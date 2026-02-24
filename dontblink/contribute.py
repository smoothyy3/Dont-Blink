"""
Data contribution pipeline for Dont-Blink.

Scans a video, selects frames where the model is uncertain or diverse,
launches a lightweight labeling UI, and packages everything into a
contribution bundle (frames + labels + metadata).
"""
import os
import json
import shutil
import logging
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from .config import Config
from .detection import DetectionService
from .utils import compute_video_fingerprint

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Frame selection
# ---------------------------------------------------------------------------

class CandidateFrame:
    """A frame selected as a candidate for contribution."""

    __slots__ = (
        "frame_idx", "timestamp_s", "p_present",
        "x_norm", "y_norm", "w_norm", "h_norm",
        "uncertainty", "image",
    )

    def __init__(self, frame_idx: int, timestamp_s: float, raw: dict,
                 image: np.ndarray, confidence_threshold: float):
        self.frame_idx = frame_idx
        self.timestamp_s = timestamp_s
        self.p_present = raw["p_present"]
        self.x_norm = raw["x_norm"]
        self.y_norm = raw["y_norm"]
        self.w_norm = raw["w_norm"]
        self.h_norm = raw["h_norm"]
        self.image = image
        self.uncertainty = abs(self.p_present - confidence_threshold)


def scan_video(
    video_path: str,
    detection_service: DetectionService,
    confidence_threshold: float = 0.5,
    sample_interval_s: float = 2.0,
    max_candidates: int = 20,
    negative_ratio: float = 0.2,
    min_time_gap_s: float = 5.0,
    progress_callback=None,
) -> List[CandidateFrame]:
    """
    Scan a video and select frames for contribution.

    Strategy:
      1. Sample every *sample_interval_s* seconds.
      2. Run raw inference (no threshold) on every sample.
      3. Rank by *uncertainty* — how close p_present is to the threshold.
      4. Include a fraction of clear negatives (low p_present) for robustness.
      5. Enforce a minimum time gap between selected frames for diversity.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    if hasattr(cv2, 'CAP_PROP_ORIENTATION_AUTO'):
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps
    frame_interval = max(1, int(fps * sample_interval_s))

    # Detect rotation
    rotation = 0
    try:
        rot = cap.get(cv2.CAP_PROP_ORIENTATION_META)
        if rot not in (-1, 0):
            rotation = int(rot)
    except AttributeError:
        pass

    inference_size = getattr(detection_service, "inference_size", 320)
    samples_expected = int(total_frames / frame_interval) + 1

    logger.info(
        f"Scanning {video_path}: {total_frames} frames, {duration_s:.0f}s, "
        f"sampling every {sample_interval_s}s (~{samples_expected} samples)"
    )

    all_candidates: List[CandidateFrame] = []
    frame_idx = 0
    batch_frames: List[np.ndarray] = []
    batch_meta: List[Tuple[int, float, np.ndarray]] = []
    batch_size = 16

    pbar = tqdm(total=total_frames, unit="frame", desc="Scanning video")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if rotation == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if frame_idx % frame_interval == 0:
            resized = cv2.resize(frame, (inference_size, inference_size))
            batch_frames.append(resized)
            timestamp_s = frame_idx / fps
            batch_meta.append((frame_idx, timestamp_s, frame.copy()))

            if len(batch_frames) >= batch_size:
                raws = detection_service.detect_raw_batch(batch_frames)
                for (fidx, ts, img), raw in zip(batch_meta, raws):
                    all_candidates.append(
                        CandidateFrame(fidx, ts, raw, img, confidence_threshold)
                    )
                batch_frames.clear()
                batch_meta.clear()

        frame_idx += 1
        pbar.update(1)
        if progress_callback:
            progress_callback(frame_idx, total_frames)

    # flush remaining batch
    if batch_frames:
        raws = detection_service.detect_raw_batch(batch_frames)
        for (fidx, ts, img), raw in zip(batch_meta, raws):
            all_candidates.append(
                CandidateFrame(fidx, ts, raw, img, confidence_threshold)
            )

    pbar.close()
    cap.release()

    logger.info(f"Scanned {len(all_candidates)} sample frames")

    return _select_diverse_frames(
        all_candidates,
        max_candidates=max_candidates,
        negative_ratio=negative_ratio,
        min_time_gap_s=min_time_gap_s,
        confidence_threshold=confidence_threshold,
    )


def _select_diverse_frames(
    candidates: List[CandidateFrame],
    max_candidates: int,
    negative_ratio: float,
    min_time_gap_s: float,
    confidence_threshold: float,
) -> List[CandidateFrame]:
    """Pick the most informative + diverse subset."""
    if not candidates:
        return []

    n_negatives = max(1, int(max_candidates * negative_ratio))
    n_uncertain = max_candidates - n_negatives

    negatives = sorted(
        [c for c in candidates if c.p_present < confidence_threshold * 0.5],
        key=lambda c: c.p_present,
    )
    uncertain = sorted(
        [c for c in candidates if c.p_present >= confidence_threshold * 0.5],
        key=lambda c: c.uncertainty,
    )

    selected: List[CandidateFrame] = []

    def _can_add(candidate: CandidateFrame) -> bool:
        for s in selected:
            if abs(candidate.timestamp_s - s.timestamp_s) < min_time_gap_s:
                return False
        return True

    for c in uncertain:
        if len(selected) >= n_uncertain:
            break
        if _can_add(c):
            selected.append(c)

    for c in negatives:
        if len(selected) >= max_candidates:
            break
        if _can_add(c):
            selected.append(c)

    # If we still have room, fill with remaining candidates by uncertainty
    remaining = sorted(
        [c for c in candidates if c not in selected],
        key=lambda c: c.uncertainty,
    )
    for c in remaining:
        if len(selected) >= max_candidates:
            break
        if _can_add(c):
            selected.append(c)

    selected.sort(key=lambda c: c.timestamp_s)
    logger.info(
        f"Selected {len(selected)} frames "
        f"(uncertain: {sum(1 for s in selected if s.p_present >= confidence_threshold * 0.5)}, "
        f"negative: {sum(1 for s in selected if s.p_present < confidence_threshold * 0.5)})"
    )
    return selected


# ---------------------------------------------------------------------------
# Contribution bundle
# ---------------------------------------------------------------------------

def prepare_review_data(
    candidates: List[CandidateFrame],
    work_dir: Path,
) -> List[Dict[str, Any]]:
    """
    Save candidate frames as JPGs and return metadata for the labeling UI.
    Each frame gets a bounding box overlay burned into a preview copy,
    while the clean original is saved separately for the contribution.
    """
    frames_dir = work_dir / "frames"
    previews_dir = work_dir / "previews"
    frames_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)

    review_items = []
    for i, c in enumerate(candidates):
        fname = f"frame_{i:04d}.jpg"

        # Save clean frame
        cv2.imwrite(str(frames_dir / fname), c.image, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Save preview with bounding box overlay
        preview = c.image.copy()
        h, w = preview.shape[:2]
        if c.p_present > 0.1:
            x1 = int((c.x_norm - c.w_norm / 2) * w)
            y1 = int((c.y_norm - c.h_norm / 2) * h)
            x2 = int((c.x_norm + c.w_norm / 2) * w)
            y2 = int((c.y_norm + c.h_norm / 2) * h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            color = (0, 255, 0)  # green
            cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
            label = f"conf: {c.p_present:.2f}"
            cv2.putText(preview, label, (x1, max(y1 - 8, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.imwrite(str(previews_dir / fname), preview, [cv2.IMWRITE_JPEG_QUALITY, 90])

        review_items.append({
            "index": i,
            "filename": fname,
            "frame_idx": c.frame_idx,
            "timestamp_s": round(c.timestamp_s, 2),
            "p_present": round(c.p_present, 4),
            "bbox": [
                round(c.x_norm, 4), round(c.y_norm, 4),
                round(c.w_norm, 4), round(c.h_norm, 4),
            ],
        })

    return review_items


def create_bundle(
    work_dir: Path,
    review_items: List[Dict[str, Any]],
    labels: Dict[int, str],
    metadata: Dict[str, Any],
    video_path: str,
) -> Path:
    """
    Package confirmed labels into a contribution bundle folder.

    Labels mapping: index -> "confirm" | "reject" | "no_printhead"

    Bundle structure:
        bundle/
            frames/          ← clean JPGs (only confirmed + no-printhead)
            labels/          ← YOLO-format .txt per frame
            meta.json        ← printer/camera/lighting info + run context
    """
    bundle_dir = work_dir / "bundle"
    bundle_frames = bundle_dir / "frames"
    bundle_labels = bundle_dir / "labels"
    bundle_frames.mkdir(parents=True, exist_ok=True)
    bundle_labels.mkdir(parents=True, exist_ok=True)

    included = 0
    for item in review_items:
        idx = item["index"]
        action = labels.get(idx, "reject")
        if action == "reject":
            continue

        src = work_dir / "frames" / item["filename"]
        dst_frame = bundle_frames / item["filename"]
        shutil.copy2(str(src), str(dst_frame))

        # YOLO label: class x_center y_center width height
        label_name = Path(item["filename"]).stem + ".txt"
        with open(bundle_labels / label_name, "w") as f:
            if action == "confirm":
                bx, by, bw, bh = item["bbox"]
                f.write(f"0 {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}\n")
            else:
                # "no_printhead" → empty label file (negative sample)
                pass
        included += 1

    meta = {
        "version": "1.0",
        "license": "CC0-1.0",
        "consent_given": True,
        "video_fingerprint": compute_video_fingerprint(video_path),
        "total_candidates": len(review_items),
        "confirmed": sum(1 for v in labels.values() if v == "confirm"),
        "rejected": sum(1 for v in labels.values() if v == "reject"),
        "no_printhead": sum(1 for v in labels.values() if v == "no_printhead"),
        "included_frames": included,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    meta.update(metadata)

    with open(bundle_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(
        f"Bundle created: {included} frames "
        f"({meta['confirmed']} confirmed, {meta['no_printhead']} negatives)"
    )
    return bundle_dir


def zip_bundle(bundle_dir: Path) -> Path:
    """Create a .zip of the bundle folder. Returns path to the zip file."""
    zip_path = bundle_dir.parent / f"{bundle_dir.name}"
    archive = shutil.make_archive(str(zip_path), "zip", str(bundle_dir.parent), bundle_dir.name)
    return Path(archive)
