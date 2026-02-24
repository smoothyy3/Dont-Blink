import os
import hashlib
import json
import time
import torch
import logging
from typing import Optional, Literal
from pathlib import Path

logger = logging.getLogger(__name__)

FINGERPRINT_BYTES = 65536  # 64KB

def detect_device(requested_device: str = 'auto') -> str:
    """
    Detect and return the best available device for inference.
    
    Args:
        requested_device: 'auto', 'cpu', 'cuda', or 'mps'
        
    Returns:
        Device string ('cpu', 'cuda', or 'mps')
    """
    if requested_device == 'cpu':
        return 'cpu'
    
    if requested_device == 'cuda':
        if torch.cuda.is_available():
            return 'cuda'
        else:
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return 'cpu'
    
    if requested_device == 'mps':
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            logger.warning("MPS requested but not available. Falling back to CPU.")
            return 'cpu'
    
    # Auto-detect
    if torch.cuda.is_available():
        device = 'cuda'
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        logger.info("Using Apple Silicon (MPS)")
    else:
        device = 'cpu'
        logger.info("Using CPU")
    
    return device


def validate_model_path(model_path: str) -> bool:
    """
    Validate that model weights file exists.
    
    Args:
        model_path: Path to model weights file
        
    Returns:
        True if valid, False otherwise
    """
    if not os.path.exists(model_path):
        logger.error(f"Model weights not found at: {model_path}")
        return False
    
    if not model_path.endswith('.pt'):
        logger.warning(f"Model file doesn't have .pt extension: {model_path}")
    
    return True


def ensure_directory(path: str) -> None:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
    """
    os.makedirs(path, exist_ok=True)
    logger.debug(f"Ensured directory exists: {path}")


def calculate_frame_skip(fps: float, frame_skip_30fps: float = 7.5, frame_skip_multiplier: int = 4) -> float:
    """
    Calculate frame skip value based on video FPS.
    
    Args:
        fps: Video frames per second
        frame_skip_30fps: Frame skip for 30fps videos
        frame_skip_multiplier: Multiplier for other FPS
        
    Returns:
        Frame skip value
    """
    if fps == 30:
        return frame_skip_30fps
    else:
        import math
        return math.ceil(fps / frame_skip_multiplier)


def compute_video_fingerprint(video_path: str) -> str:
    """
    Fast fingerprint for a video file: SHA-256 of the first 64KB + file size.
    Avoids hashing multi-GB files (which would take 10-20s).
    """
    file_size = os.path.getsize(video_path)
    h = hashlib.sha256()
    with open(video_path, 'rb') as f:
        h.update(f.read(FINGERPRINT_BYTES))
    h.update(str(file_size).encode())
    return f"{h.hexdigest()[:16]}:{file_size}"


def get_model_fingerprint(model_path: str) -> str:
    """SHA-256 of the model weights file (small enough to hash fully)."""
    h = hashlib.sha256()
    with open(model_path, 'rb') as f:
        while True:
            chunk = f.read(1 << 16)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[:16]


def write_run_json(output_dir: str, data: dict) -> str:
    """Write run.json to the output directory."""
    path = os.path.join(output_dir, "run.json")
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    logger.debug(f"Wrote run.json to {path}")
    return path


def check_video_readable(video_path: str) -> dict:
    """
    Validate that a video can be opened and frames can be read.
    Returns a dict with status info or raises with actionable message.
    """
    import cv2

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    file_size = os.path.getsize(video_path)
    if file_size == 0:
        raise ValueError(f"Video file is empty (0 bytes): {video_path}")

    cap = cv2.VideoCapture(video_path)
    if hasattr(cv2, 'CAP_PROP_ORIENTATION_AUTO'):
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    if not cap.isOpened():
        ext = Path(video_path).suffix.lower()
        raise ValueError(
            f"Could not open video file: {video_path}\n"
            f"  Possible causes:\n"
            f"  - Unsupported codec or container format ({ext})\n"
            f"  - Corrupted file\n"
            f"  - Missing ffmpeg (install with: brew install ffmpeg / apt install ffmpeg)\n"
            f"  Try converting with: ffmpeg -i \"{video_path}\" -c:v libx264 output.mp4"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ret, frame = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind

    if not ret or frame is None:
        cap.release()
        raise ValueError(
            f"Video opened but no frames could be read: {video_path}\n"
            f"  Metadata says {total_frames} frames at {fps} FPS, but first frame read failed.\n"
            f"  The file may be corrupted or use an unsupported codec.\n"
            f"  Try re-encoding: ffmpeg -i \"{video_path}\" -c:v libx264 output.mp4"
        )

    cap.release()

    if fps <= 0:
        logger.warning(f"Could not determine FPS (got {fps}), defaulting to 30")
        fps = 30.0

    if total_frames <= 0:
        logger.warning(f"Could not determine total frame count (got {total_frames})")

    return {
        'fps': fps,
        'total_frames': total_frames,
        'width': width,
        'height': height,
        'file_size': file_size,
    }


def get_organized_output_paths(video_path: str, base_output_dir: str = "outputs", organize_by_video: bool = True) -> dict:
    """
    Generate organized output paths for video processing.
    
    Default: output is placed next to the input video file:
        /path/to/video_name/
          frames/
            frame_000000.jpg
            ...
          timelapse.mp4
    
    If base_output_dir is explicitly set in config, uses that instead.
    
    Args:
        video_path: Path to input video file
        base_output_dir: Base directory for outputs (default: next to video)
        organize_by_video: Whether to create subfolder per video
        
    Returns:
        Dictionary with keys:
            - 'frames_dir': Directory for extracted frames
            - 'timelapse_path': Path for timelapse video
            - 'base_dir': Base directory for this video
    """
    from pathlib import Path
    
    video_path_obj = Path(video_path).resolve()
    video_name = video_path_obj.stem
    video_dir = video_path_obj.parent
    
    if organize_by_video:
        base_dir = Path(base_output_dir) / video_name if base_output_dir != "outputs" else video_dir / video_name
        frames_dir = base_dir / "frames"
        timelapse_path = base_dir / "timelapse.mp4"
    else:
        base_dir = Path(base_output_dir) if base_output_dir != "outputs" else video_dir
        frames_dir = base_dir / "frames"
        timelapse_path = base_dir / f"{video_name}_timelapse.mp4"
    
    return {
        'frames_dir': str(frames_dir),
        'timelapse_path': str(timelapse_path),
        'base_dir': str(base_dir)
    }