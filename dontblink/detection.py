import os
import cv2
import math
import logging
import torch
from typing import Optional, List
from .types import DetectionResult
from .ml.infer import CustomDetector
from .config import Config
from .utils import validate_model_path, detect_device
from .model_manager import resolve_model_path

logger = logging.getLogger(__name__)

class DetectionService:
    """Service for detecting printhead in video frames."""
    
    def __init__(self, config: Config):
        """
        Initialize detection service.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.model = None
        self.device = None
        self._load_model()
    
    def _load_model(self):
        """Load custom PyTorch model. Auto-downloads weights if missing."""
        model_path = resolve_model_path(self.config.model_path)
        
        try:
            device = detect_device(self.config.device)
            
            logger.info(f"Loading custom model from {model_path} on {device}")
            self.model = CustomDetector(model_path, device=device)
            self.device = device
            self.inference_size = self.model.image_size
            
            try:
                if hasattr(torch, 'compile') and device == 'cuda':
                    logger.info("Compiling model for faster inference (CUDA)...")
                    self.model.model = torch.compile(self.model.model, mode='reduce-overhead')
                    logger.info("Model compiled successfully")
                elif device == 'mps':
                    logger.debug("Skipping model compilation (MPS backend not fully supported)")
            except Exception as e:
                logger.debug(f"Model compilation not available or failed: {e}")
            
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def detect(self, frame: cv2.Mat, confidence: Optional[float] = None, original_dims: Optional[tuple] = None) -> List[DetectionResult]:
        """
        Detect printhead in a frame.
        
        Args:
            frame: OpenCV frame (numpy array, may be resized)
            confidence: Confidence threshold (overrides config if provided)
            original_dims: Optional (height, width) of original frame for coordinate conversion
            
        Returns:
            List of DetectionResult objects
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        conf_threshold = confidence if confidence is not None else self.config.confidence
        
        try:
            detections = self.model.detect(frame, confidence_threshold=conf_threshold, original_dims=original_dims)
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            raise
    
    def is_printhead_visible(self, frame: cv2.Mat, confidence: Optional[float] = None) -> bool:
        """
        Check if printhead is visible in frame.
        
        Args:
            frame: OpenCV frame
            confidence: Confidence threshold
            
        Returns:
            True if printhead detected, False otherwise
        """
        detections = self.detect(frame, confidence)
        return len(detections) > 0
    
    def get_printhead_position(self, frame: cv2.Mat, confidence: Optional[float] = None) -> Optional[DetectionResult]:
        """
        Get the primary printhead position (first detection).
        
        Args:
            frame: OpenCV frame
            confidence: Confidence threshold
            
        Returns:
            DetectionResult or None if no detection
        """
        detections = self.detect(frame, confidence)
        return detections[0] if detections else None
    
    def detect_raw(self, frame: cv2.Mat) -> dict:
        """Return raw model outputs (no threshold) for uncertainty sampling."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        return self.model.detect_raw(frame)

    def detect_raw_batch(self, frames: List[cv2.Mat]) -> list:
        """Return raw model outputs for a batch of frames (no threshold)."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        return self.model.detect_raw_batch(frames)

    def detect_batch(self, frames: List[cv2.Mat], confidence: Optional[float] = None, original_dims: Optional[List[tuple]] = None) -> List[List[DetectionResult]]:
        """
        Detect printhead in multiple frames (batch processing for better performance).
        
        Args:
            frames: List of OpenCV frames (numpy arrays, may be resized)
            confidence: Confidence threshold (overrides config if provided)
            original_dims: Optional list of (height, width) tuples for original frame dimensions
            
        Returns:
            List of DetectionResult lists (one per frame)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        if not frames:
            return []
        
        conf_threshold = confidence if confidence is not None else self.config.confidence
        
        try:
            detections_list = self.model.detect_batch(frames, confidence_threshold=conf_threshold, original_dims=original_dims)
            return detections_list
        except Exception as e:
            logger.error(f"Batch detection failed: {e}")
            logger.warning("Falling back to individual frame detection")
            if original_dims:
                return [self.detect(frame, confidence, orig_dims) for frame, orig_dims in zip(frames, original_dims)]
            else:
                return [self.detect(frame, confidence) for frame in frames]
    
    def visualize_detections(self, frame: cv2.Mat, detections: List[DetectionResult], 
                           show_confidence: bool = True, show_center: bool = True) -> cv2.Mat:
        """
        Draw bounding boxes and labels on frame for debugging.
        
        Args:
            frame: OpenCV frame (numpy array)
            detections: List of DetectionResult objects
            show_confidence: Whether to show confidence score
            show_center: Whether to mark center point
            
        Returns:
            Frame with drawn bounding boxes (copy of original)
        """
        vis_frame = frame.copy()
        
        for i, det in enumerate(detections):
            x1 = int(det.x - det.w / 2)
            y1 = int(det.y - det.h / 2)
            x2 = int(det.x + det.w / 2)
            y2 = int(det.y + det.h / 2)
            
            color = (0, 255, 0)
            thickness = 2
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, thickness)
            
            if show_center:
                center = (int(det.x), int(det.y))
                cv2.circle(vis_frame, center, 5, color, -1)
            
            label_parts = [f"Printhead {i+1}"]
            if show_confidence:
                label_parts.append(f"{det.confidence:.2f}")
            label = " | ".join(label_parts)
            label += f" | X:{det.x_normalized:.3f} Y:{det.y_normalized:.3f}"
            
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = max(y1 - 10, text_height + 10)
            cv2.rectangle(
                vis_frame,
                (x1, label_y - text_height - 5),
                (x1 + text_width + 5, label_y + baseline),
                (0, 0, 0),
                -1
            )
            cv2.putText(
                vis_frame,
                label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        return vis_frame

class PrintheadTracker:
    """
    Tracks printhead position over time to determine when to capture frames.

    Capture modes (user chooses where the head parks):
      - ``left-park``: capture when the head is stationary at its leftmost position (default).
      - ``right-park``: capture when the head is stationary at its rightmost position.
      - ``top-park``: capture when the head is stationary at its topmost position (e.g. overhead camera).
      - ``out-of-view``: capture only when the model reports no printhead.
    """

    MODES = ("left-park", "right-park", "top-park", "out-of-view")

    def __init__(self, config: Config, capture_mode: str = "left-park"):
        if capture_mode not in self.MODES:
            raise ValueError(f"Unknown capture_mode '{capture_mode}', expected one of {self.MODES}")

        self.config = config
        self.capture_mode = capture_mode
        self.wiggle_room = config.wiggle_room

        # Left-park state
        self._lp_prev_x = 1.0
        self._lp_was_same = False
        self._lp_photo_taken = False
        # Right-park state
        self._rp_prev_x = 0.0
        self._rp_was_same = False
        self._rp_photo_taken = False
        # Top-park state
        self._tp_prev_y = 1.0
        self._tp_was_same = False
        self._tp_photo_taken = False

    def should_capture(self, detection: Optional[DetectionResult]) -> bool:
        if self.capture_mode == "left-park":
            return self._left_park(detection)
        if self.capture_mode == "right-park":
            return self._right_park(detection)
        if self.capture_mode == "top-park":
            return self._top_park(detection)
        # out-of-view
        return detection is None

    def reset(self):
        self._lp_prev_x = 1.0
        self._lp_was_same = False
        self._lp_photo_taken = False
        self._rp_prev_x = 0.0
        self._rp_was_same = False
        self._rp_photo_taken = False
        self._tp_prev_y = 1.0
        self._tp_was_same = False
        self._tp_photo_taken = False
        logger.debug("Tracker reset")

    def get_stats(self) -> dict:
        """Return tracker diagnostics for run.json / CLI output."""
        return {"capture_mode": self.capture_mode}

    # ------------------------------------------------------------------
    # Park modes: left, right, top
    # ------------------------------------------------------------------

    def _left_park(self, detection: Optional[DetectionResult]) -> bool:
        if detection is None:
            self._lp_was_same = False
            self._lp_photo_taken = False
            return True
        x = detection.x_normalized
        if abs(x - self._lp_prev_x) < self.wiggle_room:
            self._lp_prev_x = x
            if not self._lp_was_same:
                self._lp_was_same = True
                return False
            if not self._lp_photo_taken:
                self._lp_photo_taken = True
                return True
            return False
        self._lp_was_same = False
        self._lp_photo_taken = False
        if self._lp_prev_x > x:
            self._lp_prev_x = x
        return False

    def _right_park(self, detection: Optional[DetectionResult]) -> bool:
        if detection is None:
            self._rp_was_same = False
            self._rp_photo_taken = False
            return True
        x = detection.x_normalized
        if abs(x - self._rp_prev_x) < self.wiggle_room:
            if not self._rp_was_same:
                self._rp_was_same = True
                return False
            if not self._rp_photo_taken:
                self._rp_photo_taken = True
                return True
            return False
        self._rp_was_same = False
        self._rp_photo_taken = False
        if self._rp_prev_x < x:
            self._rp_prev_x = x
        return False

    def _top_park(self, detection: Optional[DetectionResult]) -> bool:
        if detection is None:
            self._tp_was_same = False
            self._tp_photo_taken = False
            return True
        y = detection.y_normalized
        if abs(y - self._tp_prev_y) < self.wiggle_room:
            if not self._tp_was_same:
                self._tp_was_same = True
                return False
            if not self._tp_photo_taken:
                self._tp_photo_taken = True
                return True
            return False
        self._tp_was_same = False
        self._tp_photo_taken = False
        if self._tp_prev_y > y:
            self._tp_prev_y = y
        return False