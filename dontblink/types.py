"""
Shared type definitions for Dont-Blink.
"""
from typing import Tuple, Dict


class DetectionResult:
    """Container for detection results."""
    
    def __init__(self, x: float, y: float, w: float, h: float, confidence: float, frame_width: int, frame_height: int):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.confidence = confidence
        self.frame_width = frame_width
        self.frame_height = frame_height
    
    @property
    def x_normalized(self) -> float:
        """Normalized x coordinate (0-1)."""
        return self.x / self.frame_width
    
    @property
    def y_normalized(self) -> float:
        """Normalized y coordinate (0-1)."""
        return self.y / self.frame_height
    
    @property
    def w_normalized(self) -> float:
        """Normalized width (0-1)."""
        return self.w / self.frame_width
    
    @property
    def h_normalized(self) -> float:
        """Normalized height (0-1)."""
        return self.h / self.frame_height
    
    @property
    def center(self) -> Tuple[float, float]:
        """Center point of bounding box."""
        return (self.x, self.y)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'x': self.x,
            'y': self.y,
            'w': self.w,
            'h': self.h,
            'confidence': self.confidence,
            'x_normalized': self.x_normalized,
            'y_normalized': self.y_normalized,
        }
