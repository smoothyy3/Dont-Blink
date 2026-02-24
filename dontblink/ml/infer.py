"""
Inference module for custom printhead detection model.
Compatible with DetectionService interface.
"""
import torch
import numpy as np
import cv2
import logging
from typing import List, Optional, Tuple
from pathlib import Path
from .model import load_model, PrintheadDetector
from ..types import DetectionResult

logger = logging.getLogger(__name__)


class CustomDetector:
    """
    Custom detector.
    Maintains compatibility with DetectionService interface.
    """
    def __init__(self, model_path: str, device: str = "cpu", image_size: int = 320):
        """
        Initialize custom detector.
        
        Args:
            model_path: Path to model checkpoint (.pt file)
            device: Device to run on ("cpu", "cuda", "mps")
            image_size: Input image size (model was trained on this)
        """
        self.model_path = model_path
        self.device = device
        self.image_size = image_size
        
        # Load model
        logger.info(f"Loading custom model from {model_path}")
        self.model = load_model(model_path, device=device)
        self.model.eval()
        
        logger.info(f"Model loaded on {device}")
    
    def detect(self, frame: np.ndarray, confidence_threshold: float = 0.5, original_dims: Optional[Tuple[int, int]] = None) -> List[DetectionResult]:
        """
        Detect printhead in frame.
        
        Args:
            frame: BGR numpy array (OpenCV format) [H, W, 3] (may be resized)
            confidence_threshold: Threshold for presence probability
            original_dims: Optional (height, width) of original frame for coordinate conversion
        
        Returns:
            List of DetectionResult objects (empty if no detection)
        """
        input_tensor = self._preprocess(frame)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
        
        p_present = outputs[0, 0].item()
        x_center_norm = outputs[0, 1].item()
        y_center_norm = outputs[0, 2].item()
        w_norm = outputs[0, 3].item()
        h_norm = outputs[0, 4].item()
        
        if p_present < confidence_threshold:
            return []
        
        if original_dims:
            frame_h, frame_w = original_dims
        else:
            frame_h, frame_w = frame.shape[:2]
        
        x_center = x_center_norm * frame_w
        y_center = y_center_norm * frame_h
        w = w_norm * frame_w
        h = h_norm * frame_h
        
        x_center = max(0, min(frame_w - 1, x_center))
        y_center = max(0, min(frame_h - 1, y_center))
        w = max(1, min(frame_w, w))
        h = max(1, min(frame_h, h))
        
        detection = DetectionResult(x=x_center, y=y_center, w=w, h=h, confidence=p_present, frame_width=frame_w, frame_height=frame_h)
        
        return [detection]
    
    def detect_batch(self, frames: List[np.ndarray], confidence_threshold: float = 0.5, original_dims: Optional[List[Tuple[int, int]]] = None) -> List[List[DetectionResult]]:
        """
        Detect printhead in multiple frames.
        
        Args:
            frames: List of BGR numpy arrays (OpenCV format) [H, W, 3] (may be resized)
            confidence_threshold: Threshold for presence probability
            original_dims: Optional list of (height, width) tuples for original frame dimensions
        
        Returns:
            List of DetectionResult lists (one per frame)
        """
        if not frames:
            return []
        
        batch_tensors = []
        frame_dims = []
        for i, frame in enumerate(frames):
            tensor = self._preprocess(frame)
            batch_tensors.append(tensor)
            if original_dims and i < len(original_dims):
                frame_dims.append(original_dims[i])
            else:
                frame_dims.append(frame.shape[:2])
        
        batch = torch.cat(batch_tensors, dim=0)
        
        with torch.no_grad():
            outputs = self.model(batch)
        
        results = []
        for i, (frame_h, frame_w) in enumerate(frame_dims):
            p_present = outputs[i, 0].item()
            x_center_norm = outputs[i, 1].item()
            y_center_norm = outputs[i, 2].item()
            w_norm = outputs[i, 3].item()
            h_norm = outputs[i, 4].item()
            
            if p_present < confidence_threshold:
                results.append([])
                continue
            
            x_center = x_center_norm * frame_w
            y_center = y_center_norm * frame_h
            w = w_norm * frame_w
            h = h_norm * frame_h
            
            x_center = max(0, min(frame_w - 1, x_center))
            y_center = max(0, min(frame_h - 1, y_center))
            w = max(1, min(frame_w, w))
            h = max(1, min(frame_h, h))
            
            detection = DetectionResult(x=x_center, y=y_center, w=w, h=h, confidence=p_present, frame_width=frame_w, frame_height=frame_h)
            
            results.append([detection])
        
        return results
    
    def detect_raw(self, frame: np.ndarray) -> dict:
        """
        Return raw model outputs for a frame without applying any threshold.
        Useful for uncertainty sampling in the contribution pipeline.

        Returns:
            Dict with keys: p_present, x_norm, y_norm, w_norm, h_norm
        """
        input_tensor = self._preprocess(frame)
        with torch.no_grad():
            outputs = self.model(input_tensor)
        return {
            "p_present": outputs[0, 0].item(),
            "x_norm": outputs[0, 1].item(),
            "y_norm": outputs[0, 2].item(),
            "w_norm": outputs[0, 3].item(),
            "h_norm": outputs[0, 4].item(),
        }

    def detect_raw_batch(self, frames: List[np.ndarray]) -> List[dict]:
        """
        Return raw model outputs for a batch of frames without thresholding.
        """
        if not frames:
            return []
        batch = torch.cat([self._preprocess(f) for f in frames], dim=0)
        with torch.no_grad():
            outputs = self.model(batch)
        results = []
        for i in range(outputs.shape[0]):
            results.append({
                "p_present": outputs[i, 0].item(),
                "x_norm": outputs[i, 1].item(),
                "y_norm": outputs[i, 2].item(),
                "w_norm": outputs[i, 3].item(),
                "h_norm": outputs[i, 4].item(),
            })
        return results

    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess frame for model input.
        
        Args:
            frame: BGR numpy array [H, W, 3] (may already be resized)
        
        Returns:
            Tensor [1, 3, image_size, image_size] normalized [0, 1]
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        if h != self.image_size or w != self.image_size:
            frame_resized = cv2.resize(frame_rgb, (self.image_size, self.image_size))
        else:
            frame_resized = frame_rgb
        tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float()
        tensor = tensor / 255.0
        tensor = tensor.unsqueeze(0)
        tensor = tensor.to(self.device)
        return tensor

def benchmark_inference(model_path: str, num_iterations: int = 100, image_size: tuple = (480, 640), device: str = "cpu") -> dict:
    """
    Benchmark inference speed.
    
    Args:
        model_path: Path to model checkpoint
        num_iterations: Number of iterations to benchmark
        image_size: (height, width) of test image
        device: Device to benchmark on
    
    Returns:
        Dictionary with timing statistics
    """
    import time
    
    # Create detector
    detector = CustomDetector(model_path, device=device)
    
    # Create dummy frame (BGR)
    dummy_frame = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
    
    # Warmup
    for _ in range(10):
        _ = detector.detect(dummy_frame)
    
    # Benchmark
    times = []
    for _ in range(num_iterations):
        start = time.time()
        _ = detector.detect(dummy_frame)
        times.append((time.time() - start) * 1000)  # Convert to ms
    
    times = np.array(times)
    
    return {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'median_ms': float(np.median(times)),
        'fps': float(1000.0 / np.mean(times))
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark custom detector")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    stats = benchmark_inference(args.model, num_iterations=args.iterations, image_size=(args.height, args.width), device=args.device)
    
    print("\nBenchmark Results:")
    print(f"Mean: {stats['mean_ms']:.2f} ms")
    print(f"Std: {stats['std_ms']:.2f} ms")
    print(f"Min: {stats['min_ms']:.2f} ms")
    print(f"Max: {stats['max_ms']:.2f} ms")
    print(f"Median: {stats['median_ms']:.2f} ms")
    print(f"FPS: {stats['fps']:.2f}")