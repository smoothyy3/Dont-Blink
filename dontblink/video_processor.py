import os
import cv2
import time
import logging
from typing import Optional, Callable, List, Dict, Any
from pathlib import Path
from .config import Config
from .detection import DetectionService, PrintheadTracker
from .utils import calculate_frame_skip, ensure_directory
logger = logging.getLogger(__name__)


def _empty_stats() -> Dict[str, Any]:
    return {
        'frames_total': 0,
        'frames_analyzed': 0,
        'frames_captured': 0,
        'inference_times_ms': [],
        'rotation_applied': 0,
        'frame_skip': 0,
        'batch_size': 1,
        'video_fps': 0.0,
        'video_width': 0,
        'video_height': 0,
    }

class VideoProcessor:
    """Process video files or camera streams to extract timelapse frames."""
    
    def __init__(self, config: Config, detection_service: DetectionService):
        """
        Initialize video processor.
        
        Args:
            config: Configuration object
            detection_service: DetectionService instance
        """
        self.config = config
        self.detection_service = detection_service
        self.tracker = PrintheadTracker(config)
    
    def process_video_file(self, video_path: str, output_folder: str, progress_callback: Optional[Callable[[int, int], None]] = None) -> Dict[str, Any]:
        """
        Process a video file and extract timelapse frames.
        
        Args:
            video_path: Path to input video file
            output_folder: Folder to save extracted frames
            progress_callback: Optional callback(frame_count, total_frames)
            
        Returns:
            Dict with processing stats (frames_captured, frames_analyzed, inference_times_ms, etc.)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        ensure_directory(output_folder)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"Processing video: {video_path}")
            logger.info(f"FPS: {fps}, Total frames: {total_frames}")
            
            frame_skip = calculate_frame_skip(fps, self.config.frame_skip_30fps, self.config.frame_skip_multiplier)
            
            logger.info(f"Frame skip: {frame_skip}")
            
            stats = self._process_capture(cap, output_folder, frame_skip, total_frames, progress_callback)
            stats['video_fps'] = fps
            stats['frames_total'] = total_frames
            stats['video_width'] = width
            stats['video_height'] = height
            stats['frame_skip'] = frame_skip
            
            logger.info(f"Captured {stats['frames_captured']} frames")
            return stats
            
        finally:
            cap.release()
    
    def process_camera(self, camera_index: int, output_folder: str, stop_callback: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
        """
        Process live camera feed and extract timelapse frames.
        
        Returns:
            Dict with processing stats
        """
        ensure_directory(output_folder)
        
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise ValueError(f"Could not open camera {camera_index}")
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                fps = 30
            
            logger.info(f"Processing camera {camera_index} (FPS: {fps})")
            
            frame_skip = calculate_frame_skip(fps, self.config.frame_skip_30fps, self.config.frame_skip_multiplier)
            
            stats = self._process_capture(cap, output_folder, frame_skip, None, None, stop_callback)
            stats['video_fps'] = fps
            
            logger.info(f"Captured {stats['frames_captured']} frames")
            return stats
            
        finally:
            cap.release()
    
    def _get_video_rotation(self, cap: cv2.VideoCapture) -> int:
        """
        Try to detect video rotation from metadata or config.
        
        Args:
            cap: OpenCV VideoCapture object
            
        Returns:
            Rotation angle in degrees (0, 90, 180, 270)
        """
        manual_rotation = self.config.get('processing.video_rotation', 0)
        if manual_rotation != 0:
            logger.info(f"Using manual rotation from config: {manual_rotation} degrees")
            return int(manual_rotation)
        
        try:
            rotation = cap.get(cv2.CAP_PROP_ORIENTATION_META)
            if rotation != -1 and rotation != 0:
                logger.info(f"Detected rotation from video metadata: {rotation} degrees")
                return int(rotation)
        except AttributeError:
            pass
        
        return 0
    
    def _apply_rotation(self, frame: cv2.Mat, rotation: int) -> cv2.Mat:
        """
        Apply rotation to frame if needed.
        
        Args:
            frame: Input frame
            rotation: Rotation angle in degrees (0, 90, 180, 270)
            
        Returns:
            Rotated frame (or original if rotation is 0)
        """
        if rotation == 0:
            return frame
        
        if rotation == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            logger.warning(f"Unsupported rotation angle: {rotation}, skipping rotation")
            return frame
    
    def _process_capture(self, cap: cv2.VideoCapture, output_folder: str, frame_skip: float, total_frames: Optional[int] = None, progress_callback: Optional[Callable[[int, int], None]] = None, stop_callback: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
        """
        Internal method to process frames from a capture object.
        
        Returns:
            Dict with frames_captured, frames_analyzed, inference_times_ms, rotation_applied, batch_size
        """
        self.tracker.reset()
        frame_count = 0
        frames_captured = 0
        frames_analyzed = 0
        inference_times: List[float] = []
        
        rotation = self._get_video_rotation(cap)
        if rotation != 0:
            logger.info(f"Detected video rotation: {rotation} degrees")
        
        batch_size = self.config.get('processing.batch_size', 1)
        if batch_size > 1:
            logger.info(f"Using batch processing with batch size: {batch_size}")
        
        inference_size = getattr(self.detection_service, 'inference_size', 320)
        
        resized_batch = []
        original_frames = {}
        original_dims = {}
        frame_indices = []
        
        def process_batch():
            nonlocal frames_captured, frames_analyzed
            if not resized_batch:
                return
            
            batch_len = len(resized_batch)
            frames_analyzed += batch_len
            
            try:
                t0 = time.perf_counter()
                if batch_size > 1 and batch_len > 1:
                    detections_list = self.detection_service.detect_batch(
                        resized_batch, 
                        original_dims=[original_dims[idx] for idx in frame_indices]
                    )
                else:
                    detections_list = []
                    for resized_frame, frame_idx in zip(resized_batch, frame_indices):
                        original_h, original_w = original_dims[frame_idx]
                        detections = self.detection_service.detect(
                            resized_frame, 
                            original_dims=(original_h, original_w)
                        )
                        detections_list.append(detections)
                elapsed = (time.perf_counter() - t0) * 1000
                inference_times.append(elapsed / batch_len)
                
                for i, (resized_frame, frame_idx, detections) in enumerate(zip(resized_batch, frame_indices, detections_list)):
                    detection = detections[0] if detections else None
                    
                    if self.tracker.should_capture(detection):
                        original_frame = original_frames.get(frame_idx)
                        if original_frame is not None:
                            filename = os.path.join(output_folder, f"frame_{frame_idx:06d}.jpg")
                            cv2.imwrite(filename, original_frame, [cv2.IMWRITE_JPEG_QUALITY, self.config.get('processing.image_quality', 95)])
                            frames_captured += 1
                            logger.debug(f"Captured frame {frame_idx}")
                
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                for resized_frame, frame_idx in zip(resized_batch, frame_indices):
                    try:
                        original_h, original_w = original_dims[frame_idx]
                        t0 = time.perf_counter()
                        detections = self.detection_service.detect(
                            resized_frame, 
                            original_dims=(original_h, original_w)
                        )
                        inference_times.append((time.perf_counter() - t0) * 1000)
                        detection = detections[0] if detections else None
                        if self.tracker.should_capture(detection):
                            original_frame = original_frames.get(frame_idx)
                            if original_frame is not None:
                                filename = os.path.join(output_folder, f"frame_{frame_idx:06d}.jpg")
                                cv2.imwrite(filename, original_frame, [cv2.IMWRITE_JPEG_QUALITY, self.config.get('processing.image_quality', 95)])
                                frames_captured += 1
                    except Exception as e2:
                        logger.error(f"Error processing frame {frame_idx}: {e2}")
            
            resized_batch.clear()
            original_frames.clear()
            original_dims.clear()
            frame_indices.clear()
        
        while True:
            if stop_callback and stop_callback():
                logger.info("Processing stopped by callback")
                break
            
            ret, frame = cap.read()
            if not ret:
                break
            
            if rotation != 0:
                frame = self._apply_rotation(frame, rotation)
            
            if frame_count == 0:
                h, w = frame.shape[:2]
                meta_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                meta_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                logger.info(f"Video metadata dimensions: {meta_w}x{meta_h} (width x height)")
                logger.info(f"Actual frame dimensions (after rotation): {w}x{h} (width x height)")
                
                if meta_w != 0 and meta_h != 0:
                    meta_is_portrait = meta_h > meta_w
                    actual_is_portrait = h > w
                    if meta_is_portrait != actual_is_portrait and rotation == 0:
                        logger.warning(
                            f"Video metadata suggests {'portrait' if meta_is_portrait else 'landscape'} "
                            f"but frames are {'portrait' if actual_is_portrait else 'landscape'}. "
                            f"If frames appear rotated, set 'processing.video_rotation' in config.yaml "
                            f"or use --rotation 90/180/270."
                        )
            
            if frame_count % int(frame_skip) == 0:
                original_h, original_w = frame.shape[:2]
                resized_frame = cv2.resize(frame, (inference_size, inference_size))
                resized_batch.append(resized_frame)
                original_frames[frame_count] = frame.copy()
                original_dims[frame_count] = (original_h, original_w)
                frame_indices.append(frame_count)
                
                if len(resized_batch) >= batch_size:
                    process_batch()
            
            frame_count += 1
            
            if progress_callback and total_frames:
                progress_callback(frame_count, total_frames)
        
        if resized_batch:
            process_batch()
        
        return {
            'frames_captured': frames_captured,
            'frames_analyzed': frames_analyzed,
            'inference_times_ms': inference_times,
            'rotation_applied': rotation,
            'batch_size': batch_size,
        }
    
    def list_available_cameras(self) -> List[int]:
        """
        List available camera indices.
        
        Returns:
            List of available camera indices
        """
        available = []
        index = 0
        
        while True:
            cap = cv2.VideoCapture(index)
            if not cap.isOpened():
                break
            available.append(index)
            cap.release()
            index += 1
        
        return available