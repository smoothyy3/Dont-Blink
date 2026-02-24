import os
import cv2
import logging
from typing import Optional
from pathlib import Path
from natsort import natsorted
from .config import Config
from .utils import ensure_directory

logger = logging.getLogger(__name__)

class TimelapseGenerator:
    """Generate timelapse videos from captured frames."""
    
    def __init__(self, config: Config):
        """
        Initialize timelapse generator.
        
        Args:
            config: Configuration object
        """
        self.config = config
    
    def create_timelapse(self, frames_folder: str, output_path: Optional[str] = None, fps: Optional[int] = None) -> str:
        """
        Create timelapse video from frames in a folder.
        
        Args:
            frames_folder: Folder containing frame images
            output_path: Output video path (default: timelapse.mp4 in frames_folder)
            fps: Frames per second (overrides config if provided)
            
        Returns:
            Path to created video file
        """
        if not os.path.exists(frames_folder):
            raise FileNotFoundError(f"Frames folder not found: {frames_folder}")
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        images = [f for f in os.listdir(frames_folder) if Path(f).suffix.lower() in image_extensions]
        
        if not images:
            raise ValueError(f"No images found in {frames_folder}")
        
        images = natsorted(images)
        logger.info(f"Found {len(images)} frames to process")
        
        if output_path is None:
            output_path = os.path.join(frames_folder, "timelapse.mp4")
        
        video_fps = fps if fps is not None else self.config.timelapse_fps
        
        first_frame_path = os.path.join(frames_folder, images[0])
        first_frame = cv2.imread(first_frame_path)
        
        if first_frame is None:
            raise ValueError(f"Could not read first frame: {first_frame_path}")
        
        h, w = first_frame.shape[:2]
        logger.info(f"Video dimensions: {w}x{h}, FPS: {video_fps}")
        
        codec = self.config.timelapse_codec
        fourcc = cv2.VideoWriter_fourcc(*codec)
        
        output_dir = os.path.dirname(output_path)
        if output_dir:
            ensure_directory(output_dir)
        
        out = cv2.VideoWriter(output_path, fourcc, video_fps, (w, h))
        
        if not out.isOpened():
            raise RuntimeError(f"Could not initialize video writer for {output_path}")
        
        try:
            frames_written = 0
            for image_name in images:
                frame_path = os.path.join(frames_folder, image_name)
                frame = cv2.imread(frame_path)
                
                if frame is None:
                    logger.warning(f"Skipping unreadable image: {image_name}")
                    continue
                
                if frame.shape[:2] != (h, w):
                    frame = cv2.resize(frame, (w, h))
                    logger.debug(f"Resized frame {image_name}")
                
                out.write(frame)
                frames_written += 1
            
            logger.info(f"Created timelapse with {frames_written} frames: {output_path}")
            
        finally:
            out.release()
        
        return output_path
    
    def get_frame_count(self, frames_folder: str) -> int:
        """
        Get number of frames in folder.
        
        Args:
            frames_folder: Folder containing frames
            
        Returns:
            Number of frames
        """
        if not os.path.exists(frames_folder):
            return 0
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        images = [f for f in os.listdir(frames_folder) if Path(f).suffix.lower() in image_extensions]
        
        return len(images)