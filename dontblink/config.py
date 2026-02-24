import os
import yaml
from pathlib import Path
from typing import Optional, Literal
import logging

logger = logging.getLogger(__name__)

class Config:
    """Configuration manager for Dont-Blink application."""
    
    DEFAULT_CONFIG = {
        'detection': {
            'model_path': 'models/custom_best.pt',
            'device': 'auto',
            'confidence': 0.5,
            'wiggle_room': 0.012,
            'frame_skip_30fps': 7.5,
            'frame_skip_multiplier': 4,
            'capture_mode': 'left-park',
        },
        'processing': {
            'output_format': 'jpg',
            'image_quality': 95,
            'max_concurrent_frames': 10,
            'video_rotation': 0,
            'batch_size': 1,
            'output_base_dir': 'outputs',
            'organize_by_video': True,
        },
        'timelapse': {
            'fps': 15,
            'codec': 'mp4v',
            'fourcc': 'mp4v',
        },
        'logging': {
            'level': 'INFO',
            'file': None,
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = self.DEFAULT_CONFIG.copy()
        
        env_model_path = os.getenv('DONTBLINK_MODEL_PATH')
        if env_model_path:
            self.config['detection']['model_path'] = env_model_path
            logger.info(f"Using model path from environment: {env_model_path}")
        
        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)
        else:
            logger.info("Using default configuration")
    
    def load_from_file(self, config_path: str):
        try:
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                if file_config:
                    self._merge_config(self.config, file_config)
                    logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}. Using defaults.")
    
    def _merge_config(self, base: dict, override: dict):
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def save_to_file(self, config_path: str):
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Saved configuration to {config_path}")
    
    def get(self, key_path: str, default=None):
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value):
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    @property
    def model_path(self) -> str:
        """Get model weights path."""
        return self.get('detection.model_path')
    
    @property
    def device(self) -> str:
        """Get detection device."""
        return self.get('detection.device')
    
    @property
    def confidence(self) -> float:
        """Get detection confidence threshold."""
        return self.get('detection.confidence')
    
    @property
    def wiggle_room(self) -> float:
        """Get position wiggle room threshold."""
        return self.get('detection.wiggle_room')
    
    @property
    def frame_skip_30fps(self) -> float:
        """Get frame skip value for 30fps videos."""
        return self.get('detection.frame_skip_30fps')
    
    @property
    def frame_skip_multiplier(self) -> int:
        """Get frame skip multiplier for other FPS."""
        return self.get('detection.frame_skip_multiplier')
    
    @property
    def timelapse_fps(self) -> int:
        """Get timelapse FPS."""
        return self.get('timelapse.fps')
    
    @property
    def capture_mode(self) -> str:
        """Get capture mode (left-park, right-park, top-park, out-of-view)."""
        return self.get('detection.capture_mode', 'left-park')

    @property
    def timelapse_codec(self) -> str:
        """Get timelapse codec."""
        return self.get('timelapse.codec')

    def to_dict(self) -> dict:
        """Return a deep copy of the resolved config for serialization."""
        import copy
        return copy.deepcopy(self.config)


def setup_logging(config: Config):
    """Setup logging based on configuration."""
    log_level = getattr(logging, config.get('logging.level', 'INFO').upper())
    log_format = config.get('logging.format')
    log_file = config.get('logging.file')
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers
    )

