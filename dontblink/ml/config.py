"""
Training configuration for custom model.
"""
from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainConfig:
    """Training configuration."""
    
    # Data paths
    dataset_path: str = "data/datasets/dataset_1"
    train_split: float = 0.8
    
    # Model
    image_size: int = 320
    backbone: str = "mobilenet_v3_small"
    pretrained: bool = True
    
    # Training
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    lr_scheduler: str = "cosine"
    lr_step_size: int = 30
    lr_gamma: float = 0.1
    
    # Loss weights
    presence_loss_weight: float = 1.0
    box_loss_weight: float = 10.0
    
    # Validation
    val_metric: str = "deployment_score"
    val_frequency: int = 1
    
    # Outputs
    output_dir: str = "models"
    checkpoint_name: str = "custom_best.pt"
    log_dir: str = "logs"
    
    # Reproducibility
    seed: int = 42
    
    # Hardware
    num_workers: int = 4
    pin_memory: bool = True
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    
    def __post_init__(self):
        """Validate configuration."""
        assert 0 < self.train_split < 1, "train_split must be between 0 and 1"
        assert self.image_size > 0, "image_size must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.presence_loss_weight >= 0, "presence_loss_weight must be non-negative"
        assert self.box_loss_weight >= 0, "box_loss_weight must be non-negative"
