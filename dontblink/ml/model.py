"""
Custom lightweight model for printhead detection.
"""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple


class PrintheadDetector(nn.Module):
    """
    Lightweight CNN for printhead detection.
    
    Architecture:
    - Backbone: MobileNetV3-small (pre-trained on ImageNet)
    - Head: Custom regression head
    - Outputs: [p_present, x_center_norm, y_center_norm, w_norm, h_norm]
    """
    
    def __init__(self, backbone_name: str = "mobilenet_v3_small", pretrained: bool = True):
        """
        Initialize model.
        
        Args:
            backbone_name: Backbone architecture name
            pretrained: Use ImageNet pre-trained weights
        """
        super().__init__()
        
        # Load backbone
        if backbone_name == "mobilenet_v3_small":
            backbone = models.mobilenet_v3_small(weights='IMAGENET1K_V1' if pretrained else None)
            # Remove classifier, keep features
            self.backbone = nn.Sequential(*list(backbone.features.children()))
            # MobileNetV3-small output: 576 features at 1/32 scale
            backbone_out_features = 576
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Detection head
        # Output: [p_present, x_center, y_center, w, h] (all normalized 0-1)
        self.head = nn.Sequential(
            nn.Linear(backbone_out_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 5),
            nn.Sigmoid()
        )
        
        # Initialize head weights
        self._init_head_weights()
    
    def _init_head_weights(self):
        """Initialize detection head weights."""
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
        # Initialize presence output bias to logit(0.5) = 0
        # Initialize box outputs to center (0.5, 0.5) with small size
        with torch.no_grad():
            self.head[-2].bias[0].fill_(0.0)  # p_present
            self.head[-2].bias[1].fill_(0.0)  # x_center (sigmoid(0) = 0.5)
            self.head[-2].bias[2].fill_(0.0)  # y_center
            self.head[-2].bias[3].fill_(-1.0)  # w (sigmoid(-1) ≈ 0.27)
            self.head[-2].bias[4].fill_(-1.0)  # h
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, 3, H, W] (RGB, normalized)
        
        Returns:
            Tensor [B, 5] with [p_present, x_center, y_center, w, h]
        """
        # Extract features
        features = self.backbone(x)  # [B, C, H', W']
        
        # Global pooling
        pooled = self.avgpool(features)  # [B, C, 1, 1]
        pooled = pooled.view(pooled.size(0), -1)  # [B, C]
        
        # Detection head
        outputs = self.head(pooled)  # [B, 5]
        
        return outputs
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with separation of presence and box.
        
        Args:
            x: Input tensor [B, 3, H, W]
        
        Returns:
            p_present: [B, 1] presence probability
            box: [B, 4] normalized box [x_center, y_center, w, h]
        """
        outputs = self.forward(x)
        p_present = outputs[:, 0:1]
        box = outputs[:, 1:5]
        return p_present, box


def create_model(backbone_name: str = "mobilenet_v3_small", pretrained: bool = True) -> PrintheadDetector:
    """
    Factory function to create model.
    
    Args:
        backbone_name: Backbone architecture
        pretrained: Use pre-trained weights
    
    Returns:
        PrintheadDetector model
    """
    return PrintheadDetector(backbone_name=backbone_name, pretrained=pretrained)


def load_model(checkpoint_path: str, device: str = "cpu") -> PrintheadDetector:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to .pt checkpoint
        device: Device to load on
    
    Returns:
        Loaded model in eval mode
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            model_state = checkpoint['model']
        elif 'state_dict' in checkpoint:
            model_state = checkpoint['state_dict']
        else:
            model_state = checkpoint
    else:
        model_state = checkpoint
    
    # Create model
    model = create_model(pretrained=False)
    model.load_state_dict(model_state, strict=False)
    model.eval()
    model.to(device)
    
    return model