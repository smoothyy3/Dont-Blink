"""
Training script for custom printhead detection model.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import logging
from typing import Dict, Optional
import numpy as np
from .model import create_model, load_model
from .dataset import create_dataloaders
from .config import TrainConfig
from .eval import evaluate_model
logger = logging.getLogger(__name__)


class PrintheadLoss(nn.Module):
    """Multi-task loss for presence + box regression."""
    
    def __init__(self, presence_weight: float = 1.0, box_weight: float = 10.0):
        super().__init__()
        self.presence_weight = presence_weight
        self.box_weight = box_weight
        self.presence_loss = nn.BCELoss()
        self.box_loss = nn.SmoothL1Loss(reduction='none')
    
    def forward(self, pred_present: torch.Tensor, pred_box: torch.Tensor, target_present: torch.Tensor, target_box: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute loss.
        
        Args:
            pred_present: [B, 1] predicted presence probability
            pred_box: [B, 4] predicted box [x, y, w, h]
            target_present: [B, 1] target presence (0 or 1)
            target_box: [B, 4] target box [x, y, w, h]
        
        Returns:
            Dictionary with 'total', 'presence', 'box' losses
        """
        # Presence loss (always computed)
        presence_loss = self.presence_loss(pred_present, target_present)
        
        # Box loss (only when present)
        present_mask = target_present.squeeze() > 0.5  # [B]
        box_loss = self.box_loss(pred_box, target_box)  # [B, 4]
        box_loss = box_loss.mean(dim=1)  # [B]
        box_loss = (box_loss * present_mask.float()).mean()
        
        # Total loss
        total_loss = (self.presence_weight * presence_loss + self.box_weight * box_loss)
        
        return {
            'total': total_loss,
            'presence': presence_loss,
            'box': box_loss
        }

def train_epoch(model: nn.Module, train_loader: torch.utils.data.DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: str, epoch: int) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_presence_loss = 0.0
    total_box_loss = 0.0
    num_batches = 0
    
    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)
        
        # Split targets
        target_present = targets[:, 0:1]  # [B, 1]
        target_box = targets[:, 1:5]  # [B, 4]
        
        # Forward
        optimizer.zero_grad()
        outputs = model(images)  # [B, 5]
        
        pred_present = outputs[:, 0:1]
        pred_box = outputs[:, 1:5]
        
        # Loss
        losses = criterion(pred_present, pred_box, target_present, target_box)
        loss = losses['total']
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Accumulate
        total_loss += loss.item()
        total_presence_loss += losses['presence'].item()
        total_box_loss += losses['box'].item()
        num_batches += 1
        
        if batch_idx % 50 == 0:
            logger.info(
                f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, "
                f"Loss: {loss.item():.4f}"
            )
    
    return {
        'loss': total_loss / num_batches,
        'presence_loss': total_presence_loss / num_batches,
        'box_loss': total_box_loss / num_batches
    }


def train(config: TrainConfig):
    """Main training function."""
    # Set seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # Device selection (CUDA > MPS > CPU)
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=config.log_dir)
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(dataset_path=config.dataset_path, image_size=config.image_size, batch_size=config.batch_size, train_split=config.train_split, num_workers=config.num_workers, seed=config.seed)
    
    # Create model
    logger.info("Creating model...")
    model = create_model(backbone_name=config.backbone, pretrained=config.pretrained)
    model = model.to(device)
    
    # Loss function
    criterion = PrintheadLoss(presence_weight=config.presence_loss_weight, box_weight=config.box_loss_weight)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    # Learning rate scheduler
    if config.lr_scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    elif config.lr_scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_step_size, gamma=config.lr_gamma)
    else:
        scheduler = None
    
    # Training loop
    best_val_score = -float('inf')
    patience_counter = 0
    best_checkpoint_path = os.path.join(config.output_dir, config.checkpoint_name)
    
    logger.info("Starting training...")
    for epoch in range(1, config.epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Log training metrics
        writer.add_scalar('Train/Loss', train_metrics['loss'], epoch)
        writer.add_scalar('Train/PresenceLoss', train_metrics['presence_loss'], epoch)
        writer.add_scalar('Train/BoxLoss', train_metrics['box_loss'], epoch)
        writer.add_scalar('Train/LearningRate', optimizer.param_groups[0]['lr'], epoch)
        
        # Validate
        if epoch % config.val_frequency == 0:
            val_metrics = evaluate_model(model, val_loader, device)
            
            # Log validation metrics
            for key, value in val_metrics.items():
                writer.add_scalar(f'Val/{key}', value, epoch)
            
            # Get validation score
            val_score = val_metrics.get(config.val_metric, val_metrics.get('f1', 0.0))
            
            # Save best model
            if val_score > best_val_score + config.early_stopping_min_delta:
                best_val_score = val_score
                patience_counter = 0
                
                # Save checkpoint
                checkpoint = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'val_score': val_score,
                    'config': config
                }
                torch.save(checkpoint, best_checkpoint_path)
                logger.info(f"Saved best model (score: {val_score:.4f})")
            else:
                patience_counter += 1
            
            logger.info(
                f"Epoch {epoch}: Train Loss={train_metrics['loss']:.4f}, "
                f"Val Score={val_score:.4f}, Best={best_val_score:.4f}"
            )
        
        # Learning rate step
        if scheduler:
            scheduler.step()
        
        # Early stopping
        if patience_counter >= config.early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    writer.close()
    logger.info(f"Training complete. Best model saved to {best_checkpoint_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train custom printhead detection model")
    parser.add_argument("--config", type=str, help="Path to config YAML (optional)")
    parser.add_argument("--dataset", type=str, default="data/datasets/dataset_1", help="Path to dataset folder (should contain train/images and train/labels)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--output-dir", type=str, default="models")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create config
    config = TrainConfig(dataset_path=args.dataset, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr, output_dir=args.output_dir)
    
    # Train
    train(config)
