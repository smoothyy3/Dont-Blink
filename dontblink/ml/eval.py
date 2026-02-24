"""
Evaluation script for custom printhead detection model.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict
from sklearn.metrics import precision_score, recall_score, f1_score
from .model import PrintheadDetector

def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
    """
    Calculate IoU between boxes.
    
    Args:
        box1: [N, 4] boxes [x_center, y_center, w, h] normalized
        box2: [N, 4] boxes [x_center, y_center, w, h] normalized
    
    Returns:
        [N] IoU values
    """
    # Convert to [x_min, y_min, x_max, y_max]
    def to_corners(box):
        x, y, w, h = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
        x_min = x - w / 2
        y_min = y - h / 2
        x_max = x + w / 2
        y_max = y + h / 2
        return np.stack([x_min, y_min, x_max, y_max], axis=1)
    
    box1_corners = to_corners(box1)
    box2_corners = to_corners(box2)
    
    # Intersection
    x_min = np.maximum(box1_corners[:, 0], box2_corners[:, 0])
    y_min = np.maximum(box1_corners[:, 1], box2_corners[:, 1])
    x_max = np.minimum(box1_corners[:, 2], box2_corners[:, 2])
    y_max = np.minimum(box1_corners[:, 3], box2_corners[:, 3])
    
    intersection = np.maximum(0, x_max - x_min) * np.maximum(0, y_max - y_min)
    
    # Union
    area1 = (box1_corners[:, 2] - box1_corners[:, 0]) * (box1_corners[:, 3] - box1_corners[:, 1])
    area2 = (box2_corners[:, 2] - box2_corners[:, 0]) * (box2_corners[:, 3] - box2_corners[:, 1])
    union = area1 + area2 - intersection
    
    # IoU
    iou = intersection / (union + 1e-7)
    return iou

def evaluate_model(model: nn.Module, val_loader: torch.utils.data.DataLoader, device: str, presence_threshold: float = 0.5) -> Dict[str, float]:
    """
    Evaluate model on validation set.
    
    Args:
        model: Trained model
        val_loader: Validation dataloader
        device: Device to run on
        presence_threshold: Threshold for presence prediction
    
    Returns:
        Dictionary of metrics
    """
    model.eval()
    
    all_pred_present = []
    all_target_present = []
    all_pred_boxes = []
    all_target_boxes = []
    
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward
            outputs = model(images)  # [B, 5]
            
            pred_present = outputs[:, 0].cpu().numpy()
            pred_box = outputs[:, 1:5].cpu().numpy()
            
            target_present = targets[:, 0].cpu().numpy()
            target_box = targets[:, 1:5].cpu().numpy()
            
            all_pred_present.append(pred_present)
            all_target_present.append(target_present)
            all_pred_boxes.append(pred_box)
            all_target_boxes.append(target_box)
    
    # Concatenate
    pred_present = np.concatenate(all_pred_present)
    target_present = np.concatenate(all_target_present)
    pred_box = np.concatenate(all_pred_boxes)
    target_box = np.concatenate(all_target_boxes)
    
    # Binary predictions
    pred_present_binary = (pred_present >= presence_threshold).astype(int)
    target_present_binary = (target_present >= 0.5).astype(int)
    
    # Classification metrics
    precision = precision_score(target_present_binary, pred_present_binary, zero_division=0)
    recall = recall_score(target_present_binary, pred_present_binary, zero_division=0)
    f1 = f1_score(target_present_binary, pred_present_binary, zero_division=0)
    
    # Regression metrics (only on positive samples)
    present_mask = target_present_binary == 1
    if present_mask.sum() > 0:
        pred_box_pos = pred_box[present_mask]
        target_box_pos = target_box[present_mask]
        
        # MAE for each coordinate
        mae_x = np.mean(np.abs(pred_box_pos[:, 0] - target_box_pos[:, 0]))
        mae_y = np.mean(np.abs(pred_box_pos[:, 1] - target_box_pos[:, 1]))
        mae_w = np.mean(np.abs(pred_box_pos[:, 2] - target_box_pos[:, 2]))
        mae_h = np.mean(np.abs(pred_box_pos[:, 3] - target_box_pos[:, 3]))
        mae_size = np.mean(np.abs((pred_box_pos[:, 2] + pred_box_pos[:, 3]) / 2 - (target_box_pos[:, 2] + target_box_pos[:, 3]) / 2))
        
        # IoU
        iou = calculate_iou(pred_box_pos, target_box_pos)
        mean_iou = np.mean(iou)
    else:
        mae_x = mae_y = mae_w = mae_h = mae_size = mean_iou = 0.0
    
    # Deployment score (focuses on presence + X position accuracy)
    # Range: 0-1, higher is better
    deployment_score = (
        0.5 * f1 +  # Presence accuracy (50%)
        0.3 * max(0, 1 - mae_x * 2) +  # X position accuracy (30%, normalized)
        0.2 * max(0, 1 - mae_size * 2)  # Size accuracy (20%, normalized)
    )
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mae_x': mae_x,
        'mae_y': mae_y,
        'mae_w': mae_w,
        'mae_h': mae_h,
        'mae_size': mae_size,
        'iou': mean_iou,
        'deployment_score': deployment_score
    }
    
    return metrics

if __name__ == "__main__":
    import argparse
    from .model import load_model
    from .dataset import create_dataloaders
    from .config import TrainConfig
    
    parser = argparse.ArgumentParser(description="Evaluate custom printhead detection model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, default="data/datasets/dataset_1", help="Path to dataset folder")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5, help="Presence threshold")
    parser.add_argument("--device", type=str, default="cpu")
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.checkpoint, device=args.device)
    
    # Create dataloader (validation only)
    _, val_loader = create_dataloaders(dataset_path=args.dataset, image_size=320, batch_size=args.batch_size, train_split=0.8, num_workers=4, seed=42)
    
    # Evaluate
    metrics = evaluate_model(model, val_loader, args.device, args.threshold)
    
    # Print results
    print("\nEvaluation Results:")
    print("=" * 50)
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"\nRegression (on positive samples):")
    print(f"MAE X: {metrics['mae_x']:.4f}")
    print(f"MAE Y: {metrics['mae_y']:.4f}")
    print(f"MAE Size: {metrics['mae_size']:.4f}")
    print(f"IoU: {metrics['iou']:.4f}")
    print(f"\nDeployment Score: {metrics['deployment_score']:.4f}")
