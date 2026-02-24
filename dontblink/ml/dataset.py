"""
PyTorch Dataset for printhead detection with YOLO format labels.
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from typing import Tuple, Optional, List
import random

class PrintheadDataset(Dataset):
    """
    Dataset for printhead detection.
    Loads images and YOLO-format labels (class x_center y_center w h normalized).
    """
    
    def __init__(self, images_dir: str, labels_dir: str, image_size: int = 320, augment: bool = False, normalize: bool = True):
        """
        Init dataset.
        Args:
            images_dir: Directory containing images
            labels_dir: Directory containing YOLO format labels
            image_size: Target image size (square)
            augment: Enable data augmentation
            normalize: Normalize images to [0, 1]
        """
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.image_size = image_size
        self.augment = augment
        self.normalize = normalize
        
        self.image_files = sorted(list(self.images_dir.glob("*.jpg")) + list(self.images_dir.glob("*.png")))
        
        # Safety: Remove images without labels
        self.valid_pairs = []
        for img_path in self.image_files:
            label_path = self.labels_dir / (img_path.stem + ".txt")
            if label_path.exists():
                self.valid_pairs.append((img_path, label_path))
        
        print(f"Found {len(self.valid_pairs)} image-label pairs")
    
    def __len__(self) -> int:
        return len(self.valid_pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item.
        
        Returns:
            image: [3, H, W] RGB tensor, normalized [0, 1]
            target: [5] tensor [present, x_center, y_center, w, h] normalized
        """
        img_path, label_path = self.valid_pairs[idx]
        
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        # BGR to RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        original_h, original_w = image_rgb.shape[:2]
        
        present, x_center, y_center, w, h = self._load_yolo_label(label_path)
        
        if self.augment:
            image_rgb, x_center, y_center, w, h = self._augment(image_rgb, x_center, y_center, w, h)
        
        # Resize image
        image_resized = cv2.resize(image_rgb, (self.image_size, self.image_size))
        
        # Convert to tensor [H, W, 3] -> [3, H, W]
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float()
        
        # Normalize to [0, 1]
        if self.normalize:
            image_tensor = image_tensor / 255.0
        
        # Create target tensor [present, x_center, y_center, w, h]
        target = torch.tensor([present, x_center, y_center, w, h], dtype=torch.float32)
        
        return image_tensor, target
    
    def _load_yolo_label(self, label_path: Path) -> Tuple[float, float, float, float, float]:
        """
        Load YOLO format label.
        Format: class x_center y_center w h (all normalized 0-1)  
        Returns:
            (present, x_center, y_center, w, h)
            present = 1.0 if printhead present, 0.0 otherwise
        """
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        if len(lines) == 0:
            # No label = no printhead
            return (0.0, 0.5, 0.5, 0.1, 0.1)
        
        # Only 1 class = take first detection
        line = lines[0].strip()
        parts = line.split()
        
        if len(parts) < 5:
            return (0.0, 0.5, 0.5, 0.1, 0.1)
        
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])
        
        # Validate normalized coordinates
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        w = max(0.0, min(1.0, w))
        h = max(0.0, min(1.0, h))
        
        present = 1.0 if class_id == 0 else 0.0
        
        return (present, x_center, y_center, w, h)
    
    def _augment(self,image: np.ndarray, x_center: float, y_center: float, w: float, h: float) -> Tuple[np.ndarray, float, float, float, float]:
        """
        Apply light data augmentation.
        
        Args:
            image: RGB image [H, W, 3]
            x_center, y_center, w, h: Normalized box coordinates
        
        Returns:
            Augmented image and adjusted box coordinates
        """
        h_img, w_img = image.shape[:2]
        
        # Random brightness/contrast
        if random.random() < 0.5:
            alpha = random.uniform(0.8, 1.2)  # Contrast
            beta = random.randint(-20, 20)  # Brightness
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        
        # Random blur
        if random.random() < 0.3:
            ksize = random.choice([3, 5])
            image = cv2.GaussianBlur(image, (ksize, ksize), 0)
        
        # Random crop/translate (light)
        if random.random() < 0.5:
            crop_factor = random.uniform(0.9, 1.0)
            new_h = int(h_img * crop_factor)
            new_w = int(w_img * crop_factor)
            
            # Random crop position
            y_offset = random.randint(0, h_img - new_h)
            x_offset = random.randint(0, w_img - new_w)
            
            image = image[y_offset:y_offset+new_h, x_offset:x_offset+new_w]
            
            # Adjust box coordinates
            x_center = (x_center * w_img - x_offset) / new_w
            y_center = (y_center * h_img - y_offset) / new_h
            w = w * w_img / new_w
            h = h * h_img / new_h
            
            # Clamp to valid range
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))
        
        return image, x_center, y_center, w, h

def create_dataloaders(dataset_path: str, image_size: int = 320, batch_size: int = 32, train_split: float = 0.8, num_workers: int = 4, seed: int = 42) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders from a single dataset.
    Args:
        dataset_path: Path to dataset folder (should contain train/images and train/labels)
        image_size: Target image size
        batch_size: Batch size
        train_split: Fraction for training (rest goes to validation)
        num_workers: DataLoader workers
        seed: Random seed
    
    Returns:
        (train_loader, val_loader)
    """
    # seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load training dataset
    train_images_dir = os.path.join(dataset_path, "train", "images")
    train_labels_dir = os.path.join(dataset_path, "train", "labels")
    
    if not os.path.exists(train_images_dir) or not os.path.exists(train_labels_dir):
        raise ValueError(f"Dataset not found! Expected structure: {dataset_path}/train/images/ and {dataset_path}/train/labels/")
    
    # Load all training data (will be split into train/val)
    full_dataset = PrintheadDataset(
        images_dir=train_images_dir,
        labels_dir=train_labels_dir,
        image_size=image_size,
        augment=True,  # Augmentation will be applied during training
        normalize=True
    )
    
    # Split into train/val
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    print(f"Total dataset: {total_size} samples")
    print(f"Train: {len(train_dataset)} samples ({len(train_dataset)/total_size*100:.1f}%)")
    print(f"Val: {len(val_dataset)} samples ({len(val_dataset)/total_size*100:.1f}%)")
    
    return train_loader, val_loader
