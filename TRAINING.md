# Training Guide - Custom Printhead Detection Model

## Overview

This guide explains how to train the custom lightweight PyTorch model for printhead detection.

## Prerequisites

1. **Datasets**: Ensure you have labeled datasets in YOLO format:
   - `data/dataset_1/train/images/` and `data/dataset_1/train/labels/`
   - `data/dataset_2/train/images/` and `data/dataset_2/train/labels/`
   - `data/dataset_2/valid/images/` and `data/dataset_2/valid/labels/`

2. **Dependencies**: Install training dependencies:
   ```bash
   pip install dontblink[train]
   ```

## Quick Start

### Basic Training

```bash
python -m dontblink.ml.train \
  --dataset-1 data/dataset_1 \
  --dataset-2 data/dataset_2 \
  --epochs 100 \
  --batch-size 32 \
  --lr 0.001 \
  --output-dir models
```

### With Custom Configuration

Create a config file or modify `dontblink/ml/config.py`:

```python
from dontblink.ml.config import TrainConfig
from dontblink.ml.train import train

config = TrainConfig(
    dataset_1_path="data/dataset_1",
    dataset_2_path="data/dataset_2",
    epochs=100,
    batch_size=32,
    learning_rate=0.001,
    image_size=320,
    output_dir="models",
    checkpoint_name="custom_best.pt"
)

train(config)
```

## Training Parameters

### Key Hyperparameters

- **`image_size`**: Input image size (default: 320)
  - Smaller = faster training/inference, less detail
  - Larger = slower, more detail
  - Recommended: 320 for CPU inference

- **`batch_size`**: Batch size (default: 32)
  - Adjust based on GPU memory
  - Larger = faster training, more memory
  - CPU: Use 16-32

- **`learning_rate`**: Initial learning rate (default: 0.001)
  - Too high: unstable training
  - Too low: slow convergence
  - Recommended: 0.001 with AdamW

- **`presence_loss_weight`**: Weight for presence classification (default: 1.0)
- **`box_loss_weight`**: Weight for box regression (default: 10.0)
  - Increase if box accuracy is poor
  - Decrease if presence detection is poor

### Loss Function

The model uses a multi-task loss:
- **Presence Loss**: Binary cross-entropy (always computed)
- **Box Loss**: Smooth L1 loss (only when printhead is present)

```
Total Loss = presence_weight * BCE(p_present, target_present) +
             box_weight * SmoothL1(box, target_box) * mask
```

Where `mask = 1` when printhead is present, `0` otherwise.

## Dataset Preparation

### YOLO Format Labels

Each label file should contain:
```
0 x_center y_center w h
```

Where:
- `0` = class ID (always 0 for printheads)
- `x_center`, `y_center`, `w`, `h` = normalized coordinates (0-1)

### Dataset Structure

```
data/
├── dataset_1/
│   └── train/
│       ├── images/
│       │   ├── frame_0.jpg
│       │   └── ...
│       └── labels/
│           ├── frame_0.txt
│           └── ...
└── dataset_2/
    ├── train/
    │   ├── images/
    │   └── labels/
    └── valid/
        ├── images/
        └── labels/
```

### Data Augmentation

The training script applies light augmentations:
- Random brightness/contrast (±20%)
- Random Gaussian blur (3x3 or 5x5)
- Random crop/translate (90-100% of image)

Augmentations preserve bounding box coordinates.

## Training Process

### Monitoring Training

Training logs are saved to TensorBoard:
```bash
tensorboard --logdir logs
```

View metrics:
- `Train/Loss` - Total training loss
- `Train/PresenceLoss` - Presence classification loss
- `Train/BoxLoss` - Box regression loss
- `Val/f1` - Validation F1 score
- `Val/deployment_score` - Custom deployment metric

### Checkpoints

- **Best model**: Saved to `models/custom_best.pt` (best by validation metric)
- **Checkpoint format**: Includes model state, optimizer state, epoch, and config

### Early Stopping

Training stops early if validation score doesn't improve for `early_stopping_patience` epochs (default: 10).

## Evaluation

### Run Evaluation

```bash
python -m dontblink.ml.eval \
  --checkpoint models/custom_best.pt \
  --dataset-1 data/dataset_1 \
  --dataset-2 data/dataset_2 \
  --threshold 0.5
```

### Metrics Explained

- **Precision**: TP / (TP + FP) - How many detections are correct
- **Recall**: TP / (TP + FN) - How many printheads are found
- **F1**: Harmonic mean of precision and recall
- **MAE X**: Mean absolute error for X position (critical for tracker)
- **MAE Size**: Mean absolute error for box size
- **IoU**: Intersection over Union (box overlap)
- **Deployment Score**: Custom metric focusing on presence + X accuracy
  ```
  deployment_score = 0.5 * f1 + 0.3 * (1 - normalized_x_mae) + 0.2 * (1 - normalized_size_mae)
  ```

## Threshold Tuning

### Presence Threshold (`p_present`)

The model outputs a presence probability. You can tune the threshold:

**Lower threshold (0.3-0.4)**:
- ✅ Higher recall (finds more printheads)
- ❌ More false positives
- Use when: Missing detections is worse than false positives
- Tracker can filter false positives with position logic

**Higher threshold (0.6-0.7)**:
- ✅ Higher precision (fewer false positives)
- ❌ More false negatives (misses some printheads)
- Use when: False positives cause issues
- Tracker needs accurate detections

**Default (0.5)**:
- Balanced precision/recall
- Good starting point

### Tuning Process

1. Train model with default threshold (0.5)
2. Evaluate on validation set
3. Check precision/recall trade-off
4. Adjust threshold based on your needs:
   ```python
   # In config.yaml or code
   detection:
     confidence: 0.6  # Higher threshold
   ```

## Hyperparameter Tuning

### Learning Rate

Start with 0.001, try:
- 0.0005 (if training is unstable)
- 0.002 (if training is too slow)

### Loss Weights

If box accuracy is poor:
```python
config.box_loss_weight = 20.0  # Increase box weight
```

If presence detection is poor:
```python
config.presence_loss_weight = 2.0  # Increase presence weight
```

### Image Size

For faster inference:
```python
config.image_size = 256  # Smaller, faster
```

For better accuracy:
```python
config.image_size = 416  # Larger, more detail
```

## Troubleshooting

### Training Issues

**Loss not decreasing**:
- Check learning rate (try lower)
- Check data quality (verify labels)
- Check batch size (try smaller)

**Overfitting**:
- Increase data augmentation
- Add dropout
- Reduce model capacity
- Early stopping should help

**Out of memory**:
- Reduce batch size
- Reduce image size
- Use gradient accumulation

### Evaluation Issues

**Low precision**:
- Increase confidence threshold
- Check for label errors
- Train longer

**Low recall**:
- Decrease confidence threshold
- Check for missing labels
- Increase presence_loss_weight

**Poor box accuracy**:
- Increase box_loss_weight
- Check label quality
- Train longer

## Best Practices

1. **Start small**: Train for 20-30 epochs first to verify setup
2. **Monitor metrics**: Use TensorBoard to track training
3. **Validate early**: Check validation metrics frequently
4. **Save checkpoints**: Best model is auto-saved
5. **Test inference**: Benchmark inference speed after training
6. **Iterate**: Adjust hyperparameters based on results

## Example Training Session

```bash
# 1. Quick test (10 epochs)
python -m dontblink.ml.train --epochs 10 --batch-size 16

# 2. Full training (100 epochs)
python -m dontblink.ml.train \
  --dataset-1 data/dataset_1 \
  --dataset-2 data/dataset_2 \
  --epochs 100 \
  --batch-size 32 \
  --lr 0.001 \
  --output-dir models

# 3. Evaluate
python -m dontblink.ml.eval \
  --checkpoint models/custom_best.pt \
  --threshold 0.5

# 4. Benchmark
python -m dontblink.ml.infer \
  --model models/custom_best.pt \
  --iterations 100 \
  --device cpu
```

## Next Steps

After training:
1. Evaluate model on validation set
2. Benchmark inference speed
3. Test with video processing pipeline
4. Tune threshold if needed
5. Deploy model to production


