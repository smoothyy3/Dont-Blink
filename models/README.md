# Model Weights

This directory contains the trained custom PyTorch model weights for printhead detection.

## Model File

- `best.pt` or `custom_best.pt` - Trained custom model (~3-5MB)

## Model Architecture

- **Backbone**: MobileNetV3-small (ImageNet pre-trained)
- **Head**: Custom regression head
- **Outputs**: [p_present, x_center_norm, y_center_norm, w_norm, h_norm]
- **Framework**: PyTorch
- **Optimization**: CPU-friendly, lightweight

## Using the Model

The model will be automatically loaded when you run the application. Place your trained model weights in this directory.

### Training Your Own Model

See [TRAINING.md](../TRAINING.md) for detailed training instructions:

```bash
python -m dontblink.ml.train \
  --dataset-1 data/dataset_1 \
  --dataset-2 data/dataset_2 \
  --epochs 100 \
  --batch-size 32 \
  --output-dir models
```

The best model will be saved as `models/custom_best.pt`.

### Custom Model Path

If you want to use a different model path, configure it in `config.yaml`:
```yaml
detection:
  model_path: "/path/to/your/model.pt"
```

Or use an environment variable:
```bash
export DONTBLINK_MODEL_PATH="/path/to/your/model.pt"
```

## Model Requirements

- **Format**: PyTorch `.pt` checkpoint file
- **Architecture**: Custom PrintheadDetector (MobileNetV3-small based)
- **Input**: RGB images (BGR frames converted automatically)
- **Output**: DetectionResult objects with bounding boxes
- **Device**: CPU (CUDA/MPS optional for faster inference)

## Evaluation

Evaluate your trained model:
```bash
python -m dontblink.ml.eval \
  --checkpoint models/custom_best.pt \
  --dataset-1 data/dataset_1 \
  --dataset-2 data/dataset_2 \
  --threshold 0.5
```

## BenchmarkingBenchmark inference speed:
```bash
python -m dontblink.ml.infer \
  --model models/custom_best.pt \
  --iterations 100 \
  --device cpu
```
