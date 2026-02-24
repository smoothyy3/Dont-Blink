# Dont-Blink — AI Timelapse for 3D Printers

[![PyPI version](https://img.shields.io/pypi/v/dontblink)](https://pypi.org/project/dontblink/)
[![Python](https://img.shields.io/pypi/pyversions/dontblink)](https://pypi.org/project/dontblink/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

Dont-Blink watches your 3D print video and captures frames **only when the printhead is out of the way**, producing clean timelapses with no printhead photobombs.

<p align="center">
  <a href="https://youtu.be/nMrHcGVqUqU">
    <img src="https://img.youtube.com/vi/nMrHcGVqUqU/maxresdefault.jpg" width="600" alt="Demo video">
  </a>
  <br>
  <em>Click to watch the demo</em>
</p>

## How it works

1. A lightweight MobileNetV3-based model detects the printhead position in each frame
2. A tracker monitors printhead movement and captures frames when it parks
3. Captured frames are stitched into a smooth timelapse video

Works with **pre-recorded videos** and **live camera feeds**. Runs on CPU, Apple Silicon (MPS), or NVIDIA GPU.

## Quick start

### Install

```bash
pipx install dontblink
```

Or with pip:

```bash
pip install dontblink
```

The model weights (~13 MB) download automatically on first run.

### Process a video

```bash
dontblink process input.mp4
```

That's it. Output goes to `outputs/input/frames/` and `outputs/input/timelapse.mp4`.

## CLI reference

```
dontblink process <video>              Process a video (recommended entry point)
dontblink process <video> --rotation 90  Override video rotation
dontblink process <video> --print-config Show resolved config and exit

dontblink create-timelapse <dir> <out>  Stitch frames into a timelapse
dontblink process-camera <id> <out>     Process a live camera feed
dontblink visualize-video <in> <out>    Draw bounding boxes on every frame
dontblink test-image <image> [out]      Test model on a single image
dontblink extract-frames <dir>          Extract frames from videos at intervals

dontblink benchmark                     Benchmark inference speed
dontblink doctor                        System diagnostics (paste into bug reports)
dontblink download-model                Download / verify model weights

dontblink --verbose                     Debug logging
dontblink --config path/to/config.yaml  Custom config file
```

## Configuration

Dont-Blink works out of the box with sensible defaults. To customize, create a `config.yaml`:

```bash
dontblink create-config config.yaml
```

Key settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `detection.device` | `auto` | `auto`, `cpu`, `cuda`, or `mps` |
| `detection.confidence` | `0.5` | Detection threshold (0.0–1.0) |
| `processing.batch_size` | `1` | Frames per batch (increase for GPU) |
| `processing.video_rotation` | `0` | Manual rotation override (0/90/180/270) |
| `timelapse.fps` | `15` | Output timelapse frame rate |

See [`config.yaml.example`](config.yaml.example) for all options.

## Python API

```python
from dontblink.config import Config
from dontblink.detection import DetectionService
from dontblink.video_processor import VideoProcessor
from dontblink.timelapse import TimelapseGenerator

config = Config("config.yaml")
detection = DetectionService(config)
processor = VideoProcessor(config, detection)

stats = processor.process_video_file("input.mp4", "output/frames")

timelapse = TimelapseGenerator(config)
timelapse.create_timelapse("output/frames", "timelapse.mp4")
```

## Architecture

```
dontblink/
├── cli.py              Command-line interface
├── config.py           YAML configuration with defaults
├── detection.py        Detection service + printhead tracker
├── model_manager.py    Model download, caching, SHA-256 verification
├── video_processor.py  Video/camera processing with batch inference
├── timelapse.py        Timelapse generation from frames
├── utils.py            Device detection, validation, fingerprinting
├── types.py            DetectionResult type
└── ml/                 Training & inference
    ├── model.py        MobileNetV3-Small backbone + detection head
    ├── infer.py        Inference with batch support
    ├── train.py        Training loop with early stopping
    ├── dataset.py      YOLO-format dataset with augmentation
    ├── eval.py         Precision, recall, F1, IoU metrics
    └── config.py       Training hyperparameters
```

## Model

- **Backbone**: MobileNetV3-Small (ImageNet pre-trained)
- **Head**: presence score + bounding box (x, y, w, h) in normalized coordinates
- **Size**: ~13 MB
- **Inference**: ~16 ms/frame on CPU, ~5 ms on MPS/CUDA

Currently optimized for **Bambu Lab A1 Mini** but works with other printers given appropriate camera positioning.

## Training your own model

Install training dependencies and see the training guide:

```bash
pip install dontblink[train]
```

See [TRAINING.md](TRAINING.md) for dataset preparation, training, and evaluation.

## Troubleshooting

Run diagnostics and paste the output into bug reports:

```bash
dontblink doctor --copy
```

## Contributing

```bash
git clone https://github.com/smoothyy3/Dont-Blink.git
cd Dont-Blink
pip install -e ".[dev,train]"
```

## License

[GNU Affero General Public License v3.0](LICENSE) — Copyright 2025 Jonas Möbes
