# Dont-Blink — Clean Timelapses for 3D Printers

[![PyPI version](https://img.shields.io/pypi/v/dontblink)](https://pypi.org/project/dontblink/)
[![Python](https://img.shields.io/pypi/pyversions/dontblink)](https://pypi.org/project/dontblink/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

Dont-Blink post-processes your 3D print recordings into smooth timelapses where the object appears to grow on its own — no printhead in the frame. It uses local computer vision to detect when the printhead is parked or out of view, and selects only those frames.

**No cloud uploads. No firmware mods. No plugins. Just record your print, run one command.**

<p align="center">
  <a href="https://youtu.be/nMrHcGVqUqU">
    <img src="https://img.youtube.com/vi/nMrHcGVqUqU/maxresdefault.jpg" width="600" alt="Demo video">
  </a>
  <br>
  <em>Click to watch the demo</em>
</p>

## How it works

1. You record your 3D print with any camera (phone, webcam, IP cam).
2. Dont-Blink scans the video using a lightweight MobileNetV3 model to detect the printhead in each frame.
3. It selects frames where the printhead is **parked** (stationary) or **out of view**.
4. Those frames are stitched into a smooth timelapse video.

Everything runs locally on your machine — CPU, Apple Silicon (MPS), or NVIDIA GPU.

## What it needs to work well

Dont-Blink is **post-processing frame selection**, not magic. It needs something to select *from*:

- **Best case:** Your printer has a timelapse mode that parks the printhead (e.g., Bambu Studio Smooth Timelapse). Dont-Blink will find those parked frames automatically.
- **Easy setup:** Add a small G-code snippet to park the head every layer or every N layers. [See the printer setup guide.](docs/printer-setup.md)
- **Camera trick:** Position your camera so the printhead naturally moves out of frame during travel moves.

If your video has no consistent "clean moment," the output may be inconsistent. The tool will warn you if it detects this.

## How this differs from alternatives

| Tool | Approach | Pros | Cons |
|---|---|---|---|
| **Octolapse / Klipper timelapse** | Printer-controlled: parks head + takes snapshot per layer | Perfect frame-per-layer result | Adds print time, requires plugin/firmware config |
| **CyberBrick / hardware trigger kits** | External camera trigger synced to G-code | Works with DSLRs, high quality | Requires hardware purchase + wiring |
| **Dont-Blink** | Post-processing: finds "clean" frames from a continuous recording | No printer mods needed, any camera, runs after the fact | Needs a "clean moment" in the video (park/out-of-view) |

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
dontblink process print_recording.mp4
```

Tell the tool where your printhead parks (default is left). If it parks on the right or at the top, or moves out of frame, set `--capture-mode`:

```bash
dontblink process print_recording.mp4                           # default: left-park
dontblink process print_recording.mp4 --capture-mode right-park  # parks on the right
dontblink process print_recording.mp4 --capture-mode top-park    # parks at top
dontblink process print_recording.mp4 --capture-mode out-of-view # head moves out of frame
```

Output appears next to the input video:

```
print_recording.mp4           <- your input
print_recording/              <- created automatically
  frames/
    frame_000000.jpg
    ...
  timelapse.mp4               <- the result
```

### Help improve the model

The detection model gets better with diverse training data. You can contribute labeled frames from your own prints in about 2 minutes:

```bash
dontblink contribute print_recording.mp4
```

This scans your video, selects uncertain frames, opens a review UI in your browser, and packages the result for submission. No raw video is shared.

## CLI reference

```
dontblink process <video>              Process a video (recommended)
dontblink process <video> --rotation 90  Override video rotation
dontblink process <video> --print-config Show resolved config and exit

dontblink contribute <video>           Contribute labeled frames to improve the model

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

## Printer setup guide

For best results, your printer needs a repeatable "clean moment" — a point in each layer where the printhead is parked or out of view.

**[Read the full printer setup guide](docs/printer-setup.md)** for G-code snippets for Cura, PrusaSlicer, OrcaSlicer, Klipper, and Bambu Studio.

## Model

- **Backbone**: MobileNetV3-Small (ImageNet pre-trained)
- **Head**: presence score + bounding box (x, y, w, h) in normalized coordinates
- **Size**: ~13 MB
- **Inference**: ~16 ms/frame on CPU, ~5 ms on MPS/CUDA

Currently optimized for **Bambu Lab A1 Mini** but works with other printers given appropriate camera positioning. The `contribute` command helps expand printer coverage over time.

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

**Help improve the model** — the easiest way to contribute:

```bash
dontblink contribute your_print_video.mp4
```

**Develop locally:**

```bash
git clone https://github.com/smoothyy3/Dont-Blink.git
cd Dont-Blink
pip install -e ".[dev,train]"
```

## License

[GNU Affero General Public License v3.0](LICENSE) — Copyright 2025 Jonas Möbes
