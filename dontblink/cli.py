import argparse
import sys
import os
import cv2
import time
import logging
import traceback
from pathlib import Path
from datetime import datetime, timezone

from tqdm import tqdm

from . import __version__
from .config import Config, setup_logging
from .detection import DetectionService
from .video_processor import VideoProcessor
from .timelapse import TimelapseGenerator
from .utils import (
    get_organized_output_paths,
    compute_video_fingerprint,
    get_model_fingerprint,
    write_run_json,
    check_video_readable,
)

logger = logging.getLogger(__name__)


def _setup_global_flags(args):
    """Apply global CLI flags (--verbose, --log) to logging config."""
    log_level = logging.DEBUG if getattr(args, 'verbose', False) else logging.INFO
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    
    log_path = getattr(args, 'log', None)
    if log_path:
        os.makedirs(os.path.dirname(log_path) or '.', exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers,
        force=True,
    )


def _print_resolved_config(config: Config):
    """Print the resolved configuration and exit."""
    import yaml
    print("--- Resolved Configuration ---")
    print(yaml.dump(config.to_dict(), default_flow_style=False, sort_keys=False))
    print("---")

def download_model(args):
    """Download model weights to cache."""
    from .model_manager import download_model as do_download, get_cache_dir, get_registry_entry, MODEL_REGISTRY

    model_name = getattr(args, 'model', None)

    if getattr(args, 'list', False):
        print("Available models:")
        for name, entry in MODEL_REGISTRY.items():
            marker = " (default)" if name == "printhead_v2.0" else ""
            print(f"  {name}{marker}")
            print(f"    {entry['description']}")
            print(f"    Size: {entry['size_bytes'] / (1024*1024):.1f} MB")
        print(f"\nCache directory: {get_cache_dir()}")
        return 0

    force = getattr(args, 'force', False)

    try:
        path = do_download(model_name=model_name, force=force)
        print(f"\nModel ready: {path}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def doctor(args):
    """Run system diagnostics and print a copy-paste friendly report."""
    import platform
    import shutil
    import json as json_mod

    config = Config(args.config)
    diag = {}

    # --- System ---
    diag['system'] = {
        'os': f"{platform.system()} {platform.release()}",
        'os_version': platform.version(),
        'machine': platform.machine(),
        'python': platform.python_version(),
        'python_path': sys.executable,
    }

    try:
        cpu_brand = platform.processor() or "unknown"
    except Exception:
        cpu_brand = "unknown"
    diag['system']['cpu'] = cpu_brand

    try:
        import psutil
        mem = psutil.virtual_memory()
        diag['system']['ram_total_gb'] = round(mem.total / (1024 ** 3), 1)
        diag['system']['ram_available_gb'] = round(mem.available / (1024 ** 3), 1)
    except ImportError:
        diag['system']['ram_total_gb'] = 'unknown (install psutil)'
        diag['system']['ram_available_gb'] = 'unknown'

    # --- GPU / Torch ---
    diag['gpu'] = {}
    try:
        import torch
        diag['gpu']['torch_version'] = torch.__version__
        diag['gpu']['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            diag['gpu']['cuda_version'] = torch.version.cuda or 'unknown'
            diag['gpu']['cuda_device'] = torch.cuda.get_device_name(0)
            diag['gpu']['cuda_memory_gb'] = round(
                torch.cuda.get_device_properties(0).total_mem / (1024 ** 3), 1
            )
        mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        diag['gpu']['mps_available'] = mps_available
    except ImportError:
        diag['gpu']['torch_version'] = 'NOT INSTALLED'
        diag['gpu']['cuda_available'] = False
        diag['gpu']['mps_available'] = False

    # --- OpenCV ---
    diag['opencv'] = {}
    try:
        diag['opencv']['version'] = cv2.__version__
        diag['opencv']['build_info_ffmpeg'] = 'ffmpeg' in cv2.getBuildInformation().lower()
    except Exception as e:
        diag['opencv']['version'] = f'ERROR: {e}'

    # --- ffmpeg ---
    ffmpeg_path = shutil.which('ffmpeg')
    diag['ffmpeg'] = {
        'installed': ffmpeg_path is not None,
        'path': ffmpeg_path or 'not found',
    }

    # --- Model ---
    from .model_manager import get_cache_dir, MODEL_REGISTRY, DEFAULT_MODEL
    model_path = config.model_path
    cache_dir = get_cache_dir()
    cache_file = cache_dir / MODEL_REGISTRY[DEFAULT_MODEL]["filename"]
    diag['model'] = {
        'configured_path': model_path,
        'local_exists': os.path.exists(model_path),
        'cache_dir': str(cache_dir),
        'cached_exists': cache_file.exists(),
    }
    resolved = model_path if os.path.exists(model_path) else (str(cache_file) if cache_file.exists() else None)
    if resolved:
        from .utils import get_model_fingerprint
        diag['model']['resolved_path'] = resolved
        diag['model']['size_mb'] = round(os.path.getsize(resolved) / (1024 * 1024), 1)
        try:
            diag['model']['fingerprint'] = get_model_fingerprint(resolved)
        except Exception:
            diag['model']['fingerprint'] = 'error'
    else:
        diag['model']['resolved_path'] = None
        diag['model']['note'] = 'No model found. Run: dontblink download-model'

    # --- Disk ---
    output_dir = config.get('processing.output_base_dir', 'outputs')
    check_dir = output_dir if os.path.exists(output_dir) else '.'
    try:
        usage = shutil.disk_usage(check_dir)
        diag['disk'] = {
            'output_dir': output_dir,
            'free_gb': round(usage.free / (1024 ** 3), 1),
            'total_gb': round(usage.total / (1024 ** 3), 1),
        }
    except Exception:
        diag['disk'] = {'output_dir': output_dir, 'free_gb': 'unknown'}

    # --- Config ---
    diag['config'] = {
        'path': args.config or 'default (no file)',
        'device_setting': config.device,
        'confidence': config.confidence,
        'batch_size': config.get('processing.batch_size', 1),
    }

    # --- Version ---
    diag['dontblink_version'] = __version__

    # --- Output ---
    if getattr(args, 'json', False):
        print(json_mod.dumps(diag, indent=2, default=str))
        return 0

    lines = []
    lines.append("=" * 55)
    lines.append("  Dont-Blink System Diagnostics")
    lines.append("=" * 55)

    lines.append(f"\n  Dont-Blink version: {diag['dontblink_version']}")

    lines.append(f"\n  --- System ---")
    s = diag['system']
    lines.append(f"  OS:             {s['os']}")
    lines.append(f"  Machine:        {s['machine']}")
    lines.append(f"  CPU:            {s['cpu']}")
    lines.append(f"  RAM:            {s.get('ram_available_gb', '?')} GB free / {s.get('ram_total_gb', '?')} GB total")
    lines.append(f"  Python:         {s['python']} ({s['python_path']})")

    lines.append(f"\n  --- GPU / PyTorch ---")
    g = diag['gpu']
    lines.append(f"  PyTorch:        {g.get('torch_version', 'not installed')}")
    lines.append(f"  CUDA available: {g.get('cuda_available', False)}")
    if g.get('cuda_available'):
        lines.append(f"  CUDA version:   {g.get('cuda_version')}")
        lines.append(f"  CUDA device:    {g.get('cuda_device')}")
        lines.append(f"  CUDA memory:    {g.get('cuda_memory_gb')} GB")
    lines.append(f"  MPS available:  {g.get('mps_available', False)}")

    lines.append(f"\n  --- OpenCV ---")
    o = diag['opencv']
    lines.append(f"  Version:        {o.get('version', 'unknown')}")
    lines.append(f"  FFmpeg in build: {o.get('build_info_ffmpeg', 'unknown')}")

    lines.append(f"\n  --- ffmpeg ---")
    f = diag['ffmpeg']
    lines.append(f"  Installed:      {f['installed']}")
    lines.append(f"  Path:           {f['path']}")

    lines.append(f"\n  --- Model ---")
    m = diag['model']
    lines.append(f"  Configured:     {m['configured_path']}")
    lines.append(f"  Local exists:   {m['local_exists']}")
    lines.append(f"  Cache dir:      {m['cache_dir']}")
    lines.append(f"  Cached exists:  {m['cached_exists']}")
    if m.get('resolved_path'):
        lines.append(f"  Resolved:       {m['resolved_path']}")
        lines.append(f"  Size:           {m.get('size_mb', '?')} MB")
        lines.append(f"  Fingerprint:    {m.get('fingerprint', '?')}")
    else:
        lines.append(f"  Status:         NOT FOUND - run: dontblink download-model")

    lines.append(f"\n  --- Disk ---")
    d = diag['disk']
    lines.append(f"  Output dir:     {d['output_dir']}")
    lines.append(f"  Free space:     {d.get('free_gb', '?')} GB / {d.get('total_gb', '?')} GB")

    lines.append(f"\n  --- Config ---")
    c = diag['config']
    lines.append(f"  Config file:    {c['path']}")
    lines.append(f"  Device:         {c['device_setting']}")
    lines.append(f"  Confidence:     {c['confidence']}")
    lines.append(f"  Batch size:     {c['batch_size']}")

    lines.append("=" * 55)

    report = "\n".join(lines)
    print(report)

    if getattr(args, 'copy', False):
        try:
            import subprocess
            proc = subprocess.run(
                ['pbcopy'] if platform.system() == 'Darwin' else ['xclip', '-selection', 'clipboard'],
                input=report.encode(), check=True,
            )
            print("\n(Copied to clipboard)")
        except Exception:
            print(f"\n(Could not copy to clipboard — paste the output above)")

    return 0


def benchmark_model(args):
    """Benchmark model inference performance."""
    import numpy as np
    from statistics import mean, stdev
    
    config = Config(args.config)
    setup_logging(config)
    
    # Override device if specified
    if args.device:
        original_device = config.get('detection.device', 'auto')
        config.config['detection']['device'] = args.device
        logger.info(f"Overriding device: {original_device} -> {args.device}")
    
    logger.info("Starting model benchmark...")
    
    # Initialize detection service
    detection_service = DetectionService(config)
    device = detection_service.device
    
    # Get model info
    model = detection_service.model.model
    inference_size = getattr(detection_service, 'inference_size', 320)
    batch_size = args.batch_size if args.batch_size else config.get('processing.batch_size', 1)
    
    print("\n" + "="*70)
    print("MODEL INFERENCE BENCHMARK")
    print("="*70)
    print(f"Device: {device}")
    print(f"Model: {type(model).__name__}")
    print(f"Inference size: {inference_size}x{inference_size}")
    print(f"Batch size: {batch_size}")
    print(f"Warmup iterations: {args.warmup}")
    print(f"Test iterations: {args.iterations}")
    print("="*70 + "\n")
    
    # Generate test frames
    test_frames = []
    for _ in range(max(batch_size, 8)):  # Generate enough for max batch size
        frame = np.random.randint(0, 255, (inference_size, inference_size, 3), dtype=np.uint8)
        test_frames.append(frame)
    
    # Warmup
    print("Warming up model...")
    for _ in range(args.warmup):
        if batch_size > 1:
            detection_service.detect_batch(test_frames[:batch_size])
        else:
            detection_service.detect(test_frames[0])
    
    # Get initial memory (if psutil available)
    try:
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        has_psutil = True
    except ImportError:
        has_psutil = False
        initial_memory = 0
    
    # Benchmark single frame inference
    print(f"\n{'='*60}")
    print("SINGLE FRAME INFERENCE")
    print(f"{'='*60}")
    
    single_times = []
    for i in range(args.iterations):
        start = time.perf_counter()
        detection_service.detect(test_frames[0])
        end = time.perf_counter()
        single_times.append((end - start) * 1000)  # Convert to ms
    
    single_mean = mean(single_times)
    single_std = stdev(single_times) if len(single_times) > 1 else 0
    single_fps = 1000 / single_mean if single_mean > 0 else 0
    
    print(f"Mean inference time: {single_mean:.2f} ± {single_std:.2f} ms")
    print(f"Throughput: {single_fps:.2f} FPS")
    print(f"Min: {min(single_times):.2f} ms")
    print(f"Max: {max(single_times):.2f} ms")
    
    # Benchmark batch inference (if batch_size > 1)
    per_frame_time = single_mean
    batch_fps = single_fps
    
    if batch_size > 1:
        print(f"\n{'='*70}")
        print(f"BATCH INFERENCE (batch_size={batch_size})")
        print(f"{'='*70}")
        
        batch_times = []
        for i in range(args.iterations):
            start = time.perf_counter()
            detection_service.detect_batch(test_frames[:batch_size])
            end = time.perf_counter()
            batch_times.append((end - start) * 1000)  # Convert to ms
        
        batch_mean = mean(batch_times)
        batch_std = stdev(batch_times) if len(batch_times) > 1 else 0
        batch_fps = (batch_size * 1000) / batch_mean if batch_mean > 0 else 0
        per_frame_time = batch_mean / batch_size
        
        print(f"Mean batch time: {batch_mean:.2f} ± {batch_std:.2f} ms")
        print(f"Per frame time: {per_frame_time:.2f} ms")
        print(f"Batch throughput: {batch_fps:.2f} FPS")
        print(f"Speedup vs single: {single_mean / per_frame_time:.2f}x")
        print(f"Min: {min(batch_times):.2f} ms")
        print(f"Max: {max(batch_times):.2f} ms")
        print(f"95th percentile: {sorted(batch_times)[int(len(batch_times) * 0.95)]:.2f} ms")
    
    # Memory usage
    if has_psutil:
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = final_memory - initial_memory
        
        print(f"\n{'='*70}")
        print("MEMORY USAGE")
        print(f"{'='*70}")
        print(f"Process memory: {final_memory:.2f} MB")
        
        # GPU memory (if CUDA)
        if device == 'cuda':
            import torch
            if torch.cuda.is_available():
                print(f"\nGPU Memory:")
                print(f"  Allocated: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")
                print(f"  Reserved: {torch.cuda.memory_reserved() / 1024 / 1024:.2f} MB")
    else:
        print(f"\n{'='*70}")
        print("MEMORY USAGE")
        print(f"{'='*70}")
        print("(Install 'psutil' for memory statistics: pip install psutil)")
    
    # Video processing simulation
    print(f"\n{'='*70}")
    print("VIDEO PROCESSING SIMULATION")
    print(f"{'='*70}")
    
    # Simulate processing videos at different FPS
    video_scenarios = [
        (30, 60),      # 30 FPS, 1 minute
        (30, 300),     # 30 FPS, 5 minutes
        (30, 5400),    # 30 FPS, 1.5 hours (90 minutes)
        (60, 60),      # 60 FPS, 1 minute
        (60, 5400),    # 60 FPS, 1.5 hours (90 minutes)
    ]
    
    for video_fps, duration_sec in video_scenarios:
        total_frames = video_fps * duration_sec
        processing_time = (per_frame_time / 1000) * total_frames
        realtime_factor = (duration_sec / processing_time) if processing_time > 0 else 0
        
        # Format duration nicely
        if duration_sec < 60:
            duration_str = f"{duration_sec}s"
        elif duration_sec < 3600:
            duration_str = f"{duration_sec/60:.1f} min"
        else:
            hours = duration_sec / 3600
            duration_str = f"{hours:.1f} hours"
        
        print(f"\n{video_fps} FPS video, {duration_str} ({total_frames:,} frames):")
        print(f"  Processing time: {processing_time:.1f}s ({processing_time/60:.1f} min)")
        print(f"  Realtime factor: {realtime_factor:.2f}x", end="")
        if realtime_factor >= 1.0:
            print(" ✓ Faster than realtime")
        else:
            print(f" ⚠ {1/realtime_factor:.2f}x slower than realtime")
    
    # Add practical explanation
    print(f"\n{'='*70}")
    print("PRACTICAL INTERPRETATION")
    print(f"{'='*70}")
    print(f"Your model processes at {single_fps:.1f} FPS ({single_mean:.1f} ms per frame)")
    print(f"\nFor a 1.5 hour (90 minute) printing video:")
    
    # Calculate for 30 FPS (most common)
    video_fps_30 = 30
    duration_90min = 90 * 60  # 5400 seconds
    total_frames_30 = video_fps_30 * duration_90min
    processing_time_30 = (per_frame_time / 1000) * total_frames_30
    
    print(f"  • 30 FPS video ({total_frames_30:,} frames):")
    print(f"    → Processing time: {processing_time_30/60:.1f} minutes ({processing_time_30/3600:.2f} hours)")
    print(f"    → This is {duration_90min/processing_time_30:.2f}x faster than the video length")
    
    # Calculate for 60 FPS
    video_fps_60 = 60
    total_frames_60 = video_fps_60 * duration_90min
    processing_time_60 = (per_frame_time / 1000) * total_frames_60
    
    print(f"  • 60 FPS video ({total_frames_60:,} frames):")
    print(f"    → Processing time: {processing_time_60/60:.1f} minutes ({processing_time_60/3600:.2f} hours)")
    print(f"    → This is {duration_90min/processing_time_60:.2f}x faster than the video length")
    
    print(f"\nNote: Actual processing may be faster due to frame skipping.")
    print(f"      The system only processes every Nth frame (configurable).")
    
    # Performance recommendations
    print(f"\n{'='*70}")
    print("PERFORMANCE ASSESSMENT")
    print(f"{'='*70}")
    
    if single_fps >= 30:
        print("✓ EXCELLENT: Can process 30+ FPS videos in realtime")
        rating = "Excellent"
    elif single_fps >= 15:
        print("✓ GOOD: Can process 15-30 FPS videos in realtime")
        rating = "Good"
    elif single_fps >= 5:
        print("⚠ ACCEPTABLE: Suitable for timelapse processing (low FPS)")
        rating = "Acceptable"
    else:
        print("✗ SLOW: May struggle with real-time processing")
        rating = "Slow"
    
    print(f"\nDevice recommendations:")
    if device == 'cpu':
        print("  💡 Consider using GPU (CUDA/MPS) for 2-5x speedup")
    elif device == 'mps':
        print("  ✓ Using Apple Silicon GPU (MPS)")
    elif device == 'cuda':
        print("  ✓ Using NVIDIA GPU (CUDA)")
    
    if batch_size > 1 and batch_fps > single_fps * 1.2:
        print(f"  💡 Batch processing ({batch_size}) provides {single_mean / per_frame_time:.1f}x speedup")
    elif batch_size == 1 and device != 'cpu':
        print(f"  💡 Enable batch processing (batch_size=4-16) for better GPU utilization")
    
    # System requirements estimate
    print(f"\n{'='*70}")
    print("SYSTEM REQUIREMENTS ESTIMATE")
    print(f"{'='*70}")
    print(f"Minimum recommended:")
    if device == 'cpu':
        print(f"  - CPU: Modern multi-core processor")
        print(f"  - RAM: 2-4 GB available")
    else:
        print(f"  - GPU: {device.upper()} compatible device")
        print(f"  - RAM: 4-8 GB available")
    
    print(f"\nPerformance rating: {rating}")
    print("="*70 + "\n")
    
    return 0


def create_default_config(config_path: str):
    """Create a default configuration file."""
    from .config import Config
    config = Config()
    config.save_to_file(config_path)
    print(f"Created default configuration file: {config_path}")


def process_video(args):
    """Process a video file — the 'golden path' command."""
    config = Config(args.config)
    
    if getattr(args, 'verbose', False) or getattr(args, 'log', None):
        _setup_global_flags(args)
    else:
        setup_logging(config)
    
    # Apply CLI overrides to config
    rotation_override = getattr(args, 'rotation', None)
    if rotation_override is not None:
        config.set('processing.video_rotation', rotation_override)
        logger.info(f"CLI rotation override: {rotation_override} degrees")
    
    if getattr(args, 'print_config', False):
        _print_resolved_config(config)
        return 0
    
    input_path = args.input
    logger.info("Starting video processing")
    
    # Early validation with actionable errors
    try:
        video_info = check_video_readable(input_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    total_frames = video_info['total_frames']
    
    # Determine output paths
    if args.output:
        frames_dir = args.output
        organize_by_video = False
        base_dir = None
    else:
        base_dir = config.get('processing.output_base_dir', 'outputs')
        organize_by_video = config.get('processing.organize_by_video', True)
        paths = get_organized_output_paths(input_path, base_dir, organize_by_video)
        frames_dir = paths['frames_dir']
        logger.info(f"Using organized output structure: {frames_dir}")
    
    # Load model
    detection_service = DetectionService(config)
    video_processor = VideoProcessor(config, detection_service)
    
    progress_bar = tqdm(
        total=total_frames,
        unit="frame",
        desc="Processing",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )
    
    def progress_callback(frame_count, total):
        progress_bar.update(1)
    
    wall_start = time.perf_counter()
    
    try:
        stats = video_processor.process_video_file(
            input_path, 
            frames_dir,
            progress_callback=progress_callback
        )
        progress_bar.close()
        
        wall_elapsed = time.perf_counter() - wall_start
        
        timelapse_path = None
        if not args.output and organize_by_video:
            paths = get_organized_output_paths(input_path, base_dir, organize_by_video)
            timelapse_gen = TimelapseGenerator(config)
            timelapse_path = timelapse_gen.create_timelapse(frames_dir, paths['timelapse_path'])
        
        # End-of-run summary
        inference_times = stats.get('inference_times_ms', [])
        avg_inference = sum(inference_times) / len(inference_times) if inference_times else 0
        
        print(f"\n{'='*50}")
        print(f"  PROCESSING COMPLETE")
        print(f"{'='*50}")
        print(f"  Frames in video:  {stats.get('frames_total', total_frames):,}")
        print(f"  Frames analyzed:  {stats.get('frames_analyzed', 0):,}")
        print(f"  Frames selected:  {stats['frames_captured']:,}")
        print(f"  Avg inference:    {avg_inference:.1f} ms/frame")
        print(f"  Wall time:        {wall_elapsed:.1f}s")
        if stats.get('rotation_applied', 0) != 0:
            print(f"  Rotation applied: {stats['rotation_applied']} degrees")
        print(f"  Device:           {detection_service.device}")
        print(f"  Frames saved to:  {frames_dir}")
        if timelapse_path:
            print(f"  Timelapse:        {timelapse_path}")
        else:
            print(f"  Create timelapse: dontblink create-timelapse {frames_dir}")
        print(f"{'='*50}")
        
        # Write run.json
        output_base = paths['base_dir'] if not args.output and organize_by_video else frames_dir
        try:
            run_data = {
                'version': __version__,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'input': {
                    'video': os.path.basename(input_path),
                    'fingerprint': compute_video_fingerprint(input_path),
                    'fps': stats.get('video_fps', 0),
                    'total_frames': stats.get('frames_total', 0),
                    'resolution': f"{stats.get('video_width', 0)}x{stats.get('video_height', 0)}",
                },
                'model': {
                    'path': config.model_path,
                    'fingerprint': get_model_fingerprint(config.model_path) if os.path.exists(config.model_path) else 'unknown',
                },
                'device': detection_service.device,
                'config': config.to_dict(),
                'stats': {
                    'frames_analyzed': stats.get('frames_analyzed', 0),
                    'frames_selected': stats['frames_captured'],
                    'avg_inference_ms': round(avg_inference, 2),
                    'wall_time_s': round(wall_elapsed, 2),
                    'rotation_applied': stats.get('rotation_applied', 0),
                    'frame_skip': stats.get('frame_skip', 0),
                    'batch_size': stats.get('batch_size', 1),
                },
                'output': {
                    'frames_dir': frames_dir,
                    'timelapse': timelapse_path,
                },
            }
            run_path = write_run_json(output_base, run_data)
            logger.info(f"Run metadata saved to {run_path}")
        except Exception as e:
            logger.warning(f"Could not write run.json: {e}")
        
        return 0
    except KeyboardInterrupt:
        progress_bar.close()
        print("\nProcessing interrupted by user.", file=sys.stderr)
        return 130
    except Exception as e:
        progress_bar.close()
        logger.error(f"Processing failed: {e}")
        if getattr(args, 'verbose', False):
            traceback.print_exc()
        print(f"\nError: {e}", file=sys.stderr)
        return 1


def create_timelapse(args):
    """Create timelapse video from frames."""
    config = Config(args.config)
    setup_logging(config)
    
    timelapse_gen = TimelapseGenerator(config)
    
    fps = args.fps if args.fps else config.get('timelapse.fps', 15)
    
    try:
        output_path = timelapse_gen.create_timelapse(args.input, args.output, fps=fps)
        print(f"Timelapse created: {output_path}")
        return 0
    except Exception as e:
        logger.error(f"Timelapse creation failed: {e}")
        print(f"Error: {e}", file=sys.stderr)
        return 1


def process_camera(args):
    """Process live camera feed."""
    config = Config(args.config)
    
    if getattr(args, 'verbose', False) or getattr(args, 'log', None):
        _setup_global_flags(args)
    else:
        setup_logging(config)
    
    detection_service = DetectionService(config)
    video_processor = VideoProcessor(config, detection_service)
    
    if args.list_cameras:
        cameras = video_processor.list_available_cameras()
        if cameras:
            print("Available cameras:")
            for idx in cameras:
                print(f"  Camera {idx}")
        else:
            print("No cameras found")
        return 0
    
    try:
        stats = video_processor.process_camera(args.camera_index, args.output)
        print(f"Captured {stats['frames_captured']} frames")
        return 0
    except Exception as e:
        logger.error(f"Camera processing failed: {e}")
        if getattr(args, 'verbose', False):
            traceback.print_exc()
        print(f"Error: {e}", file=sys.stderr)
        return 1


def test_image(args):
    """Test model on image(s) with visualization."""
    config = Config(args.config)
    setup_logging(config)
    
    detection_service = DetectionService(config)
    
    # Determine if input is a file or directory
    input_path = Path(args.input)
    
    if input_path.is_file():
        image_paths = [input_path]
    elif input_path.is_dir():
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_paths = [p for p in input_path.iterdir() 
                      if p.suffix.lower() in image_extensions]
        if not image_paths:
            print(f"Error: No images found in {input_path}", file=sys.stderr)
            return 1
        image_paths.sort()
    else:
        print(f"Error: Input path does not exist: {input_path}", file=sys.stderr)
        return 1
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        if input_path.is_file():
            output_path = input_path.parent / f"{input_path.stem}_detected{input_path.suffix}"
        else:
            output_path = input_path.parent / f"{input_path.name}_detected"
    
    is_directory = input_path.is_dir()
    if is_directory:
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        total_detections = 0
        for image_path in image_paths:
            frame = cv2.imread(str(image_path))
            if frame is None:
                logger.warning(f"Could not read image: {image_path}")
                continue
            
            detections = detection_service.detect(frame, confidence=args.confidence)
            total_detections += len(detections)
            
            vis_frame = detection_service.visualize_detections(
                frame, detections, 
                show_confidence=not args.no_confidence, 
                show_center=not args.no_center
            )
            
            if is_directory:
                output_file = output_path / f"{image_path.stem}_detected{image_path.suffix}"
            else:
                output_file = output_path
            
            cv2.imwrite(str(output_file), vis_frame)
            
            print(f"{image_path.name}: {len(detections)} detection(s)")
            for i, det in enumerate(detections):
                print(f"Detection {i+1}: confidence={det.confidence:.3f}, "
                      f"x_norm={det.x_normalized:.3f}, y_norm={det.y_normalized:.3f}")
            
            if args.show:
                cv2.imshow(f"Detection: {image_path.name}", vis_frame)
                print("Press any key to continue...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        
        print(f"\nProcessed {len(image_paths)} image(s), found {total_detections} total detection(s)")
        if is_directory:
            print(f"Output saved to: {output_path}")
        else:
            print(f"Output saved to: {output_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Image testing failed: {e}")
        print(f"Error: {e}", file=sys.stderr)
        return 1


def visualize_video(args):
    """Process video and draw bounding boxes on each frame."""
    config = Config(args.config)
    setup_logging(config)
    
    input_video = Path(args.input)
    if not input_video.exists() or not input_video.is_file():
        print(f"Error: Video file not found: {input_video}", file=sys.stderr)
        return 1
    
    if args.output:
        output_video = Path(args.output)
    else:
        output_video = input_video.parent / f"{input_video.stem}_detected{input_video.suffix}"
    
    detection_service = DetectionService(config)
    video_processor = VideoProcessor(config, detection_service)
    
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        print(f"Error: Could not open video file: {input_video}", file=sys.stderr)
        return 1
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps == 0 or width == 0 or height == 0:
        print(f"Error: Could not read video properties", file=sys.stderr)
        cap.release()
        return 1
    
    rotation = video_processor._get_video_rotation(cap)
    
    if rotation in [90, 270]:
        output_width, output_height = height, width
    else:
        output_width, output_height = width, height
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video), fourcc, fps, (output_width, output_height))
    
    if not out.isOpened():
        print(f"Error: Could not create output video file: {output_video}", file=sys.stderr)
        cap.release()
        return 1
    
    confidence = args.confidence if args.confidence is not None else config.confidence
    
    progress_bar = tqdm(
        total=total_frames,
        unit="frame",
        desc="Processing video",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )
    
    frame_count = 0
    detections_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if rotation != 0:
                frame = video_processor._apply_rotation(frame, rotation)
            
            detections = detection_service.detect(frame, confidence=confidence)
            
            if detections:
                detections_count += len(detections)
                frame = detection_service.visualize_detections(
                    frame,
                    detections,
                    show_confidence=not args.no_confidence,
                    show_center=not args.no_center
                )
            
            out.write(frame)
            frame_count += 1
            progress_bar.update(1)
        
        progress_bar.close()
        cap.release()
        out.release()
        
        print(f"\nProcessed {frame_count} frames")
        print(f"Found {detections_count} total detection(s)")
        print(f"Output saved to: {output_video}")
        
        return 0
        
    except Exception as e:
        progress_bar.close()
        cap.release()
        out.release()
        logger.error(f"Video visualization failed: {e}")
        print(f"Error: {e}", file=sys.stderr)
        return 1


def extract_frames(args):
    """Extract frames from video(s) at regular intervals."""
    input_path = Path(args.input)
    if args.output:
        output_folder = Path(args.output)
    else:
        output_folder = Path("data/extracted_frames")
    
    output_folder.mkdir(parents=True, exist_ok=True)
    
    if input_path.is_file():
        video_files = [input_path]
    elif input_path.is_dir():
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV'}
        video_files = [p for p in input_path.iterdir() 
                      if p.suffix in video_extensions]
        if not video_files:
            print(f"Error: No video files found in {input_path}", file=sys.stderr)
            return 1
        video_files.sort()
    else:
        print(f"Error: Input path does not exist: {input_path}", file=sys.stderr)
        return 1
    
    interval_seconds = args.interval
    
    total_frames_extracted = 0
    
    for video_path in video_files:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning(f"Could not open video: {video_path}")
            continue
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        if fps == 0:
            logger.warning(f"Could not determine FPS for {video_path}, skipping")
            cap.release()
            continue
        
        frame_interval = int(fps * interval_seconds)
        expected_frames = int(total_frames / frame_interval) + 1
        
        video_name = video_path.stem
        video_output_folder = output_folder / video_name
        video_output_folder.mkdir(parents=True, exist_ok=True)
        
        print(f"\nProcessing: {video_path.name}")
        print(f"  FPS: {fps:.2f}, Duration: {duration:.1f}s, Expected frames: {expected_frames}")
        
        progress_bar = tqdm(
            total=expected_frames,
            unit="frame",
            desc=f"Extracting {video_name[:20]}"
        )
        
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                timestamp = frame_count / fps
                filename = video_output_folder / f"frame_{saved_count:06d}_t{timestamp:06.2f}s.jpg"
                cv2.imwrite(str(filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved_count += 1
                progress_bar.update(1)
            
            frame_count += 1
        
        progress_bar.close()
        cap.release()
        total_frames_extracted += saved_count
        print(f"  Extracted {saved_count} frames to {video_output_folder}")
    
    print(f"\nTotal: Extracted {total_frames_extracted} frames from {len(video_files)} video(s)")
    print(f"Frames saved to: {output_folder}")
    return 0


def _add_video_process_args(parser):
    """Add shared arguments for video processing commands."""
    parser.add_argument('input', help='Input video file path')
    parser.add_argument('output', nargs='?', default=None,
                        help='Output folder for frames (optional: uses organized structure if not specified)')
    parser.add_argument('--rotation', type=int, default=None, choices=[0, 90, 180, 270],
                        help='Override video rotation (degrees: 0, 90, 180, 270)')
    parser.add_argument('--print-config', action='store_true',
                        help='Print resolved configuration and exit')


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Dont-Blink AI - Printhead Tracking for 3D Printer Timelapses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a video (the "happy path" — organized output with timelapse)
  dontblink process input.mp4

  # Same thing, explicit subcommand
  dontblink process-video input.mp4 output_folder/

  # See what config values will be used
  dontblink process input.mp4 --print-config

  # Override rotation + verbose logging to a file
  dontblink process input.mp4 --rotation 90 --verbose --log run.log

  # Process live camera (camera 0)
  dontblink process-camera 0 output_folder/

  # Create timelapse from frames
  dontblink create-timelapse frames_folder/ output.mp4

  # Test model on single image
  dontblink test-image image.jpg output.jpg

  # Extract frames from video(s) every 2 seconds
  dontblink extract-frames data/raw_videos/ --interval 2

  # Visualize detections on video (draw bounding boxes)
  dontblink visualize-video input.mp4 output.mp4

  # Benchmark model inference performance
  dontblink benchmark --device cpu --iterations 100

  # System diagnostics (paste into bug reports)
  dontblink doctor
  dontblink doctor --copy
  dontblink doctor --json

  # Create default config file
  dontblink create-config config.yaml
        """
    )
    
    # Global flags
    parser.add_argument('--config', '-c', type=str, default=None,
                        help='Path to configuration file (YAML)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose (DEBUG) logging with stack traces')
    parser.add_argument('--log', type=str, default=None, metavar='PATH',
                        help='Write logs to file at PATH')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # "process" — the promoted happy-path alias
    parser_process = subparsers.add_parser('process',
                                           help='Process a video file (recommended)')
    _add_video_process_args(parser_process)
    
    # "process-video" — kept for backward compat
    parser_video = subparsers.add_parser('process-video',
                                         help='Process a video file')
    _add_video_process_args(parser_video)
    
    parser_camera = subparsers.add_parser('process-camera', help='Process live camera feed')
    parser_camera.add_argument('camera_index', type=int, help='Camera device index')
    parser_camera.add_argument('output', help='Output folder for frames')
    parser_camera.add_argument('--list-cameras', action='store_true', help='List available cameras and exit')
    
    parser_timelapse = subparsers.add_parser('create-timelapse', help='Create timelapse video')
    parser_timelapse.add_argument('input', help='Input folder containing frames')
    parser_timelapse.add_argument('output', nargs='?', default=None, help='Output video path (default: timelapse.mp4 in input folder)')
    parser_timelapse.add_argument('--fps', type=int, default=None, help='Frames per second (overrides config)')
    
    parser_config = subparsers.add_parser('create-config', help='Create default config file')
    parser_config.add_argument('output', help='Output config file path')
    
    parser_test = subparsers.add_parser('test-image', help='Test model on image(s) with visualization')
    parser_test.add_argument('input', help='Input image file or folder containing images')
    parser_test.add_argument('output', nargs='?', default=None, help='Output image file or folder (default: input_detected.jpg or input_detected/)')
    parser_test.add_argument('--confidence', type=float, default=None, help='Confidence threshold (overrides config)')
    parser_test.add_argument('--show', action='store_true', help='Display images in window (press any key to continue)')
    parser_test.add_argument('--no-confidence', action='store_true', help='Hide confidence scores in visualization')
    parser_test.add_argument('--no-center', action='store_true', help='Hide center point markers in visualization')
    
    parser_visualize = subparsers.add_parser('visualize-video', help='Process video and draw bounding boxes on each frame')
    parser_visualize.add_argument('input', help='Input video file')
    parser_visualize.add_argument('output', nargs='?', default=None, help='Output video file (default: input_detected.mp4)')
    parser_visualize.add_argument('--confidence', type=float, default=None, help='Confidence threshold (overrides config)')
    parser_visualize.add_argument('--no-confidence', action='store_true', help='Hide confidence scores in visualization')
    parser_visualize.add_argument('--no-center', action='store_true', help='Hide center point markers in visualization')
    
    parser_extract = subparsers.add_parser('extract-frames', help='Extract frames from video(s) at regular intervals')
    parser_extract.add_argument('input', help='Input video file or folder containing videos')
    parser_extract.add_argument('output', nargs='?', default=None, help='Output folder for extracted frames (default: data/extracted_frames)')
    parser_extract.add_argument('--interval', type=float, default=2.0, help='Interval in seconds between frames (default: 2.0)')
    
    parser_benchmark = subparsers.add_parser('benchmark', help='Benchmark model inference performance')
    parser_benchmark.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda', 'mps', 'auto'], help='Device to use for inference (overrides config: cpu, cuda, mps, or auto)')
    parser_benchmark.add_argument('--batch-size', type=int, default=None, help='Batch size to test (default: from config)')
    parser_benchmark.add_argument('--warmup', type=int, default=10, help='Number of warmup iterations (default: 10)')
    parser_benchmark.add_argument('--iterations', type=int, default=100, help='Number of test iterations (default: 100)')
    
    parser_doctor = subparsers.add_parser('doctor', help='Run system diagnostics for troubleshooting')
    parser_doctor.add_argument('--json', action='store_true', help='Output diagnostics as machine-readable JSON')
    parser_doctor.add_argument('--copy', action='store_true', help='Copy output to clipboard (macOS/Linux)')
    
    parser_download = subparsers.add_parser('download-model', help='Download model weights to local cache')
    parser_download.add_argument('--model', type=str, default=None, help='Model name from registry (default: latest)')
    parser_download.add_argument('--force', action='store_true', help='Re-download even if already cached')
    parser_download.add_argument('--list', action='store_true', help='List available models')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    if args.command in ('process', 'process-video'):
        return process_video(args)
    elif args.command == 'create-timelapse':
        return create_timelapse(args)
    elif args.command == 'process-camera':
        return process_camera(args)
    elif args.command == 'test-image':
        return test_image(args)
    elif args.command == 'visualize-video':
        return visualize_video(args)
    elif args.command == 'extract-frames':
        return extract_frames(args)
    elif args.command == 'benchmark':
        return benchmark_model(args)
    elif args.command == 'doctor':
        return doctor(args)
    elif args.command == 'download-model':
        return download_model(args)
    elif args.command == 'create-config':
        create_default_config(args.output)
        return 0
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
