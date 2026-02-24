"""
Model weight management: registry, caching, download, and verification.
"""
import os
import sys
import hashlib
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Update this when publishing a new model to GitHub Releases.
# After creating a GitHub release with the model attached, replace the URL below.
MODEL_REGISTRY = {
    "printhead_v2.0": {
        "filename": "custom_best.pt",
        "sha256": "d3d8e025b30340c2d4fbca99d288cd1450399c8c827ccdaf2107304a625621d0",
        "size_bytes": 13549262,
        "url": "https://github.com/smoothyy3/Dont-Blink/releases/download/2.0.0/custom_best.pt",
        "input_size": 320,
        "description": "MobileNetV3-Small printhead detector v2.0",
    },
}

DEFAULT_MODEL = "printhead_v2.0"


def get_cache_dir() -> Path:
    """
    Platform-appropriate cache directory for model weights.
    - macOS/Linux: ~/.cache/dontblink/models/
    - Windows: %LOCALAPPDATA%/dontblink/models/
    """
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / ".cache"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))

    cache_dir = base / "dontblink" / "models"
    return cache_dir


def get_registry_entry(model_name: Optional[str] = None) -> Dict[str, Any]:
    """Get registry entry for a model by name."""
    name = model_name or DEFAULT_MODEL
    if name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
    return MODEL_REGISTRY[name]


def _verify_sha256(file_path: str, expected_hash: str) -> bool:
    """Verify file integrity via SHA-256."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(1 << 16)
            if not chunk:
                break
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected_hash:
        logger.error(f"SHA-256 mismatch: expected {expected_hash[:16]}..., got {actual[:16]}...")
        return False
    return True


def download_model(model_name: Optional[str] = None, target_dir: Optional[str] = None, force: bool = False) -> str:
    """
    Download model weights if not already cached.

    Args:
        model_name: Key in MODEL_REGISTRY (default: DEFAULT_MODEL)
        target_dir: Override download location (default: platform cache dir)
        force: Re-download even if file exists and checksum matches

    Returns:
        Absolute path to the verified model file.
    """
    entry = get_registry_entry(model_name)
    filename = entry["filename"]
    expected_hash = entry["sha256"]
    url = entry["url"]
    expected_size = entry["size_bytes"]

    dest_dir = Path(target_dir) if target_dir else get_cache_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename

    if dest_path.exists() and not force:
        if _verify_sha256(str(dest_path), expected_hash):
            logger.info(f"Model already cached and verified: {dest_path}")
            return str(dest_path)
        else:
            logger.warning(f"Cached model failed verification, re-downloading...")

    logger.info(f"Downloading model from {url}")
    logger.info(f"Destination: {dest_path}")

    try:
        import urllib.request

        tmp_path = str(dest_path) + ".tmp"

        def _progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                downloaded = block_num * block_size
                pct = min(100, downloaded * 100 // total_size)
                mb_done = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                sys.stdout.write(f"\r  Downloading: {mb_done:.1f}/{mb_total:.1f} MB ({pct}%)")
                sys.stdout.flush()

        urllib.request.urlretrieve(url, tmp_path, reporthook=_progress_hook)
        print()  # newline after progress

        if not _verify_sha256(tmp_path, expected_hash):
            os.remove(tmp_path)
            raise RuntimeError(
                f"Downloaded model failed SHA-256 verification.\n"
                f"  Expected: {expected_hash}\n"
                f"  The file may be corrupted or the URL may be wrong.\n"
                f"  Try again, or download manually from the GitHub Releases page."
            )

        os.replace(tmp_path, str(dest_path))
        logger.info(f"Model downloaded and verified: {dest_path}")
        return str(dest_path)

    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Failed to download model: {e}\n"
            f"  URL: {url}\n"
            f"  Check your internet connection, or download the model manually:\n"
            f"  1. Go to: {url}\n"
            f"  2. Place the file at: models/{filename}\n"
        ) from e


def resolve_model_path(configured_path: str) -> str:
    """
    Resolve the model path: check local first, then cache, then download.

    Priority:
    1. configured_path exists → use it (covers cloned repo with models/ dir)
    2. Cached copy in platform cache dir → verify and use
    3. Neither exists → download to cache

    Returns:
        Absolute path to a verified model file.
    """
    if os.path.exists(configured_path):
        logger.debug(f"Using local model: {configured_path}")
        return configured_path

    entry = get_registry_entry(DEFAULT_MODEL)
    cache_path = get_cache_dir() / entry["filename"]

    if cache_path.exists():
        if _verify_sha256(str(cache_path), entry["sha256"]):
            logger.info(f"Using cached model: {cache_path}")
            return str(cache_path)
        else:
            logger.warning("Cached model failed verification, re-downloading...")

    logger.info(f"Model not found at '{configured_path}', downloading...")
    return download_model()
