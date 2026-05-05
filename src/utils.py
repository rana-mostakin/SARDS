"""
utils.py — SARDS Utility Module
================================
Shared helpers used across the entire SARDS pipeline:
  - Configuration loading
  - Logging setup
  - Directory management
  - Image I/O helpers
  - Color formatting for console output

Author : SARDS Team
Version: 1.0.0
"""

import os
import sys
import logging
import datetime
from pathlib import Path
from typing import Optional, Union

import yaml
import numpy as np
import cv2
import matplotlib.pyplot as plt
from colorama import Fore, Style, init as colorama_init

# Initialise colorama for cross-platform colored terminal output
colorama_init(autoreset=True)


# ── Configuration ────────────────────────────────────────────────────────────

def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load the SARDS YAML configuration file.

    Parameters
    ----------
    config_path : str
        Path to the YAML config file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            "Make sure you are running from the SARDS project root."
        )
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logger(name: str = "SARDS",
                 log_file: Optional[str] = None,
                 level: str = "INFO") -> logging.Logger:
    """
    Create and configure a logger with console + optional file output.

    Parameters
    ----------
    name     : Logger identifier string.
    log_file : Optional path to write log messages to a file.
    level    : Logging level string (DEBUG, INFO, WARNING, ERROR).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (optional)
    if log_file:
        ensure_dir(os.path.dirname(log_file))
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ── Directory / File Helpers ──────────────────────────────────────────────────

def ensure_dir(path: Union[str, Path]) -> Path:
    """Create a directory (and parents) if it does not exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_images(directory: str, extensions: tuple = (".jpg", ".jpeg", ".png",
                                                      ".tif", ".tiff")) -> list:
    """
    Return a sorted list of image file paths in *directory*.

    Parameters
    ----------
    directory  : Folder to scan.
    extensions : Allowed file extensions (lowercase).

    Returns
    -------
    list of Path objects, sorted by filename.
    """
    d = Path(directory)
    if not d.exists():
        return []
    files = sorted(
        [f for f in d.iterdir() if f.suffix.lower() in extensions],
        key=lambda f: f.name
    )
    return files


# ── Image I/O ─────────────────────────────────────────────────────────────────

def load_image_rgb(path: Union[str, Path]) -> np.ndarray:
    """
    Load an image from disk and return it as an RGB NumPy array (uint8).

    Parameters
    ----------
    path : Path to the image file.

    Returns
    -------
    np.ndarray  shape (H, W, 3), dtype=uint8
    """
    img = cv2.imread(str(path))
    if img is None:
        raise IOError(f"Could not load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_image(image: np.ndarray, path: Union[str, Path]) -> None:
    """
    Save an RGB NumPy array as an image file.

    Parameters
    ----------
    image : np.ndarray  (H, W, 3) RGB uint8.
    path  : Destination file path.
    """
    ensure_dir(Path(path).parent)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def array_to_uint8(arr: np.ndarray) -> np.ndarray:
    """
    Normalise a float array to [0, 255] uint8, safe for display / saving.
    """
    arr = arr.astype(np.float32)
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    normalised = (arr - lo) / (hi - lo) * 255.0
    return normalised.astype(np.uint8)


# ── Figure Helpers ────────────────────────────────────────────────────────────

def save_figure(fig: plt.Figure, path: Union[str, Path], dpi: int = 150) -> None:
    """Save a Matplotlib figure to disk, creating parent dirs if needed."""
    ensure_dir(Path(path).parent)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ── Console Formatting ────────────────────────────────────────────────────────

def print_banner() -> None:
    """Print the SARDS ASCII banner to the console."""
    banner = f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════════╗
║      SARDS — Satellite-Based Activity Risk Detection System  ║
║      Focus Area : Dhaka, Bangladesh                          ║
║      Version    : 1.0.0                                      ║
╚══════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
"""
    print(banner)


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{Fore.YELLOW}{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}{Style.RESET_ALL}")


def print_success(msg: str) -> None:
    print(f"{Fore.GREEN}✔  {msg}{Style.RESET_ALL}")


def print_warning(msg: str) -> None:
    print(f"{Fore.YELLOW}⚠  {msg}{Style.RESET_ALL}")


def print_error(msg: str) -> None:
    print(f"{Fore.RED}✘  {msg}{Style.RESET_ALL}")


def print_info(msg: str) -> None:
    print(f"{Fore.CYAN}ℹ  {msg}{Style.RESET_ALL}")


# ── Timestamp ─────────────────────────────────────────────────────────────────

def timestamp() -> str:
    """Return the current datetime as a compact string (for filenames)."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
