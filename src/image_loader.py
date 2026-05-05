"""
image_loader.py — SARDS Image Loading & Dataset Management
============================================================
Responsibilities:
  - Discover images in the data directory
  - Load single or entire timeline datasets
  - Validate images before processing
  - Provide a clean ImageDataset interface used by all other modules

Author : SARDS Team
Version: 1.0.0
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import numpy as np
import cv2

from src.utils import (
    load_config, setup_logger, load_image_rgb, list_images
)


# ── Logger ────────────────────────────────────────────────────────────────────
logger = setup_logger("ImageLoader")


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class SatelliteImage:
    """
    Lightweight container for a single satellite image frame.

    Attributes
    ----------
    year  : The calendar year this image represents.
    path  : Source file path on disk.
    array : RGB pixel data (H, W, 3) uint8.
    meta  : Optional dictionary of metadata (sensor, resolution, etc.).
    """
    year  : int
    path  : str
    array : np.ndarray
    meta  : dict = field(default_factory=dict)

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.array.shape

    @property
    def height(self) -> int:
        return self.array.shape[0]

    @property
    def width(self) -> int:
        return self.array.shape[1]

    def __repr__(self) -> str:
        return (f"SatelliteImage(year={self.year}, "
                f"shape={self.shape}, path={self.path})")


@dataclass
class ImageDataset:
    """
    A time-ordered collection of SatelliteImage objects for one location.

    Attributes
    ----------
    location : Human-readable location name.
    images   : Dict mapping year (int) → SatelliteImage.
    """
    location : str
    images   : Dict[int, SatelliteImage] = field(default_factory=dict)

    # ── convenience properties ─────────────────────────────────────────────

    @property
    def years(self) -> List[int]:
        """Sorted list of years present in the dataset."""
        return sorted(self.images.keys())

    @property
    def count(self) -> int:
        return len(self.images)

    def get(self, year: int) -> Optional[SatelliteImage]:
        """Return the SatelliteImage for *year*, or None if not found."""
        return self.images.get(year)

    def pairs(self) -> List[Tuple[SatelliteImage, SatelliteImage]]:
        """
        Return consecutive (before, after) image pairs sorted by year.
        Useful for change-detection pipelines.
        """
        ordered = [self.images[y] for y in self.years]
        return [(ordered[i], ordered[i + 1]) for i in range(len(ordered) - 1)]

    def as_list(self) -> List[SatelliteImage]:
        """Return images sorted by year as a flat list."""
        return [self.images[y] for y in self.years]

    def __repr__(self) -> str:
        return (f"ImageDataset(location='{self.location}', "
                f"years={self.years}, count={self.count})")


# ── Loader ────────────────────────────────────────────────────────────────────

class ImageLoader:
    """
    Loads and validates satellite imagery from the SARDS data directory.

    Usage
    -----
    >>> loader = ImageLoader()
    >>> dataset = loader.load_dhaka_timeline()
    >>> img_2020 = dataset.get(2020)
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.cfg = load_config(config_path)
        self.data_dir   = Path(self.cfg["data"]["raw_dir"])
        self.timeline   = self.cfg["data"]["timeline"]          # {year: filename}
        self.location   = self.cfg["location"]["name"]
        self.extensions = tuple(self.cfg["data"]["supported_formats"])

    # ── Public API ────────────────────────────────────────────────────────

    def load_dhaka_timeline(self) -> ImageDataset:
        """
        Load the full Dhaka multi-year timeline defined in config.yaml.

        Returns
        -------
        ImageDataset
            Populated dataset with all available years.
        """
        logger.info(f"Loading Dhaka timeline from: {self.data_dir}")
        dataset = ImageDataset(location=self.location)

        for year, filename in sorted(self.timeline.items()):
            img_path = self.data_dir / filename
            if not img_path.exists():
                logger.warning(f"  [SKIP] {year} — file not found: {img_path}")
                continue
            try:
                sat_img = self._load_single(year, img_path)
                dataset.images[year] = sat_img
                logger.info(f"  [OK]   {year} → {img_path.name}  "
                            f"shape={sat_img.shape}")
            except Exception as exc:
                logger.error(f"  [ERR]  {year} — {exc}")

        if dataset.count == 0:
            raise RuntimeError(
                "No images were loaded. Run  python scripts/generate_demo_data.py  "
                "to create sample Dhaka imagery, then try again."
            )

        logger.info(f"Loaded {dataset.count} image(s) spanning {dataset.years}")
        return dataset

    def load_from_directory(self, directory: str) -> ImageDataset:
        """
        Load all valid images from an arbitrary directory.
        Year is inferred from the filename if it contains a 4-digit number,
        otherwise an index is used.

        Parameters
        ----------
        directory : Path to the folder containing images.

        Returns
        -------
        ImageDataset
        """
        files = list_images(directory, extensions=self.extensions)
        if not files:
            raise FileNotFoundError(f"No images found in: {directory}")

        dataset = ImageDataset(location=directory)
        for idx, fpath in enumerate(files):
            year = self._infer_year(fpath.stem, fallback=2000 + idx)
            sat_img = self._load_single(year, fpath)
            # Handle duplicate years by shifting
            while year in dataset.images:
                year += 1
            dataset.images[year] = sat_img

        logger.info(f"Loaded {dataset.count} image(s) from {directory}")
        return dataset

    def load_single_image(self, path: Union[str, Path], year: int = 0) -> SatelliteImage:
        """
        Load one image from an explicit path.

        Parameters
        ----------
        path : Path to the image file.
        year : Year label (default 0 if unknown).

        Returns
        -------
        SatelliteImage
        """
        return self._load_single(year, Path(path))

    # ── Internal helpers ──────────────────────────────────────────────────

    def _load_single(self, year: int, path: Path) -> SatelliteImage:
        """Load one image, validate it, and return a SatelliteImage."""
        arr = load_image_rgb(path)
        self._validate(arr, path)
        meta = {
            "filename"  : path.name,
            "file_size" : path.stat().st_size,
            "dtype"     : str(arr.dtype),
        }
        return SatelliteImage(year=year, path=str(path), array=arr, meta=meta)

    @staticmethod
    def _validate(arr: np.ndarray, path: Path) -> None:
        """Raise ValueError if the image array looks invalid."""
        if arr is None or arr.size == 0:
            raise ValueError(f"Empty image array from: {path}")
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(
                f"Expected RGB image (H,W,3), got shape {arr.shape}: {path}"
            )
        if arr.shape[0] < 32 or arr.shape[1] < 32:
            raise ValueError(
                f"Image too small ({arr.shape}): {path}"
            )

    @staticmethod
    def _infer_year(stem: str, fallback: int = 2000) -> int:
        """
        Try to extract a 4-digit year from a filename stem
        (e.g. 'dhaka_2020_S2' → 2020).
        Falls back to *fallback* if none found.
        """
        import re
        matches = re.findall(r"\b(19\d{2}|20\d{2})\b", stem)
        if matches:
            return int(matches[0])
        return fallback
