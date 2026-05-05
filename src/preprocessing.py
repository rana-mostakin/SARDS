"""
preprocessing.py — SARDS Image Pre-processing Module
======================================================
Prepares raw satellite images for analysis:
  1. Resize to a common resolution
  2. Normalise pixel intensities
  3. Apply Gaussian blur to reduce sensor noise
  4. Histogram equalisation to improve contrast
  5. Geometric alignment (ECC-based registration) for multi-date pairs

Author : SARDS Team
Version: 1.0.0
"""

from typing import List, Tuple, Optional
import numpy as np
import cv2

from src.utils import load_config, setup_logger
from src.image_loader import SatelliteImage, ImageDataset


# ── Logger ────────────────────────────────────────────────────────────────────
logger = setup_logger("Preprocessing")


class Preprocessor:
    """
    Stateless image pre-processing pipeline driven by config.yaml.

    All public methods accept and return *SatelliteImage* objects so
    the dataset schema remains consistent throughout the pipeline.

    Parameters
    ----------
    config_path : str
        Path to SARDS config.yaml.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.cfg  = load_config(config_path)
        pc        = self.cfg["preprocessing"]

        self.target_w, self.target_h = pc["target_size"]   # (W, H)
        self.normalize               = pc["normalize"]
        self.blur_k                  = pc["gaussian_blur_kernel"]
        self.equalize                = pc["equalize_histogram"]

    # ── Public API ────────────────────────────────────────────────────────

    def process_image(self, sat_img: SatelliteImage) -> SatelliteImage:
        """
        Apply the full pre-processing pipeline to a single SatelliteImage.

        Steps:
            resize → blur → histogram equalise → (optionally store normalised)

        Parameters
        ----------
        sat_img : Input SatelliteImage (RGB uint8).

        Returns
        -------
        SatelliteImage
            New object with the processed array.  The original is untouched.
        """
        arr = sat_img.array.copy()

        # 1. Resize to common resolution
        arr = self._resize(arr)
        logger.debug(f"  resize  → {arr.shape}")

        # 2. Gaussian blur (noise reduction)
        arr = self._blur(arr)
        logger.debug(f"  blur    → kernel={self.blur_k}")

        # 3. Histogram equalisation (contrast enhancement)
        if self.equalize:
            arr = self._equalize_hist(arr)
            logger.debug("  equalize histogram applied")

        # Return a new SatelliteImage with processed array
        processed = SatelliteImage(
            year  = sat_img.year,
            path  = sat_img.path,
            array = arr,
            meta  = {**sat_img.meta, "preprocessed": True,
                     "target_size": (self.target_w, self.target_h)}
        )
        return processed

    def process_dataset(self, dataset: ImageDataset) -> ImageDataset:
        """
        Apply *process_image* to every frame in an ImageDataset.

        Parameters
        ----------
        dataset : Input ImageDataset.

        Returns
        -------
        ImageDataset
            New dataset with processed images.
        """
        logger.info(f"Pre-processing {dataset.count} image(s) …")
        processed_ds = ImageDataset(location=dataset.location)

        for year, img in sorted(dataset.images.items()):
            logger.info(f"  Processing {year} …")
            processed_ds.images[year] = self.process_image(img)

        logger.info("Pre-processing complete.")
        return processed_ds

    def align_pair(
        self,
        ref: SatelliteImage,
        target: SatelliteImage,
        max_iter: int = 200,
        eps: float = 1e-5
    ) -> SatelliteImage:
        """
        Co-register *target* to *ref* using OpenCV's ECC (Enhanced Correlation
        Coefficient) algorithm.  This corrects for slight camera or orbit drift
        between acquisition dates so pixel-level change detection is accurate.

        Parameters
        ----------
        ref     : Reference image (anchor — not moved).
        target  : Image to warp onto ref's coordinate frame.
        max_iter: ECC optimisation iterations.
        eps     : Convergence threshold.

        Returns
        -------
        SatelliteImage
            New SatelliteImage with the aligned array.
        """
        ref_gray    = cv2.cvtColor(ref.array,    cv2.COLOR_RGB2GRAY)
        target_gray = cv2.cvtColor(target.array, cv2.COLOR_RGB2GRAY)

        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria    = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            max_iter,
            eps
        )

        try:
            _, warp_matrix = cv2.findTransformECC(
                ref_gray, target_gray,
                warp_matrix,
                cv2.MOTION_EUCLIDEAN,
                criteria
            )
            aligned = cv2.warpAffine(
                target.array, warp_matrix,
                (ref.width, ref.height),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
            )
            logger.debug(f"  ECC alignment applied ({target.year} → {ref.year})")
        except cv2.error as e:
            logger.warning(f"  ECC alignment failed for {target.year}: {e} — using unaligned")
            aligned = target.array.copy()

        return SatelliteImage(
            year  = target.year,
            path  = target.path,
            array = aligned,
            meta  = {**target.meta, "aligned_to": ref.year}
        )

    def to_float(self, arr: np.ndarray) -> np.ndarray:
        """Convert a uint8 image to float32 in [0, 1]."""
        return arr.astype(np.float32) / 255.0

    def to_grayscale(self, arr: np.ndarray) -> np.ndarray:
        """Convert RGB uint8 → grayscale uint8."""
        return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # ── Internal helpers ──────────────────────────────────────────────────

    def _resize(self, arr: np.ndarray) -> np.ndarray:
        """Resize image to (target_w × target_h) using LANCZOS interpolation."""
        if arr.shape[1] == self.target_w and arr.shape[0] == self.target_h:
            return arr
        return cv2.resize(arr, (self.target_w, self.target_h),
                          interpolation=cv2.INTER_LANCZOS4)

    def _blur(self, arr: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur.  Kernel size forced to odd number."""
        k = self.blur_k
        if k % 2 == 0:
            k += 1
        if k <= 1:
            return arr
        return cv2.GaussianBlur(arr, (k, k), sigmaX=0)

    @staticmethod
    def _equalize_hist(arr: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast-Limited Adaptive Histogram Equalisation)
        per channel to avoid colour shift and improve local contrast.
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        channels = cv2.split(arr)
        eq_channels = [clahe.apply(ch) for ch in channels]
        return cv2.merge(eq_channels)
