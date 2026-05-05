"""
change_detection.py — SARDS Change Detection Module
=====================================================
Compares two satellite images and identifies spatial changes.

Supported methods
─────────────────
  1. absolute_diff   — Simple per-pixel absolute difference
  2. ssim            — Structural Similarity Index (perceptual quality)
  3. optical_flow    — Dense optical flow (Lucas-Kanade / Farneback)

Key outputs
───────────
  - Binary change mask     (pixels classified as changed / unchanged)
  - Difference map         (continuous 0–255 float showing magnitude)
  - Annotated overlay      (original image with red change regions)
  - Change statistics dict (%, area, contour count, etc.)

Author : SARDS Team
Version: 1.0.0
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

from src.utils import load_config, setup_logger, array_to_uint8
from src.image_loader import SatelliteImage
from src.preprocessing import Preprocessor


# ── Logger ────────────────────────────────────────────────────────────────────
logger = setup_logger("ChangeDetection")


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ChangeResult:
    """
    Container for all outputs produced by a single change-detection run.

    Attributes
    ----------
    year_before  : Year of the "before" image.
    year_after   : Year of the "after"  image.
    diff_map     : Grayscale float32 array (H,W) — raw change magnitude.
    binary_mask  : Boolean mask (H,W) — True where change exceeded threshold.
    overlay      : RGB uint8 image with change regions highlighted.
    stats        : Summary statistics dictionary.
    """
    year_before  : int
    year_after   : int
    diff_map     : np.ndarray
    binary_mask  : np.ndarray
    overlay      : np.ndarray
    stats        : Dict  = field(default_factory=dict)

    @property
    def change_percentage(self) -> float:
        """Fraction of pixels that are flagged as changed (0–100)."""
        return self.stats.get("change_pct", 0.0)

    def __repr__(self) -> str:
        return (f"ChangeResult({self.year_before}→{self.year_after}, "
                f"change={self.change_percentage:.2f}%)")


# ── Detector ──────────────────────────────────────────────────────────────────

class ChangeDetector:
    """
    Pixel-level change detector for multi-date satellite imagery.

    Parameters
    ----------
    config_path : str
        Path to SARDS config.yaml.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.cfg   = load_config(config_path)
        cd_cfg     = self.cfg["change_detection"]

        self.method           = cd_cfg["method"]
        self.threshold        = cd_cfg["threshold"]
        self.min_area         = cd_cfg["min_change_area"]
        self.morph_k          = cd_cfg["morph_kernel_size"]
        self.highlight_color  = tuple(cd_cfg["highlight_color"])  # (R,G,B)
        self.prep             = Preprocessor(config_path)

    # ── Public API ────────────────────────────────────────────────────────

    def detect(
        self,
        before: SatelliteImage,
        after : SatelliteImage,
        align : bool = True
    ) -> ChangeResult:
        """
        Run change detection between *before* and *after* satellite images.

        Parameters
        ----------
        before : Earlier image.
        after  : Later image.
        align  : If True, apply ECC image alignment before comparison.

        Returns
        -------
        ChangeResult
        """
        logger.info(f"Change detection: {before.year} → {after.year}  "
                    f"method='{self.method}'")

        # Ensure matching resolution
        b_arr = self._ensure_same_size(before.array, after.array)
        a_arr = after.array.copy()

        if b_arr.shape != a_arr.shape:
            a_arr = cv2.resize(a_arr, (b_arr.shape[1], b_arr.shape[0]),
                               interpolation=cv2.INTER_LANCZOS4)

        # Optional co-registration
        if align:
            try:
                aligned_after = self.prep.align_pair(
                    SatelliteImage(before.year, before.path, b_arr),
                    SatelliteImage(after.year,  after.path,  a_arr)
                )
                a_arr = aligned_after.array
            except Exception as e:
                logger.warning(f"  Alignment skipped: {e}")

        # ── Core detection ─────────────────────────────────────────────
        if self.method == "absolute_diff":
            diff_map = self._absolute_diff(b_arr, a_arr)
        elif self.method == "ssim":
            diff_map = self._ssim_diff(b_arr, a_arr)
        elif self.method == "optical_flow":
            diff_map = self._optical_flow_diff(b_arr, a_arr)
        else:
            logger.warning(f"Unknown method '{self.method}' — using absolute_diff")
            diff_map = self._absolute_diff(b_arr, a_arr)

        # ── Post-processing ────────────────────────────────────────────
        binary_mask = self._threshold_and_clean(diff_map)
        overlay     = self._draw_overlay(b_arr, a_arr, binary_mask)
        stats       = self._compute_stats(diff_map, binary_mask,
                                          before.year, after.year)

        result = ChangeResult(
            year_before = before.year,
            year_after  = after.year,
            diff_map    = diff_map,
            binary_mask = binary_mask,
            overlay     = overlay,
            stats       = stats
        )
        logger.info(f"  → Change: {stats['change_pct']:.2f}%  "
                    f"({stats['changed_pixels']} px, "
                    f"{stats['contour_count']} regions)")
        return result

    def detect_all_pairs(
        self,
        images: List[SatelliteImage],
        align : bool = True
    ) -> List[ChangeResult]:
        """
        Run *detect* on every consecutive (before, after) pair in *images*.

        Parameters
        ----------
        images : Time-sorted list of SatelliteImages.

        Returns
        -------
        List[ChangeResult]
        """
        results = []
        for i in range(len(images) - 1):
            try:
                r = self.detect(images[i], images[i + 1], align=align)
                results.append(r)
            except Exception as exc:
                logger.error(f"  Pair {images[i].year}→{images[i+1].year}: {exc}")
        return results

    # ── Core Detection Methods ────────────────────────────────────────────

    def _absolute_diff(self, b: np.ndarray, a: np.ndarray) -> np.ndarray:
        """
        Compute the per-channel absolute difference, then take the mean
        across channels to produce a single-channel change magnitude map.
        """
        b_f = b.astype(np.float32)
        a_f = a.astype(np.float32)
        diff = np.abs(b_f - a_f)          # (H, W, 3)
        return diff.mean(axis=2)           # (H, W) — mean across RGB

    def _ssim_diff(self, b: np.ndarray, a: np.ndarray) -> np.ndarray:
        """
        Compute the (1 − SSIM) difference map channel-by-channel.
        Regions that differ structurally will have high values.
        """
        try:
            from skimage.metrics import structural_similarity as ssim_fn
        except ImportError:
            logger.warning("scikit-image not installed; falling back to absolute_diff")
            return self._absolute_diff(b, a)

        b_gray = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        a_gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

        _, ssim_map = ssim_fn(b_gray, a_gray, full=True, data_range=1.0)
        # Invert: regions where SSIM is low → high difference
        diff_map = (1.0 - ssim_map) * 255.0
        return diff_map.astype(np.float32)

    def _optical_flow_diff(self, b: np.ndarray, a: np.ndarray) -> np.ndarray:
        """
        Compute Farneback dense optical flow magnitude between *b* and *a*.
        Higher flow magnitude → more motion/change between the two dates.
        """
        b_gray = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
        a_gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            b_gray, a_gray,
            flow=None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2,
            flags=0
        )
        # flow shape: (H, W, 2) — (dx, dy)
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        # Scale to 0–255 range
        mag_scaled = magnitude / (magnitude.max() + 1e-8) * 255.0
        return mag_scaled.astype(np.float32)

    # ── Post-processing Helpers ────────────────────────────────────────────

    def _threshold_and_clean(self, diff_map: np.ndarray) -> np.ndarray:
        """
        Convert a float difference map → clean binary mask.

        Steps
        ─────
        1. Threshold at self.threshold to get raw binary mask.
        2. Remove small isolated noise regions (< min_area px²).
        3. Morphological close+open to fill holes and smooth edges.
        """
        # Binary threshold
        _, mask = cv2.threshold(
            diff_map.astype(np.uint8), self.threshold, 255, cv2.THRESH_BINARY
        )

        # Morphological clean-up
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.morph_k, self.morph_k)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

        # Remove tiny contours (salt-and-pepper noise)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        clean_mask = np.zeros_like(mask)
        for c in contours:
            if cv2.contourArea(c) >= self.min_area:
                cv2.drawContours(clean_mask, [c], -1, 255, -1)

        return clean_mask.astype(bool)

    def _draw_overlay(
        self,
        before_rgb : np.ndarray,
        after_rgb  : np.ndarray,
        mask       : np.ndarray
    ) -> np.ndarray:
        """
        Produce a side-annotated overlay:
        Left half = "before" with red mask overlay.
        Right half = "after"  with red mask overlay.
        Both halves drawn on the *after* image for context.
        """
        h, w = after_rgb.shape[:2]
        overlay = after_rgb.copy()

        # Semi-transparent red fill where mask is True
        red_layer = np.zeros_like(overlay)
        red_layer[mask] = self.highlight_color

        cv2.addWeighted(red_layer, 0.45, overlay, 0.55, 0, overlay)

        # Draw contour outlines
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1,
                         (int(self.highlight_color[0]),
                          int(self.highlight_color[1]),
                          int(self.highlight_color[2])), 2)
        return overlay

    @staticmethod
    def _compute_stats(
        diff_map : np.ndarray,
        mask     : np.ndarray,
        y_before : int,
        y_after  : int
    ) -> Dict:
        """Compute a statistics summary for one change detection result."""
        total_px   = diff_map.size
        changed_px = int(mask.sum())
        change_pct = changed_px / total_px * 100.0

        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        return {
            "year_before"    : y_before,
            "year_after"     : y_after,
            "total_pixels"   : total_px,
            "changed_pixels" : changed_px,
            "change_pct"     : round(change_pct, 4),
            "mean_diff"      : float(diff_map[mask].mean()) if changed_px > 0 else 0.0,
            "max_diff"       : float(diff_map.max()),
            "contour_count"  : len(contours),
        }

    # ── Utility ───────────────────────────────────────────────────────────

    @staticmethod
    def _ensure_same_size(arr: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Resize *arr* to match *reference* dimensions if they differ."""
        if arr.shape[:2] == reference.shape[:2]:
            return arr
        return cv2.resize(arr, (reference.shape[1], reference.shape[0]),
                          interpolation=cv2.INTER_LANCZOS4)
