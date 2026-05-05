"""
heatmap_generator.py — SARDS Heatmap Generation Module
=======================================================
Converts raw change/intensity maps into publication-quality heatmaps:
  - Single-pair change heatmap
  - Cumulative heatmap across a full timeline
  - Overlay heatmap (translucent colour atop original satellite image)
  - Grid comparison panel (all years side by side)

Author : SARDS Team
Version: 1.0.0
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import seaborn as sns

from src.utils import load_config, setup_logger, ensure_dir, save_figure
from src.image_loader import SatelliteImage, ImageDataset
from src.change_detection import ChangeResult


# ── Logger ────────────────────────────────────────────────────────────────────
logger = setup_logger("HeatmapGenerator")


class HeatmapGenerator:
    """
    Generates and saves heatmap visualisations for SARDS.

    Parameters
    ----------
    config_path : str
        Path to SARDS config.yaml.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.cfg     = load_config(config_path)
        hm_cfg       = self.cfg["heatmap"]
        out_cfg      = self.cfg["output"]

        self.colormap     = hm_cfg["colormap"]           # e.g. "jet"
        self.blur_sigma   = hm_cfg["blur_sigma"]
        self.alpha        = hm_cfg["alpha_overlay"]
        self.save_fmt     = hm_cfg["save_format"]
        self.dpi          = out_cfg["dpi"]
        self.fig_size     = tuple(out_cfg["figure_size"])
        self.save_dir     = Path(self.cfg["data"]["results_dir"]) / "heatmaps"
        ensure_dir(self.save_dir)

    # ── Public API ────────────────────────────────────────────────────────

    def generate_change_heatmap(
        self,
        result   : ChangeResult,
        base_img : Optional[SatelliteImage] = None,
        save     : bool = True
    ) -> np.ndarray:
        """
        Generate a heatmap from a single ChangeResult.

        Parameters
        ----------
        result   : Output of ChangeDetector.detect().
        base_img : If provided, the heatmap is overlaid on this image.
        save     : Persist the heatmap to disk.

        Returns
        -------
        np.ndarray  RGB uint8 heatmap image (H, W, 3).
        """
        diff = result.diff_map.copy().astype(np.float32)
        heatmap_rgb = self._diff_to_heatmap(diff)

        if base_img is not None:
            base = cv2.resize(base_img.array,
                              (heatmap_rgb.shape[1], heatmap_rgb.shape[0]),
                              interpolation=cv2.INTER_LANCZOS4)
            heatmap_rgb = self._overlay(base, heatmap_rgb, self.alpha)

        if save:
            fname = (self.save_dir /
                     f"heatmap_{result.year_before}_{result.year_after}.{self.save_fmt}")
            self._save_heatmap_image(heatmap_rgb, fname)
            logger.info(f"  Saved: {fname}")

        return heatmap_rgb

    def generate_cumulative_heatmap(
        self,
        results  : List[ChangeResult],
        base_img : Optional[SatelliteImage] = None,
        save     : bool = True
    ) -> np.ndarray:
        """
        Stack all per-pair difference maps into a single cumulative heatmap.
        Each pair contributes equally to the total accumulation.

        Parameters
        ----------
        results  : List of ChangeResult objects (all pairs in the timeline).
        base_img : Optional overlay base image.

        Returns
        -------
        np.ndarray  RGB uint8 cumulative heatmap.
        """
        if not results:
            raise ValueError("No ChangeResult objects provided.")

        # Initialise accumulator from first result
        h, w = results[0].diff_map.shape
        cumulative = np.zeros((h, w), dtype=np.float32)

        for res in results:
            diff = cv2.resize(res.diff_map.astype(np.float32), (w, h),
                              interpolation=cv2.INTER_LINEAR)
            cumulative += diff

        # Normalise to 0–255
        cumulative = cumulative / (cumulative.max() + 1e-8) * 255.0
        heatmap_rgb = self._diff_to_heatmap(cumulative)

        if base_img is not None:
            base = cv2.resize(base_img.array, (w, h),
                              interpolation=cv2.INTER_LANCZOS4)
            heatmap_rgb = self._overlay(base, heatmap_rgb, self.alpha)

        if save:
            years = [r.year_before for r in results] + [results[-1].year_after]
            fname = (self.save_dir /
                     f"cumulative_{min(years)}_{max(years)}.{self.save_fmt}")
            self._save_heatmap_image(heatmap_rgb, fname)
            logger.info(f"  Saved cumulative heatmap: {fname}")

        return heatmap_rgb

    def generate_comparison_panel(
        self,
        dataset  : ImageDataset,
        results  : List[ChangeResult],
        save     : bool = True
    ) -> plt.Figure:
        """
        Create a multi-panel figure showing, for each time pair:
            [Before image]  [After image]  [Change heatmap]

        Parameters
        ----------
        dataset : Full ImageDataset (for retrieving original images).
        results : List of ChangeResult objects.

        Returns
        -------
        matplotlib Figure
        """
        n = len(results)
        if n == 0:
            raise ValueError("No results to plot.")

        fig, axes = plt.subplots(n, 3, figsize=(14, 4.5 * n))
        if n == 1:
            axes = np.array([axes])  # Ensure 2-D axes array

        fig.suptitle("SARDS — Dhaka Satellite Change Analysis",
                     fontsize=15, fontweight="bold", y=1.01)

        for row, res in enumerate(results):
            ax_before, ax_after, ax_heat = axes[row]

            before_img = dataset.get(res.year_before)
            after_img  = dataset.get(res.year_after)

            if before_img is not None:
                ax_before.imshow(before_img.array)
            ax_before.set_title(f"Before — {res.year_before}", fontsize=11)
            ax_before.axis("off")

            if after_img is not None:
                ax_after.imshow(after_img.array)
            ax_after.set_title(f"After — {res.year_after}", fontsize=11)
            ax_after.axis("off")

            heatmap = self._diff_to_heatmap(res.diff_map)
            ax_heat.imshow(heatmap)
            ax_heat.set_title(f"Change Heatmap  ({res.change_percentage:.1f}%)",
                              fontsize=11, color="darkred")
            ax_heat.axis("off")

            # Colour-bar for the heatmap column
            sm = plt.cm.ScalarMappable(
                cmap=plt.get_cmap(self.colormap),
                norm=plt.Normalize(vmin=0, vmax=255)
            )
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax_heat, fraction=0.046, pad=0.04)
            cbar.set_label("Change Intensity", fontsize=8)

        plt.tight_layout()

        if save:
            fname = self.save_dir / f"comparison_panel.{self.save_fmt}"
            save_figure(fig, fname, dpi=self.dpi)
            logger.info(f"  Saved comparison panel: {fname}")

        return fig

    def generate_intensity_grid(
        self,
        dataset : ImageDataset,
        save    : bool = True
    ) -> plt.Figure:
        """
        Show all images in the dataset as a single image grid with titles.

        Parameters
        ----------
        dataset : Full processed ImageDataset.

        Returns
        -------
        matplotlib Figure
        """
        imgs  = dataset.as_list()
        n     = len(imgs)
        ncols = min(4, n)
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 4, nrows * 4))
        axes = np.array(axes).flatten()

        for idx, img in enumerate(imgs):
            axes[idx].imshow(img.array)
            # Compute brightness as a simple intensity proxy
            brightness = img.array.mean()
            axes[idx].set_title(f"{img.year}\nBrightness={brightness:.1f}",
                                fontsize=10)
            axes[idx].axis("off")

        # Hide extra axes
        for idx in range(n, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle("Dhaka Timeline — Raw Satellite Frames",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save:
            fname = self.save_dir / f"timeline_grid.{self.save_fmt}"
            save_figure(fig, fname, dpi=self.dpi)
            logger.info(f"  Saved timeline grid: {fname}")

        return fig

    # ── Internal helpers ──────────────────────────────────────────────────

    def _diff_to_heatmap(self, diff: np.ndarray) -> np.ndarray:
        """
        Convert a single-channel float difference map → RGB uint8 heatmap.

        Steps
        ─────
        1. Normalise to [0, 1].
        2. Apply Gaussian blur to create smooth thermal appearance.
        3. Apply the selected matplotlib colormap.
        """
        diff = diff.astype(np.float32)

        # Normalise
        lo, hi = diff.min(), diff.max()
        if hi - lo < 1e-8:
            norm = np.zeros_like(diff)
        else:
            norm = (diff - lo) / (hi - lo)

        # Gaussian blur for smooth heatmap appearance
        if self.blur_sigma > 0:
            norm = cv2.GaussianBlur(norm, (0, 0), sigmaX=self.blur_sigma)
            norm = np.clip(norm, 0, 1)

        # Apply colormap
        cmap = plt.get_cmap(self.colormap)
        rgba = (cmap(norm) * 255).astype(np.uint8)   # (H, W, 4) RGBA
        return rgba[:, :, :3]                          # Drop alpha → RGB

    @staticmethod
    def _overlay(base: np.ndarray, heatmap: np.ndarray, alpha: float) -> np.ndarray:
        """
        Blend *heatmap* on top of *base* with transparency *alpha*.
        alpha=0 → only base; alpha=1 → only heatmap.
        """
        h, w = base.shape[:2]
        hm_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
        base_f  = base.astype(np.float32)
        heat_f  = hm_resized.astype(np.float32)
        blended = base_f * (1.0 - alpha) + heat_f * alpha
        return np.clip(blended, 0, 255).astype(np.uint8)

    @staticmethod
    def _save_heatmap_image(rgb: np.ndarray, path: Path) -> None:
        """Save an RGB uint8 heatmap array to disk via OpenCV."""
        ensure_dir(path.parent)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), bgr)
