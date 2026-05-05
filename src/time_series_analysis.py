"""
time_series_analysis.py — SARDS Time-Series Analysis Module
=============================================================
Analyses how satellite-derived metrics evolve across multiple years:
  - Urban index (built-up surface proxy via brightness)
  - NDVI proxy (vegetation estimate via green channel ratio)
  - Image entropy (structural complexity)
  - Mean intensity and per-channel luminance
  - Change progression from consecutive pairs

Produces:
  - Pandas DataFrame of metrics per year
  - Matplotlib multi-metric line plots
  - Trend decomposition (linear regression)
  - Summary statistics

Author : SARDS Team
Version: 1.0.0
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from scipy import stats
from scipy.signal import savgol_filter

from src.utils import load_config, setup_logger, ensure_dir, save_figure
from src.image_loader import SatelliteImage, ImageDataset
from src.change_detection import ChangeResult


# ── Logger ────────────────────────────────────────────────────────────────────
logger = setup_logger("TimeSeriesAnalysis")


class TimeSeriesAnalyzer:
    """
    Extracts time-series metrics from a multi-year satellite image dataset
    and generates publication-quality trend plots.

    Parameters
    ----------
    config_path : str
        Path to SARDS config.yaml.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.cfg     = load_config(config_path)
        ts_cfg       = self.cfg["time_series"]
        out_cfg      = self.cfg["output"]

        self.metrics          = ts_cfg["metrics"]
        self.smoothing        = ts_cfg["smoothing"]
        self.smooth_window    = ts_cfg["smoothing_window"]
        self.dpi              = out_cfg["dpi"]
        self.fig_size         = tuple(out_cfg["figure_size"])
        self.save_dir         = Path(self.cfg["data"]["results_dir"]) / "time_series"
        ensure_dir(self.save_dir)

    # ── Public API ────────────────────────────────────────────────────────

    def extract_metrics(self, dataset: ImageDataset) -> pd.DataFrame:
        """
        Extract all configured metrics for every image in *dataset*.

        Returns
        -------
        pd.DataFrame
            Indexed by year, one column per metric.
        """
        logger.info(f"Extracting time-series metrics for {dataset.count} images …")
        rows = []

        for img in dataset.as_list():
            row = {"year": img.year}
            row.update(self._compute_image_metrics(img))
            rows.append(row)
            logger.debug(f"  {img.year}: {row}")

        df = pd.DataFrame(rows).set_index("year").sort_index()
        logger.info(f"Metrics DataFrame shape: {df.shape}")
        return df

    def add_change_metrics(
        self,
        df      : pd.DataFrame,
        results : List[ChangeResult]
    ) -> pd.DataFrame:
        """
        Augment the metrics DataFrame with per-pair change statistics.

        Columns added
        ─────────────
        change_pct       — % pixels changed vs previous year
        mean_diff        — mean change magnitude
        contour_count    — number of discrete changed regions

        Parameters
        ----------
        df      : DataFrame from extract_metrics().
        results : List of ChangeResult objects (consecutive pairs).

        Returns
        -------
        pd.DataFrame  (copy with new columns filled for each "after" year)
        """
        df = df.copy()
        for col in ("change_pct", "mean_diff", "contour_count"):
            if col not in df.columns:
                df[col] = np.nan

        for res in results:
            if res.year_after in df.index:
                df.at[res.year_after, "change_pct"]    = res.stats.get("change_pct",    0.0)
                df.at[res.year_after, "mean_diff"]     = res.stats.get("mean_diff",     0.0)
                df.at[res.year_after, "contour_count"] = res.stats.get("contour_count", 0)

        return df

    def compute_trends(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Fit a simple linear trend (OLS) to each numeric column.

        Returns
        -------
        dict  { metric_name: {"slope": …, "intercept": …, "r2": …,
                               "p_value": …, "trend": "increasing"|"decreasing"|"stable"} }
        """
        trends = {}
        x = np.arange(len(df))

        for col in df.select_dtypes(include=[np.number]).columns:
            y = df[col].values.astype(float)
            valid = ~np.isnan(y)
            if valid.sum() < 2:
                continue
            slope, intercept, r, p, _ = stats.linregress(x[valid], y[valid])
            r2 = r ** 2
            if p < 0.05:
                direction = "increasing" if slope > 0 else "decreasing"
            else:
                direction = "stable"
            trends[col] = {
                "slope"     : round(slope,     6),
                "intercept" : round(intercept, 4),
                "r2"        : round(r2,         4),
                "p_value"   : round(p,          6),
                "trend"     : direction,
            }

        return trends

    def plot_metrics(
        self,
        df      : pd.DataFrame,
        trends  : Optional[Dict] = None,
        save    : bool = True
    ) -> plt.Figure:
        """
        Plot a multi-panel time-series chart showing all key metrics.

        Parameters
        ----------
        df     : Metrics DataFrame (index = year).
        trends : Optional trend dict from compute_trends().
        save   : Persist figure to disk.

        Returns
        -------
        matplotlib Figure
        """
        cols_to_plot = [c for c in df.columns if c in (
            "mean_intensity", "urban_index", "ndvi_proxy",
            "entropy", "change_pct"
        ) and not df[c].isna().all()]

        n = len(cols_to_plot)
        if n == 0:
            logger.warning("No plottable columns found.")
            return None

        fig, axes = plt.subplots(n, 1, figsize=(self.fig_size[0], 3 * n),
                                 sharex=True)
        if n == 1:
            axes = [axes]

        years = df.index.tolist()
        palette = sns.color_palette("tab10", n)

        title_map = {
            "mean_intensity" : "Mean Pixel Intensity",
            "urban_index"    : "Urban Expansion Index",
            "ndvi_proxy"     : "NDVI Proxy (Vegetation)",
            "entropy"        : "Image Entropy (Complexity)",
            "change_pct"     : "Change Percentage (%)",
        }
        unit_map = {
            "mean_intensity" : "Pixel Value (0–255)",
            "urban_index"    : "Index (0–1)",
            "ndvi_proxy"     : "NDVI Proxy (−1 to 1)",
            "entropy"        : "Bits",
            "change_pct"     : "% Changed Pixels",
        }

        for ax, col, color in zip(axes, cols_to_plot, palette):
            y = df[col].values.astype(float)

            # Raw data
            ax.plot(years, y, "o-", color=color, linewidth=2,
                    markersize=7, label=col, zorder=3)

            # Smoothed trend line
            if self.smoothing and len(y) >= 4:
                try:
                    wl = min(self.smooth_window * 2 + 1, len(y) if len(y) % 2 == 1 else len(y) - 1)
                    wl = max(wl, 3)
                    y_smooth = savgol_filter(y, window_length=wl, polyorder=2)
                    ax.plot(years, y_smooth, "--", color=color,
                            linewidth=1.5, alpha=0.6, label="trend")
                except Exception:
                    pass

            # Linear regression overlay
            if trends and col in trends:
                t = trends[col]
                x_vals = np.arange(len(years))
                y_reg  = t["slope"] * x_vals + t["intercept"]
                lbl    = (f"OLS: y={t['slope']:+.3f}x  "
                          f"R²={t['r2']:.2f}  [{t['trend']}]")
                ax.plot(years, y_reg, ":", color="black",
                        linewidth=1.2, label=lbl, zorder=2)

            ax.set_ylabel(unit_map.get(col, col), fontsize=9)
            ax.set_title(title_map.get(col, col), fontsize=11, fontweight="bold")
            ax.legend(fontsize=8, loc="upper left")
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.set_facecolor("#f9f9f9")

            # Shade every other year band for readability
            for i, yr in enumerate(years):
                if i % 2 == 0:
                    ax.axvspan(yr - 0.5, yr + 0.5, alpha=0.05, color="grey")

        axes[-1].set_xlabel("Year", fontsize=11)
        plt.xticks(years, [str(y) for y in years], rotation=45)

        fig.suptitle("SARDS — Dhaka Time-Series Metrics",
                     fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()

        if save:
            fname = self.save_dir / "time_series_metrics.png"
            save_figure(fig, fname, dpi=self.dpi)
            logger.info(f"  Saved time-series plot: {fname}")

        return fig

    def plot_change_progression(
        self,
        results : List[ChangeResult],
        save    : bool = True
    ) -> plt.Figure:
        """
        Bar + line chart showing change percentage for every consecutive pair.

        Parameters
        ----------
        results : List of ChangeResult objects.

        Returns
        -------
        matplotlib Figure
        """
        if not results:
            return None

        labels   = [f"{r.year_before}→{r.year_after}" for r in results]
        changes  = [r.stats.get("change_pct", 0.0)     for r in results]
        contours = [r.stats.get("contour_count", 0)    for r in results]

        fig, ax1 = plt.subplots(figsize=self.fig_size)
        ax2 = ax1.twinx()

        x = np.arange(len(labels))
        bars = ax1.bar(x, changes, width=0.55, color="#e74c3c",
                       alpha=0.75, label="Change %", zorder=2)
        ax2.plot(x, contours, "o--", color="#2980b9", linewidth=2,
                 markersize=8, label="Changed Regions", zorder=3)

        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
        ax1.set_ylabel("Change Percentage (%)", color="#e74c3c", fontsize=11)
        ax1.tick_params(axis="y", labelcolor="#e74c3c")
        ax2.set_ylabel("Number of Changed Regions", color="#2980b9", fontsize=11)
        ax2.tick_params(axis="y", labelcolor="#2980b9")

        # Value labels on bars
        for bar, val in zip(bars, changes):
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.2,
                     f"{val:.1f}%", ha="center", va="bottom",
                     fontsize=9, fontweight="bold")

        ax1.set_title("Change Progression — Dhaka Consecutive Year Pairs",
                      fontsize=13, fontweight="bold")
        ax1.grid(axis="y", linestyle="--", alpha=0.4)
        fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.88), fontsize=9)
        plt.tight_layout()

        if save:
            fname = self.save_dir / "change_progression.png"
            save_figure(fig, fname, dpi=self.dpi)
            logger.info(f"  Saved change progression: {fname}")

        return fig

    def save_metrics_csv(self, df: pd.DataFrame) -> str:
        """Save the metrics DataFrame to CSV and return the path."""
        path = self.save_dir / "metrics_timeline.csv"
        df.to_csv(path)
        logger.info(f"  Metrics CSV saved: {path}")
        return str(path)

    # ── Private Metric Extractors ─────────────────────────────────────────

    def _compute_image_metrics(self, img: SatelliteImage) -> Dict[str, float]:
        """
        Derive all configured per-image metrics from pixel data.

        Metric definitions
        ──────────────────
        mean_intensity  : Average brightness (0–255) across all channels.
        urban_index     : Fraction of 'bright' pixels (proxy for impervious
                          surface — concrete, roads, buildings).
        ndvi_proxy      : (G − R) / (G + R + ε) using visible channels as a
                          rough vegetation indicator (positive = more green).
        entropy         : Shannon entropy of the grayscale histogram —
                          high entropy = high textural complexity (urban).
        """
        arr = img.array.astype(np.float32)
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

        metrics: Dict[str, float] = {}

        # ── Mean intensity ─────────────────────────────────────────────
        if "mean_intensity" in self.metrics:
            metrics["mean_intensity"] = float(arr.mean())

        # ── Urban index (bright pixel fraction) ────────────────────────
        if "urban_index" in self.metrics:
            gray            = 0.299 * r + 0.587 * g + 0.114 * b
            threshold       = 160.0         # Typical concrete / road brightness
            urban_fraction  = float((gray > threshold).mean())
            metrics["urban_index"] = round(urban_fraction, 5)

        # ── NDVI proxy ────────────────────────────────────────────────
        if "ndvi_proxy" in self.metrics:
            ndvi_proxy = ((g - r) / (g + r + 1e-6)).mean()
            metrics["ndvi_proxy"] = round(float(ndvi_proxy), 5)

        # ── Image entropy ─────────────────────────────────────────────
        if "entropy" in self.metrics:
            gray_uint8 = cv2.cvtColor(img.array, cv2.COLOR_RGB2GRAY)
            hist, _    = np.histogram(gray_uint8.ravel(), bins=256,
                                      range=(0, 256))
            p          = hist / (hist.sum() + 1e-12)
            p_nonzero  = p[p > 0]
            entropy    = -float(np.sum(p_nonzero * np.log2(p_nonzero)))
            metrics["entropy"] = round(entropy, 5)

        return metrics
