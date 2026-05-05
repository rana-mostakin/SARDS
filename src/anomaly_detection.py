"""
anomaly_detection.py — SARDS Anomaly Detection Module
======================================================
Identifies statistically unusual frames or change events in the
satellite time-series using three complementary methods:

  1. Z-score   — flags frames whose metric value deviates > N sigma
                  from the series mean (fast, parametric)
  2. IQR fence — robust non-parametric outlier detection
                  (good for skewed distributions)
  3. Isolation Forest — tree-based unsupervised ML detector
                  (handles multivariate anomalies simultaneously)

Outputs
───────
  - AnomalyReport dataclass per metric
  - Combined anomaly score (0–1 per frame)
  - Annotated time-series plot with flagged frames highlighted

Author : SARDS Team
Version: 1.0.0
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from src.utils import load_config, setup_logger, ensure_dir, save_figure


# ── Logger ────────────────────────────────────────────────────────────────────
logger = setup_logger("AnomalyDetection")


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class AnomalyReport:
    """
    Result of running one detection method on one time-series metric.

    Attributes
    ----------
    metric        : Name of the metric analysed.
    method        : Detection method used.
    anomaly_years : Years flagged as anomalous.
    scores        : Per-year anomaly score (higher = more anomalous).
    details       : Human-readable explanation strings.
    """
    metric        : str
    method        : str
    anomaly_years : List[int]            = field(default_factory=list)
    scores        : Dict[int, float]     = field(default_factory=dict)
    details       : Dict[int, str]       = field(default_factory=dict)

    @property
    def has_anomalies(self) -> bool:
        return len(self.anomaly_years) > 0

    def __repr__(self) -> str:
        return (f"AnomalyReport(metric='{self.metric}', "
                f"method='{self.method}', anomalies={self.anomaly_years})")


# ── Detector ──────────────────────────────────────────────────────────────────

class AnomalyDetector:
    """
    Multi-method anomaly detector for satellite time-series data.

    Parameters
    ----------
    config_path : str
        Path to SARDS config.yaml.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.cfg  = load_config(config_path)
        ad_cfg    = self.cfg["anomaly_detection"]
        out_cfg   = self.cfg["output"]

        self.method           = ad_cfg["method"]
        self.zscore_thresh    = ad_cfg["zscore_threshold"]
        self.iqr_mult         = ad_cfg["iqr_multiplier"]
        self.contamination    = ad_cfg["contamination"]
        self.dpi              = out_cfg["dpi"]
        self.save_dir         = Path(self.cfg["data"]["results_dir"]) / "time_series"
        ensure_dir(self.save_dir)

    # ── Public API ────────────────────────────────────────────────────────

    def detect(
        self,
        df       : pd.DataFrame,
        metric   : str = "change_pct",
        method   : Optional[str] = None
    ) -> AnomalyReport:
        """
        Run anomaly detection on one column of the metrics DataFrame.

        Parameters
        ----------
        df     : Time-series DataFrame (index = year).
        metric : Column name to analyse.
        method : Override the configured detection method.

        Returns
        -------
        AnomalyReport
        """
        method = method or self.method

        if metric not in df.columns:
            logger.warning(f"Metric '{metric}' not found in DataFrame. "
                           f"Available: {list(df.columns)}")
            return AnomalyReport(metric=metric, method=method)

        series = df[metric].dropna()
        if len(series) < 3:
            logger.warning(f"Not enough data points to detect anomalies in '{metric}'.")
            return AnomalyReport(metric=metric, method=method)

        years  = series.index.tolist()
        values = series.values.astype(float)

        if method == "zscore":
            report = self._zscore_detect(years, values, metric)
        elif method == "iqr":
            report = self._iqr_detect(years, values, metric)
        elif method == "isolation_forest":
            report = self._isolation_forest_detect(years, values, metric)
        else:
            logger.warning(f"Unknown method '{method}' — using zscore.")
            report = self._zscore_detect(years, values, metric)

        if report.has_anomalies:
            logger.info(f"  Anomalies in '{metric}' ({method}): "
                        f"years={report.anomaly_years}")
        else:
            logger.info(f"  No anomalies detected in '{metric}' ({method}).")

        return report

    def detect_all(
        self,
        df : pd.DataFrame
    ) -> Dict[str, AnomalyReport]:
        """
        Run *detect* on every numeric column in *df*.

        Returns
        -------
        dict  { metric_name: AnomalyReport }
        """
        reports = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            reports[col] = self.detect(df, metric=col)
        return reports

    def combined_anomaly_score(
        self,
        reports : Dict[str, AnomalyReport],
        years   : List[int]
    ) -> Dict[int, float]:
        """
        Aggregate anomaly scores across all metrics into one composite score
        per year (0 = normal, 1 = maximally anomalous).

        Parameters
        ----------
        reports : Dict from detect_all().
        years   : Full list of years to score.

        Returns
        -------
        dict  { year: composite_score (0–1) }
        """
        composite = {y: 0.0 for y in years}
        n_reports = max(len(reports), 1)

        for report in reports.values():
            for year, score in report.scores.items():
                if year in composite:
                    composite[year] += score / n_reports

        # Normalise to [0, 1]
        max_score = max(composite.values()) if composite else 1.0
        if max_score > 0:
            composite = {y: round(v / max_score, 4) for y, v in composite.items()}

        return composite

    def plot_anomalies(
        self,
        df      : pd.DataFrame,
        reports : Dict[str, AnomalyReport],
        save    : bool = True
    ) -> plt.Figure:
        """
        Plot time-series with anomalous points highlighted.

        Parameters
        ----------
        df      : Metrics DataFrame.
        reports : Dict of AnomalyReport per metric.
        save    : Save figure to disk.

        Returns
        -------
        matplotlib Figure
        """
        cols = [c for c in df.columns if c in reports and
                not df[c].isna().all()]
        n    = len(cols)
        if n == 0:
            return None

        fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True)
        if n == 1:
            axes = [axes]

        palette = sns.color_palette("Set2", n)
        years   = df.index.tolist()

        for ax, col, color in zip(axes, cols, palette):
            y = df[col].values.astype(float)
            report = reports.get(col)

            ax.plot(years, y, "o-", color=color, linewidth=2,
                    markersize=6, label=col, zorder=3)

            # Highlight anomaly years
            if report and report.has_anomalies:
                for yr in report.anomaly_years:
                    if yr in df.index:
                        val = df.at[yr, col]
                        ax.scatter([yr], [val], s=150, color="red",
                                   zorder=5, marker="*",
                                   label=f"Anomaly ({yr})")
                        ax.annotate(f"⚠ {yr}",
                                    xy=(yr, val),
                                    xytext=(3, 6),
                                    textcoords="offset points",
                                    fontsize=8, color="darkred")

            ax.set_title(f"{col} — Anomaly Detection ({self.method})",
                         fontsize=10, fontweight="bold")
            ax.set_ylabel(col, fontsize=9)
            ax.legend(fontsize=8, loc="upper left")
            ax.grid(axis="y", linestyle="--", alpha=0.4)

        axes[-1].set_xlabel("Year", fontsize=11)
        plt.xticks(years, [str(y) for y in years], rotation=45)
        fig.suptitle("SARDS — Anomaly Detection in Dhaka Time-Series",
                     fontsize=13, fontweight="bold", y=1.01)
        plt.tight_layout()

        if save:
            fname = self.save_dir / "anomaly_detection.png"
            save_figure(fig, fname, dpi=self.dpi)
            logger.info(f"  Saved anomaly plot: {fname}")

        return fig

    # ── Detection methods ─────────────────────────────────────────────────

    def _zscore_detect(
        self, years: List[int], values: np.ndarray, metric: str
    ) -> AnomalyReport:
        """Flag years where |z-score| exceeds self.zscore_thresh."""
        mu  = values.mean()
        sig = values.std() + 1e-10
        z   = (values - mu) / sig

        report = AnomalyReport(metric=metric, method="zscore")
        for yr, zval in zip(years, z):
            score = float(abs(zval) / (self.zscore_thresh + 1e-10))
            report.scores[yr] = min(score, 1.0)
            if abs(zval) > self.zscore_thresh:
                report.anomaly_years.append(yr)
                direction = "above" if zval > 0 else "below"
                report.details[yr] = (
                    f"z={zval:+.2f} ({direction} mean by "
                    f"{abs(zval)*sig:.2f} units)"
                )
        return report

    def _iqr_detect(
        self, years: List[int], values: np.ndarray, metric: str
    ) -> AnomalyReport:
        """Flag years outside [Q1 − k*IQR, Q3 + k*IQR] fence."""
        q1, q3 = np.percentile(values, [25, 75])
        iqr    = q3 - q1
        lo     = q1 - self.iqr_mult * iqr
        hi     = q3 + self.iqr_mult * iqr

        report = AnomalyReport(metric=metric, method="iqr")
        for yr, v in zip(years, values):
            dist = max(0.0, max(lo - v, v - hi))
            score = dist / ((iqr * self.iqr_mult) + 1e-10)
            report.scores[yr] = min(float(score), 1.0)
            if v < lo or v > hi:
                report.anomaly_years.append(yr)
                side = "above upper" if v > hi else "below lower"
                report.details[yr] = (
                    f"value={v:.3f} — {side} fence ({lo:.3f}, {hi:.3f})"
                )
        return report

    def _isolation_forest_detect(
        self, years: List[int], values: np.ndarray, metric: str
    ) -> AnomalyReport:
        """Use Isolation Forest (scikit-learn) for ML-based detection."""
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            logger.warning("scikit-learn not installed; falling back to z-score.")
            return self._zscore_detect(years, values, metric)

        X = values.reshape(-1, 1)
        clf = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100
        )
        preds  = clf.fit_predict(X)       # 1=normal, -1=anomaly
        scores = clf.decision_function(X) # Higher = more normal

        # Normalise anomaly scores to [0, 1] (higher = more anomalous)
        raw_scores = -scores              # Flip so anomalous → high
        norm_scores = (raw_scores - raw_scores.min()) / \
                      (raw_scores.max() - raw_scores.min() + 1e-10)

        report = AnomalyReport(metric=metric, method="isolation_forest")
        for yr, pred, nscore in zip(years, preds, norm_scores):
            report.scores[yr] = round(float(nscore), 4)
            if pred == -1:
                report.anomaly_years.append(yr)
                report.details[yr] = (
                    f"Isolation Forest anomaly score={nscore:.3f}"
                )
        return report
