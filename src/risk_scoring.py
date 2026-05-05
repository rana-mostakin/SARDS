"""
risk_scoring.py — SARDS Risk Scoring Module
============================================
Combines multiple evidence streams into a single composite risk score
for each year and for the overall location:

  Score = Σ wᵢ × normalised_metricᵢ

Evidence streams (configurable weights in config.yaml)
───────────────────────────────────────────────────────
  1. change_percentage   — % of pixels that changed (most recent pair)
  2. heatmap_intensity   — mean heatmap intensity over the change area
  3. temporal_trend      — slope magnitude from the linear trend
  4. anomaly_flag        — combined anomaly score from AnomalyDetector

Risk categories (configurable thresholds)
──────────────────────────────────────────
  🟢 LOW      [0.0 – 0.30)
  🟡 MEDIUM   [0.30 – 0.60)
  🔴 HIGH     [0.60 – 0.80)
  🚨 CRITICAL [0.80 – 1.00]

Outputs
───────
  - RiskReport dataclass
  - Year-by-year risk score table
  - Risk radar/bar chart
  - Plain-text summary report saved to results/reports/

Author : SARDS Team
Version: 1.0.0
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from src.utils import load_config, setup_logger, ensure_dir, save_figure, timestamp


# ── Logger ────────────────────────────────────────────────────────────────────
logger = setup_logger("RiskScoring")


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class RiskReport:
    """
    Final risk assessment for one location / time period.

    Attributes
    ----------
    location        : Name of the monitored area.
    overall_score   : Composite risk score in [0, 1].
    risk_label      : Human-readable risk category (e.g. '🔴 HIGH RISK').
    year_scores     : Dict { year: score } for the full timeline.
    component_scores: Dict { component_name: normalised_value }.
    summary_text    : Multi-line plain-text narrative report.
    generated_at    : ISO-format timestamp.
    """
    location         : str
    overall_score    : float
    risk_label       : str
    year_scores      : Dict[int, float]   = field(default_factory=dict)
    component_scores : Dict[str, float]   = field(default_factory=dict)
    summary_text     : str                = ""
    generated_at     : str                = field(
        default_factory=lambda: datetime.datetime.now().isoformat(timespec="seconds")
    )

    @property
    def is_high_risk(self) -> bool:
        return self.overall_score >= 0.60

    def __repr__(self) -> str:
        return (f"RiskReport(location='{self.location}', "
                f"score={self.overall_score:.3f}, label='{self.risk_label}')")


# ── Scorer ────────────────────────────────────────────────────────────────────

class RiskScorer:
    """
    Calculates composite risk scores from multi-evidence satellite analytics.

    Parameters
    ----------
    config_path : str
        Path to SARDS config.yaml.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.cfg     = load_config(config_path)
        rs_cfg       = self.cfg["risk_scoring"]
        out_cfg      = self.cfg["output"]

        self.weights    = rs_cfg["weights"]
        self.thresholds = rs_cfg["thresholds"]
        self.labels     = rs_cfg["labels"]
        self.location   = self.cfg["location"]["name"]
        self.dpi        = out_cfg["dpi"]
        self.save_dir   = Path(self.cfg["data"]["results_dir"])
        self.report_dir = self.save_dir / "reports"
        ensure_dir(self.report_dir)

    # ── Public API ────────────────────────────────────────────────────────

    def compute(
        self,
        metrics_df       : pd.DataFrame,
        trends           : Dict,
        anomaly_scores   : Dict[int, float],
        change_results   : list,
        heatmap_array    : Optional[np.ndarray] = None,
    ) -> RiskReport:
        """
        Compute the full risk report.

        Parameters
        ----------
        metrics_df       : DataFrame from TimeSeriesAnalyzer.extract_metrics().
        trends           : Dict from TimeSeriesAnalyzer.compute_trends().
        anomaly_scores   : Dict {year: score} from AnomalyDetector.combined_anomaly_score().
        change_results   : List of ChangeResult objects.
        heatmap_array    : Optional cumulative heatmap (H,W,3) uint8.

        Returns
        -------
        RiskReport
        """
        logger.info("Computing composite risk score …")

        # ── 1. Change percentage component ────────────────────────────
        change_pct_score = self._change_pct_component(change_results)

        # ── 2. Heatmap intensity component ────────────────────────────
        heatmap_score = self._heatmap_component(heatmap_array, metrics_df)

        # ── 3. Temporal trend component ────────────────────────────────
        trend_score = self._trend_component(trends)

        # ── 4. Anomaly component ───────────────────────────────────────
        anomaly_score = self._anomaly_component(anomaly_scores)

        components = {
            "change_percentage" : change_pct_score,
            "heatmap_intensity" : heatmap_score,
            "temporal_trend"    : trend_score,
            "anomaly_flag"      : anomaly_score,
        }

        # ── Weighted sum ──────────────────────────────────────────────
        overall = sum(
            self.weights.get(k, 0.25) * v
            for k, v in components.items()
        )
        overall = float(np.clip(overall, 0.0, 1.0))

        # ── Year-by-year scores ───────────────────────────────────────
        year_scores = self._per_year_scores(
            metrics_df, change_results, anomaly_scores
        )

        # ── Label ─────────────────────────────────────────────────────
        label = self._classify(overall)

        # ── Narrative ─────────────────────────────────────────────────
        narrative = self._generate_narrative(
            overall, label, components, metrics_df,
            trends, change_results, year_scores
        )

        report = RiskReport(
            location         = self.location,
            overall_score    = round(overall, 4),
            risk_label       = label,
            year_scores      = year_scores,
            component_scores = {k: round(v, 4) for k, v in components.items()},
            summary_text     = narrative,
        )

        logger.info(f"  Overall risk: {overall:.3f}  →  {label}")
        return report

    def plot_risk_dashboard(
        self,
        report : RiskReport,
        save   : bool = True
    ) -> plt.Figure:
        """
        Create a four-panel risk dashboard:
          Top-left  : Component bar chart
          Top-right : Overall risk gauge (donut)
          Bottom    : Year-by-year risk bar chart

        Parameters
        ----------
        report : RiskReport from compute().
        save   : Save figure to disk.

        Returns
        -------
        matplotlib Figure
        """
        fig = plt.figure(figsize=(14, 9))
        gs  = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)

        ax_comp   = fig.add_subplot(gs[0, 0])
        ax_gauge  = fig.add_subplot(gs[0, 1])
        ax_years  = fig.add_subplot(gs[1, :])

        # ── Component scores bar ──────────────────────────────────────
        comp_names  = list(report.component_scores.keys())
        comp_values = list(report.component_scores.values())
        comp_colors = [self._score_color(v) for v in comp_values]

        bars = ax_comp.barh(comp_names, comp_values, color=comp_colors,
                            height=0.55, edgecolor="white")
        ax_comp.set_xlim(0, 1)
        ax_comp.set_xlabel("Normalised Score", fontsize=9)
        ax_comp.set_title("Risk Component Scores", fontsize=11, fontweight="bold")
        for bar, val in zip(bars, comp_values):
            ax_comp.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                         f"{val:.2f}", va="center", fontsize=9)
        ax_comp.axvline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
        ax_comp.grid(axis="x", linestyle="--", alpha=0.3)

        # ── Overall gauge donut ────────────────────────────────────────
        score = report.overall_score
        gauge_vals = [score, 1.0 - score]
        gauge_col  = [self._score_color(score), "#eeeeee"]
        wedges, _ = ax_gauge.pie(
            gauge_vals, colors=gauge_col,
            startangle=90, counterclock=False,
            wedgeprops=dict(width=0.45, edgecolor="white")
        )
        ax_gauge.text(0, 0, f"{score:.0%}", ha="center", va="center",
                      fontsize=22, fontweight="bold",
                      color=self._score_color(score))
        ax_gauge.set_title(f"Overall Risk Score\n{report.risk_label}",
                           fontsize=11, fontweight="bold")

        # ── Year-by-year bar chart ─────────────────────────────────────
        if report.year_scores:
            ys = sorted(report.year_scores.keys())
            vs = [report.year_scores[y] for y in ys]
            ycolors = [self._score_color(v) for v in vs]
            ax_years.bar(ys, vs, color=ycolors, width=0.6, edgecolor="white")
            ax_years.axhline(0.5,  color="gold",   linestyle="--",
                             linewidth=1.2, label="Medium threshold")
            ax_years.axhline(0.60, color="orange", linestyle="--",
                             linewidth=1.2, label="High threshold")
            ax_years.axhline(0.80, color="red",    linestyle="--",
                             linewidth=1.2, label="Critical threshold")
            ax_years.set_ylim(0, 1.05)
            ax_years.set_xlabel("Year", fontsize=11)
            ax_years.set_ylabel("Risk Score", fontsize=11)
            ax_years.set_title("Year-by-Year Risk Score", fontsize=11,
                                fontweight="bold")
            ax_years.legend(fontsize=8, loc="upper left")
            ax_years.set_xticks(ys)
            ax_years.set_xticklabels([str(y) for y in ys], rotation=30)
            for x, v in zip(ys, vs):
                ax_years.text(x, v + 0.02, f"{v:.2f}", ha="center",
                              fontsize=8, fontweight="bold")

        fig.suptitle(f"SARDS Risk Dashboard — {self.location}",
                     fontsize=15, fontweight="bold", y=1.01)

        if save:
            fname = self.report_dir / "risk_dashboard.png"
            save_figure(fig, fname, dpi=self.dpi)
            logger.info(f"  Saved risk dashboard: {fname}")

        return fig

    def save_report(self, report: RiskReport) -> str:
        """
        Write the plain-text report to disk and return the path.
        """
        ts    = timestamp()
        fname = self.report_dir / f"risk_report_{ts}.txt"
        fname.write_text(report.summary_text, encoding="utf-8")
        logger.info(f"  Saved text report: {fname}")
        return str(fname)

    # ── Score components ──────────────────────────────────────────────────

    def _change_pct_component(self, change_results: list) -> float:
        """Normalise average change % across all pairs to [0, 1]."""
        if not change_results:
            return 0.0
        pcts = [r.stats.get("change_pct", 0.0) for r in change_results]
        avg  = np.mean(pcts)
        # Saturates at 30% change → score = 1.0
        return float(np.clip(avg / 30.0, 0, 1))

    def _heatmap_component(
        self,
        heatmap : Optional[np.ndarray],
        df      : pd.DataFrame
    ) -> float:
        """Compute mean normalised intensity from the cumulative heatmap."""
        if heatmap is not None:
            return float(heatmap.astype(np.float32).mean() / 255.0)
        # Fallback: use mean_intensity from df if heatmap unavailable
        if "mean_intensity" in df.columns:
            val = df["mean_intensity"].mean()
            return float(np.clip(val / 255.0, 0, 1))
        return 0.0

    def _trend_component(self, trends: Dict) -> float:
        """Convert trend slope magnitudes to a 0–1 score."""
        if not trends:
            return 0.0
        # Collect absolute slopes from key urban/change metrics
        slopes = []
        priority_metrics = ["urban_index", "change_pct", "entropy", "mean_intensity"]
        for m in priority_metrics:
            if m in trends:
                slopes.append(abs(trends[m]["slope"]))
        if not slopes:
            slopes = [abs(t["slope"]) for t in trends.values()]
        if not slopes:
            return 0.0
        max_slope = max(slopes)
        # Normalise: slope of 0.05 per year → score ≈ 1.0
        return float(np.clip(max_slope / 0.05, 0, 1))

    def _anomaly_component(self, anomaly_scores: Dict[int, float]) -> float:
        """Average of all per-year anomaly scores."""
        if not anomaly_scores:
            return 0.0
        return float(np.mean(list(anomaly_scores.values())))

    def _per_year_scores(
        self,
        df             : pd.DataFrame,
        change_results : list,
        anomaly_scores : Dict[int, float],
    ) -> Dict[int, float]:
        """Compute a simplified risk score for every year independently."""
        year_scores = {}

        # Build a dict of change_pct per "after" year
        change_map = {r.year_after: r.stats.get("change_pct", 0.0)
                      for r in change_results}

        for year in df.index:
            # Component 1: change %
            cp = float(np.clip(change_map.get(year, 0.0) / 30.0, 0, 1))

            # Component 2: urban index
            ui = 0.0
            if "urban_index" in df.columns:
                ui_raw = df.at[year, "urban_index"]
                if not np.isnan(ui_raw):
                    ui = float(np.clip(ui_raw, 0, 1))

            # Component 3: anomaly
            anom = anomaly_scores.get(year, 0.0)

            score = (
                self.weights.get("change_percentage", 0.35) * cp
                + self.weights.get("heatmap_intensity", 0.30) * ui
                + self.weights.get("anomaly_flag", 0.15) * anom
            )
            year_scores[year] = round(float(np.clip(score, 0, 1)), 4)

        return year_scores

    # ── Classification ────────────────────────────────────────────────────

    def _classify(self, score: float) -> str:
        """Map a 0–1 score to a labelled risk category."""
        for level, (lo, hi) in self.thresholds.items():
            if lo <= score < hi:
                return self.labels[level]
        return self.labels["critical"]   # Edge case: score == 1.0

    @staticmethod
    def _score_color(score: float) -> str:
        """Map a 0–1 score to a matplotlib colour string."""
        if score < 0.30:
            return "#27ae60"    # Green
        elif score < 0.60:
            return "#f39c12"    # Amber
        elif score < 0.80:
            return "#e74c3c"    # Red
        else:
            return "#8e44ad"    # Purple — critical

    # ── Report narrative ──────────────────────────────────────────────────

    def _generate_narrative(
        self,
        overall   : float,
        label     : str,
        components: Dict,
        df        : pd.DataFrame,
        trends    : Dict,
        changes   : list,
        year_scores: Dict,
    ) -> str:
        """Build a human-readable plain-text summary report."""
        ts     = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        years  = sorted(df.index.tolist())
        yr_str = f"{min(years)}–{max(years)}" if years else "N/A"

        lines = [
            "=" * 68,
            " SARDS — SATELLITE-BASED ACTIVITY RISK DETECTION SYSTEM",
            " Risk Assessment Report",
            "=" * 68,
            f" Location    : {self.location}",
            f" Period       : {yr_str}",
            f" Generated at : {ts}",
            "=" * 68,
            "",
            f" OVERALL RISK SCORE : {overall:.4f}  →  {label}",
            "",
            " COMPONENT BREAKDOWN",
            " " + "-" * 40,
        ]

        for k, v in components.items():
            bar   = "█" * int(v * 20)
            pad   = "░" * (20 - int(v * 20))
            lines.append(f"  {k:<22} [{bar}{pad}]  {v:.3f}")

        lines += ["", " YEAR-BY-YEAR RISK SCORES", " " + "-" * 40]
        for yr in sorted(year_scores.keys()):
            sc  = year_scores[yr]
            cat = self._classify(sc).split(" ", 1)[-1] if " " in self._classify(sc) else self._classify(sc)
            lines.append(f"  {yr}  →  {sc:.3f}   {cat}")

        # Change progression section
        if changes:
            lines += ["", " CHANGE DETECTION SUMMARY", " " + "-" * 40]
            for r in changes:
                lines.append(
                    f"  {r.year_before} → {r.year_after} : "
                    f"{r.stats.get('change_pct',0):.2f}% changed, "
                    f"{r.stats.get('contour_count',0)} regions, "
                    f"mean diff={r.stats.get('mean_diff',0):.1f}"
                )

        # Trend section
        if trends:
            lines += ["", " TREND ANALYSIS (Linear OLS)", " " + "-" * 40]
            for m, t in trends.items():
                lines.append(
                    f"  {m:<22} slope={t['slope']:+.5f}  "
                    f"R²={t['r2']:.3f}  [{t['trend']}]"
                )

        # Dhaka case study interpretation
        lines += [
            "",
            " DHAKA CASE STUDY INTERPRETATION",
            " " + "-" * 40,
        ]
        ui_trend = trends.get("urban_index", {}).get("trend", "unknown")
        nd_trend = trends.get("ndvi_proxy",  {}).get("trend", "unknown")
        lines.append(
            f"  Urban Expansion  : {ui_trend.upper()} — "
            + ("Significant urban growth detected." if ui_trend == "increasing"
               else "Urban area relatively stable.")
        )
        lines.append(
            f"  Vegetation Cover : {nd_trend.upper()} — "
            + ("Vegetation loss detected (deforestation / urbanisation)."
               if nd_trend == "decreasing"
               else "Vegetation cover stable or recovering.")
        )

        lines += [
            "",
            " RECOMMENDATIONS",
            " " + "-" * 40,
        ]
        if overall >= 0.80:
            lines.append("  [CRITICAL] Immediate field verification recommended.")
            lines.append("  Consider escalating to emergency urban planning review.")
        elif overall >= 0.60:
            lines.append("  [HIGH] Detailed analysis and ground-truthing advised.")
            lines.append("  Notify relevant environmental and urban authorities.")
        elif overall >= 0.30:
            lines.append("  [MEDIUM] Routine monitoring schedule should continue.")
            lines.append("  Investigate anomalous years more closely.")
        else:
            lines.append("  [LOW] No immediate action required.")
            lines.append("  Continue standard monitoring cadence.")

        lines += ["", "=" * 68, " End of Report", "=" * 68]
        return "\n".join(lines)
