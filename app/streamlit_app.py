"""
streamlit_app.py — SARDS Interactive Dashboard
================================================
A full-featured Streamlit web application for the Satellite-Based
Activity Risk Detection System.

Features
─────────
  • Sidebar controls for dataset / year selection
  • Upload custom images OR use the Dhaka demo dataset
  • Change detection comparison viewer
  • Heatmap overlay with colormap selector
  • Time-series multi-metric charts
  • Anomaly timeline with highlighted years
  • Risk score gauge with component breakdown
  • Downloadable text report

Run:
    streamlit run app/streamlit_app.py

Author : SARDS Team
Version: 1.0.0
"""

import sys
import os
import io
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cv2
from PIL import Image

# Ensure project root is on Python path regardless of cwd
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Streamlit Cloud: auto-generate demo data if images are missing
_demo_check = ROOT / "data" / "dhaka" / "2018.jpg"
if not _demo_check.exists():
    try:
        import subprocess
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "generate_demo_data.py")],
            check=True, capture_output=True
        )
    except Exception:
        pass

from src.utils            import load_config, ensure_dir
from src.image_loader     import ImageLoader, SatelliteImage, ImageDataset
from src.preprocessing    import Preprocessor
from src.change_detection import ChangeDetector
from src.heatmap_generator import HeatmapGenerator
from src.time_series_analysis import TimeSeriesAnalyzer
from src.anomaly_detection import AnomalyDetector
from src.risk_scoring      import RiskScorer


# ══════════════════════════════════════════════════════════════════════════════
# Page configuration
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title  = "SARDS — Satellite Risk Detection",
    page_icon   = "🛰️",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)


# ══════════════════════════════════════════════════════════════════════════════
# Styling injection
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* Main header gradient */
.main-header {
    background: linear-gradient(135deg, #0a1628 0%, #1a3a5c 50%, #0d2137 100%);
    padding: 2rem 2rem 1.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.main-header h1 {
    color: #00d4ff;
    font-size: 2.2rem;
    margin: 0;
    letter-spacing: 2px;
}
.main-header p {
    color: #7fb3d3;
    font-size: 0.95rem;
    margin: 0.4rem 0 0;
}

/* Metric cards */
.metric-card {
    background: #1a2744;
    border: 1px solid #2d4a7a;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.2);
}
.metric-card .val {
    font-size: 2rem;
    font-weight: 700;
    color: #00d4ff;
}
.metric-card .lbl {
    font-size: 0.8rem;
    color: #8899aa;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Risk badge */
.risk-badge {
    display: inline-block;
    padding: 0.5rem 1.5rem;
    border-radius: 25px;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 1px;
    margin: 0.5rem 0;
}
.risk-low      { background: #1e4d2b; color: #4cdd7a; border: 2px solid #27ae60; }
.risk-medium   { background: #4d3a10; color: #ffc04d; border: 2px solid #f39c12; }
.risk-high     { background: #4d1010; color: #ff6060; border: 2px solid #e74c3c; }
.risk-critical { background: #2d1040; color: #c77dff; border: 2px solid #8e44ad; }

/* Section headers */
.section-header {
    border-left: 4px solid #00d4ff;
    padding-left: 0.75rem;
    color: #e8f4fd;
    font-size: 1.1rem;
    font-weight: 600;
    margin: 1.5rem 0 1rem;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_config():
    config_path = ROOT / "config.yaml"
    return load_config(str(config_path))

@st.cache_resource
def get_loader():
    return ImageLoader(str(ROOT / "config.yaml"))

@st.cache_data
def load_dhaka_dataset():
    """Cache the Dhaka demo dataset to avoid re-loading on every interaction."""
    loader = get_loader()
    try:
        dataset  = loader.load_dhaka_timeline()
        prep     = Preprocessor(str(ROOT / "config.yaml"))
        proc_ds  = prep.process_dataset(dataset)
        return proc_ds, None
    except Exception as e:
        return None, str(e)


def arr_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert an RGB uint8 numpy array to a PIL Image."""
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def fig_to_bytes(fig: plt.Figure) -> bytes:
    """Render a Matplotlib figure to PNG bytes for st.image()."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf.read()


def risk_badge_html(label: str) -> str:
    """Return an HTML badge for the risk label string."""
    if "LOW"      in label: cls = "risk-low"
    elif "MEDIUM" in label: cls = "risk-medium"
    elif "CRITICAL" in label: cls = "risk-critical"
    else:                   cls = "risk-high"
    return f'<span class="risk-badge {cls}">{label}</span>'


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar() -> dict:
    """Render sidebar controls and return selected parameters."""
    with st.sidebar:
        st.markdown("## 🛰️ SARDS Controls")
        st.markdown("---")

        # Dataset selection
        st.markdown("### 📂 Data Source")
        data_mode = st.radio(
            "Choose dataset",
            ["🗂️ Dhaka Demo Dataset", "📤 Upload Your Own Images"],
            index=0
        )

        uploaded_files = []
        if data_mode == "📤 Upload Your Own Images":
            uploaded_files = st.file_uploader(
                "Upload satellite images (min. 2)",
                type=["jpg", "jpeg", "png", "tif"],
                accept_multiple_files=True,
                help="Upload at least 2 images. Filename should contain year (e.g. 2020.jpg)."
            )

        st.markdown("---")
        st.markdown("### ⚙️ Analysis Settings")

        method = st.selectbox(
            "Change Detection Method",
            ["absolute_diff", "ssim", "optical_flow"],
            index=0,
            help=(
                "absolute_diff: fast pixel comparison\n"
                "ssim: perceptual structural similarity\n"
                "optical_flow: motion-based detection"
            )
        )

        colormap = st.selectbox(
            "Heatmap Colormap",
            ["jet", "hot", "inferno", "viridis", "plasma", "RdYlGn_r"],
            index=0
        )

        threshold = st.slider(
            "Change Threshold (0–255)",
            min_value=5, max_value=100, value=25, step=5,
            help="Pixels with difference above this are flagged as changed."
        )

        anomaly_method = st.selectbox(
            "Anomaly Detection Method",
            ["zscore", "iqr", "isolation_forest"],
            index=0
        )

        st.markdown("---")
        run_btn = st.button("🚀 Run Full Analysis", use_container_width=True,
                            type="primary")

        st.markdown("---")
        st.markdown("""
<small>
**SARDS v1.0.0**  
Focus: Dhaka, Bangladesh  
Python · OpenCV · Streamlit  
</small>
""", unsafe_allow_html=True)

    return {
        "data_mode"      : data_mode,
        "uploaded_files" : uploaded_files,
        "method"         : method,
        "colormap"       : colormap,
        "threshold"      : threshold,
        "anomaly_method" : anomaly_method,
        "run"            : run_btn,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Tab renderers
# ══════════════════════════════════════════════════════════════════════════════

def render_overview_tab(dataset: ImageDataset) -> None:
    """Show all images in a grid with basic metadata."""
    st.markdown('<div class="section-header">📸 Satellite Image Timeline</div>',
                unsafe_allow_html=True)

    imgs  = dataset.as_list()
    n     = len(imgs)
    ncols = min(4, n)
    cols  = st.columns(ncols)

    for idx, img in enumerate(imgs):
        with cols[idx % ncols]:
            st.image(arr_to_pil(img.array),
                     caption=f"Year: {img.year} | {img.width}×{img.height}",
                     use_container_width=True)
            brightness = img.array.mean()
            st.caption(f"Avg brightness: {brightness:.1f}")

    st.markdown("---")
    st.markdown(f"**{n} images loaded** spanning "
                f"**{dataset.years[0]}–{dataset.years[-1]}** "
                f"({dataset.location})")


def render_change_tab(dataset: ImageDataset, change_results: list,
                      cfg: dict) -> None:
    """Interactive before/after change detection viewer."""
    st.markdown('<div class="section-header">🔍 Change Detection</div>',
                unsafe_allow_html=True)

    if not change_results:
        st.warning("No change results available.")
        return

    pair_labels = [f"{r.year_before} → {r.year_after}" for r in change_results]
    selected    = st.selectbox("Select year pair", pair_labels)
    res         = change_results[pair_labels.index(selected)]

    col1, col2, col3 = st.columns(3)

    before = dataset.get(res.year_before)
    after  = dataset.get(res.year_after)

    with col1:
        st.markdown(f"**Before — {res.year_before}**")
        if before:
            st.image(arr_to_pil(before.array), use_container_width=True)

    with col2:
        st.markdown(f"**After — {res.year_after}**")
        if after:
            st.image(arr_to_pil(after.array), use_container_width=True)

    with col3:
        st.markdown(f"**Change Overlay**")
        st.image(arr_to_pil(res.overlay), use_container_width=True)

    # Stats row
    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Changed Pixels",   f"{res.stats['changed_pixels']:,}")
    m2.metric("Change %",         f"{res.stats['change_pct']:.2f}%")
    m3.metric("Changed Regions",  f"{res.stats['contour_count']}")
    m4.metric("Mean Intensity Δ", f"{res.stats['mean_diff']:.1f}")

    # Difference map
    with st.expander("🔬 View Raw Difference Map"):
        diff_disp = (res.diff_map / (res.diff_map.max() + 1e-8) * 255).astype(np.uint8)
        st.image(diff_disp, caption="Raw pixel difference (grayscale)",
                 use_container_width=True)


def render_heatmap_tab(change_results: list, dataset: ImageDataset,
                       colormap: str) -> None:
    """Show heatmap visualisations for selected pairs."""
    st.markdown('<div class="section-header">🌡️ Change Heatmaps</div>',
                unsafe_allow_html=True)

    if not change_results:
        st.warning("Run the analysis first.")
        return

    heatgen = HeatmapGenerator(str(ROOT / "config.yaml"))
    heatgen.colormap = colormap     # Override with sidebar choice

    # Cumulative heatmap
    st.markdown("#### Cumulative Change Heatmap (All Years)")
    last_img = dataset.get(dataset.years[-1])
    cum_hm   = heatgen.generate_cumulative_heatmap(
        change_results, base_img=last_img, save=False
    )
    st.image(arr_to_pil(cum_hm), use_container_width=True,
             caption="Cumulative heatmap overlaid on most recent image")

    # Per-pair selector
    st.markdown("#### Per-Pair Heatmap")
    pair_labels = [f"{r.year_before} → {r.year_after}" for r in change_results]
    sel_pair    = st.selectbox("Select pair", pair_labels, key="hm_pair")
    res         = change_results[pair_labels.index(sel_pair)]
    base        = dataset.get(res.year_before)
    hm          = heatgen.generate_change_heatmap(res, base_img=base, save=False)

    col_a, col_b = st.columns(2)
    with col_a:
        st.image(arr_to_pil(hm), use_container_width=True,
                 caption=f"Heatmap: {sel_pair}")
    with col_b:
        # Colormap legend
        fig_leg, ax_leg = plt.subplots(figsize=(3, 5))
        fig_leg.patch.set_alpha(0)
        gradient = np.linspace(0, 1, 256).reshape(256, 1)
        ax_leg.imshow(gradient, aspect="auto", cmap=colormap,
                      origin="lower", extent=[0, 1, 0, 255])
        ax_leg.set_xlabel("Intensity", fontsize=9)
        ax_leg.set_ylabel("Change Magnitude (0–255)", fontsize=9)
        ax_leg.set_xticks([])
        ax_leg.set_title("Color Scale", fontsize=10)
        st.pyplot(fig_leg, use_container_width=True)
        plt.close(fig_leg)


def render_timeseries_tab(metrics_df: pd.DataFrame, trends: dict,
                          change_results: list) -> None:
    """Plot multi-metric time-series and change progression."""
    st.markdown('<div class="section-header">📈 Time-Series Analysis</div>',
                unsafe_allow_html=True)

    ts = TimeSeriesAnalyzer(str(ROOT / "config.yaml"))

    # Metric selector
    available = [c for c in metrics_df.columns if not metrics_df[c].isna().all()]
    selected_metrics = st.multiselect(
        "Select metrics to plot", available, default=available[:4]
    )
    if not selected_metrics:
        st.info("Select at least one metric above.")
        return

    sub_df = metrics_df[selected_metrics]
    fig    = ts.plot_metrics(sub_df, trends, save=False)
    if fig:
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # Change progression
    if change_results:
        st.markdown("#### Change Percentage Progression")
        fig2 = ts.plot_change_progression(change_results, save=False)
        if fig2:
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

    # Trend table
    st.markdown("#### Trend Summary Table")
    if trends:
        trend_rows = []
        for m, t in trends.items():
            arrow = "↑" if t["trend"] == "increasing" else (
                    "↓" if t["trend"] == "decreasing" else "→")
            trend_rows.append({
                "Metric"     : m,
                "Trend"      : f"{arrow} {t['trend']}",
                "Slope"      : f"{t['slope']:+.5f}",
                "R²"         : f"{t['r2']:.3f}",
                "p-value"    : f"{t['p_value']:.4f}",
            })
        st.dataframe(pd.DataFrame(trend_rows), use_container_width=True)

    # Raw data table
    with st.expander("📊 View Raw Metrics Table"):
        st.dataframe(metrics_df.style.format("{:.4f}"),
                     use_container_width=True)

        csv_bytes = metrics_df.to_csv().encode()
        st.download_button(
            "⬇️ Download CSV",
            data    = csv_bytes,
            file_name = "sards_metrics.csv",
            mime    = "text/csv"
        )


def render_anomaly_tab(metrics_df: pd.DataFrame,
                       anomaly_reports: dict,
                       combined_scores: dict) -> None:
    """Show anomaly detection results."""
    st.markdown('<div class="section-header">🔬 Anomaly Detection</div>',
                unsafe_allow_html=True)

    ad  = AnomalyDetector(str(ROOT / "config.yaml"))
    fig = ad.plot_anomalies(metrics_df, anomaly_reports, save=False)
    if fig:
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # Combined score chart
    st.markdown("#### Combined Anomaly Score per Year")
    years  = sorted(combined_scores.keys())
    scores = [combined_scores[y] for y in years]

    fig_bar, ax = plt.subplots(figsize=(10, 3))
    colors = ["#e74c3c" if s > 0.5 else "#27ae60" for s in scores]
    ax.bar(years, scores, color=colors, width=0.6, edgecolor="white")
    ax.axhline(0.5, color="orange", linestyle="--", linewidth=1.2,
               label="Anomaly threshold (0.5)")
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Year")
    ax.set_ylabel("Composite Anomaly Score")
    ax.set_title("Combined Anomaly Score by Year")
    ax.legend(fontsize=8)
    for x, v in zip(years, scores):
        ax.text(x, v + 0.03, f"{v:.2f}", ha="center", fontsize=8)
    st.pyplot(fig_bar, use_container_width=True)
    plt.close(fig_bar)

    # Summary table
    flagged = {yr: sc for yr, sc in combined_scores.items() if sc > 0.5}
    if flagged:
        st.warning(f"⚠️  **{len(flagged)} anomalous year(s) detected:** "
                   + ", ".join(str(y) for y in sorted(flagged.keys())))
    else:
        st.success("✅ No significant anomalies detected.")


def render_risk_tab(risk_report) -> None:
    """Show the final risk assessment dashboard."""
    st.markdown('<div class="section-header">🎯 Risk Assessment</div>',
                unsafe_allow_html=True)

    # Overall score
    col_score, col_gauge = st.columns([1, 2])
    with col_score:
        score = risk_report.overall_score
        st.markdown(f"### Overall Risk Score")
        st.markdown(f"## `{score:.4f}`")
        st.markdown(risk_badge_html(risk_report.risk_label),
                    unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("**Component Scores**")
        for k, v in risk_report.component_scores.items():
            pct = int(v * 100)
            label = k.replace("_", " ").title()
            st.markdown(f"`{label}`")
            st.progress(v, text=f"{v:.3f}")

    with col_gauge:
        # Gauge chart using matplotlib
        fig_g, ax_g = plt.subplots(figsize=(5, 5),
                                   subplot_kw=dict(polar=False))
        ax_g.set_xlim(0, 1)
        ax_g.set_ylim(0, 1)

        # Background segments
        segments = [
            (0.0, 0.30, "#27ae60", "LOW"),
            (0.30, 0.60, "#f39c12", "MEDIUM"),
            (0.60, 0.80, "#e74c3c", "HIGH"),
            (0.80, 1.00, "#8e44ad", "CRITICAL"),
        ]
        for (lo, hi, col, lbl) in segments:
            ax_g.barh(0.5, hi - lo, left=lo, height=0.3,
                      color=col, alpha=0.8, edgecolor="white")
            ax_g.text((lo + hi) / 2, 0.35, lbl,
                      ha="center", va="center", fontsize=9,
                      color="white", fontweight="bold")

        # Score needle
        ax_g.axvline(score, color="black", linewidth=3, zorder=5)
        ax_g.scatter([score], [0.5], s=100, color="black", zorder=6)
        ax_g.text(score, 0.72, f"{score:.2f}",
                  ha="center", fontsize=14, fontweight="bold",
                  color="black")
        ax_g.set_yticks([])
        ax_g.set_xlabel("Risk Score (0 → 1)", fontsize=10)
        ax_g.set_title("Risk Gauge", fontsize=12, fontweight="bold")
        ax_g.set_facecolor("#f8f9fa")
        st.pyplot(fig_g, use_container_width=True)
        plt.close(fig_g)

    # Year-by-year chart
    st.markdown("#### Year-by-Year Risk Scores")
    if risk_report.year_scores:
        yrs = sorted(risk_report.year_scores.keys())
        vls = [risk_report.year_scores[y] for y in yrs]
        fig_y, ax_y = plt.subplots(figsize=(10, 3.5))
        cols_y = ["#8e44ad" if v >= 0.80 else
                  "#e74c3c" if v >= 0.60 else
                  "#f39c12" if v >= 0.30 else
                  "#27ae60" for v in vls]
        ax_y.bar(yrs, vls, color=cols_y, width=0.6, edgecolor="white")
        for threshold, col, label in [
            (0.30, "gold",   "Medium"),
            (0.60, "orange", "High"),
            (0.80, "red",    "Critical"),
        ]:
            ax_y.axhline(threshold, color=col, linestyle="--",
                         linewidth=1.0, label=label)
        ax_y.set_ylim(0, 1.05)
        ax_y.set_xticks(yrs)
        ax_y.set_xticklabels([str(y) for y in yrs])
        ax_y.set_ylabel("Risk Score")
        ax_y.set_title("Year-by-Year Risk Score")
        ax_y.legend(fontsize=8, loc="upper left")
        for x, v in zip(yrs, vls):
            ax_y.text(x, v + 0.02, f"{v:.2f}", ha="center",
                      fontsize=8, fontweight="bold")
        st.pyplot(fig_y, use_container_width=True)
        plt.close(fig_y)

    # Text report
    st.markdown("#### 📋 Full Text Report")
    st.code(risk_report.summary_text, language="")
    st.download_button(
        "⬇️ Download Report (.txt)",
        data      = risk_report.summary_text,
        file_name = "sards_risk_report.txt",
        mime      = "text/plain"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main application
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── Header ────────────────────────────────────────────────────────
    st.markdown("""
<div class="main-header">
  <h1>🛰️ SARDS</h1>
  <p>Satellite-Based Activity Risk Detection System · Dhaka, Bangladesh</p>
</div>
""", unsafe_allow_html=True)

    # ── Sidebar ────────────────────────────────────────────────────────
    params = render_sidebar()
    cfg    = get_config()

    # ── Session state for pipeline results ────────────────────────────
    if "results" not in st.session_state:
        st.session_state.results = None

    # ── Trigger analysis ──────────────────────────────────────────────
    if params["run"]:
        with st.spinner("🔄 Running full SARDS pipeline …"):
            try:
                # ── Build dataset ─────────────────────────────────────
                if (params["data_mode"] == "📤 Upload Your Own Images"
                        and params["uploaded_files"]):
                    # Save uploads to a temp directory
                    tmp_dir = Path(tempfile.mkdtemp())
                    for f in params["uploaded_files"]:
                        save_path = tmp_dir / f.name
                        save_path.write_bytes(f.read())
                    loader  = ImageLoader(str(ROOT / "config.yaml"))
                    dataset = loader.load_from_directory(str(tmp_dir))
                else:
                    # Use the built-in Dhaka demo dataset
                    dataset, err = load_dhaka_dataset()
                    if err:
                        st.error(f"Could not load Dhaka dataset: {err}\n\n"
                                 "Run:  `python scripts/generate_demo_data.py`")
                        st.stop()

                # ── Pre-process ───────────────────────────────────────
                prep       = Preprocessor(str(ROOT / "config.yaml"))
                proc_ds    = prep.process_dataset(dataset)

                # ── Change detection ──────────────────────────────────
                detector             = ChangeDetector(str(ROOT / "config.yaml"))
                detector.method      = params["method"]
                detector.threshold   = params["threshold"]
                change_results = detector.detect_all_pairs(
                    proc_ds.as_list(), align=True
                )

                # ── Time-series ───────────────────────────────────────
                ts_analyzer = TimeSeriesAnalyzer(str(ROOT / "config.yaml"))
                metrics_df  = ts_analyzer.extract_metrics(proc_ds)
                metrics_df  = ts_analyzer.add_change_metrics(metrics_df,
                                                              change_results)
                trends      = ts_analyzer.compute_trends(metrics_df)

                # ── Anomaly detection ─────────────────────────────────
                ad              = AnomalyDetector(str(ROOT / "config.yaml"))
                ad.method       = params["anomaly_method"]
                anomaly_reports = ad.detect_all(metrics_df)
                combined_scores = ad.combined_anomaly_score(
                    anomaly_reports, proc_ds.years
                )

                # ── Risk scoring ──────────────────────────────────────
                scorer      = RiskScorer(str(ROOT / "config.yaml"))
                heatgen     = HeatmapGenerator(str(ROOT / "config.yaml"))
                heatgen.colormap = params["colormap"]
                cum_hm = None
                if change_results:
                    cum_hm = heatgen.generate_cumulative_heatmap(
                        change_results,
                        base_img=proc_ds.get(proc_ds.years[-1]),
                        save=False
                    )

                risk_report = scorer.compute(
                    metrics_df     = metrics_df,
                    trends         = trends,
                    anomaly_scores = combined_scores,
                    change_results = change_results,
                    heatmap_array  = cum_hm,
                )

                # ── Store in session state ────────────────────────────
                st.session_state.results = {
                    "dataset"        : proc_ds,
                    "change_results" : change_results,
                    "metrics_df"     : metrics_df,
                    "trends"         : trends,
                    "anomaly_reports": anomaly_reports,
                    "combined_scores": combined_scores,
                    "risk_report"    : risk_report,
                    "colormap"       : params["colormap"],
                }
                st.success("✅ Analysis complete!")

            except Exception:
                st.error("Pipeline error — see details below.")
                st.code(traceback.format_exc())
                st.stop()

    # ── Display results ────────────────────────────────────────────────
    if st.session_state.results is None:
        # Landing page
        st.markdown("""
---
### 👋 Welcome to SARDS

This system uses satellite imagery to detect and quantify urban changes
over time in **Dhaka, Bangladesh**.

**To get started:**
1. Click **🚀 Run Full Analysis** in the sidebar to use the demo dataset.
2. Or upload your own satellite images and run the analysis.

**The pipeline will:**
- 🔍 Detect spatial changes between image pairs
- 🌡️ Generate change-intensity heatmaps
- 📈 Analyse time-series trends
- 🔬 Flag anomalous years
- 🎯 Compute a composite risk score

---
#### Why Dhaka?
Dhaka is one of the world's fastest-growing megacities with:
- Rapid urban sprawl eating into wetlands and green cover
- Significant river encroachment (Buriganga, Turag rivers)
- High climate vulnerability (flooding, heat islands)
- Over **21 million** people in the metro area

Satellite monitoring helps quantify these changes objectively and at scale.
""")
        return

    r = st.session_state.results

    # Quick KPIs
    st.markdown("---")
    k1, k2, k3, k4, k5 = st.columns(5)
    years  = r["dataset"].years
    latest = r["change_results"][-1] if r["change_results"] else None

    k1.metric("📅 Years", f"{len(years)}", f"{years[0]}–{years[-1]}")
    k2.metric("🔍 Pairs Analysed", len(r["change_results"]))
    k3.metric("📊 Change (latest)",
              f"{latest.stats['change_pct']:.1f}%" if latest else "N/A")
    k4.metric("🎯 Risk Score",
              f"{r['risk_report'].overall_score:.3f}")

    anomalies = sum(1 for sc in r["combined_scores"].values() if sc > 0.5)
    k5.metric("⚠️ Anomalous Years", anomalies)

    st.markdown(f"**Risk Level:** "
                + risk_badge_html(r["risk_report"].risk_label),
                unsafe_allow_html=True)
    st.markdown("---")

    # ── Tabs ───────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🗂️ Dataset Overview",
        "🔍 Change Detection",
        "🌡️ Heatmaps",
        "📈 Time-Series",
        "🎯 Risk Score",
    ])

    with tab1:
        render_overview_tab(r["dataset"])

    with tab2:
        render_change_tab(r["dataset"], r["change_results"], cfg)

    with tab3:
        render_heatmap_tab(r["change_results"], r["dataset"], r["colormap"])

    with tab4:
        render_timeseries_tab(r["metrics_df"], r["trends"], r["change_results"])
        st.markdown("---")
        st.markdown("#### 🔬 Anomaly Flags")
        render_anomaly_tab(r["metrics_df"], r["anomaly_reports"],
                           r["combined_scores"])

    with tab5:
        render_risk_tab(r["risk_report"])


if __name__ == "__main__":
    main()
