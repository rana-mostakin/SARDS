"""
main.py — SARDS Full Pipeline Runner
=====================================
Executes the complete Satellite-Based Activity Risk Detection pipeline:

  Step 0 : Validate environment and data
  Step 1 : Load satellite image dataset
  Step 2 : Pre-process images
  Step 3 : Change detection (all consecutive pairs)
  Step 4 : Heatmap generation (per-pair + cumulative)
  Step 5 : Time-series metric extraction
  Step 6 : Trend analysis (OLS)
  Step 7 : Anomaly detection
  Step 8 : Risk scoring
  Step 9 : Save all results and reports

Usage:
    python main.py                    # Full pipeline
    python main.py --help             # Show options

Author : SARDS Team
Version: 1.0.0
"""

import sys
import argparse
import traceback
from pathlib import Path

# ── Ensure project root is on path ────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import (
    print_banner, print_section, print_success,
    print_warning, print_error, print_info,
    setup_logger, load_config, ensure_dir, timestamp
)
from src.image_loader     import ImageLoader
from src.preprocessing    import Preprocessor
from src.change_detection import ChangeDetector
from src.heatmap_generator import HeatmapGenerator
from src.time_series_analysis import TimeSeriesAnalyzer
from src.anomaly_detection import AnomalyDetector
from src.risk_scoring      import RiskScorer


# ── Global logger ─────────────────────────────────────────────────────────────
logger = setup_logger("SARDS.Main", log_file="results/sards.log")


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(config_path: str = "config.yaml", verbose: bool = True) -> dict:
    """
    Execute the full SARDS pipeline.

    Parameters
    ----------
    config_path : Path to YAML config.
    verbose     : Print step summaries to stdout.

    Returns
    -------
    dict  Collection of all pipeline outputs for inspection / testing.
    """
    outputs = {}

    # ── Banner ────────────────────────────────────────────────────────
    if verbose:
        print_banner()

    cfg = load_config(config_path)
    ensure_dir(cfg["data"]["results_dir"])

    # ══════════════════════════════════════════════════════════════════
    # STEP 1: Load dataset
    # ══════════════════════════════════════════════════════════════════
    if verbose:
        print_section("STEP 1 — Loading Satellite Image Dataset")

    loader  = ImageLoader(config_path)
    dataset = loader.load_dhaka_timeline()
    outputs["dataset"] = dataset

    if verbose:
        print_success(f"Loaded {dataset.count} images: {dataset.years}")

    # ══════════════════════════════════════════════════════════════════
    # STEP 2: Pre-processing
    # ══════════════════════════════════════════════════════════════════
    if verbose:
        print_section("STEP 2 — Pre-processing Images")

    preprocessor = Preprocessor(config_path)
    proc_dataset = preprocessor.process_dataset(dataset)
    outputs["proc_dataset"] = proc_dataset

    if verbose:
        print_success(f"Pre-processing complete. "
                      f"Target resolution: {preprocessor.target_w}×{preprocessor.target_h}")

    # ══════════════════════════════════════════════════════════════════
    # STEP 3: Change Detection
    # ══════════════════════════════════════════════════════════════════
    if verbose:
        print_section("STEP 3 — Change Detection (Consecutive Pairs)")

    detector = ChangeDetector(config_path)
    images   = proc_dataset.as_list()
    change_results = detector.detect_all_pairs(images, align=True)
    outputs["change_results"] = change_results

    if verbose:
        for r in change_results:
            print_info(f"  {r.year_before} → {r.year_after} : "
                       f"{r.stats['change_pct']:.2f}% changed, "
                       f"{r.stats['contour_count']} regions")
        print_success(f"Detected changes in {len(change_results)} pairs.")

    # ══════════════════════════════════════════════════════════════════
    # STEP 4: Heatmap Generation
    # ══════════════════════════════════════════════════════════════════
    if verbose:
        print_section("STEP 4 — Generating Heatmaps")

    heatgen = HeatmapGenerator(config_path)

    # Per-pair heatmaps
    heatmaps = []
    for res in change_results:
        base = proc_dataset.get(res.year_before)
        hm   = heatgen.generate_change_heatmap(res, base_img=base, save=True)
        heatmaps.append(hm)
    outputs["heatmaps"] = heatmaps

    # Cumulative heatmap
    if change_results:
        cumulative_hm = heatgen.generate_cumulative_heatmap(
            change_results,
            base_img=proc_dataset.get(proc_dataset.years[-1]),
            save=True
        )
        outputs["cumulative_heatmap"] = cumulative_hm

    # Comparison panel figure
    panel_fig = heatgen.generate_comparison_panel(proc_dataset, change_results,
                                                  save=True)
    outputs["comparison_panel"] = panel_fig

    # Timeline grid
    grid_fig = heatgen.generate_intensity_grid(proc_dataset, save=True)
    outputs["timeline_grid"] = grid_fig

    if verbose:
        print_success(f"Generated {len(heatmaps)} heatmaps + cumulative map.")

    # ══════════════════════════════════════════════════════════════════
    # STEP 5: Time-Series Metric Extraction
    # ══════════════════════════════════════════════════════════════════
    if verbose:
        print_section("STEP 5 — Extracting Time-Series Metrics")

    ts_analyzer  = TimeSeriesAnalyzer(config_path)
    metrics_df   = ts_analyzer.extract_metrics(proc_dataset)
    metrics_df   = ts_analyzer.add_change_metrics(metrics_df, change_results)
    outputs["metrics_df"] = metrics_df

    csv_path = ts_analyzer.save_metrics_csv(metrics_df)

    if verbose:
        print_info("\n" + metrics_df.to_string())
        print_success(f"Metrics saved to: {csv_path}")

    # ══════════════════════════════════════════════════════════════════
    # STEP 6: Trend Analysis
    # ══════════════════════════════════════════════════════════════════
    if verbose:
        print_section("STEP 6 — Trend Analysis (Linear Regression)")

    trends = ts_analyzer.compute_trends(metrics_df)
    outputs["trends"] = trends

    ts_fig      = ts_analyzer.plot_metrics(metrics_df, trends, save=True)
    change_fig  = ts_analyzer.plot_change_progression(change_results, save=True)
    outputs["ts_figure"]     = ts_fig
    outputs["change_figure"] = change_fig

    if verbose:
        for m, t in trends.items():
            arrow = "↑" if t["trend"] == "increasing" else (
                    "↓" if t["trend"] == "decreasing" else "→")
            print_info(f"  {m:<22} {arrow} slope={t['slope']:+.5f}  "
                       f"R²={t['r2']:.3f}  [{t['trend']}]")
        print_success("Trend plots saved.")

    # ══════════════════════════════════════════════════════════════════
    # STEP 7: Anomaly Detection
    # ══════════════════════════════════════════════════════════════════
    if verbose:
        print_section("STEP 7 — Anomaly Detection")

    anomaly_detector = AnomalyDetector(config_path)
    anomaly_reports  = anomaly_detector.detect_all(metrics_df)
    combined_scores  = anomaly_detector.combined_anomaly_score(
        anomaly_reports, proc_dataset.years
    )
    outputs["anomaly_reports"] = anomaly_reports
    outputs["anomaly_scores"]  = combined_scores

    anom_fig = anomaly_detector.plot_anomalies(metrics_df, anomaly_reports,
                                               save=True)
    outputs["anomaly_figure"] = anom_fig

    flagged = {yr: sc for yr, sc in combined_scores.items() if sc > 0.5}
    if verbose:
        if flagged:
            print_warning(f"  Anomalous years: {list(flagged.keys())}")
        else:
            print_success("  No significant anomalies detected.")

    # ══════════════════════════════════════════════════════════════════
    # STEP 8: Risk Scoring
    # ══════════════════════════════════════════════════════════════════
    if verbose:
        print_section("STEP 8 — Computing Risk Score")

    scorer = RiskScorer(config_path)
    cum_hm = outputs.get("cumulative_heatmap")

    risk_report = scorer.compute(
        metrics_df     = metrics_df,
        trends         = trends,
        anomaly_scores = combined_scores,
        change_results = change_results,
        heatmap_array  = cum_hm,
    )
    outputs["risk_report"] = risk_report

    risk_fig = scorer.plot_risk_dashboard(risk_report, save=True)
    outputs["risk_dashboard"] = risk_fig

    report_path = scorer.save_report(risk_report)
    outputs["report_path"] = report_path

    if verbose:
        print_info("\n" + "─" * 60)
        print_info(f"  Score  : {risk_report.overall_score:.4f}")
        print_info(f"  Label  : {risk_report.risk_label}")
        print_info("─" * 60)
        print_success(f"Text report saved: {report_path}")

    # ══════════════════════════════════════════════════════════════════
    # DONE
    # ══════════════════════════════════════════════════════════════════
    if verbose:
        print_section("PIPELINE COMPLETE")
        print_success("All results saved to:  results/")
        print_success("  heatmaps/  ← change heatmaps")
        print_success("  time_series/ ← metric plots + CSV")
        print_success("  reports/   ← risk dashboard + text report")
        print_info("\nLaunch the interactive dashboard:\n"
                   "  streamlit run app/streamlit_app.py\n")

    return outputs


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SARDS — Satellite-Based Activity Risk Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py --config config.yaml
  python main.py --quiet
        """
    )
    parser.add_argument(
        "--config", "-c", default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress verbose console output"
    )
    args = parser.parse_args()

    try:
        run_pipeline(config_path=args.config, verbose=not args.quiet)
    except FileNotFoundError as e:
        print_error(str(e))
        print_info("Tip: Run  python scripts/generate_demo_data.py  first.")
        sys.exit(1)
    except Exception:
        print_error("Pipeline failed with an unexpected error:")
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
