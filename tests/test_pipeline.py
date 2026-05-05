"""
test_pipeline.py — SARDS Unit & Integration Tests
===================================================
Tests for all core SARDS modules using pytest.

Run:
    pytest tests/ -v --tb=short

Author : SARDS Team
Version: 1.0.0
"""

import sys
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import cv2

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config, ensure_dir, array_to_uint8
from src.image_loader import (
    ImageLoader, SatelliteImage, ImageDataset
)
from src.preprocessing import Preprocessor
from src.change_detection import ChangeDetector, ChangeResult
from src.heatmap_generator import HeatmapGenerator
from src.time_series_analysis import TimeSeriesAnalyzer
from src.anomaly_detection import AnomalyDetector
from src.risk_scoring import RiskScorer


# ── Fixtures ──────────────────────────────────────────────────────────────────

CONFIG_PATH = str(Path(__file__).parent.parent / "config.yaml")


def make_test_image(year: int, seed: int = 42,
                    h: int = 64, w: int = 64) -> SatelliteImage:
    """Create a small synthetic SatelliteImage for testing."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(50, 220, (h, w, 3), dtype=np.uint8)
    return SatelliteImage(year=year, path=f"/tmp/{year}.jpg", array=arr)


def make_test_dataset(years=(2018, 2020, 2022, 2024)) -> ImageDataset:
    """Create a small synthetic ImageDataset for testing."""
    ds = ImageDataset(location="Test City")
    for i, yr in enumerate(years):
        ds.images[yr] = make_test_image(yr, seed=i * 13)
    return ds


@pytest.fixture(scope="module")
def tiny_dataset():
    return make_test_dataset()


@pytest.fixture(scope="module")
def preprocessor():
    return Preprocessor(CONFIG_PATH)


@pytest.fixture(scope="module")
def proc_dataset(tiny_dataset, preprocessor):
    return preprocessor.process_dataset(tiny_dataset)


@pytest.fixture(scope="module")
def change_results(proc_dataset):
    detector = ChangeDetector(CONFIG_PATH)
    return detector.detect_all_pairs(proc_dataset.as_list(), align=False)


@pytest.fixture(scope="module")
def metrics_df(proc_dataset, change_results):
    ts = TimeSeriesAnalyzer(CONFIG_PATH)
    df = ts.extract_metrics(proc_dataset)
    df = ts.add_change_metrics(df, change_results)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Tests: utils
# ══════════════════════════════════════════════════════════════════════════════

class TestUtils:

    def test_load_config_returns_dict(self):
        cfg = load_config(CONFIG_PATH)
        assert isinstance(cfg, dict)
        assert "preprocessing" in cfg
        assert "change_detection" in cfg
        assert "risk_scoring" in cfg

    def test_ensure_dir_creates_path(self, tmp_path):
        new_dir = tmp_path / "a" / "b" / "c"
        result  = ensure_dir(new_dir)
        assert result.exists()
        assert result.is_dir()

    def test_array_to_uint8_normalises(self):
        arr    = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
        result = array_to_uint8(arr)
        assert result.dtype  == np.uint8
        assert result.min()  == 0
        assert result.max()  == 255

    def test_array_to_uint8_constant_returns_zeros(self):
        arr = np.ones((4, 4), dtype=np.float32) * 0.5
        result = array_to_uint8(arr)
        assert (result == 0).all()


# ══════════════════════════════════════════════════════════════════════════════
# Tests: SatelliteImage / ImageDataset
# ══════════════════════════════════════════════════════════════════════════════

class TestImageDataStructures:

    def test_satellite_image_shape(self):
        img = make_test_image(2020)
        assert img.shape == (64, 64, 3)
        assert img.height == 64
        assert img.width  == 64
        assert img.year   == 2020

    def test_satellite_image_repr(self):
        img = make_test_image(2020)
        assert "2020" in repr(img)

    def test_dataset_years_sorted(self):
        ds = make_test_dataset((2022, 2018, 2020))
        assert ds.years == [2018, 2020, 2022]

    def test_dataset_pairs(self):
        ds    = make_test_dataset((2018, 2020, 2022))
        pairs = ds.pairs()
        assert len(pairs) == 2
        assert pairs[0][0].year == 2018
        assert pairs[0][1].year == 2020

    def test_dataset_get(self):
        ds  = make_test_dataset()
        img = ds.get(2018)
        assert img is not None
        assert img.year == 2018

    def test_dataset_get_missing_returns_none(self):
        ds = make_test_dataset()
        assert ds.get(1900) is None

    def test_dataset_as_list_sorted(self):
        ds   = make_test_dataset((2022, 2018, 2020, 2024))
        imgs = ds.as_list()
        assert [i.year for i in imgs] == [2018, 2020, 2022, 2024]


# ══════════════════════════════════════════════════════════════════════════════
# Tests: Preprocessor
# ══════════════════════════════════════════════════════════════════════════════

class TestPreprocessor:

    def test_process_image_output_shape(self, preprocessor):
        img    = make_test_image(2020, h=128, w=128)
        result = preprocessor.process_image(img)
        # Should be resized to config target_size
        cfg  = load_config(CONFIG_PATH)
        tw, th = cfg["preprocessing"]["target_size"]
        assert result.array.shape == (th, tw, 3)

    def test_process_image_preserves_year(self, preprocessor):
        img    = make_test_image(2020)
        result = preprocessor.process_image(img)
        assert result.year == 2020

    def test_process_image_dtype_uint8(self, preprocessor):
        img    = make_test_image(2020)
        result = preprocessor.process_image(img)
        assert result.array.dtype == np.uint8

    def test_process_dataset_all_years(self, preprocessor, tiny_dataset):
        proc = preprocessor.process_dataset(tiny_dataset)
        assert proc.years == tiny_dataset.years

    def test_to_float_range(self, preprocessor):
        arr = np.array([[[0, 128, 255]]], dtype=np.uint8)
        f   = preprocessor.to_float(arr)
        assert f.min() >= 0.0
        assert f.max() <= 1.0

    def test_to_grayscale_shape(self, preprocessor):
        arr  = np.ones((32, 32, 3), dtype=np.uint8) * 100
        gray = preprocessor.to_grayscale(arr)
        assert gray.ndim == 2
        assert gray.shape == (32, 32)


# ══════════════════════════════════════════════════════════════════════════════
# Tests: ChangeDetector
# ══════════════════════════════════════════════════════════════════════════════

class TestChangeDetector:

    def test_detect_returns_change_result(self, proc_dataset):
        detector = ChangeDetector(CONFIG_PATH)
        imgs     = proc_dataset.as_list()
        result   = detector.detect(imgs[0], imgs[1], align=False)
        assert isinstance(result, ChangeResult)

    def test_change_pct_in_range(self, proc_dataset):
        detector = ChangeDetector(CONFIG_PATH)
        imgs     = proc_dataset.as_list()
        result   = detector.detect(imgs[0], imgs[1], align=False)
        assert 0.0 <= result.change_percentage <= 100.0

    def test_identical_images_low_change(self, proc_dataset):
        detector = ChangeDetector(CONFIG_PATH)
        img      = proc_dataset.as_list()[0]
        result   = detector.detect(img, img, align=False)
        assert result.change_percentage < 1.0

    def test_different_images_nonzero_change(self, proc_dataset):
        detector = ChangeDetector(CONFIG_PATH)
        imgs     = proc_dataset.as_list()
        result   = detector.detect(imgs[0], imgs[-1], align=False)
        assert result.change_percentage >= 0.0

    def test_detect_all_pairs_count(self, proc_dataset):
        detector = ChangeDetector(CONFIG_PATH)
        results  = detector.detect_all_pairs(proc_dataset.as_list(), align=False)
        assert len(results) == proc_dataset.count - 1

    def test_overlay_shape_matches_input(self, proc_dataset):
        detector = ChangeDetector(CONFIG_PATH)
        imgs     = proc_dataset.as_list()
        result   = detector.detect(imgs[0], imgs[1], align=False)
        assert result.overlay.shape == imgs[0].array.shape

    def test_diff_map_non_negative(self, proc_dataset):
        detector = ChangeDetector(CONFIG_PATH)
        imgs     = proc_dataset.as_list()
        result   = detector.detect(imgs[0], imgs[1], align=False)
        assert result.diff_map.min() >= 0.0

    def test_stats_dict_has_required_keys(self, proc_dataset):
        detector  = ChangeDetector(CONFIG_PATH)
        imgs      = proc_dataset.as_list()
        result    = detector.detect(imgs[0], imgs[1], align=False)
        required  = {"change_pct", "changed_pixels", "total_pixels",
                     "contour_count", "mean_diff", "max_diff"}
        assert required.issubset(result.stats.keys())

    def test_ssim_method_runs(self, proc_dataset):
        detector        = ChangeDetector(CONFIG_PATH)
        detector.method = "ssim"
        imgs            = proc_dataset.as_list()
        result          = detector.detect(imgs[0], imgs[1], align=False)
        assert isinstance(result, ChangeResult)

    def test_optical_flow_method_runs(self, proc_dataset):
        detector        = ChangeDetector(CONFIG_PATH)
        detector.method = "optical_flow"
        imgs            = proc_dataset.as_list()
        result          = detector.detect(imgs[0], imgs[1], align=False)
        assert isinstance(result, ChangeResult)


# ══════════════════════════════════════════════════════════════════════════════
# Tests: HeatmapGenerator
# ══════════════════════════════════════════════════════════════════════════════

class TestHeatmapGenerator:

    def test_generate_change_heatmap_shape(self, change_results, proc_dataset):
        hgen   = HeatmapGenerator(CONFIG_PATH)
        result = change_results[0]
        hm     = hgen.generate_change_heatmap(result, save=False)
        assert hm.ndim    == 3
        assert hm.shape[2] == 3
        assert hm.dtype   == np.uint8

    def test_heatmap_values_in_range(self, change_results):
        hgen = HeatmapGenerator(CONFIG_PATH)
        hm   = hgen.generate_change_heatmap(change_results[0], save=False)
        assert hm.min() >= 0
        assert hm.max() <= 255

    def test_cumulative_heatmap_shape(self, change_results):
        hgen = HeatmapGenerator(CONFIG_PATH)
        hm   = hgen.generate_cumulative_heatmap(change_results, save=False)
        assert hm.ndim    == 3
        assert hm.dtype   == np.uint8

    def test_overlay_blending(self):
        base    = np.ones((32, 32, 3), dtype=np.uint8) * 100
        heatmap = np.ones((32, 32, 3), dtype=np.uint8) * 200
        result  = HeatmapGenerator._overlay(base, heatmap, alpha=0.5)
        assert result.shape == (32, 32, 3)
        assert 100 <= result.mean() <= 200


# ══════════════════════════════════════════════════════════════════════════════
# Tests: TimeSeriesAnalyzer
# ══════════════════════════════════════════════════════════════════════════════

class TestTimeSeriesAnalyzer:

    def test_extract_metrics_returns_dataframe(self, proc_dataset):
        import pandas as pd
        ts = TimeSeriesAnalyzer(CONFIG_PATH)
        df = ts.extract_metrics(proc_dataset)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == proc_dataset.count

    def test_metrics_index_is_years(self, proc_dataset):
        ts  = TimeSeriesAnalyzer(CONFIG_PATH)
        df  = ts.extract_metrics(proc_dataset)
        assert list(df.index) == proc_dataset.years

    def test_mean_intensity_positive(self, proc_dataset):
        ts  = TimeSeriesAnalyzer(CONFIG_PATH)
        df  = ts.extract_metrics(proc_dataset)
        if "mean_intensity" in df.columns:
            assert (df["mean_intensity"] > 0).all()

    def test_urban_index_in_range(self, proc_dataset):
        ts  = TimeSeriesAnalyzer(CONFIG_PATH)
        df  = ts.extract_metrics(proc_dataset)
        if "urban_index" in df.columns:
            assert (df["urban_index"] >= 0).all()
            assert (df["urban_index"] <= 1).all()

    def test_compute_trends_returns_dict(self, metrics_df):
        ts     = TimeSeriesAnalyzer(CONFIG_PATH)
        trends = ts.compute_trends(metrics_df)
        assert isinstance(trends, dict)

    def test_trend_has_required_keys(self, metrics_df):
        ts     = TimeSeriesAnalyzer(CONFIG_PATH)
        trends = ts.compute_trends(metrics_df)
        for t in trends.values():
            assert "slope"     in t
            assert "r2"        in t
            assert "p_value"   in t
            assert "trend"     in t

    def test_trend_direction_valid(self, metrics_df):
        ts     = TimeSeriesAnalyzer(CONFIG_PATH)
        trends = ts.compute_trends(metrics_df)
        valid  = {"increasing", "decreasing", "stable"}
        for t in trends.values():
            assert t["trend"] in valid

    def test_add_change_metrics_adds_columns(self, proc_dataset, change_results):
        ts  = TimeSeriesAnalyzer(CONFIG_PATH)
        df  = ts.extract_metrics(proc_dataset)
        df2 = ts.add_change_metrics(df, change_results)
        assert "change_pct"    in df2.columns
        assert "contour_count" in df2.columns


# ══════════════════════════════════════════════════════════════════════════════
# Tests: AnomalyDetector
# ══════════════════════════════════════════════════════════════════════════════

class TestAnomalyDetector:

    def test_detect_returns_anomaly_report(self, metrics_df):
        ad     = AnomalyDetector(CONFIG_PATH)
        report = ad.detect(metrics_df, "mean_intensity", method="zscore")
        assert hasattr(report, "anomaly_years")
        assert hasattr(report, "scores")

    def test_scores_in_range(self, metrics_df):
        ad     = AnomalyDetector(CONFIG_PATH)
        report = ad.detect(metrics_df, "mean_intensity", method="zscore")
        for sc in report.scores.values():
            assert 0.0 <= sc <= 1.0

    def test_iqr_method_runs(self, metrics_df):
        ad     = AnomalyDetector(CONFIG_PATH)
        report = ad.detect(metrics_df, "mean_intensity", method="iqr")
        assert report is not None

    def test_isolation_forest_method_runs(self, metrics_df):
        ad     = AnomalyDetector(CONFIG_PATH)
        report = ad.detect(metrics_df, "mean_intensity",
                           method="isolation_forest")
        assert report is not None

    def test_detect_all_returns_dict(self, metrics_df):
        ad      = AnomalyDetector(CONFIG_PATH)
        reports = ad.detect_all(metrics_df)
        assert isinstance(reports, dict)
        assert len(reports) > 0

    def test_combined_score_in_range(self, metrics_df):
        ad      = AnomalyDetector(CONFIG_PATH)
        reports = ad.detect_all(metrics_df)
        scores  = ad.combined_anomaly_score(
            reports, metrics_df.index.tolist()
        )
        for sc in scores.values():
            assert 0.0 <= sc <= 1.0

    def test_missing_metric_handled_gracefully(self, metrics_df):
        ad     = AnomalyDetector(CONFIG_PATH)
        report = ad.detect(metrics_df, "nonexistent_column")
        assert report.anomaly_years == []


# ══════════════════════════════════════════════════════════════════════════════
# Tests: RiskScorer
# ══════════════════════════════════════════════════════════════════════════════

class TestRiskScorer:

    def test_compute_returns_risk_report(self, metrics_df, change_results):
        ts      = TimeSeriesAnalyzer(CONFIG_PATH)
        trends  = ts.compute_trends(metrics_df)
        ad      = AnomalyDetector(CONFIG_PATH)
        reports = ad.detect_all(metrics_df)
        scores  = ad.combined_anomaly_score(
            reports, metrics_df.index.tolist()
        )
        scorer = RiskScorer(CONFIG_PATH)
        report = scorer.compute(
            metrics_df     = metrics_df,
            trends         = trends,
            anomaly_scores = scores,
            change_results = change_results,
        )
        assert report is not None
        assert 0.0 <= report.overall_score <= 1.0

    def test_risk_label_non_empty(self, metrics_df, change_results):
        ts      = TimeSeriesAnalyzer(CONFIG_PATH)
        trends  = ts.compute_trends(metrics_df)
        ad      = AnomalyDetector(CONFIG_PATH)
        reports = ad.detect_all(metrics_df)
        scores  = ad.combined_anomaly_score(
            reports, metrics_df.index.tolist()
        )
        scorer = RiskScorer(CONFIG_PATH)
        report = scorer.compute(
            metrics_df=metrics_df, trends=trends,
            anomaly_scores=scores, change_results=change_results
        )
        assert len(report.risk_label) > 0

    def test_classify_low(self):
        scorer = RiskScorer(CONFIG_PATH)
        assert "LOW" in scorer._classify(0.15)

    def test_classify_medium(self):
        scorer = RiskScorer(CONFIG_PATH)
        assert "MEDIUM" in scorer._classify(0.45)

    def test_classify_high(self):
        scorer = RiskScorer(CONFIG_PATH)
        assert "HIGH" in scorer._classify(0.70)

    def test_classify_critical(self):
        scorer = RiskScorer(CONFIG_PATH)
        assert "CRITICAL" in scorer._classify(0.90)

    def test_year_scores_all_years_present(self, metrics_df, change_results):
        ts      = TimeSeriesAnalyzer(CONFIG_PATH)
        trends  = ts.compute_trends(metrics_df)
        ad      = AnomalyDetector(CONFIG_PATH)
        reports = ad.detect_all(metrics_df)
        scores  = ad.combined_anomaly_score(
            reports, metrics_df.index.tolist()
        )
        scorer = RiskScorer(CONFIG_PATH)
        report = scorer.compute(
            metrics_df=metrics_df, trends=trends,
            anomaly_scores=scores, change_results=change_results
        )
        for yr in metrics_df.index:
            assert yr in report.year_scores

    def test_narrative_contains_location(self, metrics_df, change_results):
        ts      = TimeSeriesAnalyzer(CONFIG_PATH)
        trends  = ts.compute_trends(metrics_df)
        ad      = AnomalyDetector(CONFIG_PATH)
        reports = ad.detect_all(metrics_df)
        scores  = ad.combined_anomaly_score(
            reports, metrics_df.index.tolist()
        )
        scorer = RiskScorer(CONFIG_PATH)
        report = scorer.compute(
            metrics_df=metrics_df, trends=trends,
            anomaly_scores=scores, change_results=change_results
        )
        assert "Dhaka" in report.summary_text or len(report.summary_text) > 100

    def test_score_color_green_for_low(self):
        assert RiskScorer._score_color(0.1) == "#27ae60"

    def test_score_color_red_for_high(self):
        assert RiskScorer._score_color(0.7) == "#e74c3c"

    def test_score_color_purple_for_critical(self):
        assert RiskScorer._score_color(0.9) == "#8e44ad"


# ══════════════════════════════════════════════════════════════════════════════
# Integration Test: Full pipeline
# ══════════════════════════════════════════════════════════════════════════════

class TestFullPipeline:

    def test_end_to_end_no_crash(self, tmp_path):
        """
        Run the entire SARDS pipeline from raw images to risk report
        in a temporary directory to ensure no crashes.
        """
        # 1. Create synthetic dataset
        ds = make_test_dataset(years=(2020, 2021, 2022))

        # 2. Preprocess
        prep    = Preprocessor(CONFIG_PATH)
        proc_ds = prep.process_dataset(ds)

        # 3. Change detection
        detector       = ChangeDetector(CONFIG_PATH)
        change_results = detector.detect_all_pairs(
            proc_ds.as_list(), align=False
        )
        assert len(change_results) == 2

        # 4. Time-series
        ts         = TimeSeriesAnalyzer(CONFIG_PATH)
        metrics_df = ts.extract_metrics(proc_ds)
        metrics_df = ts.add_change_metrics(metrics_df, change_results)
        trends     = ts.compute_trends(metrics_df)

        # 5. Anomaly detection
        ad             = AnomalyDetector(CONFIG_PATH)
        anomaly_rpts   = ad.detect_all(metrics_df)
        anom_scores    = ad.combined_anomaly_score(
            anomaly_rpts, proc_ds.years
        )

        # 6. Risk scoring
        scorer      = RiskScorer(CONFIG_PATH)
        risk_report = scorer.compute(
            metrics_df     = metrics_df,
            trends         = trends,
            anomaly_scores = anom_scores,
            change_results = change_results,
        )

        assert 0.0 <= risk_report.overall_score <= 1.0
        assert risk_report.risk_label != ""
        assert len(risk_report.year_scores) == 3
        assert "RISK" in risk_report.risk_label
        assert len(risk_report.summary_text) > 200
