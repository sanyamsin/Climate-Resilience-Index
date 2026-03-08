"""
Test Suite — Climate Resilience Index
======================================
Unit + integration tests covering:
  - CRI computation correctness
  - Edge cases (missing data, boundary values)
  - Alert classification logic
  - Data loading and normalization
  - Batch processing consistency
"""

import pytest
import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from climate_resilience import ClimateResilienceIndex, ClimateDataLoader, DegradationAlertSystem
from climate_resilience.indices import PillarScore, ResilienceProfile, DEFAULT_WEIGHTS
from climate_resilience.alerts import AlertLevel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cri_engine():
    return ClimateResilienceIndex(context="sahel")


@pytest.fixture
def loader():
    return ClimateDataLoader(offline_mode=True)


@pytest.fixture
def alert_system():
    return DegradationAlertSystem()


@pytest.fixture
def sample_indicators():
    """Typical semi-arid zone indicator profile."""
    return pd.Series({
        "drought_frequency":      0.35,
        "extreme_heat_days":      0.40,
        "flood_risk_index":       0.15,
        "precipitation_anomaly":  0.55,
        "poverty_rate":           0.55,
        "food_insecurity_ipc":    0.60,
        "population_density":     0.20,
        "female_headed_hh":       0.35,
        "displacement_rate":      0.18,
        "access_to_water":        0.42,
        "healthcare_coverage":    0.38,
        "road_network_density":   0.30,
        "early_warning_systems":  0.45,
        "credit_access_rate":     0.25,
        "ndvi_trend":             0.45,
        "crop_yield_variability": 0.50,
        "livelihood_diversity":   0.55,
        "market_access_index":    0.40,
        "forest_cover_change":    0.30,
        "soil_degradation_index": 0.45,
        "watershed_integrity":    0.38,
    })


@pytest.fixture
def resilient_indicators():
    """High-resilience zone profile — low hazard, high capacity."""
    exposure_keys = ["drought_frequency", "extreme_heat_days", "flood_risk_index",
                     "precipitation_anomaly", "poverty_rate", "food_insecurity_ipc",
                     "population_density", "female_headed_hh", "displacement_rate"]
    capacity_keys = ["access_to_water", "healthcare_coverage", "road_network_density",
                     "early_warning_systems", "credit_access_rate", "ndvi_trend",
                     "crop_yield_variability", "livelihood_diversity", "market_access_index",
                     "forest_cover_change", "soil_degradation_index", "watershed_integrity"]
    data = {k: 0.15 for k in exposure_keys}   # low hazard/vulnerability = bon
    data.update({k: 0.85 for k in capacity_keys})  # high capacity = bon
    return pd.Series(data)


@pytest.fixture
def synthetic_dataset(loader):
    df = loader.generate_synthetic_dataset(n_zones=20, n_periods=4, seed=0)
    return loader.normalize_indicators(df)


# ---------------------------------------------------------------------------
# ClimateResilienceIndex — unit tests
# ---------------------------------------------------------------------------

class TestClimateResilienceIndex:

    def test_compute_returns_profile(self, cri_engine, sample_indicators):
        profile = cri_engine.compute(
            sample_indicators, zone_id="NER-TEST-001",
            zone_name="Test Zone", country="Niger", period="2024-Q1"
        )
        assert isinstance(profile, ResilienceProfile)
        assert 0 <= profile.cri_score <= 100

    def test_cri_score_range(self, cri_engine, sample_indicators):
        profile = cri_engine.compute(sample_indicators, zone_id="Z01")
        assert 0 <= profile.cri_score <= 100

    def test_resilient_zone_scores_high(self, cri_engine, resilient_indicators):
        profile = cri_engine.compute(resilient_indicators, zone_id="Z-RESILIENT")
        assert profile.cri_score >= 55, f"Expected high CRI, got {profile.cri_score}"

    def test_profile_has_all_pillars(self, cri_engine, sample_indicators):
        profile = cri_engine.compute(sample_indicators, zone_id="Z02")
        expected_pillars = {"exposure", "sensitivity", "adaptive_cap", "livelihood", "ecosystem"}
        assert set(profile.pillar_scores.keys()) == expected_pillars

    def test_grade_classification(self, cri_engine, sample_indicators):
        profile = cri_engine.compute(sample_indicators, zone_id="Z03")
        assert profile.grade in [
            "HIGH RESILIENCE", "MODERATE RESILIENCE",
            "LOW RESILIENCE", "CRITICAL FRAGILITY"
        ]

    def test_missing_data_handled(self, cri_engine):
        partial = pd.Series({
            "drought_frequency": 0.4,
            "poverty_rate": 0.6,
            "access_to_water": 0.5,
            "ndvi_trend": 0.4,
            "watershed_integrity": 0.3,
        })
        profile = cri_engine.compute(partial, zone_id="Z-PARTIAL")
        assert pd.notna(profile.cri_score)
        assert 0 <= profile.cri_score <= 100

    def test_trend_computed_on_second_call(self, cri_engine, sample_indicators):
        cri_engine.compute(sample_indicators, zone_id="Z-TREND", period="2023-Q4")
        modified = sample_indicators.copy()
        modified["drought_frequency"] = 0.8
        profile2 = cri_engine.compute(modified, zone_id="Z-TREND", period="2024-Q1")
        assert profile2.trend is not None

    def test_invalid_weights_raise_error(self):
        with pytest.raises(ValueError, match="Missing pillar weights"):
            ClimateResilienceIndex(weights={"exposure": 0.5, "sensitivity": 0.5})

    def test_weights_must_sum_to_one(self):
        weights = {k: 0.15 for k in DEFAULT_WEIGHTS}  # Sum = 0.75
        with pytest.raises(ValueError, match="sum to 1.0"):
            ClimateResilienceIndex(weights=weights)

    def test_uncertainty_bounds(self, cri_engine, sample_indicators):
        bounds = cri_engine.uncertainty_bounds(sample_indicators)
        assert "mean" in bounds and "std" in bounds
        assert bounds["ci_lower_95"] <= bounds["mean"] <= bounds["ci_upper_95"]
        assert bounds["std"] >= 0

    def test_batch_compute(self, cri_engine, synthetic_dataset):
        latest = synthetic_dataset[synthetic_dataset["period"] == synthetic_dataset["period"].max()]
        results = cri_engine.compute_batch(latest)
        assert len(results) == len(latest)
        assert "cri_score" in results.columns
        assert results["cri_score"].between(0, 100).all() or results["cri_score"].isna().any()

    def test_trajectory_stored(self, cri_engine, sample_indicators):
        zone_id = "Z-TRAJ"
        for period in ["2023-Q1", "2023-Q2", "2023-Q3"]:
            cri_engine.compute(sample_indicators, zone_id=zone_id, period=period)
        traj = cri_engine.get_trajectory(zone_id)
        assert len(traj) == 3
        assert "period" in traj.columns


# ---------------------------------------------------------------------------
# ClimateDataLoader — unit tests
# ---------------------------------------------------------------------------

class TestClimateDataLoader:

    def test_synthetic_dataset_shape(self, loader):
        df = loader.generate_synthetic_dataset(n_zones=10, n_periods=4)
        assert len(df) > 0
        assert df["zone_id"].nunique() >= 1
        assert df["period"].nunique() == 4
        assert "zone_id" in df.columns
        assert "period" in df.columns

    def test_synthetic_dataset_columns(self, loader):
        df = loader.generate_synthetic_dataset(n_zones=5, n_periods=2)
        expected_indicators = ["drought_frequency", "poverty_rate", "ndvi_trend"]
        for ind in expected_indicators:
            assert ind in df.columns, f"Missing indicator: {ind}"

    def test_normalize_stays_in_range(self, loader):
        df = loader.generate_synthetic_dataset(n_zones=20, n_periods=2)
        normalized = loader.normalize_indicators(df)
        for col in ["drought_frequency", "poverty_rate", "access_to_water"]:
            if col in normalized.columns:
                valid = normalized[col].dropna()
                assert valid.between(0, 1).all(), f"{col} outside [0,1]"

    def test_get_latest_period(self, loader):
        df = loader.generate_synthetic_dataset(n_zones=10, n_periods=4)
        latest = loader.get_latest_period(df)
        assert latest["period"].nunique() == 1
        assert latest["period"].iloc[0] == df["period"].max()

    def test_reproducibility_with_seed(self, loader):
        df1 = loader.generate_synthetic_dataset(n_zones=20, n_periods=4, seed=42)
        df2 = loader.generate_synthetic_dataset(n_zones=20, n_periods=4, seed=42)
        pd.testing.assert_frame_equal(
            df1[["zone_id", "drought_frequency"]].reset_index(drop=True),
            df2[["zone_id", "drought_frequency"]].reset_index(drop=True),
        )


# ---------------------------------------------------------------------------
# DegradationAlertSystem — unit tests
# ---------------------------------------------------------------------------

class TestDegradationAlertSystem:

    def _make_zone_df(self, scores: list, zone_id: str = "Z-TEST") -> pd.DataFrame:
        """Helper to create a zone's score time series."""
        return pd.DataFrame({
            "zone_id":   [zone_id] * len(scores),
            "zone_name": [f"Test Zone"] * len(scores),
            "country":   ["Niger"] * len(scores),
            "period":    [f"2023-Q{i+1}" for i in range(len(scores))],
            "cri_score": scores,
        })

    def test_critical_alert_triggered(self, alert_system):
        df = self._make_zone_df([60.0, 50.0, 20.0])  # CRI drops below 25
        alerts = alert_system.evaluate(df)
        assert any(a.level == AlertLevel.CRITICAL for a in alerts)

    def test_degrading_alert_triggered(self, alert_system):
        df = self._make_zone_df([65.0, 58.0])  # 7 point drop
        alerts = alert_system.evaluate(df)
        assert any(a.level == AlertLevel.DEGRADING for a in alerts)

    def test_improving_alert_triggered(self, alert_system):
        df = self._make_zone_df([45.0, 52.0])  # 7 point gain
        alerts = alert_system.evaluate(df)
        assert any(a.level == AlertLevel.IMPROVING for a in alerts)

    def test_watch_alert_for_low_score(self, alert_system):
        df = self._make_zone_df([35.0, 36.0])  # Stable but low
        alerts = alert_system.evaluate(df)
        assert any(a.level == AlertLevel.WATCH for a in alerts)

    def test_no_alert_for_stable_normal(self, alert_system):
        df = self._make_zone_df([70.0, 71.0])  # Stable and high
        alerts = alert_system.evaluate(df)
        # Should generate no alert or only NORMAL (which is filtered)
        assert all(a.level not in (AlertLevel.CRITICAL, AlertLevel.DEGRADING) for a in alerts)

    def test_get_critical_zones(self, alert_system):
        df = pd.concat([
            self._make_zone_df([18.0], "CRIT-ZONE"),
            self._make_zone_df([75.0], "SAFE-ZONE"),
        ])
        alerts = alert_system.evaluate(df)
        critical = alert_system.get_critical_zones(alerts)
        assert any(a.zone_id == "CRIT-ZONE" for a in critical)
        assert all(a.zone_id != "SAFE-ZONE" for a in critical)

    def test_statistics_output(self, alert_system):
        df = self._make_zone_df([20.0, 18.0])
        alerts = alert_system.evaluate(df)
        stats = alert_system.get_statistics(alerts)
        assert "total_alerts" in stats
        assert "by_level" in stats

    def test_summary_dataframe(self, alert_system):
        df = self._make_zone_df([22.0, 20.0])
        alerts = alert_system.evaluate(df)
        summary = alert_system.get_summary(alerts)
        assert isinstance(summary, pd.DataFrame)
        assert "zone_id" in summary.columns


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestIntegration:

    def test_full_pipeline(self):
        """End-to-end: data generation → normalization → CRI → alerts."""
        loader = ClimateDataLoader(offline_mode=True)
        cri_engine = ClimateResilienceIndex(context="sahel")
        alert_system = DegradationAlertSystem()

        raw_df = loader.generate_synthetic_dataset(n_zones=15, n_periods=4, seed=7)
        norm_df = loader.normalize_indicators(raw_df)

        latest = loader.get_latest_period(norm_df)
        results = cri_engine.compute_batch(latest)

        meta = latest[["zone_id", "zone_name", "country", "period"]].drop_duplicates("zone_id")
        full = results.merge(meta, on="zone_id", how="left")

        alerts = alert_system.evaluate(full)
        stats = alert_system.get_statistics(alerts)

        assert len(results) == len(latest)
        assert 0 <= results["cri_score"].mean() <= 100
        assert isinstance(stats, dict)

    def test_multi_period_trajectory(self):
        """CRI trajectories update correctly across periods."""
        loader = ClimateDataLoader(offline_mode=True)
        cri_engine = ClimateResilienceIndex()

        df = loader.generate_synthetic_dataset(n_zones=5, n_periods=4, seed=99)
        df = loader.normalize_indicators(df)

        zone_id = df["zone_id"].iloc[0]
        zone_data = df[df["zone_id"] == zone_id].sort_values("period")

        for _, row in zone_data.iterrows():
            cri_engine.compute(row, zone_id=zone_id, period=row["period"])

        traj = cri_engine.get_trajectory(zone_id)
        assert len(traj) == len(zone_data)
        assert traj["cri_score"].between(0, 100).all()
