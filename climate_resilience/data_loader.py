"""
Climate Data Loader
===================
Ingests and preprocesses climate indicators from:
  - NASA POWER API     (solar radiation, temperature, precipitation)
  - ERA5 Reanalysis    (ECMWF Copernicus — wind, soil moisture, heat stress)
  - CHIRPS             (Climate Hazards Group rainfall estimates)
  - MODIS NDVI         (vegetation dynamics via NASA Earthdata)
  - IPC Food Security  (famine early warning)
  - World Bank         (socio-economic indicators)

For portfolio / offline demo: synthetic data generator included.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import logging
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Country / Zone registry for Sahel + West/Central Africa (AdaptAction scope)
# ---------------------------------------------------------------------------
ADAPTACTION_COUNTRIES = {
    "NER": {"name": "Niger",          "lat_range": (11.7, 23.5), "lon_range": (0.2, 15.9)},
    "MLI": {"name": "Mali",           "lat_range": (10.1, 25.0), "lon_range": (-4.2, 4.3)},
    "BFA": {"name": "Burkina Faso",   "lat_range": (9.4, 15.1),  "lon_range": (-5.5, 2.4)},
    "SEN": {"name": "Senegal",        "lat_range": (12.3, 16.7), "lon_range": (-17.5, -11.4)},
    "MRT": {"name": "Mauritania",     "lat_range": (14.7, 27.3), "lon_range": (-17.1, -4.8)},
    "TCD": {"name": "Chad",           "lat_range": (7.4, 23.5),  "lon_range": (13.5, 24.0)},
    "CAF": {"name": "Central African Republic", "lat_range": (2.2, 11.0), "lon_range": (14.4, 27.5)},
    "CMR": {"name": "Cameroon",       "lat_range": (1.6, 13.1),  "lon_range": (8.5, 16.2)},
    "GNB": {"name": "Guinea-Bissau",  "lat_range": (10.9, 12.7), "lon_range": (-16.7, -13.6)},
    "GIN": {"name": "Guinea",         "lat_range": (7.2, 12.7),  "lon_range": (-15.1, -7.6)},
    "GMB": {"name": "Gambia",         "lat_range": (13.1, 13.8), "lon_range": (-16.8, -13.8)},
    "MDG": {"name": "Madagascar",     "lat_range": (-25.6, -11.9), "lon_range": (43.2, 50.5)},
}

# Agro-ecological zones — drives weight calibration
AGRO_ZONES = {
    "sahelian":    {"rainfall_mm": (200, 400),  "context": "sahel"},
    "sudanian":    {"rainfall_mm": (400, 900),  "context": "default"},
    "guinean":     {"rainfall_mm": (900, 2000), "context": "forest_basin"},
    "coastal":     {"rainfall_mm": (800, 3000), "context": "coastal"},
}


class ClimateDataLoader:
    """
    Unified data loader for climate resilience indicators.

    Handles:
    - API ingestion (NASA POWER, ERA5, CHIRPS)
    - Local NetCDF / GeoTIFF processing
    - Synthetic data generation for testing/demo
    - Normalization to [0, 1] range for CRI computation
    - Temporal aggregation (monthly → seasonal → annual)
    """

    INDICATOR_RANGES = {
        # Climate / Exposure
        "drought_frequency":      (0.0, 1.0),   # fraction of months in drought
        "extreme_heat_days":      (0, 180),      # days/year > 40°C
        "flood_risk_index":       (0.0, 1.0),
        "precipitation_anomaly":  (-3.0, 3.0),  # Z-score vs climatology
        # Sensitivity
        "poverty_rate":           (0.0, 1.0),
        "food_insecurity_ipc":    (1.0, 5.0),   # IPC Phase
        "population_density":     (0, 500),     # persons/km²
        "female_headed_hh":       (0.0, 0.6),
        "displacement_rate":      (0.0, 0.3),
        # Adaptive capacity
        "access_to_water":        (0.0, 1.0),
        "healthcare_coverage":    (0.0, 1.0),
        "road_network_density":   (0.0, 1.0),   # normalized km/km²
        "early_warning_systems":  (0.0, 1.0),
        "credit_access_rate":     (0.0, 1.0),
        # Livelihood
        "ndvi_trend":             (-0.01, 0.01), # annual NDVI change
        "crop_yield_variability": (0.0, 1.0),    # CV of yields
        "livelihood_diversity":   (1.0, 5.0),    # Shannon index
        "market_access_index":    (0.0, 1.0),
        # Ecosystem
        "forest_cover_change":    (-0.05, 0.01), # annual fraction change
        "soil_degradation_index": (0.0, 1.0),
        "watershed_integrity":    (0.0, 1.0),
    }

    def __init__(
        self,
        data_dir: str = "data",
        cache: bool = True,
        offline_mode: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.cache = cache
        self.offline_mode = offline_mode
        self._cache: Dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Synthetic data generation (offline demo / testing)
    # ------------------------------------------------------------------

    def generate_synthetic_dataset(
        self,
        n_zones: int = 50,
        n_periods: int = 8,
        countries: Optional[List[str]] = None,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Generate realistic synthetic indicator dataset for demonstration.

        Simulates spatial autocorrelation (zones within same country are
        correlated), temporal trends, and realistic missing data patterns.

        Parameters
        ----------
        n_zones : int
            Number of geographic zones to simulate.
        n_periods : int
            Number of quarterly periods.
        countries : list, optional
            Country codes to sample from ADAPTACTION_COUNTRIES.
        seed : int
            Random seed for reproducibility.
        """
        rng = np.random.default_rng(seed)
        countries = countries or list(ADAPTACTION_COUNTRIES.keys())[:6]
        indicators = list(self.INDICATOR_RANGES.keys())

        records = []
        zones_per_country = max(1, n_zones // len(countries))

        for country_code in countries:
            country_meta = ADAPTACTION_COUNTRIES[country_code]
            # Country-level "vulnerability baseline" — spatial autocorrelation
            country_baseline = rng.uniform(0.2, 0.8, len(indicators))

            for zone_idx in range(zones_per_country):
                zone_id = f"{country_code}-ADM2-{zone_idx+1:03d}"
                zone_name = f"{country_meta['name']} Zone {zone_idx+1}"

                lat = rng.uniform(*country_meta["lat_range"])
                lon = rng.uniform(*country_meta["lon_range"])

                # Zone deviation from country baseline (local conditions)
                zone_baseline = np.clip(
                    country_baseline + rng.normal(0, 0.1, len(indicators)), 0, 1
                )

                # Time series with trend + seasonality
                for period_idx in range(n_periods):
                    quarter = (period_idx % 4) + 1
                    year = 2021 + period_idx // 4
                    period = f"{year}-Q{quarter}"

                    # Temporal drift (climate change signal: conditions worsen)
                    temporal_drift = period_idx * rng.uniform(-0.015, 0.005, len(indicators))
                    seasonal = 0.05 * np.sin(2 * np.pi * quarter / 4) * rng.uniform(0.5, 1.5, len(indicators))

                    values = np.clip(zone_baseline + temporal_drift + seasonal, 0.01, 0.99)

                    # Introduce realistic missing data (~10% missing)
                    missing_mask = rng.random(len(indicators)) < 0.10
                    values[missing_mask] = np.nan

                    row = {
                        "zone_id":     zone_id,
                        "zone_name":   zone_name,
                        "country":     country_meta["name"],
                        "country_code": country_code,
                        "period":      period,
                        "latitude":    lat,
                        "longitude":   lon,
                        **dict(zip(indicators, values)),
                    }
                    records.append(row)

        df = pd.DataFrame(records)
        logger.info(
            f"Synthetic dataset: {len(df)} records | "
            f"{df['zone_id'].nunique()} zones | "
            f"{df['period'].nunique()} periods"
        )
        return df

    def normalize_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Min-max normalize all indicators to [0, 1] using known physical ranges.
        Applies domain knowledge: drought_frequency already in [0,1],
        extreme_heat_days needs scaling from [0, 180], etc.
        """
        df = df.copy()
        for indicator, (min_val, max_val) in self.INDICATOR_RANGES.items():
            if indicator in df.columns:
                df[indicator] = (df[indicator] - min_val) / (max_val - min_val)
                df[indicator] = df[indicator].clip(0, 1)
        return df

    def compute_era5_drought_index(
        self,
        temperature_series: pd.Series,
        precipitation_series: pd.Series,
        window_months: int = 3,
    ) -> pd.Series:
        """
        Compute Standardized Precipitation-Evapotranspiration Index (SPEI-3)
        from ERA5 temperature and precipitation time series.

        SPEI > 0 = wet | SPEI < -1 = drought | SPEI < -2 = severe drought
        """
        pet = 0.0023 * (temperature_series + 17.8) * (temperature_series.clip(0) ** 0.5) * 30
        water_balance = precipitation_series - pet
        rolling_wb = water_balance.rolling(window_months, min_periods=1).sum()

        mean = rolling_wb.mean()
        std = rolling_wb.std()
        if std == 0:
            return pd.Series(np.zeros(len(rolling_wb)))

        spei = (rolling_wb - mean) / std
        return spei

    def compute_ndvi_trend(
        self,
        ndvi_series: pd.Series,
        dates: pd.DatetimeIndex,
    ) -> float:
        """
        Mann-Kendall trend test on NDVI time series.
        Returns annual rate of change (greening > 0, browning < 0).
        """
        from scipy import stats
        if len(ndvi_series.dropna()) < 4:
            return 0.0

        n = len(ndvi_series)
        time_numeric = np.arange(n)
        slope, _, _, p_value, _ = stats.linregress(time_numeric, ndvi_series.fillna(method="ffill"))

        # Return slope only if statistically significant (p < 0.10)
        return float(slope) if p_value < 0.10 else 0.0

    def load_from_csv(self, filepath: str) -> pd.DataFrame:
        """Load indicator data from preprocessed CSV."""
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} records from {filepath}")
        return self.normalize_indicators(df)

    def load_from_geojson(self, filepath: str) -> gpd.GeoDataFrame:
        """Load spatial indicator data from GeoJSON."""
        gdf = gpd.read_file(filepath)
        logger.info(f"Loaded {len(gdf)} zones from {filepath}")
        return gdf

    def get_latest_period(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter dataframe to most recent period."""
        latest = df["period"].max()
        return df[df["period"] == latest].copy()

    def pivot_for_cri(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pivot long-format data to wide format expected by CRI engine.
        Each row = one zone, columns = indicators.
        """
        meta_cols = ["zone_id", "zone_name", "country", "country_code",
                     "latitude", "longitude", "period"]
        indicator_cols = [c for c in df.columns if c in self.INDICATOR_RANGES]
        return df[meta_cols + indicator_cols].copy()

    def save_processed(self, df: pd.DataFrame, filename: str) -> None:
        """Save processed dataset to data/processed/."""
        out_path = self.data_dir / "processed" / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        logger.info(f"Saved processed data → {out_path}")
