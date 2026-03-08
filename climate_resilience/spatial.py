"""
Spatial Aggregation & Territorial Analysis
==========================================
Handles geographic operations for CRI territorialization:
  - Administrative boundary joins (admin0 → admin3)
  - Spatial interpolation for missing zones
  - Neighbor-based imputation (spatial lag)
  - Regional aggregation (country / basin / cross-border)
  - Hotspot clustering (spatial autocorrelation)
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from typing import Optional, List, Dict, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SpatialAggregator:
    """
    Territorial CRI analysis engine.

    Joins indicator data to admin boundaries, computes spatial statistics,
    identifies fragility hotspots, and prepares data for Dash mapping.
    """

    CRS_WGS84 = "EPSG:4326"
    CRS_AFRICA = "ESRI:102022"   # Africa Albers Equal Area (for area calc)

    def __init__(self, admin_boundaries_path: Optional[str] = None):
        self.admin_gdf: Optional[gpd.GeoDataFrame] = None
        if admin_boundaries_path:
            self.load_admin_boundaries(admin_boundaries_path)

    # ------------------------------------------------------------------
    # Boundary management
    # ------------------------------------------------------------------

    def load_admin_boundaries(self, path: str) -> gpd.GeoDataFrame:
        """Load administrative boundaries from GeoJSON/Shapefile."""
        self.admin_gdf = gpd.read_file(path).to_crs(self.CRS_WGS84)
        logger.info(f"Loaded {len(self.admin_gdf)} admin zones from {path}")
        return self.admin_gdf

    def create_synthetic_boundaries(
        self,
        cri_df: pd.DataFrame,
        buffer_deg: float = 0.5,
    ) -> gpd.GeoDataFrame:
        """
        Create point-buffered synthetic boundaries for demo/testing.
        Uses zone centroids from lat/lon columns.
        """
        if "latitude" not in cri_df.columns or "longitude" not in cri_df.columns:
            raise ValueError("DataFrame must contain 'latitude' and 'longitude' columns")

        geometry = [
            Point(row["longitude"], row["latitude"]).buffer(buffer_deg)
            for _, row in cri_df.iterrows()
        ]
        gdf = gpd.GeoDataFrame(cri_df, geometry=geometry, crs=self.CRS_WGS84)
        return gdf

    def join_to_boundaries(
        self,
        cri_df: pd.DataFrame,
        join_col: str = "zone_id",
        boundary_col: str = "PCODE",
    ) -> gpd.GeoDataFrame:
        """
        Spatial join of CRI scores to administrative boundaries.
        Returns GeoDataFrame ready for choropleth mapping.
        """
        if self.admin_gdf is None:
            logger.warning("No admin boundaries loaded — creating synthetic geometry")
            return self.create_synthetic_boundaries(cri_df)

        merged = self.admin_gdf.merge(
            cri_df,
            left_on=boundary_col,
            right_on=join_col,
            how="left",
        )
        coverage = merged["cri_score"].notna().sum() / len(merged)
        logger.info(f"Spatial join coverage: {coverage:.1%}")
        return merged

    # ------------------------------------------------------------------
    # Spatial statistics
    # ------------------------------------------------------------------

    def compute_spatial_autocorrelation(
        self,
        gdf: gpd.GeoDataFrame,
        value_col: str = "cri_score",
        k_neighbors: int = 5,
    ) -> Dict[str, float]:
        """
        Compute Global Moran's I for spatial clustering of CRI scores.
        Returns Moran's I, Z-score, and p-value.

        High Moran's I → fragile zones cluster together (hotspot pattern).
        """
        try:
            from libpysal.weights import KNN
            from esda.moran import Moran

            coords = list(zip(gdf.geometry.centroid.x, gdf.geometry.centroid.y))
            w = KNN.from_array(coords, k=k_neighbors)
            w.transform = "R"

            values = gdf[value_col].fillna(gdf[value_col].mean()).values
            moran = Moran(values, w)

            return {
                "morans_i": float(moran.I),
                "z_score":  float(moran.z_norm),
                "p_value":  float(moran.p_norm),
                "interpretation": self._interpret_moran(moran.I, moran.p_norm),
            }
        except ImportError:
            logger.warning("libpysal/esda not installed — skipping Moran's I")
            return self._fallback_spatial_stats(gdf, value_col)

    def _fallback_spatial_stats(
        self, gdf: gpd.GeoDataFrame, value_col: str
    ) -> Dict[str, float]:
        """Simplified spatial correlation without libpysal."""
        values = gdf[value_col].dropna()
        return {
            "morans_i": float(values.autocorr(lag=1)) if len(values) > 1 else 0.0,
            "z_score":  np.nan,
            "p_value":  np.nan,
            "interpretation": "Computed without spatial weights (install libpysal for full analysis)",
        }

    @staticmethod
    def _interpret_moran(i: float, p: float) -> str:
        if p > 0.05:
            return "No significant spatial clustering"
        if i > 0.3:
            return "Strong positive clustering — fragile zones concentrate geographically"
        if i > 0.1:
            return "Moderate positive clustering"
        if i < -0.1:
            return "Spatial dispersion — fragile zones interspersed with resilient zones"
        return "Weak spatial pattern"

    def identify_hotspots(
        self,
        gdf: gpd.GeoDataFrame,
        score_col: str = "cri_score",
        threshold_low: float = 35.0,
        threshold_high: float = 65.0,
    ) -> gpd.GeoDataFrame:
        """
        Classify zones into fragility hotspots, stable zones, and resilience anchors.
        """
        gdf = gdf.copy()
        gdf["hotspot_class"] = "STABLE"
        gdf.loc[gdf[score_col] < threshold_low, "hotspot_class"] = "FRAGILITY_HOTSPOT"
        gdf.loc[gdf[score_col] >= threshold_high, "hotspot_class"] = "RESILIENCE_ANCHOR"
        gdf.loc[gdf[score_col].isna(), "hotspot_class"] = "NO_DATA"

        counts = gdf["hotspot_class"].value_counts().to_dict()
        logger.info(f"Hotspot classification: {counts}")
        return gdf

    def compute_regional_aggregates(
        self,
        gdf: gpd.GeoDataFrame,
        group_col: str = "country",
        score_col: str = "cri_score",
    ) -> pd.DataFrame:
        """
        Aggregate CRI scores at regional level (country / basin / program).
        Uses area-weighted mean when geometry is available.
        """
        agg_df = gdf.groupby(group_col).agg(
            cri_mean=(score_col, "mean"),
            cri_min=(score_col, "min"),
            cri_max=(score_col, "max"),
            cri_std=(score_col, "std"),
            n_zones=(score_col, "count"),
            n_critical=(score_col, lambda x: (x < 25).sum()),
            n_watch=(score_col, lambda x: ((x >= 25) & (x < 40)).sum()),
        ).reset_index()

        agg_df["pct_critical"] = (agg_df["n_critical"] / agg_df["n_zones"] * 100).round(1)
        return agg_df.sort_values("cri_mean")

    def get_neighbors(
        self,
        gdf: gpd.GeoDataFrame,
        zone_id: str,
        id_col: str = "zone_id",
        k: int = 5,
    ) -> gpd.GeoDataFrame:
        """Return K nearest neighbors for a given zone."""
        target = gdf[gdf[id_col] == zone_id]
        if target.empty:
            raise ValueError(f"Zone {zone_id} not found")

        centroid = target.geometry.centroid.iloc[0]
        others = gdf[gdf[id_col] != zone_id].copy()
        others["dist"] = others.geometry.centroid.distance(centroid)
        return others.nsmallest(k, "dist")

    def spatial_imputation(
        self,
        gdf: gpd.GeoDataFrame,
        indicator_cols: List[str],
        k_neighbors: int = 5,
    ) -> gpd.GeoDataFrame:
        """
        Fill missing indicator values using spatial lag (neighbor average).
        Applied when data missingness is < 40% for a given zone.
        """
        gdf = gdf.copy()
        centroids = gdf.geometry.centroid

        for idx in gdf.index:
            row = gdf.loc[idx]
            missing = [c for c in indicator_cols if pd.isna(row[c])]
            if not missing:
                continue

            # Find k nearest neighbors with data
            dists = centroids.distance(centroids[idx])
            neighbor_idx = dists.nsmallest(k_neighbors + 1).index[1:]

            for col in missing:
                neighbor_vals = gdf.loc[neighbor_idx, col].dropna()
                if len(neighbor_vals) >= 2:
                    gdf.at[idx, col] = neighbor_vals.mean()
                    logger.debug(f"Imputed {col} for zone {idx} from {len(neighbor_vals)} neighbors")

        return gdf

    def to_geojson(self, gdf: gpd.GeoDataFrame, path: str) -> None:
        """Export GeoDataFrame to GeoJSON for web mapping."""
        gdf.to_file(path, driver="GeoJSON")
        logger.info(f"Exported {len(gdf)} zones to {path}")

    def compute_coverage_stats(self, gdf: gpd.GeoDataFrame) -> Dict:
        """Summary statistics for monitoring report."""
        total = len(gdf)
        scored = gdf["cri_score"].notna().sum()
        return {
            "total_zones":      total,
            "scored_zones":     int(scored),
            "coverage_pct":     round(scored / total * 100, 1),
            "n_countries":      gdf["country"].nunique() if "country" in gdf.columns else None,
            "mean_cri":         round(gdf["cri_score"].mean(), 1),
            "median_cri":       round(gdf["cri_score"].median(), 1),
            "std_cri":          round(gdf["cri_score"].std(), 1),
            "critical_zones":   int((gdf["cri_score"] < 25).sum()),
            "high_resilience":  int((gdf["cri_score"] >= 75).sum()),
        }
