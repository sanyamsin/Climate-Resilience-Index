#!/usr/bin/env python3
"""
CRI Pipeline Runner
===================
Scheduled pipeline for computing Climate Resilience Indices.
Runs weekly via GitHub Actions; can also be triggered manually.

Usage:
    python scripts/run_pipeline.py --context sahel --period 2024-Q1
    python scripts/run_pipeline.py --context sahel --output data/processed/
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from climate_resilience import ClimateResilienceIndex, ClimateDataLoader, DegradationAlertSystem

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cri-pipeline")


def run_pipeline(
    context: str = "sahel",
    output_dir: str = "data/processed",
    period: str = None,
    n_zones: int = 60,
    seed: int = 42,
):
    period = period or f"{datetime.now().year}-Q{(datetime.now().month - 1) // 3 + 1}"

    logger.info("=" * 60)
    logger.info("🌡️  CLIMATE RESILIENCE INDEX PIPELINE")
    logger.info(f"   Context: {context} | Period: {period}")
    logger.info("=" * 60)

    # 1. Load data
    logger.info("📥 Loading climate indicators...")
    loader = ClimateDataLoader(output_dir, offline_mode=True)
    raw_df = loader.generate_synthetic_dataset(n_zones=n_zones, n_periods=8, seed=seed)
    norm_df = loader.normalize_indicators(raw_df)
    logger.info(f"   ✓ {len(norm_df)} zone-periods loaded")

    # 2. Compute CRI
    logger.info("🧮 Computing CRI scores...")
    cri_engine = ClimateResilienceIndex(context=context)
    latest_df = loader.get_latest_period(norm_df)
    results_df = cri_engine.compute_batch(latest_df, period=period)

    meta = latest_df[["zone_id","zone_name","country","country_code","latitude","longitude","period"]].drop_duplicates("zone_id")
    full_df = results_df.merge(meta, on="zone_id", how="left")
    logger.info(f"   ✓ Mean CRI: {full_df['cri_score'].mean():.1f} | "
                f"Critical zones: {(full_df['cri_score'] < 25).sum()}")

    # 3. Generate alerts
    logger.info("🚨 Running degradation detection...")
    alert_system = DegradationAlertSystem()
    alerts = alert_system.evaluate(full_df)
    stats = alert_system.get_statistics(alerts)
    logger.info(f"   ✓ {stats.get('n_critical_zones', 0)} CRITICAL | "
                f"{stats.get('n_degrading_zones', 0)} DEGRADING | "
                f"{stats.get('pct_watch_or_above', 0):.0f}% at risk")

    # 4. Save outputs
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cri_output = out_path / f"cri_scores_{period}.csv"
    full_df.to_csv(cri_output, index=False)
    logger.info(f"   💾 CRI scores → {cri_output}")

    alert_output = out_path / f"alerts_{period}.csv"
    alert_system.get_summary(alerts).to_csv(alert_output, index=False)
    logger.info(f"   💾 Alerts    → {alert_output}")

    report_output = out_path / f"report_{period}.md"
    report = alert_system.generate_m_and_e_report(period, alerts)
    report_output.write_text(report)
    logger.info(f"   💾 M&E report → {report_output}")

    logger.info("=" * 60)
    logger.info("✅ Pipeline completed successfully")
    logger.info("=" * 60)

    return full_df, alerts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CRI Pipeline Runner")
    parser.add_argument("--context", default="sahel",
                        choices=["sahel", "coastal", "forest_basin", "default"])
    parser.add_argument("--output", default="data/processed")
    parser.add_argument("--period", default=None)
    parser.add_argument("--n-zones", type=int, default=60)

    args = parser.parse_args()
    run_pipeline(
        context=args.context,
        output_dir=args.output,
        period=args.period,
        n_zones=args.n_zones,
    )
