"""
Degradation Alert System
========================
Real-time monitoring of CRI trajectory with multi-tier alerting:

  TIER 1 — CRITICAL    : CRI < 25 | Immediate humanitarian response needed
  TIER 2 — DEGRADING   : CRI drop > 5 points in one quarter
  TIER 3 — WATCH       : CRI 25–40 or sustained 3-quarter decline
  TIER 4 — IMPROVING   : CRI gain > 3 points | Positive trajectory
  TIER 5 — NORMAL      : Stable, within expected variance

Integrates with programme monitoring cycles (quarterly M&E reporting).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    CRITICAL  = 5
    DEGRADING = 4
    WATCH     = 3
    NORMAL    = 2
    IMPROVING = 1


ALERT_CONFIG = {
    AlertLevel.CRITICAL:  {"color": "#d63031", "icon": "🔴", "threshold_score": 25},
    AlertLevel.DEGRADING: {"color": "#e17055", "icon": "🟠", "drop_threshold": 5.0},
    AlertLevel.WATCH:     {"color": "#fdcb6e", "icon": "🟡", "threshold_score": 40, "drop_threshold": 2.0},
    AlertLevel.NORMAL:    {"color": "#00b894", "icon": "🟢"},
    AlertLevel.IMPROVING: {"color": "#0984e3", "icon": "🔵", "gain_threshold": 3.0},
}


@dataclass
class Alert:
    """Single alert instance for a geographic zone."""
    zone_id:     str
    zone_name:   str
    country:     str
    period:      str
    level:       AlertLevel
    cri_score:   float
    trend:       Optional[float]
    message:     str
    affected_pillars: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    created_at:  datetime = field(default_factory=datetime.utcnow)

    @property
    def severity_score(self) -> int:
        return self.level.value

    @property
    def icon(self) -> str:
        return ALERT_CONFIG[self.level]["icon"]

    @property
    def color(self) -> str:
        return ALERT_CONFIG[self.level]["color"]

    def to_dict(self) -> dict:
        return {
            "zone_id":   self.zone_id,
            "zone_name": self.zone_name,
            "country":   self.country,
            "period":    self.period,
            "level":     self.level.name,
            "icon":      self.icon,
            "cri_score": round(self.cri_score, 1),
            "trend":     round(self.trend, 2) if self.trend else None,
            "message":   self.message,
            "affected_pillars": self.affected_pillars,
            "recommended_actions": self.recommended_actions,
            "created_at": self.created_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# Recommended action library (AdaptAction programme catalogue)
# ---------------------------------------------------------------------------
ACTION_LIBRARY = {
    "drought": [
        "Activate contingency plans for drought-affected communities",
        "Deploy cash transfer programmes via WFP pipeline",
        "Strengthen early warning dissemination at community level",
        "Coordinate with OCHA for humanitarian corridor activation",
    ],
    "food_insecurity": [
        "Escalate IPC monitoring frequency to bi-weekly",
        "Activate emergency seed distribution for next planting season",
        "Engage WFP/FAO for food basket pre-positioning",
    ],
    "adaptive_capacity": [
        "Prioritize WASH infrastructure rehabilitation",
        "Fast-track community early warning system installation",
        "Conduct emergency capacity building with local disaster risk committees",
    ],
    "ecosystem": [
        "Implement emergency reforestation/assisted natural regeneration",
        "Activate watershed protection bylaws with local authorities",
        "Deploy Farmer Managed Natural Regeneration (FMNR) extension services",
    ],
    "general_critical": [
        "Convene inter-agency emergency coordination meeting",
        "Produce flash update for donor reporting within 48h",
        "Deploy rapid assessment mission within 72h",
    ],
}


class DegradationAlertSystem:
    """
    Monitors CRI trajectories and generates prioritized alerts.

    Designed for integration with programme M&E cycles (quarterly),
    with optional webhook/email notification for field teams.
    """

    def __init__(
        self,
        critical_threshold: float = 25.0,
        watch_threshold: float = 40.0,
        degrading_drop: float = 5.0,
        watch_drop: float = 2.0,
        improving_gain: float = 3.0,
        min_periods: int = 2,
        notification_callback: Optional[Callable] = None,
    ):
        self.critical_threshold = critical_threshold
        self.watch_threshold    = watch_threshold
        self.degrading_drop     = degrading_drop
        self.watch_drop         = watch_drop
        self.improving_gain     = improving_gain
        self.min_periods        = min_periods
        self.notification_callback = notification_callback
        self._alerts: List[Alert] = []

    # ------------------------------------------------------------------
    # Alert generation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        cri_df: pd.DataFrame,
        pillar_cols: Optional[List[str]] = None,
        period_col: str = "period",
        score_col: str = "cri_score",
    ) -> List[Alert]:
        """
        Evaluate CRI results and generate alerts.

        Parameters
        ----------
        cri_df : DataFrame
            Must contain zone_id, zone_name, country, period, cri_score.
        pillar_cols : list, optional
            Column names for individual pillar scores.
        """
        alerts = []
        pillar_cols = pillar_cols or [c for c in cri_df.columns if c.startswith("pillar_")]

        # Ensure period column exists; if not, add a dummy
        if period_col not in cri_df.columns:
            cri_df = cri_df.copy()
            cri_df[period_col] = "latest"

        for zone_id in cri_df["zone_id"].unique():
            zone_data = cri_df[cri_df["zone_id"] == zone_id].sort_values(period_col)
            if zone_data.empty:
                continue

            latest = zone_data.iloc[-1]
            current_score = latest[score_col]

            if pd.isna(current_score):
                continue

            trend = None
            if len(zone_data) >= self.min_periods:
                prev_score = zone_data.iloc[-2][score_col]
                if pd.notna(prev_score):
                    trend = float(current_score - prev_score)

            # Detect which pillars are most degraded
            affected_pillars = []
            for col in pillar_cols:
                if col in latest and pd.notna(latest[col]) and latest[col] < 30:
                    affected_pillars.append(col.replace("pillar_", ""))

            alert = self._classify_alert(
                zone_id=str(latest["zone_id"]),
                zone_name=str(latest.get("zone_name", zone_id)),
                country=str(latest.get("country", "")),
                period=str(latest[period_col]),
                score=float(current_score),
                trend=trend,
                affected_pillars=affected_pillars,
                zone_history=zone_data[score_col].tolist(),
            )

            if alert:
                alerts.append(alert)
                if self.notification_callback and alert.level.value >= AlertLevel.DEGRADING.value:
                    self.notification_callback(alert)

        self._alerts.extend(alerts)
        critical_count = sum(1 for a in alerts if a.level == AlertLevel.CRITICAL)
        logger.info(
            f"Alert evaluation: {len(alerts)} alerts | "
            f"{critical_count} CRITICAL | Period: {cri_df[period_col].max()}"
        )
        return alerts

    def _classify_alert(
        self,
        zone_id: str,
        zone_name: str,
        country: str,
        period: str,
        score: float,
        trend: Optional[float],
        affected_pillars: List[str],
        zone_history: List[float],
    ) -> Optional[Alert]:
        """Classify a single zone's alert level and generate recommendations."""

        # Sustained decline: 3 consecutive periods of decline
        sustained_decline = False
        if len(zone_history) >= 3:
           recent = list(zone_history)[-3:]
           diffs = [recent[i] - recent[i-1] for i in range(1, len(recent))]
           sustained_decline = all(d < 0 for d in diffs if not np.isnan(d))

        # ----------------------------------------------------------------
        # CRITICAL
        if score < self.critical_threshold:
            return Alert(
                zone_id=zone_id, zone_name=zone_name, country=country,
                period=period, level=AlertLevel.CRITICAL,
                cri_score=score, trend=trend,
                message=(
                    f"{zone_name} ({country}) is in CRITICAL fragility "
                    f"(CRI={score:.1f}). "
                    + (f"Declining {abs(trend):.1f} pts vs previous period." if trend and trend < 0 else "")
                ),
                affected_pillars=affected_pillars,
                recommended_actions=self._get_actions(affected_pillars, "general_critical"),
            )

        # DEGRADING
        if trend is not None and trend <= -self.degrading_drop:
            return Alert(
                zone_id=zone_id, zone_name=zone_name, country=country,
                period=period, level=AlertLevel.DEGRADING,
                cri_score=score, trend=trend,
                message=(
                    f"RAPID DEGRADATION in {zone_name} ({country}): "
                    f"CRI dropped {abs(trend):.1f} pts to {score:.1f} this quarter."
                ),
                affected_pillars=affected_pillars,
                recommended_actions=self._get_actions(affected_pillars, "drought"),
            )

        # WATCH
        if score < self.watch_threshold or (trend is not None and trend <= -self.watch_drop) or sustained_decline:
            reason = []
            if score < self.watch_threshold: reason.append(f"score {score:.1f} below threshold")
            if trend and trend <= -self.watch_drop: reason.append(f"declining {abs(trend):.1f} pts")
            if sustained_decline: reason.append("3-period sustained decline")

            return Alert(
                zone_id=zone_id, zone_name=zone_name, country=country,
                period=period, level=AlertLevel.WATCH,
                cri_score=score, trend=trend,
                message=f"WATCH: {zone_name} ({country}) — {'; '.join(reason)}.",
                affected_pillars=affected_pillars,
                recommended_actions=self._get_actions(affected_pillars, "adaptive_capacity"),
            )

        # IMPROVING
        if trend is not None and trend >= self.improving_gain:
            return Alert(
                zone_id=zone_id, zone_name=zone_name, country=country,
                period=period, level=AlertLevel.IMPROVING,
                cri_score=score, trend=trend,
                message=(
                    f"POSITIVE TRAJECTORY in {zone_name} ({country}): "
                    f"CRI improved +{trend:.1f} pts to {score:.1f}."
                ),
                affected_pillars=[],
                recommended_actions=["Document and scale successful interventions"],
            )

        # NORMAL — no alert needed
        return None

    def _get_actions(self, pillars: List[str], default_key: str) -> List[str]:
        """Build action list from affected pillars."""
        actions = ACTION_LIBRARY.get(default_key, []).copy()
        for pillar in pillars[:2]:  # Limit to top 2 pillars
            if pillar in ACTION_LIBRARY:
                actions.extend(ACTION_LIBRARY[pillar][:2])
        return list(dict.fromkeys(actions))[:5]  # Deduplicate, max 5

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_summary(self, alerts: Optional[List[Alert]] = None) -> pd.DataFrame:
        """Generate alert summary DataFrame for dashboard/reporting."""
        alerts = alerts or self._alerts
        if not alerts:
            return pd.DataFrame()
        return pd.DataFrame([a.to_dict() for a in alerts]).sort_values(
            "level", ascending=False
        )

    def get_critical_zones(self, alerts: Optional[List[Alert]] = None) -> List[Alert]:
        """Return only CRITICAL and DEGRADING alerts."""
        alerts = alerts or self._alerts
        return [a for a in alerts if a.level.value >= AlertLevel.DEGRADING.value]

    def get_statistics(self, alerts: Optional[List[Alert]] = None) -> Dict:
        """Aggregate statistics for programme monitoring report."""
        alerts = alerts or self._alerts
        if not alerts:
            return {}

        level_counts = {}
        for level in AlertLevel:
            level_counts[level.name] = sum(1 for a in alerts if a.level == level)

        countries = list({a.country for a in alerts if a.level == AlertLevel.CRITICAL})

        return {
            "total_alerts":       len(alerts),
            "by_level":           level_counts,
            "critical_countries": countries,
            "n_critical_zones":   level_counts.get("CRITICAL", 0),
            "n_degrading_zones":  level_counts.get("DEGRADING", 0),
            "pct_watch_or_above": round(
                (level_counts.get("CRITICAL", 0) + level_counts.get("DEGRADING", 0) + level_counts.get("WATCH", 0))
                / max(len(alerts), 1) * 100, 1
            ),
        }

    def generate_m_and_e_report(self, period: str, alerts: Optional[List[Alert]] = None) -> str:
        """
        Generate M&E narrative summary for donor reporting.
        Compatible with Expertise France / GIZ reporting templates.
        """
        alerts = alerts or self._alerts
        stats = self.get_statistics(alerts)
        critical = self.get_critical_zones(alerts)

        report_lines = [
            f"# Climate Resilience Monitoring Report — {period}",
            f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            "",
            "## Executive Summary",
            f"- **{stats.get('n_critical_zones', 0)} zones in CRITICAL fragility** requiring immediate response",
            f"- **{stats.get('n_degrading_zones', 0)} zones in RAPID DEGRADATION** (>5 pts decline)",
            f"- **{stats.get('pct_watch_or_above', 0)}%** of monitored zones at WATCH level or above",
            "",
            "## Critical Zones Requiring Immediate Action",
        ]

        for alert in sorted(critical, key=lambda a: a.cri_score)[:10]:
            report_lines.append(
                f"\n### {alert.icon} {alert.zone_name} ({alert.country})"
            )
            report_lines.append(f"**CRI Score:** {alert.cri_score:.1f} | **Status:** {alert.level.name}")
            report_lines.append(f"**Assessment:** {alert.message}")
            if alert.recommended_actions:
                report_lines.append("**Recommended Actions:**")
                for action in alert.recommended_actions[:3]:
                    report_lines.append(f"  - {action}")

        return "\n".join(report_lines)
