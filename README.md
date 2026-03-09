---
title: Climate Resilience Index
emoji: 🌡️
colorFrom: red
colorTo: green
sdk: docker
pinned: true
---
# 🌡️ Climate-Resilience-Index

> **Territorialized Climate Resilience Monitoring System**  
> Developed for the [AdaptAction programme](https://www.expertisefrance.fr) (Expertise France) — West & Central Africa

[![CI](https://github.com/your-username/Climate-Resilience-Index/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/Climate-Resilience-Index/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 🌍 Overview

The **Climate Resilience Index (CRI)** is an open-source framework for computing multi-dimensional, territorialized resilience scores at sub-national level across Sub-Saharan Africa. It synthesizes climate exposure data (NASA POWER, ERA5), socio-economic vulnerability indicators, and ecosystem metrics into a single composite index — updated quarterly via automated pipeline.

**Designed for:** Expertise France, GIZ, Enabel, UNDP, and other international development actors integrating climate resilience into programme M&E cycles.

**Coverage:** Niger, Mali, Burkina Faso, Senegal, Mauritania, Chad, CAR, Cameroon, and 4+ additional countries (AdaptAction scope: 12–18 countries).

---

## 📊 Dashboard Preview

```
🗺️ Territorial Map        🚨 Active Alerts
┌──────────────────────┐   ┌─────────────────────┐
│  ●●● Niger           │   │ 🔴 Agadez NER-001   │
│   ●●● Mali  ●●●      │   │    CRI: 18.2 ↓ -6.1 │
│      ●● BFA          │   │ 🟠 Gao MLI-007      │
│  🔴=critical         │   │    CRI: 28.4 ↓ -5.2 │
│  🟡=watch            │   │ 🟡 Maradi NER-003   │
│  🟢=resilient        │   │    CRI: 37.8 stable  │
└──────────────────────┘   └─────────────────────┘

📈 CRI Trajectories         🕷️ Pillar Radar
   Niger ────────           Exposure ●
   Mali  ──────             Sensitivity ●
   BFA   ──────             Adaptive Cap ●
          ↑ thresholds      Livelihood ●
```

▶ **[Live Demo](https://huggingface.co/spaces/Lokozu/Climate-Resilience-Index)** — run `python dashboard/app.py`

---

## 🏗️ Architecture

```
Climate-Resilience-Index/
│
├── climate_resilience/        # Core Python package
│   ├── indices.py             # CRI computation engine (5-pillar model)
│   ├── data_loader.py         # NASA POWER / ERA5 / CHIRPS ingestion
│   ├── spatial.py             # GeoPandas territorial aggregation
│   └── alerts.py              # Degradation alert system (5 tiers)
│
├── dashboard/
│   └── app.py                 # Dash interactive dashboard
│
├── scripts/
│   ├── run_pipeline.py        # Weekly automated CRI computation
│   └── generate_alerts.py    # Alert report generation
│
├── data/
│   ├── raw/                   # ERA5 / CHIRPS source data
│   ├── processed/             # Computed CRI outputs (auto-updated)
│   └── shapefiles/            # Admin boundary GeoJSON
│
├── tests/
│   └── test_cri.py            # Unit + integration tests (30+ cases)
│
└── .github/workflows/
    └── ci.yml                 # CI/CD: lint → test → compute → Docker
```

---

## 🧮 Methodology

### The 5-Pillar CRI Model

| Pillar | Weight (Sahel) | Key Indicators |
|--------|---------------|----------------|
| **Exposure** | 35% | Drought frequency, extreme heat days, flood risk, precipitation anomaly |
| **Sensitivity** | 25% | Poverty rate, IPC food insecurity phase, displacement rate |
| **Adaptive Capacity** | 20% | WASH access, healthcare coverage, early warning systems |
| **Livelihood** | 15% | NDVI trend, crop yield variability, market access index |
| **Ecosystem** | 5% | Forest cover change, soil degradation, watershed integrity |

> **CRI = Σ (pillar_score × pillar_weight)** | Scale: 0 (critical fragility) → 100 (high resilience)

Weights are **context-calibrated** by agro-ecological zone: Sahelian, Guinean, Coastal, Forest Basin.

### Alert Classification

| Level | Trigger | Response |
|-------|---------|----------|
| 🔴 **CRITICAL** | CRI < 25 | Immediate inter-agency coordination |
| 🟠 **DEGRADING** | CRI drop > 5 pts/quarter | Flash update + rapid assessment |
| 🟡 **WATCH** | CRI < 40 or sustained 3-quarter decline | Enhanced monitoring |
| 🔵 **IMPROVING** | CRI gain > 3 pts | Document & scale successful interventions |
| 🟢 **NORMAL** | Stable | Routine monitoring |

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/sanyamsin/Climate-Resilience-Index.git
cd Climate-Resilience-Index
pip install -r requirements.txt
```

### 2. Run the Dashboard

```bash
python dashboard/app.py
# → http://localhost:8050
```

### 3. Compute CRI Programmatically

```python
from climate_resilience import ClimateResilienceIndex, ClimateDataLoader

# Load & normalize indicators
loader = ClimateDataLoader()
df = loader.generate_synthetic_dataset(n_zones=50, n_periods=8)
df = loader.normalize_indicators(df)

# Initialize CRI engine (Sahel calibration)
cri = ClimateResilienceIndex(context="sahel")

# Compute for a single zone
zone_data = df.iloc[0]
profile = cri.compute(zone_data, zone_id="NER-AGD-001",
                      zone_name="Agadez", country="Niger")

print(f"CRI: {profile.cri_score:.1f} — {profile.grade}")
# → CRI: 43.2 — LOW RESILIENCE

# Uncertainty quantification
bounds = cri.uncertainty_bounds(zone_data)
print(f"95% CI: [{bounds['ci_lower_95']:.1f}, {bounds['ci_upper_95']:.1f}]")
```

### 4. Batch Processing + Alerts

```python
from climate_resilience import DegradationAlertSystem

# Batch CRI for all zones
results = cri.compute_batch(df.groupby("zone_id").last())

# Generate alerts
alert_system = DegradationAlertSystem()
alerts = alert_system.evaluate(results)

# Print M&E report
print(alert_system.generate_m_and_e_report("2024-Q1"))
```

### 5. Docker

```bash
docker build -t climate-resilience-index .
docker run -p 8050:8050 climate-resilience-index
```

---

## 📡 Data Sources

| Source | Variables | Access |
|--------|-----------|--------|
| [NASA POWER](https://power.larc.nasa.gov/) | Temperature, precipitation, solar radiation | Free API |
| [ERA5 (Copernicus)](https://cds.climate.copernicus.eu/) | Reanalysis: soil moisture, wind, heat | Free account |
| [CHIRPS](https://www.chc.ucsb.edu/data/chirps) | Rainfall estimates (5km resolution) | Free |
| [MODIS NDVI](https://earthdata.nasa.gov/) | Vegetation dynamics | Free account |
| [IPC](https://www.ipcinfo.org/) | Food insecurity phases | Public |
| [World Bank](https://data.worldbank.org/) | Socio-economic indicators | Free API |

> In offline/demo mode, all data is replaced by a realistic synthetic generator.

---

## 🧪 Tests

```bash
pytest tests/ -v --cov=climate_resilience

# Output:
# tests/test_cri.py::TestClimateResilienceIndex::test_compute_returns_profile PASSED
# tests/test_cri.py::TestClimateResilienceIndex::test_resilient_zone_scores_high PASSED
# tests/test_cri.py::TestDegradationAlertSystem::test_critical_alert_triggered PASSED
# ... 30+ tests
# Coverage: 87%
```

---

## 🗺️ Programme Context

This tool was developed to support the **AdaptAction programme** (Expertise France, AFD-funded, €15M, 2021–2025), which builds climate resilience capacities across 12–18 African countries. The CRI framework directly feeds:

- **Quarterly M&E reports** submitted to AFD/EU
- **Early warning dashboards** for country programme coordinators
- **Resource allocation decisions** for adaptive management
- **Impact measurement** across adaptation interventions

---

## 👤 Author

**Serge Nyamsin**  
MSc Data Science & AI | 12+ years humanitarian field experience  
West & Central Africa specialist (Niger, Mali, Burkina Faso, Senegal, Mauritania, CAR)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/serge-alain-nyamsin/)
[![GitHub](https://img.shields.io/badge/GitHub-Portfolio-black)](https://github.com/sanyamsin)

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

*Part of a humanitarian data science portfolio integrating field expertise with modern ML/geospatial capabilities.*
