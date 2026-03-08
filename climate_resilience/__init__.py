from .indices import ClimateResilienceIndex
from .data_loader import ClimateDataLoader
from .spatial import SpatialAggregator
from .alerts import DegradationAlertSystem

__version__ = "1.0.0"
__author__ = "Serge Nyamsin"

__all__ = [
    "ClimateResilienceIndex",
    "ClimateDataLoader",
    "SpatialAggregator",
    "DegradationAlertSystem",
]