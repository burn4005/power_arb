"""Shared feature engineering for ML forecasters.

Extracts time-based, daylight, price-specific, and consumption-specific
features for use by GradientBoostingRegressor models.

All features are numeric floats. Cyclical time features use sin/cos
encoding to preserve periodicity (23:55 is close to 00:05).
"""

import math
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Try importing astral for sunrise/sunset computation
try:
    from astral import LocationInfo
    from astral.sun import sun
    _ASTRAL_AVAILABLE = True
except ImportError:
    _ASTRAL_AVAILABLE = False
    logger.info("astral library not installed; using estimated daylight features")


def cyclical_encode(value: float, period: float) -> tuple[float, float]:
    """Encode a cyclical value as (sin, cos) pair."""
    angle = 2 * math.pi * value / period
    return math.sin(angle), math.cos(angle)


def extract_time_features(ts: datetime) -> dict:
    """Extract time-based features from a timestamp."""
    hour_sin, hour_cos = cyclical_encode(ts.hour + ts.minute / 60.0, 24.0)
    dow_sin, dow_cos = cyclical_encode(ts.weekday(), 7.0)
    month_sin, month_cos = cyclical_encode(ts.month - 1, 12.0)

    return {
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "dow_sin": dow_sin,
        "dow_cos": dow_cos,
        "is_weekend": 1.0 if ts.weekday() >= 5 else 0.0,
        "month_sin": month_sin,
        "month_cos": month_cos,
    }


def extract_daylight_features(
    ts: datetime,
    latitude: float,
    longitude: float,
    weather_sunrise: str | None = None,
    weather_sunset: str | None = None,
) -> dict:
    """Extract sunrise/sunset/daylight features.

    Uses astral library if available, otherwise falls back to weather
    cache data or rough Brisbane-latitude estimates.
    """
    sunrise_dt = None
    sunset_dt = None

    if _ASTRAL_AVAILABLE:
        try:
            loc = LocationInfo(latitude=latitude, longitude=longitude)
            s = sun(loc.observer, date=ts.date())
            sunrise_dt = s["sunrise"].replace(tzinfo=None)
            sunset_dt = s["sunset"].replace(tzinfo=None)
        except Exception:
            pass

    if sunrise_dt is None and weather_sunrise:
        try:
            sunrise_dt = datetime.fromisoformat(weather_sunrise)
        except (ValueError, TypeError):
            pass

    if sunset_dt is None and weather_sunset:
        try:
            sunset_dt = datetime.fromisoformat(weather_sunset)
        except (ValueError, TypeError):
            pass

    features = {}
    if sunrise_dt and sunset_dt:
        daylight_hours = (sunset_dt - sunrise_dt).total_seconds() / 3600.0
        hours_since_sunrise = (ts - sunrise_dt).total_seconds() / 3600.0
        hours_until_sunset = (sunset_dt - ts).total_seconds() / 3600.0
        features["daylight_hours"] = daylight_hours
        features["hours_since_sunrise"] = hours_since_sunrise
        features["hours_until_sunset"] = hours_until_sunset
    else:
        # Rough estimate for Brisbane latitude (~-27.5)
        features["daylight_hours"] = 12.0
        features["hours_since_sunrise"] = max(0.0, ts.hour - 6.0)
        features["hours_until_sunset"] = max(0.0, 18.0 - ts.hour)

    return features


def extract_price_features(
    raw_price: float,
    lead_time_hours: float,
    spike_status: str = "none",
) -> dict:
    """Extract price-specific features."""
    return {
        "raw_price": raw_price,
        "lead_time_hours": lead_time_hours,
        "spike_potential": 1.0 if spike_status == "potential" else 0.0,
        "spike_active": 1.0 if spike_status == "spike" else 0.0,
    }


def extract_consumption_features(
    occupancy: bool,
    recent_loads_kw: list[float],
    profile_load_kw: float,
    ac_threshold_kw: float = 2.0,
) -> dict:
    """Extract consumption-specific features.

    Args:
        occupancy: Whether anyone is home.
        recent_loads_kw: Recent load readings in kW (most recent last).
        profile_load_kw: P75 profile prediction for this slot.
        ac_threshold_kw: Load above profile that indicates AC running.
    """
    # Recent load averages
    if recent_loads_kw:
        last_12 = recent_loads_kw[-12:]
        last_36 = recent_loads_kw[-36:]
        avg_1h = sum(last_12) / len(last_12)
        avg_3h = sum(last_36) / len(last_36)
        current_kw = recent_loads_kw[-1]
    else:
        avg_1h = 0.0
        avg_3h = 0.0
        current_kw = 0.0

    # AC detection: current load significantly above profile
    ac_running = 1.0 if current_kw > profile_load_kw + ac_threshold_kw else 0.0

    # AC duration: how many recent periods had AC-level loads
    ac_count = sum(
        1 for kw in (recent_loads_kw[-12:] if recent_loads_kw else [])
        if kw > profile_load_kw + ac_threshold_kw
    )
    ac_duration_mins = ac_count * 5

    return {
        "occupancy": 1.0 if occupancy else 0.0,
        "recent_load_avg_1h_kw": avg_1h,
        "recent_load_avg_3h_kw": avg_3h,
        "ac_running": ac_running,
        "ac_duration_mins": float(ac_duration_mins),
        "profile_load_kw": profile_load_kw,
    }


# Canonical feature order for each model type.
# Both models share the same feature vectors to ensure consistency
# between training and inference.
PRICE_FEATURE_NAMES = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
    "month_sin", "month_cos", "temperature_c", "daylight_hours",
    "hours_since_sunrise", "hours_until_sunset",
    "raw_price", "lead_time_hours", "spike_potential", "spike_active",
]

CONSUMPTION_FEATURE_NAMES = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
    "month_sin", "month_cos", "temperature_c", "daylight_hours",
    "hours_since_sunrise", "hours_until_sunset",
    "occupancy", "recent_load_avg_1h_kw", "recent_load_avg_3h_kw",
    "ac_running", "ac_duration_mins", "profile_load_kw",
]


def features_to_vector(features: dict, feature_names: list[str]) -> list[float]:
    """Convert a feature dict to an ordered vector, using 0.0 for missing keys."""
    return [features.get(name, 0.0) for name in feature_names]
