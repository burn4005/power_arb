"""Backtest validation: ML forecasters vs existing heuristic methods.

Loads historical AEMO prices and home usage from testdata/, trains ML models
on a rolling window, and compares forecast accuracy against the existing
PriceForecaster (time-of-day median) and ConsumptionForecaster (P75 profile).

No live APIs are used. Temperature data is fetched from Open-Meteo's
historical archive API for the test period.

Usage:
    python -m tests.test_ml_forecasters
"""

import csv
import json
import logging
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.data_loader import load_aemo_prices, load_home_usage, IMPORT_MARKUP_C
from forecasting.features import (
    extract_time_features, extract_daylight_features, extract_price_features,
    extract_consumption_features, PRICE_FEATURE_NAMES, CONSUMPTION_FEATURE_NAMES,
    features_to_vector,
)
from forecasting.price import PriceForecaster
from optimizer.battery_model import PERIOD_MINUTES, PERIODS_PER_HOUR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TESTDATA = Path(__file__).parent / "testdata"

# Brisbane coordinates for daylight features
LATITUDE = -27.47
LONGITUDE = 153.03

# Cache file for historical temperature data
TEMP_CACHE_FILE = TESTDATA / "temperature_cache.json"


# ---------------------------------------------------------------------------
# Historical temperature fetcher (Open-Meteo archive API, free, no key)
# ---------------------------------------------------------------------------

def fetch_historical_temperatures(
    start_date: str, end_date: str, cache_path: Path = TEMP_CACHE_FILE
) -> dict[str, float]:
    """Fetch hourly temperatures from Open-Meteo historical archive.

    Returns dict mapping "YYYY-MM-DDTHH:00" -> temperature_c.
    Caches to disk to avoid repeated API calls.

    Args:
        start_date: "YYYY-MM-DD"
        end_date: "YYYY-MM-DD"
    """
    if cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        # Check if cache covers our range
        if cached.get("start") <= start_date and cached.get("end") >= end_date:
            logger.info("Using cached temperature data (%d hours)", len(cached["data"]))
            return cached["data"]

    import requests

    logger.info("Fetching historical temperatures from Open-Meteo (%s to %s)...", start_date, end_date)

    # Open-Meteo archive API has a max of ~1 year per request, chunk if needed
    all_temps = {}
    chunk_start = datetime.strptime(start_date, "%Y-%m-%d")
    final_end = datetime.strptime(end_date, "%Y-%m-%d")

    while chunk_start < final_end:
        chunk_end = min(chunk_start + timedelta(days=364), final_end)
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "start_date": chunk_start.strftime("%Y-%m-%d"),
            "end_date": chunk_end.strftime("%Y-%m-%d"),
            "hourly": "temperature_2m",
            "timezone": "Australia/Brisbane",
        }

        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        times = data.get("hourly", {}).get("time", [])
        temps = data.get("hourly", {}).get("temperature_2m", [])

        for t, temp in zip(times, temps):
            if temp is not None:
                all_temps[t] = temp

        logger.info("  Fetched %d hours for %s to %s",
                     len(times), chunk_start.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d"))
        chunk_start = chunk_end + timedelta(days=1)

    # Cache to disk
    cache_data = {"start": start_date, "end": end_date, "data": all_temps}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache_data, f)
    logger.info("Cached %d temperature hours to %s", len(all_temps), cache_path)

    return all_temps


def get_temperature(temps: dict[str, float], ts: datetime) -> float:
    """Get temperature for a timestamp, interpolating between hours."""
    key = ts.strftime("%Y-%m-%dT%H:00")
    if key in temps:
        return temps[key]
    # Try adjacent hours
    prev_key = (ts.replace(minute=0, second=0)).strftime("%Y-%m-%dT%H:00")
    next_key = (ts.replace(minute=0, second=0) + timedelta(hours=1)).strftime("%Y-%m-%dT%H:00")
    prev_temp = temps.get(prev_key)
    next_temp = temps.get(next_key)
    if prev_temp is not None and next_temp is not None:
        frac = ts.minute / 60.0
        return prev_temp + frac * (next_temp - prev_temp)
    if prev_temp is not None:
        return prev_temp
    if next_temp is not None:
        return next_temp
    return 25.0  # Brisbane average fallback


# ---------------------------------------------------------------------------
# Price forecast evaluation
# ---------------------------------------------------------------------------

def evaluate_price_forecasting(
    aemo_prices: dict[str, float],
    temps: dict[str, float],
    train_days: int = 60,
    test_start_date: str = "2024/07/01",
    test_end_date: str = "2025/11/30",
    forecast_horizons: list[int] = None,
):
    """Compare ML price forecaster vs heuristic PriceForecaster.

    For each test day, builds a forecast from historical data and compares
    against actuals at various lead times.
    """
    from sklearn.ensemble import GradientBoostingRegressor

    if forecast_horizons is None:
        forecast_horizons = [1, 3, 6, 12, 24]  # hours ahead

    logger.info("=" * 70)
    logger.info("PRICE FORECAST EVALUATION")
    logger.info("=" * 70)

    # Parse test date range
    test_start = datetime.strptime(test_start_date, "%Y/%m/%d")
    test_end = datetime.strptime(test_end_date, "%Y/%m/%d")

    # Heuristic forecaster (existing system)
    heuristic = PriceForecaster(aemo_prices, import_markup_c=IMPORT_MARKUP_C, window_days=21)

    # Build sorted price list for ML training
    sorted_keys = sorted(aemo_prices.keys())
    price_by_key = aemo_prices

    # Collect results by horizon
    heuristic_errors = {h: [] for h in forecast_horizons}
    ml_errors = {h: [] for h in forecast_horizons}

    # ML model state
    ml_model = None
    last_train_week = None

    # Step through test period day by day
    current = test_start
    total_days = (test_end - test_start).days
    run_start = time.perf_counter()

    while current <= test_end:
        date_str = current.strftime("%Y/%m/%d")
        day_num = (current - test_start).days

        if day_num % 50 == 0:
            elapsed = time.perf_counter() - run_start
            pct = day_num / max(total_days, 1) * 100
            logger.info("  [%5.1f%%] Processing %s (day %d/%d, %.1fs elapsed)",
                        pct, date_str, day_num, total_days, elapsed)

        # Retrain ML model weekly (not daily - too slow for backtest)
        train_week = f"{current.year}-W{current.isocalendar()[1]:02d}"
        if ml_model is None or last_train_week != train_week:
            ml_model = _train_price_model(
                price_by_key, sorted_keys, temps, current, train_days, heuristic
            )
            last_train_week = train_week

        # Evaluate at 4 points during the day (00:00, 06:00, 12:00, 18:00)
        for eval_hour in [0, 6, 12, 18]:
            eval_time = current.replace(hour=eval_hour, minute=0)
            eval_key = eval_time.strftime("%Y/%m/%d %H:%M")

            for horizon_h in forecast_horizons:
                target_time = eval_time + timedelta(hours=horizon_h)
                target_key = target_time.strftime("%Y/%m/%d %H:%M")

                actual = price_by_key.get(target_key)
                if actual is None:
                    continue

                # Heuristic forecast
                fc = heuristic.forecast(eval_key, hours=horizon_h + 1)
                # Find the forecast period closest to target
                target_idx = horizon_h * PERIODS_PER_HOUR
                if target_idx < len(fc):
                    heuristic_pred = fc[target_idx]["export_c"]  # spot price
                    heuristic_errors[horizon_h].append(abs(heuristic_pred - actual))

                # ML forecast: heuristic + learned correction
                if ml_model is not None and target_idx < len(fc):
                    features = _build_price_features(
                        target_time, heuristic_pred, horizon_h, temps
                    )
                    vec = np.array([features_to_vector(features, PRICE_FEATURE_NAMES)])
                    correction = float(ml_model.predict(vec)[0])
                    ml_pred = heuristic_pred + correction
                    ml_pred = max(-100.0, min(1500.0, ml_pred))
                    ml_errors[horizon_h].append(abs(ml_pred - actual))

        current += timedelta(days=1)

    # Print results
    logger.info("")
    logger.info("PRICE FORECAST RESULTS (MAE in c/kWh)")
    logger.info("-" * 65)
    logger.info("%-12s %12s %12s %12s", "Horizon", "Heuristic", "ML Model", "Improvement")
    logger.info("-" * 65)

    for h in forecast_horizons:
        h_mae = np.mean(heuristic_errors[h]) if heuristic_errors[h] else 0
        m_mae = np.mean(ml_errors[h]) if ml_errors[h] else 0
        improvement = (1 - m_mae / max(h_mae, 0.001)) * 100
        logger.info(
            "%-12s %10.2f c %10.2f c %10.1f%%",
            f"{h}h ahead", h_mae, m_mae, improvement,
        )
        # Also show RMSE
    logger.info("-" * 65)

    # Overall
    all_h = sum(len(v) for v in heuristic_errors.values())
    all_m = sum(len(v) for v in ml_errors.values())
    h_all_mae = np.mean([e for errs in heuristic_errors.values() for e in errs]) if all_h else 0
    m_all_mae = np.mean([e for errs in ml_errors.values() for e in errs]) if all_m else 0
    logger.info("%-12s %10.2f c %10.2f c %10.1f%%",
                "Overall", h_all_mae, m_all_mae,
                (1 - m_all_mae / max(h_all_mae, 0.001)) * 100)
    logger.info("=" * 70)

    return heuristic_errors, ml_errors


def _train_price_model(
    prices: dict[str, float],
    sorted_keys: list[str],
    temps: dict[str, float],
    current_time: datetime,
    window_days: int,
    heuristic: "PriceForecaster",
):
    """Train a GBR residual-correction model on recent price forecast errors.

    Instead of predicting absolute prices, trains to predict:
        correction = actual_price - heuristic_forecast

    At inference: ml_prediction = heuristic_forecast + model.predict(features)
    This way the ML only needs to learn the *systematic bias* in the heuristic.
    """
    from sklearn.ensemble import GradientBoostingRegressor

    window_start = current_time - timedelta(days=window_days)
    start_key = window_start.strftime("%Y/%m/%d %H:%M")
    end_key = current_time.strftime("%Y/%m/%d %H:%M")

    import bisect
    lo = bisect.bisect_left(sorted_keys, start_key)
    hi = bisect.bisect_left(sorted_keys, end_key)

    if hi - lo < 7 * 288:  # need at least 7 days
        return None

    # Build a fast median profile from the first 2/3 of the window
    # (mimics what the heuristic forecaster does internally).
    # This avoids calling heuristic.forecast() thousands of times.
    profile_end_idx = lo + (hi - lo) * 2 // 3
    profile_buckets = defaultdict(list)
    for idx in range(lo, profile_end_idx):
        key = sorted_keys[idx]
        ts = _parse_aemo_key(key)
        day_type = "weekend" if ts.weekday() >= 5 else "weekday"
        slot = ts.hour * 2 + (1 if ts.minute >= 30 else 0)
        profile_buckets[(day_type, slot)].append(prices[key])

    profile = {}
    for k, vals in profile_buckets.items():
        vals.sort()
        profile[k] = vals[len(vals) // 2]  # median

    X, y = [], []

    # Train on last 1/3 of window using the profile as the "forecast"
    for idx in range(profile_end_idx, hi, 6):
        key = sorted_keys[idx]
        actual_price = prices[key]
        ts = _parse_aemo_key(key)

        day_type = "weekend" if ts.weekday() >= 5 else "weekday"
        slot = ts.hour * 2 + (1 if ts.minute >= 30 else 0)
        heuristic_pred = profile.get((day_type, slot), 8.0)

        # Train with various simulated lead times
        for lead_h in [1, 3, 6, 12, 24]:
            residual = actual_price - heuristic_pred
            features = _build_price_features(ts, heuristic_pred, lead_h, temps)
            X.append(features_to_vector(features, PRICE_FEATURE_NAMES))
            y.append(residual)

    if len(X) < 500:
        return None

    X_arr = np.array(X)
    y_arr = np.array(y)

    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_leaf=20,
        random_state=42,
    )
    model.fit(X_arr, y_arr)
    return model


def _build_price_features(
    ts: datetime, raw_price: float, lead_time_hours: float,
    temps: dict[str, float],
) -> dict:
    """Build full feature dict for a price prediction."""
    features = {}
    features.update(extract_time_features(ts))
    features.update(extract_daylight_features(ts, LATITUDE, LONGITUDE))
    features["temperature_c"] = get_temperature(temps, ts)
    features.update(extract_price_features(raw_price, lead_time_hours))
    return features


def _parse_aemo_key(key: str) -> datetime:
    """Parse "YYYY/MM/DD HH:MM" to datetime."""
    return datetime(
        int(key[:4]), int(key[5:7]), int(key[8:10]),
        int(key[11:13]), int(key[14:16]),
    )


# ---------------------------------------------------------------------------
# Consumption forecast evaluation
# ---------------------------------------------------------------------------

def evaluate_consumption_forecasting(
    home_usage: dict[str, float],
    temps: dict[str, float],
    train_days: int = 28,
    test_start_date: str = "2024-07-01",
    test_end_date: str = "2025-11-30",
):
    """Compare ML consumption forecaster vs P75 profile heuristic.

    Walks through the test period day by day. For each day, builds a profile
    from the preceding `train_days`, then forecasts and compares to actuals.
    """
    from sklearn.ensemble import GradientBoostingRegressor

    logger.info("")
    logger.info("=" * 70)
    logger.info("CONSUMPTION FORECAST EVALUATION")
    logger.info("=" * 70)

    test_start = datetime.strptime(test_start_date, "%Y-%m-%d")
    test_end = datetime.strptime(test_end_date, "%Y-%m-%d")

    # Sort usage by timestamp
    sorted_usage = sorted(home_usage.items(), key=lambda x: x[0])

    # Results collectors
    heuristic_errors = []
    ml_errors = []
    heuristic_errors_by_hour = defaultdict(list)
    ml_errors_by_hour = defaultdict(list)

    ml_model = None
    last_train_week = None

    current = test_start
    total_days = (test_end - test_start).days
    run_start = time.perf_counter()

    while current <= test_end:
        date_str = current.strftime("%Y-%m-%d")
        day_num = (current - test_start).days

        if day_num % 50 == 0:
            elapsed = time.perf_counter() - run_start
            pct = day_num / max(total_days, 1) * 100
            logger.info("  [%5.1f%%] Processing %s (day %d/%d, %.1fs elapsed)",
                        pct, date_str, day_num, total_days, elapsed)

        # Build P75 profile from preceding train_days (heuristic method)
        window_start = current - timedelta(days=train_days)
        profile = _build_consumption_profile(home_usage, window_start, current)

        # Retrain ML model weekly
        train_week = f"{current.year}-W{current.isocalendar()[1]:02d}"
        if ml_model is None or last_train_week != train_week:
            ml_model = _train_consumption_model(
                home_usage, temps, profile, current, train_days
            )
            last_train_week = train_week

        # Evaluate: for each half-hour slot in this day, compare forecast vs actual
        recent_loads = []  # rolling recent loads for ML features

        for hh in range(48):
            h = hh // 2
            m = (hh % 2) * 30
            ts_key = f"{date_str}T{h:02d}:{m:02d}:00"
            ts = current.replace(hour=h, minute=m)

            actual_kw = home_usage.get(ts_key)
            if actual_kw is None:
                continue

            # Heuristic: P75 profile
            day_type = "weekend" if ts.weekday() >= 5 else "weekday"
            profile_kw = profile.get((day_type, hh), 1.5)
            heuristic_errors.append(abs(profile_kw - actual_kw))
            heuristic_errors_by_hour[h].append(abs(profile_kw - actual_kw))

            # ML forecast
            if ml_model is not None:
                features = _build_consumption_features(
                    ts, temps, recent_loads, profile_kw, occupancy=True
                )
                vec = np.array([features_to_vector(features, CONSUMPTION_FEATURE_NAMES)])
                ml_pred = float(ml_model.predict(vec)[0])
                ml_pred = max(0.05, min(25.0, ml_pred))
                ml_errors.append(abs(ml_pred - actual_kw))
                ml_errors_by_hour[h].append(abs(ml_pred - actual_kw))

            # Update rolling recent loads (for next slot's features)
            # Expand 30-min to approximate 5-min readings
            for _ in range(6):
                recent_loads.append(actual_kw)
            if len(recent_loads) > 36:
                recent_loads = recent_loads[-36:]

        current += timedelta(days=1)

    # Print results
    logger.info("")
    logger.info("CONSUMPTION FORECAST RESULTS (MAE in kW)")
    logger.info("-" * 65)

    h_mae = np.mean(heuristic_errors) if heuristic_errors else 0
    m_mae = np.mean(ml_errors) if ml_errors else 0
    improvement = (1 - m_mae / max(h_mae, 0.001)) * 100

    logger.info("%-20s %12s %12s %12s", "Metric", "Heuristic", "ML Model", "Improvement")
    logger.info("-" * 65)
    logger.info("%-20s %10.3f kW %10.3f kW %10.1f%%", "Overall MAE", h_mae, m_mae, improvement)

    h_rmse = np.sqrt(np.mean(np.array(heuristic_errors) ** 2)) if heuristic_errors else 0
    m_rmse = np.sqrt(np.mean(np.array(ml_errors) ** 2)) if ml_errors else 0
    logger.info("%-20s %10.3f kW %10.3f kW %10.1f%%", "Overall RMSE",
                h_rmse, m_rmse, (1 - m_rmse / max(h_rmse, 0.001)) * 100)

    # By time of day (grouped into periods)
    logger.info("")
    logger.info("BY TIME OF DAY:")
    logger.info("-" * 65)
    logger.info("%-12s %12s %12s %12s", "Period", "Heuristic", "ML Model", "Improvement")
    logger.info("-" * 65)

    period_labels = [
        ("Night 0-6", range(0, 6)),
        ("Morning 6-9", range(6, 9)),
        ("Midday 9-15", range(9, 15)),
        ("Afternoon 15-18", range(15, 18)),
        ("Evening 18-22", range(18, 22)),
        ("Late 22-24", range(22, 24)),
    ]

    for label, hours in period_labels:
        h_errs = [e for hr in hours for e in heuristic_errors_by_hour.get(hr, [])]
        m_errs = [e for hr in hours for e in ml_errors_by_hour.get(hr, [])]
        if h_errs and m_errs:
            h_v = np.mean(h_errs)
            m_v = np.mean(m_errs)
            imp = (1 - m_v / max(h_v, 0.001)) * 100
            logger.info("%-12s %10.3f kW %10.3f kW %10.1f%%", label, h_v, m_v, imp)

    logger.info("=" * 70)

    return heuristic_errors, ml_errors


def _build_consumption_profile(
    usage: dict[str, float],
    window_start: datetime,
    window_end: datetime,
) -> dict[tuple[str, int], float]:
    """Build P75 profile from usage data in window (mimics ConsumptionForecaster)."""
    buckets = defaultdict(list)

    for ts_str, kw in usage.items():
        try:
            ts = datetime.fromisoformat(ts_str)
        except ValueError:
            continue
        if ts < window_start or ts >= window_end:
            continue
        day_type = "weekend" if ts.weekday() >= 5 else "weekday"
        slot = ts.hour * 2 + (1 if ts.minute >= 30 else 0)
        buckets[(day_type, slot)].append(kw)

    profile = {}
    for key, values in buckets.items():
        values.sort()
        p75_idx = int(len(values) * 0.75)
        profile[key] = values[min(p75_idx, len(values) - 1)]

    return profile


def _train_consumption_model(
    usage: dict[str, float],
    temps: dict[str, float],
    profile: dict[tuple[str, int], float],
    current_time: datetime,
    window_days: int,
):
    """Train GBR on recent consumption data."""
    from sklearn.ensemble import GradientBoostingRegressor

    window_start = current_time - timedelta(days=window_days)

    X, y = [], []
    recent_loads = []

    # Iterate through usage in time order within the window
    window_usage = []
    for ts_str, kw in usage.items():
        try:
            ts = datetime.fromisoformat(ts_str)
        except ValueError:
            continue
        if window_start <= ts < current_time:
            window_usage.append((ts, kw))

    window_usage.sort(key=lambda x: x[0])

    for ts, actual_kw in window_usage:
        day_type = "weekend" if ts.weekday() >= 5 else "weekday"
        slot = ts.hour * 2 + (1 if ts.minute >= 30 else 0)
        profile_kw = profile.get((day_type, slot), 1.5)

        features = _build_consumption_features(
            ts, temps, recent_loads, profile_kw, occupancy=True
        )
        X.append(features_to_vector(features, CONSUMPTION_FEATURE_NAMES))
        y.append(actual_kw)

        # Update rolling window (expand 30-min to 5-min approximation)
        for _ in range(6):
            recent_loads.append(actual_kw)
        if len(recent_loads) > 36:
            recent_loads = recent_loads[-36:]

    if len(X) < 200:
        return None

    X_arr = np.array(X)
    y_arr = np.array(y)

    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=42,
    )
    model.fit(X_arr, y_arr)
    return model


def _build_consumption_features(
    ts: datetime,
    temps: dict[str, float],
    recent_loads_kw: list[float],
    profile_load_kw: float,
    occupancy: bool = True,
) -> dict:
    """Build full feature dict for a consumption prediction."""
    features = {}
    features.update(extract_time_features(ts))
    features.update(extract_daylight_features(ts, LATITUDE, LONGITUDE))
    features["temperature_c"] = get_temperature(temps, ts)
    features.update(extract_consumption_features(
        occupancy=occupancy,
        recent_loads_kw=recent_loads_kw,
        profile_load_kw=profile_load_kw,
        ac_threshold_kw=2.0,
    ))
    return features


# ---------------------------------------------------------------------------
# Feature importance analysis
# ---------------------------------------------------------------------------

def print_feature_importance(model, feature_names: list[str], title: str):
    """Print feature importance from a trained GBR model."""
    if model is None:
        logger.info("No model to analyze for %s", title)
        return

    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    logger.info("")
    logger.info("FEATURE IMPORTANCE: %s", title)
    logger.info("-" * 45)
    for i in sorted_idx:
        bar = "#" * int(importances[i] * 50)
        logger.info("  %-25s %5.1f%% %s", feature_names[i], importances[i] * 100, bar)
    logger.info("")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("Loading test data...")
    aemo = load_aemo_prices()
    usage = load_home_usage()

    # Determine date range overlap between AEMO and home usage
    aemo_dates = sorted(set(k[:10] for k in aemo.keys()))
    usage_dates = sorted(set(k[:10] for k in usage.keys()))
    logger.info("AEMO data: %s to %s (%d days)", aemo_dates[0], aemo_dates[-1], len(aemo_dates))
    logger.info("Usage data: %s to %s (%d days)",
                usage_dates[0].replace("T", " ")[:10],
                usage_dates[-1].replace("T", " ")[:10],
                len(usage_dates))

    # Convert usage date format for comparison
    usage_date_set = set(d[:10] for d in usage_dates)
    aemo_date_set = set(d.replace("/", "-") for d in aemo_dates)
    overlap_dates = sorted(usage_date_set & aemo_date_set)
    logger.info("Overlapping dates: %d days (%s to %s)",
                len(overlap_dates), overlap_dates[0], overlap_dates[-1])

    # Use overlap range minus 60-day warmup for testing
    warmup_days = 60
    if len(overlap_dates) <= warmup_days + 30:
        logger.error("Insufficient overlapping data for meaningful evaluation")
        return

    test_start = overlap_dates[warmup_days]
    test_end = overlap_dates[-1]

    # Fetch historical temperatures
    # Use AEMO full range for price eval, overlap range for consumption
    aemo_start_iso = aemo_dates[0].replace("/", "-")
    aemo_end_iso = aemo_dates[-1].replace("/", "-")
    temps = fetch_historical_temperatures(aemo_start_iso, aemo_end_iso)

    # Run price evaluation
    price_test_start = "2024/07/01"  # 6 months warm-up from AEMO start
    price_test_end = aemo_dates[-2]  # second to last for full day
    h_price_errs, m_price_errs = evaluate_price_forecasting(
        aemo, temps,
        train_days=60,
        test_start_date=price_test_start,
        test_end_date=price_test_end,
    )

    # Run consumption evaluation
    h_cons_errs, m_cons_errs = evaluate_consumption_forecasting(
        usage, temps,
        train_days=28,
        test_start_date=test_start,
        test_end_date=test_end,
    )

    # Train final models and show feature importance
    logger.info("")
    logger.info("=" * 70)
    logger.info("TRAINING FINAL MODELS FOR FEATURE IMPORTANCE ANALYSIS")
    logger.info("=" * 70)

    # Final price model (trained on last 60 days of data)
    sorted_keys = sorted(aemo.keys())
    final_heuristic = PriceForecaster(aemo, import_markup_c=IMPORT_MARKUP_C, window_days=21)
    final_price_model = _train_price_model(
        aemo, sorted_keys, temps,
        _parse_aemo_key(sorted_keys[-1]),
        60, final_heuristic,
    )
    print_feature_importance(final_price_model, PRICE_FEATURE_NAMES, "Price Correction")

    # Final consumption model
    profile = _build_consumption_profile(
        usage,
        datetime.fromisoformat(overlap_dates[-29]),
        datetime.fromisoformat(overlap_dates[-1]),
    )
    final_cons_model = _train_consumption_model(
        usage, temps, profile,
        datetime.fromisoformat(overlap_dates[-1]),
        28,
    )
    print_feature_importance(final_cons_model, CONSUMPTION_FEATURE_NAMES, "Consumption")

    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info("")

    if m_price_errs and h_price_errs:
        all_h = [e for errs in h_price_errs.values() for e in errs]
        all_m = [e for errs in m_price_errs.values() for e in errs]
        if all_h and all_m:
            logger.info("Price forecasting:  ML MAE %.2f c/kWh vs Heuristic %.2f c/kWh (%.1f%% better)",
                        np.mean(all_m), np.mean(all_h),
                        (1 - np.mean(all_m) / max(np.mean(all_h), 0.001)) * 100)

    if m_cons_errs and h_cons_errs:
        logger.info("Consumption:        ML MAE %.3f kW vs Heuristic %.3f kW (%.1f%% better)",
                    np.mean(m_cons_errs), np.mean(h_cons_errs),
                    (1 - np.mean(m_cons_errs) / max(np.mean(h_cons_errs), 0.001)) * 100)

    logger.info("")
    logger.info("Done!")


if __name__ == "__main__":
    main()
