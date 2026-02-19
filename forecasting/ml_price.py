"""ML-based price correction model wrapping PriceDampener as fallback.

Trains a GradientBoostingRegressor to predict residual corrections on top
of the existing dampened price forecast. The model learns systematic biases
in the Amber forecast (spike overestimation, weather effects, etc.) using
features like temperature, daylight hours, and time-of-day patterns.

Falls back to PriceDampener when insufficient training data exists or
when scikit-learn is not installed.
"""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np

import config
from amber.client import PriceInterval
from amber.price_dampener import PriceDampener, DampenedPrice
from forecasting.features import (
    extract_time_features, extract_daylight_features, extract_price_features,
    PRICE_FEATURE_NAMES, features_to_vector,
)
from storage.database import Database

logger = logging.getLogger(__name__)

_MIN_PRICE_C = -100.0
_MAX_PRICE_C = 1500.0


class MLPriceForecaster:
    """GradientBoosting price correction that falls back to PriceDampener."""

    def __init__(self, db: Database, dampener: PriceDampener):
        self.db = db
        self.dampener = dampener
        self._model = None
        self._model_trained = False
        self._model_dir = Path(config.ml.model_dir)
        self._model_path = self._model_dir / "price_model.joblib"
        self._load_model()

    def _load_model(self):
        """Load persisted model from disk if available."""
        if self._model_path.exists():
            try:
                import joblib
                self._model = joblib.load(self._model_path)
                self._model_trained = True
                logger.info("Loaded price ML model from %s", self._model_path)
            except Exception as e:
                logger.warning("Failed to load price model: %s", e)

    def train(self):
        """Retrain model on historical forecast accuracy data. Called daily."""
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            import joblib
        except ImportError:
            logger.error("scikit-learn not installed; skipping ML price training")
            return

        accuracy = self.db.get_forecast_accuracy(
            channel="import", days=config.ml.price_training_window_days
        )
        snapshots = self.db.get_feature_snapshots(
            "price", days=config.ml.price_training_window_days
        )

        snap_by_ts = {s["timestamp"]: s["features"] for s in snapshots}

        X, y = [], []
        for record in accuracy:
            if record["actual_price"] is None or record["forecast_price"] == 0:
                continue
            ts = record["target_time"]
            features = snap_by_ts.get(ts)
            if features is None:
                continue
            # Train to predict residual: actual - dampened_forecast
            residual = record["actual_price"] - features.get("raw_price", record["forecast_price"])
            vec = features_to_vector(features, PRICE_FEATURE_NAMES)
            X.append(vec)
            y.append(residual)

        min_samples = config.ml.min_price_training_days * 24 * 12
        if len(X) < min_samples:
            logger.info(
                "Only %d price training samples (need %d); keeping fallback",
                len(X), min_samples,
            )
            return

        X_arr = np.array(X)
        y_arr = np.array(y)

        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=20,
            random_state=42,
        )
        model.fit(X_arr, y_arr)

        # Validate on last 20%
        split = int(len(X_arr) * 0.8)
        if split > 0:
            val_pred = model.predict(X_arr[split:])
            # ML prediction = raw_price + correction
            raw_idx = PRICE_FEATURE_NAMES.index("raw_price")
            ml_pred = X_arr[split:, raw_idx] + val_pred
            actual = X_arr[split:, raw_idx] + y_arr[split:]
            ml_rmse = np.sqrt(np.mean((ml_pred - actual) ** 2))
            # Baseline: just use raw_price (dampened forecast)
            baseline_rmse = np.sqrt(np.mean(y_arr[split:] ** 2))
            logger.info(
                "Price ML RMSE: %.2f c/kWh (dampener baseline RMSE: %.2f c/kWh, "
                "improvement: %.1f%%)",
                ml_rmse, baseline_rmse,
                (1 - ml_rmse / max(baseline_rmse, 0.01)) * 100,
            )
            if ml_rmse >= baseline_rmse * 1.05:
                logger.warning("ML price model not better than dampener; keeping fallback")
                return

        self._model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, self._model_path)
        self._model = model
        self._model_trained = True
        logger.info("Price ML model trained on %d samples, saved to %s",
                     len(X), self._model_path)

    def dampen(
        self, prices: list[PriceInterval], reference_time: datetime | None = None
    ) -> list[DampenedPrice]:
        """Apply ML correction or fall back to PriceDampener."""
        # Always get heuristic results first (needed as fallback and for blending)
        heuristic_results = self.dampener.dampen(prices, reference_time)

        if not config.ml.enabled or not self._model_trained or self._model is None:
            return heuristic_results

        now = reference_time or datetime.now()
        results = []

        for i, p in enumerate(prices):
            heuristic = heuristic_results[i]
            ts = datetime.fromisoformat(p.timestamp)
            lead_minutes = max(0, int((ts - now).total_seconds() / 60))
            lead_hours = lead_minutes / 60

            # Short-term (< 30 min): trust dampener, skip ML
            if lead_hours < 0.5:
                results.append(heuristic)
                continue

            # Never correct negative prices
            if p.per_kwh < 0:
                results.append(heuristic)
                continue

            # Build features and predict correction
            features = self._build_features(ts, heuristic.dampened_per_kwh, lead_hours, p.spike_status)
            vec = np.array([features_to_vector(features, PRICE_FEATURE_NAMES)])
            correction = float(self._model.predict(vec)[0])
            ml_pred = heuristic.dampened_per_kwh + correction
            ml_pred = max(_MIN_PRICE_C, min(_MAX_PRICE_C, ml_pred))

            # Blend ML + heuristic based on lead time
            ml_weight = min(0.8, max(0.3, 1.0 - lead_hours / 48.0))
            blended = ml_weight * ml_pred + (1 - ml_weight) * heuristic.dampened_per_kwh

            results.append(DampenedPrice(
                timestamp=p.timestamp,
                raw_per_kwh=p.per_kwh,
                dampened_per_kwh=blended,
                confidence=heuristic.confidence,
                channel=p.channel,
                spike_status=p.spike_status,
                lead_time_minutes=lead_minutes,
            ))

        return results

    def snapshot_features(self, prices: list[PriceInterval], reference_time: datetime):
        """Save feature snapshots for future training. Called each cycle."""
        now = reference_time
        for p in prices:
            if p.forecast_type != "forecast":
                continue
            ts = datetime.fromisoformat(p.timestamp)
            lead_hours = max(0, (ts - now).total_seconds() / 3600)
            features = self._build_features(ts, p.per_kwh, lead_hours, p.spike_status)
            self.db.insert_feature_snapshot(p.timestamp, "price", features)

    def _build_features(
        self, ts: datetime, raw_price: float, lead_hours: float, spike_status: str
    ) -> dict:
        """Build full feature dict for a price prediction."""
        features = {}
        features.update(extract_time_features(ts))
        features.update(extract_daylight_features(
            ts, config.weather.latitude, config.weather.longitude
        ))
        weather = self.db.get_weather_at(ts.isoformat())
        features["temperature_c"] = weather["temperature_c"] if weather else 20.0
        if weather:
            features.update(extract_daylight_features(
                ts, config.weather.latitude, config.weather.longitude,
                weather.get("sunrise"), weather.get("sunset"),
            ))
        features.update(extract_price_features(raw_price, lead_hours, spike_status))
        return features
