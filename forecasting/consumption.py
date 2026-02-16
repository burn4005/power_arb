import logging
from collections import defaultdict
from datetime import datetime, timedelta

from storage.database import Database

logger = logging.getLogger(__name__)

# Default flat consumption estimate (kW) used when insufficient history exists.
_DEFAULT_LOAD_KW = 1.5


class ConsumptionForecaster:
    """Predicts home power consumption for the next 48 hours.

    Uses historical load data grouped by day-type (weekday/weekend) and
    half-hour slot to build a consumption profile. Returns P75 estimates
    (conservative -- better to over-predict so we don't run out of battery).

    Until 7 days of data are collected, uses a flat default estimate.
    """

    def __init__(self, db: Database):
        self.db = db
        self._profile: dict[tuple[str, int], float] | None = None
        self._build_profile()

    def _build_profile(self):
        """Build consumption profile from historical data."""
        history = self.db.get_consumption_history(days=28)
        if len(history) < 7 * 24 * 2:  # need ~7 days of 30-min data
            logger.info(
                "Only %d consumption records; using flat estimate of %.1f kW",
                len(history), _DEFAULT_LOAD_KW,
            )
            self._profile = None
            return

        # Group by (day_type, half_hour_slot)
        buckets: dict[tuple[str, int], list[float]] = defaultdict(list)
        for row in history:
            ts = datetime.fromisoformat(row["timestamp"])
            day_type = "weekend" if ts.weekday() >= 5 else "weekday"
            slot = ts.hour * 2 + (1 if ts.minute >= 30 else 0)  # 0-47
            load_kw = row["load_watts"] / 1000.0
            buckets[(day_type, slot)].append(load_kw)

        # Compute P75 for each bucket (conservative)
        self._profile = {}
        for key, values in buckets.items():
            values.sort()
            p75_idx = int(len(values) * 0.75)
            self._profile[key] = values[min(p75_idx, len(values) - 1)]

        logger.info("Built consumption profile from %d records (%d buckets)",
                     len(history), len(self._profile))

    def forecast(self, hours: int = 48, start: datetime | None = None) -> list[dict]:
        """Generate consumption forecast for the next N hours.

        Returns list of dicts with keys: timestamp (ISO), load_kw, load_kwh
        (per 30-min interval), in 30-minute steps.
        """
        start = start or datetime.now()
        # Round to nearest 30-min boundary
        if start.minute < 15:
            start = start.replace(minute=0, second=0, microsecond=0)
        elif start.minute < 45:
            start = start.replace(minute=30, second=0, microsecond=0)
        else:
            start = (start + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)

        periods = hours * 2  # 30-min intervals
        results = []

        for i in range(periods):
            ts = start + timedelta(minutes=30 * i)
            load_kw = self._predict_slot(ts)
            results.append({
                "timestamp": ts.isoformat(),
                "load_kw": load_kw,
                "load_kwh": load_kw * 0.5,  # 30-min interval
            })

        return results

    def _predict_slot(self, ts: datetime) -> float:
        """Predict load for a specific time slot."""
        if self._profile is None:
            return _DEFAULT_LOAD_KW

        day_type = "weekend" if ts.weekday() >= 5 else "weekday"
        slot = ts.hour * 2 + (1 if ts.minute >= 30 else 0)
        return self._profile.get((day_type, slot), _DEFAULT_LOAD_KW)

    def refresh(self):
        """Rebuild profile from latest data. Call periodically (e.g., daily)."""
        self._build_profile()
