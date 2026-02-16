import logging
from collections import defaultdict
from datetime import datetime, timedelta

from optimizer.battery_model import PERIOD_HOURS, PERIOD_MINUTES, PERIODS_PER_HOUR
from storage.database import Database

logger = logging.getLogger(__name__)

# Default flat consumption estimate (kW) used when insufficient history exists.
_DEFAULT_LOAD_KW = 1.5


class ConsumptionForecaster:
    """Predicts home power consumption for the next 48 hours.

    Uses historical load data grouped by day-type (weekday/weekend) and
    half-hour slot to build a consumption profile. Returns P75 estimates
    (conservative -- better to over-predict so we don't run out of battery).

    Output is at 5-min resolution; each half-hour profile value is
    flat-repeated across the six 5-min periods within that slot.

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
        (per 5-min interval), in 5-minute steps.
        """
        start = start or datetime.now()
        # Round down to 5-min boundary
        start = start.replace(minute=(start.minute // 5) * 5, second=0, microsecond=0)

        periods = hours * PERIODS_PER_HOUR
        results = []

        for i in range(periods):
            ts = start + timedelta(minutes=PERIOD_MINUTES * i)
            load_kw = self._predict_slot(ts)
            results.append({
                "timestamp": ts.isoformat(),
                "load_kw": load_kw,
                "load_kwh": load_kw * PERIOD_HOURS,
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
