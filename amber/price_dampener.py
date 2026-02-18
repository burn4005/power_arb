import logging
import math
from dataclasses import dataclass
from datetime import datetime

from amber.client import PriceInterval
from storage.database import Database

logger = logging.getLogger(__name__)

# Default median prices (cents/kWh) by time-of-day bucket for QLD.
# These are starting estimates; replaced by empirical data once available.
_DEFAULT_TOD_MEDIANS = {
    # hour -> (import_median, export_median)
    0: (15.0, 5.0), 1: (13.0, 4.0), 2: (12.0, 3.5), 3: (12.0, 3.5),
    4: (13.0, 4.0), 5: (15.0, 5.0), 6: (20.0, 10.0), 7: (25.0, 12.0),
    8: (22.0, 10.0), 9: (18.0, 8.0), 10: (12.0, 5.0), 11: (8.0, 3.0),
    12: (6.0, 2.0), 13: (5.0, 1.5), 14: (6.0, 2.0), 15: (10.0, 4.0),
    16: (18.0, 8.0), 17: (30.0, 18.0), 18: (35.0, 22.0), 19: (32.0, 20.0),
    20: (28.0, 15.0), 21: (22.0, 10.0), 22: (18.0, 8.0), 23: (15.0, 6.0),
}


@dataclass
class DampenedPrice:
    timestamp: str
    raw_per_kwh: float
    dampened_per_kwh: float
    confidence: float
    channel: str
    spike_status: str
    lead_time_minutes: int


class PriceDampener:
    """Adjusts Amber forecast prices to reduce extreme forecast bias.

    Amber passes through AEMO pre-dispatch pricing which can show extreme
    spikes ($20/kWh) that rarely materialize (often settling at $0.60/kWh).
    Without dampening, the optimizer would waste battery cycles chasing
    phantom events.

    Strategy:
    - Short term (< 2h): trust forecast, high confidence
    - Medium term (2-12h): log-dampen extremes toward time-of-day median
    - Long term (> 12h): blend forecast with historical patterns
    - Never dampen negative prices (those are real)
    - Self-calibrate from forecast_accuracy table when data is available
    """

    def __init__(self, db: Database):
        self.db = db
        self._calibrated_alphas: dict[int, float] | None = None
        self._tod_medians = dict(_DEFAULT_TOD_MEDIANS)
        self._calibrate()

    def _calibrate(self):
        """Update dampening parameters from historical forecast accuracy data."""
        records = self.db.get_forecast_accuracy(channel="import", days=30)
        if len(records) < 100:
            logger.info(
                "Only %d accuracy records; using default dampening params", len(records)
            )
            return

        # Group by lead-time bucket and compute median actual/forecast ratio
        buckets: dict[int, list[float]] = {}
        for r in records:
            if r["actual_price"] is None or r["forecast_price"] == 0:
                continue
            lead_h = r["lead_time_minutes"] // 60
            bucket = min(lead_h, 48)
            ratio = r["actual_price"] / r["forecast_price"]
            buckets.setdefault(bucket, []).append(ratio)

        self._calibrated_alphas = {}
        for bucket, ratios in buckets.items():
            ratios.sort()
            median_ratio = ratios[len(ratios) // 2]
            # Alpha represents how much of the forecast to keep
            # If forecasts are accurate, ratio ~ 1.0 and alpha ~ 1.0
            # If forecasts overshoot, ratio < 1.0 and alpha < 1.0
            self._calibrated_alphas[bucket] = max(0.1, min(1.0, median_ratio))

        logger.info("Calibrated dampening from %d records across %d buckets",
                     len(records), len(self._calibrated_alphas))

        # Update time-of-day medians from actuals (both import and export)
        export_records = self.db.get_forecast_accuracy(channel="export", days=30)

        tod_import: dict[int, list[float]] = {}
        for r in records:
            if r["actual_price"] is None:
                continue
            hour = datetime.fromisoformat(r["target_time"]).hour
            tod_import.setdefault(hour, []).append(r["actual_price"])

        tod_export: dict[int, list[float]] = {}
        for r in export_records:
            if r["actual_price"] is None:
                continue
            hour = datetime.fromisoformat(r["target_time"]).hour
            tod_export.setdefault(hour, []).append(r["actual_price"])

        for hour in range(24):
            default = self._tod_medians.get(hour, (15.0, 5.0))
            import_vals = tod_import.get(hour)
            export_vals = tod_export.get(hour)
            import_med = sorted(import_vals)[len(import_vals) // 2] if import_vals else default[0]
            export_med = sorted(export_vals)[len(export_vals) // 2] if export_vals else default[1]
            self._tod_medians[hour] = (import_med, export_med)

    def dampen(
        self, prices: list[PriceInterval], reference_time: datetime | None = None
    ) -> list[DampenedPrice]:
        """Apply dampening to a list of forecast price intervals."""
        now = reference_time or datetime.now()
        results = []

        for p in prices:
            ts = datetime.fromisoformat(p.timestamp)
            lead_minutes = max(0, int((ts - now).total_seconds() / 60))
            lead_hours = lead_minutes / 60

            dampened, confidence = self._dampen_single(
                p.per_kwh, lead_hours, ts.hour, p.spike_status, p.channel
            )

            results.append(DampenedPrice(
                timestamp=p.timestamp,
                raw_per_kwh=p.per_kwh,
                dampened_per_kwh=dampened,
                confidence=confidence,
                channel=p.channel,
                spike_status=p.spike_status,
                lead_time_minutes=lead_minutes,
            ))

        return results

    def _dampen_single(
        self, price: float, lead_hours: float, hour: int,
        spike_status: str, channel: str
    ) -> tuple[float, float]:
        """Dampen a single price value. Returns (dampened_price, confidence)."""
        median_idx = 0 if channel == "import" else 1
        median = self._tod_medians.get(hour, (15.0, 5.0))[median_idx]

        # Never dampen negative prices -- they're real and reliable
        if price < 0:
            return price, 0.95

        # Short term: trust the forecast
        if lead_hours < 2:
            confidence = 0.9
            if spike_status == "spike":
                return price * 0.85, confidence
            return price, confidence

        # Get calibrated alpha if available
        alpha = self._get_alpha(lead_hours)

        # Medium term: dampen toward median
        if lead_hours < 12:
            confidence = max(0.3, 0.6 - lead_hours * 0.03)

            if spike_status == "spike":
                dampened = median + (price - median) * alpha * 0.7
            elif spike_status == "potential":
                dampened = median + (price - median) * alpha * 0.4
            else:
                dampened = median + (price - median) * alpha

            # Extra compression for extreme values using log dampening
            if dampened > median * 5:
                excess = dampened - median
                dampened = median + math.log1p(excess) * median
            return max(0, dampened), confidence

        # Long term: mostly historical with light forecast influence
        confidence = max(0.15, 0.25 - (lead_hours - 12) * 0.003)
        dampened = 0.3 * price + 0.7 * median

        if spike_status in ("spike", "potential"):
            dampened = 0.2 * price + 0.8 * median

        return max(0, dampened), confidence

    def _get_alpha(self, lead_hours: float) -> float:
        """Get dampening alpha for a given lead time."""
        if self._calibrated_alphas:
            bucket = min(int(lead_hours), 48)
            if bucket in self._calibrated_alphas:
                return self._calibrated_alphas[bucket]

        # Default: exponential decay from 0.5
        return 0.5 * math.exp(-0.1 * max(0, lead_hours - 2))
