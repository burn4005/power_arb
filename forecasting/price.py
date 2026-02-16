"""AEMO spot price forecaster for backtesting.

Uses a rolling window of historical prices grouped by (day_type, half_hour_slot)
to produce P50 (median) forecasts. Follows the same pattern as
forecasting/consumption.py.

No future data leakage: at time T, only prices before T are used.
"""

import bisect
from collections import defaultdict
from datetime import datetime, timedelta

from optimizer.battery_model import PERIOD_MINUTES, PERIODS_PER_HOUR

# Default spot price profile (c/kWh) by hour for QLD.
# Used during warm-up when insufficient history exists.
# Derived from typical QLD AEMO patterns: low overnight, solar dip midday,
# evening peak 5-7pm.
_DEFAULT_SPOT_PROFILE_C = {
    0: 6.5, 1: 5.5, 2: 5.0, 3: 5.0, 4: 5.5, 5: 7.0,
    6: 10.0, 7: 12.0, 8: 10.0, 9: 8.0, 10: 5.0, 11: 3.0,
    12: 2.0, 13: 1.5, 14: 2.0, 15: 5.0, 16: 10.0, 17: 18.0,
    18: 22.0, 19: 18.0, 20: 14.0, 21: 10.0, 22: 8.0, 23: 7.0,
}


class PriceForecaster:
    """Predicts AEMO spot prices using a rolling historical profile.

    Groups past prices by (weekday/weekend, half_hour_slot) and returns
    median (P50) as the forecast. Profile is rebuilt once per simulated day.

    Args:
        aemo_prices: Dict of "YYYY/MM/DD HH:MM" -> spot_price_c_kwh.
                     The full historical dataset; the forecaster only looks
                     backward from the query time.
        import_markup_c: Markup added to spot for import price (default 20c/kWh).
        window_days: Rolling window size for building the profile.
    """

    WINDOW_DAYS = 21
    MIN_DAYS = 7

    def __init__(
        self,
        aemo_prices: dict[str, float],
        import_markup_c: float = 20.0,
        window_days: int = 21,
    ):
        self.aemo = aemo_prices
        self.import_markup_c = import_markup_c
        self.window_days = window_days
        self._sorted_keys = sorted(aemo_prices.keys())
        self._cache_date: str | None = None
        self._cached_profile: dict[tuple[str, int], float] | None = None

    def forecast(
        self, current_time: str, hours: int = 48
    ) -> list[dict[str, float | str]]:
        """Generate a price forecast for the next N hours.

        Args:
            current_time: "YYYY/MM/DD HH:MM" format — the simulated 'now'.
            hours: Forecast horizon in hours.

        Returns:
            List of dicts with keys:
                timestamp: "YYYY/MM/DD HH:MM"
                import_c: cents/kWh (spot + markup)
                export_c: cents/kWh (spot)
        """
        date_str = current_time[:10]

        # Rebuild profile once per day
        if self._cache_date != date_str:
            self._cached_profile = self._build_profile(current_time)
            self._cache_date = date_str

        # Parse current time
        year = int(date_str[:4])
        month = int(date_str[5:7])
        day = int(date_str[8:10])
        h = int(current_time[11:13])
        m = int(current_time[14:16])
        dt = datetime(year, month, day, h, m)

        periods = hours * PERIODS_PER_HOUR
        results = []

        for i in range(periods):
            ts = dt + timedelta(minutes=PERIOD_MINUTES * i)
            day_type = "weekend" if ts.weekday() >= 5 else "weekday"
            slot = ts.hour * 2 + (1 if ts.minute >= 30 else 0)

            spot = self._predict_slot(day_type, slot)
            ts_str = ts.strftime("%Y/%m/%d %H:%M")

            results.append({
                "timestamp": ts_str,
                "import_c": spot + self.import_markup_c,
                "export_c": spot,
            })

        return results

    def _build_profile(self, current_time: str) -> dict[tuple[str, int], float] | None:
        """Build price profile from rolling window before current_time.

        Returns dict mapping (day_type, slot) -> median price in c/kWh,
        or None if insufficient data.
        """
        # Find all keys in the window [current - window_days, current)
        date_str = current_time[:10]
        year = int(date_str[:4])
        month = int(date_str[5:7])
        day = int(date_str[8:10])
        end_dt = datetime(year, month, day)
        start_dt = end_dt - timedelta(days=self.window_days)

        start_key = start_dt.strftime("%Y/%m/%d 00:00")
        end_key = current_time

        # Binary search for the window bounds
        lo = bisect.bisect_left(self._sorted_keys, start_key)
        hi = bisect.bisect_left(self._sorted_keys, end_key)

        if hi - lo < self.MIN_DAYS * PERIODS_PER_HOUR * 24:
            return None

        # Group by (day_type, half_hour_slot)
        buckets: dict[tuple[str, int], list[float]] = defaultdict(list)
        for idx in range(lo, hi):
            key = self._sorted_keys[idx]
            price = self.aemo[key]

            # Parse key: "YYYY/MM/DD HH:MM"
            k_year = int(key[:4])
            k_month = int(key[5:7])
            k_day = int(key[8:10])
            k_h = int(key[11:13])
            k_m = int(key[14:16])

            k_dt = datetime(k_year, k_month, k_day)
            day_type = "weekend" if k_dt.weekday() >= 5 else "weekday"
            slot = k_h * 2 + (1 if k_m >= 30 else 0)

            buckets[(day_type, slot)].append(price)

        # Compute median (P50) for each bucket
        profile = {}
        for key, values in buckets.items():
            values.sort()
            mid = len(values) // 2
            if len(values) % 2 == 0:
                profile[key] = (values[mid - 1] + values[mid]) / 2
            else:
                profile[key] = values[mid]

        return profile

    def _predict_slot(self, day_type: str, slot: int) -> float:
        """Predict spot price for a (day_type, slot) using profile or default."""
        if self._cached_profile is not None:
            predicted = self._cached_profile.get((day_type, slot))
            if predicted is not None:
                return predicted

        # Fallback: default time-of-day profile
        hour = slot // 2
        return _DEFAULT_SPOT_PROFILE_C.get(hour, 8.0)
