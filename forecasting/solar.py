import logging
from datetime import datetime, timedelta

from optimizer.battery_model import PERIOD_HOURS, PERIOD_MINUTES, PERIODS_PER_HOUR
from solcast.client import SolcastClient

logger = logging.getLogger(__name__)


class SolarForecaster:
    """Manages solar production forecast aligned to 5-min optimizer periods.

    Wraps SolcastClient and provides forecasts in a format the optimizer expects.
    Solcast returns 30-min data which is flat-repeated into 5-min periods.
    Uses P50 estimates by default, with P10 (pessimistic) available for
    conservative planning.
    """

    def __init__(self, solcast_client: SolcastClient):
        self.solcast = solcast_client

    def forecast(
        self, hours: int = 48, start: datetime | None = None, conservative: bool = False
    ) -> list[dict]:
        """Get solar forecast aligned to 5-min periods.

        Args:
            hours: forecast horizon
            start: start time (default: now, rounded to 5-min)
            conservative: if True, use P10 (pessimistic) instead of P50

        Returns list of dicts: timestamp, solar_kw, solar_kwh (per 5-min).
        """
        raw = self.solcast.get_forecast()
        if not raw:
            logger.warning("No solar forecast available; assuming 0 production")
            return self._zero_forecast(hours, start)

        start = start or datetime.now()
        # Round start down to 5-min boundary
        start = start.replace(minute=(start.minute // 5) * 5, second=0, microsecond=0)

        # Build lookup by half-hour period start
        forecast_map: dict[str, dict] = {}
        for entry in raw:
            # period_end is UTC, convert to AEST (UTC+10)
            pe_str = entry["period_end"]
            if pe_str.endswith("Z"):
                pe_str = pe_str[:-1] + "+00:00"
            pe_utc = datetime.fromisoformat(pe_str)
            pe_local = pe_utc + timedelta(hours=10)  # QLD is AEST, no DST
            # period_end is end of interval; start = end - 30min
            period_start = pe_local - timedelta(minutes=30)
            key = period_start.strftime("%Y-%m-%dT%H:%M")
            forecast_map[key] = entry

        periods = hours * PERIODS_PER_HOUR
        results = []
        for i in range(periods):
            ts = start + timedelta(minutes=PERIOD_MINUTES * i)
            # Look up the half-hour window this 5-min period falls in
            hh_start = ts.replace(minute=(ts.minute // 30) * 30)
            key = hh_start.strftime("%Y-%m-%dT%H:%M")
            entry = forecast_map.get(key)
            if entry:
                if conservative and entry.get("pv_estimate10_kw") is not None:
                    solar_kw = entry["pv_estimate10_kw"]
                else:
                    solar_kw = entry["pv_estimate_kw"]
                solar_p10 = entry.get("pv_estimate10_kw") or 0.0
                solar_p90 = entry.get("pv_estimate90_kw") or 0.0
            else:
                solar_kw = 0.0
                solar_p10 = 0.0
                solar_p90 = 0.0

            results.append({
                "timestamp": ts.isoformat(),
                "solar_kw": max(0.0, solar_kw),
                "solar_p10_kw": max(0.0, solar_p10),
                "solar_p90_kw": max(0.0, solar_p90),
                "solar_kwh": max(0.0, solar_kw) * PERIOD_HOURS,
            })

        return results

    def _zero_forecast(self, hours: int, start: datetime | None) -> list[dict]:
        start = start or datetime.now()
        start = start.replace(minute=(start.minute // 5) * 5, second=0, microsecond=0)

        return [
            {
                "timestamp": (start + timedelta(minutes=PERIOD_MINUTES * i)).isoformat(),
                "solar_kw": 0.0,
                "solar_p10_kw": 0.0,
                "solar_p90_kw": 0.0,
                "solar_kwh": 0.0,
            }
            for i in range(hours * PERIODS_PER_HOUR)
        ]
