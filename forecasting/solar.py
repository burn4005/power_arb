import logging
from datetime import datetime, timedelta

from solcast.client import SolcastClient

logger = logging.getLogger(__name__)


class SolarForecaster:
    """Manages solar production forecast aligned to 30-min optimizer periods.

    Wraps SolcastClient and provides forecasts in a format the optimizer expects.
    Uses P50 estimates by default, with P10 (pessimistic) available for
    conservative planning.
    """

    def __init__(self, solcast_client: SolcastClient):
        self.solcast = solcast_client

    def forecast(
        self, hours: int = 48, start: datetime | None = None, conservative: bool = False
    ) -> list[dict]:
        """Get solar forecast aligned to 30-min periods.

        Args:
            hours: forecast horizon
            start: start time (default: now, rounded to 30-min)
            conservative: if True, use P10 (pessimistic) instead of P50

        Returns list of dicts: timestamp, solar_kw, solar_kwh (per 30-min).
        """
        raw = self.solcast.get_forecast()
        if not raw:
            logger.warning("No solar forecast available; assuming 0 production")
            return self._zero_forecast(hours, start)

        start = start or datetime.now()
        # Round start to 30-min boundary
        if start.minute < 15:
            start = start.replace(minute=0, second=0, microsecond=0)
        elif start.minute < 45:
            start = start.replace(minute=30, second=0, microsecond=0)
        else:
            start = (start + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)

        # Build lookup by period_end (Solcast uses UTC, we work in local)
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

        periods = hours * 2
        results = []
        for i in range(periods):
            ts = start + timedelta(minutes=30 * i)
            key = ts.strftime("%Y-%m-%dT%H:%M")
            entry = forecast_map.get(key)
            if entry:
                if conservative and entry.get("pv_estimate10_kw") is not None:
                    solar_kw = entry["pv_estimate10_kw"]
                else:
                    solar_kw = entry["pv_estimate_kw"]
            else:
                solar_kw = 0.0

            results.append({
                "timestamp": ts.isoformat(),
                "solar_kw": max(0.0, solar_kw),
                "solar_kwh": max(0.0, solar_kw) * 0.5,
            })

        return results

    def _zero_forecast(self, hours: int, start: datetime | None) -> list[dict]:
        start = start or datetime.now()
        if start.minute < 15:
            start = start.replace(minute=0, second=0, microsecond=0)
        elif start.minute < 45:
            start = start.replace(minute=30, second=0, microsecond=0)
        else:
            start = (start + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)

        return [
            {
                "timestamp": (start + timedelta(minutes=30 * i)).isoformat(),
                "solar_kw": 0.0,
                "solar_kwh": 0.0,
            }
            for i in range(hours * 2)
        ]
