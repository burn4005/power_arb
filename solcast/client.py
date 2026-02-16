import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import requests

import config
from storage.database import Database

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.solcast.com.au"

# Solcast free tier: 10 API calls/day. Optimized schedule (AEST):
#   04:00 — overnight call for next-day forecast
#   06:00-10:00 — hourly during sunrise ramp (most volatile solar period)
#   12:00, 14:00, 16:00, 18:00 — afternoon updates through sunset
FETCH_HOURS_AEST = [4, 6, 7, 8, 9, 10, 12, 14, 16, 18]


class SolcastClient:
    """Fetches and caches solar production forecasts from Solcast.

    Free tier allows 10 API calls/day. All 10 used on an optimized
    schedule weighted toward sunrise ramp hours. Forecasts are cached
    in SQLite and served from cache between fetches.
    """

    def __init__(self, db: Database):
        self.db = db
        self.api_key = config.solcast.api_key
        self.resource_id = config.solcast.resource_id
        self._headers = {"Authorization": f"Bearer {self.api_key}"}
        self._last_fetch_time: datetime | None = None

    def should_fetch_now(self, now: datetime | None = None) -> bool:
        """Check if it's time for a scheduled fetch."""
        now = now or datetime.now()
        current_hour = now.hour

        # Only fetch at scheduled hours
        if current_hour not in FETCH_HOURS_AEST:
            return False

        # Don't fetch more than once in the same hour
        if self._last_fetch_time and self._last_fetch_time.hour == current_hour:
            return False

        return True

    def fetch_forecast(self) -> list[dict] | None:
        """Fetch 48h solar forecast from Solcast API.

        Returns list of dicts with keys: period_end, pv_estimate_kw,
        pv_estimate10_kw, pv_estimate90_kw, pv_estimate_kwh (per 30-min).
        Returns None on failure.
        """
        url = f"{_BASE_URL}/rooftop_sites/{self.resource_id}/forecasts"
        params = {
            "hours": 48,
            "period": "PT30M",
            "output_parameters": "pv_estimate,pv_estimate10,pv_estimate90",
            "format": "json",
        }

        try:
            resp = requests.get(url, headers=self._headers, params=params, timeout=30)
            if resp.status_code == 429:
                logger.warning("Solcast rate limit hit (429). Will retry next scheduled slot.")
                return None
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error("Solcast API request failed: %s", e)
            return None

        data = resp.json()
        forecasts_raw = data.get("forecasts", [])
        if not forecasts_raw:
            logger.warning("Solcast returned empty forecast")
            return None

        now_iso = datetime.now().isoformat()
        results = []
        db_records = []

        for entry in forecasts_raw:
            pv_est = entry.get("pv_estimate", 0)
            pv_10 = entry.get("pv_estimate10")
            pv_90 = entry.get("pv_estimate90")
            period_end = entry["period_end"]

            record = {
                "period_end": period_end,
                "pv_estimate_kw": pv_est,
                "pv_estimate10_kw": pv_10,
                "pv_estimate90_kw": pv_90,
                "pv_estimate_kwh": pv_est * 0.5,  # 30-min interval -> kWh
            }
            results.append(record)

            db_records.append({
                "fetch_time": now_iso,
                "period_end": period_end,
                "pv_estimate_kw": pv_est,
                "pv_estimate10_kw": pv_10,
                "pv_estimate90_kw": pv_90,
            })

        self.db.insert_solar_forecasts(db_records)
        self._last_fetch_time = datetime.now()
        logger.info("Fetched %d solar forecast intervals from Solcast", len(results))
        return results

    def get_forecast(self) -> list[dict]:
        """Get the latest cached solar forecast.

        Fetches fresh data if scheduled, otherwise returns last cached version.
        """
        if self.should_fetch_now():
            fresh = self.fetch_forecast()
            if fresh:
                return fresh

        # Serve from cache
        cached = self.db.get_latest_solar_forecast()
        if not cached:
            logger.warning("No cached solar forecast available")
            return []

        results = []
        for row in cached:
            results.append({
                "period_end": row["period_end"],
                "pv_estimate_kw": row["pv_estimate_kw"],
                "pv_estimate10_kw": row.get("pv_estimate10_kw"),
                "pv_estimate90_kw": row.get("pv_estimate90_kw"),
                "pv_estimate_kwh": row["pv_estimate_kw"] * 0.5,
            })
        return results
