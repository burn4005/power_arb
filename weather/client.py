"""Open-Meteo weather forecast client with SQLite caching.

Fetches hourly temperature and daily sunrise/sunset data from the free
Open-Meteo API (no key required). Cached in SQLite, refreshed every N hours.
"""

import logging
from datetime import datetime, timedelta

import requests

import config
from storage.database import Database

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.open-meteo.com/v1/forecast"


class WeatherClient:
    """Fetches hourly temperature + daily sunrise/sunset from Open-Meteo."""

    def __init__(self, db: Database):
        self.db = db
        self.latitude = config.weather.latitude
        self.longitude = config.weather.longitude
        self.refresh_hours = config.weather.refresh_interval_hours
        self._last_fetch_time: datetime | None = None

    def should_fetch_now(self) -> bool:
        if self._last_fetch_time is None:
            return True
        elapsed = (datetime.now() - self._last_fetch_time).total_seconds() / 3600
        return elapsed >= self.refresh_hours

    def fetch_forecast(self) -> list[dict] | None:
        """Fetch 7-day hourly temp + daily sunrise/sunset from Open-Meteo."""
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "hourly": "temperature_2m",
            "daily": "sunrise,sunset",
            "timezone": config.system.timezone,
            "forecast_days": 7,
        }
        try:
            resp = requests.get(_BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error("Weather API request failed: %s", e)
            return None

        data = resp.json()
        hourly = data.get("hourly", {})
        daily = data.get("daily", {})

        times = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        sunrises = daily.get("sunrise", [])
        sunsets = daily.get("sunset", [])

        if not times or not temps:
            logger.warning("Weather API returned empty data")
            return None

        # Build sunrise/sunset lookup by date
        sun_lookup = {}
        for i, sr in enumerate(sunrises):
            date_str = sr[:10]
            sun_lookup[date_str] = {
                "sunrise": sunrises[i],
                "sunset": sunsets[i] if i < len(sunsets) else None,
            }

        now_iso = datetime.now().isoformat()
        records = []
        for i, t in enumerate(times):
            date_str = t[:10]
            sun = sun_lookup.get(date_str, {})
            records.append({
                "fetch_time": now_iso,
                "target_time": t,
                "temperature_c": temps[i],
                "sunrise": sun.get("sunrise"),
                "sunset": sun.get("sunset"),
            })

        self.db.insert_weather_cache(records)
        self._last_fetch_time = datetime.now()
        logger.info("Fetched %d weather forecast hours from Open-Meteo", len(records))
        return records

    def get_forecast(self) -> list[dict]:
        """Get weather forecast, fetching fresh data if due."""
        if self.should_fetch_now():
            fresh = self.fetch_forecast()
            if fresh:
                return fresh
        cached = self.db.get_latest_weather()
        if not cached:
            logger.warning("No cached weather data available")
            return []
        return cached

    def get_temperature_at(self, target_time: datetime) -> float | None:
        """Get interpolated temperature for a specific time."""
        forecast = self.get_forecast()
        if not forecast:
            return None

        target_iso = target_time.isoformat()
        before = None
        after = None
        for entry in forecast:
            if entry["target_time"] <= target_iso:
                before = entry
            elif after is None:
                after = entry
                break

        if before and after:
            t0 = datetime.fromisoformat(before["target_time"])
            t1 = datetime.fromisoformat(after["target_time"])
            frac = (target_time - t0).total_seconds() / max(1, (t1 - t0).total_seconds())
            return before["temperature_c"] + frac * (after["temperature_c"] - before["temperature_c"])
        elif before:
            return before["temperature_c"]
        return None
