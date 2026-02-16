"""Custom CSV pricing provider.

Reads a CSV file with three columns: 24h time, import price, export price.
Generates PriceInterval objects for the optimizer using step-function
interpolation — the price at a given time is the most recent entry at or
before that time.

Example CSV format:
    time,import_price,export_price
    00:00,25.0,5.0
    06:00,30.0,8.0
    14:00,15.0,4.0
    20:00,35.0,10.0
"""

import csv
import logging
from datetime import datetime, timedelta
from pathlib import Path

from amber.client import PriceInterval

logger = logging.getLogger(__name__)


def load_custom_schedule(csv_path: str) -> list[dict]:
    """Load the custom pricing schedule from a CSV file.

    Returns a sorted list of dicts with keys: time_minutes (minutes since
    midnight), import_price, export_price.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Custom pricing CSV not found: {csv_path}")

    entries = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError("Custom pricing CSV is empty")

        for row_num, row in enumerate(reader, start=2):
            if len(row) < 3:
                continue
            time_str = row[0].strip()
            try:
                parts = time_str.split(":")
                hour = int(parts[0])
                minute = int(parts[1]) if len(parts) > 1 else 0
                time_minutes = hour * 60 + minute
            except (ValueError, IndexError):
                logger.warning("Skipping invalid time on row %d: %s", row_num, time_str)
                continue

            try:
                import_price = float(row[1].strip())
                export_price = float(row[2].strip())
            except ValueError:
                logger.warning("Skipping invalid prices on row %d", row_num)
                continue

            entries.append({
                "time_minutes": time_minutes,
                "import_price": import_price,
                "export_price": export_price,
            })

    if not entries:
        raise ValueError(f"No valid pricing entries found in {csv_path}")

    entries.sort(key=lambda e: e["time_minutes"])
    logger.info("Loaded %d custom pricing entries from %s", len(entries), csv_path)
    return entries


def _get_price_at(schedule: list[dict], minutes_since_midnight: int) -> tuple[float, float]:
    """Get (import_price, export_price) for a given time using step interpolation.

    Returns the price from the most recent schedule entry at or before the
    given time. If before the first entry, wraps to the last entry (previous
    day's final price).
    """
    result = schedule[-1]  # default: wrap to last entry
    for entry in schedule:
        if entry["time_minutes"] <= minutes_since_midnight:
            result = entry
        else:
            break
    return result["import_price"], result["export_price"]


def generate_price_intervals(
    csv_path: str,
    start: datetime | None = None,
    hours: int = 48,
    interval_min: int = 5,
) -> dict[str, list[PriceInterval]]:
    """Generate import and export PriceInterval lists from a custom CSV schedule.

    Produces intervals for the specified time range at the given resolution,
    using step-function interpolation from the CSV schedule.

    Returns dict with keys 'import' and 'export', matching the AmberClient
    return format.
    """
    schedule = load_custom_schedule(csv_path)
    start = start or datetime.now()

    # Round start down to nearest interval boundary
    start = start.replace(second=0, microsecond=0)
    minute_remainder = start.minute % interval_min
    if minute_remainder:
        start = start - timedelta(minutes=minute_remainder)

    n_periods = (hours * 60) // interval_min
    import_prices = []
    export_prices = []

    for i in range(n_periods):
        ts = start + timedelta(minutes=i * interval_min)
        end_ts = ts + timedelta(minutes=interval_min)
        minutes_since_midnight = ts.hour * 60 + ts.minute

        imp, exp = _get_price_at(schedule, minutes_since_midnight)

        forecast_type = "current" if i == 0 else "forecast"

        import_prices.append(PriceInterval(
            timestamp=ts.isoformat(),
            end_time=end_ts.isoformat(),
            per_kwh=imp,
            spot_per_kwh=imp,
            channel="import",
            forecast_type=forecast_type,
            spike_status="none",
            duration_min=interval_min,
        ))

        export_prices.append(PriceInterval(
            timestamp=ts.isoformat(),
            end_time=end_ts.isoformat(),
            per_kwh=exp,
            spot_per_kwh=exp,
            channel="export",
            forecast_type=forecast_type,
            spike_status="none",
            duration_min=interval_min,
        ))

    return {
        "import": import_prices,
        "export": export_prices,
    }
