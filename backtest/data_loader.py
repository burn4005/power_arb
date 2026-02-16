"""CSV data loaders for AEMO prices, home usage, and solar yield.

Loads historical data from CSV files and converts to the 5-minute interval
format expected by the DP optimizer.

CSV data assumptions:
  - AEMO_Pricing.csv: 5-min intervals, RRP in $/kWh (SETTLEMENTDATE is interval end)
  - Home_usage.csv: 30-min slots, kWh per slot, duplicate rows summed per slot
  - solar yield.csv: hourly kW average per Month/Day/Hour
  - Import price = AEMO spot + 20 c/kWh
  - Export price = AEMO spot (QLD feed-in rate)
"""

import csv
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from optimizer.battery_model import (
    PERIOD_MINUTES,
    PERIODS_PER_DAY,
    PERIODS_PER_HOUR,
)

TESTDATA = Path(__file__).parent.parent / "tests" / "testdata"
IMPORT_MARKUP_C = 20.0  # cents/kWh added to AEMO spot for import price

# Home usage CSV is at 30-min resolution; use fixed constant for kWh->kW.
_CSV_PERIOD_HOURS = 0.5


def load_aemo_prices(csv_path: Path | None = None) -> dict[str, float]:
    """Load AEMO 5-min settlement prices at native resolution.

    Returns dict mapping "YYYY/MM/DD HH:MM" (interval start) to spot price
    in c/kWh.
    """
    csv_path = csv_path or TESTDATA / "AEMO_Pricing.csv"
    prices = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            settlement_end = row["SETTLEMENTDATE"]  # interval end time
            rrp_ckwh = float(row["RRP"]) * 100  # $/kWh -> c/kWh

            # Subtract 5 min from settlement end to get interval start
            year = int(settlement_end[:4])
            month = int(settlement_end[5:7])
            day = int(settlement_end[8:10])
            h = int(settlement_end[11:13])
            m = int(settlement_end[14:16])
            dt_end = datetime(year, month, day, h, m)
            dt_start = dt_end - timedelta(minutes=5)

            key = dt_start.strftime("%Y/%m/%d %H:%M")
            prices[key] = rrp_ckwh

    return prices


def load_home_usage(csv_path: Path | None = None) -> dict[str, float]:
    """Load home consumption, summing duplicate rows per half-hour slot.

    Returns dict mapping ISO timestamp (without offset) to kW average for
    that 30-min period (kWh / 0.5 = kW average).
    """
    csv_path = csv_path or TESTDATA / "Home_usage.csv"
    slots: dict[str, float] = defaultdict(float)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            from_ts = row["From (date/time)"]
            kwh = float(row["Amount Used"])
            ts_key = from_ts[:19]
            slots[ts_key] += kwh

    return {ts: kwh / _CSV_PERIOD_HOURS for ts, kwh in slots.items()}


def load_solar_yield(csv_path: Path | None = None) -> dict[tuple[int, int, int], float]:
    """Load solar yield as kW average per (month, day, hour).

    Returns dict mapping (month, day, hour) to kW average output.
    """
    csv_path = csv_path or TESTDATA / "solar yield.csv"
    solar = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            month = int(row["Month"])
            day = int(row["Day"])
            hour = int(row["Hour"])
            kw = float(row["Total Array Output (kW)"])
            solar[(month, day, hour)] = kw
    return solar


def get_day_prices(
    aemo: dict[str, float], date_str: str, import_markup_c: float = IMPORT_MARKUP_C
) -> list[tuple[float, float]]:
    """Get 288 five-minute (import, export) price pairs for a given date.

    date_str format: "YYYY/MM/DD"
    Returns list of (import_c_kwh, export_c_kwh) tuples.
    """
    pairs = []
    for pp in range(PERIODS_PER_DAY):
        h = pp // PERIODS_PER_HOUR
        m = (pp % PERIODS_PER_HOUR) * PERIOD_MINUTES
        key = f"{date_str} {h:02d}:{m:02d}"
        spot = aemo.get(key, 8.0)
        pairs.append((spot + import_markup_c, spot))
    return pairs


def get_day_solar(solar: dict[tuple[int, int, int], float], month: int, day: int) -> list[float]:
    """Get 288 five-minute solar kW values for a given month/day.

    Hourly data is flat-repeated for all periods within each hour.
    """
    values = []
    for hour in range(24):
        kw = solar.get((month, day, hour), 0.0)
        values.extend([kw] * PERIODS_PER_HOUR)
    return values


def get_day_load(usage: dict[str, float], date_str: str) -> list[float]:
    """Get 288 five-minute load kW values for a given date.

    date_str format: "YYYY-MM-DD"
    30-min usage data flat-repeated for all 5-min periods within each slot.
    Falls back to 1.5kW if no data.
    """
    repeat = PERIODS_PER_HOUR // 2  # 6 five-min periods per half-hour
    values = []
    for hh in range(48):
        h = hh // 2
        m = (hh % 2) * 30
        ts = f"{date_str}T{h:02d}:{m:02d}:00"
        kw = usage.get(ts, 1.5)
        values.extend([kw] * repeat)
    return values


def get_spot_price(aemo: dict[str, float], date_str: str, period_idx: int) -> float:
    """Get a single spot price for a date and period index (0-287).

    date_str format: "YYYY/MM/DD"
    Returns spot price in c/kWh, or 8.0 as default.
    """
    h = period_idx // PERIODS_PER_HOUR
    m = (period_idx % PERIODS_PER_HOUR) * PERIOD_MINUTES
    key = f"{date_str} {h:02d}:{m:02d}"
    return aemo.get(key, 8.0)
