"""CSV data loaders for AEMO prices, home usage, and solar yield.

Loads historical data from CSV files and converts to the 30-minute interval
format expected by the DP optimizer.

CSV data assumptions:
  - AEMO_Pricing.csv: 5-min intervals, RRP in $/kWh (SETTLEMENTDATE is interval end)
  - Home_usage.csv: 30-min slots, kWh per slot, duplicate rows summed per slot
  - solar yield.csv: hourly kW average per Month/Day/Hour, split to 30-min
  - Import price = AEMO spot + 20 c/kWh
  - Export price = AEMO spot (QLD feed-in rate)
"""

import csv
from collections import defaultdict
from pathlib import Path

from optimizer.battery_model import PERIOD_HOURS

TESTDATA = Path(__file__).parent.parent / "tests" / "testdata"
IMPORT_MARKUP_C = 20.0  # cents/kWh added to AEMO spot for import price


def load_aemo_prices(csv_path: Path | None = None) -> dict[str, float]:
    """Load AEMO 5-min prices and aggregate to 30-min averages.

    Returns dict mapping "YYYY/MM/DD HH:MM" (half-hour start) to average
    spot price in c/kWh.
    """
    csv_path = csv_path or TESTDATA / "AEMO_Pricing.csv"
    five_min = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = row["SETTLEMENTDATE"]
            rrp_ckwh = float(row["RRP"]) * 100  # $/kWh -> c/kWh
            five_min.append((ts, rrp_ckwh))

    # Group into 30-min buckets.
    # AEMO settlement "00:05" through "00:30" -> half-hour starting at 00:00
    # "00:35" through "01:00" -> half-hour starting at 00:30, etc.
    buckets: dict[str, list[float]] = defaultdict(list)
    for ts, price in five_min:
        date_part = ts[:10]  # "2024/01/01"
        h = int(ts[11:13])
        m = int(ts[14:16])

        if m == 0:
            if h == 0:
                continue
            bucket_h = h - 1
            bucket_m = 30
        elif m <= 30:
            bucket_h = h
            bucket_m = 0
        else:
            bucket_h = h
            bucket_m = 30

        bucket_key = f"{date_part} {bucket_h:02d}:{bucket_m:02d}"
        buckets[bucket_key].append(price)

    averaged = {}
    for key in sorted(buckets.keys()):
        prices = buckets[key]
        averaged[key] = sum(prices) / len(prices)

    return averaged


def load_home_usage(csv_path: Path | None = None) -> dict[str, float]:
    """Load home consumption, summing duplicate rows per half-hour slot.

    Returns dict mapping ISO timestamp (without offset) to kW average for
    that 30-min period (kWh * 2 = kW average).
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

    return {ts: kwh / PERIOD_HOURS for ts, kwh in slots.items()}


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
    """Get 48 half-hour (import, export) price pairs for a given date.

    date_str format: "YYYY/MM/DD"
    Returns list of (import_c_kwh, export_c_kwh) tuples.
    """
    pairs = []
    for hh in range(48):
        h = hh // 2
        m = (hh % 2) * 30
        key = f"{date_str} {h:02d}:{m:02d}"
        spot = aemo.get(key)
        if spot is None:
            spot = 8.0  # ~8c/kWh default
        import_price = spot + import_markup_c
        export_price = spot
        pairs.append((import_price, export_price))
    return pairs


def get_day_solar(solar: dict[tuple[int, int, int], float], month: int, day: int) -> list[float]:
    """Get 48 half-hour solar kW values for a given month/day.

    Hourly data is repeated for both halves of each hour.
    """
    values = []
    for hour in range(24):
        kw = solar.get((month, day, hour), 0.0)
        values.append(kw)
        values.append(kw)
    return values


def get_day_load(usage: dict[str, float], date_str: str) -> list[float]:
    """Get 48 half-hour load kW values for a given date.

    date_str format: "YYYY-MM-DD"
    Returns list of kW averages; falls back to 1.5kW if no data.
    """
    values = []
    for hh in range(48):
        h = hh // 2
        m = (hh % 2) * 30
        ts = f"{date_str}T{h:02d}:{m:02d}:00"
        kw = usage.get(ts, 1.5)
        values.append(kw)
    return values


def get_spot_price(aemo: dict[str, float], date_str: str, half_hour: int) -> float:
    """Get a single spot price for a date and half-hour index (0-47).

    date_str format: "YYYY/MM/DD"
    Returns spot price in c/kWh, or 8.0 as default.
    """
    h = half_hour // 2
    m = (half_hour % 2) * 30
    key = f"{date_str} {h:02d}:{m:02d}"
    return aemo.get(key, 8.0)
