"""Export backtest results to JSON for web visualization."""

import json
import os
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("AMBER_API_KEY", "psk_test")
os.environ.setdefault("SOLCAST_API_KEY", "test")
os.environ.setdefault("SOLCAST_RESOURCE_ID", "test")
os.environ.setdefault("FOXESS_IP", "127.0.0.1")

from backtest.data_loader import load_aemo_prices, load_home_usage, load_solar_yield
from backtest.runner import BacktestRunner
import config


def export_backtest_json(output_path: str = "web/backtest_data.json"):
    """Run backtest and export period-level data as columnar JSON."""
    print("Loading CSV data...")
    aemo = load_aemo_prices()
    usage = load_home_usage()
    solar = load_solar_yield()

    runner = BacktestRunner(
        aemo_prices=aemo,
        home_usage=usage,
        solar_yield=solar,
    )

    print("Running backtest...")
    result = runner.run()
    result.print_summary()

    # Convert period records to columnar format
    records = result.period_records
    n = len(records)
    print(f"\nExporting {n} period records to {output_path}...")

    # Compute cumulative savings
    cumulative_savings = []
    running = 0.0
    for r in records:
        running += r.baseline_cost_c - r.net_cost_c
        cumulative_savings.append(round(running, 2))

    data = {
        "metadata": {
            "start_date": result.start_date,
            "end_date": result.end_date,
            "total_days": result.total_days,
            "total_periods": n,
            "capacity_kwh": config.battery.capacity_kwh,
            "min_soc_kwh": config.battery.min_soc_kwh,
            "max_power_kw": config.battery.max_power_kw,
            "total_savings_aud": round(result.total_savings_c / 100, 2),
            "total_cycles": round(result.total_cycles, 1),
            "cycle_life": config.battery.cycle_life,
            "price_forecast_mae_c": round(result.price_forecast_mae_c, 1),
        },
        "timestamps": [r.timestamp for r in records],
        "soc_kwh": [round(r.soc_kwh, 2) for r in records],
        "action": [r.action for r in records],
        "solar_kw": [round(r.solar_kw, 2) for r in records],
        "load_kw": [round(r.load_kw, 2) for r in records],
        "import_price_c": [round(r.import_price_c, 2) for r in records],
        "export_price_c": [round(r.export_price_c, 2) for r in records],
        "grid_import_kwh": [round(r.grid_import_kwh, 3) for r in records],
        "grid_export_kwh": [round(r.grid_export_kwh, 3) for r in records],
        "net_cost_c": [round(r.net_cost_c, 2) for r in records],
        "cumulative_savings_c": cumulative_savings,
    }

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(data, f)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Exported {size_mb:.1f} MB to {output_path}")


if __name__ == "__main__":
    export_backtest_json()
