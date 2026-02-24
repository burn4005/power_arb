"""Helpers to export an existing BacktestResult for the web UI."""

import json
from pathlib import Path

import config
from backtest.runner import BacktestResult


def export_result_to_web_json(
    result: BacktestResult,
    output_path: str = "web/backtest_data.json",
) -> Path:
    """Serialize a BacktestResult to the dashboard's JSON format."""
    records = result.period_records
    n = len(records)

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
            "price_forecast_mae_import_c": round(result.price_forecast_mae_import_c, 1),
            "period_minutes": 5,
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

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f)
    return out
