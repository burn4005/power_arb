# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Battery arbitrage optimizer for a FoxESS KH10 inverter with 42 kWh battery. Reads wholesale electricity prices (Amber Electric), solar forecasts (Solcast), and home consumption patterns, then uses dynamic programming to decide when to charge, discharge, hold, or self-use the battery every 5 minutes.

## Commands

```bash
# Run application (requires .env with API keys + inverter IP)
python main.py

# Run all tests (excluding slow annual backtest)
pytest tests/ --ignore=tests/test_annual_backtest.py -v

# Run annual backtest with progress output (~8 min)
pytest tests/test_annual_backtest.py -v -s

# Run a single test
pytest tests/test_optimizer.py::TestDPOptimizer::test_low_now_high_later_charges -v

# Run backtest export
python -m backtest.export
```

No build step, linter, or formatter is configured. Dependencies are in `requirements.txt`.

## Architecture

### Main Loop (`main.py`)

`PowerArbSystem.run_cycle()` runs every 5 minutes via `schedule`:

1. Read inverter state via Modbus TCP → `foxess/modbus_client.py`
2. Fetch 48h price forecast → `amber/client.py` → dampen extremes → `amber/price_dampener.py`
3. Get solar forecast → `solcast/client.py` (cached, fetched 5x/day at fixed AEST hours)
4. Predict consumption → `forecasting/consumption.py` (time-of-day + weekday/weekend profile)
5. Run DP optimizer → `optimizer/dp_optimizer.py`
6. Execute action on inverter via Modbus or remote control
7. Write `web/dashboard_status.json` for the dashboard

### DP Optimizer (`optimizer/`)

- **State:** SoC discretized in 0.5 kWh steps (~85 levels)
- **Horizon:** 48h = 576 periods × 5 minutes
- **Actions:** 6 discrete — GRID_CHARGE, GRID_CHARGE_NO_EXPORT, SELF_USE, SELF_USE_NO_EXPORT, HOLD, DISCHARGE_GRID
- **Backward pass:** Vectorized over all SoC states using NumPy, loops over time × actions in Python
- **Economic guardrails:** Per-period skip for charge (import < avg_future × η²) and discharge (export ≤ avg_future / η²)
- **Per-state mask:** `-np.inf` for GRID_CHARGE when near capacity, DISCHARGE_GRID at min SoC
- `battery_model.py` has both scalar `apply_action()` (for settlement) and vectorized `apply_action_vec()` (for DP backward pass)

### FoxESS KH Modbus (`foxess/modbus_client.py`)

KH series uses **holding registers only** (FC 0x03), not input registers. No FORCE_CHARGE/FORCE_DISCHARGE work modes — uses remote control registers (44000-44002) instead. Register 41012 (export limit) is invalid on KH10.

Key register groups:
- **31020–31029:** Battery (voltage, current, power, temp, SoC, BMS limits)
- **31002–31005:** PV power
- **31049–31054:** Grid CT (32-bit) + load
- **44000/44001/44002:** Remote control (enable, watchdog, power)

### Config (`config.py`)

Frozen dataclasses loaded from `.env` via `python-dotenv`. Access via `config.battery`, `config.foxess`, `config.system`, etc. Key computed properties: `battery.degradation_per_kwh`, `battery.one_way_efficiency = sqrt(efficiency)`.

### Backtest (`backtest/`)

`BacktestRunner` steps through historical AEMO/solar/consumption CSV data period-by-period with no future leakage. Uses `PriceForecaster` (naive, from past data) for predictions, settles at actual prices. The annual test runs one backtest in `setup_class` and shares the result across three test methods.

## Key Invariants

- `battery_model.py` scalar and vectorized paths must produce identical results — any new action needs both implementations
- `_explain_action()` in `dp_optimizer.py` must handle every Action enum value
- Remote control actions (GRID_CHARGE, GRID_CHARGE_NO_EXPORT, DISCHARGE_GRID) write to 44000-44002; non-remote actions must call `disable_remote_control()` first if it was previously active
- `emergency_self_use()` must disable remote control AND set work mode to Self-Use
- Min SoC is never set below 10% (hardware safety)
- Price dampener never dampens negative prices (they are reliable)

## Testing Notes

- Tests set dummy env vars (`AMBER_API_KEY=psk_test`, etc.) — no real API calls
- `test_annual_backtest.py` uses `reoptimize_every_n=36` (every 3h) for speed; the full suite takes ~8 min
- No Modbus calls in any test — `modbus_client.py` is only exercised against the real inverter
