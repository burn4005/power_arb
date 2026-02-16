# Power Arbitrage System -- Implementation Plan

## Overview

A Python-based battery arbitrage system that maximizes profit by deciding when to
charge, discharge, or hold a 42 kWh battery connected to a FoxESS inverter, using
Amber Electric's wholesale pricing and forecast data, Solcast solar forecasts, and
historical home consumption patterns.

---

## System Parameters

| Parameter | Value |
|---|---|
| Battery capacity | 42 kWh |
| Inverter rating | 10 kW |
| Solar array | 13.3 kW DC |
| Battery cost | $15,000 AUD |
| Expected cycle life | 6,000 cycles |
| **Degradation cost per cycle** | **$2.50** |
| **Degradation cost per kWh** | **~6.0 c/kWh** |
| Round-trip efficiency | ~90% (configurable) |
| Effective cost per kWh discharged | ~6.6 c/kWh |
| Export rate | QLD AEMO wholesale (Amber `feedIn` channel) |
| Import rate | Wholesale + network + margin (Amber `general` channel) |

**Implication:** Arbitrage is only profitable when the price spread between charge
and discharge exceeds ~13 c/kWh (6.6c degradation + 10% efficiency loss on the
energy itself). The optimizer enforces this automatically.

---

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  Amber API  │     │  Solcast API │     │  FoxESS      │
│  (prices +  │     │  (solar      │     │  Inverter    │
│   forecast) │     │   forecast)  │     │  (Modbus TCP)│
└──────┬──────┘     └──────┬───────┘     └──────┬───────┘
       │                   │                    │
       ▼                   ▼                    ▼
┌──────────────────────────────────────────────────────┐
│                    DATA LAYER                         │
│  ┌────────────┐ ┌────────────┐ ┌──────────────────┐  │
│  │ Price      │ │ Solar      │ │ Inverter State   │  │
│  │ Collector  │ │ Forecast   │ │ Reader           │  │
│  │ + Dampener │ │ Cache      │ │ (SoC, PV, load)  │  │
│  └─────┬──────┘ └─────┬──────┘ └────────┬─────────┘  │
│        │              │                  │            │
│        ▼              ▼                  ▼            │
│  ┌────────────────────────────────────────────────┐   │
│  │              SQLite Database                   │   │
│  │  prices | solar_forecast | consumption |       │   │
│  │  battery_log | decisions | actuals             │   │
│  └────────────────────┬───────────────────────────┘   │
└───────────────────────┼───────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────┐
│                 OPTIMIZER                              │
│                                                       │
│  1. Consumption Forecaster (historical patterns)      │
│  2. Price Dampener (reduce extreme forecast bias)     │
│  3. DP Optimizer (48h rolling horizon)                │
│     - State: battery SoC (0.5 kWh steps)             │
│     - Actions: charge / discharge / self-use / hold   │
│     - Objective: max profit - degradation cost        │
│  4. Decision: optimal action for current period       │
│                                                       │
└───────────────────────┬───────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────┐
│                 CONTROLLER                            │
│                                                       │
│  Translates optimizer decision -> FoxESS Modbus write │
│  - Work mode (41000): 0=self-use, 3=charge, 4=discharge│
│  - Min SoC (41009): prevent over-discharge            │
│  - Charge/discharge power limits                      │
│  - Safety checks + write verification                 │
│                                                       │
└───────────────────────────────────────────────────────┘
```

---

## Project Structure

```
power_arb/
├── config.py                 # Configuration constants, loaded from .env
├── main.py                   # Main scheduler loop (runs every 5 min)
│
├── amber/
│   ├── __init__.py
│   ├── client.py             # Amber API client (prices + forecasts)
│   └── price_dampener.py     # Dampening logic for extreme forecasts
│
├── foxess/
│   ├── __init__.py
│   └── modbus_client.py      # FoxESS Modbus TCP read/write
│
├── solcast/
│   ├── __init__.py
│   └── client.py             # Solcast API client with local caching
│
├── forecasting/
│   ├── __init__.py
│   ├── consumption.py        # Home consumption prediction model
│   └── solar.py              # Solar forecast manager
│
├── optimizer/
│   ├── __init__.py
│   ├── battery_model.py      # Battery state, efficiency, degradation
│   ├── dp_optimizer.py       # Dynamic programming engine
│   └── actions.py            # Action enum + constraints
│
├── storage/
│   ├── __init__.py
│   └── database.py           # SQLite schema + read/write helpers
│
├── requirements.txt
├── .env.example
└── tests/
    ├── test_optimizer.py
    ├── test_battery_model.py
    └── test_price_dampener.py
```

---

## Implementation Steps (in order)

### Step 1: Project Skeleton + Config

- Create directory structure, `requirements.txt`, `.env.example`
- `config.py`: load all settings from environment / `.env` file
  - Amber API key, site ID
  - Solcast API key, resource ID
  - FoxESS inverter IP, port (502), slave ID (247)
  - Battery specs: capacity (42 kWh), max power (10 kW), min SoC (10%),
    round-trip efficiency (0.90), cost ($15,000), cycle life (6,000)
  - Scheduler interval (300s)

**Dependencies:**
```
amberelectric>=2.0.0
pymodbus>=3.5.0
solcast>=1.0.0
requests>=2.31.0
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.11.0
python-dotenv>=1.0.0
schedule>=1.2.0
```

### Step 2: SQLite Database Layer (`storage/database.py`)

Tables:
- **`prices`**: `timestamp, channel (import/export), per_kwh, spot_per_kwh,
  forecast_type (actual/forecast), spike_status`
- **`solar_forecasts`**: `fetch_time, period_end, pv_estimate, pv_estimate10,
  pv_estimate90`
- **`consumption_log`**: `timestamp, load_watts, source (measured/predicted)`
- **`battery_log`**: `timestamp, soc_pct, soc_kwh, battery_power_w,
  grid_power_w, pv_power_w, load_power_w`
- **`decisions`**: `timestamp, action, reason, import_price, export_price,
  soc_at_decision, expected_profit`
- **`forecast_accuracy`**: `forecast_time, target_time, forecast_price,
  actual_price, channel` (for dampener calibration)

### Step 3: Amber API Client (`amber/client.py`)

- Use `amberelectric` SDK
- `get_current_and_forecast()`: call `get_current_prices(site_id, next=96)`
  to get 48h of 30-min forecasts
- Return separate import and export price series as list of
  `(timestamp, per_kwh_cents, spot_per_kwh_cents, spike_status, is_forecast)`
- Store all fetched prices in SQLite (for dampener training + audit)
- Store forecast vs actual pairs in `forecast_accuracy` table once actuals
  arrive (for dampener calibration)
- Poll every 5 minutes (well within rate limits)

### Step 4: Price Dampener (`amber/price_dampener.py`)

Amber forecasts for extreme events are unreliable -- a $20/kWh forecast often
settles at $0.60/kWh. The dampener adjusts forecasts based on lead time and
magnitude.

**Algorithm:**

```
For each forecast interval at lead_time T:

1. SHORT TERM (T < 2 hours):
   - Use forecast as-is (Amber's AEMO pre-dispatch data is reliable here)
   - Confidence: 0.9

2. MEDIUM TERM (2h <= T < 12h):
   - dampened = median_price + (forecast - median_price) * alpha
   - alpha = 0.5 * exp(-0.1 * (T_hours - 2))  [decays from 0.5 to ~0.18]
   - Clamp to historical P5-P95 range for that time-of-day
   - Confidence: 0.5 -> 0.3

3. LONG TERM (T >= 12h):
   - Blend: dampened = 0.3 * forecast + 0.7 * historical_avg_for_time_of_day
   - Confidence: 0.2

4. SPIKE HANDLING:
   - If spikeStatus == "spike": use forecast * 0.7 (spikes confirmed but
     magnitude still overstated)
   - If spikeStatus == "potential": use forecast * 0.4
   - Never dampen negative prices (those are real and reliable)

5. SELF-CALIBRATION (after accumulating 7+ days of data):
   - Compute actual forecast error distribution by lead_time bucket
   - Replace fixed alpha with empirically fitted values
   - Weighted: recent data weighted more heavily
```

### Step 5: FoxESS Modbus Client (`foxess/modbus_client.py`)

Using `pymodbus` for local Modbus TCP to the FoxESS inverter.

**Read operations** (input registers, function code 0x04):
- Battery SoC: register 11036 (uint16, %)
- Battery Power: register 11034 (int16, W; +charge/-discharge)
- PV1 Power: register 11002 (int16, W)
- PV2 Power: register 11005 (int16, W)
- Grid Power: register 11021 (int16, W; +import/-export)
- Load Power: register 11023 (int16, W) -- verify for your model

**Write operations** (holding registers, function code 0x06):
- Work Mode: register 41000 (0=self-use, 3=force-charge, 4=force-discharge)
- Min SoC: register 41009 (10-100%)
- Charge period enable/times: registers 41001-41007
- Max charge/discharge current: registers 41024-41025

**Safety rules built into the client:**
- 200ms minimum between writes
- Read-back verification after every write (wait 500ms, then read)
- Never set min SoC below 10%
- Always restore to self-use mode if communication fails
- Watchdog: if no optimizer heartbeat for 10 min, revert to self-use
- Log every mode change to SQLite

**Important:** Register addresses may vary by FoxESS model/firmware. The code
will include a register map config that can be adjusted. We will verify against
the `nathanmarlor/foxess_modbus` GitHub repo for your specific model.

### Step 6: Solcast Client (`solcast/client.py`)

- Fetch forecast 5x/day (05:00, 08:00, 11:00, 14:00, 17:00 AEST)
- Request full 48h horizon each time (`hours=48, period=PT30M`)
- Cache every response in SQLite `solar_forecasts` table
- Between fetches, serve cached forecast data
- Return: list of `(period_end, pv_estimate_kw, pv_estimate10_kw, pv_estimate90_kw)`
- Convert kW average to kWh per 30-min interval: `kWh = kW * 0.5`

### Step 7: Consumption Forecaster (`forecasting/consumption.py`)

Predict home consumption for the next 48 hours using historical patterns.

**Approach:**
1. Collect load_power readings from FoxESS every 5 minutes -> store in
   `consumption_log`
2. After 7+ days of data, build a time-of-day + day-of-week profile:
   - Group by (day_type [weekday/weekend], half_hour_slot)
   - Compute median and P75 consumption for each slot
3. Use P75 as the forecast (conservative -- better to over-predict consumption
   so we don't run out of battery)
4. Until enough data exists, use a flat estimate (configurable, e.g., 1.5 kW avg)
5. Return: list of `(timestamp, predicted_load_kw)` for next 48h in 30-min slots

### Step 8: Battery Model (`optimizer/battery_model.py`)

Models battery state transitions and costs.

```python
@dataclass
class BatteryState:
    soc_kwh: float          # current energy stored
    capacity_kwh: float     # 42.0
    max_power_kw: float     # 10.0
    min_soc_pct: float      # 10% = 4.2 kWh
    efficiency: float       # 0.90 round-trip (0.95 each way)
    cycle_cost: float       # $2.50 per full cycle
    degradation_per_kwh: float  # $0.0595/kWh

def apply_action(state, action, duration_h, solar_kw, load_kw):
    """
    Returns: (new_soc_kwh, grid_import_kwh, grid_export_kwh,
              degradation_cost, energy_cycled_kwh)
    """
```

**Key logic:**
- Charging: `new_soc = soc + power * duration * charge_efficiency`
- Discharging: `energy_out = power * duration`, `soc_cost = energy_out / discharge_efficiency`
- Degradation cost = `|energy_cycled| / capacity * cycle_cost`
- Enforce SoC bounds: `[min_soc, capacity]`
- Enforce power bounds: `[-max_power, +max_power]`
- Solar is "free" energy -- charging from solar has no grid cost, only degradation

### Step 9: DP Optimizer (`optimizer/dp_optimizer.py`)

**This is the core profit-maximizing engine.**

Rolling-horizon dynamic programming over the next 48 hours:

- **Time steps:** 96 periods of 30 minutes each
- **State:** Battery SoC, discretized in 0.5 kWh steps (0 to 42 kWh = 85 levels)
- **Actions per period:**

| Action | Description | Battery effect |
|--------|-------------|----------------|
| GRID_CHARGE | Import from grid to charge battery at max rate | SoC increases |
| SOLAR_CHARGE | Charge only from excess solar (self-use mode) | SoC increases (free) |
| HOLD | Power home from grid, don't touch battery | SoC unchanged |
| DISCHARGE_HOME | Power home from battery, avoid grid import | SoC decreases |
| DISCHARGE_GRID | Export from battery to grid at max rate | SoC decreases fast |

- **Reward function per period:**

```
profit = export_revenue - import_cost - degradation_cost

Where:
  export_revenue = grid_export_kwh * export_price_per_kwh
  import_cost    = grid_import_kwh * import_price_per_kwh
  degradation    = energy_cycled_kwh * degradation_per_kwh

Compared to baseline of doing nothing (HOLD):
  baseline_cost  = load_kwh * import_price  (must power home from grid)
  baseline_solar = min(solar_kwh, load_kwh) used directly, rest exported

  marginal_profit = profit(action) - profit(HOLD)
```

- **DP recursion (backward pass):**

```
V(t, soc) = max over actions [
    reward(t, soc, action) + V(t+1, new_soc(soc, action))
]
```

- **Forward pass:** starting from current SoC, follow optimal policy
- **Output:** sequence of (action, expected_profit) for each 30-min period
- **Execution:** take the action for the current period only, re-run DP next cycle

**Computational cost:** 85 states x 96 periods x 5 actions = ~40,800 evaluations.
Each is trivial arithmetic. Total runtime: <100ms on a Raspberry Pi. Runs every
5 minutes.

### Step 10: Main Scheduler (`main.py`)

The main loop ties everything together:

```
Every 5 minutes:
  1. READ INVERTER STATE
     - Battery SoC, PV power, grid power, load power via Modbus
     - Log to battery_log table

  2. FETCH PRICES
     - Get current + 48h forecast from Amber API
     - Run through price dampener
     - Store raw + dampened prices

  3. GET SOLAR FORECAST
     - Read cached Solcast forecast (fetched separately on schedule)
     - Align to 30-min periods

  4. PREDICT CONSUMPTION
     - Generate 48h consumption forecast from historical patterns

  5. RUN OPTIMIZER
     - Feed current SoC + all forecasts into DP optimizer
     - Get optimal action for current period

  6. EXECUTE ACTION
     - Translate action to FoxESS Modbus commands
     - Write mode + power limits + min SoC
     - Verify write succeeded
     - Log decision to decisions table

  7. UPDATE FORECAST ACCURACY
     - Compare past forecasts to actuals
     - Feed into dampener calibration

Separately (on its own schedule):
  - Solcast fetch: 5x/day at 05:00, 08:00, 11:00, 14:00, 17:00 AEST
```

**Safety / fallback:**
- If Amber API fails: use last known prices, bias toward HOLD
- If Modbus fails: log error, retry once, then skip cycle (inverter stays
  in last mode -- self-use is safe default)
- If optimizer produces negative expected profit for all actions: HOLD
- Watchdog thread: if main loop hasn't run in 10 min, force self-use mode

### Step 11: Tests

- `test_battery_model.py`: verify SoC transitions, efficiency, bounds
- `test_price_dampener.py`: verify dampening at various lead times, spike handling
- `test_optimizer.py`: known price scenarios -> verify optimal actions
  - Flat prices -> HOLD (no spread = no profit)
  - Low now, high later -> CHARGE now
  - High now, low later -> DISCHARGE now
  - Negative import price -> CHARGE (get paid to consume)
  - Price spread < degradation cost -> HOLD

---

## Key Design Decisions

### Why Dynamic Programming over simpler rules?

Simple threshold rules ("charge below 10c, discharge above 30c") miss context:
- They don't account for *when* the next price peak is
- They don't consider how full the battery is vs. remaining solar
- They can't weigh "moderate profit now" vs "bigger profit in 2 hours"

DP naturally handles all of this by evaluating every possible future path.
It's still very fast (sub-100ms) for this problem size.

### Why dampen Amber forecasts?

Amber passes through AEMO's pre-dispatch pricing, which includes extreme
forecasts that rarely materialize. A $20/kWh forecast might settle at
$0.60/kWh because:
- Generators respond to high price signals by increasing output
- Demand response kicks in
- The forecast is a "what-if" scenario, not a prediction

Without dampening, the optimizer would aggressively charge/discharge for
events that never happen, wasting battery cycles ($2.50 each).

### Why factor in degradation cost?

Every kWh cycled through the battery costs ~6c in degradation. Without this:
- The optimizer would trade on tiny price spreads (e.g., 2c difference)
- Each such trade *loses* money after accounting for battery wear
- The battery would die years early, destroying the economics

With it, the optimizer only trades when the spread justifies the wear.

---

## Profit Estimation (rough)

Based on typical QLD Amber pricing patterns:
- Morning off-peak: ~8-15 c/kWh import
- Afternoon solar sponge: ~3-8 c/kWh (sometimes negative)
- Evening peak: ~25-50 c/kWh import, ~20-40 c/kWh export
- Spike events (few times/month): ~100-1000+ c/kWh for 1-3 intervals

**Daily arbitrage opportunity:**
- Charge 42 kWh at ~5 c/kWh (midday solar sponge) = $2.10
- Discharge 42 kWh at ~35 c/kWh (evening peak) = $14.70 (x 0.9 efficiency = $13.23)
- Degradation cost = $2.50
- **Net profit: ~$8.63/day** from pure arbitrage (before home self-consumption savings)

**Spike events (bonus):**
- 5-10 events/month where export exceeds $1/kWh
- Even dampened, capturing a few of these adds significant value

**Self-consumption savings:**
- Using stored solar instead of grid import during peak saves ~25-40 c/kWh
- This is *in addition to* arbitrage profit

---

## Future Enhancements (not in initial build)

1. **Web dashboard** -- simple Flask/FastAPI page showing current state, decisions, profit tracking
2. **MQTT integration** -- publish state for Home Assistant
3. **Weather-aware dampening** -- cloudy day = less solar = higher evening prices
4. **Machine learning price forecaster** -- replace dampener with a trained model
5. **Multi-battery support** -- for systems with multiple battery packs
6. **Demand charge optimization** -- if network tariff has demand charges
