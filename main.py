"""Power Arbitrage System - Main Scheduler

Runs every 5 minutes:
1. Read inverter state (FoxESS Modbus)
2. Fetch current + forecast prices (Amber API)
3. Get solar forecast (Solcast, cached)
4. Predict home consumption
5. Run DP optimizer
6. Execute optimal action on inverter
7. Write dashboard status JSON for live monitoring
"""

import collections
import hmac
import http.server
import json
import logging
import os
import socket
import signal
import sys
import time
import threading
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import schedule

import config
from amber.client import AmberClient
from forecasting.price_dampener import PriceDampener
from flowpower.client import FlowPowerClient
from foxess.modbus_client import FoxESSModbusClient, WorkMode
from pricing.custom_csv import generate_price_intervals
from solcast.client import SolcastClient, FETCH_HOURS_AEST
from forecasting.consumption import ConsumptionForecaster
from forecasting.solar import SolarForecaster
from forecasting.ml_price import MLPriceForecaster
from forecasting.ml_consumption import MLConsumptionForecaster
from weather.client import WeatherClient
from homeassistant.client import HomeAssistantClient
from optimizer.actions import Action
from optimizer.battery_model import BatteryModel, PeriodInputs, PERIOD_HOURS, PERIODS_PER_HOUR
from optimizer.dp_optimizer import DPOptimizer
from storage.database import Database

logging.basicConfig(
    level=getattr(logging, config.system.log_level),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("power_arb")

# Map optimizer actions to KH work modes.
# GRID_CHARGE / DISCHARGE_GRID use remote control (44000-44002) — not a work mode.
# SELF_USE_NO_EXPORT uses Back-up (2) which suppresses solar export on KH.
# HOLD uses Self-Use with min SoC raised to current level to prevent discharge.
ACTION_TO_WORK_MODE = {
    Action.GRID_CHARGE:          WorkMode.BACKUP,        # fallback if watchdog fires
    Action.GRID_CHARGE_NO_EXPORT: WorkMode.BACKUP,       # same — Back-up already no-export
    Action.SELF_USE:             WorkMode.SELF_USE,
    Action.SELF_USE_NO_EXPORT:   WorkMode.BACKUP,        # Back-up suppresses export on KH
    Action.HOLD:                 WorkMode.SELF_USE,
    Action.DISCHARGE_GRID:       WorkMode.FEED_IN_FIRST, # fallback if watchdog fires
}

# Actions that use remote control registers (44000-44002) instead of work modes 3/4
_REMOTE_CONTROL_ACTIONS = {Action.GRID_CHARGE, Action.GRID_CHARGE_NO_EXPORT, Action.DISCHARGE_GRID}

STATUS_PATH = Path(__file__).parent / "web" / "dashboard_status.json"


class PowerArbSystem:
    def __init__(self):
        self.db = Database()
        self.amber: AmberClient | None = None
        self.dampener = PriceDampener(self.db)
        self.flowpower: FlowPowerClient | None = None
        self.foxess = FoxESSModbusClient()
        self.solcast_client = SolcastClient(self.db)
        self.solar_forecaster = SolarForecaster(self.solcast_client)
        self.consumption = ConsumptionForecaster(self.db)
        self.weather = WeatherClient(self.db)
        self.ha = HomeAssistantClient(self.db)
        self.ml_price = MLPriceForecaster(self.db, self.dampener)
        self.ml_consumption = MLConsumptionForecaster(self.db, self.consumption)
        self.optimizer = DPOptimizer()

        self._last_action: Action | None = None
        self._consecutive_failures = 0
        self._watchdog_last_run = time.time()
        self._running = True

        # Dashboard: daily financial accumulators
        self._today_date = None
        self._today_import_cost_c = 0.0
        self._today_export_revenue_c = 0.0
        self._today_degradation_c = 0.0
        self._today_baseline_c = 0.0
        self._today_energy_cycled_kwh = 0.0
        self._cycle_ms = 0
        self._errors: collections.deque = collections.deque(maxlen=20)

        # Manual override: None = AUTO (optimizer controls), or an Action name
        self._manual_override: str | None = None
        self._override_expires: datetime | None = None
        self._using_custom = False
        self._restart_requested = False
        self._remote_control_active = False  # tracks whether 44000=1 is currently set

    @staticmethod
    def _internet_available(timeout_s: float = 1.5) -> bool:
        """Quick connectivity check for internet-based API calls."""
        for host, port in (("1.1.1.1", 53), ("8.8.8.8", 53)):
            try:
                with socket.create_connection((host, port), timeout=timeout_s):
                    return True
            except OSError:
                continue

        # Fallback for networks where DNS egress checks fail but HTTPS works.
        probe_timeout = max(2.0, timeout_s)
        for url in ("https://api.open-meteo.com", "https://api.solcast.com.au"):
            try:
                req = urllib.request.Request(url, method="HEAD")
                with urllib.request.urlopen(req, timeout=probe_timeout):
                    return True
            except Exception:
                try:
                    with urllib.request.urlopen(url, timeout=probe_timeout):
                        return True
                except Exception:
                    continue
        return False

    def run_cycle(self):
        """Execute one optimization cycle."""
        cycle_start = datetime.now()
        logger.info("=== Optimization cycle starting at %s ===", cycle_start.isoformat())

        # Midnight rollover: reset daily accumulators
        if self._today_date != cycle_start.date():
            self._today_date = cycle_start.date()
            self._today_import_cost_c = 0.0
            self._today_export_revenue_c = 0.0
            self._today_degradation_c = 0.0
            self._today_baseline_c = 0.0
            self._today_energy_cycled_kwh = 0.0

        try:
            # 1. Read inverter state
            state = self.foxess.read_state()
            if state is None:
                logger.error("Cannot read inverter state; continuing with data-only cycle")
                self._handle_failure()
                current_soc = None
            else:
                self._consecutive_failures = 0
                current_soc = state.soc_kwh

                logger.info(
                    "Inverter: SoC=%.1f kWh (%.0f%%), PV=%dW, Grid=%dW, Load=%dW, "
                    "Mode=%s, BMS charge=%dW, BMS discharge=%dW",
                    state.soc_kwh, state.soc_pct, state.pv_power_w,
                    state.grid_power_w, state.load_power_w, state.work_mode.name,
                    state.bms_max_charge_w, state.bms_max_discharge_w,
                )

                # Log battery state
                self.db.insert_battery_log({
                    "timestamp": cycle_start.isoformat(),
                    "soc_pct": state.soc_pct,
                    "soc_kwh": state.soc_kwh,
                    "battery_power_w": state.battery_power_w,
                    "grid_power_w": state.grid_power_w,
                    "pv_power_w": state.pv_power_w,
                    "load_power_w": state.load_power_w,
                    "battery_temp_c": state.battery_temp_c,
                    "inverter_internal_temp_c": state.inverter_internal_temp_c,
                })

                # Log consumption for forecaster training
                self.db.insert_consumption(
                    cycle_start.isoformat(), state.load_power_w, "measured"
                )

            internet_ok = self._internet_available()
            if not internet_ok:
                logger.warning("Internet check failed; skipping API calls this cycle")

            # 1b. Poll occupancy and AC state (if HA configured)
            if internet_ok:
                occupancy = self.ha.poll_occupancy()
                ac_running = self.ha.poll_ac_state()
            else:
                occupancy = True
                ac_running = False

            # 1c. Refresh weather forecast/cache
            self.weather.get_forecast(allow_fetch=internet_ok)
            horizon_hours = max(6, min(48, int(config.system.optimizer_horizon_hours)))
            horizon_periods = horizon_hours * PERIODS_PER_HOUR

            # 2. Fetch prices
            retailer_mode = config.retailer.retailer
            if retailer_mode == "custom" and config.retailer.custom_pricing_csv:
                prices = generate_price_intervals(
                    config.retailer.custom_pricing_csv,
                    start=cycle_start,
                    hours=horizon_hours,
                )
                import_prices_raw = prices["import"]
                export_prices_raw = prices["export"]
                # Custom prices are fixed schedules — no dampening needed
                dampened_import = None
                dampened_export = None
            elif retailer_mode == "flowpower":
                if not internet_ok:
                    logger.error("Internet unavailable; cannot fetch Flow Power pricing")
                    self._handle_failure()
                    return
                if not config.flowpower.api_key:
                    logger.error("FLOWPOWER_API_KEY is missing; cannot fetch Flow Power pricing")
                    self._errors.append(f"{datetime.now().isoformat()}: flowpower api key missing")
                    self._handle_failure()
                    return
                try:
                    if self.flowpower is None:
                        self.flowpower = FlowPowerClient(self.db)
                except Exception as e:
                    logger.error("Failed to initialize Flow Power client: %s", e)
                    self._errors.append(f"{datetime.now().isoformat()}: flowpower init failed: {e}")
                    self._handle_failure()
                    return

                try:
                    prices = self.flowpower.fetch_current_and_forecast()
                except Exception as e:
                    logger.error("Flow Power price fetch failed: %s", e)
                    self._errors.append(f"{datetime.now().isoformat()}: flowpower fetch failed: {e}")
                    self._handle_failure()
                    return

                import_prices_raw = prices["import"]
                export_prices_raw = prices["export"]
                dampened_import = None
                dampened_export = None
            elif retailer_mode == "amber":
                if not internet_ok:
                    logger.error("Internet unavailable; cannot fetch Amber pricing")
                    self._handle_failure()
                    return
                if not config.amber.api_key:
                    logger.error("AMBER_API_KEY is missing; cannot fetch Amber pricing")
                    self._errors.append(f"{datetime.now().isoformat()}: amber api key missing")
                    self._handle_failure()
                    return
                try:
                    if self.amber is None:
                        self.amber = AmberClient(self.db)
                except Exception as e:
                    logger.error("Failed to initialize Amber client: %s", e)
                    self._errors.append(f"{datetime.now().isoformat()}: amber init failed: {e}")
                    self._handle_failure()
                    return

                try:
                    prices = self.amber.fetch_current_and_forecast()
                except Exception as e:
                    logger.error("Amber price fetch failed: %s", e)
                    self._errors.append(f"{datetime.now().isoformat()}: amber fetch failed: {e}")
                    self._handle_failure()
                    return

                import_prices_raw = prices["import"]
                export_prices_raw = prices["export"]
                dampened_import = None
                dampened_export = None
            elif retailer_mode == "custom":
                logger.error("RETAILER=custom but CUSTOM_PRICING_CSV is not set")
                self._errors.append(f"{datetime.now().isoformat()}: custom pricing csv missing")
                self._handle_failure()
                return
            else:
                logger.error("Invalid RETAILER value: %s", retailer_mode)
                self._errors.append(f"{datetime.now().isoformat()}: invalid retailer '{retailer_mode}'")
                self._handle_failure()
                return

            if not import_prices_raw or not export_prices_raw:
                logger.error("No price data; skipping cycle")
                self._errors.append(f"{datetime.now().isoformat()}: no price data")
                self._handle_failure()
                return

            # 3. Dampen forecast prices (shared ML path for Amber + Flow Power).
            if retailer_mode == "custom" and config.retailer.custom_pricing_csv:
                dampened_import = import_prices_raw
                dampened_export = export_prices_raw
            else:
                dampened_import = self.ml_price.dampen(import_prices_raw, cycle_start)
                dampened_export = self.ml_price.dampen(export_prices_raw, cycle_start)

                # Snapshot price features for ML training
                if config.ml.enabled:
                    self.ml_price.snapshot_features(import_prices_raw, cycle_start)

            # 4. Get solar forecast
            solar_fc = self.solar_forecaster.forecast(
                hours=horizon_hours, start=cycle_start, allow_api_fetch=internet_ok
            )

            # 5. Predict consumption
            # Get recent loads for ML features
            recent_history = self.db.get_consumption_history(days=1)
            recent_loads_kw = [r["load_watts"] / 1000.0 for r in recent_history[-36:]]
            profile_kw = self.consumption.predict_slot(cycle_start)

            # Snapshot consumption features for ML training
            if config.ml.enabled:
                self.ml_consumption.snapshot_features(
                    cycle_start.isoformat(), occupancy, recent_loads_kw, profile_kw,
                    ac_running=ac_running,
                )

            consumption_fc = self.ml_consumption.forecast(
                hours=horizon_hours, start=cycle_start,
                occupancy=occupancy, recent_loads_kw=recent_loads_kw,
                ac_running=ac_running,
            )

            # 6. Align all forecasts to same time periods
            n_periods = min(
                len(dampened_import), len(dampened_export),
                len(solar_fc), len(consumption_fc),
                horizon_periods,
            )
            if n_periods == 0:
                logger.error("No aligned forecast periods; skipping cycle")
                return

            self._using_custom = retailer_mode == "custom" and config.retailer.custom_pricing_csv
            using_custom = self._using_custom
            if using_custom:
                import_cents = [dampened_import[i].per_kwh for i in range(n_periods)]
                export_cents = [dampened_export[i].per_kwh for i in range(n_periods)]
            else:
                import_cents = [dampened_import[i].dampened_per_kwh for i in range(n_periods)]
                export_cents = [dampened_export[i].dampened_per_kwh for i in range(n_periods)]
            solar_kw = [solar_fc[i]["solar_kw"] for i in range(n_periods)]
            load_kw = [consumption_fc[i]["load_kw"] for i in range(n_periods)]
            timestamps = [dampened_import[i].timestamp for i in range(n_periods)]

            # If inverter is offline, still keep data ingest active but skip control.
            if current_soc is None:
                elapsed_ms = (datetime.now() - cycle_start).total_seconds() * 1000
                self._cycle_ms = int(elapsed_ms)
                self._watchdog_last_run = time.time()
                self._write_dashboard_status_inverter_offline(
                    cycle_start,
                    import_prices_raw, export_prices_raw,
                    dampened_import, dampened_export,
                    solar_fc,
                    import_cents, export_cents, solar_kw, load_kw,
                    timestamps, n_periods, horizon_hours,
                )
                logger.warning(
                    "Data-only cycle complete in %.1fs | captured prices/forecasts, "
                    "skipped optimization/control (inverter offline)",
                    elapsed_ms / 1000,
                )
                return

            # 7. Run optimizer
            result = self.optimizer.optimize(
                current_soc_kwh=current_soc,
                import_prices=import_cents,
                export_prices=export_cents,
                solar_forecast=solar_kw,
                load_forecast=load_kw,
                timestamps=timestamps,
            )

            # 8. Apply manual override if active
            optimizer_action = result.action
            if self._manual_override and self._manual_override != "AUTO":
                # Check expiry
                if self._override_expires and datetime.now() >= self._override_expires:
                    logger.info("Manual override expired, reverting to AUTO")
                    self._manual_override = None
                    self._override_expires = None
                    action = optimizer_action
                else:
                    action = Action[self._manual_override]
                    logger.info(
                        "Manual override active: %s (optimizer recommends %s)",
                        action.name, optimizer_action.name,
                    )
            else:
                action = optimizer_action

            work_mode = ACTION_TO_WORK_MODE[action]

            # Set min SoC based on action
            if action == Action.HOLD:
                min_soc = max(10, int(state.soc_pct))
            else:
                min_soc = int(config.battery.min_soc_pct)

            # Write to inverter when action changes, or refresh remote-control actions each cycle.
            mode_changed = False
            prev_action = self._last_action
            should_apply = (action != prev_action) or (action in _REMOTE_CONTROL_ACTIONS)
            if should_apply:
                if action != prev_action:
                    logger.info(
                        "Action changed: %s -> %s",
                        prev_action.name if prev_action else "None",
                        action.name,
                    )
                elif action in _REMOTE_CONTROL_ACTIONS:
                    logger.debug("Refreshing remote-control watchdog for %s", action.name)

                if action in _REMOTE_CONTROL_ACTIONS:
                    # GRID_CHARGE / GRID_CHARGE_NO_EXPORT / DISCHARGE_GRID:
                    # use remote control registers (44000-44002)
                    max_hw_w = int(config.battery.max_power_kw * 1000)
                    if action == Action.DISCHARGE_GRID:
                        power_w = min(state.bms_max_discharge_w, max_hw_w)
                        fallback = WorkMode.FEED_IN_FIRST
                    else:
                        power_w = -min(state.bms_max_charge_w, max_hw_w)
                        fallback = WorkMode.BACKUP
                    success = self.foxess.set_remote_control(power_w, fallback_mode=fallback)
                    if success:
                        self._remote_control_active = True
                else:
                    # SELF_USE / SELF_USE_NO_EXPORT / HOLD:
                    # disable remote control first, then set work mode
                    if self._remote_control_active:
                        if not self.foxess.disable_remote_control():
                            logger.warning("Failed to disable remote control before mode change")
                        self._remote_control_active = False
                    success = self.foxess.set_work_mode(work_mode)

                if success:
                    self.foxess.set_min_soc(min_soc)
                    self._last_action = action
                    mode_changed = action != prev_action
                else:
                    logger.error("Failed to apply action %s; keeping previous", action.name)

            # Log decision
            override_active = self._manual_override and self._manual_override != "AUTO"
            reason = (
                f"MANUAL OVERRIDE: {action.name} (optimizer: {optimizer_action.name} — {result.reason})"
                if override_active else result.reason
            )
            self.db.insert_decision({
                "timestamp": cycle_start.isoformat(),
                "action": action.value,
                "reason": reason,
                "import_price": import_prices_raw[0].per_kwh if import_prices_raw else None,
                "export_price": export_prices_raw[0].per_kwh if export_prices_raw else None,
                "dampened_import_price": import_cents[0] if import_cents else None,
                "dampened_export_price": export_cents[0] if export_cents else None,
                "soc_kwh": current_soc,
                "expected_profit": result.expected_profit_cents,
                "actual_mode_set": work_mode.value if mode_changed else None,
            })

            # 9. Update financial accumulators
            period_inputs = PeriodInputs(
                solar_kw=solar_kw[0], load_kw=load_kw[0],
                import_price=import_cents[0], export_price=export_cents[0],
            )
            period_result = self.optimizer.model.apply_action(
                current_soc, action, period_inputs
            )
            self._today_import_cost_c += period_result.grid_import_kwh * import_cents[0]
            self._today_export_revenue_c += period_result.grid_export_kwh * export_cents[0]
            self._today_degradation_c += period_result.degradation_cost_cents
            self._today_baseline_c += load_kw[0] * PERIOD_HOURS * import_cents[0]
            self._today_energy_cycled_kwh += abs(
                period_result.new_soc_kwh - current_soc
            )

            self._watchdog_last_run = time.time()

            elapsed_ms = (datetime.now() - cycle_start).total_seconds() * 1000
            self._cycle_ms = int(elapsed_ms)
            logger.info(
                "Cycle complete in %.1fs | Action: %s | Horizon profit: %.0fc",
                elapsed_ms / 1000, action.name, result.expected_profit_cents,
            )

            # Log the next few scheduled actions
            for ts, act, profit in result.schedule[:6]:
                logger.debug("  %s: %s (%.1fc)", ts, act.name, profit)

            # 10. Write dashboard status
            self._write_dashboard_status(
                cycle_start, state, result, action, optimizer_action,
                import_prices_raw, export_prices_raw,
                dampened_import, dampened_export,
                solar_fc, consumption_fc,
                import_cents, export_cents, solar_kw, load_kw,
                timestamps, n_periods, horizon_hours,
            )

        except Exception as e:
            logger.exception("Cycle failed: %s", e)
            self._errors.append(f"{datetime.now().isoformat()}: {e}")
            self._handle_failure()

    def _build_dashboard_dict(
        self, cycle_start,
        import_prices_raw, export_prices_raw,
        dampened_import, dampened_export,
        solar_fc,
        import_cents, export_cents, solar_kw, load_kw,
        timestamps, n_periods, horizon_hours,
        *, live_data=None, optimizer_data=None,
    ):
        """Build the dashboard status dict.

        Args:
            live_data: dict with inverter live readings, or None if offline.
            optimizer_data: dict with result/action/optimizer_action, or None if offline.
        """
        # Solcast health info
        solcast_last = self.solcast_client.last_fetch_time
        current_hour = cycle_start.hour
        next_solcast = next(
            (h for h in FETCH_HOURS_AEST if h > current_hour),
            FETCH_HOURS_AEST[0],
        )
        solcast_calls_today = sum(1 for h in FETCH_HOURS_AEST if h <= current_hour)

        # Solar P10/P90 arrays
        solar_p10 = [solar_fc[i].get("solar_p10_kw", 0.0) for i in range(n_periods)]
        solar_p90 = [solar_fc[i].get("solar_p90_kw", 0.0) for i in range(n_periods)]

        # History: last 48h battery log + decisions
        since_48h = (cycle_start - timedelta(hours=48)).isoformat()
        history = self.db.get_battery_log_since(since_48h)
        decisions = self.db.get_decisions_since(since_48h)
        decision_map = {d["timestamp"]: d for d in decisions}

        # Financial
        today_savings = self._today_baseline_c - (
            self._today_import_cost_c
            - self._today_export_revenue_c
            + self._today_degradation_c
        )
        capacity = config.battery.capacity_kwh
        cycles_today = (
            self._today_energy_cycled_kwh / (2 * capacity)
            if capacity > 0 else 0
        )

        # Live section: from inverter state or offline defaults
        if live_data:
            state = live_data["state"]
            action = live_data["action"]
            optimizer_action = live_data["optimizer_action"]
            result = live_data["result"]
            live = {
                "pv_power_kw": round(state.pv_power_w / 1000, 2),
                "load_power_kw": round(state.load_power_w / 1000, 2),
                "grid_power_kw": round(state.grid_power_w / 1000, 2),
                "battery_power_kw": round(state.battery_power_w / 1000, 2),
                "soc_pct": round(state.soc_pct, 1),
                "soc_kwh": round(state.soc_kwh, 1),
                "battery_temp_c": round(state.battery_temp_c, 1),
                "inverter_internal_temp_c": round(state.inverter_internal_temp_c, 1),
                "min_soc_pct": state.min_soc_pct,
                "import_price_c": round(import_prices_raw[0].per_kwh, 1) if import_prices_raw else 0,
                "export_price_c": round(export_prices_raw[0].per_kwh, 1) if export_prices_raw else 0,
                "dampened_import_c": round(import_cents[0], 1),
                "dampened_export_c": round(export_cents[0], 1),
                "spike_status": import_prices_raw[0].spike_status if import_prices_raw else "none",
                "action": action.name,
                "optimizer_action": optimizer_action.name,
                "reason": result.reason,
                "work_mode": state.work_mode.name,
                "remote_control_active": self._remote_control_active,
                "override": self._manual_override or "AUTO",
                "override_expires": self._override_expires.isoformat() if self._override_expires else None,
                "bms_max_charge_kw": round(state.bms_max_charge_w / 1000, 2),
                "bms_max_discharge_kw": round(state.bms_max_discharge_w / 1000, 2),
            }
            foxess_status = "ok"
            foxess_last_read = cycle_start.isoformat()
            horizon_profit = round(result.expected_profit_cents, 1)
            schedule_action = [a.name for _, a, _ in result.schedule]
            schedule_profit = [round(p, 2) for _, _, p in result.schedule]
            soc_trajectory = [round(s, 1) for s in result.soc_trajectory]
        else:
            live_import = import_prices_raw[0].per_kwh if import_prices_raw else 0.0
            live_export = export_prices_raw[0].per_kwh if export_prices_raw else 0.0
            live = {
                "pv_power_kw": 0.0,
                "load_power_kw": 0.0,
                "grid_power_kw": 0.0,
                "battery_power_kw": 0.0,
                "soc_pct": 0.0,
                "soc_kwh": 0.0,
                "battery_temp_c": 0.0,
                "inverter_internal_temp_c": 0.0,
                "min_soc_pct": config.battery.min_soc_pct,
                "import_price_c": round(live_import, 1),
                "export_price_c": round(live_export, 1),
                "dampened_import_c": round(import_cents[0], 1) if import_cents else 0.0,
                "dampened_export_c": round(export_cents[0], 1) if export_cents else 0.0,
                "spike_status": import_prices_raw[0].spike_status if import_prices_raw else "none",
                "action": "HOLD",
                "optimizer_action": None,
                "reason": "Inverter offline: data ingest active, control skipped",
                "work_mode": "OFFLINE",
                "remote_control_active": False,
                "override": self._manual_override or "AUTO",
                "override_expires": self._override_expires.isoformat() if self._override_expires else None,
                "bms_max_charge_kw": 0.0,
                "bms_max_discharge_kw": 0.0,
            }
            foxess_status = "offline"
            foxess_last_read = None
            horizon_profit = 0.0
            schedule_action = []
            schedule_profit = []
            soc_trajectory = []

        pricing_provider = str(config.retailer.retailer or "").strip().lower()
        if pricing_provider == "flowpower":
            pricing_source_label = "FlowPower"
        elif pricing_provider == "amber":
            pricing_source_label = "Amber"
        elif pricing_provider == "custom":
            csv_name = Path(str(config.retailer.custom_pricing_csv or "").strip()).stem
            pricing_source_label = csv_name or "Custom CSV"
        else:
            pricing_source_label = pricing_provider.upper() if pricing_provider else "Unknown"

        return {
            "updated_at": cycle_start.isoformat(),
            "cycle_ms": self._cycle_ms,
            "live": live,

            "forecast": {
                "timestamps": timestamps,
                "raw_import_price_c": [
                    round(dampened_import[i].per_kwh if self._using_custom else dampened_import[i].raw_per_kwh, 1)
                    for i in range(n_periods)
                ],
                "raw_export_price_c": [
                    round(dampened_export[i].per_kwh if self._using_custom else dampened_export[i].raw_per_kwh, 1)
                    for i in range(n_periods)
                ],
                "import_price_c": [round(v, 1) for v in import_cents],
                "export_price_c": [round(v, 1) for v in export_cents],
                "solar_kw": [round(v, 2) for v in solar_kw],
                "solar_p10_kw": [round(v, 2) for v in solar_p10],
                "solar_p90_kw": [round(v, 2) for v in solar_p90],
                "load_kw": [round(v, 2) for v in load_kw],
                "schedule_action": schedule_action,
                "schedule_profit_c": schedule_profit,
                "soc_trajectory_kwh": soc_trajectory,
            },

            "financial": {
                "today_import_cost_c": round(self._today_import_cost_c, 1),
                "today_export_revenue_c": round(self._today_export_revenue_c, 1),
                "today_degradation_cost_c": round(self._today_degradation_c, 1),
                "today_savings_c": round(today_savings, 1),
                "horizon_profit_c": horizon_profit,
            },

            "battery_health": {
                "cycles_today": round(cycles_today, 2),
                "capacity_kwh": config.battery.capacity_kwh,
                "min_soc_kwh": config.battery.min_soc_kwh,
                "max_power_kw": config.battery.max_power_kw,
                "cycle_life": config.battery.cycle_life,
            },

            "system_health": {
                "pricing_provider": config.retailer.retailer,
                "pricing_source_label": pricing_source_label,
                "pricing_last_fetch": cycle_start.isoformat(),
                "pricing_status": "ok" if n_periods > 0 else "no_data",
                "pricing_intervals": n_periods,
                # Backward-compatible aliases used by current dashboard UI.
                "amber_last_fetch": cycle_start.isoformat(),
                "amber_status": "ok" if n_periods > 0 else "no_data",
                "amber_prices_count": n_periods,
                "solcast_last_fetch": solcast_last.isoformat() if solcast_last else None,
                "solcast_status": "ok" if solcast_last else "no_data",
                "solcast_calls_today": solcast_calls_today,
                "solcast_next_scheduled_hour": next_solcast,
                "foxess_status": foxess_status,
                "foxess_last_read": foxess_last_read,
                "optimizer_last_run": cycle_start.isoformat(),
                "optimizer_execution_ms": self._cycle_ms,
                "optimizer_horizon_hours": horizon_hours,
                "consecutive_failures": self._consecutive_failures,
                "errors": list(self._errors),
                "ml_price_model": "active" if self.ml_price.model_trained else "fallback",
                "ml_consumption_model": "active" if self.ml_consumption.model_trained else "fallback",
                "ha_enabled": config.homeassistant.enabled,
                "weather_last_fetch": self.weather.last_fetch_time.isoformat() if self.weather.last_fetch_time else None,
            },

            "history": {
                "timestamps": [r["timestamp"] for r in history],
                "soc_kwh": [round(r["soc_kwh"], 1) for r in history],
                "pv_power_kw": [round(r["pv_power_w"] / 1000, 2) for r in history],
                "load_power_kw": [round(r["load_power_w"] / 1000, 2) for r in history],
                "grid_power_kw": [round(r["grid_power_w"] / 1000, 2) for r in history],
                "battery_power_kw": [round(r["battery_power_w"] / 1000, 2) for r in history],
                "battery_temp_c": [
                    round(r["battery_temp_c"], 1) if r["battery_temp_c"] is not None else None
                    for r in history
                ],
                "inverter_internal_temp_c": [
                    round(r.get("inverter_internal_temp_c"), 1) if r.get("inverter_internal_temp_c") is not None else None
                    for r in history
                ],
                "import_price_c": [
                    round(decision_map[r["timestamp"]]["import_price"], 1)
                    if r["timestamp"] in decision_map and decision_map[r["timestamp"]]["import_price"] is not None
                    else None
                    for r in history
                ],
                "export_price_c": [
                    round(decision_map[r["timestamp"]]["export_price"], 1)
                    if r["timestamp"] in decision_map and decision_map[r["timestamp"]]["export_price"] is not None
                    else None
                    for r in history
                ],
                "action": [
                    decision_map[r["timestamp"]]["action"].upper()
                    if r["timestamp"] in decision_map and decision_map[r["timestamp"]]["action"]
                    else None
                    for r in history
                ],
            },
        }

    def _write_dashboard_status(
        self, cycle_start, state, result, action, optimizer_action,
        import_prices_raw, export_prices_raw,
        dampened_import, dampened_export,
        solar_fc, consumption_fc,
        import_cents, export_cents, solar_kw, load_kw,
        timestamps, n_periods, horizon_hours,
    ):
        """Write dashboard_status.json atomically for the monitoring dashboard."""
        try:
            status = self._build_dashboard_dict(
                cycle_start,
                import_prices_raw, export_prices_raw,
                dampened_import, dampened_export,
                solar_fc,
                import_cents, export_cents, solar_kw, load_kw,
                timestamps, n_periods, horizon_hours,
                live_data={
                    "state": state, "result": result,
                    "action": action, "optimizer_action": optimizer_action,
                },
            )
            tmp_path = STATUS_PATH.with_suffix(".json.tmp")
            with open(tmp_path, "w") as f:
                json.dump(status, f)
            os.replace(str(tmp_path), str(STATUS_PATH))
            logger.debug("Dashboard status written to %s", STATUS_PATH)
        except Exception as e:
            logger.warning("Failed to write dashboard status: %s", e)

    def _write_dashboard_status_inverter_offline(
        self, cycle_start,
        import_prices_raw, export_prices_raw,
        dampened_import, dampened_export,
        solar_fc,
        import_cents, export_cents, solar_kw, load_kw,
        timestamps, n_periods, horizon_hours,
    ):
        """Write dashboard_status.json for data-only cycles (inverter offline)."""
        try:
            status = self._build_dashboard_dict(
                cycle_start,
                import_prices_raw, export_prices_raw,
                dampened_import, dampened_export,
                solar_fc,
                import_cents, export_cents, solar_kw, load_kw,
                timestamps, n_periods, horizon_hours,
                live_data=None,
            )
            tmp_path = STATUS_PATH.with_suffix(".json.tmp")
            with open(tmp_path, "w") as f:
                json.dump(status, f)
            os.replace(str(tmp_path), str(STATUS_PATH))
            logger.debug("Dashboard status written (inverter offline) to %s", STATUS_PATH)
        except Exception as e:
            logger.warning("Failed to write offline dashboard status: %s", e)

    def _handle_failure(self):
        self._consecutive_failures += 1
        if self._consecutive_failures >= 3:
            logger.error(
                "%d consecutive failures; reverting to self-use",
                self._consecutive_failures,
            )
            self.foxess.emergency_self_use()
            self._last_action = None
            self._remote_control_active = False

    def _watchdog(self):
        """Background thread: if main loop hasn't run in 10 min, force self-use."""
        while self._running:
            time.sleep(60)
            elapsed = time.time() - self._watchdog_last_run
            if elapsed > 600:  # 10 minutes
                logger.error("Watchdog: no cycle in %.0fs; forcing self-use", elapsed)
                self.foxess.emergency_self_use()
                self._last_action = None
                self._remote_control_active = False

    def _set_override(self, mode: str, duration_minutes: int = 60) -> dict:
        """Set or clear manual override. Returns status dict."""
        valid_modes = {"AUTO", "GRID_CHARGE", "GRID_CHARGE_NO_EXPORT", "SELF_USE", "SELF_USE_NO_EXPORT", "HOLD", "DISCHARGE_GRID"}
        if mode not in valid_modes:
            return {"ok": False, "error": f"Invalid mode: {mode}. Valid: {sorted(valid_modes)}"}

        if mode == "AUTO":
            self._manual_override = None
            self._override_expires = None
            logger.info("Manual override cleared — returning to AUTO")
            return {"ok": True, "override": "AUTO", "expires": None}

        self._manual_override = mode
        self._override_expires = datetime.now() + timedelta(minutes=duration_minutes)
        logger.info("Manual override set: %s for %d minutes", mode, duration_minutes)
        return {
            "ok": True,
            "override": mode,
            "expires": self._override_expires.isoformat(),
        }

    @staticmethod
    def _read_env_file() -> dict[str, str]:
        """Read the .env file and return key-value pairs."""
        env_path = Path(__file__).parent / ".env"
        settings = {}
        if env_path.exists():
            with open(env_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, _, value = line.partition("=")
                        settings[key.strip()] = value.strip()
        return settings

    @staticmethod
    def _write_env_file(settings: dict[str, str]):
        """Write settings back to the .env file, preserving comments and order."""
        env_path = Path(__file__).parent / ".env"
        lines = []
        written_keys = set()

        if env_path.exists():
            with open(env_path, "r") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped and not stripped.startswith("#") and "=" in stripped:
                        key = stripped.partition("=")[0].strip()
                        if key in settings:
                            lines.append(f"{key}={settings[key]}\n")
                            written_keys.add(key)
                        else:
                            lines.append(line)
                    else:
                        lines.append(line)

        # Append any new keys not already in file
        new_keys = set(settings.keys()) - written_keys
        if new_keys:
            if lines and not lines[-1].endswith("\n"):
                lines.append("\n")
            lines.append("\n# Added by settings page\n")
            for key in sorted(new_keys):
                lines.append(f"{key}={settings[key]}\n")

        with open(env_path, "w") as f:
            f.writelines(lines)

    @staticmethod
    def _parse_csv_preview(csv_path: str) -> list[dict]:
        """Read a custom pricing CSV and return rows for preview.

        Auto-detects whether the first row is a header by checking if
        the first column looks like a time (contains ':' and starts with a digit).
        """
        import csv as csv_mod
        rows = []
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            reader = csv_mod.reader(f)
            for row in reader:
                if len(row) < 3:
                    continue
                col0 = row[0].strip()
                # Skip header rows (first column is not a time like "00:00")
                if col0 and not col0[0].isdigit():
                    continue
                rows.append({
                    "time": col0,
                    "import_price": row[1].strip(),
                    "export_price": row[2].strip(),
                })
        return rows

    def _start_dashboard_server(self):
        """Start dashboard HTTP server in a background daemon thread."""
        web_dir = str(Path(__file__).parent / "web")
        port = config.system.dashboard_port
        bind_host = config.system.dashboard_bind_host
        api_token = config.system.dashboard_api_token.strip()
        local_hosts = {"127.0.0.1", "::1", "localhost"}
        if bind_host not in local_hosts and not api_token:
            logger.error(
                "Refusing to start dashboard on non-local host '%s' without DASHBOARD_API_TOKEN",
                bind_host,
            )
            return
        system = self  # capture for handler closure
        project_root = Path(__file__).parent.resolve()

        class DashboardHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=web_dir, **kwargs)

            def log_message(self, format, *args):
                pass  # suppress access logs

            def _path_only(self) -> str:
                return urllib.parse.urlparse(self.path).path

            def _provided_token(self) -> str:
                header_token = (self.headers.get("X-API-Token") or "").strip()
                if header_token:
                    return header_token

                auth_header = (self.headers.get("Authorization") or "").strip()
                if auth_header.lower().startswith("bearer "):
                    return auth_header[7:].strip()

                query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                return (query.get("token") or [""])[0].strip()

            def _authorized(self) -> bool:
                if not api_token:
                    return True
                return hmac.compare_digest(self._provided_token(), api_token)

            def _send_json(self, data, status=200):
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(data).encode())

            def do_POST(self):
                path = self._path_only()
                if path.startswith("/api/") and not self._authorized():
                    self._send_json({"ok": False, "error": "Unauthorized"}, 401)
                    return

                if path == "/api/override":
                    try:
                        length = int(self.headers.get("Content-Length", 0))
                        body = json.loads(self.rfile.read(length)) if length else {}
                        mode = body.get("mode", "AUTO").upper()
                        duration = int(body.get("duration_minutes", 60))
                        result = system._set_override(mode, duration)
                        self._send_json(result)
                    except Exception as e:
                        self._send_json({"ok": False, "error": str(e)}, 400)

                elif path == "/api/settings":
                    try:
                        length = int(self.headers.get("Content-Length", 0))
                        body = json.loads(self.rfile.read(length)) if length else {}
                        system._write_env_file(body)
                        logger.info("Settings saved via web UI — scheduling restart")
                        system._restart_requested = True
                        self._send_json({"ok": True})
                    except Exception as e:
                        self._send_json({"ok": False, "error": str(e)}, 400)

                elif path == "/api/settings/upload-pricing":
                    try:
                        length = int(self.headers.get("Content-Length", 0))
                        body = json.loads(self.rfile.read(length)) if length else {}
                        raw_filename = str(body.get("filename", "custom_pricing.csv")).strip()
                        content = body.get("content", "")
                        if not content:
                            self._send_json({"ok": False, "error": "No file content"}, 400)
                            return
                        if not raw_filename:
                            self._send_json({"ok": False, "error": "Invalid filename"}, 400)
                            return
                        if Path(raw_filename).name != raw_filename:
                            self._send_json({"ok": False, "error": "Filename must not contain path separators"}, 400)
                            return
                        filename = Path(raw_filename).name
                        if not filename.lower().endswith(".csv"):
                            self._send_json({"ok": False, "error": "Only .csv files are allowed"}, 400)
                            return

                        # Normalize line endings and save to project root
                        content = content.replace("\r\n", "\n").replace("\r", "\n")
                        dest = (project_root / filename).resolve()
                        if project_root not in dest.parents:
                            self._send_json({"ok": False, "error": "Invalid destination path"}, 400)
                            return
                        with open(dest, "w", encoding="utf-8", newline="\n") as f:
                            f.write(content)
                        logger.info("Custom pricing CSV saved: %s", dest)

                        # Validate and preview
                        preview = system._parse_csv_preview(str(dest))

                        # Update env file with the filename
                        env_settings = system._read_env_file()
                        env_settings["CUSTOM_PRICING_CSV"] = filename
                        system._write_env_file(env_settings)

                        self._send_json({
                            "ok": True,
                            "filename": filename,
                            "preview": preview,
                        })
                    except Exception as e:
                        self._send_json({"ok": False, "error": str(e)}, 400)

                else:
                    self.send_error(404)

            def do_GET(self):
                path = self._path_only()
                if path.startswith("/api/") and not self._authorized():
                    self._send_json({"ok": False, "error": "Unauthorized"}, 401)
                    return

                if path == "/api/override":
                    result = {
                        "override": system._manual_override or "AUTO",
                        "expires": system._override_expires.isoformat() if system._override_expires else None,
                    }
                    self._send_json(result)

                elif path == "/api/settings":
                    self._send_json(system._read_env_file())

                elif path == "/api/settings/pricing-preview":
                    try:
                        env = system._read_env_file()
                        csv_file = env.get("CUSTOM_PRICING_CSV", "")
                        if csv_file:
                            csv_path = project_root / csv_file
                            if csv_path.exists():
                                preview = system._parse_csv_preview(str(csv_path))
                                self._send_json({"ok": True, "preview": preview})
                                return
                        self._send_json({"ok": False, "error": "No pricing CSV configured"})
                    except Exception as e:
                        self._send_json({"ok": False, "error": str(e)}, 400)

                else:
                    super().do_GET()

        server = http.server.HTTPServer((bind_host, port), DashboardHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        display_host = "localhost" if bind_host in ("127.0.0.1", "::1", "0.0.0.0", "") else bind_host
        logger.info("Dashboard server at http://%s:%d/dashboard.html", display_host, port)
        if api_token:
            logger.info("Dashboard API token authentication is enabled")

    def start(self):
        """Start the scheduler."""
        logger.info("Power Arbitrage System starting")
        logger.info("Battery: %.0f kWh, %.1f kW inverter",
                     config.battery.capacity_kwh, config.battery.max_power_kw)
        logger.info("Degradation: $%.4f/kWh (%.2fc/kWh), $%.2f/cycle",
                     config.battery.degradation_per_kwh,
                     config.battery.degradation_per_kwh * 100,
                     config.battery.cost_per_cycle)
        logger.info("Scheduler interval: %ds", config.system.scheduler_interval_s)
        if config.system.dry_run:
            logger.info("*** DRY RUN MODE — no Modbus writes will be sent ***")

        # Start dashboard server
        if config.system.dashboard_enabled:
            self._start_dashboard_server()

        # Start watchdog
        watchdog = threading.Thread(target=self._watchdog, daemon=True)
        watchdog.start()

        # Run first cycle immediately
        self.run_cycle()

        # Schedule recurring cycles
        interval_min = config.system.scheduler_interval_s / 60
        schedule.every(interval_min).minutes.do(self.run_cycle)

        # Schedule daily ML model retraining + consumption profile refresh
        schedule.every().day.at("04:00").do(self.ml_consumption.refresh)
        schedule.every().day.at("04:00").do(self.ml_price.train)

        # Schedule daily database pruning (remove records older than 1 year)
        schedule.every().day.at("03:00").do(self.db.prune_old_records)

        # Schedule dampener recalibration weekly
        schedule.every(7).days.do(self.dampener._calibrate)

        # Schedule weather forecast refresh every 3 hours
        schedule.every(config.weather.refresh_interval_hours).hours.do(
            self.weather.fetch_forecast
        )

        logger.info("Scheduler running. Press Ctrl+C to stop.")
        while self._running:
            if self._restart_requested:
                self._restart()
            try:
                schedule.run_pending()
            except Exception as e:
                logger.exception("Scheduled task failed: %s", e)
                self._errors.append(f"{datetime.now().isoformat()}: scheduled task failed: {e}")
            time.sleep(1)

    def _restart(self):
        """Restart the process to pick up new .env settings."""
        logger.info("Restarting to apply new settings...")
        self.foxess.emergency_self_use()
        self.foxess.disconnect()
        os.execv(sys.executable, [sys.executable] + sys.argv)

    def stop(self):
        self._running = False
        logger.info("Shutting down; reverting to self-use mode")
        self.foxess.emergency_self_use()
        self.foxess.disconnect()


def main():
    system = PowerArbSystem()

    def signal_handler(sig, frame):
        system.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    system.start()


if __name__ == "__main__":
    main()
