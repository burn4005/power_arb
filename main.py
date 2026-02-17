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
import http.server
import json
import logging
import os
import signal
import sys
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path

import schedule

import config
from amber.client import AmberClient
from amber.price_dampener import PriceDampener
from foxess.modbus_client import FoxESSModbusClient, WorkMode, InverterState
from pricing.custom_csv import generate_price_intervals
from solcast.client import SolcastClient, FETCH_HOURS_AEST
from forecasting.consumption import ConsumptionForecaster
from forecasting.solar import SolarForecaster
from optimizer.actions import Action
from optimizer.battery_model import BatteryModel, PeriodInputs, PERIOD_HOURS
from optimizer.dp_optimizer import DPOptimizer
from storage.database import Database

logging.basicConfig(
 #   filename='app.log', filemode='w',
    level=getattr(logging, config.system.log_level),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("power_arb")

# Map optimizer actions to FoxESS work modes
ACTION_TO_MODE = {
    Action.GRID_CHARGE: WorkMode.FORCE_CHARGE,
    Action.SELF_USE: WorkMode.SELF_USE,
    Action.SELF_USE_NO_EXPORT: WorkMode.SELF_USE,  # same mode, but export limit set to 0W
    Action.HOLD: WorkMode.SELF_USE,                 # same mode, but min SoC set high to prevent discharge
    Action.DISCHARGE_GRID: WorkMode.FORCE_DISCHARGE,
}

STATUS_PATH = Path(__file__).parent / "web" / "dashboard_status.json"


class PowerArbSystem:
    def __init__(self):
        self.db = Database()
        self.amber = AmberClient(self.db)
        self.dampener = PriceDampener(self.db)
        self.foxess = FoxESSModbusClient()
        self.solcast_client = SolcastClient(self.db)
        self.solar_forecaster = SolarForecaster(self.solcast_client)
        self.consumption = ConsumptionForecaster(self.db)
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
        self._export_limited = False

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
                logger.error("Cannot read inverter state; skipping cycle")
                self._handle_failure()
                state = InverterState(
                    soc_pct=0,
                    soc_kwh=0,
                    battery_power_w=0,
                    pv_power_w=0,
                    grid_power_w=0,
                    load_power_w=0,
                    battery_temp_c=0,
                    work_mode=WorkMode.SELF_USE,
                    min_soc_pct=0,
                    export_limit_w=0
                )
                #return

            self._consecutive_failures = 0
            current_soc = state.soc_kwh

            # Sync export limit flag from inverter on first read
            if self._last_action is None:
                self._export_limited = state.export_limit_w == 0

            logger.info(
                "Inverter: SoC=%.1f kWh (%.0f%%), PV=%dW, Grid=%dW, Load=%dW, Mode=%s, ExportLimit=%dW",
                state.soc_kwh, state.soc_pct, state.pv_power_w,
                state.grid_power_w, state.load_power_w, state.work_mode.name,
                state.export_limit_w,
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
            })

            # Log consumption for forecaster training
            self.db.insert_consumption(
                cycle_start.isoformat(), state.load_power_w, "measured"
            )

            # 2. Fetch prices
            if config.retailer.retailer == "custom" and config.retailer.custom_pricing_csv:
                prices = generate_price_intervals(
                    config.retailer.custom_pricing_csv,
                    start=cycle_start,
                    hours=48,
                )
                import_prices_raw = prices["import"]
                export_prices_raw = prices["export"]
                # Custom prices are fixed schedules — no dampening needed
                dampened_import = None
                dampened_export = None
            else:
                prices = self.amber.fetch_current_and_forecast()
                import_prices_raw = prices["import"]
                export_prices_raw = prices["export"]
                dampened_import = None
                dampened_export = None

            if not import_prices_raw:
                logger.error("No price data; skipping cycle")
                return

            # 3. Dampen forecast prices (only for Amber; custom prices used as-is)
            if config.retailer.retailer == "custom" and config.retailer.custom_pricing_csv:
                dampened_import = import_prices_raw
                dampened_export = export_prices_raw
            else:
                dampened_import = self.dampener.dampen(import_prices_raw, cycle_start)
                dampened_export = self.dampener.dampen(export_prices_raw, cycle_start)

            # 4. Get solar forecast
            solar_fc = self.solar_forecaster.forecast(hours=48, start=cycle_start)

            # 5. Predict consumption
            consumption_fc = self.consumption.forecast(hours=48, start=cycle_start)

            # 6. Align all forecasts to same time periods
            n_periods = min(
                len(dampened_import), len(dampened_export),
                len(solar_fc), len(consumption_fc),
            )
            if n_periods == 0:
                logger.error("No aligned forecast periods; skipping cycle")
                return

            self._using_custom = config.retailer.retailer == "custom" and config.retailer.custom_pricing_csv
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

            mode = ACTION_TO_MODE[action]

            # Set min SoC based on action
            if action == Action.HOLD:
                # Prevent discharge by setting min SoC to current level
                min_soc = max(10, int(state.soc_pct))
            else:
                # SELF_USE, GRID_CHARGE, DISCHARGE_GRID all use normal min SoC
                min_soc = int(config.battery.min_soc_pct)

            # Determine export limit based on action
            normal_limit_w = int(config.battery.grid_export_limit_kw * 1000)
            wants_no_export = action == Action.SELF_USE_NO_EXPORT

            # Only write to inverter if action changed
            mode_changed = False
            if action != self._last_action:
                logger.info("Action changed: %s -> %s",
                           self._last_action.name if self._last_action else "None",
                           action.name)
                success = self.foxess.set_work_mode(mode)
                if success:
                    self.foxess.set_min_soc(min_soc)
                    # Set export limit based on action
                    if wants_no_export and not self._export_limited:
                        self.foxess.set_export_limit(0)
                        self._export_limited = True
                    elif not wants_no_export and self._export_limited:
                        self.foxess.set_export_limit(normal_limit_w)
                        self._export_limited = False
                    self._last_action = action
                    mode_changed = True
                else:
                    logger.error("Failed to set work mode; keeping previous mode")

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
                "actual_mode_set": mode.value if mode_changed else None,
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
                timestamps, n_periods,
            )

        except Exception as e:
            logger.exception("Cycle failed: %s", e)
            self._errors.append(f"{datetime.now().isoformat()}: {e}")
            self._handle_failure()

    def _write_dashboard_status(
        self, cycle_start, state, result, action, optimizer_action,
        import_prices_raw, export_prices_raw,
        dampened_import, dampened_export,
        solar_fc, consumption_fc,
        import_cents, export_cents, solar_kw, load_kw,
        timestamps, n_periods,
    ):
        """Write dashboard_status.json atomically for the monitoring dashboard."""
        try:
            # Solcast health info
            solcast_last = self.solcast_client._last_fetch_time
            current_hour = cycle_start.hour
            next_solcast = next(
                (h for h in FETCH_HOURS_AEST if h > current_hour),
                FETCH_HOURS_AEST[0],  # wrap to tomorrow
            )
            solcast_calls_today = sum(
                1 for h in FETCH_HOURS_AEST if h <= current_hour
            )

            # Solar P10/P90 arrays
            solar_p10 = [solar_fc[i].get("solar_p10_kw", 0.0) for i in range(n_periods)]
            solar_p90 = [solar_fc[i].get("solar_p90_kw", 0.0) for i in range(n_periods)]

            # History: last 48h of battery log + decisions (for prices)
            since_48h = (cycle_start - timedelta(hours=48)).isoformat()
            history = self.db.get_battery_log_since(since_48h)
            decisions = self.db.get_decisions_since(since_48h)

            # Build a price lookup from decisions (keyed by timestamp)
            decision_map = {d["timestamp"]: d for d in decisions}

            # Financial
            today_savings = self._today_baseline_c - (
                self._today_import_cost_c
                - self._today_export_revenue_c
                + self._today_degradation_c
            )

            # Cycles today
            capacity = config.battery.capacity_kwh
            cycles_today = (
                self._today_energy_cycled_kwh / (2 * capacity)
                if capacity > 0 else 0
            )

            status = {
                "updated_at": cycle_start.isoformat(),
                "cycle_ms": self._cycle_ms,

                "live": {
                    "pv_power_kw": round(state.pv_power_w / 1000, 2),
                    "load_power_kw": round(state.load_power_w / 1000, 2),
                    "grid_power_kw": round(state.grid_power_w / 1000, 2),
                    "battery_power_kw": round(state.battery_power_w / 1000, 2),
                    "soc_pct": round(state.soc_pct, 1),
                    "soc_kwh": round(state.soc_kwh, 1),
                    "battery_temp_c": round(state.battery_temp_c, 1),
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
                    "override": self._manual_override or "AUTO",
                    "override_expires": self._override_expires.isoformat() if self._override_expires else None,
                    "export_limited": self._export_limited,
                    "export_limit_w": state.export_limit_w,
                },

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
                    "schedule_action": [a.name for _, a, _ in result.schedule],
                    "schedule_profit_c": [round(p, 2) for _, _, p in result.schedule],
                    "soc_trajectory_kwh": [round(s, 1) for s in result.soc_trajectory],
                },

                "financial": {
                    "today_import_cost_c": round(self._today_import_cost_c, 1),
                    "today_export_revenue_c": round(self._today_export_revenue_c, 1),
                    "today_degradation_cost_c": round(self._today_degradation_c, 1),
                    "today_savings_c": round(today_savings, 1),
                    "horizon_profit_c": round(result.expected_profit_cents, 1),
                },

                "battery_health": {
                    "cycles_today": round(cycles_today, 2),
                    "capacity_kwh": config.battery.capacity_kwh,
                    "min_soc_kwh": config.battery.min_soc_kwh,
                    "max_power_kw": config.battery.max_power_kw,
                    "cycle_life": config.battery.cycle_life,
                },

                "system_health": {
                    "amber_last_fetch": cycle_start.isoformat(),
                    "amber_status": "ok",
                    "amber_prices_count": n_periods,
                    "solcast_last_fetch": solcast_last.isoformat() if solcast_last else None,
                    "solcast_status": "ok" if solcast_last else "no_data",
                    "solcast_calls_today": solcast_calls_today,
                    "solcast_next_scheduled_hour": next_solcast,
                    "foxess_status": "ok",
                    "foxess_last_read": cycle_start.isoformat(),
                    "optimizer_last_run": cycle_start.isoformat(),
                    "optimizer_execution_ms": self._cycle_ms,
                    "consecutive_failures": self._consecutive_failures,
                    "errors": list(self._errors),
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

            # Atomic write
            tmp_path = STATUS_PATH.with_suffix(".json.tmp")
            with open(tmp_path, "w") as f:
                json.dump(status, f)
            os.replace(str(tmp_path), str(STATUS_PATH))
            logger.debug("Dashboard status written to %s", STATUS_PATH)

        except Exception as e:
            logger.warning("Failed to write dashboard status: %s", e)

    def _handle_failure(self):
        self._consecutive_failures += 1
        if self._consecutive_failures >= 3:
            logger.error(
                "%d consecutive failures; reverting to self-use",
                self._consecutive_failures,
            )
            self.foxess.emergency_self_use()
            self._last_action = None

    def _watchdog(self):
        """Background thread: if main loop hasn't run in 10 min, force self-use."""
        while self._running:
            time.sleep(60)
            elapsed = time.time() - self._watchdog_last_run
            if elapsed > 600:  # 10 minutes
                logger.error("Watchdog: no cycle in %.0fs; forcing self-use", elapsed)
                self.foxess.emergency_self_use()
                self._last_action = None

    def _set_override(self, mode: str, duration_minutes: int = 60) -> dict:
        """Set or clear manual override. Returns status dict."""
        valid_modes = {"AUTO", "GRID_CHARGE", "SELF_USE", "SELF_USE_NO_EXPORT", "HOLD", "DISCHARGE_GRID"}
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
        system = self  # capture for handler closure
        project_root = Path(__file__).parent

        class DashboardHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=web_dir, **kwargs)

            def log_message(self, format, *args):
                pass  # suppress access logs

            def _send_json(self, data, status=200):
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(data).encode())

            def do_POST(self):
                if self.path == "/api/override":
                    try:
                        length = int(self.headers.get("Content-Length", 0))
                        body = json.loads(self.rfile.read(length)) if length else {}
                        mode = body.get("mode", "AUTO").upper()
                        duration = int(body.get("duration_minutes", 60))
                        result = system._set_override(mode, duration)
                        self._send_json(result)
                    except Exception as e:
                        self._send_json({"ok": False, "error": str(e)}, 400)

                elif self.path == "/api/settings":
                    try:
                        length = int(self.headers.get("Content-Length", 0))
                        body = json.loads(self.rfile.read(length)) if length else {}
                        system._write_env_file(body)
                        logger.info("Settings saved via web UI — scheduling restart")
                        system._restart_requested = True
                        self._send_json({"ok": True})
                    except Exception as e:
                        self._send_json({"ok": False, "error": str(e)}, 400)

                elif self.path == "/api/settings/upload-pricing":
                    try:
                        length = int(self.headers.get("Content-Length", 0))
                        body = json.loads(self.rfile.read(length)) if length else {}
                        filename = body.get("filename", "custom_pricing.csv")
                        content = body.get("content", "")
                        if not content:
                            self._send_json({"ok": False, "error": "No file content"}, 400)
                            return

                        # Normalize line endings and save to project root
                        content = content.replace("\r\n", "\n").replace("\r", "\n")
                        dest = project_root / filename
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
                if self.path == "/api/override":
                    result = {
                        "override": system._manual_override or "AUTO",
                        "expires": system._override_expires.isoformat() if system._override_expires else None,
                    }
                    self._send_json(result)

                elif self.path == "/api/settings":
                    self._send_json(system._read_env_file())

                elif self.path == "/api/settings/pricing-preview":
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

        server = http.server.HTTPServer(("", port), DashboardHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        logger.info("Dashboard server at http://localhost:%d/dashboard.html", port)

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

        # Schedule daily consumption profile refresh
        schedule.every().day.at("04:00").do(self.consumption.refresh)

        # Schedule dampener recalibration weekly
        schedule.every(7).days.do(self.dampener._calibrate)

        logger.info("Scheduler running. Press Ctrl+C to stop.")
        while self._running:
            if self._restart_requested:
                self._restart()
            schedule.run_pending()
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
