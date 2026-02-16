"""Power Arbitrage System - Main Scheduler

Runs every 5 minutes:
1. Read inverter state (FoxESS Modbus)
2. Fetch current + forecast prices (Amber API)
3. Get solar forecast (Solcast, cached)
4. Predict home consumption
5. Run DP optimizer
6. Execute optimal action on inverter
"""

import logging
import signal
import sys
import time
import threading
from datetime import datetime

import schedule

import config
from amber.client import AmberClient
from amber.price_dampener import PriceDampener
from foxess.modbus_client import FoxESSModbusClient, WorkMode
from solcast.client import SolcastClient
from forecasting.consumption import ConsumptionForecaster
from forecasting.solar import SolarForecaster
from optimizer.actions import Action
from optimizer.dp_optimizer import DPOptimizer
from storage.database import Database

logging.basicConfig(
    level=getattr(logging, config.system.log_level),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("power_arb")

# Map optimizer actions to FoxESS work modes
ACTION_TO_MODE = {
    Action.GRID_CHARGE: WorkMode.FORCE_CHARGE,
    Action.SELF_USE: WorkMode.SELF_USE,
    Action.HOLD: WorkMode.SELF_USE,         # same mode, but min SoC set high to prevent discharge
    Action.DISCHARGE_GRID: WorkMode.FORCE_DISCHARGE,
}


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

    def run_cycle(self):
        """Execute one optimization cycle."""
        cycle_start = datetime.now()
        logger.info("=== Optimization cycle starting at %s ===", cycle_start.isoformat())

        try:
            # 1. Read inverter state
            state = self.foxess.read_state()
            if state is None:
                logger.error("Cannot read inverter state; skipping cycle")
                self._handle_failure()
                return

            self._consecutive_failures = 0
            current_soc = state.soc_kwh
            logger.info(
                "Inverter: SoC=%.1f kWh (%.0f%%), PV=%dW, Grid=%dW, Load=%dW, Mode=%s",
                state.soc_kwh, state.soc_pct, state.pv_power_w,
                state.grid_power_w, state.load_power_w, state.work_mode.name,
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
            })

            # Log consumption for forecaster training
            self.db.insert_consumption(
                cycle_start.isoformat(), state.load_power_w, "measured"
            )

            # 2. Fetch prices
            prices = self.amber.fetch_current_and_forecast()
            import_prices_raw = prices["import"]
            export_prices_raw = prices["export"]

            if not import_prices_raw:
                logger.error("No price data from Amber; skipping cycle")
                return

            # 3. Dampen forecast prices
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

            # 8. Execute action
            action = result.action
            mode = ACTION_TO_MODE[action]

            # Set min SoC based on action
            if action == Action.HOLD:
                # Prevent discharge by setting min SoC to current level
                min_soc = max(10, int(state.soc_pct))
            else:
                # SELF_USE, GRID_CHARGE, DISCHARGE_GRID all use normal min SoC
                min_soc = int(config.battery.min_soc_pct)

            # Only write to inverter if action changed
            mode_changed = False
            if action != self._last_action:
                logger.info("Action changed: %s -> %s",
                           self._last_action.name if self._last_action else "None",
                           action.name)
                success = self.foxess.set_work_mode(mode)
                if success:
                    self.foxess.set_min_soc(min_soc)
                    self._last_action = action
                    mode_changed = True
                else:
                    logger.error("Failed to set work mode; keeping previous mode")

            # Log decision
            self.db.insert_decision({
                "timestamp": cycle_start.isoformat(),
                "action": action.value,
                "reason": result.reason,
                "import_price": import_prices_raw[0].per_kwh if import_prices_raw else None,
                "export_price": export_prices_raw[0].per_kwh if export_prices_raw else None,
                "dampened_import_price": import_cents[0] if import_cents else None,
                "dampened_export_price": export_cents[0] if export_cents else None,
                "soc_kwh": current_soc,
                "expected_profit": result.expected_profit_cents,
                "actual_mode_set": mode.value if mode_changed else None,
            })

            self._watchdog_last_run = time.time()

            elapsed = (datetime.now() - cycle_start).total_seconds()
            logger.info(
                "Cycle complete in %.1fs | Action: %s | Horizon profit: %.0fc",
                elapsed, action.name, result.expected_profit_cents,
            )

            # Log the next few scheduled actions
            for ts, act, profit in result.schedule[:6]:
                logger.debug("  %s: %s (%.1fc)", ts, act.name, profit)

        except Exception as e:
            logger.exception("Cycle failed: %s", e)
            self._handle_failure()

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
            schedule.run_pending()
            time.sleep(1)

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
