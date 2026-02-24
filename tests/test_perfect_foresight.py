"""Perfect foresight backtest: theoretical maximum profit with the battery.

Runs the DP optimizer with actual future prices (no forecast error) over
the full overlapping test period. Compares against no-battery baseline.

Usage:
    python -m tests.test_perfect_foresight
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta

os.environ.setdefault("AMBER_API_KEY", "psk_test")
os.environ.setdefault("SOLCAST_API_KEY", "test")
os.environ.setdefault("SOLCAST_RESOURCE_ID", "test")
os.environ.setdefault("FOXESS_IP", "127.0.0.1")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import config
from optimizer.dp_optimizer import DPOptimizer
from optimizer.battery_model import (
    BatteryModel, PeriodInputs, PERIOD_HOURS, PERIODS_PER_DAY, PERIODS_PER_HOUR,
)
from optimizer.actions import Action
from backtest.data_loader import (
    load_aemo_prices, load_home_usage, load_solar_yield,
    get_day_prices, get_day_solar, get_day_load, IMPORT_MARKUP_C,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
# Suppress per-period optimizer logging (288 * 625 = 180K lines)
logging.getLogger("optimizer").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def run_perfect_foresight():
    logger.info("Loading test data...")
    aemo = load_aemo_prices()
    usage = load_home_usage()
    solar = load_solar_yield()

    optimizer = DPOptimizer()
    model = BatteryModel()
    capacity = config.battery.capacity_kwh
    min_soc = config.battery.min_soc_kwh

    # Find overlapping dates
    aemo_dates = sorted(set(k[:10] for k in aemo.keys()))
    usage_dates = sorted(set(k[:10] for k in usage.keys()))
    usage_date_set = set(d[:10] for d in usage_dates)
    aemo_date_set = set(d.replace("/", "-") for d in aemo_dates)
    overlap_dates = sorted(usage_date_set & aemo_date_set)

    logger.info("Overlapping dates: %d days (%s to %s)",
                len(overlap_dates), overlap_dates[0], overlap_dates[-1])

    # Simulation state
    soc = capacity * 0.5
    total_opt_cost_c = 0.0
    total_baseline_cost_c = 0.0
    total_import_kwh = 0.0
    total_export_kwh = 0.0
    total_degradation_c = 0.0
    total_export_revenue_c = 0.0
    total_import_cost_c = 0.0
    action_counts = {a: 0 for a in Action}
    days_simulated = 0

    # Monthly tracking
    monthly_opt = {}
    monthly_base = {}

    run_start = time.perf_counter()

    for date_str in overlap_dates:
        aemo_date = date_str.replace("-", "/")
        month = int(date_str[5:7])
        day = int(date_str[8:10])
        month_key = date_str[:7]  # "YYYY-MM"

        # Load day data
        prices = get_day_prices(aemo, aemo_date)
        import_p = [p[0] for p in prices]
        export_p = [p[1] for p in prices]
        solar_kw = get_day_solar(solar, month, day)
        load_kw = get_day_load(usage, date_str)

        # Perfect foresight: optimizer sees ALL remaining periods in the day
        day_opt_cost = 0.0
        day_base_cost = 0.0

        for t in range(PERIODS_PER_DAY):
            # Give optimizer full remaining horizon with ACTUAL prices
            result = optimizer.optimize(
                current_soc_kwh=soc,
                import_prices=import_p[t:],
                export_prices=export_p[t:],
                solar_forecast=solar_kw[t:],
                load_forecast=load_kw[t:],
            )

            action = result.action
            action_counts[action] = action_counts.get(action, 0) + 1

            # Execute action
            inputs = PeriodInputs(
                solar_kw=solar_kw[t],
                load_kw=load_kw[t],
                import_price=import_p[t],
                export_price=export_p[t],
            )
            period = model.apply_action(soc, action, inputs)
            soc = period.new_soc_kwh

            total_import_kwh += period.grid_import_kwh
            total_export_kwh += period.grid_export_kwh
            imp_cost = period.grid_import_kwh * import_p[t]
            exp_rev = period.grid_export_kwh * export_p[t]
            total_import_cost_c += imp_cost
            total_export_revenue_c += exp_rev
            total_degradation_c += period.degradation_cost_cents
            day_opt_cost += imp_cost - exp_rev + period.degradation_cost_cents

            # No-battery baseline
            solar_kwh_t = solar_kw[t] * PERIOD_HOURS
            load_kwh_t = load_kw[t] * PERIOD_HOURS
            solar_to_load = min(solar_kwh_t, load_kwh_t)
            base_import = load_kwh_t - solar_to_load
            base_export = solar_kwh_t - solar_to_load
            day_base_cost += base_import * import_p[t] - base_export * export_p[t]

        total_opt_cost_c += day_opt_cost
        total_baseline_cost_c += day_base_cost
        monthly_opt[month_key] = monthly_opt.get(month_key, 0.0) + day_opt_cost
        monthly_base[month_key] = monthly_base.get(month_key, 0.0) + day_base_cost
        days_simulated += 1

        if days_simulated % 50 == 0:
            elapsed = time.perf_counter() - run_start
            logger.info("  [%d/%d days] %.1fs elapsed, running savings: $%.2f",
                        days_simulated, len(overlap_dates), elapsed,
                        (total_baseline_cost_c - total_opt_cost_c) / 100)

    elapsed = time.perf_counter() - run_start

    # Results
    savings_c = total_baseline_cost_c - total_opt_cost_c
    savings_aud = savings_c / 100
    daily_avg_savings = savings_aud / max(days_simulated, 1)
    annual_savings = daily_avg_savings * 365

    logger.info("")
    logger.info("=" * 70)
    logger.info("PERFECT FORESIGHT BACKTEST RESULTS")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Period: %s to %s (%d days)", overlap_dates[0], overlap_dates[-1], days_simulated)
    logger.info("Runtime: %.1fs", elapsed)
    logger.info("")
    logger.info("%-30s %12s", "Metric", "Value")
    logger.info("-" * 45)
    logger.info("%-30s %10.2f $", "No-battery cost", total_baseline_cost_c / 100)
    logger.info("%-30s %10.2f $", "With battery (perfect)", total_opt_cost_c / 100)
    logger.info("%-30s %10.2f $", "Total savings", savings_aud)
    logger.info("%-30s %10.2f $", "Battery degradation cost", total_degradation_c / 100)
    logger.info("")
    logger.info("%-30s %10.2f $", "Avg daily savings", daily_avg_savings)
    logger.info("%-30s %10.2f $", "Projected annual savings", annual_savings)
    logger.info("")
    logger.info("%-30s %10.1f kWh", "Total grid import", total_import_kwh)
    logger.info("%-30s %10.1f kWh", "Total grid export", total_export_kwh)
    logger.info("%-30s %10.2f $", "Total import cost", total_import_cost_c / 100)
    logger.info("%-30s %10.2f $", "Total export revenue", total_export_revenue_c / 100)
    logger.info("")

    logger.info("ACTION DISTRIBUTION:")
    logger.info("-" * 45)
    total_actions = sum(action_counts.values())
    for action in Action:
        count = action_counts.get(action, 0)
        pct = count / max(total_actions, 1) * 100
        bar = "#" * int(pct / 2)
        logger.info("  %-20s %6d (%5.1f%%) %s", action.name, count, pct, bar)

    # Monthly breakdown
    logger.info("")
    logger.info("MONTHLY BREAKDOWN:")
    logger.info("-" * 65)
    logger.info("%-10s %12s %12s %12s %12s", "Month", "No Battery", "With Battery", "Savings", "$/day")
    logger.info("-" * 65)
    for month_key in sorted(monthly_opt.keys()):
        base = monthly_base.get(month_key, 0) / 100
        opt = monthly_opt.get(month_key, 0) / 100
        sav = base - opt
        # Count days in this month
        month_days = sum(1 for d in overlap_dates if d[:7] == month_key)
        daily = sav / max(month_days, 1)
        logger.info("%-10s %10.2f $ %10.2f $ %10.2f $ %10.2f $", month_key, base, opt, sav, daily)
    logger.info("-" * 65)
    logger.info("%-10s %10.2f $ %10.2f $ %10.2f $ %10.2f $",
                "TOTAL", total_baseline_cost_c / 100, total_opt_cost_c / 100,
                savings_aud, daily_avg_savings)
    logger.info("=" * 70)


if __name__ == "__main__":
    run_perfect_foresight()
