"""Annual backtest using repeating flow-based pricing from flow_pricing.csv.

Uses a 24-hour step tariff (time, import price, export price) that repeats for
all simulated days, while keeping historical home usage and solar data.

This test is slow (~5-10 minutes) so it is marked with @pytest.mark.slow.
Run with: python -m pytest tests/test_annual_backtest_flow.py -v -s
"""

import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pytest

os.environ.setdefault("AMBER_API_KEY", "psk_test")
os.environ.setdefault("SOLCAST_API_KEY", "test")
os.environ.setdefault("SOLCAST_RESOURCE_ID", "test")
os.environ.setdefault("FOXESS_IP", "127.0.0.1")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import config
from backtest.data_loader import load_home_usage, load_solar_yield, get_day_load, get_day_solar
from backtest.runner import BacktestResult, DaySummary, PeriodRecord
from optimizer.actions import Action
from optimizer.battery_model import (
    BatteryModel,
    HORIZON_PERIODS,
    PERIOD_HOURS,
    PERIOD_MINUTES,
    PERIODS_PER_DAY,
    PERIODS_PER_HOUR,
    PeriodInputs,
)
from optimizer.dp_optimizer import DPOptimizer
from pricing.custom_csv import load_custom_schedule
from tests.backtest_web_export import export_result_to_web_json


TESTDATA = Path(__file__).parent / "testdata"
FLOW_PRICING_CSV = TESTDATA / "flow_pricing.csv"


@dataclass
class _FlowPrices:
    import_c: list[float]
    export_c: list[float]


def _price_at(schedule: list[dict], minutes_since_midnight: int) -> tuple[float, float]:
    """Step interpolation: most recent schedule row at or before time."""
    result = schedule[-1]
    for entry in schedule:
        if entry["time_minutes"] <= minutes_since_midnight:
            result = entry
        else:
            break
    return result["import_price"], result["export_price"]


def _build_day_flow_prices(schedule: list[dict]) -> _FlowPrices:
    import_p: list[float] = []
    export_p: list[float] = []
    for pp in range(PERIODS_PER_DAY):
        minutes = pp * PERIOD_MINUTES
        imp, exp = _price_at(schedule, minutes)
        import_p.append(imp)
        export_p.append(exp)
    return _FlowPrices(import_c=import_p, export_c=export_p)


def _build_price_horizon(schedule: list[dict], start_pp: int) -> tuple[list[float], list[float]]:
    import_fc: list[float] = []
    export_fc: list[float] = []
    for offset in range(HORIZON_PERIODS):
        pp = start_pp + offset
        minutes = (pp % PERIODS_PER_DAY) * PERIOD_MINUTES
        imp, exp = _price_at(schedule, minutes)
        import_fc.append(imp)
        export_fc.append(exp)
    return import_fc, export_fc


def _build_horizon_solar_load(
    usage: dict[str, float],
    solar: dict[tuple[int, int, int], float],
    date_str: str,
    start_pp: int,
) -> tuple[list[float], list[float]]:
    """Build 48h solar/load horizon from current day + up to two days ahead."""
    solar_fc: list[float] = []
    load_fc: list[float] = []

    current_dt = datetime.strptime(date_str, "%Y/%m/%d")
    month = current_dt.month
    day = current_dt.day
    iso = current_dt.strftime("%Y-%m-%d")

    today_solar = get_day_solar(solar, month, day)
    today_load = get_day_load(usage, iso)

    for pp in range(start_pp, PERIODS_PER_DAY):
        solar_fc.append(today_solar[pp])
        load_fc.append(today_load[pp])

    days_ahead = 1
    while len(solar_fc) < HORIZON_PERIODS and days_ahead <= 2:
        next_dt = current_dt + timedelta(days=days_ahead)
        next_solar = get_day_solar(solar, next_dt.month, next_dt.day)
        next_load = get_day_load(usage, next_dt.strftime("%Y-%m-%d"))

        for pp in range(PERIODS_PER_DAY):
            if len(solar_fc) >= HORIZON_PERIODS:
                break
            solar_fc.append(next_solar[pp])
            load_fc.append(next_load[pp])
        days_ahead += 1

    return solar_fc[:HORIZON_PERIODS], load_fc[:HORIZON_PERIODS]


def run_flow_annual_backtest(
    usage: dict[str, float],
    solar: dict[tuple[int, int, int], float],
    flow_schedule: list[dict],
    reoptimize_every_n: int = 36,
) -> BacktestResult:
    optimizer = DPOptimizer()
    model = BatteryModel()

    usage_dates = sorted(set(ts[:10] for ts in usage.keys()))
    start_idx = min(21, len(usage_dates) - 1)
    sim_dates_iso = usage_dates[start_idx:]
    sim_dates = [d.replace("-", "/") for d in sim_dates_iso]

    soc = config.battery.capacity_kwh * 0.5
    daily_summaries: list[DaySummary] = []
    all_action_counts: Counter = Counter()
    period_records: list[PeriodRecord] = []

    current_schedule: list[tuple[str, Action, float]] | None = None
    schedule_idx = 0
    periods_since_reopt = 999

    run_start = time.perf_counter()
    last_progress_time = run_start

    day_prices = _build_day_flow_prices(flow_schedule)

    for day_num, date_str in enumerate(sim_dates):
        now = time.perf_counter()
        if now - last_progress_time >= 30 or day_num == 0:
            elapsed = now - run_start
            total_days = len(sim_dates)
            pct = (day_num / total_days * 100) if total_days > 0 else 0
            if day_num > 0:
                rate = elapsed / day_num
                eta_min = (rate * (total_days - day_num)) / 60
                print(
                    f"  [{pct:5.1f}%] Day {day_num}/{total_days} "
                    f"({date_str}) | Elapsed: {elapsed / 60:.1f}m | ETA: {eta_min:.1f}m",
                    flush=True,
                )
            else:
                print(f"  [  0.0%] Day 0/{total_days} ({date_str}) | Starting...", flush=True)
            last_progress_time = now

        iso_date = date_str.replace("/", "-")
        current_dt = datetime.strptime(date_str, "%Y/%m/%d")
        day_solar = get_day_solar(solar, current_dt.month, current_dt.day)
        day_load = get_day_load(usage, iso_date)

        day_summary = DaySummary(date=date_str, start_soc=soc)
        day_actions: Counter = Counter()

        for pp in range(PERIODS_PER_DAY):
            actual_import = day_prices.import_c[pp]
            actual_export = day_prices.export_c[pp]

            if periods_since_reopt >= reoptimize_every_n:
                fc_import, fc_export = _build_price_horizon(flow_schedule, pp)
                solar_fc, load_fc = _build_horizon_solar_load(usage, solar, date_str, pp)

                n = min(len(fc_import), len(fc_export), len(solar_fc), len(load_fc))
                if n > 0:
                    result = optimizer.optimize(
                        current_soc_kwh=soc,
                        import_prices=fc_import[:n],
                        export_prices=fc_export[:n],
                        solar_forecast=solar_fc[:n],
                        load_forecast=load_fc[:n],
                    )
                    current_schedule = result.schedule
                    schedule_idx = 0
                else:
                    current_schedule = None
                periods_since_reopt = 0

            if current_schedule and schedule_idx < len(current_schedule):
                action = current_schedule[schedule_idx][1]
            else:
                action = Action.SELF_USE

            inputs = PeriodInputs(
                solar_kw=day_solar[pp],
                load_kw=day_load[pp],
                import_price=actual_import,
                export_price=actual_export,
            )
            soc_before = soc
            period_result = model.apply_action(soc, action, inputs)
            soc = period_result.new_soc_kwh

            day_summary.total_import_kwh += period_result.grid_import_kwh
            day_summary.total_export_kwh += period_result.grid_export_kwh
            day_summary.total_import_cost_c += period_result.grid_import_kwh * actual_import
            day_summary.total_export_revenue_c += period_result.grid_export_kwh * actual_export
            day_summary.total_degradation_c += period_result.degradation_cost_cents
            day_summary.energy_cycled_kwh += period_result.energy_cycled_kwh
            day_actions[action.name] += 1

            solar_kwh = day_solar[pp] * PERIOD_HOURS
            load_kwh = day_load[pp] * PERIOD_HOURS
            solar_to_load = min(solar_kwh, load_kwh)
            base_import = (load_kwh - solar_to_load) * actual_import
            base_export = (solar_kwh - solar_to_load) * actual_export
            baseline_cost_c = base_import - base_export
            day_summary.baseline_net_cost_c += baseline_cost_c

            h = pp // PERIODS_PER_HOUR
            m_val = (pp % PERIODS_PER_HOUR) * PERIOD_MINUTES
            period_records.append(
                PeriodRecord(
                    timestamp=f"{date_str} {h:02d}:{m_val:02d}",
                    soc_kwh=soc_before,
                    action=action.name,
                    solar_kw=day_solar[pp],
                    load_kw=day_load[pp],
                    import_price_c=actual_import,
                    export_price_c=actual_export,
                    grid_import_kwh=period_result.grid_import_kwh,
                    grid_export_kwh=period_result.grid_export_kwh,
                    net_cost_c=period_result.net_cost_cents,
                    baseline_cost_c=baseline_cost_c,
                )
            )

            schedule_idx += 1
            periods_since_reopt += 1

        day_summary.end_soc = soc
        day_summary.net_cost_c = (
            day_summary.total_import_cost_c
            - day_summary.total_export_revenue_c
            + day_summary.total_degradation_c
        )
        day_summary.savings_c = day_summary.baseline_net_cost_c - day_summary.net_cost_c
        day_summary.actions_count = dict(day_actions)

        daily_summaries.append(day_summary)
        all_action_counts.update(day_actions)

    elapsed = time.perf_counter() - run_start
    print(f"\r  [100.0%] Done! {len(sim_dates)} days in {elapsed / 60:.1f} minutes", flush=True)

    total_import_cost = sum(d.total_import_cost_c for d in daily_summaries)
    total_export_rev = sum(d.total_export_revenue_c for d in daily_summaries)
    total_degradation = sum(d.total_degradation_c for d in daily_summaries)
    total_net = sum(d.net_cost_c for d in daily_summaries)
    total_baseline = sum(d.baseline_net_cost_c for d in daily_summaries)
    total_energy_cycled = sum(d.energy_cycled_kwh for d in daily_summaries)

    capacity = config.battery.capacity_kwh
    total_cycles = total_energy_cycled / (2 * capacity) if capacity > 0 else 0.0
    total_days = len(sim_dates)
    avg_daily = total_cycles / total_days if total_days > 0 else 0.0

    return BacktestResult(
        start_date=sim_dates[0],
        end_date=sim_dates[-1],
        total_days=total_days,
        total_import_cost_c=total_import_cost,
        total_export_revenue_c=total_export_rev,
        total_degradation_c=total_degradation,
        total_net_cost_c=total_net,
        total_baseline_cost_c=total_baseline,
        total_savings_c=total_baseline - total_net,
        total_energy_cycled_kwh=total_energy_cycled,
        total_cycles=total_cycles,
        avg_daily_cycles=avg_daily,
        final_soc=soc,
        action_counts=dict(all_action_counts),
        price_forecast_mae_c=0.0,
        price_forecast_mae_import_c=0.0,
        daily_summaries=daily_summaries,
        period_records=period_records,
    )


@pytest.mark.slow
class TestAnnualBacktestFlow:
    """Full annual backtest with repeating flow pricing and profit tracking."""

    @classmethod
    def setup_class(cls):
        print("\nLoading CSV data...")
        t0 = time.perf_counter()
        cls.usage = load_home_usage()
        cls.solar = load_solar_yield()
        cls.flow_schedule = load_custom_schedule(str(FLOW_PRICING_CSV))
        print(
            f"  Loaded in {time.perf_counter() - t0:.1f}s: "
            f"{len(cls.flow_schedule)} flow price entries, "
            f"{len(cls.usage)} usage periods, "
            f"{len(cls.solar)} solar hours"
        )

        print("\nRunning annual backtest with flow pricing...", flush=True)
        cls.result = run_flow_annual_backtest(
            usage=cls.usage,
            solar=cls.solar,
            flow_schedule=cls.flow_schedule,
            reoptimize_every_n=36,
        )
        cls.result.print_summary()
        out = export_result_to_web_json(
            cls.result, str(Path(__file__).resolve().parents[1] / "web" / "backtest_data.json")
        )
        print(f"  Exported web data: {out}")

    def test_annual_backtest_full(self):
        """Run the full flow-priced backtest and verify profitability."""
        result = self.result

        assert result.total_savings_c > 0, (
            "Optimizer should be profitable. "
            f"Net cost: {result.total_net_cost_c / 100:.2f} AUD, "
            f"Baseline: {result.total_baseline_cost_c / 100:.2f} AUD"
        )

        min_soc = config.battery.min_soc_kwh
        capacity = config.battery.capacity_kwh
        assert min_soc <= result.final_soc <= capacity, (
            f"Final SoC {result.final_soc:.1f} kWh out of bounds [{min_soc}, {capacity}]"
        )

        assert len(result.action_counts) >= 2, (
            "Expected at least 2 action types over the full period, "
            f"got {list(result.action_counts.keys())}"
        )

        assert result.total_cycles < config.battery.cycle_life, (
            f"Used {result.total_cycles:.0f} cycles, exceeds {config.battery.cycle_life} cycle life"
        )

        assert result.total_days > 300, (
            f"Expected > 300 days simulated, got {result.total_days}"
        )

    def test_monthly_breakdown(self):
        """Check monthly savings under repeating flow pricing."""
        result = self.result

        monthly: dict[str, list] = {}
        for d in result.daily_summaries:
            month_key = d.date[:7]
            monthly.setdefault(month_key, []).append(d)

        print("\n  Monthly Breakdown:")
        print(
            f"  {'Month':<10} {'Days':>5} {'Savings':>10} {'Avg/day':>10} "
            f"{'Import kWh':>11} {'Export kWh':>11}"
        )
        print("  " + "-" * 60)

        for month_key in sorted(monthly.keys()):
            days = monthly[month_key]
            savings = sum(d.savings_c for d in days)
            import_kwh = sum(d.total_import_kwh for d in days)
            export_kwh = sum(d.total_export_kwh for d in days)
            avg = savings / len(days) if days else 0
            print(
                f"  {month_key:<10} {len(days):>5} "
                f"${savings / 100:>9,.2f} ${avg / 100:>9,.2f} "
                f"{import_kwh:>10,.1f} {export_kwh:>10,.1f}"
            )

        positive_months = sum(
            1 for days in monthly.values() if sum(d.savings_c for d in days) > 0
        )
        assert positive_months >= len(monthly) * 0.5, (
            "Expected at least half of months to have positive savings, "
            f"got {positive_months}/{len(monthly)}"
        )

    def test_price_forecast_reasonable_accuracy(self):
        """Flow schedule is deterministic in this test, so forecast MAE is zero."""
        result = self.result
        assert result.price_forecast_mae_c == 0.0
        assert result.price_forecast_mae_import_c == 0.0
        print("\n  Price forecast MAE: export 0.0 c/kWh, import 0.0 c/kWh")
