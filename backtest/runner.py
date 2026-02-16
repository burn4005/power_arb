"""Backtest runner for annual simulation of the battery arbitrage optimizer.

Steps through historical data period-by-period (5-minute intervals):
  1. Generates price forecast from past data (no future leakage)
  2. Runs DP optimizer with forecast prices + actual solar/load
  3. Settles each period at actual prices
  4. Tracks SoC, energy flows, costs, and compares to no-battery baseline
"""

import logging
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable

from backtest.data_loader import (
    get_day_solar, get_day_load, get_spot_price, IMPORT_MARKUP_C,
)
from forecasting.price import PriceForecaster
from optimizer.dp_optimizer import DPOptimizer
from optimizer.battery_model import (
    BatteryModel, PeriodInputs, PERIOD_HOURS, PERIOD_MINUTES,
    PERIODS_PER_HOUR, PERIODS_PER_DAY, HORIZON_PERIODS,
)
from optimizer.actions import Action
import config

logger = logging.getLogger(__name__)


@dataclass
class DaySummary:
    date: str
    total_import_kwh: float = 0.0
    total_export_kwh: float = 0.0
    total_import_cost_c: float = 0.0
    total_export_revenue_c: float = 0.0
    total_degradation_c: float = 0.0
    net_cost_c: float = 0.0
    baseline_net_cost_c: float = 0.0
    savings_c: float = 0.0
    start_soc: float = 0.0
    end_soc: float = 0.0
    energy_cycled_kwh: float = 0.0
    actions_count: dict[str, int] = field(default_factory=dict)


@dataclass
class PeriodRecord:
    """Period-level data for visualization."""
    timestamp: str
    soc_kwh: float
    action: str
    solar_kw: float
    load_kw: float
    import_price_c: float
    export_price_c: float
    grid_import_kwh: float
    grid_export_kwh: float
    net_cost_c: float
    baseline_cost_c: float


@dataclass
class BacktestResult:
    start_date: str
    end_date: str
    total_days: int

    # Financials (cents)
    total_import_cost_c: float
    total_export_revenue_c: float
    total_degradation_c: float
    total_net_cost_c: float
    total_baseline_cost_c: float
    total_savings_c: float

    # Battery stats
    total_energy_cycled_kwh: float
    total_cycles: float
    avg_daily_cycles: float
    final_soc: float

    # Action distribution
    action_counts: dict[str, int]

    # Forecast accuracy
    price_forecast_mae_c: float  # mean absolute error in c/kWh

    # Per-day data
    daily_summaries: list[DaySummary]

    # Per-period data (for visualization)
    period_records: list[PeriodRecord] = field(default_factory=list)

    def print_summary(self):
        """Print a human-readable annual summary."""
        total_days = self.total_days or 1
        net_cost_aud = self.total_net_cost_c / 100
        baseline_aud = self.total_baseline_cost_c / 100
        savings_aud = self.total_savings_c / 100
        import_aud = self.total_import_cost_c / 100
        export_aud = self.total_export_revenue_c / 100
        degradation_aud = self.total_degradation_c / 100

        print()
        print("=" * 60)
        print("  ANNUAL BACKTEST SUMMARY")
        print("=" * 60)
        print(f"  Period: {self.start_date} to {self.end_date} ({total_days} days)")
        print()
        print("  WITH BATTERY (Optimized):")
        print(f"    Grid import cost:    ${import_aud:>10,.2f}")
        print(f"    Grid export revenue: ${export_aud:>10,.2f}")
        print(f"    Degradation cost:    ${degradation_aud:>10,.2f}")
        print(f"    Net electricity cost: ${net_cost_aud:>9,.2f}")
        print()
        print("  WITHOUT BATTERY (Baseline):")
        print(f"    Net electricity cost: ${baseline_aud:>9,.2f}")
        print()
        print(f"  SAVINGS: ${savings_aud:>,.2f} "
              f"({savings_aud / max(baseline_aud, 0.01) * 100:.1f}% reduction)")
        print()
        print(f"  Battery cycles used: {self.total_cycles:,.1f} of "
              f"{config.battery.cycle_life:,}")
        print(f"  Avg daily cycles: {self.avg_daily_cycles:.2f}")
        battery_cost = config.battery.cost_aud
        annual_savings = savings_aud * (365 / total_days) if total_days > 0 else 0
        roi_years = battery_cost / annual_savings if annual_savings > 0 else float("inf")
        print(f"  Projected annual savings: ${annual_savings:,.2f}")
        print(f"  Battery ROI: {roi_years:.1f} years "
              f"(${battery_cost:,.0f} battery cost)")
        print()
        print(f"  Price forecast MAE: {self.price_forecast_mae_c:.1f} c/kWh")
        print()
        print("  Action distribution:")
        total_actions = sum(self.action_counts.values()) or 1
        for action_name, count in sorted(
            self.action_counts.items(), key=lambda x: -x[1]
        ):
            print(f"    {action_name:<16s} {count:>6d}  "
                  f"({count / total_actions * 100:5.1f}%)")
        print("=" * 60)
        print()


class BacktestRunner:
    """Runs a full backtest of the battery optimizer over historical data.

    At each re-optimization point, uses PriceForecaster for 48h price
    predictions. Between re-optimizations, follows the planned schedule.
    Settlement always uses actual historical prices at 5-minute resolution.
    """

    def __init__(
        self,
        aemo_prices: dict[str, float],
        home_usage: dict[str, float],
        solar_yield: dict[tuple[int, int, int], float],
        import_markup_c: float = IMPORT_MARKUP_C,
        reoptimize_every_n: int = 6,
        initial_soc_kwh: float | None = None,
        warm_up_days: int = 21,
    ):
        self.aemo = aemo_prices
        self.usage = home_usage
        self.solar = solar_yield
        self.import_markup_c = import_markup_c
        self.reoptimize_every = reoptimize_every_n
        self.initial_soc = initial_soc_kwh or (config.battery.capacity_kwh * 0.5)
        self.warm_up_days = warm_up_days

        self.optimizer = DPOptimizer()
        self.model = BatteryModel()
        self.forecaster = PriceForecaster(
            aemo_prices, import_markup_c=import_markup_c, window_days=warm_up_days
        )

    def run(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> BacktestResult:
        """Run the full backtest.

        Args:
            start_date: "YYYY/MM/DD" — defaults to warm_up_days after first data.
            end_date: "YYYY/MM/DD" — defaults to last full day of data.
            progress_callback: Called with (date_str, day_num, total_days).

        Returns:
            BacktestResult with full simulation results.
        """
        first_date, last_date = self._determine_date_range()
        start_date = start_date or first_date
        end_date = end_date or last_date

        # Build list of dates to simulate
        dates = self._date_range(start_date, end_date)
        total_days = len(dates)

        soc = self.initial_soc
        daily_summaries: list[DaySummary] = []
        all_action_counts: Counter = Counter()
        period_records: list[PeriodRecord] = []

        # Forecast accuracy tracking
        forecast_errors: list[float] = []

        # Schedule state across days
        current_schedule: list[tuple[str, Action, float]] | None = None
        schedule_idx = 0
        periods_since_reopt = 999  # force first optimization

        # Progress tracking
        run_start = time.perf_counter()
        last_progress_time = run_start

        for day_num, date_str in enumerate(dates):
            if progress_callback:
                progress_callback(date_str, day_num, total_days)

            # Print progress every 30 seconds
            now = time.perf_counter()
            if now - last_progress_time >= 30 or day_num == 0:
                elapsed = now - run_start
                pct = (day_num / total_days * 100) if total_days > 0 else 0
                if day_num > 0:
                    rate = elapsed / day_num  # seconds per day
                    remaining = rate * (total_days - day_num)
                    eta_min = remaining / 60
                    print(
                        f"\r  [{pct:5.1f}%] Day {day_num}/{total_days} "
                        f"({date_str}) | "
                        f"Elapsed: {elapsed / 60:.1f}m | "
                        f"ETA: {eta_min:.1f}m remaining",
                        end="", flush=True,
                    )
                else:
                    print(
                        f"\r  [{pct:5.1f}%] Day {day_num}/{total_days} "
                        f"({date_str}) | Starting...",
                        end="", flush=True,
                    )
                last_progress_time = now

            day_summary = DaySummary(date=date_str, start_soc=soc)
            iso_date = date_str.replace("/", "-")
            month = int(date_str[5:7])
            day = int(date_str[8:10])

            solar = get_day_solar(self.solar, month, day)
            load = get_day_load(self.usage, iso_date)
            day_actions: Counter = Counter()

            for pp in range(PERIODS_PER_DAY):
                # Get actual spot price for settlement
                actual_spot = get_spot_price(self.aemo, date_str, pp)
                actual_import = actual_spot + self.import_markup_c
                actual_export = actual_spot

                # Re-optimize if needed
                if periods_since_reopt >= self.reoptimize_every:
                    h = pp // PERIODS_PER_HOUR
                    m = (pp % PERIODS_PER_HOUR) * PERIOD_MINUTES
                    current_time = f"{date_str} {h:02d}:{m:02d}"

                    # Get forecast prices for 48h horizon
                    fc = self.forecaster.forecast(current_time, hours=48)

                    # Build solar + load forecast for horizon
                    solar_fc, load_fc = self._build_horizon_forecast(
                        date_str, pp, month, day, iso_date
                    )

                    # Truncate all to same length
                    n = min(len(fc), len(solar_fc), len(load_fc))
                    if n > 0:
                        fc_import = [fc[i]["import_c"] for i in range(n)]
                        fc_export = [fc[i]["export_c"] for i in range(n)]

                        # Inject actual current-period prices so the
                        # optimizer sees the real spot price, not the
                        # median forecast (critical for spike capture).
                        fc_import[0] = actual_import
                        fc_export[0] = actual_export

                        result = self.optimizer.optimize(
                            current_soc_kwh=soc,
                            import_prices=fc_import,
                            export_prices=fc_export,
                            solar_forecast=solar_fc[:n],
                            load_forecast=load_fc[:n],
                        )
                        current_schedule = result.schedule
                        schedule_idx = 0

                        # Track forecast accuracy for the current period
                        if len(fc) > 0:
                            forecast_errors.append(abs(fc[0]["export_c"] - actual_spot))
                    else:
                        current_schedule = None

                    periods_since_reopt = 0

                # Determine action
                if current_schedule and schedule_idx < len(current_schedule):
                    action = current_schedule[schedule_idx][1]
                else:
                    action = Action.SELF_USE

                # Execute action and settle at actual prices
                soc_before = soc
                inputs = PeriodInputs(
                    solar_kw=solar[pp],
                    load_kw=load[pp],
                    import_price=actual_import,
                    export_price=actual_export,
                )
                period_result = self.model.apply_action(soc, action, inputs)

                soc = period_result.new_soc_kwh
                day_summary.total_import_kwh += period_result.grid_import_kwh
                day_summary.total_export_kwh += period_result.grid_export_kwh
                day_summary.total_import_cost_c += (
                    period_result.grid_import_kwh * actual_import
                )
                day_summary.total_export_revenue_c += (
                    period_result.grid_export_kwh * actual_export
                )
                day_summary.total_degradation_c += period_result.degradation_cost_cents
                day_summary.energy_cycled_kwh += (
                    period_result.grid_import_kwh + period_result.grid_export_kwh
                ) * 0  # placeholder, use actual energy through battery
                day_actions[action.name] += 1

                # No-battery baseline for this period
                solar_kwh = solar[pp] * PERIOD_HOURS
                load_kwh = load[pp] * PERIOD_HOURS
                solar_to_load = min(solar_kwh, load_kwh)
                base_import = (load_kwh - solar_to_load) * actual_import
                base_export = (solar_kwh - solar_to_load) * actual_export
                baseline_cost_c = base_import - base_export
                day_summary.baseline_net_cost_c += baseline_cost_c

                # Record period data for visualization
                h = pp // PERIODS_PER_HOUR
                m_val = (pp % PERIODS_PER_HOUR) * PERIOD_MINUTES
                period_records.append(PeriodRecord(
                    timestamp=f"{date_str} {h:02d}:{m_val:02d}",
                    soc_kwh=soc_before,
                    action=action.name,
                    solar_kw=solar[pp],
                    load_kw=load[pp],
                    import_price_c=actual_import,
                    export_price_c=actual_export,
                    grid_import_kwh=period_result.grid_import_kwh,
                    grid_export_kwh=period_result.grid_export_kwh,
                    net_cost_c=period_result.net_cost_cents,
                    baseline_cost_c=baseline_cost_c,
                ))

                schedule_idx += 1
                periods_since_reopt += 1

            # Finalize day summary
            day_summary.end_soc = soc
            day_summary.net_cost_c = (
                day_summary.total_import_cost_c
                - day_summary.total_export_revenue_c
                + day_summary.total_degradation_c
            )
            day_summary.savings_c = day_summary.baseline_net_cost_c - day_summary.net_cost_c
            day_summary.actions_count = dict(day_actions)
            # Calculate energy cycled from degradation cost
            deg_per_kwh_c = config.battery.degradation_per_kwh * 100
            if deg_per_kwh_c > 0:
                day_summary.energy_cycled_kwh = (
                    day_summary.total_degradation_c / deg_per_kwh_c
                )

            daily_summaries.append(day_summary)
            all_action_counts.update(day_actions)

        # Final progress
        elapsed = time.perf_counter() - run_start
        print(
            f"\r  [100.0%] Done! {total_days} days in {elapsed / 60:.1f} minutes"
            + " " * 30,
            flush=True,
        )

        # Aggregate results
        total_import_cost = sum(d.total_import_cost_c for d in daily_summaries)
        total_export_rev = sum(d.total_export_revenue_c for d in daily_summaries)
        total_degradation = sum(d.total_degradation_c for d in daily_summaries)
        total_net = sum(d.net_cost_c for d in daily_summaries)
        total_baseline = sum(d.baseline_net_cost_c for d in daily_summaries)
        total_energy_cycled = sum(d.energy_cycled_kwh for d in daily_summaries)

        capacity = config.battery.capacity_kwh
        # A full cycle = charge capacity + discharge capacity = 2 * capacity through battery
        total_cycles = total_energy_cycled / (2 * capacity) if capacity > 0 else 0
        avg_daily = total_cycles / total_days if total_days > 0 else 0

        mae = sum(forecast_errors) / len(forecast_errors) if forecast_errors else 0.0

        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
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
            price_forecast_mae_c=mae,
            daily_summaries=daily_summaries,
            period_records=period_records,
        )

    def _determine_date_range(self) -> tuple[str, str]:
        """Find the valid date range from AEMO data, accounting for warm-up."""
        all_dates = sorted(set(k[:10] for k in self.aemo.keys()))
        if not all_dates:
            raise ValueError("No AEMO price data loaded")

        # Skip warm-up days
        start_idx = min(self.warm_up_days, len(all_dates) - 1)
        first_date = all_dates[start_idx]

        # Last date: need a full day of data.
        # Use second-to-last date to be safe (last day might be incomplete)
        last_date = all_dates[-2] if len(all_dates) > 1 else all_dates[-1]

        return first_date, last_date

    def _date_range(self, start: str, end: str) -> list[str]:
        """Generate list of date strings from start to end inclusive."""
        # Pre-compute set of dates that have AEMO data
        available_dates = set(k[:10] for k in self.aemo)
        start_dt = datetime.strptime(start, "%Y/%m/%d")
        end_dt = datetime.strptime(end, "%Y/%m/%d")
        dates = []
        dt = start_dt
        while dt <= end_dt:
            date_str = dt.strftime("%Y/%m/%d")
            if date_str in available_dates:
                dates.append(date_str)
            dt += timedelta(days=1)
        return dates

    def _build_horizon_forecast(
        self, date_str: str, start_pp: int, month: int, day: int, iso_date: str
    ) -> tuple[list[float], list[float]]:
        """Build 48h solar and load forecast from the current position.

        Spans across day boundaries as needed.
        """
        solar_fc = []
        load_fc = []
        periods_needed = HORIZON_PERIODS  # 576

        current_dt = datetime.strptime(date_str, "%Y/%m/%d")

        # Periods from today
        today_solar = get_day_solar(self.solar, month, day)
        today_load = get_day_load(self.usage, iso_date)

        for pp in range(start_pp, PERIODS_PER_DAY):
            solar_fc.append(today_solar[pp])
            load_fc.append(today_load[pp])

        # Periods from subsequent days
        days_ahead = 1
        while len(solar_fc) < periods_needed and days_ahead <= 2:
            next_dt = current_dt + timedelta(days=days_ahead)
            next_month = next_dt.month
            next_day = next_dt.day
            next_iso = next_dt.strftime("%Y-%m-%d")

            next_solar = get_day_solar(self.solar, next_month, next_day)
            next_load = get_day_load(self.usage, next_iso)

            for pp in range(PERIODS_PER_DAY):
                if len(solar_fc) >= periods_needed:
                    break
                solar_fc.append(next_solar[pp])
                load_fc.append(next_load[pp])

            days_ahead += 1

        return solar_fc[:periods_needed], load_fc[:periods_needed]
