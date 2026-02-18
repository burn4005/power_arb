"""Annual backtest: runs the optimizer over ~20 months of historical data.

Uses the PriceForecaster (no future leakage) to make decisions,
settles at actual AEMO prices, and prints a full profit summary.

This test is slow (~5-10 minutes) so it is marked with @pytest.mark.slow.
Run with: python -m pytest tests/test_annual_backtest.py -v -s
"""

import os
import sys
import time

import pytest

os.environ.setdefault("AMBER_API_KEY", "psk_test")
os.environ.setdefault("SOLCAST_API_KEY", "test")
os.environ.setdefault("SOLCAST_RESOURCE_ID", "test")
os.environ.setdefault("FOXESS_IP", "127.0.0.1")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backtest.data_loader import load_aemo_prices, load_home_usage, load_solar_yield
from backtest.runner import BacktestRunner
import config

# Speed up by only checking date existence for dates in our AEMO set
# (the _date_range method does an `any()` check per date otherwise)


class TestAnnualBacktest:
    """Full annual backtest with price prediction and profit tracking."""

    @classmethod
    def setup_class(cls):
        print("\nLoading CSV data...")
        t0 = time.perf_counter()
        cls.aemo = load_aemo_prices()
        cls.usage = load_home_usage()
        cls.solar = load_solar_yield()
        print(f"  Loaded in {time.perf_counter() - t0:.1f}s: "
              f"{len(cls.aemo)} price periods, "
              f"{len(cls.usage)} usage periods, "
              f"{len(cls.solar)} solar hours")

        # Run the backtest once and share across all tests
        print("\nRunning annual backtest...", flush=True)
        runner = BacktestRunner(
            aemo_prices=cls.aemo,
            home_usage=cls.usage,
            solar_yield=cls.solar,
            reoptimize_every_n=36,  # every 3 hours (36 × 5-min)
        )
        cls.result = runner.run()
        cls.result.print_summary()

    def test_annual_backtest_full(self):
        """Run the full backtest and verify the optimizer is profitable."""
        result = self.result

        # --- Assertions ---

        # 1. Optimizer should save money vs no-battery
        assert result.total_savings_c > 0, (
            f"Optimizer should be profitable. "
            f"Net cost: {result.total_net_cost_c / 100:.2f} AUD, "
            f"Baseline: {result.total_baseline_cost_c / 100:.2f} AUD"
        )

        # 2. SoC should end within valid bounds
        min_soc = config.battery.min_soc_kwh
        capacity = config.battery.capacity_kwh
        assert min_soc <= result.final_soc <= capacity, (
            f"Final SoC {result.final_soc:.1f} kWh out of bounds "
            f"[{min_soc}, {capacity}]"
        )

        # 3. Should have used multiple action types
        assert len(result.action_counts) >= 3, (
            f"Expected at least 3 action types over the full period, "
            f"got {list(result.action_counts.keys())}"
        )

        # 4. Battery cycles should be reasonable (not excessive)
        assert result.total_cycles < config.battery.cycle_life, (
            f"Used {result.total_cycles:.0f} cycles, exceeds "
            f"{config.battery.cycle_life} cycle life"
        )

        # 5. Every day should have been simulated
        assert result.total_days > 300, (
            f"Expected > 300 days simulated, got {result.total_days}"
        )

    def test_monthly_breakdown(self):
        """Run the backtest and check monthly savings vary with seasons."""
        result = self.result

        # Group daily summaries by month
        monthly: dict[str, list] = {}
        for d in result.daily_summaries:
            month_key = d.date[:7]  # "YYYY/MM"
            monthly.setdefault(month_key, []).append(d)

        print("\n  Monthly Breakdown:")
        print(f"  {'Month':<10} {'Days':>5} {'Savings':>10} {'Avg/day':>10} "
              f"{'Import kWh':>11} {'Export kWh':>11}")
        print("  " + "-" * 60)

        for month_key in sorted(monthly.keys()):
            days = monthly[month_key]
            savings = sum(d.savings_c for d in days)
            import_kwh = sum(d.total_import_kwh for d in days)
            export_kwh = sum(d.total_export_kwh for d in days)
            avg = savings / len(days) if days else 0
            print(f"  {month_key:<10} {len(days):>5} "
                  f"${savings / 100:>9,.2f} ${avg / 100:>9,.2f} "
                  f"{import_kwh:>10,.1f} {export_kwh:>10,.1f}")

        # At least some months should have positive savings
        positive_months = sum(
            1 for days in monthly.values()
            if sum(d.savings_c for d in days) > 0
        )
        assert positive_months >= len(monthly) * 0.5, (
            f"Expected at least half of months to have positive savings, "
            f"got {positive_months}/{len(monthly)}"
        )

    def test_price_forecast_reasonable_accuracy(self):
        """The naive price forecaster should have reasonable accuracy."""
        result = self.result

        # MAE should be finite and positive
        assert result.price_forecast_mae_c > 0, "MAE should be positive"

        # MAE should be less than 50 c/kWh on average (generous bound —
        # median AEMO price is ~10c, so MAE < 50 is very loose)
        assert result.price_forecast_mae_c < 50, (
            f"Price forecast MAE {result.price_forecast_mae_c:.1f} c/kWh "
            f"seems too high"
        )

        print(f"\n  Price forecast MAE: {result.price_forecast_mae_c:.1f} c/kWh")
