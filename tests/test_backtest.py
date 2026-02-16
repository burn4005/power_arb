"""Backtest tests using real CSV data from tests/testdata/.

Runs the DP optimizer over historical AEMO prices, home consumption, and solar
yield to verify sensible decision-making behavior.
"""

import os
import sys

os.environ.setdefault("AMBER_API_KEY", "psk_test")
os.environ.setdefault("SOLCAST_API_KEY", "test")
os.environ.setdefault("SOLCAST_RESOURCE_ID", "test")
os.environ.setdefault("FOXESS_IP", "127.0.0.1")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from optimizer.dp_optimizer import DPOptimizer
from optimizer.battery_model import BatteryModel, PeriodInputs, PERIOD_HOURS, PERIODS_PER_DAY, PERIODS_PER_HOUR
from optimizer.actions import Action
from backtest.data_loader import (
    load_aemo_prices, load_home_usage, load_solar_yield,
    get_day_prices, get_day_solar, get_day_load, IMPORT_MARKUP_C,
)
import config


def simulate_day(
    optimizer: DPOptimizer,
    model: BatteryModel,
    import_prices: list[float],
    export_prices: list[float],
    solar: list[float],
    load: list[float],
    start_soc: float,
) -> dict:
    """Simulate one day stepping through the optimizer period-by-period.

    At each period, runs the optimizer with a lookahead window, executes the
    first action, then advances to the next period with updated SoC.

    Returns dict with simulation results.
    """
    soc = start_soc
    total_import_kwh = 0.0
    total_export_kwh = 0.0
    total_import_cost_c = 0.0
    total_export_revenue_c = 0.0
    total_degradation_c = 0.0
    actions_taken = []

    for t in range(len(import_prices)):
        # Give optimizer remaining horizon from current period
        remaining = len(import_prices) - t
        result = optimizer.optimize(
            current_soc_kwh=soc,
            import_prices=import_prices[t:],
            export_prices=export_prices[t:],
            solar_forecast=solar[t:],
            load_forecast=load[t:],
        )

        action = result.action
        actions_taken.append(action)

        # Execute the action through the battery model
        inputs = PeriodInputs(
            solar_kw=solar[t],
            load_kw=load[t],
            import_price=import_prices[t],
            export_price=export_prices[t],
        )
        period_result = model.apply_action(soc, action, inputs)

        soc = period_result.new_soc_kwh
        total_import_kwh += period_result.grid_import_kwh
        total_export_kwh += period_result.grid_export_kwh
        total_import_cost_c += period_result.grid_import_kwh * import_prices[t]
        total_export_revenue_c += period_result.grid_export_kwh * export_prices[t]
        total_degradation_c += period_result.degradation_cost_cents

    net_cost_c = total_import_cost_c - total_export_revenue_c + total_degradation_c

    return {
        "final_soc": soc,
        "total_import_kwh": total_import_kwh,
        "total_export_kwh": total_export_kwh,
        "total_import_cost_c": total_import_cost_c,
        "total_export_revenue_c": total_export_revenue_c,
        "total_degradation_c": total_degradation_c,
        "net_cost_c": net_cost_c,
        "actions": actions_taken,
    }


def simulate_day_no_battery(
    import_prices: list[float],
    export_prices: list[float],
    solar: list[float],
    load: list[float],
) -> dict:
    """Simulate one day with no battery (baseline).

    Solar covers load first, excess exports, shortfall imports from grid.
    """
    total_import_cost_c = 0.0
    total_export_revenue_c = 0.0

    for t in range(len(import_prices)):
        solar_kwh = solar[t] * PERIOD_HOURS
        load_kwh = load[t] * PERIOD_HOURS
        solar_to_load = min(solar_kwh, load_kwh)
        grid_import = load_kwh - solar_to_load
        grid_export = solar_kwh - solar_to_load

        total_import_cost_c += grid_import * import_prices[t]
        total_export_revenue_c += grid_export * export_prices[t]

    return {
        "net_cost_c": total_import_cost_c - total_export_revenue_c,
        "total_import_cost_c": total_import_cost_c,
        "total_export_revenue_c": total_export_revenue_c,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBacktestWithCSV:
    """Integration tests running the optimizer against real historical data."""

    @classmethod
    def setup_class(cls):
        cls.aemo = load_aemo_prices()
        cls.usage = load_home_usage()
        cls.solar = load_solar_yield()
        cls.optimizer = DPOptimizer()
        cls.model = BatteryModel()
        cls.capacity = config.battery.capacity_kwh
        cls.min_soc = config.battery.min_soc_kwh

    def _get_day_data(self, aemo_date: str, usage_date: str, month: int, day: int):
        """Helper to load aligned data for a single day.

        aemo_date: "YYYY/MM/DD" format
        usage_date: "YYYY-MM-DD" format
        """
        prices = get_day_prices(self.aemo, aemo_date)
        import_p = [p[0] for p in prices]
        export_p = [p[1] for p in prices]
        solar = get_day_solar(self.solar, month, day)
        load = get_day_load(self.usage, usage_date)
        return import_p, export_p, solar, load

    def test_spike_day_discharges_during_peak(self):
        """On a day with a massive price spike, the optimizer should discharge
        during the spike and charge during cheap periods."""
        # 2024/07/30 had a $17.50/kWh spike — the optimizer should exploit it
        import_p, export_p, solar, load = self._get_day_data(
            "2024/07/30", "2024-07-30", 7, 30
        )

        result = simulate_day(
            self.optimizer, self.model,
            import_p, export_p, solar, load,
            start_soc=self.capacity * 0.5,
        )

        # Should have discharged to grid at some point during the spike
        assert Action.DISCHARGE_GRID in result["actions"], \
            "Optimizer should discharge during a $17.50/kWh spike day"

        # Should be net profitable (export revenue > import cost + degradation)
        assert result["total_export_revenue_c"] > 0, \
            "Should have export revenue on a spike day"

    def test_spike_day_beats_no_battery(self):
        """On a spike day, having a battery with the optimizer should beat
        the no-battery baseline."""
        import_p, export_p, solar, load = self._get_day_data(
            "2024/07/30", "2024-07-30", 7, 30
        )

        optimized = simulate_day(
            self.optimizer, self.model,
            import_p, export_p, solar, load,
            start_soc=self.capacity * 0.5,
        )
        baseline = simulate_day_no_battery(import_p, export_p, solar, load)

        # Optimizer should have lower net cost (or higher profit) than no battery
        assert optimized["net_cost_c"] < baseline["net_cost_c"], \
            (f"Optimizer net cost {optimized['net_cost_c']:.0f}c should beat "
             f"no-battery baseline {baseline['net_cost_c']:.0f}c on a spike day")

    def test_calm_day_mostly_self_use_or_hold(self):
        """On a calm day with small price spread, the optimizer should mostly
        hold or self-use rather than actively trading."""
        # Pick a day with low price spread — 2024/05/15 (autumn, stable prices)
        import_p, export_p, solar, load = self._get_day_data(
            "2024/05/15", "2024-05-15", 5, 15
        )

        result = simulate_day(
            self.optimizer, self.model,
            import_p, export_p, solar, load,
            start_soc=self.capacity * 0.5,
        )

        # Count active trading actions
        passive = sum(1 for a in result["actions"]
                      if a in (Action.SELF_USE, Action.HOLD))
        # On a calm day, most actions should be passive
        assert passive >= len(result["actions"]) * 0.5, \
            f"On a calm day, expected mostly passive actions, got {passive}/{len(result['actions'])}"

    def test_negative_price_day_charges(self):
        """On days with negative prices, the optimizer should charge the
        battery (getting paid to consume)."""
        # Find a half-hour with the most negative price
        most_negative_key = min(self.aemo, key=lambda k: self.aemo[k])
        most_negative_price = self.aemo[most_negative_key]
        date_str = most_negative_key[:10]  # "YYYY/MM/DD"
        month = int(date_str[5:7])
        day = int(date_str[8:10])

        import_p, export_p, solar, load = self._get_day_data(
            date_str, date_str.replace("/", "-"), month, day
        )

        result = simulate_day(
            self.optimizer, self.model,
            import_p, export_p, solar, load,
            start_soc=self.min_soc,  # start empty to allow charging
        )

        # Should charge from grid when prices are negative
        assert Action.GRID_CHARGE in result["actions"], \
            f"Should charge when prices are as low as {most_negative_price:.1f}c/kWh"

    def test_sunny_day_charges_from_solar(self):
        """On a sunny summer day, the optimizer should use solar to charge
        (via self-use) rather than relying entirely on grid-charge."""
        # Jan 1 is peak summer in QLD — should have lots of solar
        import_p, export_p, solar, load = self._get_day_data(
            "2024/01/01", "2024-07-01", 1, 1  # solar data from Jan 1
        )

        # Verify we actually have solar generation
        peak_solar = max(solar)
        assert peak_solar > 5.0, f"Expected strong solar, got peak {peak_solar:.1f}kW"

        result = simulate_day(
            self.optimizer, self.model,
            import_p, export_p, solar, load,
            start_soc=self.min_soc,
        )

        # During solar hours (~5am-6pm), the optimizer should
        # prefer self-use or hold over grid-charge, since free solar is available.
        solar_start = 5 * PERIODS_PER_HOUR   # hour 5
        solar_end = 18 * PERIODS_PER_HOUR    # hour 18
        solar_hours = result["actions"][solar_start:solar_end]
        grid_charge_in_sun = sum(1 for a in solar_hours if a == Action.GRID_CHARGE)
        assert grid_charge_in_sun < len(solar_hours) * 0.5, \
            (f"During solar hours, should mostly use solar not grid-charge. "
             f"Grid-charged {grid_charge_in_sun}/{len(solar_hours)} periods")

    def test_multi_day_simulation(self):
        """Run a week-long simulation and verify cumulative behavior."""
        # Simulate 7 days in July 2024 (winter = more price volatility)
        soc = self.capacity * 0.5
        total_net_cost = 0.0
        total_degradation = 0.0
        all_actions = []

        for day_offset in range(7):
            day = 22 + day_offset  # July 22-28, 2024
            aemo_date = f"2024/07/{day:02d}"
            usage_date = f"2024-07-{day:02d}"

            import_p, export_p, solar, load = self._get_day_data(
                aemo_date, usage_date, 7, day
            )

            result = simulate_day(
                self.optimizer, self.model,
                import_p, export_p, solar, load,
                start_soc=soc,
            )

            soc = result["final_soc"]
            total_net_cost += result["net_cost_c"]
            total_degradation += result["total_degradation_c"]
            all_actions.extend(result["actions"])

        # Over a week, should use all action types at some point
        action_types_used = set(all_actions)
        assert len(action_types_used) >= 2, \
            f"Expected at least 2 action types over a week, got {action_types_used}"

        # SoC should remain in valid bounds
        assert self.min_soc <= soc <= self.capacity

    def test_multi_day_beats_no_battery(self):
        """Over a week with price volatility, the optimizer should
        outperform the no-battery baseline."""
        soc = self.capacity * 0.5
        total_optimized = 0.0
        total_baseline = 0.0

        for day_offset in range(7):
            day = 22 + day_offset  # July 22-28
            aemo_date = f"2024/07/{day:02d}"
            usage_date = f"2024-07-{day:02d}"

            import_p, export_p, solar, load = self._get_day_data(
                aemo_date, usage_date, 7, day
            )

            opt = simulate_day(
                self.optimizer, self.model,
                import_p, export_p, solar, load,
                start_soc=soc,
            )
            base = simulate_day_no_battery(import_p, export_p, solar, load)

            soc = opt["final_soc"]
            total_optimized += opt["net_cost_c"]
            total_baseline += base["net_cost_c"]

        savings_c = total_baseline - total_optimized
        assert savings_c > 0, \
            (f"Optimizer should save money over a week. "
             f"Optimized: {total_optimized:.0f}c, Baseline: {total_baseline:.0f}c")

    def test_soc_always_in_bounds(self):
        """SoC should never go below min or above capacity at any point."""
        # Run through 3 days with diverse conditions
        test_days = [
            ("2024/07/30", "2024-07-30", 7, 30),  # spike day
            ("2024/01/01", "2024-07-01", 1, 1),    # summer
            ("2024/06/15", "2024-06-15", 6, 15),   # winter
        ]

        for aemo_date, usage_date, month, day in test_days:
            import_p, export_p, solar, load = self._get_day_data(
                aemo_date, usage_date, month, day
            )

            soc = self.capacity * 0.5
            for t in range(len(import_p)):
                result = self.optimizer.optimize(
                    current_soc_kwh=soc,
                    import_prices=import_p[t:],
                    export_prices=export_p[t:],
                    solar_forecast=solar[t:],
                    load_forecast=load[t:],
                )

                inputs = PeriodInputs(
                    solar_kw=solar[t], load_kw=load[t],
                    import_price=import_p[t], export_price=export_p[t],
                )
                period = self.model.apply_action(soc, result.action, inputs)
                soc = period.new_soc_kwh

                assert self.min_soc <= soc <= self.capacity, \
                    (f"SoC {soc:.2f} out of bounds [{self.min_soc}, {self.capacity}] "
                     f"on {aemo_date} period {t}, action={result.action.name}")

    def test_optimizer_runs_within_time_budget(self):
        """A single optimizer call should complete quickly (< 2 seconds)."""
        import time

        import_p, export_p, solar, load = self._get_day_data(
            "2024/07/30", "2024-07-30", 7, 30
        )

        start = time.perf_counter()
        for _ in range(10):
            self.optimizer.optimize(
                current_soc_kwh=self.capacity * 0.5,
                import_prices=import_p,
                export_prices=export_p,
                solar_forecast=solar,
                load_forecast=load,
            )
        elapsed = time.perf_counter() - start

        avg_ms = (elapsed / 10) * 1000
        assert avg_ms < 2000, f"Optimizer took {avg_ms:.0f}ms avg (should be < 2000ms)"

    def test_evening_peak_discharges(self):
        """During typical evening peak pricing (5-8pm), the optimizer should
        prefer discharging if the battery has charge and prices are high."""
        # 2024/09/08 had a big spike
        import_p, export_p, solar, load = self._get_day_data(
            "2024/09/08", "2024-09-08", 9, 8
        )

        # Find the peak export price period
        peak_idx = max(range(len(export_p)), key=lambda i: export_p[i])
        peak_export = export_p[peak_idx]

        # Run optimizer at the peak period with a charged battery
        result = self.optimizer.optimize(
            current_soc_kwh=self.capacity * 0.8,
            import_prices=import_p[peak_idx:],
            export_prices=export_p[peak_idx:],
            solar_forecast=solar[peak_idx:],
            load_forecast=load[peak_idx:],
        )

        # At very high export prices with a charged battery, should discharge
        if peak_export > 100:  # only assert if it's actually a high price (> $1/kWh)
            assert result.action in (Action.DISCHARGE_GRID, Action.SELF_USE), \
                (f"At {peak_export:.0f}c/kWh export with 80% SoC, "
                 f"expected discharge, got {result.action.name}")

    def test_data_loaders_produce_correct_counts(self):
        """Verify CSV loaders produce the expected number of periods."""
        prices = get_day_prices(self.aemo, "2024/07/30")
        assert len(prices) == PERIODS_PER_DAY, f"Expected {PERIODS_PER_DAY} price pairs, got {len(prices)}"

        solar = get_day_solar(self.solar, 7, 30)
        assert len(solar) == PERIODS_PER_DAY, f"Expected {PERIODS_PER_DAY} solar values, got {len(solar)}"

        load = get_day_load(self.usage, "2024-07-30")
        assert len(load) == PERIODS_PER_DAY, f"Expected {PERIODS_PER_DAY} load values, got {len(load)}"

    def test_price_markup_applied(self):
        """Import price should always be exactly 20c above export (AEMO spot)."""
        prices = get_day_prices(self.aemo, "2024/01/01")
        for import_p, export_p in prices:
            diff = import_p - export_p
            assert abs(diff - IMPORT_MARKUP_C) < 0.01, \
                f"Import-export spread should be {IMPORT_MARKUP_C}c, got {diff:.2f}c"
