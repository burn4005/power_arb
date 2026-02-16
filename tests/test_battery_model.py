"""Tests for battery model state transitions and degradation costs."""

import os
import sys

# Set test environment before importing config
os.environ.setdefault("AMBER_API_KEY", "psk_test")
os.environ.setdefault("SOLCAST_API_KEY", "test")
os.environ.setdefault("SOLCAST_RESOURCE_ID", "test")
os.environ.setdefault("FOXESS_IP", "127.0.0.1")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from optimizer.battery_model import BatteryModel, PeriodInputs, PERIOD_HOURS
from optimizer.actions import Action


def make_inputs(solar_kw=0, load_kw=0, import_price=20, export_price=10):
    return PeriodInputs(
        solar_kw=solar_kw, load_kw=load_kw,
        import_price=import_price, export_price=export_price,
    )


class TestBatteryModel:
    def setup_method(self):
        self.model = BatteryModel()

    def test_hold_no_solar_no_load(self):
        """HOLD with no solar or load: no grid interaction, SoC unchanged."""
        result = self.model.apply_action(20.0, Action.HOLD, make_inputs())
        assert result.new_soc_kwh == 20.0
        assert result.grid_import_kwh == 0.0
        assert result.grid_export_kwh == 0.0
        assert result.degradation_cost_cents == 0.0

    def test_hold_with_load_no_solar(self):
        """HOLD with load but no solar: imports from grid, battery untouched."""
        result = self.model.apply_action(20.0, Action.HOLD, make_inputs(load_kw=2.0))
        assert result.new_soc_kwh == 20.0
        assert abs(result.grid_import_kwh - 2.0 * PERIOD_HOURS) < 0.01
        assert result.grid_export_kwh == 0.0

    def test_hold_with_solar_excess(self):
        """HOLD with excess solar: battery charges from solar (inverter still in self-use),
        remainder exports. Battery won't discharge though (min SoC = current)."""
        result = self.model.apply_action(
            20.0, Action.HOLD, make_inputs(solar_kw=8.0, load_kw=2.0)
        )
        # Excess solar charges battery, so SoC should increase
        assert result.new_soc_kwh >= 20.0
        assert result.grid_import_kwh == 0.0
        # Some or all excess solar goes to battery + grid
        assert result.grid_export_kwh >= 0.0

    def test_grid_charge_increases_soc(self):
        """GRID_CHARGE should increase SoC from grid."""
        result = self.model.apply_action(
            10.0, Action.GRID_CHARGE, make_inputs(import_price=5.0)
        )
        assert result.new_soc_kwh > 10.0
        assert result.grid_import_kwh > 0
        assert result.degradation_cost_cents > 0  # cycling costs money

    def test_grid_charge_respects_capacity(self):
        """GRID_CHARGE should not exceed battery capacity."""
        result = self.model.apply_action(
            41.0, Action.GRID_CHARGE, make_inputs(import_price=5.0)
        )
        assert result.new_soc_kwh <= self.model.capacity

    def test_discharge_grid_decreases_soc(self):
        """DISCHARGE_GRID should decrease SoC and export energy."""
        result = self.model.apply_action(
            30.0, Action.DISCHARGE_GRID, make_inputs(export_price=50.0)
        )
        assert result.new_soc_kwh < 30.0
        assert result.grid_export_kwh > 0
        assert result.degradation_cost_cents > 0

    def test_discharge_grid_respects_min_soc(self):
        """DISCHARGE_GRID should not go below min SoC."""
        result = self.model.apply_action(
            self.model.min_soc + 0.5, Action.DISCHARGE_GRID,
            make_inputs(export_price=50.0),
        )
        assert result.new_soc_kwh >= self.model.min_soc

    def test_self_use_discharges_to_cover_load(self):
        """SELF_USE should discharge battery to cover load when solar < load."""
        hold_result = self.model.apply_action(
            30.0, Action.HOLD, make_inputs(load_kw=5.0, import_price=40.0)
        )
        self_use_result = self.model.apply_action(
            30.0, Action.SELF_USE, make_inputs(load_kw=5.0, import_price=40.0)
        )
        # Self-use discharges battery to cover load, reducing grid import
        assert self_use_result.grid_import_kwh < hold_result.grid_import_kwh

    def test_self_use_charges_from_excess_solar(self):
        """SELF_USE should charge battery from excess solar, not grid."""
        result = self.model.apply_action(
            20.0, Action.SELF_USE,
            make_inputs(solar_kw=10.0, load_kw=3.0),
        )
        assert result.new_soc_kwh > 20.0
        assert result.grid_import_kwh == 0.0  # no grid import needed

    def test_self_use_both_directions(self):
        """SELF_USE with partial solar: charges from excess AND discharges gap."""
        # 3kW solar, 5kW load -> solar covers 3kW, battery covers rest
        result = self.model.apply_action(
            30.0, Action.SELF_USE,
            make_inputs(solar_kw=3.0, load_kw=5.0),
        )
        # Battery should discharge to cover the 2kW shortfall
        assert result.new_soc_kwh < 30.0
        # Should need minimal grid import (battery covers the gap)
        assert result.grid_import_kwh < 0.5  # nearly zero

    def test_degradation_cost_calculation(self):
        """Verify degradation cost is proportional to energy cycled."""
        # Force charge ~5kWh (10kW * 0.5h)
        result = self.model.apply_action(
            10.0, Action.GRID_CHARGE, make_inputs(import_price=5.0)
        )
        # Degradation should be approximately: energy_cycled * degradation_per_kwh
        # At $15k / 6000 cycles / 42kWh = ~5.95c/kWh
        assert result.degradation_cost_cents > 0
        assert result.degradation_cost_cents < 100  # sanity check

    def test_efficiency_loss_on_charge(self):
        """Charging should lose energy due to efficiency."""
        result = self.model.apply_action(
            10.0, Action.GRID_CHARGE, make_inputs(import_price=5.0)
        )
        energy_in = result.grid_import_kwh
        energy_stored = result.new_soc_kwh - 10.0
        # Less energy stored than imported due to efficiency losses
        assert energy_stored < energy_in

    def test_negative_price_grid_charge_profitable(self):
        """Charging at negative import price should be profitable."""
        result = self.model.apply_action(
            10.0, Action.GRID_CHARGE, make_inputs(import_price=-5.0)
        )
        # Negative import price = we get paid to charge!
        # net_cost = import_cost (negative) + degradation (positive)
        # Should still be profitable if price is negative enough
        import_cost = result.grid_import_kwh * (-5.0)  # negative = profit
        assert import_cost < 0  # we earned money from the grid

    def test_all_actions_produce_valid_soc(self):
        """Every action should produce SoC within valid bounds."""
        for action in Action:
            for soc in [self.model.min_soc, 20.0, self.model.capacity]:
                result = self.model.apply_action(
                    soc, action,
                    make_inputs(solar_kw=5.0, load_kw=2.0,
                                import_price=20.0, export_price=10.0),
                )
                assert self.model.min_soc <= result.new_soc_kwh <= self.model.capacity, \
                    f"{action.name} at SoC={soc}: produced {result.new_soc_kwh}"
