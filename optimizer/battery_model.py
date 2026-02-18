from dataclasses import dataclass

import numpy as np

import config
from optimizer.actions import Action


@dataclass
class PeriodInputs:
    """External inputs for a single period."""
    solar_kw: float         # forecast solar generation (kW average)
    load_kw: float          # forecast home consumption (kW average)
    import_price: float     # cents/kWh to buy from grid
    export_price: float     # cents/kWh received for export to grid


@dataclass
class PeriodResult:
    """Result of applying an action for a single period."""
    new_soc_kwh: float
    grid_import_kwh: float
    grid_export_kwh: float
    degradation_cost_cents: float
    net_cost_cents: float   # positive = cost, negative = profit


PERIOD_MINUTES = 5
PERIOD_HOURS = PERIOD_MINUTES / 60          # 1/12 (~0.0833)
PERIODS_PER_HOUR = 60 // PERIOD_MINUTES     # 12
PERIODS_PER_DAY = 24 * PERIODS_PER_HOUR     # 288
HORIZON_HOURS = 48
HORIZON_PERIODS = HORIZON_HOURS * PERIODS_PER_HOUR  # 576


class BatteryModel:
    """Models battery state transitions, efficiency losses, and degradation costs.

    All energy in kWh, power in kW, prices in cents/kWh.
    """

    def __init__(self):
        self.capacity = config.battery.capacity_kwh
        self.max_power = config.battery.max_power_kw
        self.min_soc = config.battery.min_soc_kwh
        self.eta = config.battery.one_way_efficiency  # ~0.949 for 90% round-trip
        self.degradation_per_kwh = config.battery.degradation_per_kwh * 100  # convert $ to cents

    def apply_action(
        self, soc_kwh: float, action: Action, inputs: PeriodInputs
    ) -> PeriodResult:
        """Simulate one period and return the result."""
        solar_kwh = inputs.solar_kw * PERIOD_HOURS
        load_kwh = inputs.load_kw * PERIOD_HOURS
        max_charge_kwh = self.max_power * PERIOD_HOURS   # 5 kWh per period at 10kW
        max_discharge_kwh = self.max_power * PERIOD_HOURS

        grid_import = 0.0
        grid_export = 0.0
        new_soc = soc_kwh
        energy_cycled = 0.0  # absolute kWh through battery (for degradation)

        if action == Action.GRID_CHARGE:
            # Force Charge: charge battery from grid + solar at max rate.
            # Solar covers load first, excess charges battery, grid fills rest,
            # any remaining excess solar exports to grid.
            solar_to_load = min(solar_kwh, load_kwh)
            solar_excess = solar_kwh - solar_to_load
            solar_to_battery = min(solar_excess,
                                   (self.capacity - soc_kwh) / self.eta)
            solar_to_grid = solar_excess - solar_to_battery
            grid_to_load = load_kwh - solar_to_load
            room_in_battery = (self.capacity - soc_kwh) / self.eta - solar_to_battery
            grid_to_battery = min(max_charge_kwh, max(0, room_in_battery))

            new_soc = soc_kwh + (solar_to_battery + grid_to_battery) * self.eta
            new_soc = min(new_soc, self.capacity)
            grid_import = grid_to_load + grid_to_battery
            grid_export = solar_to_grid
            energy_cycled = solar_to_battery + grid_to_battery

        elif action == Action.GRID_CHARGE_NO_EXPORT:
            # Force Charge + export limit 0W: same as GRID_CHARGE but excess
            # solar that doesn't fit in the battery is curtailed, not exported.
            # Use when: import is cheap AND export price is negative.
            solar_to_load = min(solar_kwh, load_kwh)
            solar_to_battery = min(solar_kwh - solar_to_load,
                                   (self.capacity - soc_kwh) / self.eta)
            grid_to_load = load_kwh - solar_to_load
            room_in_battery = (self.capacity - soc_kwh) / self.eta - solar_to_battery
            grid_to_battery = min(max_charge_kwh, max(0, room_in_battery))

            new_soc = soc_kwh + (solar_to_battery + grid_to_battery) * self.eta
            new_soc = min(new_soc, self.capacity)
            grid_import = grid_to_load + grid_to_battery
            # grid_export stays 0.0 — excess solar curtailed
            energy_cycled = solar_to_battery + grid_to_battery

        elif action == Action.SELF_USE:
            # FoxESS Self-Use mode: the inverter natively handles:
            #   1. Solar -> home load (priority)
            #   2. Excess solar -> charge battery
            #   3. If solar < load -> discharge battery to cover shortfall
            #   4. Remaining excess solar -> export to grid
            #   5. Remaining unmet load -> import from grid
            solar_to_load = min(solar_kwh, load_kwh)
            remaining_load = load_kwh - solar_to_load
            solar_excess = solar_kwh - solar_to_load

            # Excess solar charges battery
            solar_to_battery = min(solar_excess,
                                   (self.capacity - soc_kwh) / self.eta,
                                   max_charge_kwh)
            solar_to_grid = solar_excess - solar_to_battery
            soc_after_charge = soc_kwh + solar_to_battery * self.eta

            # Battery discharges to cover remaining load
            available_discharge = max(0, soc_after_charge - self.min_soc) * self.eta
            battery_to_load = min(remaining_load, available_discharge, max_discharge_kwh)
            soc_discharged = battery_to_load / self.eta

            new_soc = soc_after_charge - soc_discharged
            grid_import = max(0, remaining_load - battery_to_load)
            grid_export = solar_to_grid
            energy_cycled = solar_to_battery + soc_discharged

        elif action == Action.SELF_USE_NO_EXPORT:
            # Self-Use mode with export limit = 0W.
            # Same as SELF_USE but excess solar is curtailed, not exported.
            solar_to_load = min(solar_kwh, load_kwh)
            remaining_load = load_kwh - solar_to_load
            solar_excess = solar_kwh - solar_to_load

            # Excess solar charges battery
            solar_to_battery = min(solar_excess,
                                   (self.capacity - soc_kwh) / self.eta,
                                   max_charge_kwh)
            # Remaining excess solar is curtailed (not exported)
            soc_after_charge = soc_kwh + solar_to_battery * self.eta

            # Battery discharges to cover remaining load
            available_discharge = max(0, soc_after_charge - self.min_soc) * self.eta
            battery_to_load = min(remaining_load, available_discharge, max_discharge_kwh)
            soc_discharged = battery_to_load / self.eta

            new_soc = soc_after_charge - soc_discharged
            grid_import = max(0, remaining_load - battery_to_load)
            grid_export = 0.0  # export limit = 0W
            energy_cycled = solar_to_battery + soc_discharged

        elif action == Action.HOLD:
            # Self-Use mode with min SoC = current level.
            # Battery won't discharge. Excess solar still charges battery.
            # Home powered from solar + grid.
            solar_to_load = min(solar_kwh, load_kwh)
            solar_excess = solar_kwh - solar_to_load

            # Excess solar still charges battery (inverter does this in self-use)
            solar_to_battery = min(solar_excess,
                                   (self.capacity - soc_kwh) / self.eta,
                                   max_charge_kwh)
            solar_to_grid = solar_excess - solar_to_battery

            new_soc = soc_kwh + solar_to_battery * self.eta
            new_soc = min(new_soc, self.capacity)
            grid_import = max(0, load_kwh - solar_to_load)
            grid_export = solar_to_grid
            energy_cycled = solar_to_battery  # charging from solar still cycles

        elif action == Action.DISCHARGE_GRID:
            # Force Discharge: export from battery at max rate + excess solar
            solar_to_load = min(solar_kwh, load_kwh)
            remaining_load = load_kwh - solar_to_load
            solar_excess = max(0, solar_kwh - load_kwh)

            # Discharge battery: cover home load + export to grid
            available_discharge = max(0, soc_kwh - self.min_soc) * self.eta
            battery_discharge = min(available_discharge, max_discharge_kwh)
            battery_to_load = min(remaining_load, battery_discharge)
            battery_to_grid = battery_discharge - battery_to_load
            soc_used = battery_discharge / self.eta

            new_soc = soc_kwh - soc_used
            grid_import = max(0, remaining_load - battery_to_load)
            grid_export = battery_to_grid + solar_excess
            energy_cycled = soc_used

        # Clamp SoC to valid range
        new_soc = max(self.min_soc, min(self.capacity, new_soc))

        # Degradation cost (cents) based on energy cycled
        degradation_cents = energy_cycled * self.degradation_per_kwh

        # Net cost: what we pay minus what we earn (negative = profit)
        import_cost = grid_import * inputs.import_price
        export_revenue = grid_export * inputs.export_price
        net_cost = import_cost - export_revenue + degradation_cents

        return PeriodResult(
            new_soc_kwh=new_soc,
            grid_import_kwh=grid_import,
            grid_export_kwh=grid_export,
            degradation_cost_cents=degradation_cents,
            net_cost_cents=net_cost,
        )

    def clamp_soc(self, soc: float) -> float:
        return max(self.min_soc, min(self.capacity, soc))

    def discretize_soc(self, soc: float, step: float = 0.5) -> float:
        """Round SoC to nearest step for DP state quantization."""
        return round(soc / step) * step

    def apply_action_vec(
        self,
        soc_arr: np.ndarray,
        action: Action,
        solar_kw: float,
        load_kw: float,
        import_price: float,
        export_price: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized apply_action over an array of SoC values.

        Returns (new_soc, profit) arrays, where profit = -net_cost.
        Used by the DP backward pass for performance.
        """
        solar_kwh = solar_kw * PERIOD_HOURS
        load_kwh = load_kw * PERIOD_HOURS
        max_charge = self.max_power * PERIOD_HOURS
        max_discharge = self.max_power * PERIOD_HOURS
        cap = self.capacity
        eta = self.eta
        min_s = self.min_soc
        deg = self.degradation_per_kwh

        soc = soc_arr  # alias for readability

        if action == Action.GRID_CHARGE:
            solar_to_load = min(solar_kwh, load_kwh)
            solar_excess = solar_kwh - solar_to_load
            solar_to_bat = np.minimum(solar_excess, (cap - soc) / eta)
            solar_to_grid = solar_excess - solar_to_bat
            grid_to_load = load_kwh - solar_to_load
            room = (cap - soc) / eta - solar_to_bat
            grid_to_bat = np.minimum(max_charge, np.maximum(0.0, room))
            new_soc = np.minimum(soc + (solar_to_bat + grid_to_bat) * eta, cap)
            grid_import = grid_to_load + grid_to_bat
            grid_export = np.float64(solar_to_grid)
            energy_cycled = solar_to_bat + grid_to_bat

        elif action == Action.GRID_CHARGE_NO_EXPORT:
            # Same as GRID_CHARGE but excess solar curtailed (no export)
            solar_to_load = min(solar_kwh, load_kwh)
            solar_to_bat = np.minimum(solar_kwh - solar_to_load, (cap - soc) / eta)
            grid_to_load = load_kwh - solar_to_load
            room = (cap - soc) / eta - solar_to_bat
            grid_to_bat = np.minimum(max_charge, np.maximum(0.0, room))
            new_soc = np.minimum(soc + (solar_to_bat + grid_to_bat) * eta, cap)
            grid_import = grid_to_load + grid_to_bat
            grid_export = np.float64(0.0)
            energy_cycled = solar_to_bat + grid_to_bat

        elif action == Action.SELF_USE:
            solar_to_load = min(solar_kwh, load_kwh)
            remaining_load = load_kwh - solar_to_load
            solar_excess = solar_kwh - solar_to_load
            solar_to_bat = np.minimum(solar_excess,
                                       np.minimum((cap - soc) / eta, max_charge))
            solar_to_grid = solar_excess - solar_to_bat
            soc_charged = soc + solar_to_bat * eta
            avail_discharge = np.maximum(0.0, soc_charged - min_s) * eta
            bat_to_load = np.minimum(remaining_load,
                                      np.minimum(avail_discharge, max_discharge))
            soc_discharged = bat_to_load / eta
            new_soc = soc_charged - soc_discharged
            grid_import = np.maximum(0.0, remaining_load - bat_to_load)
            grid_export = solar_to_grid
            energy_cycled = solar_to_bat + soc_discharged

        elif action == Action.SELF_USE_NO_EXPORT:
            # Self-Use with export limit 0W: same as SELF_USE but excess solar curtailed
            solar_to_load = min(solar_kwh, load_kwh)
            remaining_load = load_kwh - solar_to_load
            solar_excess = solar_kwh - solar_to_load
            solar_to_bat = np.minimum(solar_excess,
                                       np.minimum((cap - soc) / eta, max_charge))
            soc_charged = soc + solar_to_bat * eta
            avail_discharge = np.maximum(0.0, soc_charged - min_s) * eta
            bat_to_load = np.minimum(remaining_load,
                                      np.minimum(avail_discharge, max_discharge))
            soc_discharged = bat_to_load / eta
            new_soc = soc_charged - soc_discharged
            grid_import = np.maximum(0.0, remaining_load - bat_to_load)
            grid_export = np.float64(0.0)  # export curtailed
            energy_cycled = solar_to_bat + soc_discharged

        elif action == Action.HOLD:
            solar_to_load = min(solar_kwh, load_kwh)
            solar_excess = solar_kwh - solar_to_load
            solar_to_bat = np.minimum(solar_excess,
                                       np.minimum((cap - soc) / eta, max_charge))
            solar_to_grid = solar_excess - solar_to_bat
            new_soc = np.minimum(soc + solar_to_bat * eta, cap)
            grid_import = np.float64(max(0.0, load_kwh - solar_to_load))
            grid_export = solar_to_grid
            energy_cycled = solar_to_bat

        elif action == Action.DISCHARGE_GRID:
            solar_to_load = min(solar_kwh, load_kwh)
            remaining_load = load_kwh - solar_to_load
            solar_excess = max(0.0, solar_kwh - load_kwh)
            avail_discharge = np.maximum(0.0, soc - min_s) * eta
            bat_discharge = np.minimum(avail_discharge, max_discharge)
            bat_to_load = np.minimum(remaining_load, bat_discharge)
            bat_to_grid = bat_discharge - bat_to_load
            soc_used = bat_discharge / eta
            new_soc = soc - soc_used
            grid_import = np.maximum(0.0, remaining_load - bat_to_load)
            grid_export = bat_to_grid + solar_excess
            energy_cycled = soc_used
        else:
            new_soc = soc.copy()
            grid_import = np.float64(0.0)
            grid_export = np.float64(0.0)
            energy_cycled = np.float64(0.0)

        new_soc = np.clip(new_soc, min_s, cap)
        degradation = energy_cycled * deg
        net_cost = grid_import * import_price - grid_export * export_price + degradation
        return new_soc, -net_cost  # profit = -cost
