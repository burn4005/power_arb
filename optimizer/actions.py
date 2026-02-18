from enum import Enum


class Action(Enum):
    """Battery actions the optimizer can choose.

    These map 1:1 to distinct FoxESS inverter behaviours:
    - GRID_CHARGE           -> Force Charge (mode 3)
    - GRID_CHARGE_NO_EXPORT -> Force Charge (mode 3) + export limit 0W
    - SELF_USE              -> Self-Use (mode 0), min SoC low
    - SELF_USE_NO_EXPORT    -> Self-Use (mode 0), min SoC low, export limit 0W
    - HOLD                  -> Self-Use (mode 0), min SoC = current (prevents discharge)
    - DISCHARGE_GRID        -> Force Discharge (mode 4)
    """

    GRID_CHARGE = "grid_charge"
    """Import from grid to charge battery at max rate. Excess solar exports to grid.
    Use when: import price is very low/negative, export price is non-negative."""

    GRID_CHARGE_NO_EXPORT = "grid_charge_no_export"
    """Import from grid to charge battery at max rate. Excess solar curtailed (not exported).
    Use when: import price is very low/negative AND export price is negative
    (paying to export would add cost on top of grid charging)."""

    SELF_USE = "self_use"
    """FoxESS Self-Use mode: solar covers load, excess solar charges battery,
    battery discharges to cover load when solar is insufficient, any remaining
    excess solar exports to grid. This is the inverter's native behaviour."""

    SELF_USE_NO_EXPORT = "self_use_no_export"
    """Self-Use mode with grid export limit set to 0W. Solar covers load,
    excess solar charges battery, battery discharges to cover load, but any
    remaining excess solar is curtailed (not exported). Use when: export price
    is negative and exporting would cost money."""

    HOLD = "hold"
    """Self-Use mode but with min SoC set to current level so the battery
    won't discharge. Home powered from solar + grid only. Battery preserves
    charge for a future higher-value period."""

    DISCHARGE_GRID = "discharge_grid"
    """Force Discharge: export from battery to grid at max rate.
    Use when: export price is very high (spike events, evening peak)."""
