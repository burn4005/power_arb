from enum import Enum


class Action(Enum):
    """Battery actions the optimizer can choose.

    These map 1:1 to distinct FoxESS inverter behaviours:
    - GRID_CHARGE  -> Force Charge (mode 3)
    - SELF_USE     -> Self-Use (mode 0), min SoC low
    - HOLD         -> Self-Use (mode 0), min SoC = current (prevents discharge)
    - DISCHARGE_GRID -> Force Discharge (mode 4)
    """

    GRID_CHARGE = "grid_charge"
    """Import from grid to charge battery at max rate.
    Use when: import price is very low/negative and higher prices expected later."""

    SELF_USE = "self_use"
    """FoxESS Self-Use mode: solar covers load, excess solar charges battery,
    battery discharges to cover load when solar is insufficient, any remaining
    excess solar exports to grid. This is the inverter's native behaviour."""

    HOLD = "hold"
    """Self-Use mode but with min SoC set to current level so the battery
    won't discharge. Home powered from solar + grid only. Battery preserves
    charge for a future higher-value period."""

    DISCHARGE_GRID = "discharge_grid"
    """Force Discharge: export from battery to grid at max rate.
    Use when: export price is very high (spike events, evening peak)."""
