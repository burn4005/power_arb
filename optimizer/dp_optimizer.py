import logging
from dataclasses import dataclass

import numpy as np

import config
from optimizer.actions import Action
from optimizer.battery_model import BatteryModel, PeriodInputs, PERIOD_HOURS

logger = logging.getLogger(__name__)

# SoC discretization step (kWh)
SOC_STEP = 0.5


@dataclass
class OptimizationResult:
    """Result from the DP optimizer for the current period."""
    action: Action
    expected_profit_cents: float  # total expected profit over horizon
    schedule: list[tuple[str, Action, float]]  # (timestamp, action, marginal_profit) for each period
    soc_trajectory: list[float]  # predicted SoC (kWh) at each period
    reason: str


class DPOptimizer:
    """Rolling-horizon dynamic programming optimizer.

    Finds the sequence of battery actions over the next 48 hours that
    maximizes total profit (export revenue - import cost - degradation).

    Complexity: O(S * T * A) where S=SoC states (~85), T=periods (~576),
    A=actions (6). Total ~295,000 evaluations, runs in ~75ms.
    """

    def __init__(self):
        self.model = BatteryModel()
        self.capacity = config.battery.capacity_kwh
        self.min_soc = config.battery.min_soc_kwh
        self.max_charge_price = config.battery.max_charge_price_c
        self.min_discharge_price = config.battery.min_discharge_price_c

        # Pre-compute SoC grid
        self.soc_levels = np.arange(self.min_soc, self.capacity + SOC_STEP, SOC_STEP)
        self.n_states = len(self.soc_levels)
        self.soc_to_idx = {round(s, 2): i for i, s in enumerate(self.soc_levels)}

    def optimize(
        self,
        current_soc_kwh: float,
        import_prices: list[float],   # cents/kWh for each period
        export_prices: list[float],   # cents/kWh for each period
        solar_forecast: list[float],  # kW average for each period
        load_forecast: list[float],   # kW average for each period
        timestamps: list[str] | None = None,
    ) -> OptimizationResult:
        """Run DP optimization and return the best action for now.

        All input lists must be the same length (number of periods).
        """
        n_periods = len(import_prices)
        if n_periods == 0:
            return OptimizationResult(
                action=Action.SELF_USE,
                expected_profit_cents=0,
                schedule=[],
                soc_trajectory=[],
                reason="No forecast data available",
            )

        actions = list(Action)
        n_actions = len(actions)
        timestamps = timestamps or [str(i) for i in range(n_periods)]

        # Value function: V[t][s] = max future profit from period t in state s
        # Policy: policy[t][s] = best action index at period t in state s
        V = np.full((n_periods + 1, self.n_states), 0.0)
        policy = np.zeros((n_periods, self.n_states), dtype=int)

        # Round-trip efficiency (eta^2): charging at P_import and discharging at
        # P_future only profits when P_import < P_future * eta^2.
        rt_efficiency = self.model.eta ** 2

        # Backward pass: vectorized over all SoC states at once
        soc_arr = self.soc_levels.astype(np.float64)
        for t in range(n_periods - 1, -1, -1):
            best_values = np.full(self.n_states, -np.inf)
            best_actions = np.zeros(self.n_states, dtype=int)

            # Average import price over the next 24 hours from t — used as the
            # "expected future price" benchmark for grid charge decisions.
            lookahead_end = min(t + 12 * 24, n_periods)  # 24h = 288 periods
            avg_future_import = float(np.mean(import_prices[t:lookahead_end]))

            for a_idx, action in enumerate(actions):
                # ── Per-period guardrails (scalar, apply to all states) ──────
                if action in (Action.GRID_CHARGE, Action.GRID_CHARGE_NO_EXPORT):
                    # Hard price cap (user-configured)
                    if (self.max_charge_price > 0
                            and import_prices[t] > self.max_charge_price):
                        continue
                    # Economic guardrail: only charge from grid when the current
                    # import price is meaningfully below the expected future
                    # average (adjusted for round-trip losses).  This prevents
                    # the optimizer from doing marginal/defensive charges when
                    # the battery is near empty — the inverter handles min-SoC
                    # protection natively in SELF_USE mode.
                    if import_prices[t] >= avg_future_import * rt_efficiency:
                        continue

                if action == Action.DISCHARGE_GRID:
                    # Hard price floor (user-configured)
                    if (self.min_discharge_price > 0
                            and export_prices[t] < self.min_discharge_price):
                        continue
                    # Economic guardrail: only discharge when the current
                    # export price is meaningfully above the expected future
                    # average (adjusted for round-trip losses).  This prevents
                    # the optimizer from discharging into mediocre export
                    # periods — better to hold charge for genuine spikes.
                    avg_future_export = float(np.mean(export_prices[t:lookahead_end]))
                    if export_prices[t] <= avg_future_export / rt_efficiency:
                        continue

                new_soc, profit = self.model.apply_action_vec(
                    soc_arr, action,
                    solar_forecast[t], load_forecast[t],
                    import_prices[t], export_prices[t],
                )

                # Map new SoC to state indices
                new_s_idx = np.clip(
                    np.round((new_soc - self.min_soc) / SOC_STEP).astype(int),
                    0, self.n_states - 1,
                )

                total_value = profit + V[t + 1][new_s_idx]

                # ── Per-state guardrail: skip grid charge when battery is
                # already at or within one step of full capacity.  At full SoC
                # grid_to_bat == 0 so GRID_CHARGE is identical to HOLD/SELF_USE,
                # but wins tie-breaks by accident (it's first in the enum). ───
                if action in (Action.GRID_CHARGE, Action.GRID_CHARGE_NO_EXPORT):
                    total_value = np.where(
                        soc_arr >= self.capacity - SOC_STEP, -np.inf, total_value
                    )

                better = total_value > best_values
                best_values[better] = total_value[better]
                best_actions[better] = a_idx

            V[t] = best_values
            policy[t] = best_actions

        # Forward pass: trace optimal schedule from current SoC
        current_soc_disc = round(
            round(self.model.clamp_soc(current_soc_kwh) / SOC_STEP) * SOC_STEP, 2
        )
        s_idx = self.soc_to_idx.get(
            current_soc_disc,
            min(range(self.n_states),
                key=lambda i: abs(self.soc_levels[i] - current_soc_kwh))
        )

        schedule = []
        soc_trajectory = []
        soc = self.soc_levels[s_idx]
        total_profit = V[0][s_idx]

        for t in range(n_periods):
            soc_trajectory.append(float(soc))
            action = actions[policy[t][s_idx]]
            inputs = PeriodInputs(
                solar_kw=solar_forecast[t],
                load_kw=load_forecast[t],
                import_price=import_prices[t],
                export_price=export_prices[t],
            )
            result = self.model.apply_action(soc, action, inputs)
            marginal_profit = -result.net_cost_cents

            schedule.append((timestamps[t], action, marginal_profit))

            # Advance to next state
            new_soc_clamped = self.model.clamp_soc(result.new_soc_kwh)
            new_soc_disc = round(
                round(new_soc_clamped / SOC_STEP) * SOC_STEP, 2
            )
            s_idx = self.soc_to_idx.get(
                new_soc_disc,
                min(range(self.n_states),
                    key=lambda i: abs(self.soc_levels[i] - new_soc_clamped))
            )
            soc = self.soc_levels[s_idx]

        # The optimal action for the current period
        current_action = schedule[0][1] if schedule else Action.SELF_USE
        reason = self._explain_action(
            current_action, import_prices[0], export_prices[0],
            solar_forecast[0], load_forecast[0], current_soc_kwh, schedule
        )

        # Compare to HOLD baseline for the first period
        hold_inputs = PeriodInputs(
            solar_kw=solar_forecast[0], load_kw=load_forecast[0],
            import_price=import_prices[0], export_price=export_prices[0],
        )
        hold_result = self.model.apply_action(current_soc_kwh, Action.HOLD, hold_inputs)
        hold_cost = hold_result.net_cost_cents

        logger.info(
            "Optimal action: %s | SoC: %.1f kWh | Import: %.1f c | Export: %.1f c | "
            "Horizon profit: %.1f c | Reason: %s",
            current_action.name, current_soc_kwh,
            import_prices[0], export_prices[0], total_profit, reason,
        )

        return OptimizationResult(
            action=current_action,
            expected_profit_cents=total_profit,
            schedule=schedule,
            soc_trajectory=soc_trajectory,
            reason=reason,
        )

    def _explain_action(
        self, action: Action, import_price: float, export_price: float,
        solar_kw: float, load_kw: float, soc_kwh: float,
        schedule: list[tuple[str, Action, float]],
    ) -> str:
        """Generate a human-readable explanation for the chosen action."""
        soc_pct = soc_kwh / self.capacity * 100

        if action in (Action.GRID_CHARGE, Action.GRID_CHARGE_NO_EXPORT):
            next_discharge = None
            for ts, a, _ in schedule[1:]:
                if a == Action.DISCHARGE_GRID:
                    next_discharge = ts
                    break
            no_export_tag = " [no export]" if action == Action.GRID_CHARGE_NO_EXPORT else ""
            msg = (f"Charging from grid at {import_price:.1f}c/kWh, "
                   f"export {export_price:.1f}c/kWh (SoC {soc_pct:.0f}%){no_export_tag}")
            if next_discharge:
                msg += f"; discharge expected at {next_discharge}"
            return msg

        if action in (Action.SELF_USE, Action.SELF_USE_NO_EXPORT):
            no_export_tag = " [no export]" if action == Action.SELF_USE_NO_EXPORT else ""
            return (f"Self-use{no_export_tag}: solar {solar_kw:.1f}kW, load {load_kw:.1f}kW, "
                    f"SoC {soc_pct:.0f}%")

        if action == Action.HOLD:
            future_exports = [p for _, a, p in schedule[1:20]
                              if a == Action.DISCHARGE_GRID]
            if future_exports:
                return (f"Holding at {soc_pct:.0f}% SoC; "
                        f"discharge expected, avg profit {sum(future_exports)/len(future_exports):.1f}c")
            return f"Holding at {soc_pct:.0f}% SoC; no profitable action now"

        if action == Action.DISCHARGE_GRID:
            return (f"Exporting from battery at {export_price:.1f}c/kWh "
                    f"(SoC {soc_pct:.0f}%)")

        return action.name
