import itertools
import numpy as np
import pandas as pd
from .config import MPCConf
from .cost import sequence_cost, hard_constraint_violation

def simulate_forward_open_loop(
    state_hist: pd.DataFrame,
    forecaster,
    build_feature_fn,
    action_seq,
    action_to_state_effect,
    conf: MPCConf,
):
    """
    Rolls the model forward over the horizon under a candidate action sequence.
    - action_to_state_effect: function(state_df, action_level) -> updated_state_df
      that injects the actuator effect (e.g., aeration ON cools temps slightly).
    Returns:
      temps_traj: list of controlled temp (e.g., active1) per step
      o2_traj:    list of o2 per step (if rule-based, synthesize or clamp >= 10%)
    """
    tmp = state_hist.copy()
    temps_traj, o2_traj = [], []

    for a in action_seq:
        # Apply action's immediate effect to the latest state (domain heuristic)
        tmp = action_to_state_effect(tmp, a)

        # Predict next-step temps
        feats = build_feature_fn(tmp)
        y = forecaster.model.predict(feats.values)[0]

        # Map y -> columns (adapt indexes to your model’s output order)
        out_cols = [
            "temperature_active1","temperature_active2","temperature_active3","temperature_active4",
            "temperature_curing1","temperature_curing2"
        ]
        next_row = tmp.iloc[[-1]].copy()
        for i, c in enumerate(out_cols[:len(y)]):
            next_row[c] = float(y[i])

        # (Rule) keep oxygen >= 10% (you can simulate pump dynamics if you want)
        o2_next = max(conf.o2_floor, float(tmp.iloc[-1:].get("oxygen", 0.21)))
        next_row["oxygen"] = o2_next

        # Append and continue
        tmp = pd.concat([tmp, next_row], ignore_index=True)
        temps_traj.append(float(next_row["temperature_active1"]))
        o2_traj.append(o2_next)

    return temps_traj, o2_traj

def choose_action(
    state_hist: pd.DataFrame,
    forecaster,
    build_feature_fn,
    action_to_state_effect,
    conf: MPCConf,
):
    H = conf.horizon_steps
    action_levels = conf.action_levels
    best, bestJ = None, float("inf")

    for seq in itertools.product(action_levels, repeat=H):
        temps, o2s = simulate_forward_open_loop(
            state_hist, forecaster, build_feature_fn, seq, action_to_state_effect, conf
        )
        # Hard constraints check
        if any(hard_constraint_violation(t, o2, conf) for t, o2 in zip(temps, o2s)):
            continue
        J = sequence_cost(temps, seq, conf)
        if J < bestJ:
            best, bestJ = seq, J

    # Fallback if all sequences violate constraints: force aeration ON
    if best is None:
        return 1
    return best[0]
