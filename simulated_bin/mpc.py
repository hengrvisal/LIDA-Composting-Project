import numpy as np
from .features import build_feature_row as build_feature_fn
from .cost import compute_cost

def choose_action(state_hist, forecaster, build_feature_fn, action_to_state_effect, conf):
    """
    Tries a few candidate fan levels and picks the one with lowest cost.
    """
    actions = np.linspace(0.0, 1.0, 5)  # test 0%, 25%, 50%, 75%, 100% fan
    best_cost, best_a = float("inf"), 0.0

    for a in actions:
        df_future = action_to_state_effect(state_hist, a)
        feats = build_feature_fn(df_future)
        pred_temp = forecaster.predict(feats)
        cost = compute_cost(np.array([pred_temp]), [a], conf)
        if cost < best_cost:
            best_cost, best_a = cost, a

    return best_a

def mpc_action(state_hist_df, forecaster, conf):
    last = state_hist_df.iloc[-1]
    t_last = float(last["temperature_active1"])
    o2_last = float(last.get("oxygen", 0.21))
    if getattr(conf, "warmup_lockout", False) and t_last < getattr(conf, "warmup_temp_c", 50.0):
        # mirror warm-up O2 keeper behavior
        o2_target = getattr(conf, "warmup_o2_floor", 0.10) + getattr(conf, "warmup_o2_margin", 0.01)
        deficit = max(0.0, o2_target - o2_last)
        if deficit > 0:
            kp = getattr(conf, "warmup_kp", 4.0)
            fan = min(getattr(conf, "warmup_max_fan", 0.5),
                      max(getattr(conf, "warmup_min_fan", 0.2), kp * deficit))
        else:
            fan = 0.0
        return {"fan_level": float(fan), "lid_open": bool(getattr(conf, "warmup_lid_open", False)), "paddle_mix": False}
    ...

