# simulated_bin/controller_sim.py
from __future__ import annotations
import argparse
import time
import pandas as pd
from typing import Dict
from dataclasses import dataclass

from .config import MPCConf
from .sim_bin import BinSim
from .safety import safety_check

HAVE_MPC = True
try:
    from .mpc import choose_action
    from .models import Forecaster
    from .features import build_feature_row
except Exception as e:
    print(f"[WARN] MPC stack unavailable: {e}")
    HAVE_MPC = False


@dataclass
class SafetyStatus:
    ok: bool
    reason: str = ""


def rule_based_action(frame: Dict, conf: MPCConf) -> Dict:
    t = float(frame["temperature_active1"])
    o2 = float(frame.get("oxygen", 0.21))
    fan, lid, mix = 0.0, False, False

    if t > conf.setpoint_c + 3:
        fan, lid = 1.0, True
    elif t > conf.setpoint_c:
        fan = 0.5
    elif t < conf.setpoint_c - 2:
        fan, lid = 0.0, False

    if o2 < 0.10:
        fan, lid = 1.0, True

    return {"fan_level": fan, "lid_open": lid, "paddle_mix": mix}


def mpc_action(state_hist_df, forecaster, conf: MPCConf):
    def action_to_state_effect(df, a_level):
        last = df.iloc[[-1]].copy()
        cool = 0.55 * a_level
        last["temperature_active1"] -= cool
        out = df.copy()
        out.iloc[-1] = last.iloc[0]
        return out

    a = choose_action(
        state_hist=state_hist_df,
        forecaster=forecaster,
        build_feature_fn=lambda h: build_feature_row(h),
        action_to_state_effect=action_to_state_effect,
        conf=conf,
    )
    return {"fan_level": float(a), "lid_open": bool(a > 0.0), "paddle_mix": False}


def run_sim(steps: int, use_mpc: bool, sleep_s: float, seed: int | None = None):
    conf = MPCConf()
    # start near ambient
    sim = BinSim(start_temp_active=25.0, start_temp_curing=50.0, seed=seed)

    forecaster = Forecaster() if (use_mpc and HAVE_MPC) else None
    history_rows = []

    for k in range(steps):
        frame = sim.step()
        s = safety_check(frame)

        # inside run_sim loop, after s = safety_check(frame)
        t_active = float(frame["temperature_active1"])
        o2 = float(frame.get("oxygen", 0.21))

        if not s.ok:
            act = {"fan_level": 1.0, "lid_open": True, "paddle_mix": True}
            mode = "[safety]"

        elif conf.warmup_lockout and t_active < conf.warmup_temp_c:
            # --- Warm-up O2 keeper ---
            # Keep just enough air to maintain O2 >= floor, otherwise no cooling.
                o2_target = conf.warmup_o2_floor + conf.warmup_o2_margin  # e.g., 0.12
                o2_off = o2_target + 0.01                                  # hysteresis to turn off
                deficit = o2_target - o2

                if o2 < o2_target:
                    # Stronger push up to the cap, but never below min_fan while below target
                    fan_cmd = max(conf.warmup_min_fan,
                                min(conf.warmup_max_fan, conf.warmup_kp * max(0.0, deficit)))
                elif o2 < o2_off:
                    fan_cmd = 0.15     # small trickle across the boundary
                else:
                    fan_cmd = 0.0

                act = {"fan_level": float(fan_cmd),
                    "lid_open": bool(conf.warmup_lid_open),
                    "paddle_mix": False}
                mode = "[warmup-o2]"


        elif use_mpc and HAVE_MPC:
            history_rows.append(frame)
            df = pd.DataFrame(history_rows[-12:])
            act = mpc_action(df, forecaster, conf)
            mode = "[mpc]"

        else:
            act = rule_based_action(frame, conf)
            mode = "[rule]"


        sim.set_actuators(**act)

        print(f"step {k:03d} | "
              f"T={frame['temperature_active1']:.2f}°C  "
              f"O2={frame['oxygen']:.2f}  "
              f"M={frame['moisture']:.3f}  "
              f"fan={act['fan_level']:.2f}  {mode}")

        if sleep_s > 0:
            time.sleep(sleep_s)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run simulated compost bin (ML-based MPC).")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--sleep-s", type=float, default=0)
    p.add_argument("--mpc", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    run_sim(steps=args.steps, use_mpc=args.mpc, sleep_s=args.sleep_s, seed=args.seed)
