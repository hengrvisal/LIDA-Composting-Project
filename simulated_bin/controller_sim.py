# simulated_bin/controller_sim.py
from __future__ import annotations
import argparse, time, sys
from typing import Dict
from pathlib import Path
import pandas as pd

# --- Path bootstrap so imports work whether src/ lives at root or lida_mcp/src
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
if str(ROOT / "lida_mcp") not in sys.path: sys.path.insert(0, str(ROOT / "lida_mcp"))

from simulated_bin.sim_bin import BinSim
from simulated_bin.config import MPCConf
from simulated_bin.safety import safety_check

# Optional ML forecaster + MPC (fallback to rule-based if unavailable)
HAVE_MPC = True
try:
    try:
        from src.mpc import choose_action
        from src.models import Forecaster
        from src.features import build_feature_row
    except Exception:
        from mpc import choose_action
        from models import Forecaster
        from features import build_feature_row
except Exception:
    HAVE_MPC = False

def rule_based_action(frame: Dict, conf: MPCConf) -> Dict:
    """Gentle keeper around setpoint band with O2 guard."""
    t = float(frame["temperature_active1"])
    o2 = float(frame["oxygen"])
    fan, lid, mix = 0.0, False, False

    if t > conf.temp_hi:          # cool down when above band
        fan = 0.8; lid = True
    elif t < conf.temp_lo - 1.0:  # let it warm when below band
        fan = 0.0
    else:
        fan = 0.4                  # small background ventilation

    if o2 < conf.o2_floor:        # O2 floor always wins
        fan = max(fan, 0.6); lid = True

    return {"fan_level": fan, "lid_open": lid, "paddle_mix": mix}

def mpc_action(state_hist_df, forecaster, conf: MPCConf):
    """MPC wrapper using your ML forecaster. We approximate fan's cooling in the horizon."""
    def action_to_state_effect(df, a_level):
        # simple “fan cools” proxy for optimization
        last = df.iloc[[-1]].copy()
        cool = 0.45 * a_level
        last["temperature_active1"] = last["temperature_active1"] - cool
        out = df.copy(); out.iloc[-1] = last.iloc[0]
        return out

    a = choose_action(
        state_hist=state_hist_df,
        forecaster=forecaster,
        build_feature_fn=lambda h: build_feature_row(h),
        action_to_state_effect=action_to_state_effect,
        conf=conf,
    )
    return {"fan_level": float(a), "lid_open": bool(a > 0.0), "paddle_mix": False}

def run_sim(steps: int, use_mpc: bool, sleep_s: float, seed: int | None, dt_min: float):
    conf = MPCConf()

    # NOTE: BinSim expects dt_s (seconds per step). Convert from minutes.
    dt_s = float(dt_min) * 60.0
    sim = BinSim(start_temp_active=25.0, start_temp_curing=50.0, seed=seed, dt_s=dt_s)

    # phase tracking (driver-side flags in addition to sim’s own)
    phase = "WARMUP"   # -> ACTIVE -> CURING
    last_phase = phase

    forecaster = Forecaster() if (use_mpc and HAVE_MPC) else None
    history_rows = []

    # simple mixing policy: mix when very hot (helps dissipation) or periodically in ACTIVE
    mix_period_steps = max(1, int(round(60.0 / dt_min)))  # ~every 60 minutes by default
    last_mix_step = -10**9

    for k in range(steps):
        frame = sim.step()
        s = safety_check(frame)

        # --- phase transitions (with hysteresis) based on temperature only here
        T = float(frame["temperature_active1"])
        if phase == "WARMUP" and T >= conf.warmup_temp_c:
            phase = "ACTIVE"
        elif phase == "ACTIVE" and T <= (conf.curing_exit_c - conf.phase_hyst_c):
            phase = "CURING"
        elif phase == "CURING" and T >= (conf.curing_exit_c + conf.phase_hyst_c):
            phase = "ACTIVE"

        if phase != last_phase:
            print(f"*** PHASE → {phase} @ step {k:03d}, T={T:.2f}°C ***")
            last_phase = phase

        # --- control policy selection ---
        if not s.ok:
            act = {"fan_level": 1.0, "lid_open": True, "paddle_mix": True}
            mode = "[safety]"
        elif phase == "WARMUP":
            # O2 keeper (target ≈ 12%) with small background fan to avoid O2 crash
            o2 = float(frame["oxygen"])
            o2_target = conf.warmup_o2_floor + conf.warmup_o2_margin  # e.g., 10% + 2%
            o2_off = o2_target + 0.01
            deficit = o2_target - o2
            if o2 < o2_target:
                fan_cmd = max(conf.warmup_min_fan,
                              min(conf.warmup_max_fan, conf.warmup_kp * max(0.0, deficit)))
            elif o2 < o2_off:
                fan_cmd = 0.15
            else:
                fan_cmd = 0.0
            mix = T > (conf.warmup_temp_c + 12.0)  # rare in warmup
            act = {"fan_level": float(fan_cmd), "lid_open": False, "paddle_mix": mix}
            mode = "[warmup-o2]"
        else:
            if use_mpc and HAVE_MPC:
                history_rows.append(frame)
                df = pd.DataFrame(history_rows[-12:])  # short horizon history
                act = mpc_action(df, forecaster, conf)
                mode = "[mpc]"
            else:
                act = rule_based_action(frame, conf)
                mode = "[rule]"

            # ACTIVE-phase mixing rule:
            #  - if T gets very hot (e.g., > 65°C), mix immediately (with 10 min min-gap)
            #  - otherwise, mix periodically (once per mix_period_steps) to vent heat
            if phase == "ACTIVE":
                if T > 65.0 and k - last_mix_step > max(1, int(round(10.0 / dt_min))):
                    act["paddle_mix"] = True
                elif (k - last_mix_step) >= mix_period_steps:
                    act["paddle_mix"] = True

        # record & emit paddle-mix event flag
        if act.get("paddle_mix"):
            last_mix_step = k
            print(f"*** MIX → paddle on @ step {k:03d}, T={T:.2f}°C ***")

        sim.set_actuators(**act)

        print(f"step {k:03d} | "
              f"T={frame['temperature_active1']:.2f}°C  "
              f"O2={frame['oxygen']:.2f}  "
              f"M={frame['moisture']:.3f}  "
              f"fan={act['fan_level']:.2f}  {mode}  phase={phase}")

        if sleep_s > 0:
            time.sleep(sleep_s)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Simulated compost bin controller (flags + mixing).")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--sleep-s", type=float, default=0.0)
    p.add_argument("--mpc", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--dt-min", type=float, default=1.0, help="Sim time per step in minutes")
    args = p.parse_args()
    run_sim(steps=args.steps, use_mpc=args.mpc, sleep_s=args.sleep_s, seed=args.seed, dt_min=args.dt_min)
