# simulated_bin/controller_sim.py
from __future__ import annotations
import argparse
import time
from typing import Dict
import sys
from pathlib import Path

# ---- Robust path bootstrap so imports work whether src/ lives at root or lida_mcp/src
ROOT = Path(__file__).resolve().parents[1]  # project root (.. from simulated_bin/)
CANDIDATE_PARENTS_OF_SRC = [
    ROOT,                  # when src/ is directly under project root
    ROOT / "lida_mcp",     # your current structure: lida_mcp/src/...
]
for parent in CANDIDATE_PARENTS_OF_SRC:
    if parent.exists() and str(parent) not in sys.path:
        sys.path.insert(0, str(parent))

# Optional: keep ROOT itself on path for non-package imports in root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---- Config
try:
    from src.config import MPCConf
except Exception:
    from config import MPCConf  # fallback if modules aren’t under a src package

# ---- Safety
try:
    from src.safety import safety_check
except Exception:
    try:
        from safety import safety_check
    except Exception:
        class _S:
            def __init__(self, ok, reason=""): self.ok = ok; self.reason = reason
        def safety_check(frame):
            t = float(frame.get("temperature_active1", 0))
            o2 = float(frame.get("oxygen", 0.21))
            ok = (t < 80.0) and (o2 >= 0.10)
            return _S(ok, "" if ok else "Fallback safety tripped")

# ---- Simulator
from .sim_bin import BinSim

# ---- Optional MPC stack; fallback to rule-based if unavailable
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

# ---------- Control policies ----------
def rule_based_action(frame: Dict, conf: MPCConf) -> Dict:
    t = float(frame["temperature_active1"])
    o2 = float(frame.get("oxygen", 0.21))
    fan, lid, mix = 0.0, False, False

    if t > conf.temp_hi:
        fan = 1.0; lid = True
        if t > (conf.temp_hi + 3.0): mix = True
    elif t < (conf.temp_lo - 1.0):
        fan = 0.0; lid = False
    else:
        fan = 0.5

    if o2 < conf.o2_floor:
        fan = 1.0; lid = True

    return {"fan_level": fan, "lid_open": lid, "paddle_mix": mix}

def mpc_action(state_hist_df, forecaster, conf: MPCConf):
    def action_to_state_effect(df, a_level):
        last = df.iloc[[-1]].copy()
        cool = 0.55 * a_level
        for c in ["temperature_active1","temperature_active2","temperature_active3","temperature_active4"]:
            last[c] = last[c] - cool
        out = df.copy(); out.iloc[-1] = last.iloc[0]; return out

    a = choose_action(
        state_hist=state_hist_df,
        forecaster=forecaster,
        build_feature_fn=lambda h: build_feature_row(h),
        action_to_state_effect=action_to_state_effect,
        conf=conf,
    )
    return {"fan_level": float(a), "lid_open": bool(a > 0.0), "paddle_mix": False}

# ---------- Main loop ----------
def run_sim(steps: int, use_mpc: bool, sleep_s: float, seed: int | None = None):
    conf = MPCConf()
    bin = BinSim(start_temp_active=58.0, start_temp_curing=50.0, seed=seed)

    state_hist_df, forecaster = None, None
    if use_mpc and HAVE_MPC:
        forecaster = Forecaster()

    import pandas as pd
    history_rows = []

    for k in range(steps):
        frame = bin.step()
        s = safety_check(frame)

        if not s.ok:
            act = {"fan_level": 1.0, "lid_open": True, "paddle_mix": True}
        else:
            if use_mpc and HAVE_MPC:
                history_rows.append(frame)
                state_hist_df = pd.DataFrame(history_rows[-12:])
                act = mpc_action(state_hist_df, forecaster, conf)
            else:
                act = rule_based_action(frame, conf)

        bin.set_actuators(**act)

        print(f"[{k:03d}] Tact1={frame['temperature_active1']:.2f}C "
              f"Tcur1={frame['temperature_curing1']:.2f}C "
              f"O2={frame['oxygen']:.3f}  "
              f"fan={act['fan_level']:.2f} lid={'open' if act['lid_open'] else 'closed'} "
              f"{'mix' if act['paddle_mix'] else ''}")

        if sleep_s > 0:
            time.sleep(sleep_s)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run simulated compost bin controller (no MQTT/FastAPI).")
    p.add_argument("--steps", type=int, default=60, help="Number of control steps to simulate")
    p.add_argument("--sleep-s", type=float, default=0, help="Seconds to sleep between steps (0 = fastest)")
    p.add_argument("--mpc", action="store_true", help="Use MPC (requires trained model files).")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = p.parse_args()
    run_sim(steps=args.steps, use_mpc=args.mpc, sleep_s=args.sleep_s, seed=args.seed)
