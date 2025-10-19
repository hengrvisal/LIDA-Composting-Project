# src/main.py
"""
Offline entrypoint: controller with internal simulated bin.
Usage:
  python -m src.main controller --simulate --steps 600 --step-sleep-s 0
"""
import argparse
import sys
from pathlib import Path

# Ensure 'from src.* import ...' works if executed from within lida_mcp/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.controller import run_controller
from src.config import MPCConf

def main(argv=None):
    p = argparse.ArgumentParser(description="LIDA Composting — Offline Simulated Controller")
    sub = p.add_subparsers(dest="mode", required=True)

    pc = sub.add_parser("controller", help="Run the offline controller")
    pc.add_argument("--simulate", action="store_true", help="Use internal simulated bin (required)")
    pc.add_argument("--steps", type=int, default=600, help="Number of control steps")
    pc.add_argument("--step-sleep-s", type=int, default=0, help="Seconds between steps (0 = as fast as possible)")

    args = p.parse_args(argv)

    if args.mode == "controller":
        if not args.simulate:
            print("[ERROR] This build only supports --simulate (offline).")
            sys.exit(2)
        conf = MPCConf()
        run_controller(sensor_stream=None, conf=conf, step_sleep_s=args.step_sleep_s, steps=args.steps)

if __name__ == "__main__":
    main()
