#!/usr/bin/env python3
"""
main.py — Entry point for the LIDA Composting MPC system

Usage examples:
  # 1) Run controller with simulated sensor data (default)
  python main.py controller --simulate

  # 2) Run FastAPI prediction server
  python main.py api --host 127.0.0.1 --port 8000
"""
import argparse
import importlib
import sys
import random
import datetime as dt
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


def _import(name_variants):
    """
    Try importing a module/attribute from any of the given dotted paths.
    Returns the imported object or raises the last ImportError.
    """
    last_err = None
    for dotted in name_variants:
        try:
            module_name, attr = dotted.rsplit(":", 1)
            mod = importlib.import_module(module_name)
            return getattr(mod, attr)
        except Exception as e:
            last_err = e
    raise last_err

# Flexible imports to support both flat and 'src.' layouts
run_controller = _import(["controller:run_controller", "src.controller:run_controller"])
MPCConf       = _import(["config:MPCConf", "src.config:MPCConf"])

def fake_sensor_stream(start_temp: float = 58.0):
    """Simple synthetic generator for testing without real sensors/MQTT."""
    temp = start_temp
    while True:
        temp += random.uniform(-0.2, 0.3)
        yield {
            "time_stamp": dt.datetime.utcnow().isoformat(),
            "temperature_active1": temp,
            "temperature_active2": temp - 0.5,
            "temperature_active3": temp + 0.6,
            "temperature_active4": temp - 0.2,
            "temperature_curing1": temp - 6.0,
            "temperature_curing2": temp - 5.0,
            "oxygen": 0.21,
            "co2": 0.03,
            "methane": 0.003,
        }

def run_api(host: str, port: int, reload: bool):
    # Import lazily to avoid uvicorn/fastapi dependency for controller-only use
    try:
        import uvicorn
    except ImportError:
        print("[ERROR] uvicorn is not installed. Install with: pip install uvicorn fastapi")
        sys.exit(1)

    # server_api may live as server_api.py or src/server_api.py
    app = _import(["server_api:app", "src.server_api:app"])
    uvicorn.run(app, host=host, port=port, reload=reload)

def main(argv=None):
    p = argparse.ArgumentParser(description="Run the LIDA Composting MPC system")
    sub = p.add_subparsers(dest="mode", required=True)

    # Controller mode
    pc = sub.add_parser("controller", help="Run the MPC controller loop")
    pc.add_argument("--simulate", action="store_true", help="Use built-in fake sensor stream")
    pc.add_argument("--start-temp", type=float, default=58.0, help="Starting temp for simulated stream")
    pc.add_argument("--step-sleep-s", type=int, default=60, help="Seconds between control evaluations")

    # API mode
    pa = sub.add_parser("api", help="Run the FastAPI prediction server")
    pa.add_argument("--host", default="127.0.0.1", help="Host interface to bind")
    pa.add_argument("--port", type=int, default=8000, help="Port to bind")
    pa.add_argument("--reload", action="store_true", help="Auto-reload code on changes")

    args = p.parse_args(argv)

    if args.mode == "controller":
        conf = MPCConf()
        if args.simulate:
            stream = fake_sensor_stream(start_temp=args.start_temp)
            run_controller(sensor_stream=stream, conf=conf, step_sleep_s=args.step_sleep_s)
        else:
            print("[ERROR] Non-simulated sensor stream not implemented in main.py.")
            print("        Either pass --simulate or integrate your MQTT subscriber to yield frames.")
            sys.exit(2)

    elif args.mode == "api":
        run_api(host=args.host, port=args.port, reload=args.reload)

if __name__ == "__main__":
    main()
