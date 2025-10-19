# src/controller.py
# Offline controller with a built-in simulated bin (no MQTT/FastAPI).
from __future__ import annotations
import time, random
from dataclasses import dataclass
from typing import Dict, Generator, Optional

from .config import MPCConf
from .safety import safety_check, SafetyStatus  # keep using your existing safety checks

# ---------------------------
# Simple internal simulator
# ---------------------------
@dataclass
class SimState:
    temp_c: float
    moisture: float
    oxygen: float = 0.21

class SimulatedBin:
    """
    Very simple thermal/moisture model for offline testing:
    - Temperature increases due to microbial heat when idle
    - Aeration cools proportionally to 'cooldown_bias'
    - Moisture evaporates under aeration; recovers slowly when idle
    """
    def __init__(self, conf: MPCConf, seed: int = 42):
        self.conf = conf
        random.seed(seed)
        self.state = SimState(
            temp_c=58.0 + random.uniform(-1.5, 1.5),
            moisture=0.55 + random.uniform(-0.03, 0.03),
        )

    def step(self, dt_s: float, air_on_s: float):
        air_on_s = max(0.0, min(air_on_s, dt_s))
        cool = self.conf.cooldown_bias * air_on_s
        heat = self.conf.heat_gain_bias * dt_s

        # temperature dynamics
        self.state.temp_c += heat * (1.0 + 0.2 * random.uniform(-1, 1))
        self.state.temp_c -= cool * (1.0 + 0.2 * random.uniform(-1, 1))

        # moisture
        self.state.moisture -= self.conf.moisture_evap_per_sec * air_on_s
        if air_on_s < 1e-6:
            self.state.moisture += self.conf.moisture_recover_per_sec * dt_s

        # oxygen (keep above floor in sim)
        self.state.oxygen = max(self.conf.o2_floor, 0.21)

        # clamp
        self.state.temp_c = max(10.0, min(self.state.temp_c, self.conf.temp_hard_max))
        self.state.moisture = max(0.0, min(self.state.moisture, 1.0))

    def read(self) -> Dict[str, float]:
        # add tiny sensor noise
        return {
            "temperature_active1": self.state.temp_c + random.uniform(-0.15, 0.15),
            "oxygen": self.state.oxygen,
            "moisture": max(0.0, min(self.state.moisture + random.uniform(-0.005, 0.005), 1.0)),
        }

# ---------------------------
# Offline MPC (simple heuristic)
# ---------------------------
class OfflineMPC:
    def __init__(self, conf: MPCConf):
        self.conf = conf

    def decide_air_on_time(self, temp_c: float, moisture: float, dt_s: float) -> float:
        # safety takes precedence (handled in loop)
        err = temp_c - self.conf.setpoint_c
        if err <= self.conf.deadband_c:
            return 0.0

        on_time = self.conf.k_on_per_deg * max(0.0, err)

        # moisture protection
        if moisture < self.conf.moisture_min:
            scale = max(0.2, (moisture / self.conf.moisture_min))  # 0.2..1.0
            on_time *= scale

        # clamp
        on_time = max(self.conf.min_on_sec, min(on_time, self.conf.max_on_sec))
        return min(on_time, dt_s)

# ---------------------------
# Optional external stream (generator) for tests
# ---------------------------
def fake_sensor_stream(start_temp: float = 58.0) -> Generator[Dict[str, float], None, None]:
    temp = start_temp
    while True:
        temp += random.uniform(-0.2, 0.3)
        yield {
            "temperature_active1": temp,
            "oxygen": 0.21,
            "moisture": 0.55,
        }

# ---------------------------
# Controller loop
# ---------------------------
def run_controller(
    sensor_stream: Optional[Generator[Dict[str, float], None, None]] = None,
    conf: Optional[MPCConf] = None,
    step_sleep_s: Optional[int] = None,
    steps: int = 600,
):
    """
    If sensor_stream is None, use the internal simulator.
    Otherwise, we read from the stream and *still* compute a command (air_on_s),
    but we won't send it anywhere (offline mode).
    """
    conf = conf or MPCConf()
    dt = float(step_sleep_s if step_sleep_s is not None else conf.step_seconds)

    sim = SimulatedBin(conf) if sensor_stream is None else None
    mpc = OfflineMPC(conf)

    print("=== OFFLINE CONTROLLER (simulated bin, no MQTT/FastAPI) ===")
    print(f"Target: {conf.setpoint_c}°C  deadband: ±{conf.deadband_c}°C  dt={dt:.0f}s")
    print(f"Safety: O2≥{conf.o2_floor*100:.0f}%  hard max {conf.temp_hard_max}°C\n")

    for k in range(steps):
        # Read sensors
        frame = sim.read() if sim else next(sensor_stream)

        # Safety check (forces aeration if unsafe)
        s: SafetyStatus = safety_check(frame)
        if not s.ok:
            air_on_s = min(conf.max_on_sec, dt)  # force ON
            reason = s.reason or "safety"
        else:
            # MPC decision
            temp = float(frame["temperature_active1"])
            moist = float(frame.get("moisture", 0.55))
            air_on_s = mpc.decide_air_on_time(temp, moist, dt)
            reason = "mpc"

        # Advance sim if we own the plant
        if sim:
            sim.step(dt, air_on_s)

        # Log to console
        t = float(frame["temperature_active1"])
        o2 = float(frame.get("oxygen", 0.21))
        m = float(frame.get("moisture", 0.55))
        print(f"step {k:04d} | T={t:5.1f}°C  O2={o2:.2f}  M={m:.3f}  air={air_on_s:4.1f}s  [{reason}]")

        # Soft realtime feel (set to 0 for fast runs)
        if step_sleep_s:
            time.sleep(max(0.0, dt))
