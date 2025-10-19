# src/config.py
from dataclasses import dataclass

@dataclass(frozen=True)
class MPCConf:
    # Control cadence
    step_seconds: int = 10          # controller dt (s)
    horizon_steps: int = 6          # not used in this simple MPC, kept for future
    # Temperature targets
    temp_lo: float = 55.0           # preferred minimum (°C)
    temp_hi: float = 62.0           # preferred maximum (°C)
    temp_hard_max: float = 80.0     # hard safety cutoff (°C)
    # Oxygen limits
    o2_floor: float = 0.10          # 10%
    # Cost weights (used by cost.py helpers if you extend)
    energy_weight: float = 0.10
    over_weight: float = 1.00
    under_weight: float = 0.50

    # Simple MPC tuning for simulated bin
    deadband_c: float = 1.5
    k_on_per_deg: float = 6.0       # sec of aeration per °C above setpoint per step
    min_on_sec: float = 2.0
    max_on_sec: float = 20.0
    setpoint_c: float = 55.0        # target compost temp (°C)

    # Sim bin dynamics
    cooldown_bias: float = 0.06     # cooling effectiveness per second of aeration
    heat_gain_bias: float = 0.015   # natural microbial heating per second
    moisture_evap_per_sec: float = 0.0006
    moisture_recover_per_sec: float = 0.0001
    moisture_min: float = 0.35
