from dataclasses import dataclass

@dataclass(frozen=True)
class MPCConf:
    setpoint_c: float = 55.0
    temp_hi: float = 56.5
    temp_lo: float = 53.5
    o2_floor: float = 0.10
    moisture_min: float = 0.35
    horizon_steps: int = 6
    cost_temp_weight: float = 1.0
    cost_fan_weight: float = 0.2
    cost_moisture_weight: float = 0.5

    # Warm-up gating
    warmup_lockout: bool = True
    warmup_temp_c: float = 50.0

    # Warm-up O2 keeper (IMPORTANT: min fan must be > 0.5 in YOUR model)
    warmup_o2_floor: float = 0.10
    warmup_o2_margin: float = 0.02   # was 0.01 → target = 0.12
    warmup_min_fan: float = 0.55     # ≥ 0.5 so O2 actually rises in your sim
    warmup_max_fan: float = 0.70     # cap to avoid over-cooling
    warmup_kp: float = 30.0          # strong push when below target
    warmup_lid_open: bool = False
    warmup_temp_c: float = 50.0
