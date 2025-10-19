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

    # phase thresholds
    warmup_temp_c: float = 50.0     # enter ACTIVE when >= this
    curing_exit_c: float = 45.0     # drop below -> CURING (with hysteresis margin)
    phase_hyst_c: float = 2.0       # hysteresis margin

    # warm-up O2 controller (see controller_sim)
    warmup_o2_floor: float = 0.10
    warmup_o2_margin: float = 0.02  # target ~ 0.12
    warmup_min_fan: float = 0.55
    warmup_max_fan: float = 0.70
    warmup_kp: float = 30.0
    warmup_lid_open: bool = False
