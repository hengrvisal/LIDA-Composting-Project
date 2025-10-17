import numpy as np
from .config import MPCConf

def penalty_temperature(temp, conf: MPCConf):
    over = max(0.0, temp - conf.temp_hi)
    under = max(0.0, conf.temp_lo - temp)
    return conf.over_weight * (over**2) + conf.under_weight * (under**2)

def energy_from_action(action_level: float) -> float:
    # Simple proxy: ON=1 costs 1 unit per step; scale for duty levels
    return float(action_level)

def sequence_cost(temp_traj, action_seq, conf: MPCConf):
    """
    temp_traj: list/array of scalar temps for the controlled probe (e.g., active1)
    action_seq: same length as traj; values in {0,1} or duty fractions
    """
    J = 0.0
    for t, temp in enumerate(temp_traj):
        J += penalty_temperature(temp, conf)
        J += conf.energy_weight * energy_from_action(action_seq[t])
    return J

def hard_constraint_violation(temp, o2, conf: MPCConf) -> bool:
    if temp >= conf.temp_hard_max:
        return True
    if o2 < conf.o2_floor:
        return True
    return False
