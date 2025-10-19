# simulated_bin/cost.py
import numpy as np

def compute_cost(predicted_temps, actions, conf):
    """
    Cost = deviation from setpoint + fan energy + optional moisture term.
    """
    temp_error = np.mean((predicted_temps - conf.setpoint_c) ** 2)
    fan_energy = np.mean(np.array(actions) ** 2)
    return (conf.cost_temp_weight * temp_error +
            conf.cost_fan_weight * fan_energy)
