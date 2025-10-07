# simulated_bin/sim_bin.py
import numpy as np
from dataclasses import dataclass
from typing import Dict

@dataclass
class BinSim:
    """Compost bin simulator with simple microbial heat dynamics.

    Key ideas:
      - 'activity' (0..1) grows when O2 >= 10% and moisture in [0.35, 0.60]
      - activity generates internal heat, strongest when T is below ~60°C
      - fan adds O2 (and dries/cools), lid concept omitted for simplicity
    """
    start_temp_active: float = 25.0
    start_temp_curing: float = 50.0
    seed: int | None = None

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.temp_active1 = self.start_temp_active
        self.temp_curing1 = self.start_temp_curing
        self.oxygen = 0.20      # start fairly oxygenated
        self.moisture = 0.50
        self.fan_level = 0.0
        self.activity = 0.10    # initial microbial activity

                # Tunables (warmer + easier O2)
        self.ambient = 25.0
        self.loss_coeff = 0.002        # was 0.005 (slower loss to ambient)
        self.base_heat = 0.02          # was 0.00 (tiny baseline exotherm)
        self.heat_gain = 0.80          # was 0.30 (more microbial heat)
        self.fan_cool_coeff = 0.03     # was 0.08 (fan cools less)
        self.o2_resp = 0.060           # was 0.025 (fan raises O2 faster)
        self.o2_leak = -0.0005         # was -0.002 (O2 decays slower)
        self.moist_evap = 0.002        # was 0.004 (less drying while warming)
        self.moist_recover = 0.0010    # was 0.0006 (slightly quicker recover)
        self.activity = 0.20           # was 0.10 (seed a bit more activity)

        # In _update_activity(): bump growth a hair (find these two lines below and set:)
        k_up = 0.025    # was 0.015
        k_down = 0.006  # was 0.008


    def set_actuators(self, fan_level: float, lid_open: bool, paddle_mix: bool):
        self.fan_level = float(np.clip(fan_level, 0.0, 1.0))
        # lid_open and paddle_mix ignored in this simplified sim, but kept for API

    def _update_activity(self):
        # Conditions for growth
        ok_o2 = self.oxygen >= 0.10
        ok_moist = 0.35 <= self.moisture <= 0.60
        cond = 1.0 if (ok_o2 and ok_moist) else 0.0

        # Temperature influence: microbes love ~40–65°C, weaker outside
        t = self.temp_active1
        if t < 30: t_factor = 0.25 * (t - 20) / 10.0           # 20→30°C ramps to ~0.25
        elif t < 55: t_factor = 0.25 + 0.75 * (t - 30) / 25.0  # 30→55°C ramps to 1.0
        elif t < 70: t_factor = 1.0 - 0.5 * (t - 55) / 15.0    # 55→70°C drops to ~0.5
        else: t_factor = 0.2                                    # too hot

        # Growth/decay of activity (logistic-ish)
        k_up = 0.015     # growth rate
        k_down = 0.008   # decay rate
        a = self.activity
        a += k_up * cond * t_factor * (1.0 - a) - k_down * (1.0 - cond) * a
        self.activity = float(np.clip(a, 0.0, 1.0))

    def _microbial_heat(self):
        # Diminish heat as temp exceeds ~60°C (prevent runaway)
        t = self.temp_active1
        cap = 60.0
        if t <= cap:
            temp_factor = 1.0
        else:
            temp_factor = max(0.0, 1.0 - (t - cap) / 20.0)  # fades to 0 by ~80°C
        # Fan reduces net heat effect (forced convection)
        fan_factor = 1.0 - self.fan_level
        return self.base_heat + self.heat_gain * self.activity * temp_factor * fan_factor

    def step(self) -> Dict:
        # --- Update activity based on current state ---
        self._update_activity()

        # --- Temperature dynamics ---
        # 1) passive loss to ambient
        dT_loss = -self.loss_coeff * (self.temp_active1 - self.ambient)
        # 2) microbial heat generation
        dT_heat = self._microbial_heat()
        # 3) extra fan cooling
        dT_fan = -self.fan_cool_coeff * self.fan_level
        # 4) small noise
        noise = self.rng.normal(0.0, 0.03)

        self.temp_active1 += dT_loss + dT_heat + dT_fan + noise

        # --- Oxygen dynamics ---
        # Fan pushes O2 up; without fan, O2 slowly drops due to consumption/leak
        self.oxygen += self.o2_resp * (self.fan_level - 0.5) + self.o2_leak * (1.0 - self.fan_level)
        self.oxygen = float(np.clip(self.oxygen, 0.07, 0.21))

        # --- Moisture dynamics ---
        self.moisture -= self.moist_evap * self.fan_level
        if self.fan_level < 0.2:
            self.moisture += self.moist_recover * (0.55 - self.moisture)  # drift back up slowly
        self.moisture = float(np.clip(self.moisture, 0.30, 0.60))

        return {
            "temperature_active1": float(self.temp_active1),
            "temperature_curing1": float(self.temp_curing1),
            "oxygen": float(self.oxygen),
            "moisture": float(self.moisture),
            "fan_level": float(self.fan_level),
            "activity": float(self.activity),
        }
