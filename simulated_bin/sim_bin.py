from __future__ import annotations
from dataclasses import dataclass
import random
import time
from typing import Dict

AMBIENT_TEMP = 25.0
DT_MIN = 10  # minutes per step (aligns with MPCConf.step_minutes)

@dataclass
class Actuators:
    fan_level: float = 0.0      # 0..1 (aeration duty)
    lid_open: bool = False      # lid state via linear actuator
    paddle_mix: bool = False    # one-shot “mix” pulse this step

class BinSim:
    """
    Simplified compost bin dynamics with 3 levels:
      1) Active (hot): 4 probes (to mirror your model I/O)
      2) Curing: 2 probes
      3) Mature: implicit (we keep it near ambient internally)

    Actuators: fan (aeration), lid (linear actuator), paddle (motorized)
    Sensors returned each step:
      - temperature_active1..4, temperature_curing1..2, oxygen, co2, methane
      - lid_open, fan_level (telemetry)
    """
    def __init__(self,
                 start_temp_active: float = 58.0,
                 start_temp_curing: float = 50.0,
                 start_o2: float = 0.21,
                 seed: int | None = None):
        if seed is not None:
            random.seed(seed)

        # Internal “state”
        self.act = Actuators()
        self.time_s = 0.0

        # Temperatures (start around thermophilic)
        self.ta = [start_temp_active + d for d in (0.0, -0.5, +0.6, -0.2)]
        self.tc = [start_temp_curing - 1.0, start_temp_curing]
        self.tmature = AMBIENT_TEMP + 2.0

        # Gases
        self.o2 = start_o2
        self.co2 = 0.03
        self.ch4 = 0.003

        # process knobs (feel free to tune)
        self.exo_heat_gain = 0.18       # base exothermic gain per 10min in Active
        self.curing_gain = 0.06         # exothermic in curing
        self.cool_per_fan = 0.55        # cooling per 10min at fan_level=1 (Active only)
        self.mix_cool_boost = 0.6       # extra cooling when paddle mixes (one step)
        self.lid_cool_bonus = 0.25      # extra passive cooling when lid is open
        self.coupling = 0.06            # heat bleeding toward neighbors / ambient
        self.o2_recovery = 0.02         # O2 rises per 10min at fan_level=1
        self.o2_consumption = 0.005     # O2 drops per step from biology (rough)
        self.noise = 0.15               # random noise scale

    def set_actuators(self, fan_level: float | None = None, lid_open: bool | None = None, paddle_mix: bool | None = None):
        if fan_level is not None:
            self.act.fan_level = max(0.0, min(1.0, float(fan_level)))
        if lid_open is not None:
            self.act.lid_open = bool(lid_open)
        if paddle_mix is not None:
            self.act.paddle_mix = bool(paddle_mix)

    def _cooling_term(self) -> float:
        base = self.cool_per_fan * self.act.fan_level
        if self.act.paddle_mix:
            base += self.mix_cool_boost
        if self.act.lid_open:
            base += self.lid_cool_bonus
        return base

    def _step_temperatures(self):
        cool = self._cooling_term()

        # Active: exothermic up, cooling down, plus coupling to ambient
        for i in range(len(self.ta)):
            d_exo = self.exo_heat_gain
            d_cool = cool
            d_couple = self.coupling * (AMBIENT_TEMP - self.ta[i])
            self.ta[i] += d_exo - d_cool + d_couple + random.uniform(-self.noise, self.noise)

        # Curing follows active with lower gain & coupling
        avg_active = sum(self.ta) / len(self.ta)
        for i in range(len(self.tc)):
            target = 0.6 * avg_active + 0.4 * AMBIENT_TEMP
            d_exo = self.curing_gain
            d_couple = 0.5 * self.coupling * (target - self.tc[i])
            self.tc[i] += d_exo + d_couple + random.uniform(-self.noise*0.8, self.noise*0.8)

        # Mature drifts to ambient gently
        self.tmature += 0.2 * self.coupling * (AMBIENT_TEMP - self.tmature)

        # Reset one-shot paddle action
        self.act.paddle_mix = False

    def _step_gases(self):
        # O2: consumption vs aeration recovery
        self.o2 -= self.o2_consumption
        self.o2 += self.o2_recovery * self.act.fan_level
        self.o2 = max(0.05, min(0.21, self.o2))

        # CO2/CH4: very rough relationships
        self.co2 = 0.025 + (0.65 * (0.21 - self.o2))   # more CO2 when O2 lower
        self.ch4 = max(0.000, 0.002 + (0.21 - self.o2) * 0.01)

    def step(self) -> Dict[str, float]:
        """Advance simulation by DT_MIN and return a sensor frame."""
        self._step_temperatures()
        self._step_gases()
        self.time_s += DT_MIN * 60

        return {
            "time_stamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.time_s)),
            "temperature_active1": self.ta[0],
            "temperature_active2": self.ta[1],
            "temperature_active3": self.ta[2],
            "temperature_active4": self.ta[3],
            "temperature_curing1": self.tc[0],
            "temperature_curing2": self.tc[1],
            "oxygen": self.o2,
            "co2": self.co2,
            "methane": self.ch4,
            "lid_open": float(self.act.lid_open),
            "fan_level": float(self.act.fan_level),
        }
