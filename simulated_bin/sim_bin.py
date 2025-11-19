# simulated_bin/sim_bin.py
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

Phase = Literal["inactive", "warmup", "active", "curing"]

@dataclass
class BinSim:
    """Compost bin simulator with simple microbial heat dynamics.

    Key ideas:
      - 'activity' (0..1) grows when O2 >= 10% and moisture in [0.35, 0.60]
      - activity generates internal heat, strongest when T is below ~60°C
      - fan adds O2 (and dries/cools), lid concept omitted for simplicity

    ADDITIONS:
      - dt_s:   real-time seconds per step (default 60s), so 'steps' in the driver
                now map to a concrete time-base for reasoning and plots.
      - phase:  state machine with hysteresis (inactive → warmup → active → curing).
      - events: list of strings returned each step for phase transitions and mixes.
      - paddle mixing: temporary cooling + O2 bump + small drying for a few minutes.
      - tuned growth rates: k_up=0.025, k_down=0.006 as you noted in the header.
    """
    start_temp_active: float = 25.0
    start_temp_curing: float = 50.0
    seed: Optional[int] = None
    dt_s: float = 60.0  # seconds per step (simulation timestep)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

        # --- State variables (Active zone + one Curing probe kept simple)
        self.temp_active1 = float(self.start_temp_active)
        self.temp_curing1 = float(self.start_temp_curing)
        self.oxygen = 0.20       # start fairly oxygenated (20%)
        self.moisture = 0.50
        self.fan_level = 0.0
        self.activity = 0.20     # initial microbial activity (seed a bit more)

        # --- Actuator side-effects memory (mix effect persists for a short time)
        self._mix_cool_steps = 0  # countdown for transient cooling window
        self._mix_o2_steps = 0    # countdown for transient O2 bump window

        # --- Phase tracking + hysteresis
        self.phase: Phase = "inactive"
        self._ever_active = False  # for deciding curing eligibility
        self._last_phase: Phase = self.phase

        # ====================== Tunables (warmer + easier O2) ======================
        # Ambient and passive coupling
        self.ambient = 25.0
        self.loss_coeff = 0.002        # was 0.005 (slower loss to ambient)
        self.curing_loss_coeff = 0.001 # curing layer changes slower than active

        # Microbial heat parameters
        self.base_heat = 0.02          # was 0.00 (tiny baseline exotherm)
        self.heat_gain = 0.80          # was 0.30 (more microbial heat)
        self.cap_temp = 60.0           # heat effectiveness fades beyond this
        self.overcap_fade_span = 20.0  # fades to 0 by ~80°C

        # Fan and transport
        self.fan_cool_coeff = 0.03     # was 0.08 (fan cools less)
        self.o2_resp = 0.060           # controlls how the fans boost oxygen
        self.o2_leak = -0.0005         # o2 loss due to microbes consumption

        # Moisture dynamics
        self.moist_evap = 0.002        # was 0.004 (less drying while warming)
        self.moist_recover = 0.0010    # was 0.0006 (slightly quicker recover)

        # -------- Activity growth (your intended bump) --------
        # In _update_activity(): bump growth a hair (find these two lines below and set:)
        # k_up = 0.025    # was 0.015
        # k_down = 0.006  # was 0.008
        # We implement these as attributes to keep comments and make them obvious:
        self.k_up = 0.025
        self.k_down = 0.006

        # Paddle-mix transient parameters (effects last a few minutes)
        # (The real device would inject air and break compaction, creating short-term
        # heat loss + O2 spike; tune windows by duration in seconds.)
        self.mix_cool_dt = 10 * 60.0   # 10 minutes of extra cooling effect
        self.mix_o2_dt = 5 * 60.0      # 5 minutes of extra O2 bump
        self.mix_cool_dT = -0.10       # °C per step while cooling window active
        self.mix_o2_bump = +0.015      # absolute O2 bump per step while window active
        self.mix_dry = 0.002           # additional moisture loss per step during mixing

        # Noise magnitude (temperature)
        self.sigma_T = 0.03

    # -------------------------------------------------------------------------- #
    # External actuators (API preserved). lid_open is ignored here (kept for API)
    # paddle_mix triggers a short transient of extra cooling + O2 + drying.
    # -------------------------------------------------------------------------- #
    def set_actuators(self, fan_level: float, lid_open: bool, paddle_mix: bool):
        self.fan_level = float(np.clip(fan_level, 0.0, 1.0))
        if paddle_mix:
            # length (in steps) = duration_seconds / dt_s
            self._mix_cool_steps = int(round(self.mix_cool_dt / self.dt_s))
            self._mix_o2_steps = int(round(self.mix_o2_dt / self.dt_s))
            # instantaneous kick on the step when activated (optional)
            self.oxygen = float(np.clip(self.oxygen + 0.01, 0.07, 0.21))

    # -------------------------------------------------------------------------- #
    # Activity dynamics — “logistic-ish” with environmental gating + T preference
    # -------------------------------------------------------------------------- #
    def _update_activity(self):
        # Conditions for growth
        ok_o2 = self.oxygen >= 0.10
        ok_moist = 0.35 <= self.moisture <= 0.60
        cond = 1.0 if (ok_o2 and ok_moist) else 0.0

        # Temperature influence: microbes love ~40–65°C, weaker outside
        t = self.temp_active1
        if t < 30:
            # 20→30°C ramps to ~0.25
            t_factor = 0.25 * (t - 20.0) / 10.0
        elif t < 55:
            # 30→55°C ramps to 1.0
            t_factor = 0.25 + 0.75 * (t - 30.0) / 25.0
        elif t < 70:
            # 55→70°C drops to ~0.5
            t_factor = 1.0 - 0.5 * (t - 55.0) / 15.0
        else:
            t_factor = 0.2  # too hot

        # Growth/decay of activity (logistic-ish). Scaled by dt_s to keep rates sensible.
        a = self.activity
        k_up = self.k_up * (self.dt_s / 60.0)     # interpret base rates per minute
        k_down = self.k_down * (self.dt_s / 60.0)
        a += k_up * cond * t_factor * (1.0 - a) - k_down * (1.0 - cond) * a
        self.activity = float(np.clip(a, 0.0, 1.0))

    # -------------------------------------------------------------------------- #
    # Net microbial heat contribution (diminishes above ~60°C; fan reduces effect)
    # -------------------------------------------------------------------------- #
    def _microbial_heat(self) -> float:
        t = self.temp_active1
        if t <= self.cap_temp:
            temp_factor = 1.0
        else:
            temp_factor = max(0.0, 1.0 - (t - self.cap_temp) / self.overcap_fade_span)
        fan_factor = 1.0 - self.fan_level
        return self.base_heat + self.heat_gain * self.activity * temp_factor * fan_factor

    # -------------------------------------------------------------------------- #
    # Phase state machine with hysteresis (uses temperature + activity)
    # -------------------------------------------------------------------------- #
    def _update_phase(self, events: List[str]):
        prev = self.phase
        t = self.temp_active1
        a = self.activity

        # Thresholds + hysteresis bands
        # We keep it simple and interpretable:
        #  - warmup: t >= 30 OR a >= 0.3 (enter), leave if t < 27 AND a < 0.25
        #  - active: t >= 45 OR a >= 0.6 (enter), leave if t < 42 AND a < 0.5
        #  - curing: allowed only if "ever active", enter when t <= 40 AND a <= 0.3
        #            leave if t > 43 OR a > 0.45
        if self.phase == "inactive":
            if (t >= 30.0) or (a >= 0.30):
                self.phase = "warmup"
        elif self.phase == "warmup":
            if (t >= 45.0) or (a >= 0.60):
                self.phase = "active"
                self._ever_active = True
            elif (t < 27.0) and (a < 0.25):
                self.phase = "inactive"
        elif self.phase == "active":
            # can fall back to warmup if both T, a slip below narrower band
            if (t < 42.0) and (a < 0.50):
                self.phase = "warmup"
        # separate curing decision (only after ever being active)
        if self._ever_active:
            if self.phase in ("active", "warmup"):
                # enter curing when sufficiently cool + low activity
                if (t <= 40.0) and (a <= 0.30):
                    self.phase = "curing"
            elif self.phase == "curing":
                # leave curing if it heats/reactivates
                if (t > 43.0) or (a > 0.45):
                    self.phase = "warmup"

        if self.phase != prev:
            events.append(f"PHASE→{self.phase}")

    # -------------------------------------------------------------------------- #
    # Single simulation step
    # -------------------------------------------------------------------------- #
    def step(self) -> Dict:
        events: List[str] = []

        # --- Update activity based on current state ---
        self._update_activity()

        # --- Temperature dynamics (Active probe) ---
        # 1) passive loss to ambient
        dT_loss = -self.loss_coeff * (self.temp_active1 - self.ambient)
        # 2) microbial heat generation
        dT_heat = self._microbial_heat()
        # 3) extra fan cooling (forced convection)
        dT_fan = -self.fan_cool_coeff * self.fan_level
        # 4) paddle mixing transient cooling
        dT_mix = 0.0
        if self._mix_cool_steps > 0:
            dT_mix = self.mix_cool_dT
            self._mix_cool_steps -= 1
            if self._mix_cool_steps == 0:
                events.append("MIX→cool_end")
        # 5) small temperature noise
        noise = self.rng.normal(0.0, self.sigma_T)

        # Aggregate temp change (already scaled in coefficients to be per-step)
        self.temp_active1 += dT_loss + dT_heat + dT_fan + dT_mix + noise

        # --- Temperature dynamics (Curing probe, very simple drift toward ambient) ---
        dT_loss_curing = -self.curing_loss_coeff * (self.temp_curing1 - self.ambient)
        self.temp_curing1 += dT_loss_curing + 0.25 * dT_heat  # tiny cross-coupling

        # --- Oxygen dynamics ---
        # Fan pushes O2 up; without fan, O2 slowly drops due to consumption/leak
        self.oxygen += self.o2_resp * (self.fan_level - 0.5) + self.o2_leak * (1.0 - self.fan_level)
        # paddle mix transient O2 bump
        if self._mix_o2_steps > 0:
            self.oxygen += self.mix_o2_bump
            self._mix_o2_steps -= 1
            if self._mix_o2_steps == 0:
                events.append("MIX→o2_end")
        self.oxygen = float(np.clip(self.oxygen, 0.07, 0.21))

        # --- Moisture dynamics ---
        self.moisture -= self.moist_evap * self.fan_level
        if self._mix_cool_steps > 0:
            self.moisture -= self.mix_dry
        if self.fan_level < 0.2:
            # drift back up slowly toward ~0.55 when not ventilating much
            self.moisture += self.moist_recover * (0.55 - self.moisture)
        self.moisture = float(np.clip(self.moisture, 0.30, 0.60))

        # --- Phase + events ---
        self._update_phase(events)

        # Build frame (compatible + richer telemetry)
        return {
            "dt_s": float(self.dt_s),
            "temperature_active1": float(self.temp_active1),
            "temperature_curing1": float(self.temp_curing1),
            "oxygen": float(self.oxygen),
            "moisture": float(self.moisture),
            "fan_level": float(self.fan_level),
            "activity": float(self.activity),
            "phase": self.phase,
            "events": events,  # e.g. ["PHASE→warmup", "MIX→o2_end"]
        }
