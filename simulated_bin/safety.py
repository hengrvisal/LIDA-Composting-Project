# simulated_bin/safety.py
from dataclasses import dataclass
from typing import Dict

@dataclass
class SafetyStatus:
    ok: bool
    reason: str = ""

def safety_check(frame: Dict) -> SafetyStatus:
    """Basic safety rule: ensure T < 80°C and O2 > 10%"""
    t = float(frame.get("temperature_active1", 0))
    o2 = float(frame.get("oxygen", 0.21))
    ok = (t < 80.0) and (o2 >= 0.1)
    reason = "" if ok else f"Safety trip: T={t:.1f}°C, O2={o2:.2f}"
    return SafetyStatus(ok, reason)
