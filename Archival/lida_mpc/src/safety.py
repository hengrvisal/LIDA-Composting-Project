from dataclasses import dataclass

@dataclass
class SafetyStatus:
    ok: bool
    reason: str | None = None

def safety_check(frame) -> SafetyStatus:
    # frame = latest sensor reading (dict-like)
    if frame.get("oxygen", 0.21) < 0.10:
        return SafetyStatus(False, "O2 below 10% — forcing aeration ON")
    if frame.get("temperature_active1", 0) >= 80.0:
        return SafetyStatus(False, "Temp >= 80C — forcing aeration ON + mix if allowed")
    # add more: stale timestamps, door open, actuator cooldown, etc.
    return SafetyStatus(True, None)
