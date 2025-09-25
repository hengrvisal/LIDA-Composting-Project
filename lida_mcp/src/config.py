from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    model_dir: Path = Path("./models")
    scaler_path: Path = model_dir / "feature_scaler.joblib"
    forecaster_path: Path = model_dir / "forecaster.joblib"         #sklearn or compatible
    phase_clf_path: Path = model_dir / "phase.joblib"          #optional
    
@dataclass(frozen=True)
class MQTTConf:
    host: str = "localhost"
    port: int = 1833
    topic_cmd: str = "lida/actuators/cmd"
    topic_ack: str = "lida/actuators/ack"
    topic_sensors: str = "lida/sensors/raw"     #incoming sensors frames (JSON)
    
@dataclass(frozen=True)
class MPCConf:
    horizon_steps: int = 6           # 6 x 10min = 1h (see controller step period)
    action_levels = (0, 1)           # aeration OFF/ON; extend to (0, 0.5, 1) if needed
    step_minutes: int = 10           # controller period
    temp_hi: float = 62.0            # preferred max
    temp_lo: float = 55.0            # preferred min
    temp_hard_max: float = 80.0      # hard safety cutoff
    o2_floor: float = 0.10           # 10%
    energy_weight: float = 0.10
    over_weight: float = 1.00
    under_weight: float = 0.50