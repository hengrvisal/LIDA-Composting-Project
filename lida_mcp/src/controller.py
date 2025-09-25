import time
import pandas as pd
from datetime import datetime, timedelta

from .config import MPCConf
from .features import build_feature_row
from .models import Forecaster
from .mqtt_client import MQTTClient
from .mpc import choose_action
from .safety import safety_check

# --- Simple “state effect” for actions (domain heuristic) ---
def action_to_state_effect(state_df: pd.DataFrame, action_level: float) -> pd.DataFrame:
    """
    Inject a small immediate effect of aeration on the latest row (cooling / O2).
    Tune these numbers to your system or learn them from data.
    """
    cooling = {0.0: 0.0, 1.0: -0.4}  # degC per control step (10min) — tune!
    o2_bump = {0.0: 0.00, 1.0: 0.02} # +2% O2 proxy — if you simulate O2
    latest = state_df.iloc[[-1]].copy()

    for c in ["temperature_active1","temperature_active2","temperature_active3","temperature_active4"]:
        if c in latest.columns:
            latest[c] = latest[c].astype(float) + cooling.get(float(action_level), 0.0)

    if "oxygen" in latest.columns:
        latest["oxygen"] = float(latest["oxygen"]) + o2_bump.get(float(action_level), 0.0)

    return pd.concat([state_df, latest], ignore_index=True)

def command_payload(aeration_on: bool, seconds: int):
    return {"device": "aeration", "state": "on" if aeration_on else "off", "seconds": int(seconds)}

def run_controller(
    sensor_stream,       # iterator yielding dicts: latest sensor frame
    conf: MPCConf = MPCConf(),
    step_sleep_s: int = 60,  # run every minute; actions are 10-min, but we can re-evaluate
):
    forecaster = Forecaster()
    mqttc = MQTTClient()

    history = pd.DataFrame()

    while True:
        frame = next(sensor_stream)   # latest sensors
        ts = frame.get("time_stamp", datetime.utcnow().isoformat())
        history = pd.concat([history, pd.DataFrame([frame])], ignore_index=True).tail(120)

        # Safety overrides first
        safe = safety_check(frame)
        if not safe.ok:
            mqttc.publish_cmd(command_payload(True, conf.step_minutes * 60))
            print(f"[{ts}] SAFETY: {safe.reason} → Aeration ON")
            time.sleep(step_sleep_s); mqttc.loop(); continue

        # Build features and choose action via MPC
        try:
            action = choose_action(
                state_hist=history,
                forecaster=forecaster,
                build_feature_fn=build_feature_row,
                action_to_state_effect=action_to_state_effect,
                conf=conf
            )
        except Exception as e:
            # Fail-safe if model/feature construction fails
            print(f"[{ts}] MPC error: {e} — defaulting to safe aeration")
            action = 1

        # Translate action to command; here action in {0,1}
        seconds = conf.step_minutes * 60 if action == 1 else 0
        mqttc.publish_cmd(command_payload(action == 1, seconds))
        print(f"[{ts}] MPC action: {'ON' if action else 'OFF'} for {seconds}s")

        mqttc.loop()
        time.sleep(step_sleep_s)
