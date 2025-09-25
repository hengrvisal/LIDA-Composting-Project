# main.py (example stub)
import time, json
from .controller import run_controller
from .config import MPCConf

def fake_sensor_stream():
    import random
    import datetime as dt
    temp = 58.0
    while True:
        temp += random.uniform(-0.2, 0.3)
        yield {
            "time_stamp": dt.datetime.utcnow().isoformat(),
            "temperature_active1": temp,
            "temperature_active2": temp-0.5,
            "temperature_active3": temp+0.6,
            "temperature_active4": temp-0.2,
            "temperature_curing1": temp-6,
            "temperature_curing2": temp-5,
            "oxygen": 0.21,
            "co2": 0.03,
            "methane": 0.003
        }

if __name__ == "__main__":
    run_controller(fake_sensor_stream(), MPCConf())
