# simulated_bin/models.py
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

class Forecaster:
    """
    Wrapper around your trained ML model.
    Expects 'forecaster.joblib' to be in the same directory.
    """
    def __init__(self):
        model_path = Path(__file__).with_name("forecaster.joblib")
        if model_path.exists():
            self.model = joblib.load(model_path)
            print(f"[MPC] Loaded model from {model_path.name}")
        else:
            print("[WARN] No forecaster.joblib found — using dummy forecaster.")
            self.model = None

    def predict(self, features: pd.DataFrame) -> float:
        if self.model:
            return float(self.model.predict(features)[0])
        # Fallback dummy prediction: add random drift
        return features["temperature_active1"].iloc[0] + np.random.uniform(-0.2, 0.2)
