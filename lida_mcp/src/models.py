import joblib
import numpy as np
import pandas as pd
from .config import Paths

class Forecaster:
    """
    Wraps your saved sklearn model (e.g., MultiOutputRegressor(RandomForestRegressor))
    and exposes predict_next() and roll_forward() utilities.
    """
    def __init__(self, model_path=Paths.forecaster_path, scaler_path=Paths.scaler_path):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path) if Paths.scaler_path.exists() else None

    def predict_next(self, feature_row_df):
        X = feature_row_df if isinstance(feature_row_df, np.ndarray) else feature_row_df.values
        return self.model.predict(X)[0]  # shape: (n_outputs,)

    def roll_forward(self, state_df, build_feature_fn, steps=6):
        """
        Repeatedly builds features from 'state_df' and applies the forecaster.
        This is a simple open-loop rollout; for better fidelity, inject action effects into state_df each step.
        """
        preds = []
        tmp_state = state_df.copy()
        for _ in range(steps):
            feats = build_feature_fn(tmp_state)
            y = self.model.predict(feats.values)[0]
            preds.append(y)
            # Naive state update: write predictions back into next row (for temperatures)
            # Customize to your column order and output mapping
            tmp_row = tmp_state.iloc[[-1]].copy()
            # Example: assume outputs map to temperatures in order (adapt!):
            out_cols = [
                "temperature_active1","temperature_active2","temperature_active3","temperature_active4",
                "temperature_curing1","temperature_curing2"
            ]
            for i, c in enumerate(out_cols[:len(y)]):
                tmp_row[c] = y[i]
            tmp_state = pd.concat([tmp_state, tmp_row], ignore_index=True)
        return np.array(preds)

def load_phase_classifier():
    try:
        return joblib.load(Paths.phase_clf_path)
    except Exception:
        return None
