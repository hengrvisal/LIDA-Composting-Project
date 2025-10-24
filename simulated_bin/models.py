# simulated_bin/models.py
import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List

class Forecaster:
    """
    Wrapper around your trained ML forecaster.

    Auto-detects one of:
      - Tabular regressor:   forecaster.joblib  (expects DataFrame row)
      - LSTM sequence model: lstm_forecaster.keras + scaler_fore.joblib + feature_cols_fore.json

    LSTM expects last `window` rows and returns next-hour temperature prediction
    (extend targets as needed).
    """
    def __init__(self):
        self.dir = Path(__file__).parent
        self.tabular_path = self.dir / "forecaster.joblib"
        self.keras_path   = self.dir / "lstm_forecaster.keras"
        self.scaler_path  = self.dir / "scaler_fore.joblib"
        self.cols_path    = self.dir / "feature_cols_fore.json"
        self.window       = 24  # default; can be overridden by JSON if present

        self.mode = "fallback"
        self._tab = None
        self._scaler = None
        self._cols: Optional[List[str]] = None
        self._keras = None

        # Try Keras first
        try:
            import tensorflow as tf  # lazy import
            if self.keras_path.exists() and self.scaler_path.exists() and self.cols_path.exists():
                self._keras = tf.keras.models.load_model(self.keras_path)
                self._scaler = joblib.load(self.scaler_path)
                with open(self.cols_path, "r") as f:
                    meta = json.load(f)
                self._cols = meta["feature_cols"]
                self.window = int(meta.get("window", self.window))
                self.mode = "lstm"
                print(f"[MPC] Loaded LSTM forecaster ({self.keras_path.name}), window={self.window}")
        except Exception as e:
            print(f"[WARN] LSTM forecaster not available: {e}")

        # Else try tabular
        if self.mode != "lstm" and self.tabular_path.exists():
            try:
                self._tab = joblib.load(self.tabular_path)
                self.mode = "tabular"
                print(f"[MPC] Loaded tabular forecaster ({self.tabular_path.name})")
            except Exception as e:
                print(f"[WARN] Could not load tabular forecaster: {e}")

        if self.mode == "fallback":
            print("[WARN] No model files found — using dummy forecaster.")

    # ---- Public API used by controller/MPC ----
    def need_window(self) -> int:
        return self.window if self.mode == "lstm" else 1

    def feature_cols(self) -> Optional[List[str]]:
        return list(self._cols) if self._cols is not None else None

    def scaler(self):
        return self._scaler

    def predict(self, hist_df: pd.DataFrame) -> float:
        """
        Predict next-hour active temperature (extend as needed).
        """
        if self.mode == "lstm":
            from simulated_bin.features import build_feature_seq
            X = build_feature_seq(
                hist_df=hist_df,
                feature_cols=self._cols,
                window=self.window,
                scaler=self._scaler,
            )
            y = self._keras.predict(X, verbose=0)
            # If your LSTM is multi-target, map the index here; for now assume first is temp_active1
            return float(y[0][0])

        if self.mode == "tabular" and self._tab is not None:
            from simulated_bin.features import build_feature_row
            feats = build_feature_row(hist_df)
            return float(self._tab.predict(feats)[0])

        # Fallback dummy: small random drift
        last = float(hist_df.iloc[-1]["temperature_active1"])
        return last + float(np.random.uniform(-0.2, 0.2))


class PhaseModel:
    """
    Wrapper around your phase classifier.
      - XGBoost model:   xgb_phase.json (model), scaler_phase.joblib, feature_cols_phase.json, label_encoder.joblib
    Returns the phase string (e.g., 'active', 'curing', 'matured').
    """
    def __init__(self):
        self.dir = Path(__file__).parent
        self.model_path = self.dir / "xgb_phase.json"
        self.scaler_path = self.dir / "scaler_phase.joblib"
        self.cols_path = self.dir / "feature_cols_phase.json"
        self.le_path = self.dir / "label_encoder.joblib"

        self._xgb = None
        self._scaler = None
        self._cols = None
        self._le = None
        self.available = False

        try:
            import xgboost as xgb
            if all(p.exists() for p in [self.model_path, self.scaler_path, self.cols_path, self.le_path]):
                self._xgb = xgb.XGBClassifier()
                self._xgb.load_model(str(self.model_path))  # loads JSON
                self._scaler = joblib.load(self.scaler_path)
                with open(self.cols_path, "r") as f:
                    self._cols = json.load(f)["feature_cols"]
                self._le = joblib.load(self.le_path)
                self.available = True
                print("[MPC] Loaded phase model (XGBoost)")
            else:
                print("[WARN] Phase model files not found — phase will use heuristic.")
        except Exception as e:
            print(f"[WARN] Phase model not available: {e}")

    def predict(self, frame_or_df: pd.DataFrame) -> str:
        """
        Predict phase from a single-row DataFrame (latest state with engineered features).
        If not available, returns a heuristic based on temperature.
        """
        if not self.available:
            T = float(frame_or_df.iloc[-1]["temperature_active1"])
            return "ACTIVE" if T >= 55.0 else "CURING"

        # Expect the engineered columns that you trained on; select them in order
        x = frame_or_df.iloc[[-1]][self._cols]
        xs = self._scaler.transform(x)
        yhat = self._xgb.predict(xs)[0]
        label = self._le.inverse_transform([int(yhat)])[0]
        return str(label).upper()
