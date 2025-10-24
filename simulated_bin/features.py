# simulated_bin/features.py
import pandas as pd
import numpy as np
from typing import List, Optional

def build_feature_row(hist_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts features from history for the ML forecaster.
    """
    row = hist_df.iloc[-1]
    feats = {
        "temperature_active1": row["temperature_active1"],
        "oxygen": row["oxygen"],
        "moisture": row["moisture"],
        "fan_level": row.get("fan_level", 0),
    }
    return pd.DataFrame([feats])

# NEW: sequence builder for LSTM forecaster
def build_feature_seq(
        hist_df: pd.DataFrame,
        feature_cols: List[str],
        window: int,
        scaler=None,
) -> np.ndarray:
    """
    returns a (1, window, n_features) array for an LSTM:
        - takes the last 'window' rows from hist_df,
        - selects 'feature_cols' in order,
        - applies 'scaler.transform' if provided
    Raises if not enough history
    """
    if len(hist_df) < window:
        raise ValueError(f"Need at least {window} rows of history for LSTM")
    window_df = hist_df.iloc[-window:][feature_cols].copy()
    X = window_df.values.astype(float)
    if scaler is not None:
        X = scaler.transform(window_df)
    return X[np.newaxis, ...] # (1, window, n_features)