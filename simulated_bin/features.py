# simulated_bin/features.py
import pandas as pd
import numpy as np

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
