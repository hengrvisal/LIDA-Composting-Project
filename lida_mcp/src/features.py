"""
add_lags, scaling, slope
"""

import numpy as np
import pandas as pd

# ------- Feature engineering must match training pipeline -------
def add_lags(df: pd.DataFrame, cols, lags=(1,2,3)) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        for L in lags:
            df[f"{c}_lag{L}"] = df[c].shift(L)
    return df

def rolling_slope(x: pd.Series, w: int = 3) -> pd.Series:
    # simple slope: (x_t - x_{t-w}) / w ; robustify as needed
    return (x - x.shift(w)) / float(w)

def build_feature_row(history: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a small rolling window 'history' of the latest frames and returns a single-row
    dataframe with all features required by the forecaster.
    Assumes columns like: temperature_active1..4, temperature_curing1..2, moisture_*, gases, etc.
    """
    # Mirror your notebook’s recipe here
    cols_for_lags = [
        "temperature_active1","temperature_active2","temperature_active3","temperature_active4",
        "temperature_curing1","temperature_curing2"
    ]
    h = add_lags(history[cols_for_lags], lags=(1,2,3))
    # example slopes
    for c in cols_for_lags:
        h[f"{c}_slope3"] = rolling_slope(history[c], 3)
    # Keep only the latest row and fill NaNs conservatively (use your training fill rules)
    row = h.iloc[[-1]].fillna(method="ffill").fillna(method="bfill").fillna(0.0)
    return row

def apply_scaler(row_df, scaler):
    if scaler is None:
        return row_df.values
    return scaler.transform(row_df.values)
