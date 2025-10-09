# 60D_datagen_shapefit.py
# Generate a 60-day hourly dataset that follows the *shape* of ideal_inflated_data.csv
# - Keeps curing temperatures filled (no NaNs)
# - Adds aeration pulses and small oscillations/transients
# - Uses 'h' for hourly freq
# - OUTPUT now has 'hour' (0..1439) instead of 'day'
# - Active phase ENDS at 14 days (phase_cut = 14)

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict

# --- Paths & constants ---
# Use a safe Windows path (no backslash escapes); forward slashes also work on Windows.
REF_CSV = Path(r"LIDA-Composting-Project/ideal_inflated_data.csv")   # reference shapes (daily/sparse)
OUT_CSV = Path("ideal_dataset_hourly_60d_shapefit.csv")

TOTAL_DAYS = 60
TOTAL_HOURS = TOTAL_DAYS * 24
AMBIENT = 22.0

# Hard-set: active phase ends at day 14
PHASE_CUT_DAYS = 14

# ---------- helpers ----------
def _interp_shape_to_days(day_ref, y_ref, total_days=60):
    """Stretch/compress reference (day_ref, y_ref) to a 0..(total_days-1) day grid."""
    day_ref = np.asarray(day_ref, dtype=float)
    y_ref = np.asarray(y_ref, dtype=float)
    mask = np.isfinite(day_ref) & np.isfinite(y_ref)
    day_ref = day_ref[mask]; y_ref = y_ref[mask]
    if day_ref.size < 2:
        return None
    # shift to start at 0, then scale to target span
    day_ref = day_ref - day_ref.min()
    scale = (day_ref.max() - day_ref.min()) / max(total_days - 1, 1)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    x_target = np.arange(total_days, dtype=float) * scale
    return np.interp(x_target, day_ref, y_ref)

def _upsample_days_to_hours(y_daily):
    """Repeat each daily value 24x to make hourly; return length total_days*24."""
    return np.repeat(np.asarray(y_daily, dtype=float), 24)

def _ensure_bounds(arr, lo=None, hi=None):
    arr = np.asarray(arr, dtype=float)
    if lo is not None: arr = np.maximum(arr, lo)
    if hi is not None: arr = np.minimum(arr, hi)
    return arr

# ---------- main generator ----------
def generate_shape_fitted_dataset(ref_csv: Path = REF_CSV, out_csv: Path = OUT_CSV,
                                  total_days: int = TOTAL_DAYS, ambient: float = AMBIENT):
    assert ref_csv.exists(), f"Reference CSV not found: {ref_csv}"
    ref = pd.read_csv(ref_csv)

    # Ensure there's a Day axis for interpolation
    if "Day" not in ref.columns:
        ref = ref.reset_index().rename(columns={"index": "Day"})

    lower = {c.lower(): c for c in ref.columns}
    var_order = [
        "temperature_active1","temperature_active2","temperature_active3","temperature_active4",
        "temperature_curing1","temperature_curing2",
        "moisture_active1","moisture_active2",
        "moisture_curing1","moisture_curing2",
        "oxygen","co2","methane","methane_ppm"
    ]

    # Interpolate daily shapes from reference (stretched to total_days)
    daily_shapes: Dict[str, np.ndarray] = {}
    for v in var_order:
        col = lower.get(v)
        if col is None:
            token = v.split("_")[0]
            for c in ref.columns:
                if c.lower().startswith(token):
                    col = c; break
        if col is None:
            continue
        y = pd.to_numeric(ref[col], errors="coerce")
        x = pd.to_numeric(ref["Day"], errors="coerce")
        m = y.notna() & x.notna()
        if m.sum() >= 2:
            yN = _interp_shape_to_days(x[m].values, y[m].values, total_days=total_days)
            if yN is not None:
                daily_shapes[v] = yN

    # We FORCE the phase cut to 14 days (per your request)
    phase_cut = PHASE_CUT_DAYS
    days = np.arange(total_days, dtype=float)

    # ----------------------------------------------------------------------
    # ----- Active probes peak at phase change and COOL GRADUALLY (no cliff) -----
    # Rise: Gaussian centered on phase_cut
    # Fall: Continuous exponential decay from value at the cut (no jump)
    sigma_days = 3.0                         # width of the thermophilic peak (2.0–4.0 works well)
    peak_temp  = 60.0                        # °C at the peak
    baseline   = 35.0                        # °C baseline pre-rise & far tail
    tau_days   = 10.0                        # decay time constant (higher = slower cooling)
    floor_temp = max(AMBIENT + 2.0, 24.0)    # asymptotic floor for active probes

    # Gaussian bell centered at phase_cut
    def gauss_centered(x, mu, sigma):
        return np.exp(-((x - mu) ** 2) / (2 * sigma**2))

    rng_local = np.random.default_rng(123)
    phase_shift = 0.4                         # days; tiny horizontal shift between probes
    offsets = [0.0, +1.0, -0.8, +0.5]         # °C offsets per probe

    active_profiles = []
    for i, off in enumerate(offsets):
        # Slightly shift each probe so curves aren't identical
        mu = phase_cut + (i - 1.5) * phase_shift
        bell = baseline + (peak_temp - baseline) * gauss_centered(days, mu, sigma_days)

        # Continuous exponential cooling after the phase cut:
        # value_post(t) = floor + (value_at_cut - floor) * exp(-(t - phase_cut)/tau)
        post = days >= phase_cut
        # value right at the cut for this probe (continuous)
        value_at_cut = baseline + (peak_temp - baseline) * gauss_centered(phase_cut, mu, sigma_days)
        decay = np.exp(-(np.maximum(days - phase_cut, 0.0)) / tau_days)
        smooth_post = floor_temp + (value_at_cut - floor_temp) * decay  # starts at value_at_cut, decays to floor

        # stitch pre/post (no jump), add small noise & per-probe offset
        curve = np.where(post, smooth_post, bell) + off
        noise = rng_local.normal(0, 0.25, size=days.size)               # small daily noise
        active_profiles.append(curve + noise)

    # Assign (override if they already exist)
    daily_shapes["temperature_active1"] = active_profiles[0]
    daily_shapes["temperature_active2"] = active_profiles[1]
    daily_shapes["temperature_active3"] = active_profiles[2]
    daily_shapes["temperature_active4"] = active_profiles[3]
    # ----------------------------------------------------------------------

    # Fallback synthesis for any *other* variables missing from the reference
    def _synth_curing_temp(d):
        return np.maximum(ambient, 45 * np.exp(-np.maximum(d - phase_cut, 0) / 10.0))

    if "temperature_curing1" not in daily_shapes:
        daily_shapes["temperature_curing1"] = np.where(days < phase_cut, AMBIENT, _synth_curing_temp(days))
    if "temperature_curing2" not in daily_shapes:
        daily_shapes["temperature_curing2"] = daily_shapes["temperature_curing1"] - 0.8

    if "moisture_active1" not in daily_shapes:
        daily_shapes["moisture_active1"] = np.clip(60 - days * 0.5, 45, 62)
    if "moisture_active2" not in daily_shapes:
        daily_shapes["moisture_active2"] = np.clip(58 - days * 0.45, 44, 60)
    if "moisture_curing1" not in daily_shapes:
        daily_shapes["moisture_curing1"] = np.where(days < phase_cut, 50, 50 - (days - phase_cut) * 0.35)
    if "moisture_curing2" not in daily_shapes:
        daily_shapes["moisture_curing2"] = np.where(days < phase_cut, 48, 48 - (days - phase_cut) * 0.3)

    if "oxygen" not in daily_shapes:
        daily_shapes["oxygen"] = np.where(days < phase_cut, 15, 17)
    if "co2" not in daily_shapes:
        daily_shapes["co2"] = (
            15 * np.exp(-((days - min(2.5, 0.2 * phase_cut)) ** 2) / (2 * 1.5**2))
            + 8 * np.exp(-days / 12.0)
        )

    if "methane_ppm" not in daily_shapes:
        if "methane" in daily_shapes:
            daily_shapes["methane_ppm"] = daily_shapes["methane"]
        else:
            daily_shapes["methane_ppm"] = 20.0 + 1500 * np.exp(-((days - phase_cut) ** 2) / (2 * 0.5**2))

    # Bounds on daily shapes
    daily_shapes["oxygen"] = _ensure_bounds(daily_shapes["oxygen"], 8, 21)
    daily_shapes["co2"] = _ensure_bounds(daily_shapes["co2"], 0, 20)
    for k in ["moisture_active1","moisture_active2","moisture_curing1","moisture_curing2"]:
        daily_shapes[k] = _ensure_bounds(daily_shapes[k], 30, 70)

    # Build hourly frame (NOTE: using 'hour' instead of 'day')
    start = pd.Timestamp("2025-01-01 00:00:00")
    ts = pd.date_range(start, periods=TOTAL_HOURS, freq="h")
    hour_ix = np.arange(TOTAL_HOURS, dtype=int)     # 0..1439 for 60 days
    elapsed_days = hour_ix / 24.0
    phase_arr = np.where(elapsed_days < phase_cut, "active", "curing")

    hourly = pd.DataFrame({"timestamp": ts, "hour": hour_ix, "phase": phase_arr})

    # Fill hourly columns by repeating daily shape + small noise
    rng = np.random.default_rng(42)
    for k, y_daily in daily_shapes.items():
        yh = _upsample_days_to_hours(y_daily)
        amp = (np.nanmax(yh) - np.nanmin(yh)) if np.isfinite(yh).any() else 1.0
        noise = rng.normal(0, max(1e-6, amp * 0.005), size=TOTAL_HOURS)  # ~0.5% amplitude
        hourly[k] = yh + noise

    # Aeration pulses & gas transients
    aeration = np.zeros(TOTAL_HOURS, dtype=int)
    for i, t in enumerate(ts):
        if elapsed_days[i] < phase_cut:
            if t.hour % 6 == 0: aeration[i] = 1
        else:
            if t.hour % 12 == 0: aeration[i] = 1

    # Oxygen oscillations (+ aeration bump)
    o2_period_active, o2_period_curing = 1.5, 2.5
    osc = np.where(elapsed_days < phase_cut,
                   np.sin(2*np.pi*elapsed_days/o2_period_active),
                   np.sin(2*np.pi*elapsed_days/o2_period_curing))
    hourly["oxygen"] = _ensure_bounds(hourly["oxygen"].to_numpy() + 0.8*osc + aeration*0.8, 8, 21)

    # CO2 dips on aeration
    hourly["co2"] = _ensure_bounds(hourly["co2"].to_numpy() - aeration*0.6, 0, 20)
    hourly["aeration_on"] = aeration

    # Final bounds/rounding
    for c in hourly.columns:
        if c in ("timestamp","phase","hour","aeration_on"): 
            continue
        if "moisture" in c: hourly[c] = _ensure_bounds(hourly[c].to_numpy(), 30, 70)
        if c == "oxygen":  hourly[c] = _ensure_bounds(hourly[c].to_numpy(), 8, 21)
        if c == "co2":     hourly[c] = _ensure_bounds(hourly[c].to_numpy(), 0, 20)
        if "temperature" in c: hourly[c] = _ensure_bounds(hourly[c].to_numpy(), 10, 75)

    for c in hourly.columns:
        if pd.api.types.is_numeric_dtype(hourly[c]):
            hourly[c] = hourly[c].round(2)

    hourly.to_csv(out_csv, index=False)
    return hourly, phase_cut

if __name__ == "__main__":
    df, phase_cut = generate_shape_fitted_dataset()
    print(f"Saved {OUT_CSV} (active phase forced to day {phase_cut}, column 'hour' added).")
