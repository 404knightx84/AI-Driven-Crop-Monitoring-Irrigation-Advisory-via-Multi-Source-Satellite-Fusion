"""
utils/timeseries.py
NDVI / EVI time-series preprocessing utilities:
  - Savitzky-Golay smoothing
  - Double logistic phenology curve fitting
  - Season extraction (SOS, EOS, peak)
  - Gap-fill via linear interpolation
"""

from typing import Tuple, Optional, List

import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from loguru import logger


# ── Smoothing ─────────────────────────────────────────────────────────────────

def savitzky_golay(
    ts: np.ndarray,
    window: int = 7,
    polyorder: int = 3,
) -> np.ndarray:
    """
    Apply Savitzky-Golay filter to a 1-D or 2-D time series.
    ts: (T,) or (N, T)
    """
    if ts.ndim == 1:
        return savgol_filter(ts, window_length=window, polyorder=polyorder).astype(np.float32)
    return np.apply_along_axis(
        lambda x: savgol_filter(x, window_length=window, polyorder=polyorder),
        axis=1, arr=ts,
    ).astype(np.float32)


def gap_fill(ts: np.ndarray, fill_value: float = np.nan) -> np.ndarray:
    """
    Linear interpolation gap-fill for NaN / masked values.
    ts: (T,) — 1-D time series.
    """
    out = ts.copy()
    nans = np.isnan(out)
    if nans.any():
        x = np.arange(len(out))
        out[nans] = np.interp(x[nans], x[~nans], out[~nans])
    return out.astype(np.float32)


# ── Phenology curve fitting ───────────────────────────────────────────────────

def _double_logistic(t: np.ndarray, mn: float, mx: float,
                     t1: float, k1: float, t2: float, k2: float) -> np.ndarray:
    """Double logistic model (Eklundh & Jönsson) for NDVI seasonal curves."""
    return mn + (mx - mn) * (
        1 / (1 + np.exp(-k1 * (t - t1))) -
        1 / (1 + np.exp(-k2 * (t - t2)))
    )


def fit_double_logistic(
    ts: np.ndarray,
    doys: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Fit a double logistic curve to a 1-D NDVI time series.
    doys: day-of-year array (T,); if None, uses integer indices.
    Returns (fitted_curve, params_dict).
    """
    T = len(ts)
    if doys is None:
        doys = np.arange(T, dtype=np.float32)

    mn_init = float(ts.min())
    mx_init = float(ts.max())
    mid = T // 2

    p0 = [mn_init, mx_init, doys[mid // 2], 0.1, doys[mid + mid // 2], 0.1]
    bounds = (
        [0, 0, doys[0],   0.01, doys[0],   0.01],
        [1, 1, doys[-1],  2.0,  doys[-1],  2.0],
    )

    try:
        popt, _ = curve_fit(_double_logistic, doys, ts, p0=p0, bounds=bounds, maxfev=5000)
        fitted = _double_logistic(doys, *popt)
        params = dict(zip(["mn", "mx", "t1", "k1", "t2", "k2"], popt))
        return fitted.astype(np.float32), params
    except RuntimeError:
        logger.warning("[TS] Double logistic fit failed — returning raw series.")
        return ts.astype(np.float32), {}


# ── Phenological metrics ──────────────────────────────────────────────────────

def extract_phenometrics(
    ts: np.ndarray,
    doys: Optional[np.ndarray] = None,
    sos_threshold: float = 0.2,
    eos_threshold: float = 0.2,
) -> dict:
    """
    Extract start of season (SOS), end of season (EOS), peak, and amplitude
    from a 1-D NDVI curve.

    SOS/EOS defined as the point where NDVI crosses:
      baseline + threshold * (peak - baseline)
    """
    if doys is None:
        doys = np.arange(len(ts), dtype=np.float32)

    baseline  = float(np.percentile(ts, 10))
    peak_val  = float(ts.max())
    peak_idx  = int(ts.argmax())
    amplitude = peak_val - baseline
    sos_val   = baseline + sos_threshold * amplitude
    eos_val   = baseline + eos_threshold * amplitude

    # SOS: first crossing above sos_val before peak
    sos_idx = next((i for i in range(peak_idx) if ts[i] >= sos_val), 0)
    # EOS: last crossing below eos_val after peak
    eos_idx = next((i for i in range(len(ts) - 1, peak_idx, -1) if ts[i] >= eos_val), len(ts) - 1)

    return {
        "sos_doy":   float(doys[sos_idx]),
        "eos_doy":   float(doys[eos_idx]),
        "peak_doy":  float(doys[peak_idx]),
        "peak_ndvi": peak_val,
        "amplitude": amplitude,
        "season_length_days": float(doys[eos_idx] - doys[sos_idx]),
    }


def batch_phenometrics(
    ts_array: np.ndarray,   # (N, T)
    doys: Optional[np.ndarray] = None,
) -> List[dict]:
    """Extract phenometrics for N pixels."""
    return [extract_phenometrics(ts_array[i], doys) for i in range(ts_array.shape[0])]
