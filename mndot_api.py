import os
import time
import requests
import pandas as pd
import numpy as np

# Base URL for the real upstream service (configure via environment when deployed).
# If empty, the module will generate mock data so the UI can run end-to-end.
MN_API_BASE = os.getenv("MNDOT_API_BASE", "").rstrip("/")
TIMEOUT = 15  # seconds


# -----------------------------
# Simple in-memory TTL cache
# -----------------------------
def cache_ttl(ttl=30):
    """
    A very small in-memory cache decorator with a time-to-live.
    It helps avoid hammering the upstream API when the user tweaks the UI.
    Not meant for production-grade persistence or concurrency.
    """
    def deco(fn):
        store = {}

        def wrap(*args, **kwargs):
            key = (fn.__name__, args, tuple(sorted(kwargs.items())))
            now = time.time()
            if key in store and now - store[key][0] < ttl:
                return store[key][1]
            val = fn(*args, **kwargs)
            store[key] = (now, val)
            return val

        return wrap
    return deco


@cache_ttl(ttl=300)
def load_detector_list(csv_path="data/detectors_sample.csv"):
    """
    Load the detector metadata list (route, direction, detector_id, lat/lon, etc.).
    You can replace the CSV with an official list later without touching the UI code.
    """
    df = pd.read_csv(csv_path)
    df["detector_id"] = df["detector_id"].astype(str).str.strip()
    return df


@cache_ttl(ttl=30)
def fetch_timeseries(detector_id: str, date_str: str, start_hm: str, end_hm: str, sensor_type: str = "V30") -> pd.DataFrame:
    """
    Fetch a 30-second time series for a given detector and time window.

    Parameters
    ----------
    detector_id : str
        The detector identifier (as listed in your metadata).
    date_str : str
        Date string in YYYY-MM-DD format (local date).
    start_hm : str
        Start time "HH:MM" (local).
    end_hm : str
        End time "HH:MM" (local).
    sensor_type : str
        One of ["V30", "C30", "S30"] for volume/occupancy/speed respectively.

    Returns
    -------
    DataFrame with columns:
        - ts: naive pandas datetime (converted to local America/Chicago and tz-removed)
        - value: numeric values
    """
    # If the upstream base URL is not configured, return a realistic mock series.
    # This keeps the app usable in local development or before the API is ready.
    if not MN_API_BASE:
        # Build a 30-second index on [start, end) (lower-inclusive, upper-exclusive).
        idx = pd.date_range(
            f"{date_str} {start_hm}",
            f"{date_str} {end_hm}",
            freq="30s",                 # NOTE: lowercase 's' avoids the pandas deprecation warning
            inclusive="left"
        )
        if len(idx) == 0:
            return pd.DataFrame(columns=["ts", "value"])

        # Create a simple diurnal shape for different metrics using numpy,
        # so later slice-assignments are safe (numpy arrays are mutable).
        t = idx.hour.to_numpy() + idx.minute.to_numpy() / 60.0 + idx.second.to_numpy() / 3600.0

        if sensor_type == "V30":           # per-30s volume
            peak, amp, noise = 8.0, 900.0, 18.0
            base = amp * np.exp(-((t - peak) ** 2) / 4.0)
            val = base + np.random.normal(0, noise, size=len(idx))
            val = np.maximum(val, 0.0)
        elif sensor_type == "C30":         # occupancy in %
            peak, amp, noise = 8.5, 22.0, 2.0
            base = amp * np.exp(-((t - peak) ** 2) / 4.5) + 4.0
            val = base + np.random.normal(0, noise, size=len(idx))
            val = np.clip(val, 0.0, 100.0)
        elif sensor_type == "S30":         # speed mph
            base = 60.0 - 15.0 * np.exp(-((t - 8.0) ** 2) / 2.0)
            val = base + np.random.normal(0, 1.8, size=len(idx))
            val = np.clip(val, 0.0, 85.0)
        else:
            # Unknown metric: return zeros but keep the time axis.
            val = np.zeros(len(idx), dtype=float)

        # Ensure 'val' is a mutable numpy array before injecting a zero streak.
        val = np.asarray(val, dtype=float).copy()

        # Inject a small zero streak (e.g., 5 minutes = 10 points) to demo rule checks.
        if len(val) >= 30:
            val[20:30] = 0.0

        return pd.DataFrame({"ts": idx, "value": val})

    # ---- Real upstream path (replace with your actual endpoint contract) ----
    # Example 1 (query style):
    #   .../timeseries?date=YYYY-MM-DD&detector=5838&type=V30&start=07:00&end=09:00&step=30
    url = (
        f"{MN_API_BASE}/timeseries"
        f"?date={date_str}&detector={detector_id}&type={sensor_type}"
        f"&start={start_hm}&end={end_hm}&step=30"
    )

    # Example 2 (path style):
    #   .../{date}/{area}/{detector_id}/{sensor_type}?start=...&end=...&step=30
    # area = "I-94-EB"
    # url = f"{MN_API_BASE}/{date_str}/{area}/{detector_id}/{sensor_type}?start={start_hm}&end={end_hm}&step=30"

    r = requests.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    js = r.json()

    # Map upstream JSON into a uniform DataFrame with ['timestamp', 'value'].
    df = normalize_timeseries_json(js)

    # Parse to timezone-aware UTC, convert to America/Chicago, then drop tz to keep the app simple.
    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = pd.DataFrame(
        {
            "ts": ts.dt.tz_convert("America/Chicago").dt.tz_localize(None),
            "value": pd.to_numeric(df["value"], errors="coerce"),
        }
    ).dropna()

    return df.sort_values("ts").reset_index(drop=True)


def normalize_timeseries_json(js):
    """
    Normalize upstream JSON into a DataFrame with exactly two columns:
    ['timestamp', 'value'].

    If your upstream uses different field names, add the mapping here.
    This function centralizes the contract so UI code does not need to change.
    """
    df = pd.DataFrame(js)

    # Accept common variants and rename them to the canonical names.
    if "timestamp" not in df.columns and "ts" in df.columns:
        df = df.rename(columns={"ts": "timestamp"})
    if "value" not in df.columns and "val" in df.columns:
        df = df.rename(columns={"val": "value"})

    # Final contract check
    if "timestamp" not in df.columns or "value" not in df.columns:
        raise ValueError(
            "Upstream payload is missing 'timestamp' or 'value'. "
            "Please add field mapping in normalize_timeseries_json()."
        )

    return df[["timestamp", "value"]]


def rule_flags(df: pd.DataFrame, flat_k: int = 10):
    """
    Basic rule checks on a 30-second series:
      1) Any negative values.
      2) Any flatline of length >= flat_k (std very close to zero).
      3) Any streak of zeros of length >= flat_k.

    Parameters
    ----------
    df : DataFrame
        Must contain a numeric 'value' column.
    flat_k : int
        Window length for flatline and zero-streak checks.

    Returns
    -------
    dict with boolean flags.
    """
    if df.empty:
        return {"empty": True}

    v = pd.to_numeric(df["value"], errors="coerce").to_numpy()
    out = {}

    # 1) Negative values
    out["negative_any"] = bool((v < 0).any())

    # 2) Flatline: rolling std within a window nearly zero
    if len(v) >= flat_k:
        rolling_std = pd.Series(v).rolling(flat_k).std().to_numpy()
        out["flatline_any"] = bool((np.nan_to_num(rolling_std, nan=0.0) < 1e-3).any())
    else:
        out["flatline_any"] = False

    # 3) Zero-value streak
    run_zero = 0
    zero_flag = False
    for x in v:
        if abs(x) < 1e-9:
            run_zero += 1
            if run_zero >= flat_k:
                zero_flag = True
                break
        else:
            run_zero = 0
    out["zero_streak"] = zero_flag

    return out
