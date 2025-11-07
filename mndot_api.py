import os
import json
import time
from datetime import datetime, timedelta, date
from pathlib import Path

import requests
import pandas as pd
import numpy as np

# Base URL for the real upstream service (configure via environment when deployed).
# If empty, the module will generate mock data so the UI can run end-to-end.
MN_API_BASE = os.getenv("MNDOT_API_BASE", "").rstrip("/")
LOCAL_DATA_ROOT = Path(os.getenv("MNDOT_LOCAL_DATA", "data/mndot_raw")).expanduser()
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
def load_detector_list(csv_path="data/I-94_detectors_converted.csv"):
    """
    Load the detector metadata list (route, direction, detector_id, lat/lon, etc.).
    You can replace the CSV with an official list later without touching the UI code.
    """
    df = pd.read_csv(csv_path)
    df["detector_id"] = df["detector_id"].astype(str).str.strip()
    return df


@cache_ttl(ttl=30)
def fetch_timeseries(detector_id: str, start_dt: datetime, end_dt: datetime, sensor_type: str = "V30") -> pd.DataFrame:
    """
    Fetch a 30-second time series for a given detector and time window.

    Parameters
    ----------
    detector_id : str
        The detector identifier (as listed in your metadata).
    start_dt : datetime-like
        Start timestamp (interpreted as local America/Chicago if timezone-aware).
    end_dt : datetime-like
        End timestamp (interpreted as local America/Chicago if timezone-aware).
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
    start_ts = pd.Timestamp(start_dt)
    end_ts = pd.Timestamp(end_dt)

    if pd.isna(start_ts) or pd.isna(end_ts) or end_ts <= start_ts:
        return pd.DataFrame(columns=["ts", "value"])

    # Normalize to naive local timestamps for downstream use.
    if start_ts.tz is not None:
        start_ts = start_ts.tz_convert("America/Chicago").tz_localize(None).to_pydatetime()
    else:
        start_ts = start_ts.to_pydatetime()

    if end_ts.tz is not None:
        end_ts = end_ts.tz_convert("America/Chicago").tz_localize(None).to_pydatetime()
    else:
        end_ts = end_ts.to_pydatetime()

    local_df = _load_local_timeseries(detector_id, sensor_type, start_ts, end_ts)
    if local_df is not None:
        mask = (local_df["ts"] >= start_ts) & (local_df["ts"] < end_ts)
        return local_df.loc[mask].sort_values("ts").reset_index(drop=True)
    if not MN_API_BASE:
        return pd.DataFrame(columns=["ts", "value"])

    # ---- Real upstream path (replace with your actual endpoint contract) ----
    params = {
        "detector": detector_id,
        "type": sensor_type,
        "start": start_ts.strftime("%Y-%m-%dT%H:%M"),
        "end": end_ts.strftime("%Y-%m-%dT%H:%M"),
        "step": "30",
    }
    if start_ts.date() == end_ts.date():
        params["date"] = start_ts.strftime("%Y-%m-%d")

    r = requests.get(f"{MN_API_BASE}/timeseries", params=params, timeout=TIMEOUT)
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


def _load_local_timeseries(detector_id: str, sensor_type: str, start_ts: datetime, end_ts: datetime) -> pd.DataFrame | None:
    sensor_type_slug = sensor_type.lower()
    if LOCAL_DATA_ROOT is None:
        return None

    frames: list[pd.DataFrame] = []

    day = start_ts.date()
    end_day = end_ts.date()

    while day <= end_day:
        year_dir = LOCAL_DATA_ROOT / f"{day:%Y}"
        file_path = year_dir / f"{day:%Y%m%d}" / f"{detector_id}.{sensor_type_slug}.json"
        print("file_path:", file_path)
        if not file_path.exists():
            return None
        try:
            with file_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception:
            return None

        df_day = _payload_to_dataframe(payload, day)
        if df_day is None:
            return None

        frames.append(df_day)
        day += timedelta(days=1)

    if not frames:
        return None

    return pd.concat(frames, ignore_index=True)


def _payload_to_dataframe(payload, day: date) -> pd.DataFrame | None:
    if isinstance(payload, list):
        if not payload:
            return pd.DataFrame(columns=["ts", "value"])
        start = pd.Timestamp(day, tz="America/Chicago")
        idx = pd.date_range(start=start, periods=len(payload), freq="30S")
        values = pd.to_numeric(pd.Series(payload, dtype="float"), errors="coerce")
        df = pd.DataFrame(
            {
                "ts": idx.tz_convert("America/Chicago").tz_localize(None),
                "value": values,
            }
        )
        return df.dropna(subset=["ts", "value"])

    try:
        df_raw = normalize_timeseries_json(payload)
    except Exception:
        return None

    ts = pd.to_datetime(df_raw["timestamp"], errors="coerce", utc=True)
    df = pd.DataFrame(
        {
            "ts": ts.dt.tz_convert("America/Chicago").dt.tz_localize(None),
            "value": pd.to_numeric(df_raw["value"], errors="coerce"),
        }
    )
    return df.dropna()


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
