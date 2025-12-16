from pathlib import Path
import json
from datetime import datetime, date

import numpy as np
import pandas as pd
import clickhouse_connect


# -----------------------------
# Config
# -----------------------------
BASE_DIR = Path("/data/pouya_data/mndot_raw_data")
DETECTORS_CSV = Path("/home/MnDoT_project/preprocess/detectors_minneapolis_radius_25.0km.csv")
YEAR = 2020
MONTH = 3  # March

CH_HOST = "127.0.0.1"
CH_PORT = 8123
DB = "sensors"
TABLE = "raw_30s"
CH_TABLE = TABLE

TARGET_ROWS_PER_BATCH = 100_000


def ensure_schema():
    client = clickhouse_connect.get_client(host=CH_HOST, port=CH_PORT, database="default")
    client.command(f"CREATE DATABASE IF NOT EXISTS {DB}")

    client.command(f"""
    CREATE TABLE IF NOT EXISTS {DB}.{TABLE}
    (
        sensor_id   String,
        sensor_type Enum8('c30' = 1, 's30' = 2, 'v30' = 3),
        ts          DateTime,
        value       Nullable(Int32),
        day Date MATERIALIZED toDate(ts)
    )
    ENGINE = MergeTree
    PARTITION BY toYYYYMM(day)
    ORDER BY (sensor_id, sensor_type, ts)
    """)
    return client


def load_allowed_sensors(csv_path: Path) -> set[str]:
    # Force detector_name to string to avoid losing formatting (e.g., leading zeros)
    df = pd.read_csv(csv_path, dtype={"detector_name": "string"})
    if "detector_name" not in df.columns:
        raise ValueError(f"'detector_name' column not found in {csv_path}")

    allowed = set(df["detector_name"].dropna().astype(str).str.strip())
    if not allowed:
        raise ValueError(f"No detector_name values found in {csv_path}")
    return allowed


# -----------------------------
# Helpers
# -----------------------------
def parse_day_folder_yyyymmdd(folder_name: str) -> date:
    return datetime.strptime(folder_name, "%Y%m%d").date()


def file_to_df(day: date, json_path: Path) -> pd.DataFrame:
    """
    Reads one JSON file: <SENSOR_ID>.<SENSOR_TYPE>.json
    JSON content: list of 2880 items, each is either an integer or null
    """
    sensor_id, sensor_type, ext = json_path.name.rsplit(".", 2)
    if ext.lower() != "json":
        raise ValueError(f"Unexpected extension: {json_path}")
    if sensor_type not in ("c30", "s30", "v30"):
        raise ValueError(f"Unexpected sensor_type '{sensor_type}' in {json_path}")

    with json_path.open("r") as f:
        values = json.load(f)

    if not isinstance(values, list):
        raise ValueError(f"JSON content is not a list: {json_path}")
    if len(values) != 2880:
        raise ValueError(f"Expected 2880 samples, got {len(values)} in {json_path}")

    # allow None or int
    for i, v in enumerate(values):
        if v is None:
            continue
        if isinstance(v, (int, np.integer)):
            continue
        raise ValueError(f"Non-integer non-null value at index {i}: {v!r} in {json_path}")

    value_arr = pd.array(values, dtype="Int32")  # nullable int

    base = np.datetime64(f"{day:%Y-%m-%d}T00:00:00", "s")
    offsets = np.arange(2880, dtype=np.int64) * np.timedelta64(30, "s")
    ts = base + offsets

    return pd.DataFrame(
        {
            "sensor_id": sensor_id,
            "sensor_type": sensor_type,
            "ts": ts.astype("datetime64[s]"),
            "value": value_arr,
        }
    )


def iter_month_day_folders(base_dir: Path, year: int, month: int):
    year_dir = base_dir / str(year)
    if not year_dir.exists():
        raise FileNotFoundError(f"Year directory not found: {year_dir}")

    prefix = f"{year}{month:02d}"  # e.g., "202003"
    for p in sorted(year_dir.iterdir()):
        if not p.is_dir():
            continue
        name = p.name
        if len(name) == 8 and name.startswith(prefix) and name.isdigit():
            yield parse_day_folder_yyyymmdd(name), p


def get_sensors_client():
    return clickhouse_connect.get_client(host=CH_HOST, port=CH_PORT, database=DB)


# -----------------------------
# Main ingestion
# -----------------------------
def ingest_march_2020(allowed_sensors: set[str]):
    client = get_sensors_client()

    batch = []
    batch_rows = 0

    insert_settings = {
        # "async_insert": 1,
        # "wait_for_async_insert": 1,
    }

    for day, folder in iter_month_day_folders(BASE_DIR, YEAR, MONTH):
        for json_path in folder.glob("*.json"):
            # robust split in case sensor_id has dots
            sensor_id, _, _ = json_path.name.rsplit(".", 2)

            # FILTER: only ingest sensors present in CSV
            if sensor_id not in allowed_sensors:
                continue

            df = file_to_df(day, json_path)
            batch.append(df)
            batch_rows += len(df)

            if batch_rows >= TARGET_ROWS_PER_BATCH:
                big = pd.concat(batch, ignore_index=True)
                client.insert_df(CH_TABLE, big, settings=insert_settings)
                print(f"Inserted {len(big):,} rows through {day} (folder {folder.name})")
                batch.clear()
                batch_rows = 0

    if batch:
        big = pd.concat(batch, ignore_index=True)
        client.insert_df(CH_TABLE, big, settings=insert_settings)
        print(f"Inserted final {len(big):,} rows")


if __name__ == "__main__":
    ensure_schema()
    allowed = load_allowed_sensors(DETECTORS_CSV)
    ingest_march_2020(allowed)
