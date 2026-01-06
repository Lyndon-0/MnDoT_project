import argparse
import json
from datetime import datetime, date, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
import time

import clickhouse_connect
import numpy as np
import pandas as pd


# -----------------------------
# Config
# -----------------------------
BASE_DIR = Path("/data/pouya_data/mndot_raw_data")
DETECTORS_CSV = Path("/home/MnDoT_project/preprocess/detectors_minneapolis_radius_25.0km.csv")
YEAR = 2020

CH_HOST = "127.0.0.1"
CH_PORT = 8123
DB = "sensors"
TABLE = "raw_30s"
CH_TABLE = TABLE

TARGET_ROWS_PER_BATCH = 100_000
DEFAULT_START_DATE = date(YEAR, 1, 1)
DEFAULT_END_DATE = date(YEAR, 12, 31)

SENSOR_TYPE_TO_ENDPOINT = {
    "c30": "occupancy",
    "s30": "speed",
    "v30": "counts",
}


def ensure_schema():
    client = clickhouse_connect.get_client(host=CH_HOST, port=CH_PORT, database="default")
    client.command(f"CREATE DATABASE IF NOT EXISTS {DB}")

    client.command(f"""
    CREATE TABLE IF NOT EXISTS {DB}.{TABLE}
    (
        sensor_id   String,
        sensor_type Enum8('c30' = 1, 's30' = 2, 'v30' = 3),
        ts          DateTime,
        value       Nullable(Float64),
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
def parse_yyyymmdd(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y%m%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date '{value}'. Expected YYYYMMDD.") from exc


def iter_dates(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def day_folder(base_dir: Path, day: date) -> Path:
    return base_dir / str(day.year) / f"{day:%Y%m%d}"


def values_to_df(
    sensor_id: str,
    sensor_type: str,
    day: date,
    values,
    *,
    allow_empty: bool = False,
) -> pd.DataFrame:
    if sensor_type not in SENSOR_TYPE_TO_ENDPOINT:
        raise ValueError(f"Unexpected sensor_type '{sensor_type}' for sensor {sensor_id}")

    if allow_empty and not values:
        values = [None] * 2880

    if not isinstance(values, list):
        raise ValueError(f"Expected a list of values for {sensor_id} {sensor_type} on {day}")
    if len(values) != 2880:
        raise ValueError(f"Expected 2880 samples, got {len(values)} for {sensor_id} {sensor_type} on {day}")

    cleaned = []
    for i, v in enumerate(values):
        if v is None:
            cleaned.append(None)
            continue
        is_int = isinstance(v, (int, np.integer))
        is_float = isinstance(v, float)
        if sensor_type == "c30":
            if is_int or is_float:
                cleaned.append(float(v))
                continue
        else:
            if is_int:
                cleaned.append(int(v))
                continue
        raise ValueError(
            f"Unexpected value type at index {i}: {v!r} for {sensor_id} {sensor_type} on {day} "
            "(occupancy allows float; speed/volume must be int)"
        )

    base = np.datetime64(f"{day:%Y-%m-%d}T00:00:00", "s")
    offsets = np.arange(2880, dtype=np.int64) * np.timedelta64(30, "s")
    ts = base + offsets

    return pd.DataFrame(
        {
            "sensor_id": sensor_id,
            "sensor_type": sensor_type,
            "ts": ts.astype("datetime64[s]"),
            "value": pd.array(cleaned, dtype="Float64"),  # nullable float (occupancy can be float)
        }
    )


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

    return values_to_df(sensor_id, sensor_type, day, values)


def get_sensors_client():
    return clickhouse_connect.get_client(host=CH_HOST, port=CH_PORT, database=DB)


def fetch_api_values(sensor_id: str, sensor_type: str, day: date):
    endpoint = SENSOR_TYPE_TO_ENDPOINT[sensor_type]
    url = f"https://data.dot.state.mn.us/mayfly/{endpoint}?date={day:%Y%m%d}&detector={sensor_id}"
    attempts = 3
    for attempt in range(1, attempts + 1):
        try:
            with urlopen(url, timeout=30) as resp:
                status = getattr(resp, "status", None) or resp.getcode()
                if status == 404:
                    return [None] * 2880
                if status != 200:
                    raise RuntimeError(f"API request failed with status {status} for {url}")
                payload = json.load(resp)
                break
        except HTTPError as e:
            if e.code == 404:
                return [None] * 2880
            if attempt == attempts:
                raise RuntimeError(f"API request failed for {url}: {e}") from e
        except URLError as e:
            if attempt == attempts:
                raise RuntimeError(f"API request error for {url}: {e}") from e

        # retry on next loop
        time.sleep(1)

    if payload is None:
        return [None] * 2880
    if not isinstance(payload, list):
        raise ValueError(f"API response was not a list for {url}")
    return payload


# -----------------------------
# Main ingestion
# -----------------------------
def ingest_from_files(
    allowed_sensors: set[str],
    start_date: date,
    end_date: date,
    base_dir: Path = BASE_DIR,
    *,
    target_rows_per_batch: int = TARGET_ROWS_PER_BATCH,
):
    client = get_sensors_client()

    batch = []
    batch_rows = 0

    insert_settings = {}

    for day in iter_dates(start_date, end_date):
        folder = day_folder(base_dir, day)
        if not folder.exists():
            print(f"Skipping missing folder {folder}")
            continue
        if not folder.is_dir():
            print(f"Skipping non-directory path {folder}")
            continue

        for json_path in folder.glob("*.json"):
            # robust split in case sensor_id has dots
            sensor_id, _, _ = json_path.name.rsplit(".", 2)

            # FILTER: only ingest sensors present in CSV
            if sensor_id not in allowed_sensors:
                continue

            df = file_to_df(day, json_path)
            batch.append(df)
            batch_rows += len(df)

            if batch_rows >= target_rows_per_batch:
                big = pd.concat(batch, ignore_index=True)
                client.insert_df(CH_TABLE, big, settings=insert_settings)
                print(f"Inserted {len(big):,} rows through {day} (folder {folder.name})")
                batch.clear()
                batch_rows = 0

    if batch:
        big = pd.concat(batch, ignore_index=True)
        client.insert_df(CH_TABLE, big, settings=insert_settings)
        print(f"Inserted final {len(big):,} rows from files")


def ingest_from_api(
    allowed_sensors: set[str],
    start_date: date,
    end_date: date,
    *,
    target_rows_per_batch: int = TARGET_ROWS_PER_BATCH,
):
    client = get_sensors_client()

    batch = []
    batch_rows = 0

    insert_settings = {
        # "async_insert": 1,
        # "wait_for_async_insert": 1,
    }

    sensor_ids = sorted(allowed_sensors)
    for day in iter_dates(start_date, end_date):
        for sensor_id in sensor_ids:
            for sensor_type in SENSOR_TYPE_TO_ENDPOINT:
                values = fetch_api_values(sensor_id, sensor_type, day)
                df = values_to_df(sensor_id, sensor_type, day, values, allow_empty=True)
                batch.append(df)
                batch_rows += len(df)

                if batch_rows >= target_rows_per_batch:
                    big = pd.concat(batch, ignore_index=True)
                    client.insert_df(CH_TABLE, big, settings=insert_settings)
                    print(f"Inserted {len(big):,} rows through {day} (API)")
                    batch.clear()
                    batch_rows = 0

    if batch:
        big = pd.concat(batch, ignore_index=True)
        client.insert_df(CH_TABLE, big, settings=insert_settings)
        print(f"Inserted final {len(big):,} rows from API")


def ingest_year(
    allowed_sensors: set[str],
    year: int = YEAR,
    *,
    base_dir: Path = BASE_DIR,
    target_rows_per_batch: int = TARGET_ROWS_PER_BATCH,
):
    start = date(year, 1, 1)
    end = date(year, 12, 31)
    ingest_from_files(
        allowed_sensors,
        start,
        end,
        base_dir=base_dir,
        target_rows_per_batch=target_rows_per_batch,
    )


def main():
    parser = argparse.ArgumentParser(description="Insert MnDOT detector values into ClickHouse.")
    parser.add_argument(
        "--data-source",
        choices=["api", "files"],
        default="files",
        help="Use 'files' to read local JSON dumps or 'api' to fetch from the Mayfly API.",
    )
    parser.add_argument(
        "--start-date",
        type=parse_yyyymmdd,
        default=DEFAULT_START_DATE,
        help="Start date (inclusive) in YYYYMMDD format. Default: 20200101.",
    )
    parser.add_argument(
        "--end-date",
        type=parse_yyyymmdd,
        default=DEFAULT_END_DATE,
        help="End date (inclusive) in YYYYMMDD format. Default: 20201231.",
    )
    parser.add_argument(
        "--detectors-csv",
        type=Path,
        default=DETECTORS_CSV,
        help="CSV file containing a detector_name column.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=BASE_DIR,
        help="Base directory containing <year>/<yyyymmdd> folders (used with data-source=files).",
    )
    parser.add_argument(
        "--target-rows",
        type=int,
        default=TARGET_ROWS_PER_BATCH,
        help=f"Target number of rows per ClickHouse insert (default: {TARGET_ROWS_PER_BATCH}).",
    )
    args = parser.parse_args()

    if args.start_date > args.end_date:
        parser.error("start-date must be on or before end-date")

    ensure_schema()
    allowed = load_allowed_sensors(args.detectors_csv)

    if args.data_source == "files":
        ingest_from_files(
            allowed,
            args.start_date,
            args.end_date,
            base_dir=args.base_dir,
            target_rows_per_batch=args.target_rows
        )
    else:
        ingest_from_api(
            allowed,
            args.start_date,
            args.end_date,
            target_rows_per_batch=args.target_rows
        )


if __name__ == "__main__":
    raise SystemExit(main())
