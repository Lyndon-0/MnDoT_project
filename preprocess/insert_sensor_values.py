import argparse
import asyncio
import json
import time
from datetime import datetime, date, timedelta
from pathlib import Path

import aiohttp

import clickhouse_connect
import numpy as np
import pandas as pd
import orjson

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

TARGET_ROWS_PER_BATCH = 8_000_000
DEFAULT_START_DATE = date(YEAR, 1, 1)
DEFAULT_END_DATE = date(YEAR, 12, 31)
SAMPLES_PER_DAY = 2880
API_CONCURRENCY = 300
API_CHUNK_SIZE = 100
API_TIMEOUT_SECONDS = 30
API_RETRY_DELAY_SECONDS = 0.1
API_MAX_RETRIES = 3

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


def chunked(seq, size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def normalize_values(
    sensor_id: str,
    sensor_type: str,
    day: date,
    values,
    *,
    allow_empty: bool = False,
) -> np.ndarray:
    if sensor_type not in SENSOR_TYPE_TO_ENDPOINT:
        raise ValueError(f"Unexpected sensor_type '{sensor_type}' for sensor {sensor_id}")

    if allow_empty and not values:
        values = [None] * SAMPLES_PER_DAY

    if not isinstance(values, list):
        raise ValueError(f"Expected a list of values for {sensor_id} {sensor_type} on {day}")
    arr = np.asarray(values, dtype=object)
    if arr.shape[0] != SAMPLES_PER_DAY:
        raise ValueError(f"Expected {SAMPLES_PER_DAY} samples, got {arr.shape[0]} for {sensor_id} {sensor_type} on {day}")

    is_null = pd.isna(arr)
    non_null = arr[~is_null]

    if sensor_type == "c30":
        try:
            non_null_cast = np.asarray(non_null, dtype=float)
        except Exception as exc:
            raise ValueError(
                f"Non-numeric value encountered for occupancy {sensor_id} on {day}: {exc}"
            ) from exc
    else:
        try:
            non_null_int = np.asarray(non_null, dtype=int)
        except Exception as exc:
            raise ValueError(
                f"Non-integer value encountered for {sensor_type} {sensor_id} on {day}: {exc}"
            ) from exc
        non_null_float = np.asarray(non_null, dtype=float)
        if not np.allclose(non_null_int, non_null_float, equal_nan=True):
            raise ValueError(
                f"Non-integer value encountered for {sensor_type} {sensor_id} on {day}"
            )
        non_null_cast = non_null_int.astype(float)

    cleaned = np.full(arr.shape[0], np.nan, dtype=float)
    cleaned[~is_null] = non_null_cast
    return cleaned


def build_batch_dataframe(batch_buffer: list[tuple[str, str, date, np.ndarray]]) -> pd.DataFrame:
    """
    Convert buffered (sensor_id, sensor_type, day, cleaned_values ndarray) tuples into one DataFrame.
    """
    if not batch_buffer:
        return pd.DataFrame()

    sensor_ids, sensor_types, days, values_list = zip(*batch_buffer)
    rows_per_sensor = values_list[0].shape[0]

    ids_col = np.repeat(np.asarray(sensor_ids, dtype=object), rows_per_sensor)
    types_col = np.repeat(np.asarray(sensor_types, dtype=object), rows_per_sensor)

    one_day_offsets = np.arange(rows_per_sensor, dtype="timedelta64[30s]")
    day_starts = np.array(days, dtype="datetime64[s]")
    ts_col = (day_starts[:, np.newaxis] + one_day_offsets).reshape(-1)

    values_col = np.concatenate(values_list)

    return pd.DataFrame(
        {
            "sensor_id": ids_col,
            "sensor_type": types_col,
            "ts": ts_col,
            "value": values_col, # It's already np.nan-filled float array
        }
    )


def values_to_df(
    sensor_id: str,
    sensor_type: str,
    day: date,
    values,
    *,
    allow_empty: bool = False,
) -> pd.DataFrame:
    sensor_ids, sensor_types, ts, cleaned = values_to_columns(sensor_id, sensor_type, day, values, allow_empty=allow_empty)

    return pd.DataFrame(
        {
            "sensor_id": sensor_ids,
            "sensor_type": sensor_types,
            "ts": ts,
            "value": pd.array(cleaned, dtype="Float64"),  # nullable float (occupancy can be float)
        }
    )


def values_to_columns(
    sensor_id: str,
    sensor_type: str,
    day: date,
    values,
    *,
    allow_empty: bool = False,
):
    cleaned = normalize_values(sensor_id, sensor_type, day, values, allow_empty=allow_empty)
    sensor_ids = np.full(cleaned.shape[0], sensor_id, dtype=object)
    sensor_types = np.full(cleaned.shape[0], sensor_type, dtype=object)
    base = np.datetime64(f"{day:%Y-%m-%d}T00:00:00", "s")
    offsets = np.arange(cleaned.shape[0], dtype=np.int64) * np.timedelta64(30, "s")
    ts = (base + offsets).astype("datetime64[s]")
    return sensor_ids, sensor_types, ts, cleaned


def file_to_df(day: date, json_path: Path) -> pd.DataFrame:
    """Reads one sensor JSON file into a DataFrame."""
    df, _, _ = file_to_df_with_timings(day, json_path)
    return df


def file_to_df_with_timings(day: date, json_path: Path) -> tuple[pd.DataFrame, float, float]:
    """
    Reads one JSON file: <SENSOR_ID>.<SENSOR_TYPE>.json
    JSON content: list of 2880 items, each is either an integer or null.
    Returns a tuple of (df, read_seconds, transform_seconds).
    """
    sensor_id, sensor_type, ext = json_path.name.rsplit(".", 2)
    if ext.lower() != "json":
        raise ValueError(f"Unexpected extension: {json_path}")
    if sensor_type not in ("c30", "s30", "v30"):
        raise ValueError(f"Unexpected sensor_type '{sensor_type}' in {json_path}")

    read_start = time.perf_counter()
    with json_path.open("r") as f:
        values = json.load(f)
    read_elapsed = time.perf_counter() - read_start

    transform_start = time.perf_counter()
    df = values_to_df(sensor_id, sensor_type, day, values)
    transform_elapsed = time.perf_counter() - transform_start

    return df, read_elapsed, transform_elapsed


def get_sensors_client():
    return clickhouse_connect.get_client(host=CH_HOST, port=CH_PORT, database=DB)


async def fetch_api_values(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    sensor_id: str,
    sensor_type: str,
    day: date,
):
    endpoint = SENSOR_TYPE_TO_ENDPOINT[sensor_type]
    url = f"https://data.dot.state.mn.us/mayfly/{endpoint}?date={day:%Y%m%d}&detector={sensor_id}"
    payload = None
    for attempt in range(1, API_MAX_RETRIES + 1):
        try:
            async with semaphore:
                async with session.get(url) as resp:
                    if resp.status == 404:
                        return [None] * 2880
                    if resp.status != 200:
                        raise aiohttp.ClientResponseError(
                            resp.request_info,
                            resp.history,
                            status=resp.status,
                            message=f"status {resp.status}",
                            headers=resp.headers,
                        )
                    # payload = await resp.json(content_type=None)
                    raw_data = await resp.read()
                    payload = orjson.loads(raw_data)
                    break
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt == API_MAX_RETRIES:
                print(f"API request failed after {attempt} attempts for {url}: {e}")
                return [None] * 2880
        await asyncio.sleep(API_RETRY_DELAY_SECONDS)

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
    batch_timers = {"read": 0.0, "transform": 0.0}

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

            df, read_time, transform_time = file_to_df_with_timings(day, json_path)
            batch_timers["read"] += read_time
            batch_timers["transform"] += transform_time
            batch.append(df)
            batch_rows += len(df)

            if batch_rows >= target_rows_per_batch:
                concat_start = time.perf_counter()
                big = pd.concat(batch, ignore_index=True)
                concat_elapsed = time.perf_counter() - concat_start

                insert_start = time.perf_counter()
                client.insert_df(CH_TABLE, big, settings=insert_settings)
                insert_elapsed = time.perf_counter() - insert_start

                print(
                    f"Inserted {len(big):,} rows through {day} (folder {folder.name}) | "
                    f"timings (s): read+decode={batch_timers['read']:.2f}, "
                    f"to_df={batch_timers['transform']:.2f}, "
                    f"concat={concat_elapsed:.2f}, insert={insert_elapsed:.2f}"
                )
                batch.clear()
                batch_rows = 0
                batch_timers = {"read": 0.0, "transform": 0.0}

    if batch:
        concat_start = time.perf_counter()
        big = pd.concat(batch, ignore_index=True)
        concat_elapsed = time.perf_counter() - concat_start

        insert_start = time.perf_counter()
        client.insert_df(CH_TABLE, big, settings=insert_settings)
        insert_elapsed = time.perf_counter() - insert_start

        print(
            f"Inserted final {len(big):,} rows from files | "
            f"timings (s): read+decode={batch_timers['read']:.2f}, "
            f"to_df={batch_timers['transform']:.2f}, "
            f"concat={concat_elapsed:.2f}, insert={insert_elapsed:.2f}"
        )


def ingest_from_api(
    allowed_sensors: set[str],
    start_date: date,
    end_date: date,
    *,
    target_rows_per_batch: int = TARGET_ROWS_PER_BATCH,
    concurrency: int = API_CONCURRENCY,
    chunk_size: int = API_CHUNK_SIZE,
):
    asyncio.run(
        ingest_from_api_async(
            allowed_sensors,
            start_date,
            end_date,
            target_rows_per_batch=target_rows_per_batch,
            concurrency=concurrency,
            chunk_size=chunk_size,
        )
    )


async def ingest_from_api_async(
    allowed_sensors: set[str],
    start_date: date,
    end_date: date,
    *,
    target_rows_per_batch: int = TARGET_ROWS_PER_BATCH,
    concurrency: int = API_CONCURRENCY,
    chunk_size: int = API_CHUNK_SIZE,
):
    client = get_sensors_client()
    insert_settings = {'async_insert': 1, 'wait_for_async_insert': 1}

    sensor_ids = sorted(allowed_sensors)
    expected_rows_per_day = len(sensor_ids) * len(SENSOR_TYPE_TO_ENDPOINT) * SAMPLES_PER_DAY if sensor_ids else 0
    timeout = aiohttp.ClientTimeout(total=API_TIMEOUT_SECONDS)
    semaphore = asyncio.Semaphore(concurrency)
    queue: asyncio.Queue = asyncio.Queue(maxsize=5000)

    async def producer():
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for day in iter_dates(start_date, end_date):
                for sensor_chunk in chunked(sensor_ids, chunk_size):
                    jobs = [
                        (
                            sensor_id,
                            sensor_type,
                            fetch_api_values(session, semaphore, sensor_id, sensor_type, day),
                        )
                        for sensor_id in sensor_chunk
                        for sensor_type in SENSOR_TYPE_TO_ENDPOINT
                    ]
                    results = await asyncio.gather(*(job[2] for job in jobs))
                    for (sensor_id, sensor_type, _), values in zip(jobs, results):
                        await queue.put((sensor_id, sensor_type, day, values))
        await queue.put(None)

    async def insert_batch(batch_buffer):
        loop = asyncio.get_running_loop()
        build_start = time.perf_counter()
        big = await loop.run_in_executor(None, build_batch_dataframe, batch_buffer)
        build_elapsed = time.perf_counter() - build_start
        insert_elapsed = 0.0
        if not big.empty:
            insert_start = time.perf_counter()
            await loop.run_in_executor(
                None, lambda: client.insert_df(CH_TABLE, big, settings=insert_settings)
            )
            insert_elapsed = time.perf_counter() - insert_start
        return build_elapsed, insert_elapsed, len(big)

    async def consumer():
        batch_buffer = []
        batch_rows = 0
        current_day: date | None = None
        day_rows = 0
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                break
            sensor_id, sensor_type, day, values = item
            if current_day is None:
                current_day = day
            if day != current_day:
                print(
                    f"[{current_day}] loaded {day_rows:,}"
                    + (f"/{expected_rows_per_day:,}" if expected_rows_per_day else "")
                )
                current_day = day
                day_rows = 0
            cleaned = normalize_values(sensor_id, sensor_type, day, values, allow_empty=True)
            batch_buffer.append((sensor_id, sensor_type, day, cleaned))
            batch_rows += len(cleaned)
            day_rows += len(cleaned)
            if batch_rows >= target_rows_per_batch:
                build_elapsed, insert_elapsed, _ = await insert_batch(batch_buffer)
                if expected_rows_per_day:
                    pct = (day_rows / expected_rows_per_day) * 100
                    print(
                        f"[{current_day}] {day_rows:,}/{expected_rows_per_day:,} rows ({pct:.1f}%) | "
                        f"timings (s): build_df={build_elapsed:.2f}, insert={insert_elapsed:.2f}"
                    )
                else:
                    print(
                        f"[{current_day}] {day_rows:,} rows | "
                        f"timings (s): build_df={build_elapsed:.2f}, insert={insert_elapsed:.2f}"
                    )
                batch_buffer.clear()
                batch_rows = 0
            queue.task_done()

        if batch_rows:
            build_elapsed, insert_elapsed, _ = await insert_batch(batch_buffer)
            if current_day is not None:
                if expected_rows_per_day:
                    pct = (day_rows / expected_rows_per_day) * 100
                    print(
                        f"[{current_day}] {day_rows:,}/{expected_rows_per_day:,} rows ({pct:.1f}%) | "
                        f"timings (s): build_df={build_elapsed:.2f}, insert={insert_elapsed:.2f}"
                    )
                else:
                    print(
                        f"[{current_day}] {day_rows:,} rows | "
                        f"timings (s): build_df={build_elapsed:.2f}, insert={insert_elapsed:.2f}"
                    )
            print("Inserted final batch from API")

    prod_task = asyncio.create_task(producer())
    cons_task = asyncio.create_task(consumer())
    await asyncio.gather(prod_task, cons_task)


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
    parser.add_argument(
        "--api-concurrency",
        type=int,
        default=API_CONCURRENCY,
        help=f"Maximum concurrent API requests (default: {API_CONCURRENCY}).",
    )
    parser.add_argument(
        "--api-chunk-size",
        type=int,
        default=API_CHUNK_SIZE,
        help=f"Number of sensors to schedule per batch of API tasks (default: {API_CHUNK_SIZE}).",
    )
    args = parser.parse_args()

    if args.start_date > args.end_date:
        parser.error("start-date must be on or before end-date")
    if args.api_concurrency < 1:
        parser.error("api-concurrency must be at least 1")
    if args.api_chunk_size < 1:
        parser.error("api-chunk-size must be at least 1")

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
            target_rows_per_batch=args.target_rows,
            concurrency=args.api_concurrency,
            chunk_size=args.api_chunk_size,
        )


if __name__ == "__main__":
    raise SystemExit(main())
