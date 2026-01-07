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

TARGET_ROWS_PER_BATCH = 4_000_000  # Kept at 4M for optimal balance
DEFAULT_START_DATE = date(YEAR, 1, 1)
DEFAULT_END_DATE = date(YEAR, 12, 31)
SAMPLES_PER_DAY = 2880

# PERFORMANCE TUNING
API_CONCURRENCY = 1000   # 3000 might be too aggressive for open files/sockets, 1000 is safer
API_CHUNK_SIZE = 200
API_TIMEOUT_SECONDS = 60 # Increased timeout for safety
API_RETRY_DELAY_SECONDS = 0.1
API_MAX_RETRIES = 3
QUEUE_SIZE = 15_000      # Buffer for ~3-4 batches

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


def build_batch_columns(batch_buffer: list[tuple[str, str, date, np.ndarray]]) -> tuple[list[str], list[np.ndarray]]:
    column_names = ["sensor_id", "sensor_type", "ts", "value"]
    if not batch_buffer:
        return column_names, []

    sensor_ids, sensor_types, days, values_list = zip(*batch_buffer)
    rows_per_sensor = values_list[0].shape[0]

    ids_col = np.repeat(np.asarray(sensor_ids, dtype=object), rows_per_sensor)
    types_col = np.repeat(np.asarray(sensor_types, dtype=object), rows_per_sensor)

    one_day_offsets = np.arange(rows_per_sensor, dtype="timedelta64[30s]")
    day_starts = np.array(days, dtype="datetime64[s]")
    
    ts_col = (day_starts[:, np.newaxis] + one_day_offsets).reshape(-1)
    values_col = np.concatenate(values_list)

    order = np.lexsort((ts_col.view("int64"), types_col, ids_col))
    ids_col = ids_col[order]
    types_col = types_col[order]
    ts_col = ts_col[order].astype("datetime64[s]").astype(object)
    values_col = _nan_to_none(values_col[order])

    return column_names, [ids_col, types_col, ts_col, values_col]


def values_to_df(sensor_id, sensor_type, day, values, allow_empty=False):
    cleaned = normalize_values(sensor_id, sensor_type, day, values, allow_empty=allow_empty)
    sensor_ids = np.full(cleaned.shape[0], sensor_id, dtype=object)
    sensor_types = np.full(cleaned.shape[0], sensor_type, dtype=object)
    base = np.datetime64(f"{day:%Y-%m-%d}T00:00:00", "s")
    offsets = np.arange(cleaned.shape[0], dtype=np.int64) * np.timedelta64(30, "s")
    ts = (base + offsets).astype("datetime64[s]")
    
    return pd.DataFrame(
        {
            "sensor_id": sensor_ids,
            "sensor_type": sensor_types,
            "ts": ts,
            "value": pd.array(cleaned, dtype="Float64"),
        }
    )


def file_to_df_with_timings(day: date, json_path: Path):
    sensor_id, sensor_type, ext = json_path.name.rsplit(".", 2)
    
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


def _nan_to_none(values: np.ndarray) -> np.ndarray:
    if not np.issubdtype(values.dtype, np.floating):
        return values
    mask = np.isnan(values)
    if not mask.any():
        return values
    out = values.astype(object)
    out[mask] = None
    return out


def dataframe_to_column_arrays(df: pd.DataFrame) -> tuple[list[str], list[np.ndarray]]:
    column_names = ["sensor_id", "sensor_type", "ts", "value"]
    if df.empty:
        return column_names, []
    
    ts_col = df["ts"].to_numpy(copy=False).astype("datetime64[s]").astype(object)
    data = [
        df["sensor_id"].to_numpy(copy=False),
        df["sensor_type"].to_numpy(copy=False),
        ts_col,
        _nan_to_none(df["value"].to_numpy(dtype="float64", na_value=np.nan)),
    ]
    return column_names, data


def insert_column_arrays(client, table: str, column_names: list[str], data: list[np.ndarray], *, settings: dict | None = None):
    if not data:
        return
    client.insert(
        table,
        data,
        column_oriented=True,
        column_names=column_names,
        settings=settings or {},
    )


async def fetch_api_values(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    sensor_id: str,
    sensor_type: str,
    day: date,
):
    endpoint = SENSOR_TYPE_TO_ENDPOINT[sensor_type]
    url = f"https://data.dot.state.mn.us/mayfly/{endpoint}?date={day:%Y%m%d}&detector={sensor_id}"
    
    for attempt in range(1, API_MAX_RETRIES + 1):
        try:
            async with semaphore:
                async with session.get(url) as resp:
                    if resp.status == 404:
                        return [None] * 2880
                    if resp.status != 200:
                        raise aiohttp.ClientResponseError(
                            resp.request_info, resp.history, status=resp.status,
                            message=f"status {resp.status}", headers=resp.headers,
                        )
                    # Optimize: Read bytes directly and pass to orjson
                    raw_data = await resp.read()
                    return orjson.loads(raw_data)
        except Exception as e:
            if attempt == API_MAX_RETRIES:
                return [None] * 2880
        await asyncio.sleep(API_RETRY_DELAY_SECONDS)
    return [None] * 2880


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
        if not folder.exists() or not folder.is_dir():
            continue

        for json_path in folder.glob("*.json"):
            sensor_id, _, _ = json_path.name.rsplit(".", 2)
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
                col_names, data = dataframe_to_column_arrays(big)
                insert_column_arrays(client, CH_TABLE, col_names, data, settings=insert_settings)
                insert_elapsed = time.perf_counter() - insert_start

                print(
                    f"Inserted {len(big):,} rows from files | "
                    f"Read: {batch_timers['read']:.2f}s, Trans: {batch_timers['transform']:.2f}s, "
                    f"Concat: {concat_elapsed:.2f}s, Insert: {insert_elapsed:.2f}s"
                )
                batch.clear()
                batch_rows = 0
                batch_timers = {"read": 0.0, "transform": 0.0}

    if batch:
        big = pd.concat(batch, ignore_index=True)
        col_names, data = dataframe_to_column_arrays(big)
        insert_column_arrays(client, CH_TABLE, col_names, data, settings=insert_settings)
        print(f"Inserted final {len(big):,} rows from files.")


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
    # Async Insert + Wait reduces client-side blocking
    insert_settings = {'async_insert': 1, 'wait_for_async_insert': 1}

    sensor_ids = sorted(allowed_sensors)
    timeout = aiohttp.ClientTimeout(total=API_TIMEOUT_SECONDS)
    
    # CRITICAL FIX: Uncap connection limit. Without this, concurrency is capped at 100.
    connector = aiohttp.TCPConnector(limit=None, ttl_dns_cache=300)
    
    semaphore = asyncio.Semaphore(concurrency)
    queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_SIZE)

    async def producer():
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            for day in iter_dates(start_date, end_date):
                for sensor_chunk in chunked(sensor_ids, chunk_size):
                    jobs = []
                    for sensor_id in sensor_chunk:
                        for sensor_type in SENSOR_TYPE_TO_ENDPOINT:
                            jobs.append(fetch_api_values(session, semaphore, sensor_id, sensor_type, day))
                    
                    results = await asyncio.gather(*jobs)
                    
                    # Manually map results back to metadata
                    # Since we iterate strictly in order: Chunk -> Sensor -> Types
                    idx = 0
                    for sensor_id in sensor_chunk:
                        for sensor_type in SENSOR_TYPE_TO_ENDPOINT:
                            val = results[idx]
                            await queue.put((sensor_id, sensor_type, day, val))
                            idx += 1
                            
        # Signal ALL consumers to stop (we have 2 consumers)
        for _ in range(2): 
            await queue.put(None)

    async def insert_batch(batch_buffer, cid):
        loop = asyncio.get_running_loop()
        build_start = time.perf_counter()
        
        # CPU work: Build column arrays
        column_names, data = await loop.run_in_executor(None, build_batch_columns, batch_buffer)
        build_elapsed = time.perf_counter() - build_start
        insert_elapsed = 0.0
        row_count = data[0].shape[0] if data else 0
        
        if data:
            insert_start = time.perf_counter()
            # IO work: Send to ClickHouse (blocks thread but other consumer continues)
            await loop.run_in_executor(
                None, lambda: insert_column_arrays(client, CH_TABLE, column_names, data, settings=insert_settings)
            )
            insert_elapsed = time.perf_counter() - insert_start
            
        return build_elapsed, insert_elapsed, row_count

    async def consumer(cid):
        batch_buffer = []
        batch_rows = 0
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                break
            
            s_id, s_type, day, val = item
            cleaned = normalize_values(s_id, s_type, day, val, allow_empty=True)
            batch_buffer.append((s_id, s_type, day, cleaned))
            batch_rows += len(cleaned)

            if batch_rows >= target_rows_per_batch:
                b_t, i_t, count = await insert_batch(batch_buffer, cid)
                print(f"[Consumer-{cid}] Inserted {count:,} rows | Build: {b_t:.2f}s | Insert: {i_t:.2f}s")
                batch_buffer.clear()
                batch_rows = 0
            
            queue.task_done()

        if batch_rows:
            await insert_batch(batch_buffer, cid)

    # Launch Pipeline: 1 Producer, 2 Consumers
    prod = asyncio.create_task(producer())
    c1 = asyncio.create_task(consumer(1))
    c2 = asyncio.create_task(consumer(2))
    
    await asyncio.gather(prod, c1, c2)


# -----------------------------
# CLI Entry Point
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-source", choices=["api", "files"], default="files")
    parser.add_argument("--start-date", type=parse_yyyymmdd, default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", type=parse_yyyymmdd, default=DEFAULT_END_DATE)
    parser.add_argument("--detectors-csv", type=Path, default=DETECTORS_CSV)
    parser.add_argument("--base-dir", type=Path, default=BASE_DIR)
    parser.add_argument("--target-rows", type=int, default=TARGET_ROWS_PER_BATCH)
    parser.add_argument("--api-concurrency", type=int, default=API_CONCURRENCY)
    parser.add_argument("--api-chunk-size", type=int, default=API_CHUNK_SIZE)
    args = parser.parse_args()

    ensure_schema()
    allowed = load_allowed_sensors(args.detectors_csv)

    if args.data_source == "files":
        ingest_from_files(
            allowed, args.start_date, args.end_date,
            base_dir=args.base_dir, target_rows_per_batch=args.target_rows
        )
    else:
        ingest_from_api(
            allowed, args.start_date, args.end_date,
            target_rows_per_batch=args.target_rows,
            concurrency=args.api_concurrency, chunk_size=args.api_chunk_size
        )

if __name__ == "__main__":
    raise SystemExit(main())
