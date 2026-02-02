import argparse
import csv
import os
import sys
from typing import List, Tuple

import clickhouse_connect


CH_HOST = os.environ.get("CH_HOST", "127.0.0.1")
CH_PORT = int(os.environ.get("CH_PORT", "8123"))
DB = os.environ.get("CH_DB", "sensors")
CH_USER = os.environ.get("CH_USER", "")
CH_PASSWORD = os.environ.get("CH_PASSWORD", "")


def ensure_db_and_table(client, table: str) -> None:
    client.command(f"CREATE DATABASE IF NOT EXISTS {DB}")

    client.command(f"""
    CREATE TABLE IF NOT EXISTS {DB}.{table}
    (
        sensor_id  String,
        route      String,
        direction  String,
        lat        Float64,
        lon        Float64,
        lane       UInt8
    )
    ENGINE = MergeTree
    ORDER BY (sensor_id)
    """)


def parse_lane(raw: str) -> int:
    lane = int(raw)
    if not (0 <= lane <= 255):
        raise ValueError(f"lane out of range [0,255]: {lane}")
    return lane


def load_rows(csv_path: str) -> List[Tuple]:
    rows: List[Tuple] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"detector_name", "route", "direction", "lat", "lon", "lane"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

        for i, r in enumerate(reader, start=2):  # start=2 to account for header row being line 1
            try:
                sensor_id = str(r["detector_name"]).strip()
                route = str(r["route"]).strip()
                direction = str(r["direction"]).strip()
                lat = float(r["lat"])
                lon = float(r["lon"])
                lane = parse_lane(r["lane"])

                if not sensor_id:
                    raise ValueError("empty detector_name")

                rows.append((sensor_id, route, direction, lat, lon, lane))
            except Exception as e:
                raise ValueError(f"Error parsing CSV at line {i}: {e}") from e

    return rows


def insert_rows(client, table: str, rows: List[Tuple], batch_size: int) -> None:
    cols = ["sensor_id", "route", "direction", "lat", "lon", "lane"]

    if batch_size <= 1:
        # Row-by-row insert (as requested; slower for large files)
        for idx, row in enumerate(rows, start=1):
            client.insert(f"{DB}.{table}", [row], column_names=cols)
            if idx % 1000 == 0:
                print(f"Inserted {idx} rows...")
        print(f"Inserted {len(rows)} rows total.")
        return

    # Batched insert (recommended for performance)
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        client.insert(f"{DB}.{table}", chunk, column_names=cols)
        print(f"Inserted {min(start + batch_size, len(rows))}/{len(rows)} rows...")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", help="Path to the detector metadata CSV file")
    ap.add_argument("--table", default="detector_meta", help="ClickHouse table name (default: detector_meta)")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="Insert batch size. Use 1 for row-by-row inserts (default: 1). "
                         "Use e.g. 5000 for much faster loads.")
    args = ap.parse_args()

    kwargs = {}
    if CH_USER:
        kwargs["username"] = CH_USER
    if CH_PASSWORD:
        kwargs["password"] = CH_PASSWORD
    client = clickhouse_connect.get_client(host=CH_HOST, port=CH_PORT, database="default", **kwargs)

    ensure_db_and_table(client, args.table)

    rows = load_rows(args.csv_path)
    if not rows:
        print("No rows found in CSV; nothing to insert.")
        return 0

    insert_rows(client, args.table, rows, args.batch_size)
    print(f"Done. Loaded into {DB}.{args.table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
