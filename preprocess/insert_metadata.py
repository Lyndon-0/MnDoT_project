import argparse
import csv
import os
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
        pkey        String,
        name        String,
        station     String,
        type        String,
        location    String,
        lane_number Nullable(UInt8),
        lane_type   String,
        lat         Float64,
        lon         Float64
    )
    ENGINE = MergeTree
    ORDER BY (pkey)
    """)


def load_rows(csv_path: str) -> List[Tuple]:
    rows: List[Tuple] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)

        for r in reader:
            lane_raw = r["lane_number"]
            lane_number = None if lane_raw in ("", "<Null>") else int(lane_raw)

            rows.append(
                (
                    str(r["pkey"]),
                    str(r["name"]),
                    str(r["station"]),
                    str(r["type"]),
                    str(r["location"]),
                    lane_number,
                    str(r["lane_type"]),
                    float(r["lat"]),
                    float(r["lon"]),
                )
            )

    return rows


def insert_rows(client, table: str, rows: List[Tuple]) -> None:
    cols = ["pkey", "name", "station", "type", "location", "lane_number", "lane_type", "lat", "lon"]
    client.insert(f"{DB}.{table}", rows, column_names=cols)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", nargs="?", default="data/detectors.csv", help="Path to detectors CSV")
    ap.add_argument("--table", default="detectors_meta_info", help="ClickHouse table name")
    args = ap.parse_args()

    kwargs = {}
    if CH_USER:
        kwargs["username"] = CH_USER
    if CH_PASSWORD:
        kwargs["password"] = CH_PASSWORD
    client = clickhouse_connect.get_client(host=CH_HOST, port=CH_PORT, database="default", **kwargs)

    ensure_db_and_table(client, args.table)

    rows = load_rows(args.csv_path)
    insert_rows(client, args.table, rows)
    print(f"Done. Loaded into {DB}.{args.table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
