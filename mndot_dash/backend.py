import json
from datetime import date
from functools import lru_cache

import pandas as pd

import clickhouse_connect

from .config import CH_HOST, CH_PORT, CH_DB


@lru_cache(maxsize=1)
def ch():
    return clickhouse_connect.get_client(host=CH_HOST, port=CH_PORT, database=CH_DB)


def choose_bucket_seconds(start_date: date, end_date: date) -> int:
    days = (end_date - start_date).days + 1
    if days <= 2:
        return 30
    if days <= 14:
        return 300  # 5 min
    if days <= 60:
        return 3600  # 1 hour
    return 86400  # 1 day


def fetch_meta_with_presence(routes, directions, sensor_type_db, start_day, end_day) -> pd.DataFrame:
    sql = """
        SELECT
            m.sensor_id, m.route, m.direction, m.lat, m.lon, m.lane,
            (ifNull(d.cnt, 0) > 0) AS has_data
        FROM detector_meta AS m
        LEFT JOIN
        (
            SELECT sensor_id, count() AS cnt
            FROM raw_30s
            WHERE toString(sensor_type) = {sensor_type:String}
              AND day >= {start_day:Date} AND day <= {end_day:Date}
            GROUP BY sensor_id
        ) AS d USING (sensor_id)
        WHERE m.route IN {routes:Array(String)}
          AND m.direction IN {directions:Array(String)}
        ORDER BY m.sensor_id
    """
    return ch().query_df(
        sql,
        parameters={
            "routes": routes,
            "directions": directions,
            "sensor_type": sensor_type_db,
            "start_day": start_day,
            "end_day": end_day,
        },
    )


def fetch_ts_joined(sensor_id, routes, directions, sensor_type_db, start_dt, end_dt, bucket_s) -> pd.DataFrame:
    sql = f"""
        WITH toStartOfInterval(r.ts, INTERVAL {int(bucket_s)} SECOND) AS t
        SELECT
            t AS ts,
            avg(r.value) AS value
        FROM raw_30s AS r
        INNER JOIN detector_meta AS m ON r.sensor_id = m.sensor_id
        WHERE r.sensor_id = {{sensor_id:String}}
          AND toString(r.sensor_type) = {{sensor_type:String}}
          AND r.ts >= {{start:DateTime}} AND r.ts <= {{end:DateTime}}
          AND m.route IN {{routes:Array(String)}}
          AND m.direction IN {{directions:Array(String)}}
        GROUP BY ts
        ORDER BY ts
    """
    return ch().query_df(
        sql,
        parameters={
            "sensor_id": str(sensor_id),
            "sensor_type": str(sensor_type_db),
            "start": start_dt,
            "end": end_dt,
            "routes": routes,
            "directions": directions,
        },
    )


def fetch_raw_count(sensor_id, routes, directions, sensor_type_db, start_dt, end_dt) -> int:
    sql = """
        SELECT count() AS raw_rows
        FROM raw_30s AS r
        INNER JOIN detector_meta AS m ON r.sensor_id = m.sensor_id
        WHERE r.sensor_id = {sensor_id:String}
          AND toString(r.sensor_type) = {sensor_type:String}
          AND r.ts >= {start:DateTime} AND r.ts <= {end:DateTime}
          AND m.route IN {routes:Array(String)}
          AND m.direction IN {directions:Array(String)}
    """
    df = ch().query_df(
        sql,
        parameters={
            "sensor_id": str(sensor_id),
            "sensor_type": str(sensor_type_db),
            "start": start_dt,
            "end": end_dt,
            "routes": routes,
            "directions": directions,
        },
    )
    return int(df["raw_rows"].iloc[0]) if not df.empty else 0


def fetch_con_zero_vol(sensor_id, routes, directions, sensor_type_db, start_dt, end_dt, min_run_slots: int = 20) -> int:
    """
    conZeroVol: maximum length of a consecutive run of 30s slots where value == 0.
    Runs shorter than min_run_slots are ignored (default: 20 slots = 10 minutes).
    """
    min_run_slots = int(min_run_slots)
    if min_run_slots <= 0:
        raise ValueError("min_run_slots must be >= 1")

    sql = f"""
        SELECT ifNull(maxIf(run_len, run_len >= {min_run_slots}), 0) AS conZeroVol
        FROM
        (
            SELECT day, grp, count() AS run_len
            FROM
            (
                SELECT
                    day,
                    ts,
                    is_zero,
                    sum(toUInt8(is_zero != prev_is_zero)) OVER (PARTITION BY day ORDER BY ts) AS grp
                FROM
                (
                    SELECT
                        toDate(r.ts) AS day,
                        r.ts AS ts,
                        (ifNull(r.value, 1) = 0) AS is_zero,
                        lagInFrame((ifNull(r.value, 1) = 0), 1, 0) OVER (PARTITION BY toDate(r.ts) ORDER BY r.ts) AS prev_is_zero
                    FROM raw_30s AS r
                    INNER JOIN detector_meta AS m ON r.sensor_id = m.sensor_id
                    WHERE r.sensor_id = {{sensor_id:String}}
                      AND toString(r.sensor_type) = {{sensor_type:String}}
                      AND r.ts >= {{start:DateTime}} AND r.ts <= {{end:DateTime}}
                      AND m.route IN {{routes:Array(String)}}
                      AND m.direction IN {{directions:Array(String)}}
                )
            )
            WHERE is_zero
            GROUP BY day, grp
        )
    """

    df = ch().query_df(
        sql,
        parameters={
            "sensor_id": str(sensor_id),
            "sensor_type": str(sensor_type_db),
            "start": start_dt,
            "end": end_dt,
            "routes": routes,
            "directions": directions,
        },
    )
    if df.empty:
        return 0
    return int(df["conZeroVol"].iloc[0])


def meta_cache_key(routes, directions, sensor_type_db, start_day, end_day) -> str:
    payload = {
        "routes": list(routes),
        "directions": list(directions),
        "sensor_type": sensor_type_db,
        "start_day": str(start_day),
        "end_day": str(end_day),
    }
    return "meta:" + json.dumps(payload, sort_keys=True)


def ts_cache_key(sensor_id, routes, directions, sensor_type_db, start_dt, end_dt, bucket_s) -> str:
    payload = {
        "sensor_id": str(sensor_id),
        "routes": list(routes),
        "directions": list(directions),
        "sensor_type": sensor_type_db,
        "start": str(start_dt),
        "end": str(end_dt),
        "bucket_s": int(bucket_s),
    }
    return "ts:" + json.dumps(payload, sort_keys=True)
