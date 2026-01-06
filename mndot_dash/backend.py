import json
from datetime import date
from functools import lru_cache
from typing import Optional

import pandas as pd

import clickhouse_connect

from .config import CH_HOST, CH_PORT, CH_DB, DAILY_METRICS_TABLE


PRECOMPUTED_MIN_RUN_SLOTS = 20
PRECOMPUTED_OCC_THRESHOLD = 0.2


@lru_cache(maxsize=1)
def ch():
    return clickhouse_connect.get_client(host=CH_HOST, port=CH_PORT, database=CH_DB)


def fetch_precomputed_metric(metric: str, sensor_id, start_day: date, end_day: date, sensor_type: Optional[str] = None) -> Optional[int]:
    """
    Look up a metric in the daily_metrics table for the given sensor/day range.
    Returns None if the table is missing, the query fails, or no rows are present.
    """
    where_sensor_type = " AND sensor_type = {sensor_type:String}" if sensor_type else ""
    sql = f"""
        SELECT toUInt32(ifNull(max(value), 0)) AS metric_value
        FROM {CH_DB}.{DAILY_METRICS_TABLE} FINAL
        WHERE sensor_id = {{sensor_id:String}}
          AND metric = {{metric:String}}
          AND day >= {{start_day:Date}} AND day <= {{end_day:Date}}
          {where_sensor_type}
    """

    params = {
        "sensor_id": str(sensor_id),
        "metric": metric,
        "start_day": start_day,
        "end_day": end_day,
    }
    if sensor_type:
        params["sensor_type"] = sensor_type

    try:
        df = ch().query_df(sql, parameters=params)
    except Exception:
        return None

    if df.empty:
        return None

    value = df["metric_value"].iloc[0]
    if pd.isna(value):
        return None
    return int(value)


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

    metric_name = "conZeroOcc" if str(sensor_type_db) == "c30" else "conZeroVol"
    if min_run_slots == PRECOMPUTED_MIN_RUN_SLOTS:
        start_day = start_dt.date()
        end_day = end_dt.date()
        precomputed = fetch_precomputed_metric(metric_name, sensor_id, start_day, end_day, str(sensor_type_db))
        if precomputed is not None:
            return precomputed

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


def fetch_neg_vol_cnt(sensor_id, routes, directions, sensor_type_db, start_dt, end_dt) -> int:
    """
    negVolCnt: for each day in range, count 30s slots with negative values; return the maximum daily count.
    """
    start_day = start_dt.date()
    end_day = end_dt.date()

    metric_name = "negOccCnt" if str(sensor_type_db) == "c30" else "negVolCnt"
    precomputed = fetch_precomputed_metric(metric_name, sensor_id, start_day, end_day, str(sensor_type_db))
    if precomputed is not None:
        return precomputed

    sql = """
        SELECT ifNull(max(neg_cnt), 0) AS negVolCnt
        FROM
        (
            SELECT r.day AS day, countIf(ifNull(r.value, 0) < 0) AS neg_cnt
            FROM raw_30s AS r
            INNER JOIN detector_meta AS m ON r.sensor_id = m.sensor_id
            WHERE r.sensor_id = {sensor_id:String}
              AND toString(r.sensor_type) = {sensor_type:String}
              AND r.day >= {start_day:Date} AND r.day <= {end_day:Date}
              AND r.ts >= {start:DateTime} AND r.ts <= {end:DateTime}
              AND m.route IN {routes:Array(String)}
              AND m.direction IN {directions:Array(String)}
            GROUP BY day
        )
    """

    df = ch().query_df(
        sql,
        parameters={
            "sensor_id": str(sensor_id),
            "sensor_type": str(sensor_type_db),
            "start_day": start_day,
            "end_day": end_day,
            "start": start_dt,
            "end": end_dt,
            "routes": routes,
            "directions": directions,
        },
    )
    if df.empty:
        return 0
    return int(df["negVolCnt"].iloc[0])


def fetch_occ_lock_on(sensor_id, routes, directions, sensor_type_db, start_dt, end_dt) -> int:
    """
    occLockOn: for each day in range, count 30s slots where 99 < value <= 100; return the maximum daily count.
    Intended for occupancy (c30).
    """
    start_day = start_dt.date()
    end_day = end_dt.date()

    precomputed = fetch_precomputed_metric("occLockOn", sensor_id, start_day, end_day, str(sensor_type_db))
    if precomputed is not None:
        return precomputed

    sql = """
        SELECT ifNull(max(lock_cnt), 0) AS occLockOn
        FROM
        (
            SELECT r.day AS day, countIf((ifNull(r.value, -1) >= 99) AND (ifNull(r.value, -1) <= 100)) AS lock_cnt
            FROM raw_30s AS r
            INNER JOIN detector_meta AS m ON r.sensor_id = m.sensor_id
            WHERE r.sensor_id = {sensor_id:String}
              AND toString(r.sensor_type) = {sensor_type:String}
              AND r.day >= {start_day:Date} AND r.day <= {end_day:Date}
              AND r.ts >= {start:DateTime} AND r.ts <= {end:DateTime}
              AND m.route IN {routes:Array(String)}
              AND m.direction IN {directions:Array(String)}
            GROUP BY day
        )
    """

    df = ch().query_df(
        sql,
        parameters={
            "sensor_id": str(sensor_id),
            "sensor_type": str(sensor_type_db),
            "start_day": start_day,
            "end_day": end_day,
            "start": start_dt,
            "end": end_dt,
            "routes": routes,
            "directions": directions,
        },
    )
    if df.empty:
        return 0
    return int(df["occLockOn"].iloc[0])


def fetch_zvol_on_occ(sensor_id, routes, directions, start_dt, end_dt, vol_sensor_type_db: str = "v30", occ_sensor_type_db: str = "c30") -> int:
    """
    zvolOnOcc: for each day in range, count 30s slots where volume is zero (v30) while occupancy is non-zero (c30);
    return the maximum daily count.
    """
    start_day = start_dt.date()
    end_day = end_dt.date()

    metric_sensor_type = f"{vol_sensor_type_db}+{occ_sensor_type_db}"
    precomputed = fetch_precomputed_metric("zvolOnOcc", sensor_id, start_day, end_day, metric_sensor_type)
    if precomputed is not None:
        return precomputed

    sql = """
        SELECT ifNull(max(z_cnt), 0) AS zvolOnOcc
        FROM
        (
            SELECT day, countIf((vol = 0) AND (occ > 0)) AS z_cnt
            FROM
            (
                SELECT
                    r.day AS day,
                    r.ts AS ts,
                    anyIf(r.value, toString(r.sensor_type) = {vol_type:String}) AS vol,
                    anyIf(r.value, toString(r.sensor_type) = {occ_type:String}) AS occ
                FROM raw_30s AS r
                INNER JOIN detector_meta AS m ON r.sensor_id = m.sensor_id
                WHERE r.sensor_id = {sensor_id:String}
                  AND r.day >= {start_day:Date} AND r.day <= {end_day:Date}
                  AND r.ts >= {start:DateTime} AND r.ts <= {end:DateTime}
                  AND (toString(r.sensor_type) = {vol_type:String} OR toString(r.sensor_type) = {occ_type:String})
                  AND m.route IN {routes:Array(String)}
                  AND m.direction IN {directions:Array(String)}
                GROUP BY day, ts
            )
            GROUP BY day
        )
    """

    df = ch().query_df(
        sql,
        parameters={
            "sensor_id": str(sensor_id),
            "vol_type": str(vol_sensor_type_db),
            "occ_type": str(occ_sensor_type_db),
            "start_day": start_day,
            "end_day": end_day,
            "start": start_dt,
            "end": end_dt,
            "routes": routes,
            "directions": directions,
        },
    )
    if df.empty:
        return 0
    return int(df["zvolOnOcc"].iloc[0])


def fetch_vol_on_low_occ(
    sensor_id,
    routes,
    directions,
    start_dt,
    end_dt,
    vol_sensor_type_db: str = "v30",
    occ_sensor_type_db: str = "c30",
    occ_threshold: float = 0.2,
) -> int:
    """
    volOnLowOcc: for each day in range, count 30s slots where volume > 1 (v30) while occupancy <= occ_threshold (c30);
    return the maximum daily count.
    """
    start_day = start_dt.date()
    end_day = end_dt.date()

    metric_sensor_type = f"{vol_sensor_type_db}+{occ_sensor_type_db}"
    if abs(float(occ_threshold) - PRECOMPUTED_OCC_THRESHOLD) < 1e-9:
        precomputed = fetch_precomputed_metric("volOnLowOcc", sensor_id, start_day, end_day, metric_sensor_type)
        if precomputed is not None:
            return precomputed

    sql = """
        SELECT ifNull(max(cnt), 0) AS volOnLowOcc
        FROM
        (
            SELECT day, countIf((ifNull(vol, -1) > 1) AND (ifNull(occ, 999) <= {occ_threshold:Float64})) AS cnt
            FROM
            (
                SELECT
                    r.day AS day,
                    r.ts AS ts,
                    anyIf(r.value, toString(r.sensor_type) = {vol_type:String}) AS vol,
                    anyIf(r.value, toString(r.sensor_type) = {occ_type:String}) AS occ
                FROM raw_30s AS r
                INNER JOIN detector_meta AS m ON r.sensor_id = m.sensor_id
                WHERE r.sensor_id = {sensor_id:String}
                  AND r.day >= {start_day:Date} AND r.day <= {end_day:Date}
                  AND r.ts >= {start:DateTime} AND r.ts <= {end:DateTime}
                  AND (toString(r.sensor_type) = {vol_type:String} OR toString(r.sensor_type) = {occ_type:String})
                  AND m.route IN {routes:Array(String)}
                  AND m.direction IN {directions:Array(String)}
                GROUP BY day, ts
            )
            GROUP BY day
        )
    """

    df = ch().query_df(
        sql,
        parameters={
            "sensor_id": str(sensor_id),
            "vol_type": str(vol_sensor_type_db),
            "occ_type": str(occ_sensor_type_db),
            "occ_threshold": float(occ_threshold),
            "start_day": start_day,
            "end_day": end_day,
            "start": start_dt,
            "end": end_dt,
            "routes": routes,
            "directions": directions,
        },
    )
    if df.empty:
        return 0
    return int(df["volOnLowOcc"].iloc[0])


def fetch_over_cnt(sensor_id, routes, directions, sensor_type_db, start_dt, end_dt) -> int:
    """
    overCnt: for each day in range, count 30s slots where 25 < value < 128; return the maximum daily count.
    Intended for volume (v30).
    """
    start_day = start_dt.date()
    end_day = end_dt.date()

    precomputed = fetch_precomputed_metric("overCnt", sensor_id, start_day, end_day, str(sensor_type_db))
    if precomputed is not None:
        return precomputed

    sql = """
        SELECT ifNull(max(over_cnt), 0) AS overCnt
        FROM
        (
            SELECT r.day AS day, countIf((ifNull(r.value, -1) > 25) AND (ifNull(r.value, -1) < 128)) AS over_cnt
            FROM raw_30s AS r
            INNER JOIN detector_meta AS m ON r.sensor_id = m.sensor_id
            WHERE r.sensor_id = {sensor_id:String}
              AND toString(r.sensor_type) = {sensor_type:String}
              AND r.day >= {start_day:Date} AND r.day <= {end_day:Date}
              AND r.ts >= {start:DateTime} AND r.ts <= {end:DateTime}
              AND m.route IN {routes:Array(String)}
              AND m.direction IN {directions:Array(String)}
            GROUP BY day
        )
    """

    df = ch().query_df(
        sql,
        parameters={
            "sensor_id": str(sensor_id),
            "sensor_type": str(sensor_type_db),
            "start_day": start_day,
            "end_day": end_day,
            "start": start_dt,
            "end": end_dt,
            "routes": routes,
            "directions": directions,
        },
    )
    if df.empty:
        return 0
    return int(df["overCnt"].iloc[0])


def fetch_high_occ(sensor_id, routes, directions, sensor_type_db, start_dt, end_dt) -> int:
    """
    highOcc: for each day in range, count 30s slots where value > 35; return the maximum daily count.
    Intended for occupancy (c30).
    """
    start_day = start_dt.date()
    end_day = end_dt.date()

    precomputed = fetch_precomputed_metric("highOcc", sensor_id, start_day, end_day, str(sensor_type_db))
    if precomputed is not None:
        return precomputed

    sql = """
        SELECT ifNull(max(high_cnt), 0) AS highOcc
        FROM
        (
            SELECT r.day AS day, countIf(ifNull(r.value, -1) > 35) AS high_cnt
            FROM raw_30s AS r
            INNER JOIN detector_meta AS m ON r.sensor_id = m.sensor_id
            WHERE r.sensor_id = {sensor_id:String}
              AND toString(r.sensor_type) = {sensor_type:String}
              AND r.day >= {start_day:Date} AND r.day <= {end_day:Date}
              AND r.ts >= {start:DateTime} AND r.ts <= {end:DateTime}
              AND m.route IN {routes:Array(String)}
              AND m.direction IN {directions:Array(String)}
            GROUP BY day
        )
    """

    df = ch().query_df(
        sql,
        parameters={
            "sensor_id": str(sensor_id),
            "sensor_type": str(sensor_type_db),
            "start_day": start_day,
            "end_day": end_day,
            "start": start_dt,
            "end": end_dt,
            "routes": routes,
            "directions": directions,
        },
    )
    if df.empty:
        return 0
    return int(df["highOcc"].iloc[0])


def fetch_const_vol(sensor_id, routes, directions, sensor_type_db, start_dt, end_dt, min_run_slots: int = 20) -> int:
    """
    constVol: maximum length (in 30s slots) of a consecutive run where volume stays constant
    and 0 < volume < 128. Runs shorter than min_run_slots are ignored (default: 20 slots = 10 minutes).
    Returns the maximum over all days in the range.
    Intended for v30.
    """
    min_run_slots = int(min_run_slots)
    if min_run_slots <= 0:
        raise ValueError("min_run_slots must be >= 1")

    start_day = start_dt.date()
    end_day = end_dt.date()

    if min_run_slots == PRECOMPUTED_MIN_RUN_SLOTS:
        precomputed = fetch_precomputed_metric("constVol", sensor_id, start_day, end_day, str(sensor_type_db))
        if precomputed is not None:
            return precomputed

    sql = f"""
        SELECT ifNull(maxIf(run_len, run_len >= {min_run_slots}), 0) AS constVol
        FROM
        (
            SELECT day, grp, count() AS run_len
            FROM
            (
                SELECT
                    day,
                    ts,
                    ok,
                    sum(toUInt8(boundary)) OVER (PARTITION BY day ORDER BY ts) AS grp
                FROM
                (
                    SELECT
                        r.day AS day,
                        r.ts AS ts,
                        (ifNull(r.value, -1) > 0 AND ifNull(r.value, -1) < 128) AS ok,
                        (
                            ((ifNull(r.value, -1) > 0 AND ifNull(r.value, -1) < 128) != prev_ok)
                            OR (
                                (ifNull(r.value, -1) > 0 AND ifNull(r.value, -1) < 128)
                                AND prev_ok
                                AND (ifNull(r.value, -1) != prev_val)
                            )
                        ) AS boundary
                    FROM
                    (
                        SELECT
                            r.day AS day,
                            r.ts AS ts,
                            r.value AS value,
                            lagInFrame((ifNull(r.value, -1) > 0 AND ifNull(r.value, -1) < 128), 1, 0)
                                OVER (PARTITION BY r.day ORDER BY r.ts) AS prev_ok,
                            lagInFrame(ifNull(r.value, -1), 1, -1)
                                OVER (PARTITION BY r.day ORDER BY r.ts) AS prev_val
                        FROM raw_30s AS r
                        INNER JOIN detector_meta AS m ON r.sensor_id = m.sensor_id
                        WHERE r.sensor_id = {{sensor_id:String}}
                          AND toString(r.sensor_type) = {{sensor_type:String}}
                          AND r.day >= {{start_day:Date}} AND r.day <= {{end_day:Date}}
                          AND r.ts >= {{start:DateTime}} AND r.ts <= {{end:DateTime}}
                          AND m.route IN {{routes:Array(String)}}
                          AND m.direction IN {{directions:Array(String)}}
                    ) AS r
                )
            )
            WHERE ok
            GROUP BY day, grp
        )
    """

    df = ch().query_df(
        sql,
        parameters={
            "sensor_id": str(sensor_id),
            "sensor_type": str(sensor_type_db),
            "start_day": start_day,
            "end_day": end_day,
            "start": start_dt,
            "end": end_dt,
            "routes": routes,
            "directions": directions,
        },
    )
    if df.empty:
        return 0
    return int(df["constVol"].iloc[0])


def fetch_const_occ(sensor_id, routes, directions, sensor_type_db, start_dt, end_dt, min_run_slots: int = 20) -> int:
    """
    constOcc: maximum length (in 30s slots) of a consecutive run where occupancy stays constant
    and 0.2 < occupancy < 100. Runs shorter than min_run_slots are ignored (default: 20 slots = 10 minutes).
    Returns the maximum over all days in the range.
    Intended for c30.
    """
    min_run_slots = int(min_run_slots)
    if min_run_slots <= 0:
        raise ValueError("min_run_slots must be >= 1")

    start_day = start_dt.date()
    end_day = end_dt.date()

    if min_run_slots == PRECOMPUTED_MIN_RUN_SLOTS:
        precomputed = fetch_precomputed_metric("constOcc", sensor_id, start_day, end_day, str(sensor_type_db))
        if precomputed is not None:
            return precomputed

    sql = f"""
        SELECT ifNull(maxIf(run_len, run_len >= {min_run_slots}), 0) AS constOcc
        FROM
        (
            SELECT day, grp, count() AS run_len
            FROM
            (
                SELECT
                    day,
                    ts,
                    ok,
                    sum(toUInt8(boundary)) OVER (PARTITION BY day ORDER BY ts) AS grp
                FROM
                (
                    SELECT
                        r.day AS day,
                        r.ts AS ts,
                        (ifNull(r.value, -1) > 0.2 AND ifNull(r.value, -1) < 100) AS ok,
                        (
                            ((ifNull(r.value, -1) > 0.2 AND ifNull(r.value, -1) < 100) != prev_ok)
                            OR (
                                (ifNull(r.value, -1) > 0.2 AND ifNull(r.value, -1) < 100)
                                AND prev_ok
                                AND (ifNull(r.value, -1) != prev_val)
                            )
                        ) AS boundary
                    FROM
                    (
                        SELECT
                            r.day AS day,
                            r.ts AS ts,
                            r.value AS value,
                            lagInFrame((ifNull(r.value, -1) > 0.2 AND ifNull(r.value, -1) < 100), 1, 0)
                                OVER (PARTITION BY r.day ORDER BY r.ts) AS prev_ok,
                            lagInFrame(ifNull(r.value, -1), 1, -1)
                                OVER (PARTITION BY r.day ORDER BY r.ts) AS prev_val
                        FROM raw_30s AS r
                        INNER JOIN detector_meta AS m ON r.sensor_id = m.sensor_id
                        WHERE r.sensor_id = {{sensor_id:String}}
                          AND toString(r.sensor_type) = {{sensor_type:String}}
                          AND r.day >= {{start_day:Date}} AND r.day <= {{end_day:Date}}
                          AND r.ts >= {{start:DateTime}} AND r.ts <= {{end:DateTime}}
                          AND m.route IN {{routes:Array(String)}}
                          AND m.direction IN {{directions:Array(String)}}
                    ) AS r
                )
            )
            WHERE ok
            GROUP BY day, grp
        )
    """

    df = ch().query_df(
        sql,
        parameters={
            "sensor_id": str(sensor_id),
            "sensor_type": str(sensor_type_db),
            "start_day": start_day,
            "end_day": end_day,
            "start": start_dt,
            "end": end_dt,
            "routes": routes,
            "directions": directions,
        },
    )
    if df.empty:
        return 0
    return int(df["constOcc"].iloc[0])


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
