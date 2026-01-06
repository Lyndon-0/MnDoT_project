from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime
from string import Template
from typing import Dict, Iterable, Optional, Tuple

import clickhouse_connect

from config import CH_DB, CH_HOST, CH_PORT, DAILY_METRICS_TABLE


RAW_TABLE = f"{CH_DB}.raw_30s"


@dataclass
class MetricJob:
    name: str
    sensor_type: str
    sql: str
    params: Dict[str, object]


def zero_run_sql(raw_table: str) -> str:
    template = Template(
        """
        SELECT
            sensor_id,
            day,
            toUInt32(ifNull(maxIf(run_len, run_len >= {min_run_slots:Int32}), 0)) AS value
        FROM
        (
            SELECT sensor_id, day, grp, count() AS run_len
            FROM
            (
                SELECT
                    sensor_id,
                    day,
                    ts,
                    is_zero,
                    sum(toUInt8(is_zero != prev_is_zero)) OVER (PARTITION BY sensor_id, day ORDER BY ts) AS grp
                FROM
                (
                    SELECT
                        r.sensor_id AS sensor_id,
                        r.day AS day,
                        r.ts AS ts,
                        (ifNull(r.value, 1) = 0) AS is_zero,
                        lagInFrame((ifNull(r.value, 1) = 0), 1, 0)
                            OVER (PARTITION BY r.sensor_id, r.day ORDER BY r.ts) AS prev_is_zero
                    FROM $raw_table AS r
                    WHERE r.day >= {start_day:Date} AND r.day <= {end_day:Date}
                      AND toString(r.sensor_type) = {sensor_type:String}
                )
            )
            WHERE is_zero
            GROUP BY sensor_id, day, grp
        )
        GROUP BY sensor_id, day
        """
    )
    return template.substitute(raw_table=raw_table)


def const_run_sql(raw_table: str, ok_expr: str) -> str:
    template = Template(
        """
        SELECT
            sensor_id,
            day,
            toUInt32(ifNull(maxIf(run_len, run_len >= {min_run_slots:Int32}), 0)) AS value
        FROM
        (
            SELECT sensor_id, day, grp, count() AS run_len
            FROM
            (
                SELECT
                    sensor_id,
                    day,
                    ts,
                    ok,
                    sum(toUInt8(boundary)) OVER (PARTITION BY sensor_id, day ORDER BY ts) AS grp
                FROM
                (
                    SELECT
                        r.sensor_id AS sensor_id,
                        r.day AS day,
                        r.ts AS ts,
                        ($ok_expr) AS ok,
                        (
                            (($ok_expr) != prev_ok)
                            OR (($ok_expr) AND prev_ok AND (ifNull(r.value, -1) != prev_val))
                        ) AS boundary
                    FROM
                    (
                        SELECT
                            r.sensor_id AS sensor_id,
                            r.day AS day,
                            r.ts AS ts,
                            r.value AS value,
                            lagInFrame(($ok_expr), 1, 0)
                                OVER (PARTITION BY r.sensor_id, r.day ORDER BY r.ts) AS prev_ok,
                            lagInFrame(ifNull(r.value, -1), 1, -1)
                                OVER (PARTITION BY r.sensor_id, r.day ORDER BY r.ts) AS prev_val
                        FROM $raw_table AS r
                        WHERE r.day >= {start_day:Date} AND r.day <= {end_day:Date}
                          AND toString(r.sensor_type) = {sensor_type:String}
                    ) AS r
                )
            )
            WHERE ok
            GROUP BY sensor_id, day, grp
        )
        GROUP BY sensor_id, day
        """
    )
    return template.substitute(raw_table=raw_table, ok_expr=ok_expr)


def simple_daily_count_sql(raw_table: str, condition: str) -> str:
    template = Template(
        """
        SELECT
            r.day AS day,
            r.sensor_id AS sensor_id,
            toUInt32(countIf($condition)) AS value
        FROM $raw_table AS r
        WHERE r.day >= {start_day:Date} AND r.day <= {end_day:Date}
          AND toString(r.sensor_type) = {sensor_type:String}
        GROUP BY r.sensor_id, r.day
        """
    )
    return template.substitute(raw_table=raw_table, condition=condition)


def zvol_on_occ_sql(raw_table: str) -> str:
    template = Template(
        """
        SELECT
            sensor_id,
            day,
            toUInt32(countIf((ifNull(vol, 1) = 0) AND (ifNull(occ, 0) > 0))) AS value
        FROM
        (
            SELECT
                r.sensor_id AS sensor_id,
                r.day AS day,
                r.ts AS ts,
                anyIf(r.value, toString(r.sensor_type) = {vol_type:String}) AS vol,
                anyIf(r.value, toString(r.sensor_type) = {occ_type:String}) AS occ
            FROM $raw_table AS r
            WHERE r.day >= {start_day:Date} AND r.day <= {end_day:Date}
              AND (toString(r.sensor_type) = {vol_type:String} OR toString(r.sensor_type) = {occ_type:String})
            GROUP BY sensor_id, day, ts
        )
        GROUP BY sensor_id, day
        """
    )
    return template.substitute(raw_table=raw_table)


def vol_on_low_occ_sql(raw_table: str) -> str:
    template = Template(
        """
        SELECT
            sensor_id,
            day,
            toUInt32(countIf((ifNull(vol, -1) > 1) AND (ifNull(occ, 999) <= {occ_threshold:Float64}))) AS value
        FROM
        (
            SELECT
                r.sensor_id AS sensor_id,
                r.day AS day,
                r.ts AS ts,
                anyIf(r.value, toString(r.sensor_type) = {vol_type:String}) AS vol,
                anyIf(r.value, toString(r.sensor_type) = {occ_type:String}) AS occ
            FROM $raw_table AS r
            WHERE r.day >= {start_day:Date} AND r.day <= {end_day:Date}
              AND (toString(r.sensor_type) = {vol_type:String} OR toString(r.sensor_type) = {occ_type:String})
            GROUP BY sensor_id, day, ts
        )
        GROUP BY sensor_id, day
        """
    )
    return template.substitute(raw_table=raw_table)


def ensure_metrics_table(client, table: str) -> None:
    table_name = f"{CH_DB}.{table}"
    client.command(
        f"""
        CREATE TABLE IF NOT EXISTS {table_name}
        (
            day Date,
            sensor_id String,
            metric String,
            sensor_type String,
            value UInt32,
            computed_at DateTime DEFAULT now()
        )
        ENGINE = ReplacingMergeTree(computed_at)
        PARTITION BY toYYYYMM(day)
        ORDER BY (metric, sensor_id, day)
        """
    )


def delete_existing_range(client, table: str, start_day: date, end_day: date) -> None:
    client.command(
        f"ALTER TABLE {CH_DB}.{table} DELETE WHERE day >= {{start_day:Date}} AND day <= {{end_day:Date}}",
        parameters={"start_day": start_day, "end_day": end_day},
    )


def insert_metric(client, table: str, metric: MetricJob) -> None:
    wrapped = Template(
        """
        INSERT INTO $table (day, sensor_id, metric, sensor_type, value)
        SELECT day, sensor_id, {metric_value:String} AS metric, {metric_sensor_type:String} AS sensor_type, value
        FROM
        (
            $inner_sql
        )
        """
    ).substitute(table=f"{CH_DB}.{table}", inner_sql=metric.sql)

    params = {
        "metric_value": metric.name,
        "metric_sensor_type": metric.sensor_type,
    }
    params.update(metric.params)

    client.command(wrapped, parameters=params)


def build_jobs(start_day: date, end_day: date, min_run_slots: int, occ_threshold: float) -> Iterable[MetricJob]:
    common_range = {"start_day": start_day, "end_day": end_day}

    zero_sql = zero_run_sql(RAW_TABLE)
    const_vol_expr = "(ifNull(r.value, -1) > 0 AND ifNull(r.value, -1) < 128)"
    const_occ_expr = "(ifNull(r.value, -1) > 0.2 AND ifNull(r.value, -1) < 100)"
    const_vol_sql = const_run_sql(RAW_TABLE, const_vol_expr)
    const_occ_sql = const_run_sql(RAW_TABLE, const_occ_expr)

    yield MetricJob(
        name="conZeroVol",
        sensor_type="v30",
        sql=zero_sql,
        params={**common_range, "min_run_slots": min_run_slots, "sensor_type": "v30"},
    )
    yield MetricJob(
        name="conZeroOcc",
        sensor_type="c30",
        sql=zero_sql,
        params={**common_range, "min_run_slots": min_run_slots, "sensor_type": "c30"},
    )
    yield MetricJob(
        name="constVol",
        sensor_type="v30",
        sql=const_vol_sql,
        params={**common_range, "min_run_slots": min_run_slots, "sensor_type": "v30"},
    )
    yield MetricJob(
        name="constOcc",
        sensor_type="c30",
        sql=const_occ_sql,
        params={**common_range, "min_run_slots": min_run_slots, "sensor_type": "c30"},
    )

    yield MetricJob(
        name="negVolCnt",
        sensor_type="v30",
        sql=simple_daily_count_sql(RAW_TABLE, "(ifNull(r.value, 0) < 0)"),
        params={**common_range, "sensor_type": "v30"},
    )
    yield MetricJob(
        name="negOccCnt",
        sensor_type="c30",
        sql=simple_daily_count_sql(RAW_TABLE, "(ifNull(r.value, 0) < 0)"),
        params={**common_range, "sensor_type": "c30"},
    )
    yield MetricJob(
        name="overCnt",
        sensor_type="v30",
        sql=simple_daily_count_sql(RAW_TABLE, "(ifNull(r.value, -1) > 25) AND (ifNull(r.value, -1) < 128)"),
        params={**common_range, "sensor_type": "v30"},
    )
    yield MetricJob(
        name="highOcc",
        sensor_type="c30",
        sql=simple_daily_count_sql(RAW_TABLE, "(ifNull(r.value, -1) > 35)"),
        params={**common_range, "sensor_type": "c30"},
    )
    yield MetricJob(
        name="occLockOn",
        sensor_type="c30",
        sql=simple_daily_count_sql(RAW_TABLE, "(ifNull(r.value, -1) >= 99) AND (ifNull(r.value, -1) <= 100)"),
        params={**common_range, "sensor_type": "c30"},
    )

    yield MetricJob(
        name="zvolOnOcc",
        sensor_type="v30+c30",
        sql=zvol_on_occ_sql(RAW_TABLE),
        params={**common_range, "vol_type": "v30", "occ_type": "c30"},
    )
    yield MetricJob(
        name="volOnLowOcc",
        sensor_type="v30+c30",
        sql=vol_on_low_occ_sql(RAW_TABLE),
        params={**common_range, "vol_type": "v30", "occ_type": "c30", "occ_threshold": float(occ_threshold)},
    )


def parse_day(value: Optional[str]) -> Optional[date]:
    if value is None:
        return None
    return date.fromisoformat(value)


def detect_range(client, start_day: Optional[date], end_day: Optional[date]) -> Tuple[date, date]:
    if start_day and end_day:
        if end_day < start_day:
            raise ValueError("end-day must be on or after start-day")
        return start_day, end_day

    df = client.query_df("SELECT min(day) AS min_day, max(day) AS max_day FROM raw_30s")
    if df.empty or df["min_day"].isna().iloc[0] or df["max_day"].isna().iloc[0]:
        raise ValueError("raw_30s is empty; nothing to aggregate")

    detected_start = df["min_day"].iloc[0].date()
    detected_end = df["max_day"].iloc[0].date()

    start = start_day or detected_start
    end = end_day or detected_end
    if end < start:
        raise ValueError("end-day must be on or after start-day")
    return start, end


def run_jobs(client, table: str, jobs: Iterable[MetricJob]) -> None:
    for job in jobs:
        print(f"• computing {job.name} ({job.sensor_type})...", end="", flush=True)
        insert_metric(client, table, job)
        print(" done")


def main() -> int:
    parser = argparse.ArgumentParser(description="Precompute daily MnDOT metrics into ClickHouse.")
    parser.add_argument("--table", default=DAILY_METRICS_TABLE, help="Destination ClickHouse table (default: daily_metrics)")
    parser.add_argument("--start-day", dest="start_day", type=parse_day, help="Inclusive start day (YYYY-MM-DD). Defaults to min(raw_30s).")
    parser.add_argument("--end-day", dest="end_day", type=parse_day, help="Inclusive end day (YYYY-MM-DD). Defaults to max(raw_30s).")
    parser.add_argument("--min-run-slots", type=int, default=20, help="Minimum run length (30s slots) for const*/conZero* metrics (default: 20).")
    parser.add_argument("--occ-threshold", type=float, default=0.2, help="Occupancy threshold for volOnLowOcc (default: 0.2).")
    parser.add_argument("--delete-existing", action="store_true",
                        help="Delete existing rows in the target day range before inserting (uses ALTER TABLE ... DELETE).")
    args = parser.parse_args()

    client = clickhouse_connect.get_client(host=CH_HOST, port=CH_PORT, database=CH_DB)
    ensure_metrics_table(client, args.table)

    start_day, end_day = detect_range(client, args.start_day, args.end_day)
    print(f"Aggregating metrics for {start_day} → {end_day} into {CH_DB}.{args.table}")

    if args.delete_existing:
        print("Removing existing rows in range...", end="", flush=True)
        delete_existing_range(client, args.table, start_day, end_day)
        print(" done")

    jobs = build_jobs(start_day, end_day, args.min_run_slots, args.occ_threshold)
    run_jobs(client, args.table, jobs)
    print(f"Finished writing daily metrics to {CH_DB}.{args.table} at {datetime.now():%Y-%m-%d %H:%M:%S}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
