from __future__ import annotations

import argparse
import json
import math
import os
import re
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Mapping

import clickhouse_connect
import pandas as pd


DEFAULT_MAX_TIMESTEPS = 288
DEFAULT_BASE_SECONDS = 30
DEFAULT_TABLE = "raw_30s"


def get_clickhouse_client(
    ch_host: str | None = None,
    ch_port: int | None = None,
    ch_database: str | None = None,
    ch_user: str | None = None,
    ch_password: str | None = None,
):
    host = ch_host or os.environ.get("CH_HOST", "127.0.0.1")
    port = int(ch_port or os.environ.get("CH_PORT", 8123))
    database = ch_database or os.environ.get("CH_DB", "sensors")
    user = ch_user if ch_user is not None else os.environ.get("CH_USER", "")
    password = ch_password if ch_password is not None else os.environ.get("CH_PASSWORD", "")

    kwargs = {}
    if user:
        kwargs["username"] = user
    if password:
        kwargs["password"] = password
    return clickhouse_connect.get_client(host=host, port=port, database=database, **kwargs)


def _coerce_datetime(value: date | datetime | str, *, end_of_day: bool) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.astimezone().replace(tzinfo=None)
        return value.replace(microsecond=0)

    if isinstance(value, date):
        base_t = time(23, 59, 59) if end_of_day else time(0, 0, 0)
        return datetime.combine(value, base_t)

    text = str(value).strip()
    if not text:
        raise ValueError("Datetime input cannot be empty.")

    date_only = bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", text))
    ts = pd.Timestamp(text)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)

    dt = ts.to_pydatetime().replace(microsecond=0)
    if date_only:
        base_t = time(23, 59, 59) if end_of_day else time(0, 0, 0)
        return datetime.combine(dt.date(), base_t)
    return dt


def normalize_time_range(
    start_time: date | datetime | str,
    end_time: date | datetime | str,
) -> tuple[datetime, datetime]:
    start_dt = _coerce_datetime(start_time, end_of_day=False)
    end_dt = _coerce_datetime(end_time, end_of_day=True)
    if end_dt < start_dt:
        raise ValueError(f"end_time must be on or after start_time (got {start_dt} > {end_dt}).")
    return start_dt, end_dt


def choose_bucket_seconds(
    start_time: date | datetime | str,
    end_time: date | datetime | str,
    max_timesteps: int = DEFAULT_MAX_TIMESTEPS,
    base_seconds: int = DEFAULT_BASE_SECONDS,
) -> int:
    start_dt, end_dt = normalize_time_range(start_time, end_time)
    span_seconds = int((end_dt - start_dt).total_seconds()) + 1
    raw_bucket = max(1, math.ceil(span_seconds / max_timesteps))
    return max(base_seconds, int(math.ceil(raw_bucket / base_seconds) * base_seconds))


def _build_bucket_index(start_dt: datetime, end_dt: datetime, bucket_seconds: int) -> pd.DatetimeIndex:
    steps = int((end_dt - start_dt).total_seconds() // bucket_seconds) + 1
    return pd.date_range(start=start_dt, periods=steps, freq=f"{bucket_seconds}s")


def _unique_nonempty(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def station_to_detector_names(
    station_id: str,
    detectors_path: Path = Path("data/detectors.csv"),
    detectors_df: pd.DataFrame | None = None,
) -> list[str]:
    station_text = str(station_id).strip()
    detectors = detectors_df if detectors_df is not None else pd.read_csv(detectors_path, dtype=str)

    station_rows = detectors.loc[detectors["station"].astype(str).str.strip() == station_text].copy()
    lane_type = station_rows["lane_type"].fillna("").astype(str).str.strip().str.lower()
    station_rows = station_rows.loc[~lane_type.isin(["exit", "merge"])]
    return _unique_nonempty(station_rows["name"].astype(str).tolist())


def _query_aggregated_sensor_values(
    sensor_ids: list[str],
    start_dt: datetime,
    end_dt: datetime,
    bucket_seconds: int,
    sensor_type: str,
    *,
    ch_table: str = DEFAULT_TABLE,
    ch_database: str | None = None,
    client=None,
) -> pd.Series:
    bucket_index = _build_bucket_index(start_dt, end_dt, bucket_seconds)
    zeros = pd.Series(0.0, index=bucket_index, dtype=float)

    sensor_ids = _unique_nonempty(sensor_ids)
    if not sensor_ids:
        return zeros

    database_name = ch_database or os.environ.get("CH_DB", "sensors")
    ch_client = client or get_clickhouse_client(ch_database=database_name)

    sql = f"""
        SELECT
            toDateTime(
                toUnixTimestamp({{start:DateTime}})
                + intDiv(
                    toUnixTimestamp(ts) - toUnixTimestamp({{start:DateTime}}),
                    {{bucket:Int32}}
                ) * {{bucket:Int32}}
            ) AS ts_bucket,
            sum(ifNull(value, 0.0)) AS total_value
        FROM {database_name}.{ch_table}
        WHERE sensor_id IN {{sensor_ids:Array(String)}}
          AND toString(sensor_type) = {{sensor_type:String}}
          AND ts >= {{start:DateTime}}
          AND ts <= {{end:DateTime}}
        GROUP BY ts_bucket
        ORDER BY ts_bucket
    """

    df = ch_client.query_df(
        sql,
        parameters={
            "sensor_ids": sensor_ids,
            "sensor_type": str(sensor_type),
            "start": start_dt,
            "end": end_dt,
            "bucket": int(bucket_seconds),
        },
    )
    if df.empty:
        return zeros

    agg = (
        df.assign(
            ts_bucket=pd.to_datetime(df["ts_bucket"]),
            total_value=pd.to_numeric(df["total_value"], errors="coerce").fillna(0.0),
        )
        .groupby("ts_bucket")["total_value"]
        .sum()
    )
    return zeros.add(agg, fill_value=0.0).astype(float).sort_index()


def sensor_vehicle_count(
    station_id: str,
    start_time: date | datetime | str,
    end_time: date | datetime | str,
    *,
    detectors_path: Path = Path("data/detectors.csv"),
    detectors_df: pd.DataFrame | None = None,
    sensor_type: str = "v30",
    bucket_seconds: int | None = None,
    max_timesteps: int = DEFAULT_MAX_TIMESTEPS,
    ch_table: str = DEFAULT_TABLE,
    ch_database: str | None = None,
    client=None,
) -> pd.Series:
    start_dt, end_dt = normalize_time_range(start_time, end_time)
    bucket = bucket_seconds or choose_bucket_seconds(start_dt, end_dt, max_timesteps=max_timesteps)

    node_text = str(station_id).strip()
    if node_text.startswith(("B", "I")):
        raise ValueError(f"Cannot query inferred node {node_text!r} directly from ClickHouse.")

    if node_text and node_text[0].isdigit():
        detector_names = [node_text]
    elif node_text.startswith("S"):
        detector_names = station_to_detector_names(node_text, detectors_path=detectors_path, detectors_df=detectors_df)
    else:
        detector_names = [node_text]

    return _query_aggregated_sensor_values(
        detector_names,
        start_dt,
        end_dt,
        bucket,
        sensor_type,
        ch_table=ch_table,
        ch_database=ch_database,
        client=client,
    )


def _normalize_operation(raw: str) -> str:
    op = str(raw).strip().lower()
    if not op:
        return "identity"
    if op in {"addition", "add", "+", "plus", "sum"}:
        return "addition"
    if op in {"subtraction", "sub", "subtract", "-", "minus"}:
        return "subtraction"
    if op in {"identity", "copy", "alias", "pass", "none"}:
        return "identity"
    raise ValueError(f"Unsupported inferred-node operation: {raw!r}")


def _normalize_rule(node: str, spec: Any) -> dict[str, str | None]:
    if isinstance(spec, str):
        return {"left_node": spec.strip(), "right_node": None, "operation": "identity"}

    left = spec.get("left_node", spec.get("left"))
    right = spec.get("right_node", spec.get("right"))
    op = _normalize_operation(str(spec.get("operation", spec.get("op", "identity"))))
    return {
        "left_node": str(left).strip() if left is not None else None,
        "right_node": str(right).strip() if right is not None else None,
        "operation": op,
    }


def _clean_token(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "<null>", "<na>"}:
        return ""
    return text


def _normalize_node_token(value: Any) -> str:
    text = _clean_token(value)
    if not text:
        return ""
    if re.fullmatch(r"\d+\.0+", text):
        return str(int(float(text)))
    return text


def load_inferred_rules(
    inferred_rules: Mapping[str, Any] | None = None,
    inferred_rules_path: Path | None = None,
) -> dict[str, dict[str, str | None]]:
    merged: dict[str, Any] = {}

    if inferred_rules_path is not None:
        payload = json.loads(Path(inferred_rules_path).read_text())
        if isinstance(payload, Mapping):
            for key, value in payload.items():
                merged[str(key).strip()] = value
        elif isinstance(payload, list):
            for item in payload:
                node = str(item["node"]).strip()
                merged[node] = {
                    "left_node": item.get("left_node", item.get("left")),
                    "right_node": item.get("right_node", item.get("right")),
                    "operation": item.get("operation", item.get("op", "identity")),
                }

    if inferred_rules:
        for key, value in inferred_rules.items():
            merged[str(key).strip()] = value

    normalized: dict[str, dict[str, str | None]] = {}
    for node, spec in merged.items():
        if node:
            normalized[node] = _normalize_rule(node, spec)
    return normalized


def load_virtual_sensor_rules(
    virtual_sensors_path: Path | None = Path("data/virtual_sensors.csv"),
) -> dict[str, dict[str, str | None]]:
    if virtual_sensors_path is None:
        return {}

    path = Path(virtual_sensors_path)
    if not path.exists():
        return {}

    df = pd.read_csv(path, dtype=str)
    if df.empty:
        return {}

    col_map = {str(col).strip().lower(): col for col in df.columns}
    left_col = col_map.get("inferred_left_pkey")
    right_col = col_map.get("inferred_right_pkey")
    op_col = col_map.get("inferred_operator")

    missing = []
    if col_map.get("name") is None:
        missing.append("name")
    if left_col is None:
        missing.append("inferred_left_pkey")
    if right_col is None:
        missing.append("inferred_right_pkey")
    if op_col is None:
        missing.append("inferred_operator")
    if missing:
        raise ValueError(
            f"{path} is missing required virtual-sensor columns: {', '.join(missing)}"
        )

    rules: dict[str, dict[str, str | None]] = {}

    def row_value(row: pd.Series, key: str) -> str:
        col = col_map.get(key)
        if col is None:
            return ""
        return _normalize_node_token(row[col])

    def infer_node_id(row: pd.Series) -> str:
        # `name` is the canonical inferred-node id column in virtual_sensors.csv.
        for key in ("name", "station", "node", "sensor_id"):
            token = row_value(row, key)
            if not token:
                continue
            if token.startswith(("B", "I")):
                return token
            m_vbs = re.fullmatch(r"(?i)VBS(\d+)", token)
            if m_vbs:
                return f"B{m_vbs.group(1)}"

        pkey = (
            row_value(row, "pkey")
            or row_value(row, "sensor_pkey")
            or row_value(row, "virtual_sensor_pkey")
        )
        if not pkey:
            return ""
        if pkey.startswith(("B", "I")):
            return pkey

        sensor_kind = _clean_token(
            row[col_map["sensor_kind"]] if col_map.get("sensor_kind") else ""
        ).lower()
        if pkey[0].isdigit():
            if sensor_kind.startswith("inferred"):
                return f"I{pkey}"
            if sensor_kind.startswith("vbs"):
                return f"B{pkey}"
        return ""

    for _, row in df.iterrows():
        node = infer_node_id(row)
        if not node or not node.startswith(("B", "I")):
            continue

        left_node = _normalize_node_token(row[left_col])
        right_node = _normalize_node_token(row[right_col])
        sensor_kind = _clean_token(
            row[col_map["sensor_kind"]] if col_map.get("sensor_kind") else ""
        ).lower()
        if not left_node and not right_node:
            # For VBS rows without inferred fields, fall back to the explicit virtual sensor id
            # (e.g. VBS23257) so B-nodes can still resolve via sensor_vehicle_count.
            if sensor_kind.startswith("vbs"):
                for key in ("name", "station", "sensor_id"):
                    token = row_value(row, key)
                    if token and token != node and not token.startswith(("B", "I")):
                        left_node = token
                        break
            if not left_node:
                continue

        op_raw = _clean_token(row[op_col]) or ("identity" if not right_node else "subtraction")
        op = _normalize_operation(op_raw)
        if op != "identity" and not right_node:
            op = "identity"

        rules[node] = {
            "left_node": left_node or None,
            "right_node": right_node or None,
            "operation": op,
        }

    return rules


def _infer_rules_from_network(
    network_df: pd.DataFrame,
    existing_rules: Mapping[str, dict[str, str | None]] | None = None,
) -> dict[str, dict[str, str | None]]:
    rules = dict(existing_rules or {})

    origins = network_df["origin"].astype(str).str.strip()
    dests = network_df["dest"].astype(str).str.strip()

    incoming: dict[str, list[str]] = {}
    outgoing: dict[str, list[str]] = {}
    for origin, dest in zip(origins, dests):
        outgoing.setdefault(origin, []).append(dest)
        incoming.setdefault(dest, []).append(origin)

    all_nodes = sorted(set(origins).union(set(dests)))
    for node in all_nodes:
        if not node or node in rules or not node.startswith(("B", "I")):
            continue

        in_nodes = incoming.get(node, [])
        out_nodes = outgoing.get(node, [])

        if node.startswith("I") and not in_nodes and len(out_nodes) == 1:
            merge_node = out_nodes[0]
            merge_in = incoming.get(merge_node, [])
            merge_out = outgoing.get(merge_node, [])
            other_upstreams = [n for n in merge_in if n != node]
            if len(other_upstreams) == 1 and len(merge_out) == 1:
                rules[node] = {
                    "left_node": merge_out[0],
                    "right_node": other_upstreams[0],
                    "operation": "subtraction",
                }
                continue

        if node.startswith("B") and len(out_nodes) == 1:
            rules[node] = {"left_node": out_nodes[0], "right_node": None, "operation": "identity"}
            continue

        if len(in_nodes) == 1:
            rules[node] = {"left_node": in_nodes[0], "right_node": None, "operation": "identity"}
            continue
        if len(out_nodes) == 1:
            rules[node] = {"left_node": out_nodes[0], "right_node": None, "operation": "identity"}

    return rules


def resolve_inferred_node(
    left_node: str,
    right_node: str,
    operation: str,
    start_time: date | datetime | str,
    end_time: date | datetime | str,
    *,
    detectors_path: Path = Path("data/detectors.csv"),
    detectors_df: pd.DataFrame | None = None,
    sensor_type: str = "v30",
    bucket_seconds: int | None = None,
    max_timesteps: int = DEFAULT_MAX_TIMESTEPS,
    ch_table: str = DEFAULT_TABLE,
    ch_database: str | None = None,
    client=None,
    virtual_sensors_path: Path | None = Path("data/virtual_sensors.csv"),
    inferred_rules: Mapping[str, Any] | None = None,
    inferred_rules_path: Path | None = None,
) -> pd.DataFrame:
    start_dt, end_dt = normalize_time_range(start_time, end_time)
    bucket = bucket_seconds or choose_bucket_seconds(start_dt, end_dt, max_timesteps=max_timesteps)
    bucket_index = _build_bucket_index(start_dt, end_dt, bucket)
    rules = load_virtual_sensor_rules(virtual_sensors_path)
    rules.update(load_inferred_rules(inferred_rules, inferred_rules_path))
    cache: dict[str, pd.Series] = {}

    def resolve_node(node: str) -> pd.Series:
        node_text = str(node).strip()
        if node_text in cache:
            return cache[node_text]

        if node_text.startswith(("B", "I")):
            rule = rules.get(node_text)
            if rule is None:
                raise KeyError(
                    f"Missing inferred-node rule for {node_text!r}. "
                    f"Add it to {virtual_sensors_path} with inferred_left_pkey, "
                    f"inferred_right_pkey, inferred_operator."
                )
            op = _normalize_operation(str(rule.get("operation", "identity")))
            left = rule.get("left_node")
            right = rule.get("right_node")
            if op == "identity" and not left:
                raise ValueError(f"Inferred-node rule for {node_text!r} has identity op but missing left_node.")
            if op in {"addition", "subtraction"} and (not left or not right):
                raise ValueError(
                    f"Inferred-node rule for {node_text!r} has {op} op but missing left/right node."
                )

            if op == "identity":
                series = resolve_node(str(left))
            elif op == "addition":
                series = resolve_node(str(left)).add(resolve_node(str(right)), fill_value=0.0)
            else:
                series = resolve_node(str(left)).sub(resolve_node(str(right)), fill_value=0.0)
            series = series.clip(lower=0.0)
        else:
            series = sensor_vehicle_count(
                node_text,
                start_dt,
                end_dt,
                detectors_path=detectors_path,
                detectors_df=detectors_df,
                sensor_type=sensor_type,
                bucket_seconds=bucket,
                max_timesteps=max_timesteps,
                ch_table=ch_table,
                ch_database=ch_database,
                client=client,
            )

        out = series.reindex(bucket_index, fill_value=0.0).astype(float)
        cache[node_text] = out
        return out

    left_series = resolve_node(str(left_node))
    right_series = resolve_node(str(right_node))

    operation_norm = _normalize_operation(operation)
    if operation_norm == "addition":
        result = left_series.add(right_series, fill_value=0.0)
    else:
        result = left_series.sub(right_series, fill_value=0.0)

    result = result.clip(lower=0.0).rename("v30")
    return result.reset_index().rename(columns={"index": "ts"})


def export_flow_matrix(
    start_time: date | datetime | str,
    end_time: date | datetime | str,
    network_path: Path = Path("data/network.dat"),
    *,
    output_path: Path = Path("data/flow_matrix.txt"),
    detectors_path: Path = Path("data/detectors.csv"),
    detectors_df: pd.DataFrame | None = None,
    sensor_type: str = "v30",
    bucket_seconds: int | None = None,
    max_timesteps: int = DEFAULT_MAX_TIMESTEPS,
    value_node_col: str = "origin",
    ch_table: str = DEFAULT_TABLE,
    ch_database: str | None = None,
    client=None,
    virtual_sensors_path: Path | None = Path("data/virtual_sensors.csv"),
    inferred_rules: Mapping[str, Any] | None = None,
    inferred_rules_path: Path | None = None,
    auto_infer_rules: bool = True,
) -> pd.DataFrame:
    start_dt, end_dt = normalize_time_range(start_time, end_time)
    bucket = bucket_seconds or choose_bucket_seconds(start_dt, end_dt, max_timesteps=max_timesteps)
    bucket_index = _build_bucket_index(start_dt, end_dt, bucket)

    network_df = pd.read_csv(network_path, sep=r"\s+", engine="python", dtype=str)
    network_df["origin"] = network_df["origin"].astype(str).str.strip()
    network_df["dest"] = network_df["dest"].astype(str).str.strip()
    node_values = network_df[value_node_col].astype(str).str.strip()

    # For edges with exactly one inferred endpoint, use the non-inferred endpoint
    # as the flow source. This avoids zeroing rows when B/I nodes do not map to
    # direct sensor ids (for example, B23298 -> 2605 should use 2605).
    origin_is_inferred = network_df["origin"].str.startswith(("B", "I"))
    dest_is_inferred = network_df["dest"].str.startswith(("B", "I"))
    one_inferred = origin_is_inferred ^ dest_is_inferred
    if one_inferred.any():
        non_inferred_end = network_df["origin"].where(~origin_is_inferred, network_df["dest"])
        node_values = node_values.where(~one_inferred, non_inferred_end)

    detectors_local = detectors_df if detectors_df is not None else pd.read_csv(detectors_path, dtype=str)
    rules = load_virtual_sensor_rules(virtual_sensors_path)
    rules.update(load_inferred_rules(inferred_rules, inferred_rules_path))
    if auto_infer_rules:
        rules = _infer_rules_from_network(network_df, existing_rules=rules)

    cache: dict[str, pd.Series] = {}

    def resolve_node(node: str) -> pd.Series:
        node_text = str(node).strip()
        if node_text in cache:
            return cache[node_text]

        if node_text.startswith(("B", "I")):
            rule = rules.get(node_text)
            if rule is None:
                raise KeyError(
                    f"Missing inferred-node rule for {node_text!r}. "
                    f"Add it to {virtual_sensors_path} with inferred_left_pkey, "
                    f"inferred_right_pkey, inferred_operator."
                )
            op = _normalize_operation(str(rule.get("operation", "identity")))
            left = rule.get("left_node")
            right = rule.get("right_node")
            if op == "identity" and not left:
                raise ValueError(f"Inferred-node rule for {node_text!r} has identity op but missing left_node.")
            if op in {"addition", "subtraction"} and (not left or not right):
                raise ValueError(
                    f"Inferred-node rule for {node_text!r} has {op} op but missing left/right node."
                )

            if op == "identity":
                series = resolve_node(str(left))
            elif op == "addition":
                series = resolve_node(str(left)).add(resolve_node(str(right)), fill_value=0.0)
            else:
                series = resolve_node(str(left)).sub(resolve_node(str(right)), fill_value=0.0)
            series = series.clip(lower=0.0)
        else:
            series = sensor_vehicle_count(
                node_text,
                start_dt,
                end_dt,
                detectors_path=detectors_path,
                detectors_df=detectors_local,
                sensor_type=sensor_type,
                bucket_seconds=bucket,
                max_timesteps=max_timesteps,
                ch_table=ch_table,
                ch_database=ch_database,
                client=client,
            )

        out = series.reindex(bucket_index, fill_value=0.0).astype(float)
        cache[node_text] = out
        return out

    row_vectors = [resolve_node(node).tolist() for node in node_values]
    matrix_df = pd.DataFrame(row_vectors, columns=[f"t{i:03d}" for i in range(len(bucket_index))])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_df.to_csv(output_path, sep="\t", index=False, header=False)

    metadata = network_df[["origin", "dest"]].copy()
    return pd.concat([metadata, matrix_df], axis=1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export link flow matrix from ClickHouse raw_30s.")
    parser.add_argument("--start", required=True, help="Start datetime/date. Examples: 2020-03-12 or 2020-03-12 00:00:00")
    parser.add_argument("--end", required=True, help="End datetime/date. Examples: 2020-03-13 or 2020-03-13 23:59:59")
    parser.add_argument("--network-path", type=Path, default=Path("data/network.dat"))
    parser.add_argument("--output-path", type=Path, default=Path("data/flow_matrix.txt"))
    parser.add_argument("--detectors-path", type=Path, default=Path("data/detectors.csv"))
    parser.add_argument("--sensor-type", default="v30")
    parser.add_argument("--bucket-seconds", type=int, default=None)
    parser.add_argument("--max-timesteps", type=int, default=DEFAULT_MAX_TIMESTEPS)
    parser.add_argument("--value-node-col", default="origin", choices=["origin", "dest"])
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--database", default=None)
    parser.add_argument("--virtual-sensors-path", type=Path, default=Path("data/virtual_sensors.csv"))
    parser.add_argument("--inferred-rules-path", type=Path, default=None)
    parser.add_argument(
        "--disable-auto-infer-rules",
        action="store_true",
        help="Disable network-based fallback rules for unresolved B*/I* nodes.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out = export_flow_matrix(
        args.start,
        args.end,
        args.network_path,
        output_path=args.output_path,
        detectors_path=args.detectors_path,
        sensor_type=args.sensor_type,
        bucket_seconds=args.bucket_seconds,
        max_timesteps=args.max_timesteps,
        value_node_col=args.value_node_col,
        ch_table=args.table,
        ch_database=args.database,
        virtual_sensors_path=args.virtual_sensors_path,
        inferred_rules_path=args.inferred_rules_path,
        auto_infer_rules=not args.disable_auto_infer_rules,
    )
    timestep_cols = [c for c in out.columns if c.startswith("t")]
    print(f"Wrote {len(out)} rows to {args.output_path} with {len(timestep_cols)} timesteps (node column: {args.value_node_col}).")


if __name__ == "__main__":
    main()
