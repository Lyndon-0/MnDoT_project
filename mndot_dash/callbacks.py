from __future__ import annotations

from datetime import date, datetime, time as dtime

import pandas as pd
import plotly.graph_objects as go

import dash
from dash import Input, Output, State, html, no_update
from dash.exceptions import PreventUpdate

from .backend import (
    choose_bucket_seconds,
    fetch_con_zero_vol,
    fetch_const_vol,
    fetch_const_occ,
    fetch_over_cnt,
    fetch_high_occ,
    fetch_meta_with_presence,
    fetch_neg_vol_cnt,
    fetch_occ_lock_on,
    fetch_precomputed_metrics_for_sensors,
    fetch_raw_counts,
    fetch_ts_joined,
    fetch_vol_on_low_occ,
    fetch_zvol_on_occ,
    meta_cache_key,
    ts_cache_key,
)
from .config import UI2DB_SENSOR
from .ui import (
    build_map_figure,
    build_ts_figure,
    empty_map_figure,
)


def register_callbacks(app, cache) -> None:
    def normalize_corridor(value: str) -> str:
        return " ".join(str(value).replace(",", " ").split())

    def split_corridors(corridors):
        if corridors is None:
            corridors_list = []
        elif isinstance(corridors, str):
            corridors_list = [corridors]
        else:
            corridors_list = list(corridors)

        normalized = [normalize_corridor(c) for c in corridors_list if c]
        normalized = [c for c in normalized if c]

        routes = set()
        directions = set()
        for corridor in normalized:
            parts = corridor.split()
            if len(parts) < 2:
                continue
            routes.add(parts[0])
            directions.add(parts[1])
        return normalized, sorted(routes), sorted(directions)

    def match_corridor_from_location(location_text: str, corridors_norm: list[str]) -> str:
        loc = str(location_text).upper()
        for corridor in corridors_norm:
            parts = corridor.split()
            if len(parts) < 2:
                continue
            route, direction = parts[0].upper(), parts[1].upper()
            if route in loc and direction in loc:
                return corridor
        return ""

    @app.callback(
        Output("df-show-store", "data"),
        Output("map-graph", "figure"),
        Output("validation-alert", "is_open"),
        Output("validation-alert", "children"),
        Input("corridors", "value"),
        Input("sensor-label", "value"),
        Input("date-range", "start_date"),
        Input("date-range", "end_date"),
        Input("threshold-conZeroVol", "value"),
        Input("threshold-negVolCnt", "value"),
        Input("threshold-conZeroOcc", "value"),
        Input("threshold-negOccCnt", "value"),
        Input("threshold-occLockOn", "value"),
        Input("threshold-zvolOnOcc", "value"),
        Input("threshold-overCnt", "value"),
        Input("threshold-highOcc", "value"),
        Input("threshold-constVol", "value"),
        Input("threshold-constOcc", "value"),
        Input("threshold-volOnLowOcc", "value"),
    )
    def update_meta_and_map(
        corridors,
        sensor_label,
        start_date_s,
        end_date_s,
        threshold_con_zero_vol,
        threshold_neg_vol_cnt,
        threshold_con_zero_occ,
        threshold_neg_occ_cnt,
        threshold_occ_lock_on,
        threshold_zvol_on_occ,
        threshold_over_cnt,
        threshold_high_occ,
        threshold_const_vol,
        threshold_const_occ,
        threshold_vol_on_low_occ,
    ):
        corridors_norm, routes, directions = split_corridors(corridors)
        if not corridors_norm:
            return None, empty_map_figure(), True, "No corridors selected."

        if not start_date_s or not end_date_s:
            return None, empty_map_figure(), True, "Start and end dates are required."

        start_date = date.fromisoformat(start_date_s)
        end_date = date.fromisoformat(end_date_s)
        if end_date < start_date:
            return None, empty_map_figure(), True, "End date must be on or after the start date."

        sensor_type_db = UI2DB_SENSOR[sensor_label]
        start_day, end_day = start_date, end_date

        key = meta_cache_key(tuple(sorted(routes)), tuple(sorted(directions)), sensor_type_db, start_day, end_day)
        cached = cache.get(key)

        try:
            if cached is None:
                df_show = fetch_meta_with_presence(routes, directions, sensor_type_db, start_day, end_day)
                for c in ["lat", "lon"]:
                    df_show[c] = pd.to_numeric(df_show[c], errors="coerce")
                df_show["name"] = df_show["name"].astype(str)
                df_show["station"] = df_show["station"].astype(str)
                df_show["location"] = df_show["location"].astype(str)
                df_show["sensor_id"] = df_show["name"]
                df_show["lane"] = pd.to_numeric(df_show["lane"], errors="coerce")
                df_show["has_data"] = df_show["has_data"].astype(bool)

                raw_records = df_show.to_dict("records")
                cache.set(key, raw_records)
            else:
                raw_records = cached

            df_show = pd.DataFrame.from_records(raw_records) if raw_records else pd.DataFrame(
                columns=["sensor_id", "name", "station", "location", "lat", "lon", "lane", "has_data"]
            )

        except Exception as e:
            msg = f"ClickHouse error while loading detectors_meta_info: {e}"
            return None, empty_map_figure(), True, msg

        if not df_show.empty:
            df_show["matched_corridor"] = df_show["location"].apply(
                lambda loc: match_corridor_from_location(loc, corridors_norm)
            )
            df_show = df_show.loc[df_show["matched_corridor"] != ""].reset_index(drop=True)

        if df_show.empty:
            fig = empty_map_figure()
            records = []
            return records, fig, False, ""

        def parse_threshold(value):
            try:
                return float(value)
            except (TypeError, ValueError):
                return float("inf")

        thresholds = {
            "conZeroVol": parse_threshold(threshold_con_zero_vol),
            "negVolCnt": parse_threshold(threshold_neg_vol_cnt),
            "overCnt": parse_threshold(threshold_over_cnt),
            "constVol": parse_threshold(threshold_const_vol),
            "conZeroOcc": parse_threshold(threshold_con_zero_occ),
            "constOcc": parse_threshold(threshold_const_occ),
            "negOccCnt": parse_threshold(threshold_neg_occ_cnt),
            "occLockOn": parse_threshold(threshold_occ_lock_on),
            "highOcc": parse_threshold(threshold_high_occ),
            "zvolOnOcc": parse_threshold(threshold_zvol_on_occ),
            "volOnLowOcc": parse_threshold(threshold_vol_on_low_occ),
        }

        df_show["sensor_id"] = df_show["sensor_id"].astype(str)
        has_data_map = {sid: bool(hd) for sid, hd in zip(df_show["sensor_id"], df_show["has_data"])}
        sensor_ids = df_show["sensor_id"].tolist()
        metrics_map = fetch_precomputed_metrics_for_sensors(sensor_ids, start_day, end_day)

        def sensor_anomalous(sid: str) -> bool:
            if not has_data_map.get(sid, False):
                return False
            mv = metrics_map.get(sid, {})
            metric_values = {
                "conZeroVol": mv.get("conZeroVol", 0),
                "negVolCnt": mv.get("negVolCnt", 0),
                "overCnt": mv.get("overCnt", 0),
                "constVol": mv.get("constVol", 0),
                "conZeroOcc": mv.get("conZeroOcc", 0),
                "constOcc": mv.get("constOcc", 0),
                "negOccCnt": mv.get("negOccCnt", 0),
                "occLockOn": mv.get("occLockOn", 0),
                "highOcc": mv.get("highOcc", 0),
                "zvolOnOcc": mv.get("zvolOnOcc", 0),
                "volOnLowOcc": mv.get("volOnLowOcc", 0),
            }
            exceed_count = sum(1 for key, value in metric_values.items() if value > thresholds[key])
            total_metrics = len(metric_values)
            required = (total_metrics + 1) // 2
            return exceed_count >= required

        df_show["anomalous"] = df_show["sensor_id"].apply(sensor_anomalous)
        df_show["status_label"] = df_show.apply(
            lambda r: "No data" if not r.get("has_data", False) else ("Anomalous" if r["anomalous"] else "Healthy"),
            axis=1,
        )

        fig = build_map_figure(df_show)

        records = df_show.to_dict("records")
        return records, fig, False, ""

    @app.callback(
        Output("ts-modal", "is_open"),
        Output("active-sensor-store", "data"),
        Output("active-signature-store", "data"),
        Output("dismissed-signature-store", "data"),
        Input("map-graph", "clickData"),
        Input("ts-modal", "is_open"),
        State("active-sensor-store", "data"),
        State("active-signature-store", "data"),
        State("dismissed-signature-store", "data"),
    )
    def handle_open_close(
        clickData,
        modal_is_open,
        active_sensor,
        active_sig,
        dismissed_sig,
    ):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate

        trigger = ctx.triggered[0]["prop_id"]

        if trigger == "ts-modal.is_open" and modal_is_open is False:
            if active_sig:
                return False, active_sensor, active_sig, active_sig
            return False, active_sensor, active_sig, dismissed_sig

        if trigger == "map-graph.clickData":
            if not clickData or "points" not in clickData or not clickData["points"]:
                raise PreventUpdate

            cd = clickData["points"][0].get("customdata", None)
            if not cd or len(cd) < 4:
                raise PreventUpdate

            sensor_name = str(cd[0])
            station = "" if cd[1] is None else str(cd[1])
            lat = float(cd[2])
            lon = float(cd[3])
            signature = f"{sensor_name}|{station}|{lat:.6f},{lon:.6f}"
            payload = {"name": sensor_name, "station": station}

            if signature == dismissed_sig:
                return False, active_sensor, active_sig, dismissed_sig
            if signature == active_sig:
                return True, active_sensor, active_sig, dismissed_sig

            return True, payload, signature, dismissed_sig

        raise PreventUpdate

    @app.callback(
        Output("ts-modal-title", "children"),
        Output("ts-modal-meta", "children"),
        Output("ts-modal-status", "children"),
        Output("ts-modal-debug", "children"),
        Output("ts-modal-alert", "is_open"),
        Output("ts-modal-alert", "children"),
        Output("ts-graph", "figure"),
        Output("ts-modal-metrics", "children"),
        Input("ts-modal", "is_open"),
        State("active-sensor-store", "data"),
        State("corridors", "value"),
        State("sensor-label", "value"),
        State("date-range", "start_date"),
        State("date-range", "end_date"),
        State("threshold-conZeroVol", "value"),
        State("threshold-negVolCnt", "value"),
        State("threshold-overCnt", "value"),
        State("threshold-constVol", "value"),
        State("threshold-conZeroOcc", "value"),
        State("threshold-constOcc", "value"),
        State("threshold-negOccCnt", "value"),
        State("threshold-occLockOn", "value"),
        State("threshold-highOcc", "value"),
        State("threshold-zvolOnOcc", "value"),
        State("threshold-volOnLowOcc", "value"),
    )
    def update_ts_modal(
        is_open,
        sensor_payload,
        corridors,
        sensor_label,
        start_date_s,
        end_date_s,
        threshold_con_zero_vol,
        threshold_neg_vol_cnt,
        threshold_over_cnt,
        threshold_const_vol,
        threshold_con_zero_occ,
        threshold_const_occ,
        threshold_neg_occ_cnt,
        threshold_occ_lock_on,
        threshold_high_occ,
        threshold_zvol_on_occ,
        threshold_vol_on_low_occ,
    ):
        if isinstance(sensor_payload, dict):
            sensor_id = str(sensor_payload.get("name", "")).strip()
            sensor_station = str(sensor_payload.get("station", "")).strip()
        else:
            sensor_id = str(sensor_payload or "").strip()
            sensor_station = ""

        if not is_open or not sensor_id:
            return "", "", "", "", False, "", go.Figure(), ""

        corridors_norm, routes, directions = split_corridors(corridors)
        if not corridors_norm:
            return "Time series", "", "", "", True, "A corridor must be selected.", go.Figure(), ""

        start_date = date.fromisoformat(start_date_s)
        end_date = date.fromisoformat(end_date_s)
        if end_date < start_date:
            return "Time series", "", "", "", True, "End date must be on or after the start date.", go.Figure(), ""

        sensor_type_db = UI2DB_SENSOR[sensor_label]
        metrics_sensor_type_db = "v30"
        occ_sensor_type_db = "c30"
        start_dt = datetime.combine(start_date, dtime.min)
        end_dt = datetime.combine(end_date, dtime.max)
        bucket_s = choose_bucket_seconds(start_date, end_date)

        meta_line = (
            f"Detector: {sensor_id} | Station: {sensor_station} | Sensor: {sensor_type_db} | "
            f"Range: {start_date:%Y-%m-%d} → {end_date:%Y-%m-%d} | "
            f"Corridor(s): {', '.join(sorted(corridors_norm))} | "
            f"Bucket: {bucket_s}s"
        )

        try:
            ts_key = ts_cache_key(
                sensor_id,
                tuple(sorted(routes)),
                tuple(sorted(directions)),
                sensor_type_db,
                start_dt,
                end_dt,
                bucket_s,
            )
            cached_ts = cache.get(ts_key)

            raw_rows, nonnull_raw_rows = fetch_raw_counts(sensor_id, routes, directions, sensor_type_db, start_dt, end_dt)

            if cached_ts is None:
                df_ts = fetch_ts_joined(sensor_id, routes, directions, sensor_type_db, start_dt, end_dt, bucket_s)
                cache.set(ts_key, df_ts.to_dict("records"))
            else:
                df_ts = pd.DataFrame.from_records(cached_ts)

            if not df_ts.empty:
                df_ts["ts"] = pd.to_datetime(df_ts["ts"], errors="coerce")
                df_ts["value"] = pd.to_numeric(df_ts["value"], errors="coerce")

            metrics_key_base = ts_cache_key(
                sensor_id,
                tuple(sorted(routes)),
                tuple(sorted(directions)),
                metrics_sensor_type_db,
                start_dt,
                end_dt,
                30,
            )

            metric_key = f"metric:conZeroVol:{metrics_key_base}"
            cached_metric = cache.get(metric_key)
            if cached_metric is None:
                con_zero_vol = fetch_con_zero_vol(
                    sensor_id,
                    routes,
                    directions,
                    metrics_sensor_type_db,
                    start_dt,
                    end_dt,
                )
                cache.set(metric_key, int(con_zero_vol))
            else:
                con_zero_vol = int(cached_metric)

            occ_key_base = ts_cache_key(
                sensor_id,
                tuple(sorted(routes)),
                tuple(sorted(directions)),
                occ_sensor_type_db,
                start_dt,
                end_dt,
                30,
            )
            occ_key = f"metric:conZeroOcc:{occ_key_base}"
            cached_occ = cache.get(occ_key)
            if cached_occ is None:
                con_zero_occ = fetch_con_zero_vol(
                    sensor_id,
                    routes,
                    directions,
                    occ_sensor_type_db,
                    start_dt,
                    end_dt,
                )
                cache.set(occ_key, int(con_zero_occ))
            else:
                con_zero_occ = int(cached_occ)

            const_occ_key = f"metric:constOcc:{occ_key_base}"
            cached_const_occ = cache.get(const_occ_key)
            if cached_const_occ is None:
                const_occ = fetch_const_occ(sensor_id, routes, directions, occ_sensor_type_db, start_dt, end_dt)
                cache.set(const_occ_key, int(const_occ))
            else:
                const_occ = int(cached_const_occ)

            neg_occ_key = f"metric:negOccCnt:{occ_key_base}"
            cached_neg_occ = cache.get(neg_occ_key)
            if cached_neg_occ is None:
                neg_occ_cnt = fetch_neg_vol_cnt(sensor_id, routes, directions, occ_sensor_type_db, start_dt, end_dt)
                cache.set(neg_occ_key, int(neg_occ_cnt))
            else:
                neg_occ_cnt = int(cached_neg_occ)

            lock_on_key = f"metric:occLockOn:{occ_key_base}"
            cached_lock_on = cache.get(lock_on_key)
            if cached_lock_on is None:
                occ_lock_on = fetch_occ_lock_on(sensor_id, routes, directions, occ_sensor_type_db, start_dt, end_dt)
                cache.set(lock_on_key, int(occ_lock_on))
            else:
                occ_lock_on = int(cached_lock_on)

            high_occ_key = f"metric:highOcc:{occ_key_base}"
            cached_high_occ = cache.get(high_occ_key)
            if cached_high_occ is None:
                high_occ = fetch_high_occ(sensor_id, routes, directions, occ_sensor_type_db, start_dt, end_dt)
                cache.set(high_occ_key, int(high_occ))
            else:
                high_occ = int(cached_high_occ)

            over_key = f"metric:overCnt:{metrics_key_base}"
            cached_over = cache.get(over_key)
            if cached_over is None:
                over_cnt = fetch_over_cnt(sensor_id, routes, directions, metrics_sensor_type_db, start_dt, end_dt)
                cache.set(over_key, int(over_cnt))
            else:
                over_cnt = int(cached_over)

            const_key = f"metric:constVol:{metrics_key_base}"
            cached_const = cache.get(const_key)
            if cached_const is None:
                const_vol = fetch_const_vol(sensor_id, routes, directions, metrics_sensor_type_db, start_dt, end_dt)
                cache.set(const_key, int(const_vol))
            else:
                const_vol = int(cached_const)

            z_key_base = ts_cache_key(
                sensor_id,
                tuple(sorted(routes)),
                tuple(sorted(directions)),
                "v30+c30",
                start_dt,
                end_dt,
                30,
            )
            z_key = f"metric:zvolOnOcc:{z_key_base}"
            cached_z = cache.get(z_key)
            if cached_z is None:
                zvol_on_occ = fetch_zvol_on_occ(sensor_id, routes, directions, start_dt, end_dt)
                cache.set(z_key, int(zvol_on_occ))
            else:
                zvol_on_occ = int(cached_z)

            vlo_key = f"metric:volOnLowOcc:{z_key_base}"
            cached_vlo = cache.get(vlo_key)
            if cached_vlo is None:
                vol_on_low_occ = fetch_vol_on_low_occ(sensor_id, routes, directions, start_dt, end_dt)
                cache.set(vlo_key, int(vol_on_low_occ))
            else:
                vol_on_low_occ = int(cached_vlo)

            neg_key = f"metric:negVolCnt:{metrics_key_base}"
            cached_neg = cache.get(neg_key)
            if cached_neg is None:
                neg_vol_cnt = fetch_neg_vol_cnt(sensor_id, routes, directions, metrics_sensor_type_db, start_dt, end_dt)
                cache.set(neg_key, int(neg_vol_cnt))
            else:
                neg_vol_cnt = int(cached_neg)

        except Exception as e:
            return (
                f"Detector {sensor_id}",
                meta_line,
                "",
                "",
                True,
                f"ClickHouse error while loading time series: {e}",
                go.Figure(),
                "",
            )

        def parse_threshold(value):
            try:
                return float(value)
            except (TypeError, ValueError):
                return float("inf")

        thresholds = {
            "conZeroVol": parse_threshold(threshold_con_zero_vol),
            "negVolCnt": parse_threshold(threshold_neg_vol_cnt),
            "overCnt": parse_threshold(threshold_over_cnt),
            "constVol": parse_threshold(threshold_const_vol),
            "conZeroOcc": parse_threshold(threshold_con_zero_occ),
            "constOcc": parse_threshold(threshold_const_occ),
            "negOccCnt": parse_threshold(threshold_neg_occ_cnt),
            "occLockOn": parse_threshold(threshold_occ_lock_on),
            "highOcc": parse_threshold(threshold_high_occ),
            "zvolOnOcc": parse_threshold(threshold_zvol_on_occ),
            "volOnLowOcc": parse_threshold(threshold_vol_on_low_occ),
        }
        metric_values = {
            "conZeroVol": con_zero_vol,
            "negVolCnt": neg_vol_cnt,
            "overCnt": over_cnt,
            "constVol": const_vol,
            "conZeroOcc": con_zero_occ,
            "constOcc": const_occ,
            "negOccCnt": neg_occ_cnt,
            "occLockOn": occ_lock_on,
            "highOcc": high_occ,
            "zvolOnOcc": zvol_on_occ,
            "volOnLowOcc": vol_on_low_occ,
        }
        exceed_count = sum(1 for key, value in metric_values.items() if value > thresholds[key])
        total_metrics = len(metric_values)
        required = (total_metrics + 1) // 2
        is_anomalous = exceed_count >= required

        ts_points = int(len(df_ts))
        ts_points_nonnull = int(df_ts["value"].notna().sum()) if (not df_ts.empty and "value" in df_ts.columns) else 0

        if nonnull_raw_rows <= 0 or ts_points_nonnull <= 0:
            status_color = "#9CA3AF"
            status_text = "No data"
        else:
            status_color = "#DC2626" if is_anomalous else "#16A34A"
            status_text = "Anomalous" if is_anomalous else "Healthy"

        status_component = html.Div(
            [
                html.Span(
                    style={
                        "display": "inline-block",
                        "width": "10px",
                        "height": "10px",
                        "borderRadius": "50%",
                        "backgroundColor": status_color,
                    }
                ),
                html.Span(
                    f"Status: {status_text}",
                    style={"color": status_color, "fontWeight": "600"},
                ),
            ],
            style={"display": "flex", "alignItems": "center", "gap": "8px"},
        )

        debug = (
            f"Debug: raw rows matching filters = {raw_rows:,} ({nonnull_raw_rows:,} non-null) | "
            f"chart points returned (aggregated) = {ts_points:,} ({ts_points_nonnull:,} non-null)"
        )

        if df_ts.empty:
            return (
                f"Detector {sensor_id}",
                meta_line,
                status_component,
                debug,
                True,
                "No time-series rows returned for this sensor under the current filters/range.",
                go.Figure(),
                "",
            )

        if ts_points_nonnull <= 0:
            return (
                f"Detector {sensor_id}",
                meta_line,
                status_component,
                debug,
                True,
                "Time-series rows exist, but all aggregated values are NULL (no non-null readings in this range).",
                go.Figure(),
                "",
            )

        fig = build_ts_figure(df_ts, sensor_type_db)
        if con_zero_vol <= 0:
            con_zero_line = "conZeroVol: 0 (no ≥10 minute all-zero run, v30)"
        else:
            con_zero_line = f"conZeroVol: {con_zero_vol} slots ({con_zero_vol / 2:.1f} minutes, v30)"

        neg_line = f"negVolCnt: {neg_vol_cnt} slots/day (max over range, v30)"
        over_line = f"overCnt: {over_cnt} slots/day (max over range, v30; 25<vol<128)"
        if const_vol <= 0:
            const_line = "constVol: 0 (no ≥10 minute constant run, v30; 0<vol<128)"
        else:
            const_line = f"constVol: {const_vol} slots ({const_vol / 2:.1f} minutes, v30; 0<vol<128)"

        if con_zero_occ <= 0:
            con_occ_line = "conZeroOcc: 0 (no ≥10 minute all-zero run, c30)"
        else:
            con_occ_line = f"conZeroOcc: {con_zero_occ} slots ({con_zero_occ / 2:.1f} minutes, c30)"

        if const_occ <= 0:
            const_occ_line = "constOcc: 0 (no ≥10 minute constant run, c30; 0.2<occ<100)"
        else:
            const_occ_line = f"constOcc: {const_occ} slots ({const_occ / 2:.1f} minutes, c30; 0.2<occ<100)"

        neg_occ_line = f"negOccCnt: {neg_occ_cnt} slots/day (max over range, c30)"
        lock_on_line = f"occLockOn: {occ_lock_on} slots/day (max over range, c30)"
        high_occ_line = f"highOcc: {high_occ} slots/day (max over range, c30; occ>35)"
        zvol_on_occ_line = f"zvolOnOcc: {zvol_on_occ} slots/day (max over range, v30==0 & c30>0)"
        vol_on_low_occ_line = f"volOnLowOcc: {vol_on_low_occ} slots/day (max over range, v30>1 & c30<=0.2)"
        metrics = [
            html.Div(con_zero_line),
            html.Div(neg_line),
            html.Div(over_line),
            html.Div(const_line),
            html.Div(con_occ_line),
            html.Div(const_occ_line),
            html.Div(neg_occ_line),
            html.Div(lock_on_line),
            html.Div(high_occ_line),
            html.Div(zvol_on_occ_line),
            html.Div(vol_on_low_occ_line),
        ]

        return f"Detector {sensor_id}", meta_line, status_component, debug, False, "", fig, metrics
