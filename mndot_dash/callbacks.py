from __future__ import annotations

from datetime import date, datetime, time as dtime

import pandas as pd
import plotly.graph_objects as go

import dash
from dash import Input, Output, State, no_update
from dash.exceptions import PreventUpdate

from .backend import (
    choose_bucket_seconds,
    fetch_meta_with_presence,
    fetch_raw_count,
    fetch_ts_joined,
    meta_cache_key,
    ts_cache_key,
)
from .config import UI2DB_SENSOR
from .ui import (
    build_map_figure,
    build_ts_figure,
    choose_manual_value,
    empty_map_figure,
    make_debug_caption,
    make_manual_sensor_options,
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

    @app.callback(
        Output("df-show-store", "data"),
        Output("map-graph", "figure"),
        Output("manual-sensor", "options"),
        Output("manual-sensor", "value"),
        Output("debug-caption", "children"),
        Output("validation-alert", "is_open"),
        Output("validation-alert", "children"),
        Input("clear-cache", "n_clicks"),
        Input("corridors", "value"),
        Input("sensor-label", "value"),
        Input("date-range", "start_date"),
        Input("date-range", "end_date"),
        State("active-sensor-store", "data"),
    )
    def update_meta_and_map(clear_cache_n, corridors, sensor_label, start_date_s, end_date_s, active_sensor):
        ctx = dash.callback_context
        if clear_cache_n and any(t.get("prop_id") == "clear-cache.n_clicks" for t in (ctx.triggered or [])):
            cache.clear()
            return no_update, no_update, no_update, no_update, no_update, True, "Cache cleared (server-side)."

        corridors_norm, routes, directions = split_corridors(corridors)
        if not corridors_norm:
            return None, empty_map_figure(), [], None, "", True, "No corridors selected."

        if not start_date_s or not end_date_s:
            return None, empty_map_figure(), [], None, "", True, "Start and end dates are required."

        start_date = date.fromisoformat(start_date_s)
        end_date = date.fromisoformat(end_date_s)
        if end_date < start_date:
            return None, empty_map_figure(), [], None, "", True, "End date must be on or after the start date."

        sensor_type_db = UI2DB_SENSOR[sensor_label]
        start_day, end_day = start_date, end_date

        key = meta_cache_key(tuple(sorted(routes)), tuple(sorted(directions)), sensor_type_db, start_day, end_day)
        cached = cache.get(key)

        try:
            if cached is None:
                df_show = fetch_meta_with_presence(routes, directions, sensor_type_db, start_day, end_day)
                for c in ["lat", "lon"]:
                    df_show[c] = pd.to_numeric(df_show[c], errors="coerce")
                df_show["sensor_id"] = df_show["sensor_id"].astype(str)
                df_show["lane"] = pd.to_numeric(df_show["lane"], errors="coerce")
                df_show["has_data"] = df_show["has_data"].astype(bool)

                raw_records = df_show.to_dict("records")
                cache.set(key, raw_records)
            else:
                raw_records = cached

            df_show = pd.DataFrame.from_records(raw_records) if raw_records else pd.DataFrame(
                columns=["sensor_id", "route", "direction", "lat", "lon", "lane", "has_data"]
            )

        except Exception as e:
            msg = f"ClickHouse error while loading detector_meta: {e}"
            return None, empty_map_figure(), [], None, "", True, msg

        if not df_show.empty:
            selected_set = set(corridors_norm)
            corridor_col = df_show["route"].astype(str).str.strip() + " " + df_show["direction"].astype(str).str.strip()
            df_show = df_show.loc[corridor_col.isin(selected_set)].reset_index(drop=True)

        debug_caption = make_debug_caption(df_show)
        sensor_options = make_manual_sensor_options(df_show)
        manual_value = choose_manual_value(sensor_options, active_sensor)
        fig = build_map_figure(df_show)

        records = df_show.to_dict("records")
        return records, fig, sensor_options, manual_value, debug_caption, False, ""

    @app.callback(
        Output("ts-modal", "is_open"),
        Output("active-sensor-store", "data"),
        Output("active-signature-store", "data"),
        Output("dismissed-signature-store", "data"),
        Input("map-graph", "clickData"),
        Input("manual-open", "n_clicks"),
        Input("ts-modal", "is_open"),
        State("manual-sensor", "value"),
        State("active-sensor-store", "data"),
        State("active-signature-store", "data"),
        State("dismissed-signature-store", "data"),
        State("corridors", "value"),
        State("sensor-label", "value"),
        State("date-range", "start_date"),
        State("date-range", "end_date"),
    )
    def handle_open_close(
        clickData,
        manual_open_n,
        modal_is_open,
        manual_sensor_id,
        active_sensor,
        active_sig,
        dismissed_sig,
        corridors,
        sensor_label,
        start_date_s,
        end_date_s,
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
            if not cd or len(cd) < 3:
                raise PreventUpdate

            sensor_id = str(cd[0])
            lat = float(cd[1])
            lon = float(cd[2])
            signature = f"{sensor_id}|{lat:.6f},{lon:.6f}"

            if signature == dismissed_sig:
                return False, active_sensor, active_sig, dismissed_sig
            if signature == active_sig:
                return True, active_sensor, active_sig, dismissed_sig

            return True, sensor_id, signature, dismissed_sig

        if trigger == "manual-open.n_clicks":
            if not manual_open_n:
                raise PreventUpdate
            if not manual_sensor_id:
                raise PreventUpdate

            sensor_type_db = UI2DB_SENSOR[sensor_label]
            corridors_norm, _, _ = split_corridors(corridors)
            signature = (
                f"manual:{manual_sensor_id}:{sensor_type_db}:{start_date_s}:{end_date_s}:"
                f"{','.join(sorted(corridors_norm or []))}"
            )
            return True, str(manual_sensor_id), signature, dismissed_sig

        raise PreventUpdate

    @app.callback(
        Output("ts-modal-title", "children"),
        Output("ts-modal-meta", "children"),
        Output("ts-modal-debug", "children"),
        Output("ts-modal-alert", "is_open"),
        Output("ts-modal-alert", "children"),
        Output("ts-graph", "figure"),
        Input("ts-modal", "is_open"),
        State("active-sensor-store", "data"),
        State("corridors", "value"),
        State("sensor-label", "value"),
        State("date-range", "start_date"),
        State("date-range", "end_date"),
    )
    def update_ts_modal(is_open, sensor_id, corridors, sensor_label, start_date_s, end_date_s):
        if not is_open or not sensor_id:
            return "", "", "", False, "", go.Figure()

        corridors_norm, routes, directions = split_corridors(corridors)
        if not corridors_norm:
            return "Time series", "", "", True, "A corridor must be selected.", go.Figure()

        start_date = date.fromisoformat(start_date_s)
        end_date = date.fromisoformat(end_date_s)
        if end_date < start_date:
            return "Time series", "", "", True, "End date must be on or after the start date.", go.Figure()

        sensor_type_db = UI2DB_SENSOR[sensor_label]
        start_dt = datetime.combine(start_date, dtime.min)
        end_dt = datetime.combine(end_date, dtime.max)
        bucket_s = choose_bucket_seconds(start_date, end_date)

        meta_line = (
            f"Detector: {sensor_id} | Sensor: {sensor_type_db} | "
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

            raw_rows = fetch_raw_count(sensor_id, routes, directions, sensor_type_db, start_dt, end_dt)

            if cached_ts is None:
                df_ts = fetch_ts_joined(sensor_id, routes, directions, sensor_type_db, start_dt, end_dt, bucket_s)
                df_ts["ts"] = pd.to_datetime(df_ts["ts"])
                cache.set(ts_key, df_ts.to_dict("records"))
            else:
                df_ts = pd.DataFrame.from_records(cached_ts)
                if not df_ts.empty:
                    df_ts["ts"] = pd.to_datetime(df_ts["ts"])

        except Exception as e:
            return (
                f"Detector {sensor_id}",
                meta_line,
                "",
                True,
                f"ClickHouse error while loading time series: {e}",
                go.Figure(),
            )

        debug = f"Debug: raw rows matching filters = {raw_rows:,} | chart points returned (aggregated) = {len(df_ts):,}"

        if df_ts.empty:
            return (
                f"Detector {sensor_id}",
                meta_line,
                debug,
                True,
                "No time-series rows returned for this sensor under the current filters/range.",
                go.Figure(),
            )

        fig = build_ts_figure(df_ts, sensor_type_db)
        return f"Detector {sensor_id}", meta_line, debug, False, "", fig
