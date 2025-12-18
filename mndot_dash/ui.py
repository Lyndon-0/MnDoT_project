from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dash import dcc, html
import dash_bootstrap_components as dbc

from .config import BLUE, CORRIDOR_OPTIONS, GRAY, SENSOR_LABELS


def empty_map_figure(center=(44.97, -93.20), zoom=11) -> go.Figure:
    df0 = pd.DataFrame({"lat": [center[0]], "lon": [center[1]], "status": [""]})
    fig = px.scatter_map(
        df0,
        lat="lat",
        lon="lon",
        zoom=zoom,
        center={"lat": center[0], "lon": center[1]},
        map_style="open-street-map",
        height=700,
    )
    fig.update_traces(marker={"opacity": 0})
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    return fig


def build_layout() -> dbc.Container:
    today = date.today()

    return dbc.Container(
        fluid=True,
        children=[
            dcc.Store(id="df-show-store"),  # holds df_show as records
            dcc.Store(id="active-sensor-store", data=None),
            dcc.Store(id="active-signature-store", data=None),
            dcc.Store(id="dismissed-signature-store", data=None),
            dbc.Row(
                [
                    dbc.Col(
                        width=3,
                        children=[
                            html.H4("Filters"),
                            html.Hr(),
                            dbc.Label("Corridor(s)"),
                            dcc.Dropdown(
                                id="corridors",
                                options=[{"label": c, "value": c} for c in CORRIDOR_OPTIONS],
                                value=CORRIDOR_OPTIONS,
                                multi=True,
                                clearable=False,
                            ),
                            html.Div(style={"height": "10px"}),
                            dbc.Label("Sensor Type"),
                            dcc.Dropdown(
                                id="sensor-label",
                                options=[{"label": s, "value": s} for s in SENSOR_LABELS],
                                value="C30 (occupancy)",
                                clearable=False,
                            ),
                            html.Div(style={"height": "10px"}),
                            dbc.Label("Date Range"),
                            dcc.DatePickerRange(
                                id="date-range",
                                start_date=today,
                                end_date=today,
                                display_format="YYYY-MM-DD",
                                minimum_nights=0,
                            ),
                            html.Div(style={"height": "10px"}),
                            dbc.Button(
                                "Clear cache (debug)",
                                id="clear-cache",
                                color="secondary",
                                outline=True,
                            ),
                            html.Div(style={"height": "10px"}),
                            dbc.Alert(id="validation-alert", color="warning", is_open=False),
                        ],
                    ),
                    dbc.Col(
                        width=9,
                        children=[
                            html.H3("MnDOT Detector Monitor"),
                            html.Div(
                                "Blue markers have data under current filters; gray markers do not. Click a marker to open the time-series dialog.",
                                style={"color": "#6B7280"},
                            ),
                            html.Div(style={"height": "8px"}),
                            html.Div(id="debug-caption", style={"color": "#6B7280"}),
                            html.Hr(),
                            html.H5("Map / Click a sensor (opens time-series panel)"),
                            dcc.Graph(
                                id="map-graph",
                                figure=empty_map_figure(),
                                config={"displayModeBar": False},
                                style={"height": "700px"},
                            ),
                            html.Div(style={"height": "10px"}),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        width=4,
                                        children=[
                                            dbc.Label("Manual open target"),
                                            dcc.Dropdown(
                                                id="manual-sensor",
                                                options=[],
                                                value=None,
                                                clearable=False,
                                            ),
                                            html.Div(style={"height": "8px"}),
                                            dbc.Button(
                                                "Open time-series panel",
                                                id="manual-open",
                                                color="primary",
                                            ),
                                        ],
                                    ),
                                ]
                            ),
                        ],
                    ),
                ]
            ),
            dbc.Modal(
                id="ts-modal",
                is_open=False,
                size="xl",
                backdrop="static",
                children=[
                    dbc.ModalHeader(dbc.ModalTitle(id="ts-modal-title"), close_button=True),
                    dbc.ModalBody(
                        children=[
                            html.Div(id="ts-modal-meta", style={"marginBottom": "8px"}),
                            html.Div(
                                id="ts-modal-debug",
                                style={"color": "#6B7280", "marginBottom": "8px"},
                            ),
                            dbc.Alert(id="ts-modal-alert", color="warning", is_open=False),
                            dcc.Graph(
                                id="ts-graph",
                                figure=go.Figure(),
                                config={"displayModeBar": False},
                            ),
                            html.Div(id="ts-modal-metrics", style={"marginTop": "10px"}),
                        ]
                    ),
                ],
            ),
        ],
    )


def make_debug_caption(df_show: pd.DataFrame) -> str:
    shown = len(df_show)
    with_data = int(df_show["has_data"].sum()) if (shown and "has_data" in df_show.columns) else 0
    return f"Debug: sensors shown={shown:,}, with_data={with_data:,}"


def make_manual_sensor_options(df_show: pd.DataFrame) -> List[Dict[str, str]]:
    if df_show.empty or "sensor_id" not in df_show.columns:
        return []
    return [{"label": sid, "value": sid} for sid in df_show["sensor_id"].astype(str).tolist()]


def choose_manual_value(sensor_options: List[Dict[str, str]], active_sensor: Optional[str]) -> Optional[str]:
    values = {o.get("value") for o in sensor_options}
    if active_sensor and active_sensor in values:
        return active_sensor
    return sensor_options[0]["value"] if sensor_options else None


def build_map_figure(df_show: pd.DataFrame) -> go.Figure:
    if df_show.empty or df_show[["lat", "lon"]].dropna().empty:
        return empty_map_figure()

    center_lat = float(np.nanmean(df_show["lat"].to_numpy(dtype=float)))
    center_lon = float(np.nanmean(df_show["lon"].to_numpy(dtype=float)))
    if np.isnan(center_lat) or np.isnan(center_lon):
        center_lat, center_lon = 44.97, -93.20

    df_plot = df_show.copy()
    df_plot["status"] = np.where(df_plot["has_data"], "has data", "no data")
    lane_str = df_plot["lane"].apply(lambda x: "None" if pd.isna(x) else str(int(x)))
    df_plot["tooltip"] = (
        df_plot["route"].astype(str)
        + " "
        + df_plot["direction"].astype(str)
        + " lane "
        + lane_str
        + " ("
        + df_plot["sensor_id"].astype(str)
        + ") — "
        + df_plot["status"]
    )

    fig = px.scatter_map(
        df_plot,
        lat="lat",
        lon="lon",
        color="status",
        hover_name="tooltip",
        hover_data={
            # "lat": True,
            # "lon": True,
            "status": False,
            "tooltip": False,
        },
        custom_data=["sensor_id", "lat", "lon"],
        zoom=12,
        center={"lat": center_lat, "lon": center_lon},
        map_style="open-street-map",
        height=700,
        color_discrete_map={"has data": BLUE, "no data": GRAY},
    )
    fig.update_traces(marker={"size": 10, "opacity": 0.9})
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend_title_text="")
    return fig


def build_ts_figure(df_ts: pd.DataFrame, sensor_type_db: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_ts["ts"], y=df_ts["value"], mode="lines", name=f"{sensor_type_db} (avg)"))
    fig.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Time",
        yaxis_title=f"{sensor_type_db} (avg over bucket)",
    )
    return fig
