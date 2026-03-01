from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dash import dcc, html
import dash_bootstrap_components as dbc

from .config import BLUE, CORRIDOR_OPTIONS, GRAY, SENSOR_LABELS


DEFAULT_START_DAY = date(2026, 1, 1)


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
                        className="filters-panel",
                        children=[
                            html.H4("Filters"),
                            html.Hr(),
                            html.Div(
                                className="filter-block",
                                children=[
                                    dbc.Label("Corridor(s)"),
                                    dcc.Dropdown(
                                        id="corridors",
                                        options=[{"label": c, "value": c} for c in CORRIDOR_OPTIONS],
                                        value=CORRIDOR_OPTIONS,
                                        multi=True,
                                        clearable=False,
                                    ),
                                ],
                            ),
                            html.Div(
                                className="filter-block",
                                children=[
                                    dbc.Label("Sensor Type"),
                                    dcc.Dropdown(
                                        id="sensor-label",
                                        options=[{"label": s, "value": s} for s in SENSOR_LABELS],
                                        value="C30 (occupancy)",
                                        clearable=False,
                                    ),
                                ],
                            ),
                            html.Div(
                                className="filter-block",
                                children=[
                                    dbc.Label("Date Range"),
                                    dcc.DatePickerRange(
                                        id="date-range",
                                        start_date=DEFAULT_START_DAY,
                                        end_date=today,
                                        display_format="YYYY-MM-DD",
                                        minimum_nights=0,
                                    ),
                                ],
                            ),
                            dbc.Alert(id="validation-alert", color="warning", is_open=False),
                            dbc.Accordion(
                                [
                                    dbc.AccordionItem(
                                        title="Thresholds",
                                        children=[
                                            dbc.Label("conZeroVol"),
                                            dbc.Input(
                                                id="threshold-conZeroVol",
                                                type="text",
                                                value="2870",
                                            ),
                                            html.Div(style={"height": "6px"}),
                                            dbc.Label("negVolCnt"),
                                            dbc.Input(
                                                id="threshold-negVolCnt",
                                                type="text",
                                                value="1440",
                                            ),
                                            html.Div(style={"height": "6px"}),
                                            dbc.Label("conZeroOcc"),
                                            dbc.Input(
                                                id="threshold-conZeroOcc",
                                                type="text",
                                                value="-1",
                                            ),
                                            html.Div(style={"height": "6px"}),
                                            dbc.Label("negOccCnt"),
                                            dbc.Input(
                                                id="threshold-negOccCnt",
                                                type="text",
                                                value="-1",
                                            ),
                                            html.Div(style={"height": "6px"}),
                                            dbc.Label("occLockOn"),
                                            dbc.Input(
                                                id="threshold-occLockOn",
                                                type="text",
                                                value="2304",
                                            ),
                                            html.Div(style={"height": "6px"}),
                                            dbc.Label("zvolOnOcc"),
                                            dbc.Input(
                                                id="threshold-zvolOnOcc",
                                                type="text",
                                                value="2304",
                                            ),
                                            html.Div(style={"height": "6px"}),
                                            dbc.Label("overCnt"),
                                            dbc.Input(
                                                id="threshold-overCnt",
                                                type="text",
                                                value="2304",
                                            ),
                                            html.Div(style={"height": "6px"}),
                                            dbc.Label("highOcc"),
                                            dbc.Input(
                                                id="threshold-highOcc",
                                                type="text",
                                                value="2592",
                                            ),
                                            html.Div(style={"height": "6px"}),
                                            dbc.Label("constVol"),
                                            dbc.Input(
                                                id="threshold-constVol",
                                                type="text",
                                                value="-1",
                                            ),
                                            html.Div(style={"height": "6px"}),
                                            dbc.Label("constOcc"),
                                            dbc.Input(
                                                id="threshold-constOcc",
                                                type="text",
                                                value="-1",
                                            ),
                                            html.Div(style={"height": "6px"}),
                                            dbc.Label("volOnLowOcc"),
                                            dbc.Input(
                                                id="threshold-volOnLowOcc",
                                                type="text",
                                                value="-1",
                                            ),
                                        ],
                                    )
                                ],
                                start_collapsed=True,
                                flush=True,
                            ),
                        ],
                    ),
                    dbc.Col(
                        width=9,
                        className="main-panel",
                        children=[
                            html.H3("MnDOT Detector Monitor"),
                            html.Div(
                                "Red markers are anomalous, green are healthy, gray have no data under current filters. Click a marker to open the time-series dialog.",
                                style={"color": "#6B7280"},
                            ),
                            html.Div(style={"height": "8px"}),
                            html.Hr(),
                            html.H5("Map / Click a sensor (opens time-series)"),
                            dcc.Graph(
                                id="map-graph",
                                figure=empty_map_figure(),
                                config={"displayModeBar": False},
                                style={"height": "700px"},
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
                                id="ts-modal-status",
                                style={"marginBottom": "8px"},
                            ),
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

def build_map_figure(df_show: pd.DataFrame) -> go.Figure:
    if df_show.empty or df_show[["lat", "lon"]].dropna().empty:
        return empty_map_figure()

    center_lat = float(np.nanmean(df_show["lat"].to_numpy(dtype=float)))
    center_lon = float(np.nanmean(df_show["lon"].to_numpy(dtype=float)))
    if np.isnan(center_lat) or np.isnan(center_lon):
        center_lat, center_lon = 44.97, -93.20

    df_plot = df_show.copy()
    if "status_label" in df_plot.columns:
        df_plot["status"] = df_plot["status_label"]
    else:
        df_plot["status"] = np.where(df_plot["has_data"], "Healthy", "No data")

    if "sensor_id" not in df_plot.columns:
        df_plot["sensor_id"] = df_plot["name"].astype(str) if "name" in df_plot.columns else ""
    if "station" not in df_plot.columns:
        df_plot["station"] = ""
    if "location" not in df_plot.columns:
        route_col = df_plot["route"].astype(str) if "route" in df_plot.columns else ""
        direction_col = df_plot["direction"].astype(str) if "direction" in df_plot.columns else ""
        df_plot["location"] = (route_col + " " + direction_col).astype(str).str.strip()

    lane_col = "lane" if "lane" in df_plot.columns else ("lane_number" if "lane_number" in df_plot.columns else None)
    if lane_col is None:
        lane_str = pd.Series(["None"] * len(df_plot), index=df_plot.index)
    else:
        lane_str = df_plot[lane_col].apply(lambda x: "None" if pd.isna(x) else str(int(x)))

    df_plot["tooltip"] = (
        df_plot["location"].astype(str)
        + " lane "
        + lane_str
        + " ("
        + df_plot["sensor_id"].astype(str)
        + " / "
        + df_plot["station"].astype(str)
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
        custom_data=["sensor_id", "station", "lat", "lon"],
        zoom=12,
        center={"lat": center_lat, "lon": center_lon},
        map_style="open-street-map",
        height=700,
        color_discrete_map={
            "Anomalous": "#DC2626",
            "Healthy": "#16A34A",
            "No data": GRAY,
        },
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
