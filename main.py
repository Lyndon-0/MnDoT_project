import streamlit as st
import pandas as pd
import altair as alt
from streamlit_folium import st_folium
import folium
import numpy as np
from datetime import datetime, time
from typing import Optional, Tuple
import clickhouse_connect

# ----------------------
# Config
# ----------------------
CH_HOST, CH_PORT, CH_DB = "127.0.0.1", 8123, "sensors"

ROUTE_OPTIONS = ["I-94", "I-494", "I-35E", "I-35W", "I-694"]
DIRECTION_OPTIONS = ["EB", "NB", "SB", "WB"]

SENSOR_LABELS = ["V30 (volume)", "C30 (occupancy)", "S30 (speed)"]
UI2DB_SENSOR = {"V30 (volume)": "v30", "C30 (occupancy)": "c30", "S30 (speed)": "s30"}

BLUE = "#2563EB"
GRAY = "#9CA3AF"


# ----------------------
# ClickHouse client
# ----------------------
@st.cache_resource
def ch():
    return clickhouse_connect.get_client(host=CH_HOST, port=CH_PORT, database=CH_DB)


def choose_bucket_seconds(start_date, end_date) -> int:
    days = (end_date - start_date).days + 1
    if days <= 2:
        return 30
    if days <= 14:
        return 300       # 5 min
    if days <= 60:
        return 3600      # 1 hour
    return 86400         # 1 day


# ----------------------
# Queries
# ----------------------
@st.cache_data(show_spinner=False)
def fetch_meta_with_presence(routes, directions, sensor_type_db, start_day, end_day):
    """
    detector_meta + has_data computed in ClickHouse.
    Uses day range for partition pruning.
    """
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


@st.cache_data(show_spinner=False)
def fetch_ts_joined(sensor_id, routes, directions, sensor_type_db, start_dt, end_dt, bucket_s):
    """
    Aggregated time series for a sensor_id under current filters (join in ClickHouse).
    """
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


@st.cache_data(show_spinner=False)
def fetch_raw_count(sensor_id, routes, directions, sensor_type_db, start_dt, end_dt) -> int:
    """
    Debug: count raw_30s rows matching filters (server-side).
    """
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


# -----------------------
# Modal / click state (stable signature)
# -----------------------
if "active_sensor_id" not in st.session_state:
    st.session_state.active_sensor_id = None
if "active_signature" not in st.session_state:
    st.session_state.active_signature = None
if "dismissed_signature" not in st.session_state:
    st.session_state.dismissed_signature = None
if "show_ts_modal" not in st.session_state:
    st.session_state.show_ts_modal = False


def dismiss_ts_modal() -> None:
    st.session_state.show_ts_modal = False
    st.session_state.dismissed_signature = st.session_state.active_signature


def parse_folium_click(ret: dict) -> Tuple[Optional[str], Optional[str]]:
    if not ret:
        return None, None

    label = ret.get("last_object_clicked_tooltip") or ret.get("last_object_clicked_popup")
    latlng = ret.get("last_object_clicked") or ret.get("last_clicked")
    if not label or "(" not in label or ")" not in label:
        return None, None

    a = label.find("(")
    b = label.find(")", a + 1)
    if a == -1 or b == -1 or b <= a:
        return None, None

    sensor_id = label[a + 1 : b].strip()

    lat = lng = None
    if isinstance(latlng, dict):
        lat, lng = latlng.get("lat"), latlng.get("lng")
    elif isinstance(latlng, (list, tuple)) and len(latlng) >= 2:
        lat, lng = latlng[0], latlng[1]

    signature = f"{sensor_id}|{float(lat):.6f},{float(lng):.6f}" if (lat is not None and lng is not None) else sensor_id
    return sensor_id, signature


def handle_sensor_click(sensor_id: str, signature: str) -> None:
    if signature == st.session_state.dismissed_signature:
        return
    if signature == st.session_state.active_signature:
        return
    st.session_state.active_sensor_id = sensor_id
    st.session_state.active_signature = signature
    st.session_state.show_ts_modal = True


def manual_open(sensor_id: str, sensor_type_db: str, start_date, end_date, routes, directions) -> None:
    """
    Manual open bypasses dismissed-signature guard (useful to reopen same sensor after dismiss).
    """
    st.session_state.active_sensor_id = str(sensor_id)
    st.session_state.active_signature = (
        f"manual:{sensor_id}:{sensor_type_db}:{start_date}:{end_date}:"
        f"{','.join(routes)}:{','.join(directions)}"
    )
    st.session_state.show_ts_modal = True


# ----------------------
# Cache slots to avoid re-query / rebuild on modal close
# ----------------------
if "meta_sig" not in st.session_state:
    st.session_state.meta_sig = None
if "df_show" not in st.session_state:
    st.session_state.df_show = None

if "map_sig" not in st.session_state:
    st.session_state.map_sig = None
if "map_obj" not in st.session_state:
    st.session_state.map_obj = None


# ----------------------
# Page chrome
# ----------------------
st.set_page_config(page_title="MnDOT Detector Monitor", layout="wide")
st.title("MnDOT Detector Monitor")
st.caption("Blue markers have data under current filters; gray markers do not. Click a marker to open the time-series dialog.")


# ----------------------
# Sidebar filters
# ----------------------
with st.sidebar:
    st.header("Filters")

    selected_routes = st.multiselect("Corridor(s)", ROUTE_OPTIONS, default=ROUTE_OPTIONS)

    selected_directions = st.multiselect(
        "Direction(s)",
        options=DIRECTION_OPTIONS,
        default=DIRECTION_OPTIONS,
    )

    sensor_label = st.selectbox("Sensor Type", SENSOR_LABELS, index=1)
    sensor_type_db = UI2DB_SENSOR[sensor_label]

    today = pd.Timestamp.today().date()
    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input("Start Date", today)
    with c2:
        end_date = st.date_input("End Date", today)

    if end_date < start_date:
        st.warning("End date must be on or after the start date.")
        st.stop()

    if st.button("Clear cache (debug)"):
        st.cache_data.clear()
        st.cache_resource.clear()
        # also clear session_state caches
        st.session_state.meta_sig = None
        st.session_state.df_show = None
        st.session_state.map_sig = None
        st.session_state.map_obj = None
        st.rerun()

if not selected_routes:
    st.warning("No corridors selected.")
    st.stop()
if not selected_directions:
    st.warning("No directions selected.")
    st.stop()

start_dt = datetime.combine(start_date, time.min)
end_dt = datetime.combine(end_date, time.max)
start_day = start_date
end_day = end_date

# A stable signature for "map data state" (used for caching df_show and the map object)
meta_sig = (
    tuple(sorted(selected_routes)),
    tuple(sorted(selected_directions)),
    sensor_type_db,
    str(start_date),
    str(end_date),
)

# ----------------------
# Load meta + presence ONLY when filters change
# ----------------------
if st.session_state.df_show is None or st.session_state.meta_sig != meta_sig:
    try:
        st.session_state.df_show = fetch_meta_with_presence(
            selected_routes, selected_directions, sensor_type_db, start_day, end_day
        )
        st.session_state.meta_sig = meta_sig
    except Exception as e:
        st.error(f"ClickHouse error while loading detector_meta: {e}")
        st.session_state.df_show = pd.DataFrame(columns=["sensor_id", "route", "direction", "lat", "lon", "lane", "has_data"])
        st.session_state.meta_sig = meta_sig

df_show = st.session_state.df_show

if not df_show.empty and "has_data" in df_show.columns:
    st.caption(f"Debug: sensors shown={len(df_show):,}, with_data={int(df_show['has_data'].sum()):,}")


# ----------------------
# Build Folium map ONLY when df_show (meta_sig) changes
# ----------------------
if st.session_state.map_obj is None or st.session_state.map_sig != meta_sig:
    if df_show.empty:
        m = folium.Map(location=[44.97, -93.20], zoom_start=12)
    else:
        center_lat = float(pd.to_numeric(df_show["lat"], errors="coerce").mean())
        center_lon = float(pd.to_numeric(df_show["lon"], errors="coerce").mean())
        if np.isnan(center_lat) or np.isnan(center_lon):
            center_lat, center_lon = 44.97, -93.20

        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

        for _, r in df_show.iterrows():
            sid = str(r.get("sensor_id", "")).strip()
            try:
                lat, lon = float(r["lat"]), float(r["lon"])
            except Exception:
                continue

            has_data = bool(r.get("has_data", False))
            color = BLUE if has_data else GRAY

            lane = int(r["lane"]) if pd.notna(r.get("lane")) else None
            status = "has data" if has_data else "no data"
            tooltip = f"{r['route']} {r['direction']} lane {lane} ({sid}) — {status}"

            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                tooltip=tooltip,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.9,
            ).add_to(m)

    st.session_state.map_obj = m
    st.session_state.map_sig = meta_sig

m = st.session_state.map_obj


# ----------------------
# Map (click-only returned objects to avoid reruns on zoom/pan)
# ----------------------
st.subheader("Map / Click a sensor (opens time-series panel)")

ret = st_folium(
    m,
    height=700,
    use_container_width=True,
    key="mndot_map",
    returned_objects=[
        "last_object_clicked_tooltip",
        "last_object_clicked_popup",
        "last_object_clicked",
        "last_clicked",
    ],
)

sensor_id, signature = parse_folium_click(ret)
if sensor_id and signature:
    handle_sensor_click(sensor_id, signature)


# ----------------------
# Manual open button (optional; allows reopening after dismiss)
# ----------------------
default_target = None
if st.session_state.active_sensor_id:
    default_target = str(st.session_state.active_sensor_id)
elif not df_show.empty:
    default_target = str(df_show["sensor_id"].astype(str).iloc[0])

btn_col, _ = st.columns([0.3, 0.7])
with btn_col:
    if st.button("Open time-series panel", disabled=not bool(default_target)):
        manual_open(default_target, sensor_type_db, start_date, end_date, selected_routes, selected_directions)


# ----------------------
# Time-series dialog
# Query happens ONLY when modal is open (i.e., only after a click/manual open).
# Closing modal triggers a rerun, but df_show + map are reused; no requery/rebuild.
# ----------------------
if st.session_state.show_ts_modal and st.session_state.active_sensor_id:
    @st.dialog("\u00A0", width="large", on_dismiss=dismiss_ts_modal)
    def time_series_dialog():
        sid = str(st.session_state.active_sensor_id)
        bucket_s = choose_bucket_seconds(start_date, end_date)

        st.write(
            f"**Detector:** `{sid}` | **Sensor:** `{sensor_type_db}` | "
            f"**Range:** `{start_date:%Y-%m-%d}` → `{end_date:%Y-%m-%d}` | "
            f"**Corridor(s):** `{', '.join(selected_routes)}` | "
            f"**Direction(s):** `{', '.join(selected_directions)}` | "
            f"**Bucket:** `{bucket_s}s`"
        )

        # Debug counts (remove if you don't want any extra queries)
        raw_rows = fetch_raw_count(sid, selected_routes, selected_directions, sensor_type_db, start_dt, end_dt)

        df_ts = fetch_ts_joined(
            sid, selected_routes, selected_directions, sensor_type_db, start_dt, end_dt, bucket_s
        )

        st.caption(f"Debug: raw rows matching filters = {raw_rows:,}")
        st.caption(f"Debug: chart points returned (aggregated) = {len(df_ts):,}")

        if df_ts.empty:
            st.warning("No time-series rows returned for this sensor under the current filters/range.")
            return

        df_ts["ts"] = pd.to_datetime(df_ts["ts"])
        chart = (
            alt.Chart(df_ts)
            .mark_line()
            .encode(
                x=alt.X("ts:T", title="Time"),
                y=alt.Y("value:Q", title=f"{sensor_type_db} (avg over bucket)"),
                tooltip=[alt.Tooltip("ts:T", title="Time"), alt.Tooltip("value:Q", title="Value")],
            )
            .properties(height=320)
        )
        st.altair_chart(chart, use_container_width=True)

    time_series_dialog()
