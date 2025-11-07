import streamlit as st, pandas as pd, altair as alt
from streamlit_folium import st_folium
import folium
import numpy as np  
from datetime import datetime, time  

from mndot_api import load_detector_list, fetch_timeseries, rule_flags

st.set_page_config(page_title="MnDOT Detector Monitor — I-94", layout="wide")
st.title("MnDOT Detector Monitor — I-94")
st.caption("Click a sensor on the map → fetch 30-sec data → aggregate to 5-min + basic rule checks (ready for MnDOT real-time API).")

# Maintain UI state across interactions
if "clicked_id" not in st.session_state:
    st.session_state.clicked_id = None
if "show_ts_modal" not in st.session_state:
    st.session_state.show_ts_modal = False
if "last_click_signature" not in st.session_state:
    st.session_state.last_click_signature = None
if "dismissed_signature" not in st.session_state:
    st.session_state.dismissed_signature = None

# ======================
# helpers
# ======================
@st.cache_data(ttl=60)
def agg_5min(df_30s: pd.DataFrame, value_col: str = "value") -> pd.DataFrame:
    """Aggregate 30-second series to 5-min mean. Output: [ts, val]."""
    if df_30s.empty:
        return pd.DataFrame(columns=["ts", "val"])
    out = (
        df_30s.set_index("ts")[value_col]
        .resample("5min")
        .mean()
        .reset_index()
        .rename(columns={value_col: "val"})
    )
    return out


def _close_ts_panel() -> None:
    """Reset modal state when the time-series dialog is dismissed."""
    st.session_state.show_ts_modal = False
    st.session_state.dismissed_signature = st.session_state.last_click_signature
    st.session_state.last_click_signature = None


@st.cache_data(ttl=60, show_spinner=False)
def sensor_has_records(detector_id: str, start_dt: datetime, end_dt: datetime, sensor_key: str) -> bool:
    """Return True if MnDOT API yields any rows for the detector/time window."""
    df = fetch_timeseries(str(detector_id), start_dt, end_dt, sensor_type=sensor_key)
    return not df.empty


@st.cache_data(ttl=30, show_spinner=False)
def fetch_anomaly_flags(sensor_ids: list[str], start_dt: datetime, end_dt: datetime, sensor_key: str) -> dict[str, int]:
    """
    Placeholder for backend anomaly API.
    Replace the body with a real request that returns 0/1 flags per sensor.
    """
    if not sensor_ids:
        return {}
    flags = np.random.randint(0, 2, size=len(sensor_ids))
    return {sid: int(flag) for sid, flag in zip(sensor_ids, flags)}

@st.cache_data(ttl=60, show_spinner=False)
def build_heatmap_long(df_meta_subset: pd.DataFrame,
                       start_dt: datetime, end_dt: datetime,
                       sensor_key: str, max_det: int = 40) -> pd.DataFrame:
    """
    Build long-form table for time–space heatmap using a start/end window.
    Columns: ['time','order','detector_id','value'] where 'time' is 5min timestamp.
    Ordering heuristic:
      - if 'order' column exists -> use it;
      - else choose the axis with larger spread (lon or lat) and sort by it.
    """
    dets = df_meta_subset.copy()
    if "order" in dets.columns:
        dets = dets.sort_values("order")
    else:
        if dets["lon"].std() >= dets["lat"].std():
            dets = dets.sort_values("lon")
        else:
            dets = dets.sort_values("lat")

    frames = []
    # limit to avoid hammering upstream
    dets = dets.head(max_det).reset_index(drop=True)


    prog = st.progress(0)

    n = int(len(dets))
    if n == 0:
        prog.empty()
        return pd.DataFrame(columns=["time", "order", "detector_id", "value"])

    for i, r in dets.iterrows():
        df30 = fetch_timeseries(str(r.detector_id), start_dt, end_dt, sensor_type=sensor_key)
        if df30.empty:

            percent = min(100, max(0, int(round((i + 1) * 100 / n))))
            prog.progress(percent)
            continue

        df5 = agg_5min(df30)
        if df5.empty:
            percent = min(100, max(0, int(round((i + 1) * 100 / n))))
            prog.progress(percent)
            continue

        df5 = df5.rename(columns={"ts": "time", "val": "value"})
        df5["detector_id"] = r.detector_id
        df5["order"] = i + 1
        frames.append(df5)


        percent = min(100, max(0, int(round((i + 1) * 100 / n))))
        prog.progress(percent)

    prog.progress(100)
    prog.empty()

    if not frames:
        return pd.DataFrame(columns=["time", "order", "detector_id", "value"])
    return pd.concat(frames, ignore_index=True)

# —— Sidebar filters —— #
with st.sidebar:
    st.header("Filters")
    df_meta = load_detector_list()
    corridors = sorted(df_meta["route"].dropna().unique().tolist())
    route = st.selectbox("Corridor", corridors, index=0 if corridors else None)
    directions = sorted(df_meta[df_meta["route"]==route]["direction"].dropna().unique().tolist()) if route else []
    direction = st.selectbox("Direction", directions, index=0 if directions else None)
    sensor_type = st.selectbox("Sensor Type", ["V30 (volume)","C30 (occupancy)","S30 (speed)"], index=0)
    sensor_key = {"V30 (volume)":"V30", "C30 (occupancy)":"C30", "S30 (speed)":"S30"}[sensor_type]

    today_date = pd.Timestamp.today().date()
    default_start_time = time(7, 0)
    default_end_time = time(9, 0)

    start_col, end_col = st.columns(2)
    with start_col:
        start_date = st.date_input("Start Date", today_date)
        start_time = st.time_input("Start Time", default_start_time, step=60)
    with end_col:
        end_date = st.date_input("End Date", today_date)
        end_time = st.time_input("End Time", default_end_time, step=60)

    start_dt = datetime.combine(start_date, start_time)
    end_dt = datetime.combine(end_date, end_time)

    if end_dt <= start_dt:
        st.warning("End time must be after the start time.")
        st.stop()

# —— Subset of sensors —— #
df_show = df_meta.copy()
if route: df_show = df_show[df_show["route"]==route]
if direction: df_show = df_show[df_show["direction"]==direction]

# ======================
# TABS
# ======================
tab_map, tab_kpi = st.tabs(["Map", "KPI"])

# ======================
# Map Tab
# ======================
with tab_map:
    st.subheader("① Map / Click a sensor (time-series panel opens here)")
    if df_show.empty:
        st.warning("No sensors under current filters (use data/I-94_detectors_converted.csv first; replace with the official list later).")
        m = folium.Map(location=[44.97, -93.20], zoom_start=12)
    else:
        m = folium.Map(location=[df_show["lat"].mean(), df_show["lon"].mean()], zoom_start=12)
        COLOR_ANOMALOUS = "#DC2626"
        COLOR_HEALTHY = "#16A34A"
        COLOR_NODATA = "#9CA3AF"
        COLOR_UNKNOWN = "#FCD34D"

        sensor_ids = df_show["detector_id"].astype(str).tolist()
        anomaly_flags = fetch_anomaly_flags(sensor_ids, start_dt, end_dt, sensor_key)

        for _, r in df_show.iterrows():
            det_id = str(r.detector_id)
            has_data = sensor_has_records(det_id, start_dt, end_dt, sensor_key)
            flag = anomaly_flags.get(det_id)

            if not has_data:
                marker_color = COLOR_NODATA
                status_text = "no recent data"
            else:
                if flag == 1:
                    marker_color = COLOR_ANOMALOUS
                    status_text = "anomalous"
                elif flag == 0:
                    marker_color = COLOR_HEALTHY
                    status_text = "normal"
                else:
                    marker_color = COLOR_UNKNOWN
                    status_text = "anomaly unknown"

            folium.CircleMarker(
                location=[r.lat, r.lon],
                radius=6,
                tooltip=f'{r.name} ({r.detector_id}) — {status_text}',
                color=marker_color,
                fill=True,
                fill_color=marker_color,
                fill_opacity=0.9 if has_data else 0.5,
            ).add_to(m)

    # Stretch the Folium map to use the available viewport height inside the tab
    full_height_css = """
    <style>
        .folium-map {
            height: calc(100vh - 160px) !important;
            width: 100% !important;
        }
    </style>
    """
    m.get_root().header.add_child(folium.Element(full_height_css))

    ret = st_folium(
        m,
        height=700,
        use_container_width=True,
    )

    clicked_id = None
    if ret:
        label = ret.get("last_object_clicked_tooltip") or ret.get("last_object_clicked_popup")
        latlng = ret.get("last_object_clicked") or ret.get("last_clicked")
        if label and "(" in label and ")" in label:
            if isinstance(latlng, dict):
                lat = latlng.get("lat")
                lng = latlng.get("lng")
            elif isinstance(latlng, (list, tuple)) and len(latlng) >= 2:
                lat, lng = latlng[0], latlng[1]
            else:
                lat = lng = None

            if lat is not None and lng is not None:
                signature = f"{label}|{float(lat):.6f},{float(lng):.6f}"
            else:
                signature = label

            dismissed_sig = st.session_state.dismissed_signature

            if signature == dismissed_sig:
                pass
            elif signature != st.session_state.last_click_signature:
                clicked_id = label.split("(")[-1].split(")")[0].strip()
                st.session_state.last_click_signature = signature
                st.session_state.dismissed_signature = None

    if clicked_id:
        st.session_state.clicked_id = clicked_id
        st.session_state.show_ts_modal = True

    default_target = st.session_state.get("clicked_id")
    if not default_target and not df_show.empty:
        default_target = str(df_show["detector_id"].astype(str).iloc[0])

    button_col, _ = st.columns([0.3, 0.7])
    with button_col:
        st.button(
            "Open time-series panel",
            disabled=not default_target,
            key="open_ts_panel",
            on_click=lambda: st.session_state.update({
                "show_ts_modal": True,
            }),
        )

    if st.session_state.get("show_ts_modal") and default_target:
        @st.dialog("\u00A0", width="large", on_dismiss=_close_ts_panel)
        def time_series_dialog():
            target_id = str(default_target) if default_target else None

            if not target_id:
                st.info("Please click a sensor on the Map.")
            else:
                st.write(
                    f"**Detector:** `{target_id}`  | **Metric:** `{sensor_key}`  | "
                    f"**Range:** `{start_dt:%Y-%m-%d %H:%M}` → `{end_dt:%Y-%m-%d %H:%M}`  | "
                    f"**Corridor:** `{route or 'N/A'}-{direction or 'N/A'}`"
                )
                df_30s = fetch_timeseries(str(target_id), start_dt, end_dt, sensor_type=sensor_key)
                if df_30s.empty:
                    st.warning("No data returned: if upstream is not configured yet, mock data will be used; or adjust the time window.")
                else:
                    df_5m = agg_5min(df_30s)

                    line = alt.Chart(df_5m).mark_line().encode(
                        x=alt.X('ts:T', title='Time'),
                        y=alt.Y('val:Q', title=sensor_key),
                        tooltip=[alt.Tooltip('ts:T', title='Time'), alt.Tooltip('val:Q', title=sensor_key)]
                    ).properties(height=280)
                    st.altair_chart(line, use_container_width=True)

                    flags = rule_flags(df_30s)
                    c1,c2,c3 = st.columns(3)
                    c1.metric("Negative values present", "Yes" if flags.get("negative_any") else "No")
                    c2.metric("Flatline present", "Yes" if flags.get("flatline_any") else "No")
                    c3.metric("Zero-value streak present", "Yes" if flags.get("zero_streak") else "No")

                    st.caption("Note: V30=30-sec volume; C30=30-sec occupancy; S30=30-sec speed. Current rules are demo-only; we will add the 14 health metrics and VBS next.")

        time_series_dialog()

# ======================
# ======================
# KPI Tab
# ======================
with tab_kpi:
    st.subheader("② KPI / Health Summary (within window)")
    if df_show.empty:
        st.info("No sensors under current filter.")
    else:
        st.caption("Rule score = negative(1) + flatline(1) + zero-streak(1). Table shows Top N (by score descending).")
        sample_n = st.slider("Number of sensors to check (KPI)", min_value=10, max_value=200, value=50, step=10)
        rows = []
        dets = df_show.head(sample_n)
        prog = st.progress(0)  # 用整数模式

        n = int(len(dets))
        if n == 0:
            prog.empty()
        else:
            for i, r in dets.iterrows():
                df30 = fetch_timeseries(str(r.detector_id), start_dt, end_dt, sensor_type=sensor_key)
                if not df30.empty:
                    f = rule_flags(df30)
                    sev = int(bool(f.get("negative_any"))) + int(bool(f.get("flatline_any"))) + int(bool(f.get("zero_streak")))
                    rows.append({
                        "detector_id": r.detector_id,
                        "name": r.name,
                        "severity": sev,
                        "neg": bool(f.get("negative_any")),
                        "flat": bool(f.get("flatline_any")),
                        "zero": bool(f.get("zero_streak")),
                    })

                percent = min(100, max(0, int(round((i + 1) * 100 / n))))
                prog.progress(percent)

            prog.progress(100)
            prog.empty()


        # KPI cards
        total = len(df_show)
        checked = len(rows)
        anyhit = sum(1 for x in rows if x["severity"] > 0)
        c1, c2, c3 = st.columns(3)
        c1.metric("Sensors in corridor", f"{total}")
        c2.metric("Sensors checked", f"{checked}")
        c3.metric("Sensors with any rule hit", f"{anyhit}")

        if rows:
            df_rank = pd.DataFrame(rows).sort_values(["severity", "neg", "flat", "zero"], ascending=[False]*4)
            st.dataframe(df_rank.head(20), use_container_width=True, height=360)
        else:
            st.info("No usable data fetched, or the window is too short.")
