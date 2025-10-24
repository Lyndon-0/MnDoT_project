import streamlit as st, pandas as pd, altair as alt
from streamlit_folium import st_folium
import folium
import numpy as np  
from datetime import datetime  

from mndot_api import load_detector_list, fetch_timeseries, rule_flags

st.set_page_config(page_title="MnDOT Detector Monitor — I-94 MVP", layout="wide")
st.title("MnDOT Detector Monitor — I-94 MVP")
st.caption("Click a sensor on the map → fetch 30-sec data → aggregate to 5-min + basic rule checks (ready for MnDOT real-time API).")

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

@st.cache_data(ttl=60, show_spinner=False)
def build_heatmap_long(df_meta_subset: pd.DataFrame,
                       date_str: str, start_hm: str, end_hm: str,
                       sensor_key: str, max_det: int = 40) -> pd.DataFrame:
    """
    Build long-form table for time–space heatmap:
    columns: ['time','order','detector_id','value'] where 'time' is 5min timestamp.
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
        df30 = fetch_timeseries(str(r.detector_id), date_str, start_hm, end_hm, sensor_type=sensor_key)
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

    date = st.date_input("Date", pd.Timestamp.today().date())
    c1, c2 = st.columns(2)
    start_hm = c1.text_input("Start (HH:MM)", "07:00")
    end_hm   = c2.text_input("End (HH:MM)", "09:00")

# —— Subset of sensors —— #
df_show = df_meta.copy()
if route: df_show = df_show[df_show["route"]==route]
if direction: df_show = df_show[df_show["direction"]==direction]

# ======================
# TABS
# ======================
tab_map, tab_ts, tab_heat, tab_kpi = st.tabs(["Map", "Time Series", "Heatmap", "KPI"])

# ======================
# Map Tab
# ======================
with tab_map:
    st.subheader("① Map / Click a sensor")
    if df_show.empty:
        st.warning("No sensors under current filters (use data/detectors_sample.csv first; replace with the official list later).")
        m = folium.Map(location=[44.97, -93.20], zoom_start=12)
    else:
        m = folium.Map(location=[df_show["lat"].mean(), df_show["lon"].mean()], zoom_start=12)
        for _, r in df_show.iterrows():
            folium.CircleMarker(
                location=[r.lat, r.lon],
                radius=6,
                tooltip=f'{r.name} ({r.detector_id})',
                popup=f'{r.name} ({r.detector_id})',
                color="#2563EB",
                fill=True
            ).add_to(m)

    ret = st_folium(m, height=500)
    clicked_id = None
    if ret and ret.get("last_object_clicked_popup"):
        pop = ret["last_object_clicked_popup"]
        if "(" in pop and ")" in pop:
            clicked_id = pop.split("(")[-1].split(")")[0].strip()
    # Save clicked result into session for other tabs
    if clicked_id:
        st.session_state.clicked_id = clicked_id

# ======================
# Time Series Tab
# ======================
with tab_ts:
    st.subheader("② Time Series (30s → 5-min) & Rule Checks")
    # Read clicked id from Map tab; manual override is allowed
    clicked_id = st.session_state.get("clicked_id")
    manual_id = st.text_input("Enter detector_id manually (overrides map click)", value=clicked_id or "")
    target_id = manual_id or clicked_id or (df_show["detector_id"].astype(str).iloc[0] if not df_show.empty else None)

    if not target_id:
        st.info("Please click a sensor on the Map, or enter a detector_id above.")
    else:
        st.write(f"**Detector:** `{target_id}`  | **Metric:** `{sensor_key}`  | **Corridor:** `{route or 'N/A'}-{direction or 'N/A'}`")
        df_30s = fetch_timeseries(str(target_id), str(date), start_hm, end_hm, sensor_type=sensor_key)
        if df_30s.empty:
            st.warning("No data returned: if upstream is not configured yet, mock data will be used; or adjust the time window.")
        else:
            # 5-min aggregation (use helper for consistency across tabs)
            df_5m = agg_5min(df_30s)

            # Chart
            line = alt.Chart(df_5m).mark_line().encode(
                x=alt.X('ts:T', title='Time'),
                y=alt.Y('val:Q', title=sensor_key),
                tooltip=[alt.Tooltip('ts:T', title='Time'), alt.Tooltip('val:Q', title=sensor_key)]
            ).properties(height=280)
            st.altair_chart(line, use_container_width=True)

            # Rules
            flags = rule_flags(df_30s)
            c1,c2,c3 = st.columns(3)
            c1.metric("Negative values present", "Yes" if flags.get("negative_any") else "No")
            c2.metric("Flatline present", "Yes" if flags.get("flatline_any") else "No")
            c3.metric("Zero-value streak present", "Yes" if flags.get("zero_streak") else "No")

            st.caption("Note: V30=30-sec volume; C30=30-sec occupancy; S30=30-sec speed. Current rules are demo-only; we will add the 14 health metrics and VBS next.")

# ======================
# Heatmap Tab
# ======================
with tab_heat:
    st.subheader("③ Time–Space Heatmap (5-min)")
    st.caption("For the current corridor/direction, fetch a subset of sensors, aggregate, and render. To avoid load, default max 40 sensors.")
    max_det = st.slider("Max sensors to load (Heatmap)", min_value=10, max_value=120, value=40, step=10)
    if df_show.empty:
        st.info("No sensors under current filter; cannot render.")
    else:
        dfh = build_heatmap_long(df_show, str(date), start_hm, end_hm, sensor_key, max_det=max_det)
        if dfh.empty:
            st.warning("Heatmap has no data. Try narrowing the time window or check upstream.")
        else:
            # Optional normalization: z-score per detector for better visual contrast
            norm = st.checkbox("Normalize per detector (z-score)", value=True)
            plot_df = dfh.copy()
            if norm:
                plot_df["value"] = plot_df.groupby("detector_id")["value"].transform(
                    lambda x: (x - x.mean()) / (x.std() + 1e-9)
                )

            heat = (
                alt.Chart(plot_df)
                .mark_rect()
                .encode(
                    x=alt.X("time:T", title="Time (5-min)"),
                    y=alt.Y("order:O", title="Sensor order (along corridor)", sort="ascending"),
                    color=alt.Color("value:Q", title=sensor_key),
                    tooltip=["detector_id:N", "time:T", alt.Tooltip("value:Q", format=".2f")],
                )
                .properties(height=420)
            )
            st.altair_chart(heat, use_container_width=True)

# ======================
# KPI Tab
# ======================
with tab_kpi:
    st.subheader("④ KPI / Health Summary (within window)")
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
                df30 = fetch_timeseries(str(r.detector_id), str(date), start_hm, end_hm, sensor_type=sensor_key)
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
