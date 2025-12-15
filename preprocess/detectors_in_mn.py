import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from geopy.distance import geodesic
import io

# --- Page config ---
st.set_page_config(page_title="Minneapolis Traffic Detectors Filter", layout="wide")
st.title("Filter I-35/I-94/I-494/I-694 Detectors in Minneapolis Area")

# --- Upload CSV ---
uploaded_file = st.file_uploader("Upload detectors_i35_i94_i494_i694.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(f"Loaded {len(df)} detectors.")

    # Minneapolis center (city center approximation)
    minneapolis_center = (44.9778, -93.2650)
    lat_center, lon_center = minneapolis_center

    # --- Radius slider ---
    st.sidebar.header("Filter Settings")
    radius_km = st.sidebar.slider(
        "Radius (km) from Minneapolis center",
        min_value=5.0,
        max_value=50.0,
        value=20.0,
        step=1.0,
        help="Adjust to include the desired area of Minneapolis and immediate suburbs."
    )

    # --- Calculate distance and filter ---
    def dist_to_center(row):
        return geodesic(minneapolis_center, (row['lat'], row['lon'])).km

    df['distance_km'] = df.apply(dist_to_center, axis=1)
    df_filtered = df[df['distance_km'] <= radius_km].copy()

    st.write(f"Detectors within {radius_km} km: **{len(df_filtered)}** out of {len(df)}")

    # --- Create Folium map ---
    m = folium.Map(location=minneapolis_center, zoom_start=10, tiles="CartoDB positron")

    # Add center marker
    folium.Marker(
        location=minneapolis_center,
        popup="Minneapolis Center",
        icon=folium.Icon(color="red", icon="star")
    ).add_to(m)

    # Add radius circle
    folium.Circle(
        location=minneapolis_center,
        radius=radius_km * 1000,  # meters
        color="red",
        fill=True,
        fill_opacity=0.1,
        popup=f"Radius: {radius_km} km"
    ).add_to(m)

    # Add all detectors (gray)
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=(row['lat'], row['lon']),
            radius=4,
            color="gray",
            fill=True,
            fill_opacity=0.7,
            popup=f"{row['detector_name']} ({row['route']} {row['direction']})<br>Dist: {row['distance_km']:.1f} km"
        ).add_to(m)

    # Add filtered detectors (blue)
    for _, row in df_filtered.iterrows():
        folium.CircleMarker(
            location=(row['lat'], row['lon']),
            radius=6,
            color="blue",
            fill=True,
            fill_opacity=0.9,
            popup=f"<b>{row['detector_name']}</b><br>{row['route']} {row['direction']}<br>Lane {row['lane']}<br>Dist: {row['distance_km']:.1f} km"
        ).add_to(m)

    # Display map
    st_folium(m, width=1200, height=600)

    # --- Download filtered CSV ---
    if not df_filtered.empty:
        output = io.BytesIO()
        df_filtered.drop(columns=['distance_km']).to_csv(output, index=False)
        output.seek(0)

        st.download_button(
            label=f"Download filtered CSV ({len(df_filtered)} detectors, radius {radius_km} km)",
            data=output,
            file_name=f"detectors_minneapolis_radius_{radius_km}km.csv",
            mime="text/csv"
        )
    else:
        st.warning("No detectors in selected radius.")

else:
    st.info("Please upload the CSV file to begin.")