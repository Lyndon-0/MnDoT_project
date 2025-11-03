"""Streamlit UI for bulk downloading MnDOT detector JSON data."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import List

import streamlit as st

from mndot_api import load_detector_list
from mndot_fetcher import BulkDownloadSummary, fetch_date_range


@st.cache_data(ttl=300, show_spinner=False)
def _detector_ids() -> List[str]:
    df = load_detector_list()
    if "detector_id" not in df.columns:
        raise ValueError("detector_id column missing from detector list")
    ids = df["detector_id"]
    print("ids:", ids)
    return sorted({str(det).strip() for det in ids if str(det).strip()})


st.set_page_config(page_title="MnDOT Data Fetcher", layout="wide")

st.title("MnDOT Detector Data Fetcher")
st.caption(
    "Download MnDOT detector JSON files day-by-day; cached files are reused to avoid redundant API calls."
)

with st.sidebar:
    st.header("Configuration")
    storage_dir_input = st.text_input("Storage directory", value="data/mndot_raw")
    start_date = st.date_input("Start date", value=date(2025, 1, 1))
    end_date = st.date_input("End date", value=date(2025, 1, 1))
    try:
        available_sensor_ids = _detector_ids()
    except Exception as exc:  # noqa: BLE001
        available_sensor_ids = []
        st.caption(f"⚠️ Unable to load detector list: {exc}")

    if available_sensor_ids:
        st.write(f"Using **{len(available_sensor_ids)}** detector IDs from metadata list.")
        with st.expander("Show detector IDs", expanded=False):
            st.code("\n".join(available_sensor_ids), language="text")

    sensor_types = st.multiselect(
        "Sensor types",
        options=["V30", "C30", "S30"],
        help="Restrict downloads to specific 30-second data types.",
    )
    force_download = st.checkbox("Force re-download even if file exists", value=False)
    fetch_button = st.button("Fetch data", type="primary")

storage_dir = Path(storage_dir_input).expanduser().resolve()

status_placeholder = st.empty()
progress_placeholder = st.empty()
log_placeholder = st.container()


def _run_fetch() -> BulkDownloadSummary:
    if not available_sensor_ids:
        raise RuntimeError("No detector IDs available; cannot perform download.")

    storage_dir.mkdir(parents=True, exist_ok=True)

    progress_bar = progress_placeholder.progress(0)
    status_placeholder.info("Preparing download...")

    day_span = (end_date - start_date).days + 1
    total_days = max(day_span, 1)

    def _progress_callback(current_day: date, index: int, total: int) -> None:
        day_offset = (current_day - start_date).days
        if total:
            ratio = (day_offset + (min(index, total) / max(total, 1))) / total_days
        else:
            ratio = max(day_offset / total_days, 0)
        progress_bar.progress(min(100, int(ratio * 100)))
        if total:
            status_placeholder.write(f"{current_day:%Y-%m-%d}: {index}/{total} files processed")
        else:
            status_placeholder.write(f"{current_day:%Y-%m-%d}: scanning directory...")

    summary = fetch_date_range(
        start_date=start_date,
        end_date=end_date,
        base_dir=storage_dir,
        sensor_ids=available_sensor_ids,
        sensor_types=sensor_types,
        force=force_download,
        progress_callback=_progress_callback,
    )

    progress_bar.progress(100)
    status_placeholder.success("Download complete")
    return summary


if fetch_button:
    if end_date < start_date:
        status_placeholder.error("End date must be on or after the start date.")
    else:
        with st.spinner("Fetching data from MnDOT..."):
            summary = _run_fetch()
        st.subheader("Run summary")
        st.json(
            {
                "storage_dir": str(summary.base_dir),
                "days_processed": summary.days_processed,
                "requested_files": summary.requested_files,
                "downloaded": summary.downloaded,
                "skipped": summary.skipped,
                "errors": summary.errors,
            }
        )
        if summary.errors:
            with log_placeholder.expander("Errors", expanded=False):
                for err in summary.errors:
                    st.write(f"- {err}")

st.markdown(
    """
### Storage layout

Files are saved within the selected directory using the pattern `YEAR/yyyymmdd/<sensor>.<type>.json`. Each download attempt is appended to `download_manifest.csv`, so you can audit or resume runs. Cached files are reused unless **Force re-download** is enabled.

### Tips

- The MnDOT listings can expose thousands of files per day. Narrow the sensor IDs or sensor types to reduce network traffic.
- Leave the page open during long runs; progress updates in real time.
- Combine this tool with the monitoring UI to analyse cached data offline.
"""
)
