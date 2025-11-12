"""Streamlit UI for bulk downloading MnDOT detector JSON data."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import date, datetime, timedelta
from pathlib import Path
from queue import Empty, SimpleQueue
from threading import Lock
from typing import List, Sequence, Tuple

import streamlit as st

from mndot_api import load_detector_list
from mndot_fetcher import BulkDownloadSummary, fetch_date_range


@st.cache_data(ttl=300, show_spinner=False)
def _detector_ids() -> List[str]:
    df = load_detector_list("/home/MnDoT_project/data/all_detectors_converted.csv")
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
    storage_dir_input = st.text_input("Storage directory", value="/data/pouya_data/mndot_raw_data")
    covid_next_day = date(2020, 3, 12)
    today = date.today()
    start_date = st.date_input("Start date", value=covid_next_day)
    end_date = st.date_input("End date", value=today)
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
        default=["V30", "C30", "S30"],
    )

    parallel_toggle = st.checkbox(
        "Parallelize by year",
        value=False,
        help="Run each year-long chunk in its own background worker. Speeds up downloads but increases load.",
    )
    worker_count = (
        st.slider("Parallel workers", min_value=2, max_value=8, value=4, help="Max simultaneous year chunks.")
        if parallel_toggle
        else 1
    )
    force_download = st.checkbox("Force re-download even if file exists", value=False)
    fetch_button = st.button("Fetch data", type="primary")

storage_dir = Path(storage_dir_input).expanduser().resolve()

status_placeholder = st.empty()
progress_placeholder = st.empty()
log_placeholder = st.empty()
daily_progress_placeholder = st.empty()
error_placeholder = st.container()

_log_messages: List[str] = []
_day_messages: List[str] = []
_log_lock = Lock()
_day_lock = Lock()
_log_queue: SimpleQueue[str] = SimpleQueue()
_day_queue: SimpleQueue[str] = SimpleQueue()
_progress_queue: SimpleQueue[float] = SimpleQueue()


def _emit_log(message: str) -> None:
    timestamp = datetime.utcnow().strftime("%H:%M:%S")
    _log_queue.put(f"[{timestamp}] {message}")


def _record_day_status(entry: str) -> None:
    _day_queue.put(entry)


def _enqueue_progress(ratio: float) -> None:
    _progress_queue.put(max(0.0, min(1.0, ratio)))


def _drain_feedback(progress_bar) -> None:
    drained = False
    while True:
        try:
            message = _log_queue.get_nowait()
        except Empty:
            break
        else:
            drained = True
            with _log_lock:
                _log_messages.append(message)
                if len(_log_messages) > 200:
                    _log_messages.pop(0)
                log_placeholder.text(_log_messages[-1])
    while True:
        try:
            entry = _day_queue.get_nowait()
        except Empty:
            break
        else:
            drained = True
            with _day_lock:
                _day_messages.append(entry)
                if len(_day_messages) > 200:
                    _day_messages.pop(0)
                daily_progress_placeholder.text(
                    "; ".join(_day_messages[-10:])
                )
    while True:
        try:
            ratio = _progress_queue.get_nowait()
        except Empty:
            break
        else:
            drained = True
            progress_bar.progress(min(100, int(ratio * 100)))
    return drained


def _reset_run_state() -> None:
    with _log_lock:
        _log_messages.clear()
        log_placeholder.empty()
    with _day_lock:
        _day_messages.clear()
        daily_progress_placeholder.empty()
    while True:
        try:
            _log_queue.get_nowait()
        except Empty:
            break
    while True:
        try:
            _day_queue.get_nowait()
        except Empty:
            break
    while True:
        try:
            _progress_queue.get_nowait()
        except Empty:
            break


def _chunk_by_year(start: date, end: date) -> List[Tuple[date, date]]:
    """Split the requested window into contiguous year-sized chunks."""
    chunks: List[Tuple[date, date]] = []
    current = start
    while current <= end:
        year_end = date(current.year, 12, 31)
        chunk_end = min(end, year_end)
        chunks.append((current, chunk_end))
        current = chunk_end + timedelta(days=1)
    return chunks


def _merge_summaries(summaries: Sequence[BulkDownloadSummary], base_dir: Path) -> BulkDownloadSummary:
    total = BulkDownloadSummary(
        days_processed=0,
        requested_files=0,
        downloaded=0,
        skipped=0,
        errors=[],
        base_dir=base_dir,
    )
    for summary in summaries:
        total.days_processed += summary.days_processed
        total.requested_files += summary.requested_files
        total.downloaded += summary.downloaded
        total.skipped += summary.skipped
        total.errors.extend(summary.errors)
    return total


def _run_parallel_fetch(max_workers: int) -> BulkDownloadSummary:
    chunks = _chunk_by_year(start_date, end_date)
    if len(chunks) <= 1:
        return _run_sequential_fetch()

    total_days = sum((chunk_end - chunk_start).days + 1 for chunk_start, chunk_end in chunks)
    summaries: List[BulkDownloadSummary] = []
    worker_target = min(max_workers, len(chunks))
    completed_days = 0
    completed_lock = Lock()

    _emit_log(f"Parallel fetch queued: {len(chunks)} chunk(s) spanning {total_days} day(s).")
    progress_bar = progress_placeholder.progress(0)
    status_placeholder.info(f"Launching {worker_target} worker(s) for {len(chunks)} year chunk(s)...")

    def _make_parallel_callback(label: str):
        def _callback(current_day: date, index: int, total: int) -> None:
            nonlocal completed_days
            if index == 0:
                _emit_log(f"[{label}] Downloading data for date {current_day:%Y-%m-%d} ({total} file(s)).")
                _record_day_status(f"{current_day:%Y-%m-%d} — started ({label})")
            elif total and index == total:
                _emit_log(f"[{label}] Completed date {current_day:%Y-%m-%d}.")
                _record_day_status(f"{current_day:%Y-%m-%d} — completed ({label})")
                with completed_lock:
                    completed_days += 1
                    ratio = completed_days / max(total_days, 1)
                _enqueue_progress(ratio)
        return _callback

    with ThreadPoolExecutor(max_workers=worker_target) as executor:
        pending: dict = {}
        for chunk_start, chunk_end in chunks:
            label = f"{chunk_start:%Y-%m-%d} → {chunk_end:%Y-%m-%d}"
            future = executor.submit(
                fetch_date_range,
                chunk_start,
                chunk_end,
                base_dir=storage_dir,
                sensor_ids=available_sensor_ids,
                sensor_types=sensor_types,
                force=force_download,
                progress_callback=_make_parallel_callback(label),
            )
            pending[future] = (chunk_start, chunk_end)
            chunk_days = (chunk_end - chunk_start).days + 1
            _emit_log(f"Worker queued for {label} ({chunk_days} day(s)).")

        while pending:
            done, _ = wait(list(pending.keys()), timeout=0.5, return_when=FIRST_COMPLETED)
            _drain_feedback(progress_bar)
            if not done:
                continue
            for future in done:
                chunk_start, chunk_end = pending.pop(future)
                label = f"{chunk_start:%Y-%m-%d} → {chunk_end:%Y-%m-%d}"
                try:
                    summary = future.result()
                except Exception as exc:  # noqa: BLE001
                    err = f"{label}: {exc}"
                    summaries.append(
                        BulkDownloadSummary(
                            days_processed=0,
                            requested_files=0,
                            downloaded=0,
                            skipped=0,
                            errors=[err],
                            base_dir=storage_dir,
                        )
                    )
                    status_placeholder.error(err)
                    _emit_log(f"Chunk {label} failed: {exc}")
                else:
                    summaries.append(summary)
                    status_placeholder.write(
                        f"Finished {label} ({summary.downloaded} downloaded / {summary.skipped} skipped)"
                    )
                    _emit_log(
                        f"Chunk {label} done: {summary.downloaded} downloaded, {summary.skipped} skipped, {len(summary.errors)} error(s)."
                    )
            _drain_feedback(progress_bar)

    _drain_feedback(progress_bar)
    progress_bar.progress(100)
    return _merge_summaries(summaries, storage_dir)


def _run_sequential_fetch() -> BulkDownloadSummary:
    if not available_sensor_ids:
        raise RuntimeError("No detector IDs available; cannot perform download.")

    progress_bar = progress_placeholder.progress(0)
    status_placeholder.info("Preparing download...")

    day_span = (end_date - start_date).days + 1
    total_days = max(day_span, 1)
    _emit_log(
        f"Sequential fetch starting: {start_date:%Y-%m-%d} → {end_date:%Y-%m-%d} ({total_days} day(s))."
    )
    _drain_feedback(progress_bar)

    completed_days = 0

    def _progress_callback(current_day: date, index: int, total: int) -> None:
        nonlocal completed_days
        fractional = (min(index, total) / max(total, 1)) if total else 0.0
        ratio = (completed_days + fractional) / total_days
        _enqueue_progress(ratio)
        if total:
            status_placeholder.write(
                f"Downloading data for date {current_day:%Y-%m-%d}: {index}/{total} files processed"
            )
        else:
            status_placeholder.write(f"{current_day:%Y-%m-%d}: scanning directory...")
        if index == 0:
            _emit_log(f"Downloading data for date {current_day:%Y-%m-%d} (total {total} file(s)).")
            _record_day_status(f"{current_day:%Y-%m-%d} — started ({total} file(s))")
        elif total and index == total:
            _emit_log(f"Completed download for date {current_day:%Y-%m-%d}.")
            _record_day_status(f"{current_day:%Y-%m-%d} — completed")
            completed_days = min(completed_days + 1, total_days)
        _drain_feedback(progress_bar)

    summary = fetch_date_range(
        start_date=start_date,
        end_date=end_date,
        base_dir=storage_dir,
        sensor_ids=available_sensor_ids,
        sensor_types=sensor_types,
        force=force_download,
        progress_callback=_progress_callback,
    )

    _drain_feedback(progress_bar)
    progress_bar.progress(100)
    status_placeholder.success("Download complete")
    return summary


def _run_fetch(parallel_workers: int) -> BulkDownloadSummary:
    _reset_run_state()
    storage_dir.mkdir(parents=True, exist_ok=True)
    if parallel_workers > 1:
        return _run_parallel_fetch(parallel_workers)
    return _run_sequential_fetch()


if fetch_button:
    if end_date < start_date:
        status_placeholder.error("End date must be on or after the start date.")
    else:
        with st.spinner("Fetching data from MnDOT..."):
            summary = _run_fetch(worker_count)
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
        _emit_log(
            f"Run complete: {summary.downloaded} downloaded, {summary.skipped} skipped, {len(summary.errors)} error(s)."
        )
        if summary.errors:
            with error_placeholder.expander("Errors", expanded=False):
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
