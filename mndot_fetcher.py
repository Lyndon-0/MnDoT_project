"""Utilities for downloading and caching MnDOT detector data."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional
import requests

BASE_URL = "https://data.dot.state.mn.us/trafdat/metro"
DEFAULT_TIMEOUT = 15
DEFAULT_SENSOR_TYPES = ("V30", "C30", "S30")


class MnDOTFetcherError(Exception):
    """Wrapper for recoverable fetch errors."""


@dataclass
class DownloadResult:
    filename: str
    saved: bool
    skipped: bool
    size: int
    url: str
    path: Path
    error: Optional[str] = None


def _day_url(target_date: date) -> str:
    return f"{BASE_URL}/{target_date:%Y%m%d}/"


def _ensure_storage_dir(base_dir: Path, target_date: date) -> Path:
    dest = base_dir / f"{target_date:%Y}" / f"{target_date:%Y%m%d}"
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def _manifest_path(base_dir: Path) -> Path:
    return base_dir / "download_manifest.csv"


def _append_manifest_row(base_dir: Path, result: DownloadResult) -> None:
    manifest = _manifest_path(base_dir)
    is_new = not manifest.exists()
    with manifest.open("a", newline="") as fh:
        writer = csv.writer(fh)
        if is_new:
            writer.writerow(
                [
                    "timestamp",
                    "filename",
                    "saved",
                    "skipped",
                    "size_bytes",
                    "url",
                    "path",
                    "error",
                ]
            )
        writer.writerow(
            [
                datetime.utcnow().isoformat(timespec="seconds") + "Z",
                result.filename,
                int(result.saved),
                int(result.skipped),
                result.size,
                result.url,
                str(result.path),
                result.error or "",
            ]
        )


def download_sensor_file(
    target_date: date,
    filename: str,
    base_dir: Path,
    session: Optional[requests.Session] = None,
    force: bool = False,
    timeout: int = DEFAULT_TIMEOUT,
) -> DownloadResult:
    """Fetch a single sensor JSON file and persist it locally."""

    sess = session or requests.Session()
    day_dir = _ensure_storage_dir(base_dir, target_date)
    dest = day_dir / filename
    url = _day_url(target_date) + filename

    if dest.exists() and not force:
        return DownloadResult(filename=filename, saved=False, skipped=True, size=dest.stat().st_size, url=url, path=dest)

    resp = sess.get(url, timeout=timeout)
    resp.raise_for_status()

    dest.write_text(resp.text)
    size = dest.stat().st_size

    result = DownloadResult(filename=filename, saved=True, skipped=False, size=size, url=url, path=dest)
    _append_manifest_row(base_dir, result)
    return result


@dataclass
class BulkDownloadSummary:
    days_processed: int
    requested_files: int
    downloaded: int
    skipped: int
    errors: List[str]
    base_dir: Path


def fetch_date_range(
    start_date: date,
    end_date: date,
    base_dir: Path,
    sensor_ids: Optional[Iterable[str]] = None,
    sensor_types: Optional[Iterable[str]] = None,
    force: bool = False,
    progress_callback: Optional[callable] = None,
    session: Optional[requests.Session] = None,
) -> BulkDownloadSummary:
    """Download all files for each day, respecting existing cached files."""

    if end_date < start_date:
        raise ValueError("End date must be on or after start date")

    current = start_date
    days = 0
    requested = 0
    downloaded = 0
    skipped = 0
    errors: List[str] = []

    types = [stype.strip() for stype in (sensor_types or DEFAULT_SENSOR_TYPES)]
    detectors = [sid.strip() for sid in sensor_ids] if sensor_ids is not None else None
    if not detectors:
        raise ValueError("At least one detector_id is required (sensor_ids cannot be empty).")

    sess = session or requests.Session()

    while current <= end_date:
        filenames = [f"{sid}.{stype.lower()}.json" for sid in detectors for stype in types]
        requested += len(filenames)

        if progress_callback:
            progress_callback(current, 0, len(filenames))

        for idx, name in enumerate(filenames):
            if progress_callback:
                progress_callback(current, idx + 1, len(filenames))
            try:
                result = download_sensor_file(
                    current,
                    name,
                    base_dir=base_dir,
                    session=sess,
                    force=force,
                )
            except requests.exceptions.ConnectTimeout as exc:
                logging.warning("Connect timeout for %s on %s: %s", name, current, exc)
                errors.append(f"{current} {name}: connection timed out")
                continue
            except requests.exceptions.ReadTimeout as exc:
                logging.warning("Read timeout for %s on %s: %s", name, current, exc)
                errors.append(f"{current} {name}: read timed out")
                continue
            except requests.HTTPError as exc:
                errors.append(f"{current} {name}: {exc}")
                continue
            if result.saved:
                downloaded += 1
            elif result.skipped:
                skipped += 1

        days += 1
        current = current + timedelta(days=1)

    return BulkDownloadSummary(
        days_processed=days,
        requested_files=requested,
        downloaded=downloaded,
        skipped=skipped,
        errors=errors,
        base_dir=base_dir,
    )
