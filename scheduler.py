from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from apscheduler.schedulers.background import BackgroundScheduler


REPO_ROOT = Path(__file__).resolve().parent
INSERT_SENSOR_VALUES = REPO_ROOT / "preprocess" / "insert_sensor_values.py"
PRECOMPUTE_DAILY_METRICS = REPO_ROOT / "mndot_dash" / "precompute_daily_metrics.py"
DETECTORS_CSV = REPO_ROOT / "preprocess" / "detectors_minneapolis_radius_25.0km.csv"


def _load_timezone(name: str):
    try:
        return ZoneInfo(name)
    except Exception:
        return timezone.utc


def _now(tz):
    return datetime.now(tz)


def _run(cmd: list[str]) -> None:
    print(f"[{datetime.utcnow().isoformat()}Z] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def _validate_paths() -> None:
    missing = [p for p in (INSERT_SENSOR_VALUES, PRECOMPUTE_DAILY_METRICS, DETECTORS_CSV) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required file(s): {', '.join(str(p) for p in missing)}")


def daily_task(*, tz) -> None:
    """
    Runs at 00:05 and processes the day that just finished ("yesterday" in the scheduler timezone).
    """
    _validate_paths()

    day: date = _now(tz).date() - timedelta(days=1)
    yyyymmdd = day.strftime("%Y%m%d")
    iso_day = day.isoformat()

    _run(
        [
            sys.executable,
            str(INSERT_SENSOR_VALUES),
            "--data-source",
            "api",
            "--start-date",
            yyyymmdd,
            "--end-date",
            yyyymmdd,
            "--detectors-csv",
            str(DETECTORS_CSV),
            "--delete-existing",
        ]
    )

    _run(
        [
            sys.executable,
            str(PRECOMPUTE_DAILY_METRICS),
            "--start-day",
            iso_day,
            "--end-day",
            iso_day,
            "--delete-existing",
        ]
    )

    print(f"[{datetime.utcnow().isoformat()}Z] Completed pipeline for {iso_day}", flush=True)


def main() -> int:
    tz_name = os.environ.get("JOB_TZ", "America/Chicago")
    tz = _load_timezone(tz_name)
    if tz is timezone.utc and tz_name.upper() not in {"UTC", "ETC/UTC", "GMT"}:
        print(f"WARNING: Could not load timezone {tz_name!r}; falling back to UTC.", flush=True)

    scheduler = BackgroundScheduler(timezone=tz)
    scheduler.add_job(
        daily_task,
        "cron",
        hour=0,
        minute=5,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=6 * 3600,
        kwargs={"tz": tz},
    )
    scheduler.start()

    print(
        f"Scheduler started. job_tz={tz_name!r} next_run={scheduler.get_jobs()[0].next_run_time}",
        flush=True,
    )

    try:
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
