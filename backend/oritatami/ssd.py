"""Wear on the internal SSD, read from the NVMe SMART log.

A prediction whose peak memory exceeds physical memory does not fail, it swaps — and swap
is a sustained stream of writes to a soldered drive that cannot be replaced without
replacing the logic board. Measured on this machine on 2026-09-14, a synthetic 40 GB
working set on 24 GB of physical memory produced 3.2M page-outs against 0.97M page-ins,
about 190 MB/s of writing, held for as long as the run lasted.

macOS exposes none of this. ``system_profiler SPNVMeDataType`` reports only
``S.M.A.R.T. status: Verified``, and ``ioreg`` carries no written/percent fields, so the
numbers here come from ``smartctl`` (smartmontools). It is not a dependency: when it is
absent the cost of a run is still stated in terabytes, just not as a share of the drive.

Apple publishes no endurance rating for Mac internal SSDs and does not limit the warranty
by write cycles, so there is no TBW figure to divide by. The drive's own two counters are
used as the calibration instead: ``percentage_used`` is the controller's estimate of life
consumed and ``data_units_written`` is what it took to get there, so their ratio is
terabytes per percent *for this individual drive*.

Read once, that ratio is weak, and it is labelled as such. Measured here on 2026-09-14:
74.5 TB at 3%, so 24.8 TB per percent and about 2,480 TB of implied total endurance. Three
things are wrong with taking that at face value. The line is drawn through a single point
and the origin, and nothing says the counter starts at zero. ``percentage_used`` is reported
as a whole number, so 3% is really somewhere in 3.0-3.99% and the ratio moves by a quarter
across that band. And the NVMe specification calls the field a "vendor specific estimate":
a controller basing it on the worst block's erase count moves slowly while wear levelling
still has room and faster later, which is not a straight line at all.

So the ratio is measured over time instead. Every read appends ``(written, percent)`` to a
small log, and once the percentage has moved even one point the slope comes from the
difference between two readings — which cancels both the unknown intercept and whatever
the counter did before we started watching. Until then the single-point number is served
with ``basis="single"`` and the UI says it is provisional.

For scale, the single-point figure is not implausible: consumer TBW ratings (600 TB for a
1 TB Samsung 990 PRO) are warranty limits set well under the physical endurance of TLC,
which runs 1,000-3,000 program/erase cycles. Reader reports collected by The Eclectic Light
Company imply 1,360 TB for a 256 GB MacBook Air and 13,370 TB for a Mac Studio, and that
article puts recent Macs at "conservative estimates ... around 3,000 cycles". 2,480 cycles
for this drive sits just under it. That is corroboration, not confirmation.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .config import app_home, get_settings

log = logging.getLogger("oritatami.ssd")

# An NVMe data unit is 1000 * 512 bytes (NVMe 1.4, Figure 194), not 512 * 1024.
DATA_UNIT_BYTES = 512_000
# Sustained page-out rate once the machine is swapping, measured on this machine with a
# 40 GB working set on 24 GB of physical memory. It is bounded by random 16 KB queue-depth-1
# IO (255 MB/s measured on this SSD), not by how far over the limit the job is, which is why
# it is a constant here rather than a function of the overflow.
SWAP_WRITE_MB_PER_SEC = 190.0

# Launched from the .app bundle the process inherits the GUI PATH, which contains neither
# /opt/homebrew/bin nor /usr/local/bin, so shutil.which finds nothing that a shell would.
_CANDIDATES = ("/opt/homebrew/sbin/smartctl", "/opt/homebrew/bin/smartctl",
               "/usr/local/sbin/smartctl", "/usr/local/bin/smartctl")
_CACHE_TTL_SEC = 1800.0
_cache: tuple[float, dict[str, Any] | None] = (0.0, None)

# Readings are appended here so the wear slope can be measured rather than extrapolated.
HISTORY_NAME = "ssd_wear.jsonl"
# A new line is only kept when it says something the last one did not: the percentage moved,
# or another terabyte went past. At 1.4 TB/day that is one line a day and a few dozen over
# the life of the machine, so the trim below is a backstop, not a working limit.
_HISTORY_MIN_TB_STEP = 1.0
_HISTORY_MAX_LINES = 2000


def smartctl_bin() -> str | None:
    for path in _CANDIDATES:
        if Path(path).exists():
            return path
    return shutil.which("smartctl")


def _history_path() -> Path:
    return app_home() / HISTORY_NAME


def _load_history() -> list[dict[str, Any]]:
    path = _history_path()
    if not path.exists():
        return []
    rows = []
    for line in path.read_text("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue        # a torn final line from a power loss; the rest is still good
        if isinstance(row, dict) and isinstance(row.get("percentage_used"), int) \
                and isinstance(row.get("written_tb"), (int, float)):
            rows.append(row)
    return rows


def _record(history: list[dict[str, Any]], written_tb: float, used: int) -> None:
    """Append this reading if it adds information. Best effort — losing one costs nothing."""
    if history:
        last = history[-1]
        if last["percentage_used"] == used and written_tb - last["written_tb"] < _HISTORY_MIN_TB_STEP:
            return
    row = {"ts": round(time.time()), "written_tb": round(written_tb, 3), "percentage_used": used}
    history.append(row)
    path = _history_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if len(history) > _HISTORY_MAX_LINES:
            # The span between the oldest reading and the newest is what makes the slope
            # measurable, so a trim that drops the front would throw away the only part that
            # matters. Keep both ends and rewrite the file — appending after an in-memory
            # trim would leave the file growing forever while pretending to be capped.
            keep = _HISTORY_MAX_LINES // 4
            history[:] = history[:keep] + history[-(_HISTORY_MAX_LINES - keep):]
            path.write_text("".join(json.dumps(r) + "\n" for r in history), "utf-8")
            return
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
    except OSError as exc:
        log.debug("SSD 摩耗の記録を書けませんでした: %s", exc)


def _calibrate(history: list[dict[str, Any]], written_tb: float, used: int) -> tuple[float | None, str, int]:
    """(TB per percent, how it was derived, percent spanned).

    Prefers the oldest reading that was at a lower percentage: the widest span available
    gives the least noise, and a difference between two readings needs no assumption about
    where the counter started.
    """
    for row in history:
        if row["percentage_used"] < used:
            d_pct = used - row["percentage_used"]
            d_tb = written_tb - row["written_tb"]
            if d_tb > 0:
                return round(d_tb / d_pct, 1), "delta", d_pct
            break       # the counter went backwards; that is not a measurement
    if used > 0:
        return round(written_tb / used, 1), "single", used
    return None, "none", 0


def _read(device: str = "/dev/disk0") -> dict[str, Any] | None:
    binary = smartctl_bin()
    if binary is None or sys.platform != "darwin":
        return None
    try:
        # smartctl sets low bits of its exit status for conditions that are not failures
        # here (Apple's controller has no Error Information Log page, which is bit 2), so
        # the output is parsed whatever the status and judged on its content.
        out = subprocess.run([binary, "-j", "-a", device], capture_output=True, text=True, timeout=10)
        data = json.loads(out.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, ValueError):
        return None
    log = data.get("nvme_smart_health_information_log")
    if not isinstance(log, dict):
        return None
    written = log.get("data_units_written")
    used = log.get("percentage_used")
    if not isinstance(written, int) or not isinstance(used, int):
        return None
    written_tb = written * DATA_UNIT_BYTES / 1e12
    history = _load_history()
    _record(history, written_tb, used)
    per_percent, basis, span = _calibrate(history, written_tb, used)
    return {
        "model": data.get("model_name"),
        "percentage_used": used,
        "written_tb": round(written_tb, 1),
        "read_tb": round(int(log.get("data_units_read") or 0) * DATA_UNIT_BYTES / 1e12, 1),
        "available_spare": log.get("available_spare"),
        "available_spare_threshold": log.get("available_spare_threshold"),
        "media_errors": log.get("media_errors"),
        "power_on_hours": log.get("power_on_hours"),
        "healthy": bool((data.get("smart_status") or {}).get("passed")),
        # None until the drive has worn a whole point: below that the ratio is a division by
        # zero dressed up as a measurement.
        "tb_per_percent": per_percent,
        # "delta" — measured across a real change in the counter, no assumptions.
        # "single" — one reading extrapolated through the origin. Provisional; say so.
        "tb_per_percent_basis": basis,
        "tb_per_percent_span": span,
        "readings": len(history),
    }


def wear(*, fresh: bool = False) -> dict[str, Any] | None:
    """SMART wear for the boot drive, or None when it cannot be read.

    Cached for half an hour: the counters move slowly and each read spawns a process.
    """
    global _cache
    now = time.time()
    if not fresh and now - _cache[0] < _CACHE_TTL_SEC:
        return _cache[1]
    value = _read()
    _cache = (now, value)
    return value


def swap_write_forecast(*, fresh: bool = False) -> dict[str, Any]:
    """What a run that swaps costs the SSD, per day of running.

    Stated as a rate rather than a total on purpose: once a job swaps, the runtime estimate
    that would turn a rate into a total is itself wrong by about two orders of magnitude
    (97 GB/s of page traffic resident against 296 MB/s swapping), so multiplying the two
    would dress a guess up as arithmetic.
    """
    tb_per_day = SWAP_WRITE_MB_PER_SEC * 86400 / 1e6
    w = wear(fresh=fresh)
    per_percent = (w or {}).get("tb_per_percent")
    return {
        "mb_per_sec": round(SWAP_WRITE_MB_PER_SEC),
        "tb_per_day": round(tb_per_day, 1),
        "life_percent_per_day": round(tb_per_day / per_percent, 2) if per_percent else None,
        "life_basis": (w or {}).get("tb_per_percent_basis", "none"),
        # Uncovered, a drive worn out by this is a logic board at full price; covered, it is
        # at least arguable. Either way it is the person's call, so the warning says which
        # one they told us they are in rather than picking a tone at random.
        "applecare": get_settings().applecare,
        "wear": w,
    }
