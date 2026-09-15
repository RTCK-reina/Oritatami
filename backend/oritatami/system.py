"""Machine information, notifications and disk usage of the data directory."""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import app_home, get_settings, imports_dir, jobs_dir, msa_cache_dir
from .estimate import memory_gb

log = logging.getLogger("oritatami.system")


@lru_cache(maxsize=1)
def machine() -> dict[str, Any]:
    chip = None
    if sys.platform == "darwin":
        try:
            chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                                  timeout=2).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            chip = None
    return {
        "os": f"macOS {platform.mac_ver()[0]}" if sys.platform == "darwin" else platform.platform(),
        "chip": chip or platform.machine(),
        "memory_gb": round(memory_gb() or 0, 1) or None,
        "python": platform.python_version(),
    }


def disk_free_gb() -> float:
    return round(shutil.disk_usage(app_home()).free / 1024**3, 1)


def notify(title: str, message: str, *, respect_setting: bool = True) -> bool:
    """Show a macOS notification (Notification Center). Returns False when not supported.

    ``respect_setting=False`` bypasses ``notify_on_finish``: used by the autopilot,
    whose notifications are gated by ``autopilot_notify_improvement`` instead and
    must fire even when the UI is closed.
    """
    if sys.platform != "darwin":
        return False
    if respect_setting and not get_settings().notify_on_finish:
        return False

    def q(text: str) -> str:
        return '"' + text.replace("\\", "\\\\").replace('"', '\\"')[:240] + '"'

    script = f"display notification {q(message)} with title {q(title)} sound name \"Glass\""
    try:
        subprocess.Popen(["osascript", "-e", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except OSError as exc:
        log.warning("通知を表示できませんでした: %s", exc)
        return False


def _size(path: Path) -> int:
    total = 0
    if path.is_file():
        return path.stat().st_size
    for f in path.rglob("*"):
        try:
            if f.is_file() and not f.is_symlink():
                total += f.stat().st_size
        except OSError:
            continue
    return total


def storage() -> dict[str, Any]:
    s = get_settings()
    boltz_cache = Path(s.boltz_cache).expanduser()
    return {
        "home": str(app_home()),
        "jobs_bytes": _size(jobs_dir()),
        "imports_bytes": _size(imports_dir()),
        "msa_cache_bytes": _size(msa_cache_dir()),
        "boltz_cache_bytes": _size(boltz_cache) if boltz_cache.exists() else 0,
        "disk_free_gb": disk_free_gb(),
    }


def cleanup(*, intermediate: bool, aligned_older_than_days: float | None, skip_jobs: set[str]) -> dict[str, Any]:
    """Remove files that are safe to lose. Results (structures, scores) are never touched."""
    from .engines.boltz import cleanup_intermediate

    freed = 0
    jobs_cleaned = 0
    if intermediate:
        for job_dir in jobs_dir().iterdir():
            if job_dir.is_dir() and job_dir.name not in skip_jobs:  # never touch a job Boltz is still using
                n = cleanup_intermediate(job_dir)
                if n:
                    freed += n
                    jobs_cleaned += 1
    removed_imports = 0
    if aligned_older_than_days is not None:
        cutoff = time.time() - aligned_older_than_days * 86400
        for f in imports_dir().glob("aligned_*"):  # superposed .cif and its cached .json
            try:
                if f.stat().st_mtime < cutoff:
                    freed += f.stat().st_size
                    f.unlink()
                    removed_imports += 1
            except OSError:
                continue
    return {"freed_bytes": freed, "jobs_cleaned": jobs_cleaned, "aligned_removed": removed_imports}


# ------------------------------------------------------------------ this process's own memory
# Boltz runs in a subprocess and takes its memory with it when it exits, so the thing that can
# quietly grow over a long batch is the app itself: gemmi structures, PAE matrices, the ESM-2
# model and torch's MPS allocator pool all live here. RSS does not describe it — Metal
# allocations never appear in RSS — so read the same physical footprint the supervisor reads.
_SLOT_PHYS_FOOTPRINT = 9
_RUSAGE_INFO_V4 = 4
_libc: Any = None
# (jobs finished, footprint GB) sampled at the end of each job, newest last.
_footprints: list[tuple[int, float]] = []
_FOOTPRINT_KEEP = 200
# Growth over a batch that is worth saying out loud rather than leaving in a log.
FOOTPRINT_GROWTH_WARN_GB = 4.0


def process_footprint_gb() -> float | None:
    """Physical footprint of this process, in GB. None when the kernel call is unavailable."""
    global _libc
    import ctypes
    import ctypes.util

    if _libc is None:
        try:
            _libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.dylib", use_errno=True)
        except OSError:
            _libc = False
    if _libc is False:
        return None
    buf = (ctypes.c_uint64 * 64)()
    try:
        rc = _libc.proc_pid_rusage(ctypes.c_int(os.getpid()), ctypes.c_int(_RUSAGE_INFO_V4),
                                   ctypes.byref(buf))
    except (AttributeError, OSError):
        return None
    if rc != 0:
        return None
    return round(int(buf[_SLOT_PHYS_FOOTPRINT]) / 1024**3, 2)


def record_footprint() -> float | None:
    """Sample the footprint at the end of a job. Cheap: one syscall."""
    gb = process_footprint_gb()
    if gb is None:
        return None
    _footprints.append((len(_footprints) + 1, gb))
    del _footprints[:-_FOOTPRINT_KEEP]
    return gb


def memory_trend() -> dict[str, Any]:
    """Whether the app's own footprint is climbing job after job.

    A long batch that ends where it started is fine no matter how big each job was. One that
    ends several GB higher than it began is holding something it no longer needs, and the next
    prediction gets that much less of unified memory to work in.
    """
    samples = list(_footprints)
    now = process_footprint_gb()
    if len(samples) < 2:
        return {"current_gb": now, "samples": len(samples), "growth_gb": None, "climbing": False}
    first = min(gb for _, gb in samples[:3])
    last = max(gb for _, gb in samples[-3:])
    growth = round(last - first, 2)
    return {
        "current_gb": now,
        "samples": len(samples),
        "first_gb": first,
        "last_gb": last,
        "growth_gb": growth,
        "climbing": growth >= FOOTPRINT_GROWTH_WARN_GB,
        "series": [{"job": n, "gb": gb} for n, gb in samples[-60:]],
    }
