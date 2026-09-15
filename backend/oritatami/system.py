"""Machine information, notifications and disk usage of the data directory."""

from __future__ import annotations

import logging
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
