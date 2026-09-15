"""How much of unified memory Metal will let one process hold, and how to change it.

On Apple Silicon the GPU and the CPU share the same memory, but Metal still caps what a
single process may keep resident: ``recommendedMaxWorkingSetSize``. On this 24 GB machine
the default works out to 17.76 GB, and ``sysctl iogpu.wired_limit_mb`` raises or lowers it
(0 means "use the default").

Why it matters here, measured on this machine 2026-09-14:

* A 1,696-residue three-chain complex reached a 30 GB physical footprint and then Boltz
  logged ``ran out of memory, skipping batch`` and produced nothing, after about 16 minutes.
  Metal gave up; the machine itself was fine. A 609-residue monomer on the same day did
  finish, in 997 s, so the wall sits somewhere between the two.
* Plain memory does not behave that way. A synthetic allocation of 40 GB — 1.67x physical —
  ran to completion on swap, at 245 MB/s against 97,000 MB/s while resident. So the wall a
  prediction hits is Metal's working-set limit, not the machine running out of room.

Raising the limit therefore buys real headroom, and costs the rest of the system the memory
it takes. The kernel does not protect you from setting it to everything you have; leaving a
few GB for macOS and the app is the caller's job, which is why :func:`bounds` reports a
maximum below the installed memory rather than trusting the input.

The setting does not survive a reboot, and Metal reads it when the process opens the device,
so a running prediction keeps the limit it started with and the app has to be restarted for
its own torch session to see a change.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from typing import Any

SYSCTL = "iogpu.wired_limit_mb"
# Absolute path on purpose. Launched from the .app bundle the process inherits the GUI PATH,
# which does not contain /usr/sbin, and a bare "sysctl" raised FileNotFoundError — the whole
# settings section answered 500 while the same code worked from a terminal.
SYSCTL_BIN = "/usr/sbin/sysctl"

# macOS needs room to run; below this the machine starts fighting the prediction for pages.
HEADROOM_GB = 4.0
_PROBE_TTL = 30.0
_probe: dict[str, Any] = {"at": 0.0, "value": None}


def _sysctl(name: str) -> str | None:
    try:
        return subprocess.run([SYSCTL_BIN, "-n", name], capture_output=True, text=True,
                              timeout=4, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def total_bytes() -> int:
    """Installed memory. Falls back to sysconf so this never depends on a binary being found."""
    raw = _sysctl("hw.memsize")
    if raw:
        try:
            return int(raw)
        except ValueError:
            pass
    return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))


def wired_limit_mb() -> int:
    """Current value of the sysctl. 0 means the OS default is in force."""
    raw = _sysctl(SYSCTL)
    try:
        return int(raw) if raw else 0
    except ValueError:
        return 0


def _probe_metal() -> float | None:
    """What Metal actually reports right now, in GB. Runs out of process so it is never stale.

    Importing torch costs seconds, and the in-process torch has already opened the Metal
    device, so asking it would return the limit as it was at start-up rather than as it is.
    """
    code = ("import json, torch;"
            "print(json.dumps(torch.mps.recommended_max_memory()))")
    try:
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             timeout=90, check=True).stdout
        return float(json.loads(out.strip().splitlines()[-1])) / 1024**3
    except (OSError, ValueError, json.JSONDecodeError, IndexError, subprocess.SubprocessError):
        return None


def metal_limit_gb(*, fresh: bool = False) -> float | None:
    now = time.time()
    if not fresh and _probe["value"] is not None and now - _probe["at"] < _PROBE_TTL:
        return _probe["value"]
    value = _probe_metal()
    if value is not None:
        _probe.update(at=now, value=value)
    return value


def bounds() -> dict[str, Any]:
    total_gb = total_bytes() / 1024**3
    return {
        "total_gb": round(total_gb, 1),
        "min_mb": 0,                                   # 0 = OS default
        "max_mb": int((total_gb - HEADROOM_GB) * 1024),
        "headroom_gb": HEADROOM_GB,
    }


def state(*, fresh: bool = False) -> dict[str, Any]:
    out = bounds()
    out["wired_limit_mb"] = wired_limit_mb()
    out["metal_limit_gb"] = metal_limit_gb(fresh=fresh)
    out["is_default"] = out["wired_limit_mb"] == 0
    return out


def apply(mb: int) -> dict[str, Any]:
    """Ask macOS to change the limit. The user authorises it; we never see the password.

    ``do shell script ... with administrator privileges`` puts up the system's own
    authentication dialog. Nothing here reads, stores or forwards what is typed into it.
    """
    limits = bounds()
    mb = int(mb)
    if mb < 0:
        raise ValueError("0 以上で指定してください (0 = OS の既定値に戻す)")
    if mb > limits["max_mb"]:
        raise ValueError(
            f"{mb} MB は大きすぎます。搭載 {limits['total_gb']} GB のうち "
            f"{HEADROOM_GB:.0f} GB は macOS とアプリのために残す必要があるので、"
            f"上限は {limits['max_mb']} MB です")
    if mb != 0 and mb < 1024:
        raise ValueError("1024 MB 未満にすると GPU が使えなくなります。0 なら既定値に戻せます")

    inner = f"{SYSCTL_BIN} -w {SYSCTL}={mb}"
    script = f"do shell script {json.dumps(inner)} with administrator privileges"
    try:
        proc = subprocess.run(["osascript", "-e", script], capture_output=True, text=True,
                              timeout=180)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"設定を変更できませんでした: {exc}") from exc
    if proc.returncode != 0:
        message = (proc.stderr or "").strip()
        if "-128" in message or "User canceled" in message:
            raise PermissionError("認証がキャンセルされました。設定は変更していません")
        raise RuntimeError(f"設定を変更できませんでした: {message or shlex.quote(inner)}")

    after = state(fresh=True)
    after["changed_to_mb"] = mb
    # Metal reads the limit when a process opens the device, so this app's own torch session
    # still has the old one until it is restarted. Say so rather than implying it took effect
    # everywhere.
    after["restart_required"] = True
    after["persists_across_reboot"] = False
    return after
