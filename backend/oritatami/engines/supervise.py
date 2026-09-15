"""Run a command, stop it if the app that started it goes away, and record what it used.

    python -m oritatami.engines.supervise PARENT_PID -- command args...

Boltz runs in its own session so a cancel can kill its whole process group, which also
means it would outlive an app that quits abruptly (⌘Q, force quit, crash) and keep the GPU
busy. This shim is that group's leader: it runs the command as a child, polls the parent
process, and terminates the group when the parent disappears. The exit status of the
command is passed through.

It is also the only place that sees the run while it is running, so it samples peak memory
into ``peak_memory.txt`` in the working directory. The app estimates peak memory from the
token count, and that estimate was extrapolated from a single observation — with this it can
be fitted from history the way the time estimate already is. Past physical memory the machine
does not fail, it slows by about 300x (measured on this machine: 97 GB/s of page traffic
resident, 296 MB/s once swapping), so knowing the peak before submitting is the difference
between a 2-minute job and an abandoned one.

The measurement is the process's *physical footprint*, not its resident size. On Metal the
tensors live in IOAccelerator allocations that never appear in RSS, and under memory pressure
RSS collapses further as pages are compressed or swapped. Measured here on a 1,696-residue
three-chain complex, at the same instant: ``ps`` reported 6.5 MB resident while the kernel
reported a 27 GB footprint and a 30 GB lifetime peak. An RSS-based reading is not an
underestimate of this, it is unrelated to it.

The same sampling also writes ``live_memory.json`` every few seconds, which is the only
thing a long run can be read by. Boltz reports no progress inside the diffusion phase — its
own bar sits at 1/1 from start to finish — so "has this stalled or is it just big" can only
be answered by whether the footprint is still under physical memory or the machine has
fallen into swap, and by how far over it is.

``proc_pid_rusage`` carries ``ri_lifetime_max_phys_footprint``, so the kernel keeps the peak
for us and the sampling interval cannot miss it — we only have to read each process before it
exits. Summing per-process lifetime maxima overstates the group's true simultaneous peak when
several processes peak at different times; for a pre-flight warning erring high is the right
direction, and in practice one process dominates.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time

# Written into the command's working directory, which is the job directory.
PEAK_NAME = "peak_memory.txt"
LIVE_NAME = "live_memory.json"

# struct rusage_info_v4, uint64 slots after the 16-byte uuid. Verified against
# /usr/bin/footprint on this machine: slot 9 matched "phys_footprint" and slot 30 matched
# "phys_footprint_peak" on a live 30 GB process.
_SLOT_PHYS_FOOTPRINT = 9
_SLOT_LIFETIME_MAX = 30
_RUSAGE_INFO_V4 = 4

_libc = None


def _rusage_footprint(pid: int) -> tuple[int, int]:
    """(current, lifetime peak) physical footprint of one process, in bytes. (0, 0) if unreadable."""
    global _libc
    if _libc is None:
        try:
            _libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.dylib", use_errno=True)
        except OSError:
            _libc = False
    if _libc is False:
        return 0, 0
    buf = (ctypes.c_uint64 * 64)()
    try:
        rc = _libc.proc_pid_rusage(ctypes.c_int(pid), ctypes.c_int(_RUSAGE_INFO_V4), ctypes.byref(buf))
    except (AttributeError, OSError):
        return 0, 0
    if rc != 0:
        return 0, 0
    peak = int(buf[_SLOT_LIFETIME_MAX])
    now = int(buf[_SLOT_PHYS_FOOTPRINT])
    # The lifetime max can only be >= the current value; if it is not, the layout is wrong
    # for this OS version and the number would be nonsense. Prefer the current reading.
    return now, (peak if peak >= now else now)


def _group_pids(pgid: int) -> list[int]:
    try:
        out = subprocess.run(["ps", "-Ao", "pgid=,pid="], capture_output=True, text=True,
                             timeout=4, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    pids = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit() and int(parts[0]) == pgid:
            pids.append(int(parts[1]))
    return pids


def _sample_group(pgid: int, seen: dict[int, int]) -> int:
    """Update the per-pid peak footprints for our group; return what it holds right now."""
    live = 0
    for pid in _group_pids(pgid):
        now, peak = _rusage_footprint(pid)
        live += now
        if peak > seen.get(pid, 0):
            seen[pid] = peak
    return live


_SWAP = re.compile(r"used\s*=\s*([0-9.]+)([MGK])")


def _swap_used_gb() -> float | None:
    """Bytes macOS currently has paged out, from vm.swapusage. None if it cannot be read.

    Absolute path on purpose: this runs as a child of an app that may have been launched
    from the Finder, whose PATH contains no /usr/sbin.
    """
    try:
        out = subprocess.run(["/usr/sbin/sysctl", "-n", "vm.swapusage"], capture_output=True,
                             text=True, timeout=4, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = _SWAP.search(out)
    if not m:
        return None
    scale = {"K": 1 / 1024**2, "M": 1 / 1024, "G": 1.0}[m.group(2)]
    return float(m.group(1)) * scale


def _total_gb() -> float | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    except (ValueError, OSError, AttributeError):
        return None


def _write_live(live_bytes: int, peak_bytes: int) -> None:
    """Publish the current sample for the app to read. Best effort; a miss costs one tick."""
    total = _total_gb()
    try:
        free_gb = shutil.disk_usage("/").free / 1024**3
    except OSError:
        free_gb = None
    sample = {
        "ts": time.time(),
        "footprint_gb": round(live_bytes / 1024**3, 2),
        "peak_gb": round(peak_bytes / 1024**3, 2),
        "total_gb": round(total, 1) if total else None,
        "swap_used_gb": None,
        "free_disk_gb": round(free_gb, 1) if free_gb is not None else None,
    }
    swap = _swap_used_gb()
    sample["swap_used_gb"] = round(swap, 2) if swap is not None else None
    # Write-then-rename so the reader never sees half a file. Same directory, so the rename
    # is atomic; a partial temp file left behind by a kill is overwritten next tick.
    tmp = LIVE_NAME + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(sample, fh)
        os.replace(tmp, LIVE_NAME)
    except OSError:
        pass


def _stop_group(child: subprocess.Popen) -> None:
    """Stop the command and everything it spawned, without touching anything else.

    Boltz forks dataloader workers, so signalling only the child leaves them holding the GPU;
    the group is the right unit. But ``killpg(getpgrp())`` is only ours to send when we lead
    the group — Boltz starts us with ``start_new_session=True`` precisely so that we do.
    Started any other way we are a guest in somebody else's group, and the signal goes to
    processes that have nothing to do with this run. That is not hypothetical: this shim was
    once launched by hand with a PARENT_PID that was not its parent, took the "parent died"
    branch on its first tick, and SIGTERMed the shell that started it and an unrelated app
    sharing that group. The check below is one syscall and makes that impossible.
    """
    leader = os.getpgrp() == os.getpid()
    try:
        if leader:
            os.killpg(os.getpgrp(), signal.SIGTERM)
        elif child.poll() is None:
            child.terminate()
    except (OSError, ProcessLookupError):
        return
    try:
        child.wait(timeout=8)
    except subprocess.TimeoutExpired:
        try:
            if leader:
                os.killpg(os.getpgrp(), signal.SIGKILL)
            else:
                child.kill()
        except (OSError, ProcessLookupError):
            pass


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def main(argv: list[str]) -> int:
    if len(argv) < 3 or argv[1] != "--":
        print("usage: supervise PARENT_PID -- command ...", file=sys.stderr)
        return 2
    parent = int(argv[0])
    child = subprocess.Popen(argv[2:])

    def forward(signum: int, _frame: object) -> None:
        if child.poll() is None:
            child.send_signal(signum)

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    pgid = os.getpgrp()
    seen: dict[int, int] = {}
    samples = 0

    def record_peak() -> None:
        _sample_group(pgid, seen)          # one last look before anything is reaped
        # Our own footprint is noise next to the command's; leave it in rather than guess.
        peak = sum(seen.values())
        if not peak:
            return
        try:
            with open(PEAK_NAME, "w", encoding="utf-8") as fh:
                fh.write(f"{peak / 1024**3:.3f}\n")
        except OSError:
            pass

    while child.poll() is None:
        samples += 1
        if samples % 3 == 0:            # every ~3 s; enumerating processes is not free
            live = _sample_group(pgid, seen)
            _write_live(live, sum(seen.values()))
        if not _alive(parent) or os.getppid() != parent:
            print("[oritatami] アプリが終了したため計算を停止します", flush=True)
            record_peak()               # read the footprints before the group is torn down
            _stop_group(child)
            return 143
        time.sleep(1.0)
    record_peak()
    return child.returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
