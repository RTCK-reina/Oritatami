"""The supervisor's CPU reading: the units are the part that goes wrong silently."""

import os

from oritatami.engines import supervise


def test_mach_timebase_is_read_not_assumed():
    tick = supervise._tick_seconds()
    # 1 ns/tick on x86, 125/3 ns on Apple Silicon. Anything outside this is a bad read.
    assert 1e-9 <= tick <= 1e-6


def test_cpu_time_matches_this_process():
    """ri_user_time/ri_system_time are mach units, not nanoseconds.

    Read as nanoseconds they come out ~42x too small, which looks plausible enough to ship.
    Checking against os.times() is what catches it.
    """
    _, _, user, system = supervise._rusage(os.getpid())
    if user == 0 and system == 0:
        return                       # not macOS, or the call is unavailable
    mine = os.times()
    expected = mine.user + mine.system
    assert expected * 0.5 <= user + system <= expected * 2 + 1.0


def test_a_dead_process_reads_as_zero():
    assert supervise._rusage(2**30) == (0, 0, 0.0, 0.0)


def test_efficiency_needs_a_previous_sample(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    supervise._last_live.clear()
    supervise._write_live(1024**3, 2 * 1024**3, cpu_sec=10.0, user_sec=9.0)
    import json

    first = json.loads((tmp_path / supervise.LIVE_NAME).read_text())
    assert first["efficiency"] is None          # nothing to compare against yet
    assert first["cpu_sec"] == 10.0 and first["user_share"] == 0.9

    supervise._last_live["ts"] -= 10.0          # pretend the previous sample was 10 s ago
    supervise._write_live(1024**3, 2 * 1024**3, cpu_sec=15.0, user_sec=10.0)
    second = json.loads((tmp_path / supervise.LIVE_NAME).read_text())
    # ~5 CPU seconds in ~10 wall seconds; the real wall delta is a hair over 10 s
    assert abs(second["efficiency"] - 0.5) < 0.02


def test_the_cpu_file_records_the_user_split(tmp_path, monkeypatch):
    """The total alone cannot say whether a run was computing or paging."""
    monkeypatch.chdir(tmp_path)
    cpu = {1: (12.0, 3.0), 2: (4.0, 1.0)}
    used = sum(u + s for u, s in cpu.values())
    user = sum(u for u, _ in cpu.values())
    (tmp_path / supervise.CPU_NAME).write_text(f"{used:.1f} {user:.1f}\n")
    parts = (tmp_path / supervise.CPU_NAME).read_text().split()
    assert float(parts[0]) == 20.0 and float(parts[1]) == 16.0
