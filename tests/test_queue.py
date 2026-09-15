"""Reordering and bulk-cancelling what is waiting, and reading the Metal working-set limit.

Both came out of one afternoon: an LLM proposal round queued twenty variants of a 1,140-residue
sequence that could not fit in memory, each of which would have spent ~10 minutes reaching the
same out-of-memory failure. There was no way to put a short job in front of them and no way to
drop them except one at a time.
"""

import threading
import time

import pytest
from oritatami import gpu
from oritatami.db import Database
from oritatami.jobs import JobManager


@pytest.fixture
def manager(tmp_path):
    db = Database(tmp_path / "jobs.sqlite3")
    m = JobManager(db)
    # Register the lane but never start the workers: these tests are about the waiting list,
    # and a worker would race us by pulling jobs off it.
    m.register("predict", lambda ctx: {}, lane="predict")
    m.register("scan", lambda ctx: {}, lane="esm")
    return m


def _ids(manager, lane="predict"):
    q = manager._queues[lane]
    with q.mutex:
        return list(q.queue)


def test_a_waiting_job_can_be_moved_to_the_front(manager):
    a = manager.submit("predict", {}, "A")["id"]
    b = manager.submit("predict", {}, "B")["id"]
    c = manager.submit("predict", {}, "C")["id"]
    assert _ids(manager) == [a, b, c]

    manager.reorder(c, "top")
    assert _ids(manager) == [c, a, b]
    manager.reorder(c, "down")
    assert _ids(manager) == [a, c, b]
    manager.reorder(c, "bottom")
    assert _ids(manager) == [a, b, c]
    manager.reorder(c, "up")
    assert _ids(manager) == [a, c, b]


def test_reordering_the_ends_is_not_an_error(manager):
    a = manager.submit("predict", {}, "A")["id"]
    b = manager.submit("predict", {}, "B")["id"]
    manager.reorder(a, "up")
    manager.reorder(b, "down")
    assert _ids(manager) == [a, b]


def test_only_waiting_jobs_can_be_reordered(manager):
    a = manager.submit("predict", {}, "A")["id"]
    manager.db.update_job(a, status="running")
    with pytest.raises(ValueError):
        manager.reorder(a, "top")
    with pytest.raises(ValueError):
        manager.reorder(manager.submit("predict", {}, "B")["id"], "sideways")


def test_cancelling_removes_the_job_from_the_waiting_list(manager):
    """A cancelled job used to stay in the lane, so the positions shown behind it were fiction."""
    a = manager.submit("predict", {}, "A")["id"]
    b = manager.submit("predict", {}, "B")["id"]
    c = manager.submit("predict", {}, "C")["id"]
    manager.cancel(a)
    assert _ids(manager) == [b, c]
    assert manager.describe(manager.db.get_job(c))["queue_position"] == 2


def test_cancel_all_clears_the_queue_but_not_the_running_job(manager):
    running = manager.submit("predict", {}, "走っている")["id"]
    manager.db.update_job(running, status="running")
    manager._forget("predict", [running])          # the worker took it off the list
    waiting = [manager.submit("predict", {}, f"待ち{i}")["id"] for i in range(5)]
    scan = manager.submit("scan", {}, "スキャン")["id"]

    cancelled = manager.cancel_queued()
    assert set(cancelled) == set(waiting) | {scan}
    assert _ids(manager) == [] and _ids(manager, "esm") == []
    assert manager.db.get_job(running)["status"] == "running"


def test_stopping_everything_includes_the_job_that_is_running(manager):
    """Having decided the whole batch was wrong, leaving the one job mid-flight — usually the
    longest — is not what "stop" means to the person who pressed it."""
    running = manager.submit("predict", {}, "走っている")["id"]
    manager.db.update_job(running, status="running")
    manager._forget("predict", [running])
    waiting = [manager.submit("predict", {}, f"待ち{i}")["id"] for i in range(3)]

    stopped = manager.cancel_batch(include_running=True)
    assert set(stopped) == set(waiting) | {running}
    assert manager._cancel_flags[running].is_set()
    # Still "running" in the database: the handler has to notice the flag, kill Boltz and
    # write the final status itself, or a job would read as cancelled while its process was
    # still holding the GPU.
    assert manager.db.get_job(running)["status"] == "running"


def test_cancel_all_can_be_limited_to_one_kind(manager):
    predicts = [manager.submit("predict", {}, f"P{i}")["id"] for i in range(3)]
    scan = manager.submit("scan", {}, "S")["id"]
    assert set(manager.cancel_queued("predict")) == set(predicts)
    assert manager.db.get_job(scan)["status"] == "queued"
    assert _ids(manager, "esm") == [scan]


def test_the_worker_is_not_left_waiting_on_jobs_that_were_removed(manager):
    """Queue.get() counts every put(); dropping ids without fixing that hangs join() forever."""
    for i in range(4):
        manager.submit("predict", {}, f"J{i}")
    manager.cancel_queued("predict")
    q = manager._queues["predict"]
    assert q.unfinished_tasks == 0
    finished = threading.Event()
    threading.Thread(target=lambda: (q.join(), finished.set()), daemon=True).start()
    assert finished.wait(2.0), "join() が返らない = キューの計数が壊れている"


def test_gpu_limits_are_readable_and_leave_room_for_the_system():
    state = gpu.state()
    assert state["total_gb"] > 0
    assert state["max_mb"] < state["total_gb"] * 1024, "全部を GPU に渡す値を上限にしない"
    assert state["wired_limit_mb"] >= 0
    assert state["is_default"] == (state["wired_limit_mb"] == 0)


def test_an_over_large_gpu_limit_is_refused_before_anything_is_run(monkeypatch):
    """The refusal has to happen here, not in sysctl: asking would pop an auth dialog first."""
    installed_mb = int(gpu.total_bytes() / 1024**2)          # read it before stubbing sysctl out
    called = []
    real_run = gpu.subprocess.run

    def watched(args, *a, **k):
        if args and args[0] == "osascript":
            called.append(args)
            raise AssertionError("検証前に osascript を呼んではいけない")
        return real_run(args, *a, **k)

    monkeypatch.setattr(gpu.subprocess, "run", watched)
    with pytest.raises(ValueError):
        gpu.apply(installed_mb)                              # everything installed
    with pytest.raises(ValueError):
        gpu.apply(512)                                       # too small to run the GPU at all
    with pytest.raises(ValueError):
        gpu.apply(-1)
    assert not called, "検証前に osascript を呼んではいけない"


def test_the_metal_probe_is_cached_but_can_be_forced(monkeypatch):
    calls = []

    def fake():
        calls.append(time.time())
        return 17.76

    monkeypatch.setattr(gpu, "_probe_metal", fake)
    gpu._probe.update(at=0.0, value=None)
    assert gpu.metal_limit_gb() == 17.76
    assert gpu.metal_limit_gb() == 17.76
    assert len(calls) == 1, "torch の起動は数秒かかるので、続けて聞かれたら使い回すこと"
    assert gpu.metal_limit_gb(fresh=True) == 17.76
    assert len(calls) == 2, "変更直後は測り直せること"


def test_reading_the_gpu_limit_does_not_depend_on_path(monkeypatch):
    """Launched from the .app bundle, PATH has no /usr/sbin and a bare "sysctl" is not found.

    That made the whole settings section answer 500 while the identical code worked from a
    terminal, so the binary is addressed absolutely and there is a sysconf fallback behind it.
    """
    assert gpu.SYSCTL_BIN.startswith("/"), "PATH に頼らないこと"
    monkeypatch.setattr(gpu, "_sysctl", lambda name: None)      # as if the binary were missing
    assert gpu.total_bytes() > 0
    assert gpu.wired_limit_mb() == 0
    assert gpu.bounds()["total_gb"] > 0
