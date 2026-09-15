"""The app's own footprint across jobs, and handing Metal back what torch is not using."""

from oritatami import system
from oritatami.engines import esm


def test_footprint_is_readable_and_positive():
    gb = system.process_footprint_gb()
    assert gb is None or gb > 0


def test_trend_needs_two_samples_before_it_says_anything():
    system._footprints.clear()
    first = system.memory_trend()
    assert first["samples"] == 0 and first["growth_gb"] is None and not first["climbing"]


def test_trend_reports_growth_across_jobs():
    system._footprints.clear()
    system._footprints.extend([(1, 2.0), (2, 2.1), (3, 9.0), (4, 9.2)])
    t = system.memory_trend()
    assert t["growth_gb"] == 7.2 and t["climbing"] is True
    assert len(t["series"]) == 4


def test_flat_usage_is_not_flagged():
    system._footprints.clear()
    system._footprints.extend([(i, 2.0 + (i % 2) * 0.1) for i in range(1, 30)])
    t = system.memory_trend()
    assert t["climbing"] is False


def test_release_cache_is_free_when_torch_was_never_imported(monkeypatch):
    """Importing torch to empty its cache would cost more memory than it returns."""
    import sys

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    assert esm.release_cache() == 0.0


def test_every_finished_job_leaves_a_footprint_sample(tmp_path):
    """The sampling has to sit in the worker's finally block, or a failed job never records."""
    import time

    from oritatami.db import Database
    from oritatami.jobs import JobManager

    system._footprints.clear()
    m = JobManager(Database(tmp_path / "jobs.sqlite3"))
    m.register("ok", lambda ctx: {"done": True}, lane="esm")
    m.register("boom", lambda ctx: (_ for _ in ()).throw(RuntimeError("失敗")), lane="esm")
    m.start()
    try:
        m.submit("ok", {}, "A")
        m.submit("boom", {}, "B")
        deadline = time.time() + 20
        while len(system._footprints) < 2 and time.time() < deadline:
            time.sleep(0.05)
    finally:
        m.stop()
    assert len(system._footprints) >= 2, "成功したジョブも失敗したジョブも記録されること"
