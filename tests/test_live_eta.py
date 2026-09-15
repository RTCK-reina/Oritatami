"""How much longer a running prediction needs, measured rather than assumed.

Boltz reports no progress from inside the diffusion phase, so the old answer was the
token-based estimate minus elapsed, clamped to a 15-second floor. That floor is a lie in the
one case it matters: a 1,278-token job that had been running 162 minutes against a 26-minute
estimate reported "15 秒" because the subtraction went negative.
"""

import pytest
from oritatami import estimate

SPEC = {"token_estimate": 800, "params": {"diffusion_samples": 1, "sampling_steps": 200,
                                          "recycling_steps": 4}}


def _finished(tokens, cpu_seconds, elapsed, peak=None):
    res = {"token_estimate": tokens, "cpu_seconds": cpu_seconds, "elapsed_sec": elapsed,
           "cpu_efficiency": round(cpu_seconds / elapsed, 3),
           "normalized_spec": {"token_estimate": tokens, "params": SPEC["params"]}}
    if peak is not None:
        res["peak_memory_gb"] = peak
        res["peak_memory_method"] = "footprint"
    return {"id": "j", "result": res}


HISTORY = [_finished(200, 60, 65, 6.0), _finished(400, 260, 280, 9.0),
           _finished(800, 1000, 1080, 20.0)]


def test_cpu_cost_needs_two_measured_runs():
    assert estimate.cpu_cost(SPEC, [])[0] is None
    assert estimate.cpu_cost(SPEC, HISTORY[:1])[0] is None
    cost, n = estimate.cpu_cost(SPEC, HISTORY)
    assert n == 3 and cost > 0


def test_remaining_is_measured_from_what_the_run_has_computed():
    cost, _ = estimate.cpu_cost(SPEC, HISTORY)
    live = {"cpu_sec": cost / 2, "efficiency": 1.0}
    out = estimate.remaining(SPEC, elapsed=cost / 2, baseline=cost, live=live, history=HISTORY)
    assert out["basis"] == "cpu"
    assert out["progress"] == pytest.approx(0.5, abs=0.02)
    assert out["seconds"] == pytest.approx(cost / 2, rel=0.05)


def test_a_slow_run_gets_a_longer_estimate_not_a_floor():
    """Same work done, one tenth of the speed: ten times the remaining time."""
    cost, _ = estimate.cpu_cost(SPEC, HISTORY)
    fast = estimate.remaining(SPEC, 100, cost, {"cpu_sec": cost / 2, "efficiency": 1.0}, HISTORY)
    slow = estimate.remaining(SPEC, 100, cost, {"cpu_sec": cost / 2, "efficiency": 0.1}, HISTORY)
    assert slow["seconds"] == pytest.approx(fast["seconds"] * 10, rel=0.02)
    assert "実効速度" in slow["note"]


def test_a_thrashing_run_reports_no_estimate_rather_than_a_number():
    out = estimate.remaining(SPEC, 9800, 1500, {"cpu_sec": 2000, "efficiency": 0.005}, HISTORY)
    assert out["seconds"] is None and out["basis"] == "unknown"
    assert "スワップ" in out["note"]


def test_an_overrun_with_no_measurement_says_so_instead_of_15_seconds():
    """The case that prompted this: 162 minutes against a 26-minute estimate."""
    out = estimate.remaining(SPEC, elapsed=9800, baseline=1560, live={}, history=[])
    assert out["seconds"] is None
    assert out["overrun"] is True
    assert "163 分" in out["note"] and "26 分" in out["note"]


def test_before_the_estimate_runs_out_the_baseline_is_still_used():
    out = estimate.remaining(SPEC, elapsed=100, baseline=1000, live={}, history=[])
    assert out["basis"] == "baseline" and out["seconds"] == 900
    assert out["overrun"] is False


def test_normal_efficiency_ignores_runs_that_swapped(monkeypatch):
    monkeypatch.setattr(estimate, "memory_gb", lambda: 24.0)
    history = HISTORY + [_finished(1278, 2000, 20000, 42.0)]     # 10 % efficiency, swapped
    assert estimate.normal_efficiency(history) > 0.8


def test_efficiency_is_ignored_when_the_supervisor_did_not_report_it():
    out = estimate.remaining(SPEC, 100, 1000, {"cpu_sec": 50}, HISTORY)
    assert out["basis"] == "baseline"


# ---------------------------------------------------------------- through the endpoint
def test_queue_eta_marks_an_untimeable_queue_as_a_lower_bound(monkeypatch):
    """A clock time only means something when every job in the queue could be timed."""
    from fastapi.testclient import TestClient
    from oritatami.app import app, state

    with TestClient(app) as client:
        job = state.db.insert_job(
            kind="predict", title="テスト", parent_id=None, origin="user",
            spec={"components": [{"type": "protein", "sequence": "MQIFVKTLTGKTITLEVEPSD",
                                  "chains": ["A"], "msa": "single"}], "params": {}})
        state.db.update_job(job["id"], status="running", started_at=0.0)
        # 3 hours in, almost no compute earned: the thrashing case.
        monkeypatch.setattr(state.jobs, "live",
                            lambda _id: {"started_at": __import__("time").time() - 10800,
                                         "cpu_sec": 40.0, "efficiency": 0.004})
        body = client.get("/api/jobs/eta").json()

    assert body["complete"] is False
    assert body["finish_at"] is None
    assert body["at_least_until"] is not None or body["counted"] == 0
    item = next(i for i in body["items"] if i["id"] == job["id"])
    assert item["seconds"] is None
    assert item["overrun"] is True
    assert item["basis"] == "unknown"
