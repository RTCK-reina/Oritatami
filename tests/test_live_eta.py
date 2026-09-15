"""Timing a prediction that does not fit in memory.

Boltz does not fail when the working set stops fitting — it pages, and keeps going at the
speed of the SSD. A 1,278-token job estimated at 40 minutes was still running after 212, and
the queue view said "15 秒" the whole time because the estimate had gone negative and been
clamped to a floor.

Two measurements shape what is here, both taken on this machine:

* three calibration runs inside memory — 76, 304 and 608 tokens — where wall time went
  36 s → 116 s → 389 s while CPU time went 20 s → 27 s → 40 s. CPU time is not a progress
  meter when the GPU does the arithmetic.
* the 1,278-token run while it was paging: 1.6 % user time against 64-86 % for the healthy
  three, and about 565 MB/s of swap traffic.
"""

import pytest
from oritatami import estimate

PARAMS = {"diffusion_samples": 1, "sampling_steps": 200, "recycling_steps": 4}
SPEC = {"token_estimate": 1278, "params": PARAMS}
TOTAL = 24.0


def test_a_job_that_fits_pays_nothing():
    p = estimate.swap_penalty(14.75, PARAMS, TOTAL)       # 608 トークンの実測ピーク
    assert p["overage_gb"] == 0.0 and p["seconds"] == 0.0


def test_the_capacity_leaves_room_for_the_system(monkeypatch):
    assert estimate.resident_capacity_gb(24.0) == 20.0
    # None means "this machine", not "unknown"
    monkeypatch.setattr(estimate, "memory_gb", lambda: 36.0)
    assert estimate.resident_capacity_gb(None) == 32.0
    monkeypatch.setattr(estimate, "memory_gb", lambda: None)
    assert estimate.resident_capacity_gb(None) is None


def test_the_anchor_case_lands_in_the_right_hours():
    """45.56 GB on a 24 GB machine ran past 212 minutes without finishing.

    The model is allowed to be wrong about the minutes and not about the scale: anything
    under two hours would have been just as useless as the 40-minute answer it replaced.
    """
    p = estimate.swap_penalty(45.56, PARAMS, TOTAL)
    assert p["overage_gb"] == pytest.approx(25.6, abs=0.1)
    assert 2.5 * 3600 <= p["seconds"] <= 6 * 3600
    assert p["passes"] == 280


def test_the_penalty_grows_with_the_overage():
    small = estimate.swap_penalty(22.0, PARAMS, TOTAL)["seconds"]
    large = estimate.swap_penalty(30.0, PARAMS, TOTAL)["seconds"]
    assert large > small > 0
    # linear in the overage: twice as much outside memory, twice the paging
    a = estimate.swap_penalty(25.0, PARAMS, TOTAL)["seconds"]
    b = estimate.swap_penalty(30.0, PARAMS, TOTAL)["seconds"]
    assert b / a == pytest.approx(10 / 5, rel=0.01)


def test_fewer_diffusion_steps_cost_less_paging():
    many = estimate.swap_penalty(40.0, {**PARAMS, "sampling_steps": 200}, TOTAL)["seconds"]
    few = estimate.swap_penalty(40.0, {**PARAMS, "sampling_steps": 50}, TOTAL)["seconds"]
    assert few < many


def test_the_estimate_carries_the_paging_term():
    spec = {"token_estimate": 1278, "params": PARAMS}
    out = estimate.estimate(spec, False, [])
    assert out["breakdown"]["paging"] > 0
    assert out["seconds"] >= out["breakdown"]["paging"]
    assert out["memory"]["overage_gb"] > 0


# ---------------------------------------------------------------- which regime, live
def test_user_share_separates_the_regimes():
    assert estimate.regime({"user_share": 0.642, "cpu_sec": 40}) == "resident"   # 608 実測
    assert estimate.regime({"user_share": 0.016, "cpu_sec": 2455}) == "paging"   # 1278 実測


def test_cpu_efficiency_is_not_used_as_a_health_signal():
    """A healthy 608-token run sat at 0.104 and the paging one at 0.21 — higher."""
    healthy = {"user_share": 0.642, "cpu_sec": 40, "efficiency": 0.104}
    paging = {"user_share": 0.016, "cpu_sec": 2455, "efficiency": 0.21}
    assert estimate.regime(healthy) == "resident"
    assert estimate.regime(paging) == "paging"


def test_the_footprint_decides_before_enough_cpu_has_accrued():
    assert estimate.regime({"footprint_gb": 34.0, "total_gb": 24.0, "cpu_sec": 1}) == "paging"
    assert estimate.regime({"footprint_gb": 12.0, "total_gb": 24.0, "cpu_sec": 1}) == "resident"
    assert estimate.regime({}) == "unknown"


def test_remaining_adds_the_paging_time_from_the_measured_footprint():
    live = {"peak_gb": 45.56, "total_gb": TOTAL, "user_share": 0.016, "cpu_sec": 2455}
    out = estimate.remaining(SPEC, elapsed=600, baseline=2398, live=live, history=[])
    assert out["basis"] == "swap" and out["regime"] == "paging"
    assert out["seconds"] > 2 * 3600
    assert out["overage_gb"] == pytest.approx(25.6, abs=0.1)
    assert "物理メモリ" in out["note"]


def test_remaining_is_the_plain_baseline_when_it_fits():
    live = {"peak_gb": 14.75, "total_gb": TOTAL, "user_share": 0.642, "cpu_sec": 40}
    out = estimate.remaining({"token_estimate": 608, "params": PARAMS},
                             elapsed=100, baseline=389, live=live, history=[])
    assert out["basis"] == "baseline" and out["seconds"] == 289
    assert out["regime"] == "resident" and out["overage_gb"] == 0.0


def test_past_everything_the_model_accounts_for_it_says_so():
    live = {"peak_gb": 45.56, "total_gb": TOTAL, "user_share": 0.016, "cpu_sec": 2455}
    out = estimate.remaining(SPEC, elapsed=60 * 3600, baseline=2398, live=live, history=[])
    assert out["seconds"] is None and out["basis"] == "unknown" and out["overrun"] is True
    assert "ページング" in out["note"]


def test_progress_is_elapsed_against_the_corrected_total():
    live = {"peak_gb": 45.56, "total_gb": TOTAL, "user_share": 0.016, "cpu_sec": 2455}
    out = estimate.remaining(SPEC, elapsed=out_total(SPEC, live) / 2, baseline=2398,
                             live=live, history=[])
    assert out["progress"] == pytest.approx(0.5, abs=0.01)


def out_total(spec, live):
    return 2398 + estimate.swap_penalty(live["peak_gb"], spec["params"], live["total_gb"])["seconds"]
