"""Wear accounting for the internal SSD, and what a swapping run costs it.

The point of all of this is one sentence in the workbench: "this job writes N TB a day,
which is M% of your drive". Both halves can be wrong in ways that are hard to see — a
single-point extrapolation that looks like a measurement, or a rate quietly multiplied by
a runtime estimate that stops being true the moment the job swaps — so they are pinned here.
"""

from __future__ import annotations

import json

import pytest
from oritatami import ssd


@pytest.fixture(autouse=True)
def _clear_cache():
    ssd._cache = (0.0, None)
    yield
    ssd._cache = (0.0, None)


def test_one_reading_is_labelled_as_the_extrapolation_it_is():
    per, basis, span = ssd._calibrate([], 74.5, 3)
    assert per == 24.8            # 74.5 / 3, drawn through an origin nobody verified
    assert basis == "single"
    assert span == 3


def test_a_moved_counter_gives_a_real_slope():
    # 20 TB later the drive admitted one more percent: that difference needs no assumption
    # about where the counter started or what it did before we were watching.
    history = [{"ts": 1, "written_tb": 74.5, "percentage_used": 3},
               {"ts": 2, "written_tb": 94.5, "percentage_used": 4}]
    per, basis, span = ssd._calibrate(history, 94.5, 4)
    assert (per, basis, span) == (20.0, "delta", 1)


def test_the_widest_span_wins():
    # Three percent of wear measured across 60 TB beats the last percent measured across 20:
    # the integer counter rounds each end, and a wider span dilutes that error.
    history = [{"ts": 1, "written_tb": 40.0, "percentage_used": 2},
               {"ts": 2, "written_tb": 60.0, "percentage_used": 3},
               {"ts": 3, "written_tb": 80.0, "percentage_used": 4}]
    per, basis, span = ssd._calibrate(history, 100.0, 5)
    assert (per, basis, span) == (20.0, "delta", 3)


def test_a_counter_that_went_backwards_is_not_a_measurement():
    # A replaced drive, or a restored machine: the written total is lower than a record we
    # kept. Dividing a negative by a positive would report a negative endurance.
    history = [{"ts": 1, "written_tb": 500.0, "percentage_used": 3}]
    per, basis, _ = ssd._calibrate(history, 10.0, 4)
    assert basis == "single" and per == 2.5


def test_no_wear_yet_means_no_answer_rather_than_a_division_by_zero():
    assert ssd._calibrate([], 12.0, 0) == (None, "none", 0)


def test_the_log_only_keeps_readings_that_say_something_new(isolated_home):
    history: list[dict] = []
    ssd._record(history, 74.50, 3)
    ssd._record(history, 74.52, 3)          # same percent, 20 GB later — nothing to learn
    assert len(history) == 1
    ssd._record(history, 76.00, 3)          # a whole terabyte on
    ssd._record(history, 76.10, 4)          # the percent moved: always worth keeping
    assert [r["percentage_used"] for r in history] == [3, 3, 4]
    lines = (isolated_home / ssd.HISTORY_NAME).read_text("utf-8").strip().splitlines()
    assert [json.loads(ln)["written_tb"] for ln in lines] == [74.5, 76.0, 76.1]


def test_a_torn_line_does_not_lose_the_rest_of_the_log(isolated_home):
    isolated_home.mkdir(parents=True, exist_ok=True)
    (isolated_home / ssd.HISTORY_NAME).write_text(
        '{"ts": 1, "written_tb": 40.0, "percentage_used": 2}\n'
        '{"ts": 2, "written_tb": 60.0, "per\n'          # power cut mid-write
        '{"ts": 3, "written_tb": 80.0, "percentage_used": 4}\n', "utf-8")
    rows = ssd._load_history()
    assert [r["percentage_used"] for r in rows] == [2, 4]


def test_the_log_is_capped_by_rewriting_and_keeps_both_ends(isolated_home, monkeypatch):
    monkeypatch.setattr(ssd, "_HISTORY_MAX_LINES", 8)
    monkeypatch.setattr(ssd, "_HISTORY_MIN_TB_STEP", 0.0)
    history: list[dict] = []
    for i in range(12):
        ssd._record(history, float(i), i)
    # The oldest reading is what makes the span wide, so a trim keeps the front as well as
    # the back — and it rewrites the file, because appending after an in-memory trim would
    # leave the file growing forever while the cap looked like it was working.
    assert len(history) <= 8
    assert history[0]["percentage_used"] == 0
    assert history[-1]["percentage_used"] == 11
    on_disk = (isolated_home / ssd.HISTORY_NAME).read_text("utf-8").strip().splitlines()
    assert len(on_disk) == len(history)


def test_the_write_forecast_is_a_rate_not_a_total(monkeypatch):
    # Deliberately not seconds x rate: once a job swaps the runtime estimate is wrong by two
    # orders of magnitude, and multiplying by it would dress a guess up as arithmetic.
    monkeypatch.setattr(ssd, "wear", lambda **_: {"tb_per_percent": 24.8,
                                                  "tb_per_percent_basis": "single"})
    out = ssd.swap_write_forecast()
    assert out["mb_per_sec"] == 190
    assert out["tb_per_day"] == 16.4                    # 190 MB/s held for a day
    assert out["life_percent_per_day"] == 0.66          # 16.4 / 24.8
    assert out["life_basis"] == "single"
    assert "seconds" not in out


def test_without_smartmontools_the_cost_is_still_stated_in_terabytes(monkeypatch):
    monkeypatch.setattr(ssd, "wear", lambda **_: None)
    out = ssd.swap_write_forecast()
    assert out["tb_per_day"] == 16.4
    assert out["life_percent_per_day"] is None          # no drive to measure it against
    assert out["life_basis"] == "none"


def test_nothing_is_read_when_smartctl_is_not_installed(monkeypatch):
    monkeypatch.setattr(ssd, "smartctl_bin", lambda: None)
    assert ssd._read() is None


def test_a_real_smartctl_payload_is_turned_into_wear(monkeypatch, isolated_home):
    # Captured from this machine on 2026-09-14. smartctl exits non-zero here because Apple's
    # controller has no Error Information Log page, which is not a failure of the read.
    payload = {"model_name": "APPLE SSD AP1024Z",
               "smart_status": {"passed": True},
               "nvme_smart_health_information_log": {
                   "percentage_used": 3, "data_units_written": 145_524_999,
                   "data_units_read": 311_659_099, "available_spare": 100,
                   "available_spare_threshold": 99, "media_errors": 0, "power_on_hours": 1245}}

    class _Done:
        stdout = json.dumps(payload)

    monkeypatch.setattr(ssd, "smartctl_bin", lambda: "/opt/homebrew/bin/smartctl")
    monkeypatch.setattr(ssd.sys, "platform", "darwin")
    monkeypatch.setattr(ssd.subprocess, "run", lambda *a, **k: _Done())
    out = ssd._read()
    assert out is not None
    assert out["written_tb"] == 74.5 and out["read_tb"] == 159.6
    assert out["percentage_used"] == 3 and out["healthy"] is True
    assert out["tb_per_percent"] == 24.8 and out["tb_per_percent_basis"] == "single"
    assert out["readings"] == 1
