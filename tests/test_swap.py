"""Running past physical memory on purpose, and being able to read a run that does.

Boltz used to die at PyTorch's allocator ceiling: 17.76 GB recommended x 1.7 default = 30.2 GB,
and a 1,696-residue three-chain complex hit exactly that after 16 minutes and produced nothing.
The machine was fine — a synthetic 40 GB working set on 24 GB of physical memory completed on
swap — so the ceiling was throwing away work to avoid something survivable. It is now removed
by default, which moves the cost from "job lost" to "SSD written to", and moves the job's
duration from minutes to possibly days. Neither of those may be silent, hence this file.
"""

from __future__ import annotations

import json
import signal

import pytest
from oritatami import estimate
from oritatami.config import update_settings
from oritatami.engines import boltz, supervise

# ---- the ceiling ---------------------------------------------------------------------

def test_the_allocator_ceiling_is_removed_by_default():
    env = boltz.run_env("mps")
    # 0.0 is PyTorch's "no limit". The alternative was a number, and any number would have
    # been this machine's number: the ceiling that killed a job at 30.2 GB was also a number
    # somebody picked once.
    assert env["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] == "0.00"
    # The low watermark is deliberately left alone: at its 1.4 default the allocator keeps
    # releasing cached blocks under pressure, which is exactly what a swapping run wants.
    assert "PYTORCH_MPS_LOW_WATERMARK_RATIO" not in env


def test_the_ceiling_can_be_put_back():
    update_settings({"mps_memory_ratio": 1.7})
    assert boltz.run_env("mps")["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] == "1.70"


def test_a_cpu_run_is_not_given_a_metal_setting():
    assert "PYTORCH_MPS_HIGH_WATERMARK_RATIO" not in boltz.run_env("cpu")


# ---- reading a long run --------------------------------------------------------------

def test_swap_usage_is_parsed_in_every_unit_macos_uses(monkeypatch):
    def fake(cmd, **kwargs):
        class R:
            stdout = fake.text
        return R()

    monkeypatch.setattr(supervise.subprocess, "run", fake)
    fake.text = "total = 4096.00M  used = 3005.44M  free = 1090.56M  (encrypted)"
    assert round(supervise._swap_used_gb(), 2) == 2.94
    fake.text = "total = 8.00G  used = 6.50G  free = 1.50G  (encrypted)"
    assert supervise._swap_used_gb() == 6.5
    fake.text = "total = 1024.00K  used = 512.00K  free = 512.00K"
    assert round(supervise._swap_used_gb(), 4) == 0.0005
    fake.text = "nothing like swap at all"
    assert supervise._swap_used_gb() is None


def test_a_sample_is_published_for_the_app_to_read(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(supervise, "_swap_used_gb", lambda: 6.9)
    supervise._write_live(30 * 1024**3, 31 * 1024**3)
    sample = json.loads((tmp_path / supervise.LIVE_NAME).read_text("utf-8"))
    assert sample["footprint_gb"] == 30.0 and sample["peak_gb"] == 31.0
    assert sample["swap_used_gb"] == 6.9
    assert sample["ts"] > 0
    # Free space matters here in a way it does not elsewhere: swap lives on the boot volume,
    # and a run that fills it takes the whole machine down with it, not just the job.
    assert "free_disk_gb" in sample
    assert not (tmp_path / (supervise.LIVE_NAME + ".tmp")).exists()


def test_a_half_written_sample_is_skipped_rather_than_crashing(tmp_path):
    seen: list[dict] = []
    (tmp_path / supervise.LIVE_NAME).write_text('{"ts": 5, "footprint_gb": 3', "utf-8")
    assert boltz._report_live(tmp_path, 0.0, seen.append) == 0.0
    assert seen == []


def test_nothing_is_reported_until_the_sample_is_newer(tmp_path):
    seen: list[dict] = []
    path = tmp_path / supervise.LIVE_NAME
    path.write_text(json.dumps({"ts": 100.0, "footprint_gb": 12.0}), "utf-8")
    assert boltz._report_live(tmp_path, 0.0, seen.append) == 100.0
    assert seen == [{"memory": {"ts": 100.0, "footprint_gb": 12.0}}]
    # Same file again: the UI already has it, and re-sending would bump the live revision
    # every 0.7 s and make every client re-poll for nothing.
    assert boltz._report_live(tmp_path, 100.0, seen.append) == 100.0
    assert len(seen) == 1


def test_a_missing_sample_is_normal(tmp_path):
    assert boltz._report_live(tmp_path, 0.0, lambda _: pytest.fail("報告してはいけない")) == 0.0


# ---- stopping it ---------------------------------------------------------------------

class _Child:
    """Stands in for the Boltz process."""

    def __init__(self):
        self.terminated = False
        self.killed = False

    def poll(self):
        return None

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return 0


def test_the_whole_group_is_stopped_when_the_supervisor_leads_it(monkeypatch):
    # Boltz forks dataloader workers; signalling only the child leaves them on the GPU.
    sent = []
    monkeypatch.setattr(supervise.os, "getpgrp", lambda: 333)
    monkeypatch.setattr(supervise.os, "getpid", lambda: 333)
    monkeypatch.setattr(supervise.os, "killpg", lambda pgid, sig: sent.append((pgid, sig)))
    child = _Child()
    supervise._stop_group(child)
    assert sent == [(333, signal.SIGTERM)]
    assert not child.terminated          # the group signal already covered it


def test_a_group_the_supervisor_does_not_lead_is_left_alone(monkeypatch):
    """Measured the hard way: started by hand with a PARENT_PID that was not its parent, this
    shim took the "parent died" branch on its first tick and SIGTERMed the shell that launched
    it plus an unrelated app sharing that process group. Boltz always starts it with
    start_new_session=True, so in production it does lead the group — but "in production it
    cannot happen" is what was believed before it happened."""
    sent = []
    monkeypatch.setattr(supervise.os, "getpgrp", lambda: 111)
    monkeypatch.setattr(supervise.os, "getpid", lambda: 222)
    monkeypatch.setattr(supervise.os, "killpg", lambda pgid, sig: sent.append((pgid, sig)))
    child = _Child()
    supervise._stop_group(child)
    assert sent == []
    assert child.terminated


def test_a_command_that_ignores_sigterm_is_killed(monkeypatch):
    monkeypatch.setattr(supervise.os, "getpgrp", lambda: 111)
    monkeypatch.setattr(supervise.os, "getpid", lambda: 222)
    monkeypatch.setattr(supervise.os, "killpg", lambda *a: None)
    child = _Child()
    child.wait = _raise_timeout
    supervise._stop_group(child)
    assert child.killed


def _raise_timeout(timeout=None):
    raise supervise.subprocess.TimeoutExpired("boltz", timeout or 0)


# ---- what it costs -------------------------------------------------------------------

def _spec(tokens):
    return {"token_estimate": tokens,
            "params": {"diffusion_samples": 1, "recycling_steps": 3, "sampling_steps": 200},
            "affinity_binder": None}


def test_the_submit_warning_carries_no_figures_that_move():
    """Terabytes-per-day and percent-of-life were on the submit button and were worse there.
    The decision at that point is go or do not go; a number that changes between two looks
    invites comparing them instead. The figures live in settings, and keeping them out of
    here also keeps smartctl off a path that runs on every keystroke."""
    out = estimate.estimate(_spec(1198), needs_msa_search=False, history=[])["memory"]
    assert out["level"] == "danger" and out["beyond_physical"] is True
    assert "swap_write" not in out
    assert out["applecare"] is False        # decides the wording, nothing else


def test_a_job_that_fits_is_not_lectured_about_the_drive():
    out = estimate.estimate(_spec(76), needs_msa_search=False, history=[])["memory"]
    assert out["level"] == "ok"
    assert out["beyond_physical"] is False


def test_brushing_swap_and_swimming_in_it_are_not_the_same_warning(monkeypatch):
    """Both are "danger", but one costs a slow afternoon and the other costs the drive.

    879 tokens lands at 20 GB on a 24 GB machine: tight, some paging, still finishes. 1,198
    lands at 31.4 GB, which is the size that ran for eighty minutes without finishing — every
    second of it paging out at 190 MB/s. The second one is what gets told not to run
    uninsured; saying that about the first would train the warning out of usefulness.
    """
    monkeypatch.setattr(estimate, "memory_gb", lambda: 24.0)
    tight = estimate.estimate(_spec(879), needs_msa_search=False, history=[])["memory"]
    over = estimate.estimate(_spec(1198), needs_msa_search=False, history=[])["memory"]
    assert tight["level"] == "danger" and tight["beyond_physical"] is False
    assert over["level"] == "danger" and over["beyond_physical"] is True
