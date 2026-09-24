"""What a prediction costs in memory, and what happens when it does not fit.

Measured on this machine 2026-09-14 with incompressible pages (macOS compresses before it
swaps, so a test written with zeroed pages measures the compressor and never reaches
swap):

    8 GB resident   swap 0            97,000 MB/s of page traffic
    16 GB           swap 6.9 GB          296 MB/s      — 328x slower

The machine does not fall over; it becomes unusable instead. So the estimate that decides
whether to warn before submitting matters more than any crash handling, and until now it
was extrapolated from a single observation and never learned from what actually ran.
"""
from oritatami import estimate


def _spec(tokens, affinity=False):
    return {"token_estimate": tokens, "params": {"diffusion_samples": 1, "recycling_steps": 3,
                                                 "sampling_steps": 200},
            "affinity_binder": "B" if affinity else None}


def _job(tokens, peak_gb, seconds=60.0, method="footprint"):
    return {"result": {"token_estimate": tokens, "elapsed_sec": seconds,
                       "peak_memory_gb": peak_gb, "peak_memory_method": method,
                       "accelerator": "mps",
                       "timings": {"structure": seconds},
                       "normalized_spec": {"params": {"diffusion_samples": 1,
                                                      "recycling_steps": 3,
                                                      "sampling_steps": 200},
                                           "token_estimate": tokens}}}


def test_memory_is_estimated_and_graded(monkeypatch):
    monkeypatch.setattr(estimate, "memory_gb", lambda: 24.0)
    out = estimate.estimate(_spec(76), needs_msa_search=False, history=[])
    assert out["memory"]["peak_gb"] > 0
    assert out["memory"]["level"] in ("ok", "caution", "danger")

    small = estimate.estimate(_spec(76), needs_msa_search=False, history=[])
    huge = estimate.estimate(_spec(3000), needs_msa_search=False, history=[])
    assert huge["memory"]["peak_gb"] > small["memory"]["peak_gb"]
    assert huge["memory"]["level"] == "danger"


def test_the_memory_estimate_learns_from_what_actually_ran():
    """The time estimate is fitted from history; the memory estimate was a constant
    extrapolated from one job and never improved."""
    history = [_job(290, 13.0), _job(580, 40.0), _job(150, 5.0)]
    fitted = estimate.estimate(_spec(580), needs_msa_search=False, history=history)
    blind = estimate.estimate(_spec(580), needs_msa_search=False, history=[])
    assert fitted["memory"]["basis"] == "history"
    assert blind["memory"]["basis"] == "default"
    assert abs(fitted["memory"]["peak_gb"] - 40.0) < abs(blind["memory"]["peak_gb"] - 40.0), \
        "実測が 3 件あるなら、外挿より実測に寄ること"


def test_a_job_without_a_measurement_does_not_poison_the_fit():
    history = [_job(290, 13.0), _job(580, 40.0), {"result": {"token_estimate": 400}}]
    out = estimate.estimate(_spec(580), needs_msa_search=False, history=history)
    assert out["memory"]["basis"] == "history"
    assert out["memory"]["samples"] == 2


def test_a_narrow_cluster_of_small_jobs_does_not_set_the_slope():
    """Measured on this machine: three real runs at 51 and 76 tokens (2.59, 3.47, 4.85 GB).

    Least squares over that cluster gives k=48 GB per unit, so a 290-token prediction was
    estimated at 50 GB — four times the ~13 GB a job that size actually costs — and every
    ordinary prediction came up red. The slope is not identifiable from sizes that close
    together, so the measurements may move the offset and nothing else.
    """
    history = [_job(51, 2.59), _job(51, 3.47), _job(76, 4.85)]
    out = estimate.estimate(_spec(290), needs_msa_search=False, history=history)
    assert out["memory"]["basis"] == "history"
    assert out["memory"]["samples"] == 3
    assert out["memory"]["peak_gb"] < 20, "近いサイズだけから傾きを当ててはならない"

    # It still tracks what it measured at the sizes it has seen.
    near = estimate.estimate(_spec(76), needs_msa_search=False, history=history)
    assert abs(near["memory"]["peak_gb"] - 4.85) < 2.0


def test_an_absurd_fitted_slope_is_rejected():
    """Wide sizes but nonsense values: the fit is discarded in favour of the prior."""
    history = [_job(100, 2.0), _job(300, 2.1), _job(900, 900.0)]
    out = estimate.estimate(_spec(300), needs_msa_search=False, history=history)
    assert out["memory"]["peak_gb"] < 50


def test_resident_size_readings_are_not_used():
    """The first supervisor summed RSS. On Metal that is not a small number, it is a wrong one.

    Measured here on one 1,696-residue three-chain complex, at the same instant: ps reported
    6.5 MB resident while the kernel reported a 27 GB footprint and a 30 GB lifetime peak — the
    tensors live in IOAccelerator allocations that never enter RSS. Those rows stay in the
    database (nothing is deleted) but must not reach the fit.
    """
    stale = [_job(290, 2.1, method=None), _job(580, 3.0, method=None), _job(150, 1.2, method=None)]
    out = estimate.estimate(_spec(580), needs_msa_search=False, history=stale)
    assert out["memory"]["basis"] == "default"
    assert out["memory"]["samples"] == 0

    mixed = stale + [_job(290, 13.0), _job(580, 40.0), _job(150, 5.0)]
    out = estimate.estimate(_spec(580), needs_msa_search=False, history=mixed)
    assert out["memory"]["samples"] == 3, "footprint で測った 3 件だけを使うこと"
    assert abs(out["memory"]["peak_gb"] - 40.0) < 5.0


def test_the_default_model_matches_what_this_machine_actually_did():
    """Three runs measured 2026-09-14 (single-sequence MSA, 1 sample, 3 recycles, 200 steps).

    The defaults used to be five times too steep — 609 tokens came out at 34 GB against a
    measured 14.8 GB — so every job past about 400 residues showed a red warning and the
    warning stopped meaning anything.
    """
    for tokens, measured, tolerance in ((76, 5.52, 2.0), (609, 14.81, 3.0), (1198, 31.0, 4.0)):
        out = estimate.estimate(_spec(tokens), needs_msa_search=False, history=[])
        assert out["memory"]["basis"] == "default"
        assert abs(out["memory"]["peak_gb"] - measured) < tolerance, (tokens, out["memory"])


def test_the_grades_land_where_the_measurements_do(monkeypatch):
    """609 tokens finished in five minutes; 1,198 did not finish in eighty. The line is between."""
    # The grades are fractions of installed memory, so pin the machine the measurements
    # were taken on — on a smaller machine the same peaks legitimately grade higher.
    monkeypatch.setattr(estimate, "memory_gb", lambda: 24.0)
    ok = estimate.estimate(_spec(609), needs_msa_search=False, history=[])["memory"]
    bad = estimate.estimate(_spec(1198), needs_msa_search=False, history=[])["memory"]
    assert ok["level"] in ("ok", "caution"), "5 分で終わる仕事を danger にしない"
    assert bad["level"] == "danger"
