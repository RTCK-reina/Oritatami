"""A finished prediction can still be unusable: NaN coordinates, atoms inside each other.

The confidence numbers do not say so — a NaN residue carries a pLDDT like any other — so the
warning has to come from the geometry block, and it has to reach the same place the
function-risk warnings do.
"""

from oritatami import function_risk


def _job(geometry, **extra):
    return {"id": "j1", "status": "succeeded",
            "result": {"models": [{"index": 0}], "chains": [{"chain": "A"}], "geometry": geometry},
            "spec": {"components": []}, **extra}


def test_nan_coordinates_are_critical():
    f = function_risk._geometry_findings(_job({
        "model_index": 0, "atoms": 600, "nonfinite_atoms": 9,
        "nonfinite_examples": ["A/LYS11/CA"], "other_models_nonfinite": [],
        "clashes": None, "severe_clashes": None, "clashscore": None,
        "worst_clashes": [], "chain_breaks": [],
    })["result"])
    assert [x["severity"] for x in f] == ["critical"]
    assert "NaN" not in f[0]["text"]          # written for a person, not a stack trace
    assert "CPU" in f[0]["text"]              # and it says what to actually do


def test_nan_in_another_sample_is_still_reported():
    f = function_risk._geometry_findings(_job({
        "model_index": 0, "atoms": 600, "nonfinite_atoms": 0, "nonfinite_examples": [],
        "other_models_nonfinite": [{"model_index": 2, "nonfinite_atoms": 4}],
        "clashes": 0, "severe_clashes": 0, "clashscore": 0.0,
        "worst_clashes": [], "chain_breaks": [],
    })["result"])
    assert len(f) == 1 and f[0]["kind"] == "nonfinite"
    assert "モデル 2" in f[0]["text"]


def test_severe_clash_warns_and_names_the_pair():
    f = function_risk._geometry_findings(_job({
        "model_index": 0, "atoms": 600, "nonfinite_atoms": 0, "nonfinite_examples": [],
        "other_models_nonfinite": [], "clashes": 5, "severe_clashes": 2, "clashscore": 8.3,
        "worst_clashes": [{"a": "A/LYS6/CB", "b": "A/GLN41/CA", "dist": 1.26, "overlap": 1.74}],
        "chain_breaks": [],
    })["result"])
    assert [x["kind"] for x in f] == ["clash"] and f[0]["severity"] == "high"
    assert "A/LYS6/CB" in f[0]["text"]


def test_a_handful_of_minor_clashes_is_not_a_warning():
    """Median clashscore on this machine is 0 and the 90th percentile is 4.9; 1.7 is normal."""
    f = function_risk._geometry_findings(_job({
        "model_index": 0, "atoms": 600, "nonfinite_atoms": 0, "nonfinite_examples": [],
        "other_models_nonfinite": [], "clashes": 1, "severe_clashes": 0, "clashscore": 1.7,
        "worst_clashes": [], "chain_breaks": [],
    })["result"])
    assert f == []


def test_many_shallow_clashes_warn_at_a_lower_severity():
    f = function_risk._geometry_findings(_job({
        "model_index": 0, "atoms": 600, "nonfinite_atoms": 0, "nonfinite_examples": [],
        "other_models_nonfinite": [], "clashes": 30, "severe_clashes": 0, "clashscore": 50.0,
        "worst_clashes": [], "chain_breaks": [],
    })["result"])
    assert [x["severity"] for x in f] == ["medium"]


def test_broken_backbone_is_reported():
    f = function_risk._geometry_findings(_job({
        "model_index": 0, "atoms": 600, "nonfinite_atoms": 0, "nonfinite_examples": [],
        "other_models_nonfinite": [], "clashes": 0, "severe_clashes": 0, "clashscore": 0.0,
        "worst_clashes": [],
        "chain_breaks": [{"chain": "A", "after": "I30", "before": "Q31", "dist": 22.1, "limit": 5.0}],
    })["result"])
    assert [x["kind"] for x in f] == ["chain_break"] and f[0]["severity"] == "high"


def test_missing_or_failed_geometry_says_nothing():
    assert function_risk._geometry_findings({"geometry": None}) == []
    assert function_risk._geometry_findings({"geometry": {"error": "boom"}}) == []
    assert function_risk._geometry_findings({}) == []


def test_assess_surfaces_geometry_and_ranks_it_first(db_and_jobs):
    db = db_and_jobs
    job = _job({
        "model_index": 0, "atoms": 600, "nonfinite_atoms": 3, "nonfinite_examples": [],
        "other_models_nonfinite": [], "clashes": None, "severe_clashes": None,
        "clashscore": None, "worst_clashes": [], "chain_breaks": [],
    })
    out = function_risk.assess(db, job)
    assert out["level"] == "danger"
    assert out["findings"][0]["kind"] == "nonfinite"
