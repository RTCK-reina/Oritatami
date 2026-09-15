"""Comparing scores computed under different conditions.

Boltz reports its own confidence, not a property of the molecule, so the same sequence
scores differently depending on how the prediction was run. On this machine's 893
finished predictions the MSA regime alone moved the mean core pLDDT from 96.21 (reused)
to 89.26 (single sequence) — fourteen times the 0.5 the autopilot calls an improvement.
"""
from oritatami import regime


def _result(msa=None, params=None, **kw):
    out = {"models": [{"plddt": {"A": [90.0]}}]}
    if msa is not None:
        out["msa"] = msa
    if params is not None:
        out["normalized_spec"] = {"params": params}
    out.update(kw)
    return out


PARAMS = {"diffusion_samples": 1, "recycling_steps": 3, "sampling_steps": 200,
          "use_potentials": False}


def test_the_msa_route_is_part_of_the_regime():
    reused = regime.of(_result(msa={"reused_from": "job_1"}, params=PARAMS))
    server = regime.of(_result(msa={"server": True}, params=PARAMS))
    single = regime.of(_result(msa={"single_sequence_chains": ["A"]}, params=PARAMS))
    assert {reused.msa, server.msa, single.msa} == {"reused", "server", "single"}
    assert not regime.comparable(reused, server)
    assert not regime.comparable(server, single)
    assert regime.comparable(reused, regime.of(_result(msa={"reused_from": "job_2"},
                                                       params=PARAMS)))


def test_sample_count_is_part_of_the_regime():
    one = regime.of(_result(msa={"reused_from": "j"}, params=PARAMS))
    three = regime.of(_result(msa={"reused_from": "j"}, params={**PARAMS, "diffusion_samples": 3}))
    assert not regime.comparable(one, three)
    assert "サンプル数 (1 → 3)" in regime.differences(three, one)


def test_an_unrecorded_regime_never_matches_a_recorded_one():
    """This is the gap a settings change hides in."""
    known = regime.of(_result(msa={"reused_from": "j"}, params=PARAMS))
    assert not regime.comparable(known, regime.UNKNOWN)
    assert not regime.comparable(regime.of({}), known)
    # two equally unrecorded ones are no worse off than before the guard existed
    assert regime.comparable(regime.UNKNOWN, regime.UNKNOWN)


def test_the_label_hides_the_defaults_and_shows_the_rest():
    plain = regime.of(_result(msa={"reused_from": "j"}, params=PARAMS))
    assert plain.label() == "MSA 使い回し"
    loud = regime.of(_result(msa={"server": True},
                             params={**PARAMS, "diffusion_samples": 3, "use_potentials": True}))
    assert "サンプル 3" in loud.label() and "ポテンシャル" in loud.label()


def test_a_mixed_population_is_reported_as_mixed():
    s = regime.summarize([
        _result(msa={"reused_from": "j"}, params=PARAMS),
        _result(msa={"reused_from": "k"}, params=PARAMS),
        _result(msa={"single_sequence_chains": ["A"]}, params=PARAMS),
    ])
    assert s["mixed"] is True
    assert s["regimes"][0]["count"] == 2 and s["regimes"][0]["msa"] == "reused"

    same = regime.summarize([_result(msa={"reused_from": "j"}, params=PARAMS)] * 3)
    assert same["mixed"] is False


def test_a_variant_is_not_announced_as_better_than_a_parent_run_differently(db_and_jobs):
    """Otherwise the settings change is reported as a discovery."""
    import time

    from oritatami import autopilot, config, system

    db = db_and_jobs
    config.update_settings({"autopilot_notify_improvement": True,
                            "autopilot_improvement_delta": 0.5})
    sent = []
    original = system.notify
    system.notify = lambda *a, **k: sent.append(a)
    try:
        parent = db.insert_job(kind="predict", title="親", spec={}, parent_id=None, origin="user")
        db.update_job(parent["id"], status="succeeded", finished_at=time.time(),
                      result={"models": [{"plddt": {"A": [90.0]}, "confidence": {}}],
                              "msa": {"single_sequence_chains": ["A"]},
                              "normalized_spec": {"params": PARAMS}})
        child = {"id": "job_child", "title": "子", "parent_id": parent["id"],
                 "result": {"models": [{"plddt": {"A": [95.0]}, "confidence": {}}],
                            "msa": {"reused_from": parent["id"]},
                            "normalized_spec": {"params": PARAMS}}}
        autopilot._notify_if_improved(child, db)
        assert sent == [], "条件が違う親子で改善を通知しないこと"

        child["result"]["msa"] = {"single_sequence_chains": ["A"]}
        autopilot._notify_if_improved(child, db)
        assert len(sent) == 1, "条件が同じなら通知すること"
    finally:
        system.notify = original
