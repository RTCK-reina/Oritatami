"""Reading the run's own history back.

893 predictions were made and none of them were ever read back. These pin the three
things that were in there: 9.4% of predictions recomputed a sequence already computed,
the repeat spread (sigma 0.219) is half the improvement threshold, and which position is
mutated predicts the outcome.
"""
import time

from oritatami import autopilot, history, regime
from oritatami.db import Database

UBQ = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
PARAMS = {"diffusion_samples": 1, "recycling_steps": 3, "sampling_steps": 200,
          "use_potentials": False}


def _score(result):
    return autopilot.metric_value(result, autopilot.resolve_metric(result))


def _add(db, seq, plddt, parent=None, mutation=None, msa=None):
    spec = {"components": [{"type": "protein", "label": "U", "chains": ["A"], "sequence": seq}]}
    if mutation:
        spec["autopilot_mutation"] = mutation
    j = db.insert_job(kind="predict", title=mutation or "seed", spec=spec,
                      parent_id=parent, origin="autopilot_variant")
    db.update_job(j["id"], status="succeeded", finished_at=time.time(), result={
        "models": [{"plddt": {"A": [plddt] * len(seq)}, "ligand_plddt": {}, "confidence": {}}],
        "msa": msa or {"reused_from": "j0"},
        "normalized_spec": {"params": PARAMS, "components": spec["components"]},
    })
    history.invalidate()
    return db.get_job(j["id"])


def test_repeats_of_one_sequence_give_the_noise_floor(tmp_path):
    db = Database(tmp_path / "t.sqlite3")
    for v in (95.0, 95.4, 94.6, 95.2):          # same molecule, four times
        _add(db, UBQ, v)
    data = history.build(db, _score)
    assert data["repeats"] == 3
    assert data["sigma"] > 0
    # the threshold the noise supports, not the one someone typed
    assert history.suggested_delta(data) == round(2 * data["sigma"] * 2 ** 0.5, 2)
    assert not history.detectable(data, 0.05)
    assert history.detectable(data, 5.0)


def test_a_sequence_already_predicted_is_not_predicted_again(tmp_path):
    """Two branches reach the same molecule by different routes; the loop paid twice."""
    db = Database(tmp_path / "t.sqlite3")
    mutated = UBQ[:8] + "E" + UBQ[9:]
    _add(db, mutated, 96.0)
    data = history.build(db, _score)
    key = history.sequence_key([mutated], [], regime.of({
        "msa": {"reused_from": "j0"}, "normalized_spec": {"params": PARAMS}}))
    assert key in data["known"]
    assert data["known"][key][0] == 96.0
    # a different sequence is not a hit
    other = history.sequence_key([UBQ], [], regime.of({
        "msa": {"reused_from": "j0"}, "normalized_spec": {"params": PARAMS}}))
    assert other not in data["known"]


def test_the_same_sequence_under_other_conditions_is_a_different_experiment(tmp_path):
    db = Database(tmp_path / "t.sqlite3")
    _add(db, UBQ, 96.0, msa={"reused_from": "j0"})
    data = history.build(db, _score)
    single = history.sequence_key([UBQ], [], regime.of({
        "msa": {"single_sequence_chains": ["A"]}, "normalized_spec": {"params": PARAMS}}))
    assert single not in data["known"], "条件が違えば測り直す価値がある"


def test_deltas_are_grouped_by_the_position_mutated(tmp_path):
    db = Database(tmp_path / "t.sqlite3")
    seed = _add(db, UBQ, 95.0)
    _add(db, UBQ[:8] + "E" + UBQ[9:], 94.0, parent=seed["id"], mutation="T9E")
    _add(db, UBQ[:8] + "A" + UBQ[9:], 94.4, parent=seed["id"], mutation="T9A")
    _add(db, UBQ[:74] + "C" + UBQ[75:], 92.0, parent=seed["id"], mutation="G75C")
    data = history.build(db, _score)
    assert data["pairs"] == 3
    assert data["by_position"][9]["n"] == 2
    assert data["by_position"][75]["mean"] < data["by_position"][9]["mean"]
    assert history.position_prior(data, 9, min_n=2) < 0
    assert history.position_prior(data, 75, min_n=5) is None, "件数が足りなければ使わない"


def test_a_pair_across_a_settings_change_is_not_counted(tmp_path):
    db = Database(tmp_path / "t.sqlite3")
    seed = _add(db, UBQ, 90.0, msa={"single_sequence_chains": ["A"]})
    _add(db, UBQ[:8] + "E" + UBQ[9:], 96.0, parent=seed["id"], mutation="T9E",
         msa={"reused_from": "j0"})
    data = history.build(db, _score)
    assert data["pairs"] == 0 and data["skipped_regime"] == 1
    assert data["by_position"] == {}


def test_the_endpoint_reports_the_threshold_the_noise_supports():
    from fastapi.testclient import TestClient
    from oritatami.app import app

    with TestClient(app) as client:
        body = client.get("/api/history").json()
    for key in ("predictions", "pairs", "sigma", "configured_delta", "suggested_delta",
                "worst_positions", "repeated_substitutions", "distinct_experiments"):
        assert key in body, key
    assert "known" not in body, "重い内部データを UI に送らないこと"


def test_the_strategy_that_produced_an_attempt_is_recorded(tmp_path, monkeypatch):
    """Unrecorded, comparing hill climb with random walk meant splitting the history by
    date — and the two periods differ in sequence, depth and settings too."""
    from oritatami import config

    config.update_settings({"autopilot_strategy": "walk", "autopilot_experiment": ""})
    spec = {"name": "u", "workbench_name": "u",
            "components": [{"type": "protein", "label": "U", "chains": ["A"], "sequence": UBQ}]}
    results = {"analyses": [{"mode": "mutations", "chain": "A", "proposals": [
        {"type": "mutation_set", "status": "ok", "chain": "A",
         "apply": {"action": "mutate", "chain": "A", "mutations": ["A:T9A"]}}]}]}
    cands = autopilot._collect_variant_candidates({"spec": spec, "id": "j1"}, results)
    assert cands and cands[0]["spec"]["autopilot_strategy"] == "walk"
    assert "autopilot_climb_patience" not in cands[0]["spec"]

    config.update_settings({"autopilot_strategy": "climb", "autopilot_climb_patience": 8})
    cands = autopilot._collect_variant_candidates({"spec": spec, "id": "j1"}, results)
    assert cands[0]["spec"]["autopilot_strategy"] == "climb"
    assert cands[0]["spec"]["autopilot_climb_patience"] == 8


def test_deltas_are_split_by_the_strategy_that_produced_them(tmp_path):
    db = Database(tmp_path / "t.sqlite3")
    seed = _add(db, UBQ, 95.0)

    def child(seq, score, strategy):
        spec = {"components": [{"type": "protein", "label": "U", "chains": ["A"],
                                "sequence": seq}],
                "autopilot_mutation": "T9E", "autopilot_strategy": strategy}
        j = db.insert_job(kind="predict", title=strategy, spec=spec,
                          parent_id=seed["id"], origin="autopilot_variant")
        db.update_job(j["id"], status="succeeded", finished_at=time.time(), result={
            "models": [{"plddt": {"A": [score] * len(seq)}, "ligand_plddt": {}, "confidence": {}}],
            "msa": {"reused_from": "j0"},
            "normalized_spec": {"params": PARAMS, "components": spec["components"]}})

    child(UBQ[:8] + "E" + UBQ[9:], 96.0, "climb")
    child(UBQ[:8] + "A" + UBQ[9:], 94.0, "walk")
    history.invalidate()
    data = history.build(db, _score)
    assert data["by_strategy"]["climb"]["improved"] == 1.0
    assert data["by_strategy"]["walk"]["improved"] == 0.0


def test_how_many_predictions_a_strategy_comparison_would_need():
    """One attempt in ten improves anything, so the comparison is expensive."""
    assert 90 <= history.required_samples(0.099, 2.0) <= 110
    assert history.required_samples(0.099, 1.5) > 300
    assert history.required_samples(0.0) is None


def test_explore_mode_prefers_positions_the_run_has_barely_tried(tmp_path, monkeypatch):
    """Greedy ranking kept returning to the same residues: 23% of this run's attempts
    went to six of them."""
    from oritatami import config

    db = Database(tmp_path / "t.sqlite3")
    seed = _add(db, UBQ, 95.0)
    for i, score in enumerate((94.0, 94.2, 94.4, 94.1)):      # position 9, tried often
        _add(db, UBQ[:8] + "EAQV"[i] + UBQ[9:], score, parent=seed["id"], mutation="T9" + "EAQV"[i])
    history.invalidate()

    tried = {"mut_str": "T9E", "llr": 3.0, "chain": "A", "spec": {}}
    fresh = {"mut_str": "S57A", "llr": 1.0, "chain": "A", "spec": {}}

    config.update_settings({"autopilot_selection": "greedy"})
    out = autopilot._apply_history(db, [dict(tried), dict(fresh)])
    assert out[0]["mut_str"] == "T9E", "良さそうな順なら ESM-2 が高いほうが先"

    config.update_settings({"autopilot_selection": "explore"})
    out = autopilot._apply_history(db, [dict(tried), dict(fresh)])
    assert out[0]["mut_str"] == "S57A", "探索モードなら手つかずの位置が先"
    assert out[0]["position_tries"] == 0
    config.update_settings({"autopilot_selection": "greedy"})


def test_the_pareto_front_keeps_the_cheap_result_the_column_hides():
    """'+0.9 with sixteen mutations' and '+0.7 with two' are not the same result."""
    from fastapi.testclient import TestClient
    from oritatami.app import app

    with TestClient(app) as client:
        body = client.get("/api/leaderboard", params={"limit": 5}).json()
    assert "pareto_count" in body
    for row in body["rows"]:
        assert "pareto" in row and "lineage_root" in row


def test_the_front_is_drawn_inside_a_lineage_not_across_molecules():
    """A carbonic anhydrase complex at 98 with no mutations dominates every ubiquitin
    variant ever made, which says nothing about either."""
    from oritatami.app import pareto_front

    rows = [
        {"id": "other", "mutations": [], "m": 98.0},          # 別の分子・別系統
        {"id": "seed", "mutations": [], "m": 95.0},
        {"id": "cheap", "mutations": ["T9E"], "m": 96.0},     # 1 変異で 96
        {"id": "rich", "mutations": ["T9E", "E24A"], "m": 95.5},   # 2 変異で 95.5 = 負け
        {"id": "best", "mutations": ["T9E", "E24A", "S57A"], "m": 96.4},
    ]
    parents = {"other": None, "seed": None, "cheap": "seed", "rich": "cheap", "best": "rich"}
    front, roots = pareto_front(rows, "m", parents)

    assert "cheap" in front, "同じ系統で、より少ない変異で勝っているものは残る"
    assert "best" in front, "変異は多いが最高点なので残る"
    assert "rich" not in front, "1 変異少ない cheap に負けている"
    assert "other" not in front, "1 件だけの系統は前線にしない (比べる相手がいない)"
    assert roots["best"] == "seed" and roots["other"] == "other"


def test_a_cycle_in_the_lineage_does_not_hang_the_front():
    """Job ids come from the database; a broken parent link must not spin."""
    from oritatami.app import pareto_front

    rows = [{"id": "a", "mutations": [], "m": 1.0}, {"id": "b", "mutations": [], "m": 2.0}]
    front, roots = pareto_front(rows, "m", {"a": "b", "b": "a"})
    assert set(roots) == {"a", "b"}
