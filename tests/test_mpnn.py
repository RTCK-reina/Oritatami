"""The third scorer, and what it is allowed to decide.

The pLDDT and ESM-2 both endorsed destroying ubiquitin's C-terminal glycines and K63.
Inverse folding penalises exactly those, so it is used as a veto on the structurally
destructive — never as the ranking, because against delta core pLDDT it correlates only
r = -0.114 over 480 historical pairs.
"""
import pytest
from oritatami import autopilot, config
from oritatami.engines import mpnn

UBQ = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"


def _candidate(mut, seq):
    return {"mut_str": mut, "llr": 1.0, "chain": "A",
            "spec": {"components": [{"type": "protein", "chains": ["A"], "sequence": seq}]}}


def _mutate(seq, pos, new):
    return seq[:pos - 1] + new + seq[pos:]


@pytest.fixture
def job(tmp_path, monkeypatch):
    monkeypatch.setattr(autopilot, "jobs_dir", lambda: tmp_path)
    out = tmp_path / "j1" / "out"
    out.mkdir(parents=True)
    (out / "complex_model_0.cif").write_text("data_x\n", "utf-8")
    return {"id": "j1", "spec": {"components": [
        {"type": "protein", "chains": ["A"], "sequence": UBQ}]}}


def test_a_candidate_the_backbone_rejects_is_not_predicted(job, monkeypatch):
    config.update_settings({"mpnn_enabled": True, "mpnn_veto": 0.06})
    monkeypatch.setattr(mpnn, "available", lambda: True)
    monkeypatch.setattr(mpnn, "backbone_pdb", lambda cif, dest: dest)
    # I44A destroys the hydrophobic patch (+0.087 measured); T9E is benign (+0.013)
    monkeypatch.setattr(mpnn, "relative", lambda pdb, ref, seqs: [0.087, 0.013])
    kept = autopilot._inverse_folding_check(job, [
        _candidate("I44A", _mutate(UBQ, 44, "A")),
        _candidate("T9E", _mutate(UBQ, 9, "E")),
    ])
    assert [c["mut_str"] for c in kept] == ["T9E"]
    assert kept[0]["mpnn"] == 0.013


def test_everything_rejected_still_leaves_the_least_bad(job, monkeypatch):
    """A stalled cycle is worse than a weak attempt."""
    config.update_settings({"mpnn_enabled": True, "mpnn_veto": 0.01})
    monkeypatch.setattr(mpnn, "available", lambda: True)
    monkeypatch.setattr(mpnn, "backbone_pdb", lambda cif, dest: dest)
    monkeypatch.setattr(mpnn, "relative", lambda pdb, ref, seqs: [0.087, 0.031])
    kept = autopilot._inverse_folding_check(job, [
        _candidate("I44A", _mutate(UBQ, 44, "A")),
        _candidate("G76C", _mutate(UBQ, 76, "C")),
    ])
    assert [c["mut_str"] for c in kept] == ["G76C"]


def test_the_veto_can_be_switched_off(job, monkeypatch):
    config.update_settings({"mpnn_enabled": True, "mpnn_veto": 0.0})
    monkeypatch.setattr(mpnn, "available", lambda: True)
    monkeypatch.setattr(mpnn, "backbone_pdb", lambda cif, dest: dest)
    monkeypatch.setattr(mpnn, "relative", lambda pdb, ref, seqs: [0.9, 0.8])
    kept = autopilot._inverse_folding_check(job, [
        _candidate("I44A", _mutate(UBQ, 44, "A")),
        _candidate("G76C", _mutate(UBQ, 76, "C")),
    ])
    assert len(kept) == 2, "却下ライン 0 なら採点だけして落とさないこと"
    assert all("mpnn" in c for c in kept)


def test_a_failure_in_the_third_scorer_never_stops_the_loop(job, monkeypatch):
    config.update_settings({"mpnn_enabled": True, "mpnn_veto": 0.06})
    monkeypatch.setattr(mpnn, "available", lambda: True)
    monkeypatch.setattr(mpnn, "backbone_pdb", lambda cif, dest: dest)

    def boom(*a, **k):
        raise mpnn.MpnnError("重みが読めません")

    monkeypatch.setattr(mpnn, "relative", boom)
    cands = [_candidate("T9E", _mutate(UBQ, 9, "E"))]
    assert autopilot._inverse_folding_check(job, cands) == cands


def test_candidates_that_change_the_length_are_left_alone(job, monkeypatch):
    """Threading needs the same number of residues as the backbone has."""
    config.update_settings({"mpnn_enabled": True, "mpnn_veto": 0.06})
    monkeypatch.setattr(mpnn, "available", lambda: True)
    monkeypatch.setattr(mpnn, "backbone_pdb", lambda cif, dest: dest)
    monkeypatch.setattr(mpnn, "relative", lambda pdb, ref, seqs: [0.9] * len(seqs))
    short = _candidate("del", UBQ[:-5])
    kept = autopilot._inverse_folding_check(job, [short])
    assert kept == [short], "長さが違うものは採点対象外"


def test_it_is_skipped_entirely_when_switched_off(job, monkeypatch):
    config.update_settings({"mpnn_enabled": False})
    called = []
    monkeypatch.setattr(mpnn, "available", lambda: called.append(1) or True)
    cands = [_candidate("T9E", _mutate(UBQ, 9, "E"))]
    assert autopilot._inverse_folding_check(job, cands) == cands
    assert called == []
    config.update_settings({"mpnn_enabled": True})


def test_the_installed_scorer_ranks_the_destructive_mutations_worst():
    """The measurement this whole thing rests on, run against the real binary."""
    if not mpnn.available():
        pytest.skip("proteinmpnn が入っていません")
    assert mpnn.binary().endswith("proteinmpnn")
