import json

import pytest
import yaml
from oritatami.engines import boltz

UBQ = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"


def spec(**kw):
    base = {
        "name": "t",
        "components": [
            {"type": "protein", "sequence": UBQ, "label": "UBQ", "copies": 2},
            {"type": "ligand", "smiles": "CC(=O)Nc1nnc(s1)S(N)(=O)=O", "label": "AZM"},
        ],
    }
    base.update(kw)
    return base


def test_normalize_assigns_chains_and_defaults():
    n = boltz.normalize_spec(spec())
    assert n["components"][0]["chains"] == ["A", "B"]
    assert n["components"][1]["chains"] == ["C"]
    assert n["components"][0]["msa"] == "server"
    assert n["params"]["diffusion_samples"] == 1


def test_normalize_respects_explicit_chains():
    s = spec()
    s["components"][1]["chains"] = ["A"]
    s["components"][0]["copies"] = 1
    n = boltz.normalize_spec(s)
    assert n["components"][1]["chains"] == ["A"]
    assert n["components"][0]["chains"] == ["B"]


def test_affinity_binder_must_be_single_ligand():
    with pytest.raises(ValueError, match="リガンド"):
        boltz.normalize_spec(spec(affinity_binder="A"))
    n = boltz.normalize_spec(spec(affinity_binder="C"))
    assert n["affinity_binder"] == "C"


def test_ligand_needs_exactly_one_identifier():
    s = spec()
    s["components"][1] = {"type": "ligand", "label": "x"}
    with pytest.raises(ValueError, match="どちらか一方"):
        boltz.normalize_spec(s)


def test_invalid_smiles_is_reported():
    s = spec()
    s["components"][1]["smiles"] = "C1CC("
    with pytest.raises(ValueError, match="SMILES"):
        boltz.normalize_spec(s)


def test_malformed_constraints_are_input_errors_not_crashes():
    """A bad constraint shape used to surface as a KeyError→404 or TypeError→500."""
    s = spec(constraints=[{"type": "contact"}])
    with pytest.raises(ValueError, match="チェーン ID"):
        boltz.normalize_spec(s)
    s = spec(constraints=[{"type": "pocket", "binder": "C", "contacts": [["A"]]}])
    with pytest.raises(ValueError, match="チェーン ID"):
        boltz.normalize_spec(s)
    s = spec(constraints=[{"type": "contact", "token1": ["A", "x"], "token2": ["B", 1]}])
    with pytest.raises(ValueError, match="チェーン ID"):
        boltz.normalize_spec(s)
    s = spec(constraints=["not-a-dict"])
    with pytest.raises(ValueError, match="オブジェクト"):
        boltz.normalize_spec(s)
    n = boltz.normalize_spec(spec(constraints=[{"type": "contact", "token1": ["A", 3],
                                              "token2": ["C", 5], "max_distance": 8.0}]))
    assert n["constraints"] == [{"type": "contact", "token1": ["A", 3], "token2": ["C", 5],
                                 "max_distance": 8.0, "force": False}]


def test_build_yaml_shape():
    n = boltz.normalize_spec(spec(affinity_binder="C"))
    n["components"][0]["msa"] = "single"
    doc = yaml.safe_load(boltz.build_yaml(n, None))
    assert doc["sequences"][0]["protein"]["id"] == ["A", "B"]
    assert doc["sequences"][0]["protein"]["msa"] == "empty"
    assert doc["sequences"][1]["ligand"]["id"] == "C"
    assert doc["properties"] == [{"affinity": {"binder": "C"}}]


def test_msa_reuse_for_point_mutant(isolated_home, tmp_path):
    n = boltz.normalize_spec({"components": [{"type": "protein", "sequence": UBQ}]})
    # fake a finished job with a Boltz-style MSA csv
    root = tmp_path / "out"
    (root / "msa").mkdir(parents=True)
    (root / "msa" / "complex_0.csv").write_text("key,sequence\n-1," + UBQ + "\n-1," + UBQ.replace("K", "R", 1))
    boltz._store_msa(tmp_path, n, root, "job_1")
    mutant = UBQ[:47] + "R" + UBQ[48:]
    m = boltz.normalize_spec({"components": [{"type": "protein", "sequence": mutant}]})
    entry = boltz.find_reusable_msa(m)
    assert entry is not None and entry["job_id"] == "job_1" and entry["differences"] == 1
    paths = boltz._write_reused_msa(entry, [mutant], tmp_path / "reuse")
    lines = paths[0].read_text().splitlines()
    assert lines[0] == "key,sequence" and lines[1] == "-1," + mutant
    # a different protein must not reuse it
    other = boltz.normalize_spec({"components": [{"type": "protein", "sequence": "M" * len(UBQ)}]})
    assert boltz.find_reusable_msa(other) is None


def test_scan_log_phases_only_move_forward(tmp_path):
    log = tmp_path / "boltz.log"
    log.write_text("Checking input data.\nCalling MSA server for target x\nRunning structure prediction for 1 input.\n"
                   "Predicting property: affinity\nChecking input data.\n")
    seen = []
    timings: dict[str, float] = {}
    tracker = boltz._PhaseTracker(lambda k, lbl: seen.append(k), timings, 0.0)
    pos = boltz._scan_log(log, 0, tracker)
    assert pos == log.stat().st_size
    assert seen == ["preprocess", "msa", "structure", "affinity"]
    assert tracker.phase == "affinity"
    tracker.finish()
    assert set(timings) == {"preprocess", "msa", "structure", "affinity"}


def test_scan_log_reports_msa_server_queue(tmp_path):
    log = tmp_path / "boltz.log"
    log.write_text("Calling MSA server for target x\nSleeping for 7s. Reason: RATELIMIT\n")
    labels = []
    tracker = boltz._PhaseTracker(lambda k, lbl: labels.append(lbl), {}, 0.0)
    boltz._scan_log(log, 0, tracker)
    assert labels[-1].endswith("サーバー混雑のため待機中")


def test_collect_results_on_real_output(tmp_path):
    from conftest import DATA

    pred = DATA / "ubq_prediction"
    if not pred.exists():
        pytest.skip("prediction fixture missing")
    n = boltz.normalize_spec({"components": [{"type": "protein", "sequence": UBQ}]})
    res = boltz.collect_results(n, DATA, pred)
    assert len(res["models"]) == 1
    m = res["models"][0]
    assert len(m["plddt"]["A"]) == len(UBQ)
    assert 0 < m["confidence"]["ptm"] <= 1
    assert res["pae"]["size"] == len(UBQ)
    assert res["pae"]["segments"] == [{"chain": "A", "kind": "polymer", "start": 0, "end": len(UBQ)}]
    json.dumps(res)
