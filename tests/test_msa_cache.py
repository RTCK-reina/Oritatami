"""The MSA cache decides whether a variant costs 33 s or 63 s, and how far the alignment
it reuses may be from the sequence it was computed for."""
import json

import pytest
from oritatami.config import msa_cache_dir
from oritatami.engines import boltz

UBQ = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"


def _spec(seq):
    return {"components": [{"type": "protein", "sequence": seq, "msa": "server",
                            "chains": ["A"], "label": "U"}]}


def _mutate(seq, n):
    """n substitutions, spread out so they never collide."""
    out = list(seq)
    for i in range(n):
        p = i * 3
        out[p] = "A" if out[p] != "A" else "G"
    return "".join(out)


@pytest.fixture
def cached(tmp_path, monkeypatch):
    """One cached alignment, fetched for the wild type."""
    d = msa_cache_dir()
    (d / "wt_0.csv").write_text(f"h\n101,{UBQ}\n", "utf-8")
    (d / "index.json").write_text(json.dumps(
        [{"job_id": "wt", "sequences": [UBQ], "origin": [UBQ], "files": ["wt_0.csv"],
          "created": 0}]), "utf-8")
    return d


def test_near_variants_reuse_and_far_ones_refetch(cached):
    assert boltz.find_reusable_msa(_spec(UBQ)) is not None, "同一配列は当然再利用"
    assert boltz.find_reusable_msa(_spec(_mutate(UBQ, 3))) is not None, "3置換は窓の中"
    # 76 residues: the one-hop window is 3 substitutions
    assert boltz.find_reusable_msa(_spec(_mutate(UBQ, 8))) is None, "8置換は1ホップの窓の外"
    assert boltz.find_reusable_msa(_spec(UBQ[:-10])) is None, "長さが違えば使えない"


def test_reanchoring_keeps_the_lineage_reusing_but_bounds_total_drift(cached):
    """Without re-anchoring a lineage leaves the window every couple of generations;
    with unbounded re-anchoring it would drift to 16+ substitutions, which measured
    -0.22 pLDDT against a fresh alignment."""
    seq3 = _mutate(UBQ, 3)
    boltz._reanchor_msa(_spec(seq3), boltz.find_reusable_msa(_spec(UBQ)), "child1")
    entry = boltz.find_reusable_msa(_spec(seq3))
    assert entry is not None and entry["origin"] == [UBQ], "原点は引き継ぐこと"

    seq6 = _mutate(UBQ, 6)
    hop = boltz.find_reusable_msa(_spec(seq6))
    assert hop is not None, "3置換の子から見れば次の3置換は窓の中"
    boltz._reanchor_msa(_spec(seq6), hop, "child2")

    # 6 substitutions from the origin is the measured-safe limit; 9 is past it
    seq9 = _mutate(UBQ, 9)
    assert boltz.find_reusable_msa(_spec(seq9)) is None, \
        "原点から離れすぎたら取り直すこと (無制限にドリフトさせない)"


def test_a_reanchored_entry_never_duplicates(cached):
    seq3 = _mutate(UBQ, 3)
    e = boltz.find_reusable_msa(_spec(UBQ))
    boltz._reanchor_msa(_spec(seq3), e, "a")
    boltz._reanchor_msa(_spec(seq3), e, "b")
    index = json.loads((cached / "index.json").read_text())
    assert sum(1 for x in index if x["sequences"] == [seq3]) == 1
