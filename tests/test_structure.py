import pytest
from conftest import DATA
from oritatami import structure

CA2 = DATA / "ca2_prediction" / "complex_model_0.cif"
UBQ = DATA / "ubq_prediction" / "complex_model_0.cif"


def need(path):
    if not path.exists():
        pytest.skip(f"fixture missing: {path.name}")


def test_summarize_complex_with_ligands():
    need(CA2)
    s = structure.summarize(CA2)
    prot = [c for c in s["chains"] if c["kind"] == "protein"]
    assert len(prot) == 1 and len(prot[0]["sequence"]) == 260
    assert {lig["chain"] for lig in s["ligands"]} >= {"B", "C"}


def test_bfactors_are_plddt():
    need(UBQ)
    vals = structure.residue_bfactors(UBQ)["A"]
    assert len(vals) == 76
    assert all(0 <= v <= 100 for v in vals)


def test_contacts_find_ligand_pocket():
    need(CA2)
    c = structure.contacts(CA2)
    pairs = {tuple(i["chains"]) for i in c["interfaces"]}
    assert ("A", "C") in pairs
    zn = next(i for i in c["interfaces"] if i["chains"] == ["A", "B"])
    # carbonic anhydrase II coordinates zinc with His94, His96, His119
    his = {r for r in zn["residues"]["A"] if r.startswith("H")}
    assert {"H94", "H96", "H119"} <= his


def test_superpose_self_is_zero(tmp_path):
    need(UBQ)
    r = structure.superpose(UBQ, UBQ, tmp_path / "moved.cif")
    assert r["rmsd"] < 1e-3 and r["matched_ca"] == 76
    assert (tmp_path / "moved.cif").exists()


def test_token_segments_match_pae():
    need(CA2)
    import numpy as np

    segs = structure.token_segments(CA2)
    with np.load(DATA / "ca2_prediction" / "pae_complex_model_0.npz") as d:
        n = d["pae"].shape[0]
    assert segs[-1]["end"] == n
