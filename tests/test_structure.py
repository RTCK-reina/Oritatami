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


# ---------------------------------------------------------------- geometry sanity
def _damaged(src, tmp_path, name, edit):
    import gemmi

    st = gemmi.read_structure(str(src))
    st.setup_entities()
    edit(st)
    out = tmp_path / name
    st.make_mmcif_document().write_file(str(out))
    return out


def test_geometry_check_passes_a_real_prediction():
    need(UBQ)
    g = structure.geometry_check(UBQ)
    assert g["nonfinite_atoms"] == 0
    assert g["severe_clashes"] == 0
    assert g["chain_breaks"] == []
    assert g["atoms"] > 0 and g["clashscore"] is not None


def test_geometry_check_finds_overlapping_atoms(tmp_path):
    need(UBQ)

    def overlap(st):
        chain = st[0][0]
        target = chain[5][1].pos
        chain[40][1].pos = type(target)(target.x + 0.4, target.y, target.z)

    g = structure.geometry_check(_damaged(UBQ, tmp_path, "clash.cif", overlap))
    assert g["clashes"] > 0 and g["severe_clashes"] > 0
    assert g["worst_clashes"][0]["overlap"] > g["worst_clashes"][-1]["overlap"] - 1e-9


def test_geometry_check_finds_nan_coordinates(tmp_path):
    need(UBQ)

    def nan(st):
        import gemmi

        for atom in st[0][0][10]:
            atom.pos = gemmi.Position(float("nan"), float("nan"), float("nan"))

    g = structure.geometry_check(_damaged(UBQ, tmp_path, "nan.cif", nan))
    assert g["nonfinite_atoms"] > 0
    assert g["nonfinite_examples"]
    # With NaN in the file the neighbour search says nothing useful, so it is not run.
    assert g["clashes"] is None


def test_geometry_check_finds_a_broken_backbone(tmp_path):
    need(UBQ)

    def split(st):
        import gemmi

        for res in list(st[0][0])[30:]:
            for atom in res:
                atom.pos = gemmi.Position(atom.pos.x + 25.0, atom.pos.y, atom.pos.z)

    g = structure.geometry_check(_damaged(UBQ, tmp_path, "break.cif", split))
    assert len(g["chain_breaks"]) == 1
    assert g["chain_breaks"][0]["dist"] > structure.CA_BREAK


def test_salt_bridges_are_not_clashes():
    """A hydrogen bond pulls donor and acceptor inside their vdW contact; that is not an error."""
    need(CA2)
    g = structure.geometry_check(CA2)
    # Zinc sits 2.1 A from three histidine nitrogens. Coordination, not a collision.
    assert all("ZN" not in c["a"] and "ZN" not in c["b"] for c in g["worst_clashes"])


def _dna(tmp_path, name, step):
    """A three-nucleotide strand whose C1' atoms are `step` apart."""
    import gemmi

    st = gemmi.Structure()
    st.spacegroup_hm = "P 1"
    model = gemmi.Model("1")
    chain = gemmi.Chain("A")
    for i, code in enumerate(("DG", "DC", "DA"), 1):
        res = gemmi.Residue()
        res.name = code
        res.seqid = gemmi.SeqId(i, " ")
        for atom_name, element in (("C1'", "C"), ("P", "P")):
            atom = gemmi.Atom()
            atom.name = atom_name
            atom.element = gemmi.Element(element)
            atom.pos = gemmi.Position((i - 1) * step, 0.0 if atom_name == "C1'" else 2.0, 0.0)
            atom.b_iso = 80.0
            res.add_atom(atom)
        chain.add_residue(res)
    model.add_chain(chain)
    st.add_model(model)
    st.setup_entities()
    out = tmp_path / name
    st.make_mmcif_document().write_file(str(out))
    return out


def test_dna_strand_spacing_is_not_a_break(tmp_path):
    """C1' atoms along one strand of B-DNA sit about 5.4 A apart; the protein limit is 5.0."""
    g = structure.geometry_check(_dna(tmp_path, "dna_ok.cif", 5.4))
    assert g["chain_breaks"] == []


def test_a_dna_strand_that_really_came_apart_is_reported(tmp_path):
    g = structure.geometry_check(_dna(tmp_path, "dna_broken.cif", 14.0))
    assert len(g["chain_breaks"]) == 2
    assert g["chain_breaks"][0]["limit"] == structure.NUCLEIC_BREAK
