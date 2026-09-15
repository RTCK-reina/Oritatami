"""Structure-file analysis with gemmi: chains, sequences, pLDDT, contacts, superposition."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gemmi

from .seq import AMINO_ACIDS, identity

WATER = {"HOH", "WAT", "DOD", "H2O"}
NUCLEOTIDES = {
    "DA": "A", "DC": "C", "DG": "G", "DT": "T", "DU": "U", "DI": "N",
    "A": "A", "C": "C", "G": "G", "U": "U", "I": "N",
}
# Common crystallisation additives that are rarely what a user means by "the ligand"
ADDITIVES = {
    "SO4", "PO4", "GOL", "EDO", "PEG", "PG4", "PGE", "1PE", "ACT", "DMS", "CL", "NA", "K", "IOD",
    "BR", "FMT", "MES", "EPE", "TRS", "CIT", "ACY", "IMD", "BME", "MPD", "NO3", "SCN", "NH4",
}


def read_structure(path: Path | str) -> gemmi.Structure:
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    return st


def _polymer_kind(ptype: gemmi.PolymerType) -> str | None:
    if ptype in (gemmi.PolymerType.PeptideL, gemmi.PolymerType.PeptideD):
        return "protein"
    if ptype == gemmi.PolymerType.Dna:
        return "dna"
    if ptype == gemmi.PolymerType.Rna:
        return "rna"
    return None


def _one_letter(name: str, kind: str) -> str | None:
    name = name.split(",")[0].strip().upper()
    if kind == "protein":
        info = gemmi.find_tabulated_residue(name)
        if info is None:
            return None
        code = info.one_letter_code.upper()
        return code if code in AMINO_ACIDS else None
    code = NUCLEOTIDES.get(name)
    if code is None:
        info = gemmi.find_tabulated_residue(name)
        if info is not None and info.one_letter_code.strip():
            code = info.one_letter_code.upper()
    if code is None:
        return None
    if kind == "dna" and code == "U":
        return None
    if kind == "rna" and code == "T":
        return None
    return code


def summarize(path: Path | str) -> dict[str, Any]:
    """Chains, full sequences (SEQRES when present) and ligands of the first model."""
    st = read_structure(path)
    if len(st) == 0:
        raise ValueError("構造ファイルにモデルがありません")
    model = st[0]
    chains: list[dict[str, Any]] = []
    ligands: list[dict[str, Any]] = []
    warnings: list[str] = []
    for chain in model:
        polymer = chain.get_polymer()
        if len(polymer) > 0:
            kind = _polymer_kind(polymer.check_polymer_type())
            if kind is None:
                warnings.append(f"チェーン {chain.name}: 未対応のポリマー種別のため除外しました")
            else:
                entity = st.get_entity_of(polymer)
                names = list(entity.full_sequence) if entity is not None and entity.full_sequence else [r.name for r in polymer]
                letters = []
                dropped = 0
                for n in names:
                    code = _one_letter(n, kind)
                    if code is None:
                        dropped += 1
                    else:
                        letters.append(code)
                if dropped:
                    warnings.append(f"チェーン {chain.name}: 非標準残基 {dropped} 個を配列から除外しました")
                seq = "".join(letters)
                if seq:
                    entity_name = entity.name if entity is not None else chain.name
                    chains.append({
                        "chain": chain.name,
                        "kind": kind,
                        "sequence": seq,
                        "entity": entity_name,
                        "observed_residues": len(polymer),
                    })
        for res in chain:
            if res.het_flag != "H" or res.name in WATER:
                continue
            ent = st.get_entity_of(chain.get_subchain(res.subchain)) if res.subchain else None
            if ent is not None and ent.entity_type == gemmi.EntityType.Polymer:
                continue
            ligands.append({
                "chain": chain.name,
                "ccd": res.name,
                "seqid": str(res.seqid),
                "atoms": len(res),
                "additive": res.name in ADDITIVES,
            })
    return {
        "title": st.info["_struct.title"] if "_struct.title" in st.info else "",
        "chains": chains,
        "ligands": ligands,
        "warnings": warnings,
    }


def residue_bfactors(path: Path | str) -> dict[str, list[float]]:
    """Mean B-factor per polymer residue, by chain. For predictions this is pLDDT (0-100)."""
    st = read_structure(path)
    out: dict[str, list[float]] = {}
    for chain in st[0]:
        polymer = chain.get_polymer()
        if len(polymer) == 0:
            continue
        vals = []
        for res in polymer:
            if len(res) == 0:
                continue
            ca = res.find_atom("CA", "*") or res.find_atom("C1'", "*")
            vals.append(round(float(ca.b_iso if ca else sum(a.b_iso for a in res) / len(res)), 2))
        out[chain.name] = vals
    return out


def ligand_bfactors(path: Path | str) -> dict[str, float]:
    st = read_structure(path)
    out: dict[str, float] = {}
    for chain in st[0]:
        if len(chain.get_polymer()) > 0:
            continue
        atoms = [a.b_iso for res in chain for a in res]
        if atoms:
            out[chain.name] = round(float(sum(atoms) / len(atoms)), 2)
    return out


def _res_label(name: str, num: int) -> str:
    info = gemmi.find_tabulated_residue(name)
    if info is not None and info.is_amino_acid():
        code = info.one_letter_code.upper()
        if code in AMINO_ACIDS:
            return f"{code}{num}"
    return f"{name}{num}"


def contacts(path: Path | str, cutoff: float = 4.5) -> dict[str, Any]:
    """Inter-chain residue contacts. Returns per chain pair the residues involved."""
    st = read_structure(path)
    st.remove_hydrogens()
    model = st[0]
    ns = gemmi.NeighborSearch(model, st.cell, cutoff + 1.0).populate()
    cs = gemmi.ContactSearch(cutoff)
    cs.ignore = gemmi.ContactSearch.Ignore.SameChain
    pairs: dict[tuple[str, str], set[tuple[str, int, str, int]]] = {}
    for r in cs.find_contacts(ns):
        c1, c2 = r.partner1.chain.name, r.partner2.chain.name
        if c1 == c2:
            continue
        a = (c1, r.partner1.residue.seqid.num, r.partner1.residue.name)
        b = (c2, r.partner2.residue.seqid.num, r.partner2.residue.name)
        if (c1, c2) > (c2, c1):
            a, b = b, a
            c1, c2 = c2, c1
        pairs.setdefault((c1, c2), set()).add((a[2], a[1], b[2], b[1]))
    result = []
    for (c1, c2), items in sorted(pairs.items()):
        res1 = sorted({(n, i) for n, i, _, _ in items}, key=lambda x: x[1])
        res2 = sorted({(n, i) for _, _, n, i in items}, key=lambda x: x[1])
        result.append({
            "chains": [c1, c2],
            "residues": {
                c1: [_res_label(n, i) for n, i in res1],
                c2: [_res_label(n, i) for n, i in res2],
            },
            "contact_pairs": len(items),
        })
    return {"cutoff": cutoff, "interfaces": result}


def _ca_trace(chain: gemmi.Chain) -> tuple[str, list[gemmi.Position], list[int]]:
    letters, pos, nums = [], [], []
    for res in chain.get_polymer():
        info = gemmi.find_tabulated_residue(res.name)
        if info is None or not info.is_amino_acid():
            continue
        ca = res.find_atom("CA", "*")
        if ca is None:
            continue
        code = info.one_letter_code.upper()
        letters.append(code if code in AMINO_ACIDS else "X")
        pos.append(ca.pos)
        nums.append(res.seqid.num)
    return "".join(letters), pos, nums


def superpose(fixed_path: Path | str, moving_path: Path | str, out_path: Path) -> dict[str, Any]:
    """Superpose `moving` onto `fixed` on matched protein CA atoms and write the moved copy as mmCIF."""
    from Bio import Align

    fixed = read_structure(fixed_path)
    moving = read_structure(moving_path)
    fixed_chains = [(c.name, *_ca_trace(c)) for c in fixed[0]]
    fixed_chains = [c for c in fixed_chains if c[1]]
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -4
    aligner.extend_gap_score = -0.5

    fixed_pos: list[gemmi.Position] = []
    moving_pos: list[gemmi.Position] = []
    mapping: list[dict[str, Any]] = []
    used: set[str] = set()
    for mchain in moving[0]:
        mseq, mpos, mnums = _ca_trace(mchain)
        if not mseq:
            continue
        best = None
        for fname, fseq, fpos, fnums in fixed_chains:
            if fname in used:
                continue
            ident = identity(fseq, mseq)
            if best is None or ident > best[0]:
                best = (ident, fname, fseq, fpos, fnums)
        if best is None or best[0] < 0.3:
            continue
        ident, fname, fseq, fpos, fnums = best
        used.add(fname)
        aln = aligner.align(fseq, mseq)[0]
        for (s1, e1), (s2, _e2) in zip(*aln.aligned, strict=True):
            for k in range(e1 - s1):
                fixed_pos.append(fpos[s1 + k])
                moving_pos.append(mpos[s2 + k])
                mapping.append({"fixed_chain": fname, "fixed_res": fnums[s1 + k],
                                "moving_chain": mchain.name, "moving_res": mnums[s2 + k]})
    if len(fixed_pos) < 3:
        raise ValueError("重ね合わせに使える対応残基が 3 個未満です")
    sup = gemmi.superpose_positions(fixed_pos, moving_pos)
    moving[0].transform_pos_and_adp(sup.transform)
    deviations = []
    for m, fp, mp in zip(mapping, fixed_pos, moving_pos, strict=True):
        moved = gemmi.Position(sup.transform.apply(mp))
        deviations.append({**m, "distance": round(fp.dist(moved), 2)})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = moving.make_mmcif_document()
    doc.write_file(str(out_path))
    return {
        "rmsd": round(float(sup.rmsd), 3),
        "matched_ca": int(sup.count),
        "deviations": deviations,
        "output": str(out_path),
    }


def token_segments(path: Path | str) -> list[dict[str, Any]]:
    """Approximate Boltz tokenisation per chain: one token per standard residue, one per ligand heavy atom."""
    st = read_structure(path)
    st.remove_hydrogens()
    segments = []
    start = 0
    for chain in st[0]:
        polymer = chain.get_polymer()
        if len(polymer) > 0:
            n = len(polymer)
            kind = "polymer"
        else:
            n = sum(len(res) for res in chain)
            kind = "ligand"
        segments.append({"chain": chain.name, "kind": kind, "start": start, "end": start + n})
        start += n
    return segments


def to_pdb(path: Path | str) -> str:
    """Convert a structure to PDB text. Raises ValueError when the PDB format cannot hold it."""
    st = read_structure(path)
    if any(len(ch.name) > 1 for ch in st[0]):
        # PDB chain ids are one character; renaming silently would mislead users.
        raise ValueError("チェーン ID が 2 文字以上あるため PDB 形式に変換できません (mmCIF を使ってください)")
    if sum(1 for _ in st[0].all()) > 99999:
        raise ValueError("原子数が PDB 形式の上限を超えています")
    st.shorten_ccd_codes()
    return st.make_pdb_string(gemmi.PdbWriteOptions(minimal=False))


# ------------------------------------------------------------------ geometry sanity
# A prediction can finish, write a file, and still be unusable: half-precision denoising on
# MPS can turn coordinates into NaN, and with the physical-correction step off the raw
# diffusion output keeps atom pairs sitting inside each other. Neither shows up in pLDDT —
# a NaN residue still carries a confidence number, and two overlapping side chains can both
# be "confident". The only way to know is to measure the geometry that came out.

# What counts as a collision. Two atoms are clashing when they are closer than their van der
# Waals radii allow, with the same 0.4 A tolerance MolProbity uses. Nitrogen, oxygen and
# sulfur get a further 0.6 A because a hydrogen bond legitimately pulls a donor and an
# acceptor inside their vdW contact — without that allowance every salt bridge in every
# structure reads as an error (measured here: 141 of 973 finished models, almost all of them
# arginine-aspartate pairs at 2.0-2.2 A, which is short but not broken).
VDW_TOLERANCE = 0.4
HBOND_ALLOWANCE = 0.6
# How deep the interpenetration has to be before the two atoms are simply in the same place.
SEVERE_OVERLAP = 0.8
_VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80, "SE": 1.90, "F": 1.47,
        "CL": 1.75, "BR": 1.85, "I": 1.98}
_VDW_DEFAULT = 1.70
_POLAR = {"N", "O", "S", "F"}
# Metals coordinate at 2.0-2.2 A by design, and a disulfide sits at 2.05 A. Neither is a
# clash, and neither is described by van der Waals radii.
_METALS = {"ZN", "FE", "MG", "MN", "CU", "CO", "NI", "CD", "HG", "NA", "K", "CA", "MO", "W",
           "AG", "AU", "PT", "PD", "LI", "AL", "BA", "SR", "CS", "RB", "PB", "V", "CR"}
# Search radius: the widest pair we can flag is carbon-carbon at 1.70 + 1.70 - 0.4 = 3.0 A.
_CLASH_SEARCH = 3.1
# Consecutive CA atoms sit 3.8 A apart (2.9 A across a cis peptide bond). Past this the
# backbone has come apart.
CA_BREAK = 5.0
# Nucleic acids are not measured on the same scale: consecutive C1' atoms along one strand of
# B-DNA sit about 5.4 A apart, so the protein threshold flags every duplex that comes out of
# the app (it flagged two, both correctly built). 9 A is well past any stacked step and well
# short of a strand that has actually come apart.
NUCLEIC_BREAK = 9.0
_MAX_REPORTED = 12


def _allowed_distance(e1: str, e2: str) -> float | None:
    """How close these two elements may legitimately come. None when the pair is a bond."""
    if e1 in _METALS or e2 in _METALS:
        return None
    if e1 == "S" and e2 == "S":
        return None                                   # disulfide
    limit = _VDW.get(e1, _VDW_DEFAULT) + _VDW.get(e2, _VDW_DEFAULT) - VDW_TOLERANCE
    if e1 in _POLAR and e2 in _POLAR:
        limit -= HBOND_ALLOWANCE
    return limit


def _atom_label(cra: gemmi.CRA) -> str:
    return f"{cra.chain.name}/{cra.residue.name}{cra.residue.seqid.num}/{cra.atom.name}"


def _nonfinite(st: gemmi.Structure) -> tuple[int, list[str]]:
    import math

    bad, examples = 0, []
    for chain in st[0]:
        for res in chain:
            for atom in res:
                p = atom.pos
                if not (math.isfinite(p.x) and math.isfinite(p.y) and math.isfinite(p.z)
                        and math.isfinite(atom.b_iso)):
                    bad += 1
                    if len(examples) < _MAX_REPORTED:
                        examples.append(f"{chain.name}/{res.name}{res.seqid.num}/{atom.name}")
    return bad, examples


def _clashes(st: gemmi.Structure) -> tuple[int, int, list[dict[str, Any]]]:
    model = st[0]
    ns = gemmi.NeighborSearch(model, st.cell, _CLASH_SEARCH + 1.0).populate()
    cs = gemmi.ContactSearch(_CLASH_SEARCH)
    # AdjacentResidues also covers the same residue, so bond lengths inside a residue and
    # across the peptide bond are excluded without having to know the chemistry.
    cs.ignore = gemmi.ContactSearch.Ignore.AdjacentResidues
    total = severe = 0
    worst: list[dict[str, Any]] = []
    for r in cs.find_contacts(ns):
        e1 = r.partner1.atom.element.name.upper()
        e2 = r.partner2.atom.element.name.upper()
        limit = _allowed_distance(e1, e2)
        if limit is None or r.dist >= limit:
            continue
        overlap = limit - float(r.dist)
        total += 1
        if overlap >= SEVERE_OVERLAP:
            severe += 1
        worst.append({"a": _atom_label(r.partner1), "b": _atom_label(r.partner2),
                      "dist": round(float(r.dist), 2), "overlap": round(overlap, 2)})
    worst.sort(key=lambda x: -x["overlap"])
    return total, severe, worst[:_MAX_REPORTED]


def _chain_breaks(st: gemmi.Structure) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for chain in st[0]:
        polymer = chain.get_polymer()
        limit = CA_BREAK if _polymer_kind(polymer.check_polymer_type()) == "protein" else NUCLEIC_BREAK
        prev = None
        for res in polymer:
            ca = res.find_atom("CA", "*") or res.find_atom("C1'", "*")
            if ca is None:
                prev = None
                continue
            if prev is not None:
                d = prev[1].pos.dist(ca.pos)
                if d > limit and len(out) < _MAX_REPORTED:
                    out.append({"chain": chain.name,
                                "after": _res_label(prev[0].name, prev[0].seqid.num),
                                "before": _res_label(res.name, res.seqid.num),
                                "dist": round(float(d), 2), "limit": limit})
            prev = (res, ca)
    return out


def nonfinite_atoms(path: Path | str) -> tuple[int, list[str]]:
    """Count atoms whose coordinates or confidence are NaN/inf. Cheap enough for every model."""
    return _nonfinite(read_structure(path))


def geometry_check(path: Path | str) -> dict[str, Any]:
    """Is this model physically possible? Non-finite coordinates, clashes, broken backbone.

    Cheap enough to run on every finished prediction: a 600-residue monomer takes well under
    a second, because the neighbour search is the same one the contact map already builds.
    """
    st = read_structure(path)
    st.remove_hydrogens()          # predicted hydrogens are placed, not observed
    atoms = sum(1 for _ in st[0].all())
    bad, examples = _nonfinite(st)
    if bad:
        # With NaN in the coordinates the neighbour search is meaningless (and gemmi may
        # bin those atoms anywhere), so stop here and report the real problem.
        return {"atoms": atoms, "nonfinite_atoms": bad, "nonfinite_examples": examples,
                "clashes": None, "severe_clashes": None, "clashscore": None,
                "worst_clashes": [], "chain_breaks": [], "vdw_tolerance": VDW_TOLERANCE}
    total, severe, worst = _clashes(st)
    return {
        "atoms": atoms,
        "nonfinite_atoms": 0,
        "nonfinite_examples": [],
        "clashes": total,
        "severe_clashes": severe,
        # Clashes per 1000 atoms, so models of different sizes can be compared. Measured over
        # the 973 models this machine has produced: median 0, 75th percentile 1.7, 90th 4.9.
        "clashscore": round(total * 1000.0 / atoms, 1) if atoms else 0.0,
        "worst_clashes": worst,
        "chain_breaks": _chain_breaks(st),
        "vdw_tolerance": VDW_TOLERANCE,
    }
