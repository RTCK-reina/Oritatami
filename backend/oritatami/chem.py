"""Small-molecule helpers backed by RDKit."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import get_settings


class ChemError(ValueError):
    pass


def _rdkit():
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    return Chem


def mol_from_smiles(smiles: str):
    Chem = _rdkit()
    smiles = (smiles or "").strip()
    if not smiles:
        raise ChemError("SMILES が空です")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ChemError(f"RDKit が解釈できない SMILES です: {smiles}")
    return mol


def describe_smiles(smiles: str) -> dict[str, Any]:
    from rdkit.Chem import Descriptors, rdMolDescriptors

    Chem = _rdkit()
    mol = mol_from_smiles(smiles)
    frags = Chem.GetMolFrags(mol)
    heavy = mol.GetNumHeavyAtoms()
    # Boltz-2 affinity counts heavy atoms plus hydrogens kept by RemoveHs; heavy atoms is the
    # practical lower bound and what users need to see (training limit ~56, hard limit 128).
    return {
        "smiles": smiles,
        "canonical_smiles": Chem.MolToSmiles(mol),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "molecular_weight": round(Descriptors.MolWt(mol), 2),
        "heavy_atoms": heavy,
        "fragments": len(frags),
        "hbd": rdMolDescriptors.CalcNumHBD(mol),
        "hba": rdMolDescriptors.CalcNumHBA(mol),
        "logp": round(Descriptors.MolLogP(mol), 2),
        "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "affinity_ok": heavy <= 56 and len(frags) == 1,
    }


def smiles_svg(smiles: str, width: int = 260, height: int = 200) -> str:
    from rdkit.Chem.Draw import rdMolDraw2D

    mol = mol_from_smiles(smiles)
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.clearBackground = False
    opts.bondLineWidth = 1.6
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def ccd_dir() -> Path:
    return Path(get_settings().boltz_cache).expanduser() / "mols"


@lru_cache(maxsize=4096)
def _ccd_exists(code: str, root: str) -> bool:
    return (Path(root) / f"{code}.pkl").exists()


def validate_ccd(code: str) -> str:
    code = (code or "").strip().upper()
    if not code or len(code) > 5 or not code.isalnum():
        raise ChemError(f"CCD コードの書式が不正です: {code}")
    root = ccd_dir()
    if root.exists() and not _ccd_exists(code, str(root)):
        raise ChemError(f"CCD コード {code} は Boltz の化学成分辞書にありません")
    return code
