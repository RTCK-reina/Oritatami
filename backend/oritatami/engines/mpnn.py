"""Inverse folding as a third opinion: does this sequence fit the backbone it claims?

Boltz reports how confident it is, and ESM-2 reports how ordinary the sequence looks.
Both endorsed destroying ubiquitin's C-terminal glycines and K63 — the two changes that
raise the score and kill the molecule. Two scorers with the same blind spot are one
scorer.

ProteinMPNN asks a different question: given these backbone coordinates, how likely is
this sequence? Measured on this machine against a predicted ubiquitin backbone
(2026-09-14, negative log likelihood, lower fits better, relative to wild type):

    E24A   -0.053    benign surface change      — favoured
    K48R   +0.012    breaks chain formation
    T9E    +0.013    benign surface change
    K63N   +0.030    what the loop chased       — penalised
    G76C   +0.031    what the loop chased       — penalised
    G75C   +0.078    destroys the C-terminus
    L67D   +0.081    charge buried in the core
    I44A   +0.087    destroys the hydrophobic patch

It ranks the four structurally destructive changes worst and the two benign ones best,
which is the ordering the other two scorers get wrong.

As a predictor of the metric the loop optimises it is weak — over 480 historical
parent/child pairs the correlation with delta core pLDDT is r = -0.114. As a filter it
still pays: keeping the best quarter by this score lifts the share of attempts that beat
their parent from 4.6% to 8.3%, for a couple of seconds against a 78-second prediction.
So it is used to veto the obviously destructive and to break ties, never as the ranking.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

log = logging.getLogger("oritatami.mpnn")

_SCORE = re.compile(r"from FASTA, mean:\s*([-\d.]+)")
# Batched against one backbone; a run of ~8 sequences takes a couple of seconds on MPS.
TIMEOUT = 300.0


class MpnnError(RuntimeError):
    pass


def binary() -> str | None:
    """The proteinmpnn CLI that ships with the installed package, if it is there."""
    local = Path(sys.executable).with_name("proteinmpnn")
    if local.exists():
        return str(local)
    return shutil.which("proteinmpnn")


def available() -> bool:
    return binary() is not None


def backbone_pdb(cif_path: Path, dest: Path) -> Path:
    """ProteinMPNN reads PDB; Boltz writes mmCIF."""
    import gemmi

    st = gemmi.read_structure(str(cif_path))
    st.setup_entities()
    st.write_pdb(str(dest))
    return dest


def score(pdb: Path, sequences: list[str], *, seed: int = 37) -> list[float]:
    """Negative log likelihood of each sequence given the backbone. Lower fits better.

    Deterministic for a fixed seed — the scoring path adds no backbone noise — so unlike
    the pLDDT it can be compared between runs without averaging.
    """
    exe = binary()
    if exe is None:
        raise MpnnError("proteinmpnn が見つかりません (uv pip install proteinmpnn-mps)")
    if not sequences:
        return []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        fasta = tmp_path / "sequences.fa"
        fasta.write_text("".join(f">{i}\n{s}\n" for i, s in enumerate(sequences)), "utf-8")
        try:
            proc = subprocess.run(
                [exe, "--pdb_path", str(pdb), "--path_to_fasta", str(fasta),
                 "--score_only", "1", "--out_folder", str(tmp_path),
                 "--seed", str(seed), "--batch_size", "1"],
                capture_output=True, text=True, timeout=TIMEOUT)
        except subprocess.TimeoutExpired as exc:
            raise MpnnError(f"ProteinMPNN が {TIMEOUT:.0f} 秒で終わりませんでした") from exc
    if proc.returncode != 0:
        raise MpnnError(f"ProteinMPNN が失敗しました: {(proc.stderr or proc.stdout)[-400:]}")
    values = [float(v) for v in _SCORE.findall(proc.stdout)]
    if len(values) != len(sequences):
        raise MpnnError(f"スコアの数が合いません ({len(values)} / {len(sequences)})")
    return values


def relative(pdb: Path, reference: str, candidates: list[str], *, seed: int = 37) -> list[float]:
    """Each candidate's score minus the reference's, on the same backbone.

    Positive means ProteinMPNN thinks the change makes the sequence fit this fold worse.
    Scoring the reference in the same batch keeps the comparison free of any per-run
    offset.
    """
    values = score(pdb, [reference, *candidates], seed=seed)
    base = values[0]
    return [round(v - base, 4) for v in values[1:]]
