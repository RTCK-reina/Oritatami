"""Sequence validation, FASTA parsing and point-mutation notation."""

from __future__ import annotations

import re
from dataclasses import dataclass

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
ALPHABETS = {
    "protein": set(AMINO_ACIDS),
    "dna": set("ACGTN"),
    "rna": set("ACGUN"),
}
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
AA_NAMES_JA = {
    "A": "アラニン", "C": "システイン", "D": "アスパラギン酸", "E": "グルタミン酸", "F": "フェニルアラニン",
    "G": "グリシン", "H": "ヒスチジン", "I": "イソロイシン", "K": "リシン", "L": "ロイシン",
    "M": "メチオニン", "N": "アスパラギン", "P": "プロリン", "Q": "グルタミン", "R": "アルギニン",
    "S": "セリン", "T": "トレオニン", "V": "バリン", "W": "トリプトファン", "Y": "チロシン",
}

MAX_PROTEIN_LENGTH = 2500


class SequenceError(ValueError):
    pass


def clean_sequence(raw: str, kind: str = "protein") -> str:
    """Clean a pasted sequence, FASTA header and all.

    Copying a sequence out of UniProt or NCBI brings its ">sp|P0CG48|..." line along, and
    that is how most sequences arrive. Rejecting it as "使えない文字があります: >" blamed
    the user for the ordinary case; parse_fasta had existed for this since the start and
    was never wired to anything.
    """
    if kind not in ALPHABETS:
        raise SequenceError(f"unknown polymer type: {kind}")
    text = raw or ""
    if ">" in text or text.lstrip().startswith(";"):
        records = parse_fasta(text)
        if len(records) > 1:
            raise SequenceError(
                f"FASTA に配列が {len(records)} 本あります。1 本ずつ入力してください")
        if records:
            text = records[0][1]
    seq = re.sub(r"[\s\d*]", "", text).upper()
    if not seq:
        raise SequenceError("配列が空です")
    bad = sorted({c for c in seq if c not in ALPHABETS[kind]})
    if bad:
        raise SequenceError(f"{kind} 配列に使えない文字があります: {''.join(bad)}")
    if kind == "protein" and len(seq) > MAX_PROTEIN_LENGTH:
        raise SequenceError(f"配列が長すぎます ({len(seq)} > {MAX_PROTEIN_LENGTH})")
    return seq


def parse_fasta(text: str) -> list[tuple[str, str]]:
    """Return (header, sequence) pairs. A bare sequence with no header is allowed."""
    records: list[tuple[str, str]] = []
    header: str | None = None
    chunks: list[str] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(";"):
            continue  # comment line in the older FASTA dialect
        if line.startswith(">"):
            if header is not None or chunks:
                records.append((header or "sequence", "".join(chunks)))
            header = line[1:].strip()
            chunks = []
        else:
            chunks.append(line)
    if header is not None or chunks:
        records.append((header or "sequence", "".join(chunks)))
    return records


@dataclass(frozen=True)
class Mutation:
    wt: str
    position: int  # 1-based
    mt: str
    chain: str | None = None

    @property
    def code(self) -> str:
        prefix = f"{self.chain}:" if self.chain else ""
        return f"{prefix}{self.wt}{self.position}{self.mt}"


_MUT_RE = re.compile(r"^(?:(?P<chain>[A-Za-z0-9]{1,4}):)?(?P<wt>[A-Za-z])(?P<pos>\d+)(?P<mt>[A-Za-z])$")


def parse_mutations(text: str | list[str]) -> list[Mutation]:
    tokens = text if isinstance(text, list) else re.split(r"[\s,;/+]+", text or "")
    out: list[Mutation] = []
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        m = _MUT_RE.match(tok)
        if not m:
            raise SequenceError(f"変異の書式が不正です: {tok} (例: K48R, A:K48R)")
        wt, mt = m.group("wt").upper(), m.group("mt").upper()
        if wt not in AMINO_ACIDS or mt not in AMINO_ACIDS:
            raise SequenceError(f"標準アミノ酸ではありません: {tok}")
        out.append(Mutation(wt=wt, position=int(m.group("pos")), mt=mt, chain=m.group("chain")))
    return out


def apply_mutations(seq: str, mutations: list[Mutation]) -> str:
    chars = list(seq)
    seen: dict[int, Mutation] = {}
    for mut in mutations:
        if mut.position < 1 or mut.position > len(chars):
            raise SequenceError(f"{mut.code}: 位置 {mut.position} は配列長 {len(chars)} の範囲外です")
        if mut.position in seen and seen[mut.position].mt != mut.mt:
            raise SequenceError(f"{mut.code}: 同じ位置に別の変異が指定されています")
        actual = seq[mut.position - 1]
        if actual != mut.wt:
            raise SequenceError(
                f"{mut.code}: 位置 {mut.position} の残基は {actual} で、{mut.wt} ではありません"
            )
        chars[mut.position - 1] = mut.mt
        seen[mut.position] = mut
    return "".join(chars)


def diff_substitutions(wt: str, mt: str) -> list[str] | None:
    """Substitution list if both sequences have the same length, else None."""
    if len(wt) != len(mt):
        return None
    return [f"{a}{i + 1}{b}" for i, (a, b) in enumerate(zip(wt, mt, strict=True)) if a != b]


def identity(a: str, b: str) -> float:
    """Global identity via Biopython's aligner (fraction of aligned identical positions / max length)."""
    if a == b:
        return 1.0
    from Bio import Align

    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1
    aligner.mismatch_score = 0
    aligner.open_gap_score = -1
    aligner.extend_gap_score = -0.5
    aln = aligner.align(a, b)[0]
    same = 0
    for (s1, e1), (s2, e2) in zip(*aln.aligned, strict=True):
        same += sum(1 for x, y in zip(a[s1:e1], b[s2:e2], strict=True) if x == y)
    return same / max(len(a), len(b))
