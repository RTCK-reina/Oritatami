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


# ---------------------------------------------------------------- repairing a proposal
# How far from the position it named a mutation may be moved. A language model that has the
# right residue in mind and the wrong index is usually off by a handful — a leading methionine,
# a signal peptide that UniProt numbers and the construct does not. Past this the "correction"
# would be a different mutation wearing the same name.
REPAIR_WINDOW = 12
# An offset is only believed when this many of the proposal's own mutations agree on it.
OFFSET_QUORUM = 2


def wt_matches(seq: str, mut: Mutation) -> bool:
    return 1 <= mut.position <= len(seq) and seq[mut.position - 1] == mut.wt


def infer_offset(seq: str, mutations: list[Mutation], *, window: int = REPAIR_WINDOW) -> int | None:
    """A single shift that explains several mismatched mutations at once.

    When a model numbers from the UniProt entry and the workbench holds a construct, every
    position in the proposal is wrong by the same amount. That is worth finding before moving
    mutations one at a time, because the shared offset is evidence and a nearest-match is a
    guess.
    """
    wrong = [m for m in mutations if not wt_matches(seq, m)]
    if len(wrong) < OFFSET_QUORUM:
        return None
    best: tuple[int, int] | None = None                    # (matches, -|k|) for the best k
    for k in range(-window, window + 1):
        if k == 0:
            continue
        hits = sum(1 for m in wrong
                   if 1 <= m.position + k <= len(seq) and seq[m.position + k - 1] == m.wt)
        if hits >= OFFSET_QUORUM and (best is None or (hits, -abs(k)) > best):
            best = (hits, -abs(k))
            best_k = k
    return best_k if best else None


def repair_mutation(seq: str, mut: Mutation, *, offset: int | None = None,
                    window: int = REPAIR_WINDOW) -> tuple[Mutation | None, str | None]:
    """Put a mutation on the position its wild-type residue actually occupies.

    Returns the mutation to use and a note for the person, or ``(None, reason)`` when there is
    nothing near enough to be the same mutation. The note is never silent: a proposal that was
    moved says so on its card, because "S257P" and "S252P" are different claims about the
    molecule even when one of them is what the model meant.
    """
    if wt_matches(seq, mut):
        return mut, None
    if mut.position < 1:
        return None, f"位置 {mut.position} は配列の外です"
    actual = seq[mut.position - 1] if mut.position <= len(seq) else None
    if offset is not None:
        shifted = mut.position + offset
        if 1 <= shifted <= len(seq) and seq[shifted - 1] == mut.wt:
            moved = Mutation(wt=mut.wt, position=shifted, mt=mut.mt, chain=mut.chain)
            return moved, (f"{mut.code} → {moved.code} (提案全体が {offset:+d} ずれていました)")
    candidates = [q for q in range(max(1, mut.position - window), min(len(seq), mut.position + window) + 1)
                  if seq[q - 1] == mut.wt]
    if not candidates:
        where = f"位置 {mut.position} の残基は {actual}" if actual else f"位置 {mut.position} は配列長 {len(seq)} の外"
        return None, f"{where} で、前後 {window} 残基に {mut.wt} がありません"
    candidates.sort(key=lambda q: (abs(q - mut.position), q))
    moved = Mutation(wt=mut.wt, position=candidates[0], mt=mut.mt, chain=mut.chain)
    others = [str(q) for q in candidates[1:3]]
    note = (f"{mut.code} → {moved.code} (位置 {mut.position} は {actual}、"
            f"最も近い {mut.wt} は {candidates[0]}"
            + (f"。他に {', '.join(others)}" if others else "") + ")")
    return moved, note


def repair_mutations(seq: str, mutations: list[Mutation]) -> tuple[list[Mutation], list[str], list[str]]:
    """Repair a whole proposal at once. Returns (usable mutations, notes, dropped)."""
    offset = infer_offset(seq, mutations)
    kept: list[Mutation] = []
    notes: list[str] = []
    dropped: list[str] = []
    for mut in mutations:
        moved, note = repair_mutation(seq, mut, offset=offset)
        if moved is None:
            dropped.append(f"{mut.code}: {note}")
            continue
        if note:
            notes.append(note)
        kept.append(moved)
    # Moving two mutations onto one position would silently drop one of them.
    seen: dict[int, Mutation] = {}
    final: list[Mutation] = []
    for mut in kept:
        clash = seen.get(mut.position)
        if clash is not None:
            if clash.mt != mut.mt:
                dropped.append(f"{mut.code}: 直した先の位置 {mut.position} が {clash.code} と重なります")
            else:
                notes.append(f"{mut.code} は {clash.code} と同じ変異になったのでまとめました")
            continue
        seen[mut.position] = mut
        final.append(mut)
    return final, notes, dropped


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
