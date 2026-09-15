"""Which residues carry the molecule's function — and whether a result has broken them.

Two questions, one body of evidence.

*Before* a search: which positions should be off limits, worked out from the molecule itself
rather than typed in by hand. A hill-climb on pLDDT has no notion of what a protein is *for*,
so the cheapest way to raise the number is to delete the parts the predictor is least sure
about — which are disproportionately the functional ones. Ubiquitin made this concrete: of 491
autonomous attempts, 91 % had rewritten the C-terminal G76 that the whole molecule exists to
conjugate through, and 94 % had destroyed K63.

*After* a prediction: whether the variant in hand scored well by breaking something. The same
evidence answers it, plus one check the evidence cannot give on its own — where the pLDDT gain
actually came from. A variant whose improvement is concentrated in residues the parent was
already unsure about has not been made better; it has been made blander.

Nothing here decides anything on its own. It produces a list with reasons attached, for the
person to accept or overrule, and a warning on a result that already exists.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from . import sources
from .config import get_settings
from .db import Database

log = logging.getLogger("oritatami.function_risk")

_POS = re.compile(r"(\d+)")

# UniProt feature types worth protecting, with the label shown to the person and how strongly
# the suggestion is made. "Domain" and "Region" are deliberately absent: they routinely span
# most of the sequence, so protecting them protects everything and the search has nowhere left
# to go. "Mutagenesis" is absent for the opposite reason — it marks positions somebody has
# already tested, which is information, not a prohibition.
FEATURE_RULES: dict[str, tuple[str, str]] = {
    "Active site": ("活性部位", "critical"),
    "Binding site": ("結合部位", "critical"),
    "Metal binding": ("金属結合", "critical"),
    "Disulfide bond": ("ジスルフィド結合", "critical"),
    "DNA binding": ("DNA 結合", "high"),
    "Zinc finger": ("ジンクフィンガー", "high"),
    "Modified residue": ("翻訳後修飾", "high"),
    "Glycosylation": ("糖鎖付加", "high"),
    # "Site" is UniProt's catch-all — on haemoglobin alpha it is mostly pathogen cleavage
    # points, which are not what this list is for. Reported, not ticked.
    "Site": ("機能部位", "medium"),
    "Motif": ("モチーフ", "medium"),
    "Signal": ("シグナルペプチド", "medium"),
    "Propeptide": ("プロペプチド", "medium"),
    "Transmembrane": ("膜貫通", "medium"),
}

# A feature wider than this is reported but not ticked by default: a 25-residue transmembrane
# helix is a real annotation and a quarter of a small protein.
WIDE_FEATURE = 12
SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}
DEFAULT_ON = {"critical", "high"}


def _severity(reasons: list[dict[str, Any]]) -> str:
    return min((r["severity"] for r in reasons), key=lambda s: SEVERITY_ORDER.get(s, 9), default="low")


def _component_for_chain(spec: dict[str, Any], chain: str) -> dict[str, Any] | None:
    for comp in spec.get("components") or []:
        if chain in (comp.get("chains") or []):
            return comp
    return None


def _add(out: dict[int, list[dict[str, Any]]], pos: int, kind: str, label: str, severity: str) -> None:
    if pos < 1:
        return
    bucket = out.setdefault(pos, [])
    if any(r["kind"] == kind and r["label"] == label for r in bucket):
        return
    bucket.append({"kind": kind, "label": label, "severity": severity})


def _uniprot_positions(comp: dict[str, Any], out: dict[int, list[dict[str, Any]]]) -> str | None:
    """Annotated functional residues, from the entry this chain was taken from.

    Offsets are UniProt's own numbering. That matches the workbench only while the sequence is
    the full entry — which is the case for anything added through the UniProt search, and is
    checked here by length before any of it is used.
    """
    source = comp.get("source") or {}
    if str(source.get("db") or "").lower() != "uniprot" or not source.get("id"):
        return None
    acc = str(source["id"])
    try:
        entry = sources.uniprot_entry(acc)
    except Exception as exc:  # network, rate limit, a withdrawn accession
        log.info("function_risk: UniProt %s を読めませんでした: %s", acc, exc)
        return None
    seq = comp.get("parent_sequence") or comp.get("sequence") or ""
    if entry.get("sequence") and seq and len(entry["sequence"]) != len(seq):
        log.info("function_risk: %s は配列長が UniProt と違う (%d != %d) ため注釈を使いません",
                 acc, len(seq), len(entry["sequence"]))
        return None
    for f in entry.get("features") or []:
        rule = FEATURE_RULES.get(f.get("type"))
        if not rule:
            continue
        label, severity = rule
        start, end = int(f.get("start") or 0), int(f.get("end") or 0)
        if start < 1 or end < start:
            continue
        if end - start + 1 > WIDE_FEATURE:
            severity = "medium"
        desc = (f.get("description") or "").strip()
        text = f"{label}{f' ({desc})' if desc else ''}"
        for pos in range(start, end + 1):
            _add(out, pos, "uniprot", text, severity)
    return acc


def _interface_positions(job: dict[str, Any], chain: str, out: dict[int, list[dict[str, Any]]]) -> int:
    block = ((job.get("result") or {}).get("interfaces") or {})
    n = 0
    for itf in (block.get("interfaces") or []):
        partner = " / ".join(str(c) for c in (itf.get("chains") or []) if c != chain)
        for code in (itf.get("residues") or {}).get(chain) or []:
            digits = "".join(ch for ch in str(code) if ch.isdigit())
            if digits:
                _add(out, int(digits), "interface", f"界面 (相手 {partner})" if partner else "界面", "high")
                n += 1
    return n


def _plddt(job: dict[str, Any], chain: str) -> list[float]:
    models = ((job.get("result") or {}).get("models") or [])
    if not models:
        return []
    values = (models[0].get("plddt") or {}).get(chain) or []
    return [v for v in values if isinstance(v, (int, float))]


def _disordered_positions(root: dict[str, Any], chain: str, threshold: float,
                          out: dict[int, list[dict[str, Any]]]) -> int:
    n = 0
    for i, v in enumerate(_plddt(root, chain), 1):
        if v < threshold:
            _add(out, i, "disorder", f"起点で乱れていた (pLDDT {v:.0f})", "high")
            n += 1
    return n


def _terminal_positions(length: int, out: dict[int, list[dict[str, Any]]]) -> None:
    """The last two residues, and the first.

    Not a heuristic about proteins in general — a heuristic about this search. Termini are
    where pLDDT is lowest and where function is often carried (ubiquitin's G75-G76 is the whole
    point of ubiquitin), which is exactly the combination a score-driven search walks into.
    """
    if length <= 6:
        return
    _add(out, 1, "terminus", "N 末端", "medium")
    _add(out, length - 1, "terminus", "C 末端 (残り 2)", "medium")
    _add(out, length, "terminus", "C 末端", "medium")


def _conserved_positions(scan: dict[str, Any] | None, out: dict[int, list[dict[str, Any]]]) -> int:
    """Positions ESM-2 will not accept any substitution at.

    ``position_tolerance`` is the mean log-likelihood ratio over all 19 substitutions at a
    position. The bottom tenth is where the language model says every alternative is unnatural,
    which is the model's version of "conserved".
    """
    values = ((scan or {}).get("result") or {}).get("position_tolerance") or []
    values = [v for v in values if isinstance(v, (int, float))]
    if len(values) < 20:
        return 0
    ranked = sorted(values)
    cutoff = ranked[max(0, len(ranked) // 10 - 1)]
    n = 0
    for i, v in enumerate(values, 1):
        if v <= cutoff:
            _add(out, i, "conserved", f"ESM-2 がどの置換も不自然と見る (許容度 {v:.1f})", "high")
            n += 1
    return n


def _lineage_root(db: Database, job: dict[str, Any]) -> dict[str, Any]:
    seen: set[str] = set()
    cur = job
    while cur.get("parent_id") and cur["parent_id"] not in seen:
        seen.add(cur["parent_id"])
        nxt = db.get_job(cur["parent_id"])
        if nxt is None:
            break
        cur = nxt
    return cur


def _scan_for(db: Database, sequence: str) -> dict[str, Any] | None:
    if not sequence:
        return None
    for j in db.list_jobs(limit=400):
        if j.get("kind") != "scan" or j.get("status") != "succeeded":
            continue
        full = db.get_job(j["id"]) or {}
        if ((full.get("result") or {}).get("sequence") or "") == sequence:
            return full
    return None


def suggest_protected(db: Database, job: dict[str, Any], chain: str) -> dict[str, Any]:
    """Positions worth putting off limits for this molecule, each with its reason.

    Returns everything found, flagged rather than filtered: the person decides. ``default_on``
    marks the ones proposed as ticked — annotated function, contacts, disorder and conservation
    — leaving wide annotations and termini as suggestions to look at.
    """
    spec = job.get("spec") or {}
    comp = _component_for_chain(spec, chain)
    if comp is None:
        return {"chain": chain, "positions": [], "sources": {}, "notes": ["この鎖が作業台にありません"]}

    sequence = comp.get("sequence") or ""
    found: dict[int, list[dict[str, Any]]] = {}
    notes: list[str] = []
    s = get_settings()

    acc = _uniprot_positions(comp, found)
    if acc is None and (comp.get("source") or {}).get("db", "").lower() == "uniprot":
        notes.append("UniProt の注釈は使えませんでした (配列が改変されている、または取得に失敗)")
    elif acc is None:
        notes.append("UniProt 由来ではないので、データベースの機能注釈は使えません")

    contacts = _interface_positions(job, chain, found)
    root = _lineage_root(db, job)
    disordered = _disordered_positions(root, chain, s.autopilot_disorder_plddt, found)
    scan = _scan_for(db, comp.get("parent_sequence") or sequence)
    conserved = _conserved_positions(scan, found)
    if scan is None:
        notes.append("この配列の変異スキャン (ESM-2) がないので、保存度からの推定は入っていません")
    _terminal_positions(len(sequence), found)

    positions = []
    for pos in sorted(found):
        reasons = sorted(found[pos], key=lambda r: SEVERITY_ORDER.get(r["severity"], 9))
        severity = _severity(reasons)
        positions.append({
            "position": pos,
            "residue": sequence[pos - 1] if 0 < pos <= len(sequence) else "",
            "severity": severity,
            "reasons": reasons,
            "default_on": severity in DEFAULT_ON,
        })
    on = sum(1 for p in positions if p["default_on"])
    if sequence and on > len(sequence) * 0.4:
        notes.append(f"既定で {on} / {len(sequence)} 残基が保護対象です。"
                     "これだけ塞ぐと探索できる範囲がほとんど残らないので、絞り込みを検討してください")
    return {
        "chain": chain,
        "length": len(sequence),
        "positions": positions,
        "default_on_count": on,
        "sources": {
            "uniprot": acc,
            "interface_residues": contacts,
            "disordered_residues": disordered,
            "conserved_residues": conserved,
            "scan_job": (scan or {}).get("id"),
            "root_job": root.get("id"),
        },
        "notes": notes,
    }


def format_list(positions: list[dict[str, Any]]) -> str:
    """The settings field wants "K48, K63, G76"; build it from the ticked suggestions."""
    return ", ".join(f"{p.get('residue') or ''}{p['position']}" for p in positions)


# ---------------------------------------------------------------- after the fact
def _mutation_positions(spec: dict[str, Any], chain: str) -> list[tuple[str, int]]:
    comp = _component_for_chain(spec, chain)
    out = []
    for code in (comp or {}).get("mutations") or []:
        m = _POS.search(str(code))
        if m:
            out.append((str(code), int(m.group(1))))
    return out


def _gain_from_disorder(parent: dict[str, Any], child: dict[str, Any], chain: str,
                        threshold: float) -> dict[str, Any] | None:
    """How much of this variant's pLDDT gain came from residues the parent was unsure about.

    A hill-climb that only ever tidies up the floppy end of a molecule produces a rising score
    and a worse protein. The number that matters is not the mean but where the mean moved.
    """
    a, b = _plddt(parent, chain), _plddt(child, chain)
    if not a or len(a) != len(b):
        return None
    total = sum(b) - sum(a)
    if total <= 0:
        return None
    from_low = sum(y - x for x, y in zip(a, b, strict=True) if x < threshold)
    share = from_low / total
    return {
        "mean_gain": round(total / len(a), 2),
        "share_from_disordered": round(share, 3),
        "disordered_residues": sum(1 for x in a if x < threshold),
    }


def assess(db: Database, job: dict[str, Any]) -> dict[str, Any]:
    """Whether this finished prediction looks like a win or like a broken molecule.

    Three findings, any of which can stand alone:

    * a mutation sits on a residue with a known job (annotation, contact, conservation)
    * the score went up mainly by rewriting residues the parent was already unsure about
    * the model itself came back unusable (no coordinates, no confidence)
    """
    result = job.get("result") or {}
    spec = job.get("spec") or {}
    findings: list[dict[str, Any]] = []

    models = result.get("models") or []
    if job.get("status") == "succeeded" and not models:
        findings.append({"kind": "empty", "severity": "critical",
                         "text": "構造モデルが入っていません。結果として使えません"})

    chains = [c.get("chain") for c in (result.get("chains") or []) if c.get("chain")]
    parent = db.get_job(job["parent_id"]) if job.get("parent_id") else None
    s = get_settings()

    for chain in chains:
        muts = _mutation_positions(spec, chain)
        if not muts:
            continue
        try:
            suggestion = suggest_protected(db, job, chain)
        except Exception as exc:
            log.info("function_risk: %s の評価に失敗: %s", job.get("id"), exc)
            continue
        by_pos = {p["position"]: p for p in suggestion["positions"]}
        for code, pos in muts:
            hit = by_pos.get(pos)
            if hit and hit["severity"] in ("critical", "high"):
                findings.append({
                    "kind": "hit", "severity": hit["severity"], "chain": chain,
                    "mutation": code, "position": pos,
                    "reasons": [r["label"] for r in hit["reasons"]],
                    "text": f"{code} は{'・'.join(r['label'] for r in hit['reasons'])}に当たります",
                })

        if parent:
            gain = _gain_from_disorder(parent, job, chain, s.autopilot_disorder_plddt)
            if gain and gain["share_from_disordered"] >= 0.6 and gain["mean_gain"] >= 0.5:
                findings.append({
                    "kind": "disorder_gain", "severity": "high", "chain": chain,
                    "text": (f"平均 pLDDT の上がり幅 {gain['mean_gain']} のうち "
                             f"{gain['share_from_disordered'] * 100:.0f}% が、親で乱れていた残基の書き換えによるものです。"
                             "折りたたみが良くなったのではなく、予測器が迷う部分を潰しただけの可能性があります"),
                    "detail": gain,
                })

    level = "ok"
    if any(f["severity"] == "critical" for f in findings):
        level = "danger"
    elif findings:
        level = "warn"
    return {"job_id": job.get("id"), "level": level, "findings": findings}
