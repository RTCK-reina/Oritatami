"""LLM assistant: builds grounded context, asks for structured proposals, then verifies them.

The model's output is never applied blindly. Every proposal is checked against the
actual workbench (residue identities, chain ids), external databases (UniProt,
PubChem, CCD) and RDKit, and scored with ESM-2 where that is meaningful. Proposals keep
their issues list so the UI can show what was wrong instead of silently fixing it.

One exception, because it was the common case rather than a rare one: a mutation whose
wild-type residue is right and whose index is wrong is moved onto the position that residue
actually occupies, and the move is reported on the proposal. Rejecting it outright threw away
work the model had mostly right, and left the person to redo the arithmetic by hand.
"""

from __future__ import annotations

import json
import logging
import re
import time
from functools import lru_cache
from typing import Any

from . import chem, llm, sources
from .config import get_settings
from .db import Database
from .engines import esm
from .llm import SafeguardError as SafeguardError  # noqa: F401  (re-export so callers can catch it)
from .seq import SequenceError, clean_sequence, parse_mutations, repair_mutations

log = logging.getLogger("oritatami.assistant")

MODES = ("chat", "mutations", "complex", "design", "explain")

PROPOSAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "reply": {"type": "string"},
        "proposals": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "type": {"type": "string",
                             "enum": ["mutation_set", "add_ligand", "add_protein", "add_nucleic", "new_protein"]},
                    "title": {"type": "string"},
                    "rationale": {"type": "string"},
                    "chain": {"type": "string"},
                    "mutations": {"type": "array", "items": {"type": "string"}},
                    "name": {"type": "string"},
                    "smiles": {"type": "string"},
                    "ccd": {"type": "string"},
                    "uniprot": {"type": "string"},
                    "sequence": {"type": "string"},
                    "nucleic_type": {"type": "string", "enum": ["dna", "rna"]},
                },
                "required": ["type", "title", "rationale"],
            },
        },
    },
    "required": ["reply", "proposals"],
}

SYSTEM_PROMPT = """あなたは「Oritatami」という計算構造生物学ソフトウェアに組み込まれた科学支援アシスタントです。
このソフトウェアは学術・教育目的のタンパク質立体構造予測プラットフォームであり、
Boltz-2 (深層学習構造予測)・ESM-2 (タンパク質言語モデル)・外部データベース (UniProt / PubChem / CCD) を統合しています。

あなたの役割:
- 提示された分子系に対し、計算科学的に検証可能な提案を行う
- 提案はすべて自動検証される (残基照合・ESM-2 スコア・Boltz-2 構造予測)

必須の制約:
- アミノ酸置換は「K48R」の形式 (元残基・1始まり位置・新残基)。提示配列の実際の残基と一致させること。不一致はシステムが自動却下する
- 金属イオンは name ではなく ccd で指定する (ZN, CA, MG, FE, CU, MN, NA など)。PubChem での名前検索は不安定
- UniProt アクセッションと SMILES を未確認のまま生成しない。不明な場合は uniprot / smiles を省略し name だけ書く。システムがデータベース検索する
- uniprot・smiles・ccd・sequence の各欄には値そのものだけを書く。説明・注釈・「例：」・括弧書きを混ぜない (説明は rationale に書く)
- sequence 欄に「...」などの省略記号やプレースホルダを書かない。全長を書けないなら sequence を省略する
- 計算スコアは予測値であり実験値ではない。断定を避ける
- 回答は日本語。reply はユーザーへの科学的説明、proposals は実行可能な計算ジョブの案。案が不要なら proposals は空配列
- type=mutation_set の提案では mutations 配列を必ず 1 つ以上埋める。題名だけに変異を書いて配列を空にした提案は却下される
- 出力は指定 JSON スキーマに厳密に従う"""

MODE_INSTRUCTIONS = {
    "chat": "ユーザーの発言に答えてください。作業台に対して試せる具体案があれば proposals に入れてください。",
    "mutations": ("チェーン {chain} に対する変異セットを {count} 個提案してください (type=mutation_set、各 1〜5 変異)。"
                  "目的: {goal}\n"
                  "ESM-2 の上位候補、位置ごとの許容度、UniProt 注釈、予測の低信頼領域や界面を手がかりにしてください。"
                  "機能部位や結合部位を変える案はその影響を rationale に書いてください。\n"
                  "【重要】変異コードの元残基は下の表と必ず一致させること。"
                  "表にない残基を書いた提案はシステムが自動却下する。\n"
                  "コンテキストに「ESM-2 変異スキャン」がある場合、そこに並ぶ置換は"
                  "元残基が今の配列と一致することを確認済みなので、まずその中から選ぶこと。"
                  "一覧にない置換を使うときは下の表で元残基を必ず確認する。\n"
                  "「導入済み」と書かれた変異はもう適用されている。同じコードを書くと必ず却下されるので、"
                  "その位置を触るなら今の残基から書き直すこと。\n"
                  "チェーン {chain} の配列 (位置範囲 → 残基):\n{residues}"),
    "complex": ("この作業台に加えると意味のある、または面白い結合相手を {count} 個提案してください "
                "(type=add_protein / add_ligand / add_nucleic)。目的: {goal}\n"
                "天然の相互作用相手、既知の補因子・基質・阻害剤を優先し、UniProt 注釈の結合部位と対応づけてください。\n"
                "title は検索に使われる。add_ligand では PubChem で引ける化合物名 (英語の一般名か IUPAC 名) か "
                "PDB の CCD コード (ATP, ZN, HEM など) だけを書き、説明や用途を混ぜないこと "
                "(「金属イオン配位子 (ZN)」ではなく「ZN」)。用途は rationale に書く。\n"
                "add_protein では UniProt アクセッションを推測で書かないこと。確実でなければ空欄にし、"
                "title にタンパク質の一般名を書けばシステムが検索する。"),
    "design": ("新しいタンパク質配列を {count} 個設計してください (type=new_protein、sequence 必須)。"
               "目的: {goal}\n"
               "【必須】各配列は 60〜200 残基で設計する。40 残基未満はシステムが却下し、"
               "60 残基未満は「短すぎる」と警告される。\n"
               "rationale に設計意図 (二次構造の配置、疎水性パターン、モチーフ) を具体的に書く。\n"
               "ランダムな羅列は不可。ヘリックスなら 3〜4 残基ごとの疎水性配置 (a・d 位置: AILVMFW)、"
               "シートなら交互疎水性 (奇数: 疎水性, 偶数: 親水性)、"
               "連結には GGS や GGSGG のリンカーを使うなど、折り畳みの根拠が明確な配列にしてください。"
               "確認: sequence フィールドに書く文字数を数えて、60 以上であることを確認してから出力する。"),
    "explain": ("直近の予測結果を、構造生物学に詳しくない人にもわかるように解説してください。"
                "pLDDT・pTM・ipTM・PAE・親和性の数値が何を意味するか、この結果で注目すべき点、"
                "次に試すと面白いことを書き、試せる案は proposals に入れてください。補足: {goal}\n"
                "変異を提案する場合、元残基は下の表と必ず一致させ、mutations 配列に必ず入れること。\n"
                "チェーン {chain} の配列 (位置範囲 → 残基):\n{residues}"),
}


@lru_cache(maxsize=64)
def _uniprot_cached(accession: str) -> dict[str, Any]:
    return sources.uniprot_entry(accession)


def _component_accession(comp: dict[str, Any]) -> str | None:
    src = comp.get("source") or {}
    if src.get("db") in ("UniProt", "AFDB") and src.get("id"):
        return str(src["id"]).split("-")[0]
    return None


def _ranges(values: list[float], threshold: float) -> list[str]:
    out, start = [], None
    for i, v in enumerate(values + [threshold + 1]):
        if v < threshold and start is None:
            start = i
        elif v >= threshold and start is not None:
            out.append(f"{start + 1}-{i}" if i - start > 1 else f"{start + 1}")
            start = None
    return out


def numbered_residues(seq: str, per_row: int = 10, fixed: set[int] | None = None) -> str:
    """The chain laid out as position-labelled rows, with off-limits residues bracketed.

    The every-10 markers in the workbench context are not enough once a prediction
    result is also in the prompt: the model starts picking positions out of the
    pLDDT range list and guessing the residue, landing a few positions off (asking
    for T68A where 68 is H). Spelling out short numbered rows next to the actual
    question removes the counting step entirely.

    ``fixed`` marks positions the answer may not touch. A sentence saying so was
    ignored by 28% of proposals in a 901-proposal run — the two most-proposed mutations
    of all were the two most prominently forbidden ones. The model picks positions by
    reading this table, so the ban belongs in the table.
    """
    fixed = fixed or set()
    rows = []
    for start in range(0, len(seq), per_row):
        chunk = seq[start:start + per_row]
        cells = [f"[{c}]" if start + i + 1 in fixed else f" {c} " for i, c in enumerate(chunk)]
        rows.append(f"{start + 1:>4}-{start + len(chunk):<4}{''.join(cells)}")
    table = "\n".join(rows)
    if fixed:
        table = ("[ ] で囲まれた残基は変更禁止。これらの位置を含む提案はシステムが自動却下する。\n"
                 + table)
    return table


# 数えものはモデルにやらせない。9B は「リジンは 10 個」と平然と書き、27B でも直らない
# （どちらも実測）。組成と両末端はここで数えて渡し、本文はそれを引き写すだけにする。
_AA_JP = {"A": "アラニン", "R": "アルギニン", "N": "アスパラギン", "D": "アスパラギン酸",
          "C": "システイン", "Q": "グルタミン", "E": "グルタミン酸", "G": "グリシン",
          "H": "ヒスチジン", "I": "イソロイシン", "L": "ロイシン", "K": "リジン",
          "M": "メチオニン", "F": "フェニルアラニン", "P": "プロリン", "S": "セリン",
          "T": "トレオニン", "W": "トリプトファン", "Y": "チロシン", "V": "バリン"}
_FACT_POS_LIMIT = 30
_SUB_CODE = re.compile(r"^([ACDEFGHIKLMNPQRSTVWY])(\d{1,4})([ACDEFGHIKLMNPQRSTVWY])$")


def _sequence_facts(seq: str) -> list[str]:
    """Counts and terminal residues, computed rather than recalled."""
    seq = (seq or "").strip().upper()
    if not seq:
        return []
    out = ["確定値 (この配列から機械的に数えたもの。本文ではこの数字をそのまま使うこと):"]
    out.append(f"  長さ: {len(seq)} 残基")
    out.append(f"  N 末端の 5 残基: {' '.join(seq[:5])}")
    out.append(f"  C 末端の 5 残基: {' '.join(seq[-5:])} (最後は {len(seq)} 番の {seq[-1]})")
    for aa in sorted(set(seq)):
        pos = [i + 1 for i, c in enumerate(seq) if c == aa]
        name = _AA_JP.get(aa, aa)
        where = (", ".join(str(p) for p in pos[:_FACT_POS_LIMIT])
                 + ("…" if len(pos) > _FACT_POS_LIMIT else ""))
        out.append(f"  {aa} ({name}): {len(pos)} 個 — {where}")
    return out


def build_context(workbench: dict[str, Any], job: dict[str, Any] | None, scan: dict[str, Any] | None,
                  scan_chain: str | None) -> str:
    lines = ["# 現在の作業台"]
    comps = workbench.get("components") or []
    if not comps:
        lines.append("(空です)")
    for comp in comps:
        chains = ",".join(comp.get("chains") or []) or "未割当"
        label = comp.get("label") or ""
        if comp.get("type") in ("protein", "dna", "rna"):
            seq = comp.get("sequence", "")
            acc = _component_accession(comp)
            extra = f" UniProt {acc}" if acc else ""
            muts = comp.get("mutations") or []
            lines.append(f"- チェーン {chains} ({comp['type']}, {label}, {len(seq)} 残基{extra})")
            if muts:
                # Written as a plain list this reads as a menu of good moves and the model
                # proposes the same codes straight back; every one is then rejected because
                # the position already holds the new residue.
                lines.append(f"  元配列から導入済み (適用ずみ。同じ変異コードを再提案しないこと): {' '.join(muts)}")
            shown = seq if len(seq) <= 2500 else seq[:2500] + "…(省略)"
            lines.append(f"  配列: {shown}")
            if comp["type"] == "protein" and seq:
                marks = " ".join(f"{i}:{seq[i - 1]}" for i in range(10, len(seq) + 1, 10))
                lines.append(f"  位置の目印: {marks}")
                lines.extend("  " + line for line in _sequence_facts(seq))
        else:
            what = f"SMILES {comp.get('smiles')}" if comp.get("smiles") else f"CCD {comp.get('ccd')}"
            lines.append(f"- チェーン {chains} (ligand, {label}, {what})")
    if workbench.get("affinity_binder"):
        lines.append(f"親和性予測の対象: チェーン {workbench['affinity_binder']}")

    for comp in comps:
        acc = _component_accession(comp)
        if not acc:
            continue
        try:
            entry = _uniprot_cached(acc)
        except sources.SourceError as exc:
            lines.append(f"\n# UniProt {acc}: 取得失敗 ({exc})")
            continue
        lines.append(f"\n# UniProt {acc} ({entry['name']}, {entry.get('organism')})")
        for f in entry["function"][:6]:
            lines.append(f"- {f[:600]}")
        feats = [f for f in entry["features"] if f["type"] in
                 ("Active site", "Binding site", "Site", "Metal binding", "Disulfide bond", "Motif", "Domain",
                  "Mutagenesis")]
        if feats:
            lines.append("注釈 (位置は UniProt 配列基準。作業台の配列が同一なら同じ番号):")
            for f in feats[:40]:
                pos = f"{f['start']}" if f["start"] == f["end"] else f"{f['start']}-{f['end']}"
                detail = f.get("ligand") or f.get("description") or ""
                if f["type"] == "Mutagenesis" and f.get("alternative"):
                    detail = f"{f.get('original')}→{'/'.join(f['alternative'])}: {detail}"
                lines.append(f"- {f['type']} {pos} {detail}"[:300])

    if job and job.get("result"):
        res = job["result"]
        model = res["models"][0]
        conf = model.get("confidence", {})
        lines.append(f"\n# 直近の予測結果 ({job.get('title')})")
        lines.append("信頼度: " + ", ".join(
            f"{k}={conf[k]:.3f}" for k in ("confidence_score", "ptm", "iptm", "complex_plddt") if
            isinstance(conf.get(k), (int, float))))
        if conf.get("pair_chains_iptm"):
            pairs = []
            for a, row in conf["pair_chains_iptm"].items():
                for b, v in row.items():
                    if a < b and isinstance(v, (int, float)):
                        pairs.append(f"{a}-{b}:{v:.2f}")
            if pairs:
                lines.append("チェーン間 ipTM: " + " ".join(pairs))
        for chain, vals in (model.get("plddt") or {}).items():
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            low = _ranges(vals, 70.0)
            lines.append(f"チェーン {chain}: 平均 pLDDT {mean:.1f}; pLDDT<70 の区間 {', '.join(low) if low else 'なし'}")
        for chain, v in (model.get("ligand_plddt") or {}).items():
            lines.append(f"リガンド {chain}: 平均 pLDDT {v:.1f}")
        inter = res.get("interfaces") or {}
        for itf in (inter.get("interfaces") or [])[:8]:
            a, b = itf["chains"]
            lines.append(f"界面 {a}-{b}: {a} 側 {' '.join(itf['residues'][a][:30])} / {b} 側 {' '.join(itf['residues'][b][:30])}")
        aff = res.get("affinity")
        if aff:
            lines.append(
                f"親和性 (チェーン {aff.get('binder')}): log10(IC50/µM)={aff.get('affinity_pred_value'):.2f}"
                f" (IC50≈{aff.get('ic50_um')} µM, ΔG≈{aff.get('delta_g_kcal')} kcal/mol),"
                f" 結合する確率={aff.get('affinity_probability_binary'):.2f}")

    if scan:
        # A scan is reused whenever its sequence is close enough, so on a mutated workbench
        # part of it describes residues that are no longer there. Left in, those entries are
        # the single biggest source of rejected proposals: the instruction tells the model the
        # listed substitutions have verified wild-type letters, and for a stale entry that is
        # a lie, so it proposes E24A on a chain whose 24 is already A. Measured on the running
        # loop: 35 of 50 proposals rejected, and every mutation-set rejection was this.
        current = ""
        for comp in comps:
            if scan_chain in (comp.get("chains") or []) and comp.get("type") == "protein":
                current = comp.get("sequence") or ""
                break
        seq = scan["sequence"]
        lines.append(f"\n# ESM-2 変異スキャン (チェーン {scan_chain})")
        lines.append(f"疑似パープレキシティ {scan['pseudo_perplexity']} (低いほど自然な配列)")
        if current and current != seq:
            lines.append(f"注意: このスキャンは今の配列とは別の配列 ({len(seq)} 残基) に対して実行されたものです。"
                         "今の配列と残基が一致する項目だけを残してあります。")

        def _live(pos: int, wt: str) -> bool:
            if not current or current == seq:
                return True
            return 1 <= pos <= len(current) and current[pos - 1] == wt

        subs, dropped = [], 0
        for t in scan["top_substitutions"]:
            m = _SUB_CODE.match(str(t["mutation"]))
            if m and not _live(int(m.group(2)), m.group(1)):
                dropped += 1
                continue
            subs.append(f"{t['mutation']}({t['llr']:+.1f})")
            if len(subs) >= 25:
                break
        if subs:
            lines.append("LLR が高い置換 (ESM-2 が自然と見なす置換。元残基は今の配列と一致済み) 上位: "
                         + " ".join(subs))
        if dropped:
            lines.append(f"({dropped} 件は今の配列で元残基が変わっているため除外しました。"
                         "同じ変異を再提案しないこと)")
        tol = scan["position_tolerance"]
        rigid = sorted(range(len(tol)), key=lambda i: tol[i])[:20]
        rigid = [i for i in sorted(rigid) if _live(i + 1, seq[i])]
        if rigid:
            lines.append("置換に弱い位置 (保存的): " + " ".join(f"{seq[i]}{i + 1}" for i in rigid))
    return "\n".join(lines)



# ---------------------------------------------------------------- reply grounding
# Measured on ubiquitin: asked to name the residue at a given position, the model gets
# it right 1/10 of the time from the raw sequence and 4/10 with a numbered table. So any
# position it mentions in prose is a coin flip at best. Proposals are already verified
# residue by residue; this does the same for the explanation text, which the reader would
# otherwise have no way to check.
_JA_RESIDUE = {
    "アラニン": "A", "アルギニン": "R", "アスパラギン酸": "D", "アスパラギン": "N",
    "システイン": "C", "グルタミン酸": "E", "グルタミン": "Q", "グリシン": "G",
    "ヒスチジン": "H", "イソロイシン": "I", "ロイシン": "L", "リジン": "K", "リシン": "K",
    "メチオニン": "M", "フェニルアラニン": "F", "プロリン": "P", "セリン": "S",
    "トレオニン": "T", "スレオニン": "T", "トリプトファン": "W", "チロシン": "Y",
    "バリン": "V",
}
_AA1 = "ACDEFGHIKLMNPQRSTVWY"
_MUT_CODE = re.compile(rf"\b([{_AA1}])(\d{{1,4}})([{_AA1}])\b")
_RES_REF = re.compile(rf"\b([{_AA1}])(\d{{1,4}})\b")
_POS_JA = re.compile(r"位置\s*(\d{1,4})\s*(?:番目)?\s*(?:の|は|には)?\s*([ァ-ヴー]+酸?)")
# The first spelling of each residue wins, so the reverse map keeps the canonical name.
_JA_NAME: dict[str, str] = {}
for _ja, _aa in _JA_RESIDUE.items():
    _JA_NAME.setdefault(_aa, _ja)
# "グルタミン酸 (Q40)" and "Q40 (グルタミン酸)" — a name sitting next to a code.
_JA_BESIDE_CODE = re.compile(
    rf"(?:(?P<ja1>[ァ-ヴー]+酸?)\s*[（(]\s*(?P<aa>[{_AA1}])(?P<pos>\d{{1,4}})[A-Z]?\s*[)）]"
    rf"|(?P<aa2>[{_AA1}])(?P<pos2>\d{{1,4}})\s*[（(]\s*(?P<ja2>[ァ-ヴー]+酸?)\s*[)）])")


def _protein_sequences(workbench: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for comp in workbench.get("components") or []:
        if comp.get("type") != "protein" or not comp.get("sequence"):
            continue
        for chain in comp.get("chains") or ["?"]:
            out[chain] = comp["sequence"]
    return out


def check_reply_claims(reply: str, workbench: dict[str, Any], limit: int = 6) -> list[str]:
    """Residue-position claims in the prose that the actual sequences contradict.

    Three classes are caught: a mutation code or residue reference whose wild-type letter
    is not what the sequence holds, a position past the end of every chain, and a
    Japanese residue name written next to a code it does not match (the model wrote
    "グルタミン酸 (Q40)" — the letter was right, the name was glutamic acid instead of
    glutamine). Conservative: a claim is only reported when the sequences positively
    contradict it.
    """
    seqs = _protein_sequences(workbench)
    if not seqs or not reply:
        return []
    longest = max(len(v) for v in seqs.values())
    issues: list[str] = []
    seen: set[str] = set()

    def add(msg: str) -> None:
        if msg not in seen and len(issues) < limit:
            seen.add(msg)
            issues.append(msg)

    def where(pos: int) -> str:
        return ", ".join(f"チェーン {c} は {q[pos - 1]}" for c, q in sorted(seqs.items())
                         if pos <= len(q))

    # 1 & 2 — mutation codes and bare residue references.
    consumed: list[tuple[int, int]] = []
    codes: list[tuple[int, int, str, str]] = []  # start, position, residue, as written
    for m in _MUT_CODE.finditer(reply):
        codes.append((m.start(), int(m.group(2)), m.group(1), m.group(0)))
        consumed.append(m.span())
    strict = {c[0] for c in codes}  # a 3-part code is unambiguous; a bare ref may not be
    for m in _RES_REF.finditer(reply):
        if any(a <= m.start() < b for a, b in consumed):
            continue
        codes.append((m.start(), int(m.group(2)), m.group(1), m.group(0)))

    for at, pos, aa, written in codes:
        if pos < 1:
            continue
        in_range = {c: q for c, q in seqs.items() if pos <= len(q)}
        if not in_range:
            if at in strict or pos <= longest * 4:
                add(f"本文の「{written}」: 位置 {pos} はどのチェーンの配列長 (最長 {longest}) も超えています")
            continue
        if not any(q[pos - 1] == aa for q in in_range.values()):
            add(f"本文の「{written}」: 位置 {pos} の残基は {aa} ではなく {where(pos)} です")

    # 3 — "位置 N の <日本語名>"
    for m in _POS_JA.finditer(reply):
        aa = _JA_RESIDUE.get(m.group(2))
        pos = int(m.group(1))
        if not aa or pos < 1:
            continue
        in_range = {c: q for c, q in seqs.items() if pos <= len(q)}
        if in_range and not any(q[pos - 1] == aa for q in in_range.values()):
            add(f"本文の「{m.group(0)}」: 位置 {pos} の残基は {aa} ではなく {where(pos)} です")

    # 4 — a Japanese name written beside a code that means a different residue.
    for m in _JA_BESIDE_CODE.finditer(reply):
        name = m.group("ja1") or m.group("ja2")
        letter = m.group("aa") or m.group("aa2")
        pos = int(m.group("pos") or m.group("pos2"))
        if not name or not letter:
            continue
        expected = _JA_RESIDUE.get(name)
        if expected and expected != letter:
            add(f"本文の「{m.group(0).strip()}」: {letter}{pos} は{_JA_NAME.get(letter, letter)}で、"
                f"{name} ({expected}) ではありません")
    return issues


def _history_messages(thread: dict[str, Any], limit: int = 6) -> list[dict[str, str]]:
    msgs = []
    for m in thread["messages"][-limit * 2:]:
        if m["role"] == "user":
            msgs.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant":
            titles = "; ".join(p.get("title", "") for p in m.get("proposals", []))
            text = m["content"] + (f"\n(このとき出した案: {titles})" if titles else "")
            msgs.append({"role": "assistant", "content": text})
    return msgs


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))


# Small models often frame the object with a sentence or a code fence. Rather than
# throwing the whole answer away, hand it back once and let the model re-emit it.
_JSON_REPAIR = ("直前の出力は JSON として読めませんでした。"
                "指定された JSON スキーマに厳密に従う JSON だけを出力し直してください。"
                "説明文・コードフェンス・前後の文は付けないこと。")


def _record(db: Database, **kw: Any) -> None:
    """The call log is a side effect: it must never be the reason an answer is lost."""
    try:
        db.record_llm_call(**kw)
    except Exception as exc:  # noqa: BLE001 - logging must not break the answer
        log.warning("LLM 呼び出しの記録に失敗しました: %s", exc)


def _thread_title(mode: str, message: str, workbench: dict[str, Any],
                  job: dict[str, Any] | None) -> str:
    """A name the conversation picker can tell apart.

    Panel buttons ("LLM に解説させる", the starter chips) send no text of their own, so every
    thread they opened was called "変異の提案" — a dropdown of identical rows. Fall back to
    what the conversation is *about*: the workbench, or the result being discussed.
    """
    text = (message or "").strip()
    if text:
        return text[:40]
    subject = (workbench.get("workbench_name") or workbench.get("name") or "").strip()
    if not subject and job:
        subject = str(job.get("title") or "").strip()
    if not subject:
        comps = workbench.get("components") or []
        subject = str((comps[0] or {}).get("label") or "").strip() if comps else ""
    label = MODE_LABELS.get(mode, mode)
    return f"{label} · {subject}"[:40] if subject else f"{label} · {time.strftime('%m/%d %H:%M')}"


def _keep_alive(origin: str) -> float | str | None:
    """Machine-made calls drop the model the moment they are done.

    A person asking follow-up questions benefits from the model staying warm; the autopilot
    asks twice per generation and then hands the machine back to Boltz, so holding 6-8 GB of
    unified memory for the 15-minute default is pure contention.
    """
    return 0 if origin != "user" else None


def ask(db: Database, *, thread_id: str | None, mode: str, message: str, workbench: dict[str, Any],
        job: dict[str, Any] | None, scan: dict[str, Any] | None, scan_chain: str | None,
        focus_chain: str | None, count: int, persist: bool = True,
        fixed_positions: set[int] | None = None, verify_reply: bool = True,
        model: str | None = None, origin: str = "user") -> dict[str, Any]:
    """Ask the LLM and, unless ``persist`` is off, keep the exchange as a chat thread.

    The autopilot asks twice per generation and never reads the thread back — its own
    ``autopilot.json`` already holds the reply and the proposals. Left persisting, an
    overnight run buried the conversation picker under ~700 machine-made threads.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode}")
    if mode == "explain" and not (job and job.get("result")):
        raise ValueError("解説する予測結果がありません。先に予測を実行してください")
    count = max(1, min(int(count or 3), 8))
    thread = db.get_thread(thread_id) if thread_id else None
    if thread is None:
        if persist:
            thread = db.create_thread(_thread_title(mode, message, workbench, job))
        else:
            thread = {"id": None, "messages": []}

    protein_chains = [cid for c in workbench.get("components") or [] if c.get("type") == "protein"
                      for cid in c.get("chains") or []]
    chain = focus_chain or (protein_chains[0] if protein_chains else "A")
    goal = message.strip() or "特に指定なし。構造的に面白い変化が見られそうなもの"
    focus_comp = _find_component(workbench, chain)
    residues = (numbered_residues(focus_comp.get("sequence", ""), fixed=fixed_positions)
                if focus_comp and focus_comp.get("type") == "protein" else "(配列なし)")
    instruction = MODE_INSTRUCTIONS[mode].format(chain=chain, count=count, goal=goal,
                                                 residues=residues)
    context = build_context(workbench, job, scan, scan_chain)
    user_text = message.strip() if mode == "chat" else f"[{MODE_LABELS[mode]}] {goal}"

    # One system message: strict templates (qwen3.5) reject any system role that is
    # not the very first message, so prompt and context travel in one.
    messages = [{"role": "system", "content": f"{SYSTEM_PROMPT}\n\n{context}"},
                *_history_messages(thread),
                {"role": "user", "content": instruction if mode != "chat" else f"{instruction}\n\n{message}"}]
    used_model = model or get_settings().llm_model
    logged = dict(model=used_model, mode=mode, origin=origin,
                  job_id=(job or {}).get("id"), thread_id=thread["id"],
                  keep=get_settings().llm_log_limit)
    started = time.time()
    try:
        raw = llm.chat(messages, schema=PROPOSAL_SCHEMA, model=model, keep_alive=_keep_alive(origin))
    except Exception as exc:
        # A failure is the most informative row in the log, so it must not be the one row
        # that goes missing.
        _record(db, messages=messages, raw="", reply="", proposals=[], reply_issues=[],
                corrected=False, elapsed_sec=time.time() - started,
                error=f"{type(exc).__name__}: {exc}", **logged)
        raise
    elapsed = time.time() - started
    try:
        data = _parse_json(raw)
    except json.JSONDecodeError as exc:
        # A repair retry is cheap insurance for the common small-model failure — a
        # preamble sentence or a code fence around an otherwise valid object — and it
        # only runs on the failure path, so good answers never pay for it.
        try:
            raw2 = llm.chat([*messages, {"role": "assistant", "content": raw},
                             {"role": "user", "content": _JSON_REPAIR}],
                            schema=PROPOSAL_SCHEMA, model=model, keep_alive=_keep_alive(origin))
            data = _parse_json(raw2)
            raw = raw2
            log.info("JSON の出力し直しに成功しました")
        except (llm.LlmError, json.JSONDecodeError):
            _record(db, messages=messages, raw=raw, reply="", proposals=[], reply_issues=[],
                    corrected=False, elapsed_sec=time.time() - started,
                    error=f"JSONDecodeError: {exc}", **logged)
            raise llm.LlmError(
                f"LLM の出力を JSON として読めませんでした: {exc}\n---\n{raw[:800]}") from exc
        elapsed = time.time() - started
    reply = str(data.get("reply", "")).strip()
    proposals = [verify_proposal(p, workbench, chain) for p in data.get("proposals") or [] if isinstance(p, dict)]
    reply_issues = check_reply_claims(reply, workbench)
    corrected = False
    if reply_issues:
        log.info("LLM の本文に配列と矛盾する記述が %d 件ありました", len(reply_issues))
    if reply_issues and verify_reply:
        # The detector already knows exactly which claims are false, so handing them back is
        # cheaper and more reliable than hoping a bigger model gets it right the first time.
        fix = ("直前の回答の本文に、作業台の配列と矛盾する記述がありました:\n"
               + "\n".join(f"- {t}" for t in reply_issues)
               + "\n\n配列を数え直し、同じ質問にもう一度答えてください。"
                 "指摘された記述は削除するか正しい値に直すこと。"
                 "確信が持てない位置には言及しないこと。提案 (proposals) は前回と同じ内容で構いません。")
        try:
            raw2 = llm.chat([*messages, {"role": "assistant", "content": raw},
                             {"role": "user", "content": fix}], schema=PROPOSAL_SCHEMA, model=model,
                            keep_alive=_keep_alive(origin))
            data2 = _parse_json(raw2)
        except (llm.LlmError, json.JSONDecodeError) as exc:
            log.info("本文の訂正パスに失敗したので初回の回答を使います: %s", exc)
        else:
            reply2 = str(data2.get("reply", "")).strip()
            issues2 = check_reply_claims(reply2, workbench)
            if reply2 and len(issues2) < len(reply_issues):
                props2 = [verify_proposal(p, workbench, chain)
                          for p in data2.get("proposals") or [] if isinstance(p, dict)]
                log.info("本文を訂正しました (矛盾 %d 件 → %d 件)", len(reply_issues), len(issues2))
                reply, reply_issues, corrected = reply2, issues2, True
                if props2:
                    proposals = props2
        elapsed = time.time() - started

    _record(db, messages=messages, raw=raw, reply=reply, proposals=proposals,
            reply_issues=reply_issues, corrected=corrected, elapsed_sec=elapsed, **logged)

    now = time.time()
    thread["messages"].append({"role": "user", "content": user_text, "mode": mode, "created_at": now})
    thread["messages"].append({"role": "assistant", "content": reply, "proposals": proposals, "mode": mode,
                               "reply_issues": reply_issues, "corrected": corrected,
                               "created_at": time.time(), "elapsed_sec": round(elapsed, 1),
                               "model": model or get_settings().llm_model})
    if thread["id"] is not None:
        db.save_thread(thread["id"], thread["messages"])
    return {"thread_id": thread["id"], "reply": reply, "proposals": proposals,
            "reply_issues": reply_issues, "corrected": corrected,
            "model": model or get_settings().llm_model, "elapsed_sec": round(elapsed, 1)}


MODE_LABELS = {"chat": "会話", "mutations": "変異の提案", "complex": "複合体の提案",
               "design": "新しい配列の設計", "explain": "結果の解説"}


# ------------------------------------------------------------------ verification
def _find_component(workbench: dict[str, Any], chain: str) -> dict[str, Any] | None:
    for comp in workbench.get("components") or []:
        if chain in (comp.get("chains") or []):
            return comp
    return None


def verify_proposal(p: dict[str, Any], workbench: dict[str, Any], default_chain: str) -> dict[str, Any]:
    ptype = p.get("type")
    out: dict[str, Any] = {
        "type": ptype,
        "title": str(p.get("title", "")).strip() or "(無題)",
        "rationale": str(p.get("rationale", "")).strip(),
        "issues": [],
        "status": "ok",
        "apply": None,
    }
    try:
        if ptype == "mutation_set":
            _verify_mutations(p, out, workbench, default_chain)
        elif ptype == "add_ligand":
            _verify_ligand(p, out)
        elif ptype == "add_protein":
            _verify_protein(p, out)
        elif ptype == "add_nucleic":
            kind = p.get("nucleic_type") or "dna"
            seq = clean_sequence(p.get("sequence", ""), kind)
            out["apply"] = {"action": "add_component",
                            "component": {"type": kind, "label": p.get("name") or out["title"], "sequence": seq}}
        elif ptype == "new_protein":
            _verify_new_protein(p, out)
        else:
            out["status"] = "invalid"
            out["issues"].append(f"未知の提案種別: {ptype}")
    except (SequenceError, chem.ChemError, sources.SourceError, ValueError) as exc:
        out["status"] = "invalid"
        out["issues"].append(str(exc))
    if out["status"] == "ok" and (out["issues"] or out.get("repaired")):
        out["status"] = "warning"
    return out


def _verify_mutations(p: dict[str, Any], out: dict[str, Any], workbench: dict[str, Any], default_chain: str) -> None:
    chain = (p.get("chain") or default_chain).strip()
    comp = _find_component(workbench, chain)
    if comp is None or comp.get("type") != "protein":
        raise ValueError(f"チェーン {chain} は作業台のタンパク質にありません")
    seq = comp["sequence"]
    # The model names the residue it means and then gets the index wrong — a leading
    # methionine, UniProt numbering against a construct. Dropping the mutation throws away
    # the part it had right, so the position is moved to where that residue actually is and
    # the move is written on the card. What cannot be placed is still dropped.
    parsed, rejected = [], []
    for token in p.get("mutations") or []:
        token = str(token).strip()
        if ":" in token:
            token = token.split(":", 1)[1]
        try:
            parsed.extend(parse_mutations(token))
        except SequenceError as exc:
            rejected.append(f"{token}: {exc}")
    valid, repairs, dropped = repair_mutations(seq, parsed)
    rejected.extend(dropped)
    # Repairs are kept apart from problems: the proposal is usable, and the card says what
    # was moved. They still make the status "warning" — the applied mutation is not the one
    # the model wrote, and that has to be visible without opening anything.
    out["repaired"] = repairs
    out["issues"].extend(rejected)
    if not valid:
        raise ValueError("有効な変異がありません")
    out["chain"] = chain
    out["mutations"] = [m.code for m in valid]
    out["rejected"] = rejected
    try:
        out["esm"] = esm.score_mutations(seq, valid, wait=20)
    except Exception as exc:  # ESM is an optional signal; report why it is missing
        out["esm"] = None
        out["issues"].append(f"ESM-2 スコアを計算できませんでした: {exc}")
    out["apply"] = {"action": "mutate", "chain": chain, "mutations": out["mutations"]}


# Common inorganic ions: keyed by lowercase name/formula → CCD code.
# PubChem's name search is unreliable for these; go straight to CCD.
_ION_CCD: dict[str, str] = {
    "zn": "ZN", "zn2+": "ZN", "zinc": "ZN", "zinc ion": "ZN", "zn(ii)": "ZN",
    "ca": "CA", "ca2+": "CA", "calcium": "CA", "calcium ion": "CA",
    "mg": "MG", "mg2+": "MG", "magnesium": "MG", "magnesium ion": "MG",
    "fe": "FE", "fe2+": "FE2", "fe3+": "FE", "iron": "FE", "iron ion": "FE",
    "cu": "CU", "cu2+": "CU", "copper": "CU", "copper ion": "CU",
    "mn": "MN", "mn2+": "MN", "manganese": "MN", "manganese ion": "MN",
    "na": "NA", "na+": "NA", "sodium": "NA", "sodium ion": "NA",
    "k": "K", "k+": "K", "potassium": "K", "potassium ion": "K",
    "cl": "CL", "cl-": "CL", "chloride": "CL",
    "po4": "PO4", "phosphate": "PO4",
    "so4": "SO4", "sulfate": "SO4",
    "no3": "NO3", "nitrate": "NO3",
    "co": "CO", "cobalt": "CO", "co2+": "CO",
    "ni": "NI", "nickel": "NI", "ni2+": "NI",
    "zn(2+)": "ZN", "ca(2+)": "CA", "mg(2+)": "MG",
}


def _ion_ccd_from_name(name: str) -> str | None:
    """Return a CCD code if name unambiguously identifies a common inorganic ion."""
    return _ION_CCD.get(name.lower().strip())


def _verify_ligand(p: dict[str, Any], out: dict[str, Any]) -> None:
    name = (p.get("name") or out["title"]).strip()
    smiles = (p.get("smiles") or "").strip()
    ccd = (p.get("ccd") or "").strip().upper()

    # Shortcut: if the model forgot to set ccd for a well-known ion, infer it from the name.
    if not ccd:
        inferred = _ion_ccd_from_name(name)
        if inferred:
            ccd = inferred
            out["issues"].append(f"名前「{name}」から CCD コード {ccd} を推定しました")
    component: dict[str, Any] = {"type": "ligand", "label": name}
    if ccd:
        try:
            component["ccd"] = chem.validate_ccd(ccd)
            info = sources.ccd_info(ccd)
            if info and info.get("name"):
                out["resolved"] = {"via": "CCD", "name": info["name"], "formula": info.get("formula")}
        except chem.ChemError as exc:
            out["issues"].append(str(exc))
            ccd = ""
    if "ccd" not in component and smiles:
        try:
            out["chem"] = chem.describe_smiles(smiles)
            component["smiles"] = smiles
        except chem.ChemError as exc:
            out["issues"].append(f"LLM の SMILES は無効でした: {exc}")
    if "ccd" not in component and "smiles" not in component:
        found = sources.pubchem_lookup(name)
        component["smiles"] = found["smiles"]
        out["chem"] = chem.describe_smiles(found["smiles"])
        out["resolved"] = {"via": "PubChem", "name": found["name"], "cid": found["cid"]}
        if smiles:
            out["issues"].append("PubChem の SMILES に置き換えました")
    if "smiles" in component and out.get("chem") and not out["chem"]["affinity_ok"]:
        out["issues"].append("親和性予測の推奨範囲 (重原子 56 以下・1 分子) を外れています")
    out["apply"] = {"action": "add_component", "component": component}


# The model is told to put only an accession in `uniprot`, but it routinely appends a
# gloss ("Q9Y2Z3 (例：SCF のサブユニット)"). Matching the accession body anywhere in the
# field recovers those instead of failing the whole proposal on a format check.
_ACCESSION_IN_TEXT = re.compile(
    r"\b(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})(?:-\d+)?\b")


# Words too generic to count as evidence that an accession is the protein asked for.
_NAME_STOPWORDS = frozenset({
    "protein", "human", "homo", "sapiens", "chain", "subunit", "complex", "domain",
    "isoform", "putative", "probable", "uncharacterized", "family", "type", "like",
})


def _name_matches(proposed: str, entry: dict[str, Any]) -> bool:
    """Does the fetched entry plausibly correspond to the name the model asked for?

    The model invents accessions that happen to exist but belong to something else
    entirely (it asked for UBAP1 and Q9Y2Z4 is a tRNA ligase). A shared meaningful word
    between the requested name and the entry's name, gene symbols or id is weak
    evidence, but its complete absence is a strong signal the accession is wrong.
    """
    def tokens(text: str) -> set[str]:
        return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower())
                if len(w) > 2 and w not in _NAME_STOPWORDS}

    want = tokens(proposed)
    if not want:
        return True  # nothing to check against
    have = tokens(entry.get("name", "")) | tokens(entry.get("entry_name", "")) \
        | tokens(" ".join(entry.get("genes") or [])) | tokens(entry.get("accession", ""))
    return bool(want & have)


def _clean_accession(raw: str) -> tuple[str, bool]:
    """Return (accession, salvaged). `salvaged` marks a field that carried extra prose."""
    text = (raw or "").strip().upper()
    if not text or sources.ACCESSION_RE.match(text):
        return text, False
    m = _ACCESSION_IN_TEXT.search(text)
    return (m.group(0), True) if m else (text, False)


def _verify_protein(p: dict[str, Any], out: dict[str, Any]) -> None:
    acc, salvaged = _clean_accession(p.get("uniprot") or "")
    name = (p.get("name") or out["title"]).strip()
    entry = None
    mismatched: dict[str, Any] | None = None
    if acc:
        try:
            entry = _uniprot_cached(acc)
            if salvaged:
                out["issues"].append(
                    f"UniProt 欄に説明文が混ざっていたため「{acc}」として解釈しました。"
                    "意図した相手か確認してください")
            # An accession can be well-formed, resolve, and still be useless: obsolete
            # and demerged entries come back with an empty sequence. Adding one would
            # queue a prediction with no residues in it.
            if not (entry.get("sequence") or "").strip():
                out["issues"].append(
                    f"UniProt {acc} は配列を持たないエントリ (廃止・統合済みの可能性) でした。名前で検索します")
                entry = None
            elif name and not _name_matches(name, entry):
                out["issues"].append(
                    f"UniProt {acc} の中身は「{entry.get('name')}」で、指定された「{name}」と一致しません。"
                    "アクセッションの誤りとみなし、名前で検索し直します")
                mismatched, entry = entry, None
        except sources.SourceError as exc:
            out["issues"].append(f"UniProt {acc} を取得できませんでした: {exc}")
    if entry is None and p.get("sequence"):
        # A sequence the model mangled (placeholders, ellipses) must not abort the
        # proposal — the name search below is the more reliable route anyway.
        try:
            seq = clean_sequence(p["sequence"], "protein")
        except SequenceError as exc:
            out["issues"].append(f"LLM が書いた配列は使えませんでした ({exc})。名前で検索します")
        else:
            out["apply"] = {"action": "add_component",
                            "component": {"type": "protein", "label": name,
                                          "sequence": seq, "msa": "server"}}
            out["issues"].append("配列は LLM が書いたもので、データベースで確認されていません")
            return
    if entry is None:
        hits = sources.uniprot_search(name, size=5)
        if not hits:
            raise ValueError(f"UniProt で「{name}」が見つかりません")
        entry = next(
            (e for e in (_uniprot_cached(h["accession"]) for h in hits)
             if (e.get("sequence") or "").strip()),
            None,
        )
        if entry is None and mismatched is not None:
            entry = mismatched
            out["issues"].append(
                f"名前検索でも見つからなかったため、指定された {entry['accession']} をそのまま使います")
        if entry is None:
            raise ValueError(f"UniProt で「{name}」に配列を持つエントリが見つかりません")
        out["issues"].append(f"名前検索で {entry['accession']} ({entry['name']}, {entry.get('organism')}) に解決しました。意図した相手か確認してください")
    seq = entry["sequence"]
    if not seq.strip():
        raise ValueError(f"UniProt {entry.get('accession')} の配列が空です")
    if len(seq) > 1200:
        out["issues"].append(f"{len(seq)} 残基と長く、予測に時間とメモリを要します。ドメインを切り出すことを検討してください")
    out["resolved"] = {"via": "UniProt", "accession": entry["accession"], "name": entry["name"],
                       "organism": entry.get("organism"), "length": len(seq)}
    out["apply"] = {"action": "add_component", "component": {
        "type": "protein", "label": entry["name"][:60], "sequence": seq, "msa": "server",
        "source": {"db": "UniProt", "id": entry["accession"]}}}


def _verify_new_protein(p: dict[str, Any], out: dict[str, Any]) -> None:
    seq = clean_sequence(p.get("sequence", ""), "protein")
    if len(seq) < 40:
        raise ValueError(f"配列が短すぎます ({len(seq)} 残基)。de novo 設計は 60 残基以上を推奨します")
    if len(seq) < 60:
        out["issues"].append(f"配列が {len(seq)} 残基と短めです。60 残基以上にすると折り畳みの評価が安定します")
    stats = {"length": len(seq)}
    hydrophobic = sum(seq.count(a) for a in "AILMFVWY") / len(seq)
    stats["hydrophobic_fraction"] = round(hydrophobic, 2)
    top = max(set(seq), key=seq.count)
    stats["most_common"] = f"{top} {seq.count(top) / len(seq):.0%}"
    if seq.count(top) / len(seq) > 0.4:
        out["issues"].append(f"{top} が {seq.count(top) / len(seq):.0%} を占める低複雑度の配列です")
    try:
        stats["pseudo_perplexity"] = esm.pseudo_perplexity(seq, wait=20)
    except Exception as exc:
        out["issues"].append(f"ESM-2 の評価を計算できませんでした: {exc}")
    out["stats"] = stats
    out["sequence"] = seq
    out["apply"] = {"action": "new_protein",
                    "component": {"type": "protein", "label": p.get("name") or out["title"], "sequence": seq,
                                  "msa": "single"}}
