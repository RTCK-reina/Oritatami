"""Export job results as a self-contained zip (structures, scores, inputs, logs, a README)."""

from __future__ import annotations

import csv
import io
import json
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

from . import __version__, structure
from .config import exports_dir, jobs_dir

_SAFE = re.compile(r"[^\w\-]+")  # \w is Unicode-aware, so Japanese titles survive


def safe_name(text: str, fallback: str = "oritatami") -> str:
    name = _SAFE.sub("_", text or "").strip("._")
    return (name or fallback)[:80]


def _fasta(rows: list[tuple[str, str]]) -> str:
    out = []
    for header, seq in rows:
        out.append(f">{header}")
        out.extend(seq[i:i + 60] for i in range(0, len(seq), 60))
    return "\n".join(out) + "\n"


def _readme(job: dict[str, Any], files: list[str]) -> str:
    lines = [
        f"# {job['title']}",
        "",
        f"Oritatami {__version__} で書き出したジョブ {job['id']} ({job['kind']})。",
        f"作成: {time.strftime('%Y-%m-%d %H:%M', time.localtime(job['created_at']))}",
        "",
        "値はすべて計算による予測で、実験値ではありません。",
        "",
        "## ファイル",
    ]
    notes = {
        "summary.json": "結果の全データ (信頼度・pLDDT・PAE 縮約・界面・親和性など)",
        "sequences.fasta": "入力配列",
        "input/": "Boltz に渡した入力 YAML",
        "structures/": "予測構造 (mmCIF。B-factor 列が pLDDT)",
        "structures_pdb/": "同じ構造の PDB 形式 (変換できたもののみ)",
        "boltz/": "Boltz の出力 (confidence / affinity JSON, pLDDT / PAE の npz)",
        "scan_matrix.csv": "ESM-2 の変異スコア行列 (行=位置, 列=置換先アミノ酸, 値=LLR)",
        "methods_ja.txt": "論文メソッド記述向けテキスト (日本語)",
        "methods_en.txt": "論文メソッド記述向けテキスト (英語)",
        "logs/": "実行ログ",
    }
    for key, text in notes.items():
        if any(f == key or f.startswith(key) for f in files):
            lines.append(f"- {key}: {text}")
    return "\n".join(lines) + "\n"


def build_zip(job: dict[str, Any]) -> tuple[bytes, str]:
    """Return (zip bytes, suggested file name)."""
    job_dir = jobs_dir() / job["id"]
    buf = io.BytesIO()
    names: list[str] = []
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        def add_bytes(name: str, data: bytes | str) -> None:
            zf.writestr(name, data)
            names.append(name)

        def add_file(name: str, path: Path) -> None:
            if path.is_file():
                zf.write(path, name)
                names.append(name)

        result = job.get("result") or {}
        add_bytes("summary.json", json.dumps({k: v for k, v in job.items() if k != "live"}, ensure_ascii=False, indent=2))
        if job["kind"] == "predict":
            spec = result.get("normalized_spec") or job["spec"]
            seqs = [(f"{','.join(c['chains'])}|{c['type']}|{c.get('label') or ''}", c["sequence"])
                    for c in spec.get("components", []) if c.get("sequence")]
            if seqs:
                add_bytes("sequences.fasta", _fasta(seqs))
            for path in sorted((job_dir / "input").glob("*.yaml")):
                add_file(f"input/{path.name}", path)
            for model in result.get("models", []):
                cif = job_dir / model["file"]
                add_file(f"structures/model_{model['index']}.cif", cif)
                try:
                    pdb_text = structure.to_pdb(cif)
                    add_bytes(f"structures_pdb/model_{model['index']}.pdb", pdb_text)
                except Exception:  # PDB format has hard limits (chain ids, atom count); mmCIF is always included
                    pass
            pred = job_dir / "out" / "boltz_results_complex" / "predictions" / "complex"
            if pred.exists():
                for path in sorted(pred.iterdir()):
                    if path.suffix in (".json", ".npz"):
                        add_file(f"boltz/{path.name}", path)
        elif job["kind"] == "scan" and result:
            seq = result.get("sequence", "")
            add_bytes("sequences.fasta", _fasta([(job["title"], seq)]))
            rows = io.StringIO()
            writer = csv.writer(rows)
            alphabet = result.get("alphabet", "")
            writer.writerow(["position", "wild_type", *alphabet, "tolerance"])
            for i, row in enumerate(result.get("matrix", [])):
                writer.writerow([i + 1, seq[i] if i < len(seq) else "", *row,
                                 (result.get("position_tolerance") or [None] * (i + 1))[i]])
            add_bytes("scan_matrix.csv", rows.getvalue())
        elif job["kind"] == "refine" and result:
            add_bytes("sequences.fasta", _fasta([
                ("start", (result.get("history") or [{}])[0].get("sequence", "")),
                (f"refined|PPPL={result.get('pseudo_perplexity')}", result.get("sequence", "")),
            ]))
        add_bytes("methods_ja.txt", methods_text(job, "ja"))
        add_bytes("methods_en.txt", methods_text(job, "en"))
        for log_name in ("job.log", "boltz.log"):
            add_file(f"logs/{log_name}", job_dir / log_name)
        zf.writestr("README.md", _readme(job, names))
    return buf.getvalue(), f"{safe_name(job['title'])}_{job['id'][-6:]}.zip"


def _compositions(spec: dict[str, Any]) -> tuple[str, str]:
    """Chain counts by type, (English, Japanese) — e.g. '2 protein chains + 1 ligand'."""
    counts: dict[str, int] = {}
    for comp in spec.get("components") or []:
        n = len(comp.get("chains") or []) or 1
        counts[comp.get("type", "?")] = counts.get(comp.get("type", "?"), 0) + n
    _EN = {"protein": "protein chain", "dna": "DNA strand", "rna": "RNA strand", "ligand": "ligand"}
    _JA = {"protein": "タンパク質鎖", "dna": "DNA 鎖", "rna": "RNA 鎖", "ligand": "リガンド"}
    en = " + ".join(f"{n} {_EN.get(t, t)}{'s' if n > 1 else ''}" for t, n in counts.items()) or "none"
    ja = "、".join(f"{_JA.get(t, t)} {n} 本" for t, n in counts.items()) or "なし"
    return en, ja


def _msa_desc(spec: dict[str, Any]) -> tuple[str, str]:
    proteins = [c for c in spec.get("components") or [] if c.get("type") == "protein"]
    server = sum(1 for c in proteins if c.get("msa") == "server")
    single = len(proteins) - server
    if server and not single:
        return ("with multiple sequence alignments from the ColabFold MMseqs2 server",
                "MSA は ColabFold MMseqs2 サーバーで生成")
    if single and not server:
        return ("as single sequences without an MSA", "MSA は使わず単一配列のみ")
    return (f"with server MSAs for {server} protein(s) and single-sequence input for {single}",
            f"{server} 本はサーバー生成の MSA、{single} 本は単一配列")


def methods_text(job: dict[str, Any], lang: str = "ja") -> str:
    """A methods-section-ready text: the engine, parameters and versions behind this
    job's numbers, so a paper or lab note can state exactly how they were produced."""
    result = job.get("result") or {}
    spec = result.get("normalized_spec") or job.get("spec") or {}
    created = time.strftime("%Y-%m-%d", time.localtime(job["created_at"]))
    kind = job["kind"]
    comp_en, comp_ja = _compositions(spec)
    if kind == "predict":
        p = spec.get("params") or {}
        msa_en, msa_ja = _msa_desc(spec)
        pot_en = " and Boltz-2 potentials" if p.get("use_potentials") else ""
        pot_ja = "、Boltz-2 ポテンシャルあり" if p.get("use_potentials") else ""
        seed_en = f", random seed {p['seed']}" if p.get("seed") is not None else ""
        seed_ja = f"、シード {p['seed']}" if p.get("seed") is not None else ""
        if lang == "en":
            return (
                "Methods\n"
                "=======\n\n"
                f"Structure prediction was performed with Boltz-2 (Abramson et al., Nature 2024) "
                f"via Oritatami {__version__}.\n"
                f"The input was {comp_en}, modelled {msa_en}.\n"
                f"Sampling used {p.get('diffusion_samples')} diffusion sample(s), "
                f"{p.get('recycling_steps')} recycling steps and {p.get('sampling_steps')} sampling steps"
                f"{pot_en}{seed_en}; the compute device setting was '{p.get('accelerator', 'auto')}'.\n\n"
                f"Job {job['id']} ran on {created}. All values are computational predictions, "
                "not measurements.\n")
        return (
            "方法\n"
            "====\n\n"
            f"構造予測は Boltz-2 (Abramson et al., Nature 2024) を Oritatami {__version__} 経由で実行した。\n"
            f"入力は {comp_ja}。{msa_ja}。\n"
            f"サンプリングは拡散サンプル {p.get('diffusion_samples')} 個、リサイクル {p.get('recycling_steps')} 回、"
            f"サンプリングステップ {p.get('sampling_steps')}{pot_ja}{seed_ja}。"
            f"計算デバイス設定は '{p.get('accelerator', 'auto')}'。\n\n"
            f"ジョブ {job['id']}、{created} に実行。値はすべて計算による予測で、実験値ではない。\n")
    if kind == "scan":
        model = result.get("model") or "ESM-2"
        seq = result.get("sequence") or ""
        if lang == "en":
            return (
                "Methods\n"
                "=======\n\n"
                f"Every possible single substitution of the {len(seq)}-residue sequence was scored "
                f"with the protein language model {model} via Oritatami {__version__}, reported as "
                "per-mutation log-likelihood ratios against the wild type.\n"
                f"Pseudo-perplexity of the wild-type sequence was {result.get('pseudo_perplexity')}.\n\n"
                f"Job {job['id']} ran on {created}. All values are computational predictions, "
                "not measurements.\n")
        return (
            "方法\n"
            "====\n\n"
            f"{len(seq)} 残基の配列に対し、可能なすべての一点置換をタンパク質言語モデル {model} で "
            f"Oritatami {__version__} 経由スコアリングし、野生型に対する対数尤度比 (LLR) として報告した。\n"
            f"野生型配列の疑似パープレキシティは {result.get('pseudo_perplexity')}。\n\n"
            f"ジョブ {job['id']}、{created} に実行。値はすべて計算による予測で、実験値ではない。\n")
    if kind == "refine":
        s = job.get("spec") or {}
        seed_en = f", seed {s['seed']}" if s.get("seed") is not None else ""
        seed_ja = f"、シード {s['seed']}" if s.get("seed") is not None else ""
        if lang == "en":
            return (
                "Methods\n"
                "=======\n\n"
                f"The sequence was refined by masked resampling with {result.get('model') or 'ESM-2'} "
                f"via Oritatami {__version__}: {s.get('rounds')} rounds masking {s.get('fraction')} of the "
                f"least-likely positions, sampled at temperature {s.get('temperature')}{seed_en}; each round "
                "was kept only when pseudo-perplexity did not worsen.\n"
                f"Pseudo-perplexity moved {result.get('start_pseudo_perplexity')} → "
                f"{result.get('pseudo_perplexity')}.\n\n"
                f"Job {job['id']} ran on {created}. All values are computational predictions, "
                "not measurements.\n")
        return (
            "方法\n"
            "====\n\n"
            f"{result.get('model') or 'ESM-2'} によるマスクド再サンプリングで配列を最適化した "
            f"(Oritatami {__version__}): 尤度の低い位置を {s.get('fraction')} の割合でマスクし、"
            f"温度 {s.get('temperature')} で {s.get('rounds')} ラウンドサンプリング{seed_ja}。"
            "各ラウンドは疑似パープレキシティが悪化しない場合のみ採用した。\n"
            f"疑似パープレキシティは {result.get('start_pseudo_perplexity')} → "
            f"{result.get('pseudo_perplexity')} となった。\n\n"
            f"ジョブ {job['id']}、{created} に実行。値はすべて計算による予測で、実験値ではない。\n")
    return f"Oritatami {__version__} job {job['id']} ({kind}), ran on {created}.\n"


def save_to_exports(data: bytes, filename: str) -> Path:
    folder = exports_dir()
    stem = safe_name(Path(filename).stem)
    suffix = re.sub(r"[^.A-Za-z0-9]", "", Path(filename).suffix)[:10]
    target = folder / f"{stem}{suffix}"
    n = 1
    while target.exists():
        n += 1
        target = folder / f"{stem}-{n}{suffix}"
    target.write_bytes(data)
    return target


def reveal(path: Path) -> None:
    """Select the file in Finder (macOS). Silently does nothing elsewhere."""
    if sys.platform == "darwin":
        subprocess.Popen(["open", "-R", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
