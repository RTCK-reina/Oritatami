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
        for log_name in ("job.log", "boltz.log"):
            add_file(f"logs/{log_name}", job_dir / log_name)
        zf.writestr("README.md", _readme(job, names))
    return buf.getvalue(), f"{safe_name(job['title'])}_{job['id'][-6:]}.zip"


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
