"""Boltz-2 structure prediction through the ``boltz predict`` CLI.

The CLI runs as a child process in its own session so a cancel can terminate the
whole process group (Boltz spawns dataloader workers). Outputs are parsed into a
JSON result that the UI renders; the raw files stay in the job directory.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .. import chem, structure
from ..config import get_settings, msa_cache_dir, resolve_boltz_bin
from ..seq import SequenceError, clean_sequence
from .supervise import LIVE_NAME as LIVE_MEMORY_NAME
from .supervise import PEAK_NAME as PEAK_MEMORY_NAME

# Boltz runs out of chain letters long before this; the cap is here so a runaway
# client cannot make the server build a 200-chain YAML before failing.
MAX_COMPONENTS = 60
CHAIN_IDS = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + [f"{a}{b}" for a in "ABCDEFGH" for b in "0123456789"]
INPUT_NAME = "complex"


class PredictionError(RuntimeError):
    """A failed prediction. `kind` drives the recovery actions the UI offers (retry without MSA, on CPU, ...)."""

    def __init__(self, message: str, kind: str = "error") -> None:
        super().__init__(message)
        self.kind = kind


FAILURE_HINTS = {
    "oom": ("メモリが足りませんでした。サンプル数を 1 にする、構成要素やコピー数を減らす、長い配列をドメインに"
            "切り出す、のいずれかを試してください。「CPU で再実行」も選べます (かなり遅くなります)。"),
    "msa": ("MSA サーバー (ColabFold) との通信に失敗しました。時間をおいて再実行するか、「MSA なしで再実行」を"
            "試してください (進化情報を使わないため精度は下がります)。"),
    "download": "Boltz の重み・化学辞書をダウンロードできませんでした。ネットワーク接続を確認して再実行してください。",
    "input": "入力を Boltz が解釈できませんでした。配列、リガンドの SMILES / CCD コード、拘束の指定を確認してください。",
}

_OOM_RE = re.compile(r"out of memory|Invalid buffer size|MPSNDArray.*(?:allocate|limit)|Cannot allocate memory", re.I)
_MSA_RE = re.compile(r"MSA server|MMseqs2 API|Too many failed attempts for the MSA|msa_server|colabfold", re.I)
_DOWNLOAD_RE = re.compile(r"Failed to download|checkpoint .*(?:corrupt|incomplete)|Could not read .* checkpoint", re.I)
_INPUT_RE = re.compile(r"Unable to parse|Invalid SMILES|Could not find binder|CCD component|not found in the CCD|"
                       r"KeyError: '[A-Z0-9]{1,5}'|Failed to parse|input\(s\) failed", re.I)


def classify_failure(log_text: str, returncode: int | None, last_phase: str) -> str:
    tail = log_text[-20000:]
    if returncode in (-9, 137) or _OOM_RE.search(tail):
        return "oom"
    if _DOWNLOAD_RE.search(tail) or last_phase == "download":
        return "download"
    if _INPUT_RE.search(tail):
        return "input"
    if last_phase == "msa" or (_MSA_RE.search(tail) and "Running structure prediction" not in tail
                               and re.search(r"Exception|Error", tail)):
        return "msa"
    if last_phase == "preprocess":
        return "input"
    return "error"


def _failure(summary: str, log_text: str, returncode: int | None, last_phase: str) -> PredictionError:
    kind = classify_failure(log_text, returncode, last_phase)
    tail = "\n".join(log_text.replace("\r", "\n").splitlines()[-40:])
    hint = FAILURE_HINTS.get(kind)
    head = f"{hint}\n\n" if hint else ""
    return PredictionError(f"{head}{summary}\n{tail}", kind=kind)


class Cancelled(RuntimeError):
    pass


# ------------------------------------------------------------------ spec
def normalize_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Validate a prediction spec and fill chain ids. Raises ValueError with a user-facing message."""
    comps_in = spec.get("components") or []
    if not isinstance(comps_in, list):
        # A dict here iterates as its keys — every component becomes a string and the
        # first .get() blows up with AttributeError, i.e. a 500 with no explanation.
        raise ValueError("構成要素はリストで指定してください")
    if not comps_in:
        raise ValueError("構成要素がありません。タンパク質・核酸・リガンドを 1 つ以上追加してください")
    if len(comps_in) > MAX_COMPONENTS:
        raise ValueError(f"構成要素が多すぎます (最大 {MAX_COMPONENTS} 個)")
    for i, comp in enumerate(comps_in):
        if not isinstance(comp, dict):
            raise ValueError(f"構成要素 {i + 1}: オブジェクトで指定してください")
    used: set[str] = set()
    for i, comp in enumerate(comps_in):
        raw_chains = comp.get("chains") or []
        if not isinstance(raw_chains, list):
            raise ValueError(f"構成要素 {i + 1}: チェーン ID はリストで指定してください")
        for cid in raw_chains:
            if not isinstance(cid, str) or not cid.strip():
                # A nested list here is unhashable and takes the whole request down.
                raise ValueError(f"構成要素 {i + 1}: チェーン ID は文字列で指定してください")
            if cid in used:
                raise ValueError(f"チェーン ID {cid} が重複しています")
            used.add(cid)
    free = (c for c in CHAIN_IDS if c not in used)
    comps: list[dict[str, Any]] = []
    total_tokens = 0
    for i, comp in enumerate(comps_in):
        ctype = comp.get("type")
        try:
            copies = int(comp.get("copies") or len(comp.get("chains") or []) or 1)
        except (TypeError, ValueError):
            raise ValueError(f"構成要素 {i + 1}: コピー数は数値で指定してください") from None
        if copies < 1 or copies > 12:
            raise ValueError(f"構成要素 {i + 1}: コピー数は 1〜12 です")
        chains = list(comp.get("chains") or [])
        while len(chains) < copies:
            chains.append(next(free))
        chains = chains[:copies]
        label = (comp.get("label") or "").strip()
        out: dict[str, Any] = {"type": ctype, "chains": chains, "label": label}
        if ctype in ("protein", "dna", "rna"):
            try:
                seq = clean_sequence(comp.get("sequence", ""), ctype)
            except SequenceError as exc:
                raise ValueError(f"構成要素 {i + 1} ({label or ctype}): {exc}") from exc
            out["sequence"] = seq
            total_tokens += len(seq) * copies
            if ctype == "protein":
                msa = comp.get("msa", "server")
                if msa not in ("server", "single"):
                    raise ValueError(f"構成要素 {i + 1}: msa は server か single です")
                out["msa"] = msa
            if comp.get("cyclic"):
                out["cyclic"] = True
        elif ctype == "ligand":
            smiles = (comp.get("smiles") or "").strip()
            ccd = (comp.get("ccd") or "").strip()
            if bool(smiles) == bool(ccd):
                raise ValueError(f"構成要素 {i + 1}: リガンドは SMILES か CCD コードのどちらか一方を指定してください")
            try:
                if smiles:
                    info = chem.describe_smiles(smiles)
                    out["smiles"] = smiles
                    total_tokens += info["heavy_atoms"] * copies
                else:
                    out["ccd"] = chem.validate_ccd(ccd)
                    total_tokens += 30 * copies
            except chem.ChemError as exc:
                raise ValueError(f"構成要素 {i + 1} ({label or 'ligand'}): {exc}") from exc
        else:
            raise ValueError(f"構成要素 {i + 1}: 未知の種類 {ctype}")
        for key in ("source", "mutations", "parent_sequence", "notes"):
            if key in comp:
                out[key] = comp[key]
        comps.append(out)

    binder = spec.get("affinity_binder") or None
    if binder:
        owner = next((c for c in comps if binder in c["chains"]), None)
        if owner is None or owner["type"] != "ligand":
            raise ValueError("親和性予測の対象は、リガンドのチェーン ID である必要があります")
        if len(owner["chains"]) > 1:
            raise ValueError("親和性予測の対象リガンドはコピー数 1 にしてください")
        if not any(c["type"] == "protein" for c in comps):
            raise ValueError("親和性予測にはタンパク質が必要です")

    constraints = []
    for con in spec.get("constraints") or []:
        if con.get("type") == "pocket":
            if con.get("binder") not in used | {cid for c in comps for cid in c["chains"]}:
                raise ValueError("ポケット拘束の binder が構成要素のチェーンにありません")
            contacts = [[str(c[0]), int(c[1])] for c in con.get("contacts") or []]
            if not contacts:
                raise ValueError("ポケット拘束には接触残基が 1 つ以上必要です")
            constraints.append({"type": "pocket", "binder": con["binder"], "contacts": contacts,
                                "max_distance": float(con.get("max_distance", 6.0)),
                                "force": bool(con.get("force", False))})
        elif con.get("type") == "contact":
            constraints.append({"type": "contact", "token1": [str(con["token1"][0]), int(con["token1"][1])],
                                "token2": [str(con["token2"][0]), int(con["token2"][1])],
                                "max_distance": float(con.get("max_distance", 6.0)),
                                "force": bool(con.get("force", False))})
        else:
            raise ValueError(f"未知の拘束: {con.get('type')}")

    s = get_settings()
    p = spec.get("params") or {}
    params = {
        "diffusion_samples": int(p.get("diffusion_samples", s.diffusion_samples)),
        "recycling_steps": int(p.get("recycling_steps", s.recycling_steps)),
        "sampling_steps": int(p.get("sampling_steps", s.sampling_steps)),
        "use_potentials": bool(p.get("use_potentials", s.use_potentials)),
        "seed": int(p["seed"]) if p.get("seed") not in (None, "") else None,
        "accelerator": str(p.get("accelerator") or "auto"),
    }
    if params["accelerator"] not in ("auto", "mps", "cpu"):
        raise ValueError("計算デバイスは auto / mps / cpu のいずれかです")
    if not 1 <= params["diffusion_samples"] <= 10:
        raise ValueError("サンプル数は 1〜10 です")
    if not 1 <= params["recycling_steps"] <= 10:
        raise ValueError("リサイクル回数は 1〜10 です")
    if not 10 <= params["sampling_steps"] <= 500:
        raise ValueError("サンプリングステップは 10〜500 です")
    return {
        "name": (spec.get("name") or "").strip() or "prediction",
        "components": comps,
        "affinity_binder": binder,
        "constraints": constraints,
        "params": params,
        "token_estimate": total_tokens,
    }


# ------------------------------------------------------------------ MSA cache
def _protein_entities(spec: dict[str, Any]) -> list[str]:
    return [c["sequence"] for c in spec["components"] if c["type"] == "protein" and c.get("msa") == "server"]


def _cache_index_path() -> Path:
    return msa_cache_dir() / "index.json"


def _load_cache_index() -> list[dict[str, Any]]:
    path = _cache_index_path()
    if not path.exists():
        return []
    return json.loads(path.read_text("utf-8"))


# How far a cached alignment may be reused, as a fraction of the sequence length.
#   STEP   — distance from the entry the index records, i.e. one hop.
#   ORIGIN — distance from the sequence the alignment was actually computed for.
# Measured on ubiquitin (76 aa), three runs each: at 6 substitutions from the origin a
# reused alignment scored +0.13 pLDDT against a freshly fetched one (pooled sigma 0.13,
# i.e. free), at 16 substitutions it scored -0.22 — the same size as a generation's gain
# in the search, so it would quietly eat the signal. 8% of 76 is 6, the distance measured
# safe. Without the ORIGIN bound, re-anchoring on every hop would let a lineage drift
# without limit; without re-anchoring at all, a search taking 2-3 substitutions per step
# leaves the window every couple of generations and pays ~30 s to refetch.
MSA_REUSE_STEP = 0.05
MSA_REUSE_ORIGIN = 0.08


def _substitutions(a: list[str], b: list[str]) -> int | None:
    """Total substitutions between two equal-shaped sequence sets, or None if incomparable."""
    if len(a) != len(b):
        return None
    total = 0
    for x, y in zip(a, b, strict=True):
        if len(x) != len(y):
            return None
        total += sum(1 for p, q in zip(x, y, strict=True) if p != q)
    return total


def _budget(seqs: list[str], fraction: float) -> int:
    return max(1, int(sum(len(s) for s in seqs) * fraction))


def find_reusable_msa(spec: dict[str, Any]) -> dict[str, Any] | None:
    """The nearest cached MSA usable for this spec, or None if every one is too far."""
    seqs = _protein_entities(spec)
    if not seqs:
        return None
    step_budget = _budget(seqs, MSA_REUSE_STEP)
    origin_budget = _budget(seqs, MSA_REUSE_ORIGIN)
    best = None
    for entry in _load_cache_index():
        step = _substitutions(entry["sequences"], seqs)
        if step is None or step > step_budget:
            continue
        origin = _substitutions(entry.get("origin") or entry["sequences"], seqs)
        if origin is None or origin > origin_budget:
            continue
        if not all(Path(msa_cache_dir() / f).exists() for f in entry["files"]):
            continue
        if best is None or origin < best[0]:
            best = (origin, entry)
    if best is None:
        return None
    return {**best[1], "differences": best[0]}


def _store_msa(job_dir: Path, spec: dict[str, Any], out_root: Path, job_id: str) -> None:
    seqs = _protein_entities(spec)
    if not seqs:
        return
    msa_dir = out_root / "msa"
    csvs = sorted(msa_dir.glob(f"{INPUT_NAME}_*.csv")) if msa_dir.exists() else []
    by_query: dict[str, Path] = {}
    for csv in csvs:
        lines = csv.read_text("utf-8").splitlines()
        if len(lines) >= 2:
            query = lines[1].split(",", 1)[1].replace("-", "").upper()
            by_query.setdefault(query, csv)
    files = []
    for seq in seqs:
        src = by_query.get(seq)
        if src is None:
            return  # MSA came from the cache or is incomplete; nothing new to store
        dest_name = f"{job_id}_{len(files)}.csv"
        shutil.copyfile(src, msa_cache_dir() / dest_name)
        files.append(dest_name)
    index = _load_cache_index()
    index.insert(0, {"job_id": job_id, "sequences": seqs, "origin": seqs, "files": files,
                     "created": time.time()})
    del index[200:]
    tmp = _cache_index_path().with_suffix(".tmp")
    tmp.write_text(json.dumps(index), "utf-8")
    tmp.replace(_cache_index_path())


def _reanchor_msa(spec: dict[str, Any], entry: dict[str, Any], job_id: str) -> None:
    """Record this run's sequences against the alignment it reused.

    Nothing is copied and no alignment changes — the index just gains a row, so the next
    variant measures its one-hop distance from here instead of from a distant ancestor.
    The origin travels with the row, so the total-drift bound still applies.
    """
    seqs = _protein_entities(spec)
    if not seqs or len(seqs) != len(entry["files"]):
        return
    index = _load_cache_index()
    if any(e["sequences"] == seqs for e in index):
        return
    index.insert(0, {"job_id": job_id, "sequences": seqs,
                     "origin": entry.get("origin") or entry["sequences"],
                     "files": list(entry["files"]), "created": time.time()})
    del index[200:]
    tmp = _cache_index_path().with_suffix(".tmp")
    tmp.write_text(json.dumps(index), "utf-8")
    tmp.replace(_cache_index_path())


def _write_reused_msa(entry: dict[str, Any], seqs: list[str], dest: Path) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, (fname, seq) in enumerate(zip(entry["files"], seqs, strict=True)):
        lines = (msa_cache_dir() / fname).read_text("utf-8").splitlines()
        header, first = lines[0], lines[1]
        key = first.split(",", 1)[0]
        lines[1] = f"{key},{seq}"
        path = dest / f"reused_{i}.csv"
        path.write_text("\n".join([header, *lines[1:]]), "utf-8")
        paths.append(path)
    return paths


# ------------------------------------------------------------------ YAML
def build_yaml(spec: dict[str, Any], msa_paths: list[Path] | None) -> str:
    sequences = []
    server_idx = 0
    for comp in spec["components"]:
        ids = comp["chains"] if len(comp["chains"]) > 1 else comp["chains"][0]
        ctype = comp["type"]
        if ctype == "protein":
            entry: dict[str, Any] = {"id": ids, "sequence": comp["sequence"]}
            if comp["msa"] == "single":
                entry["msa"] = "empty"
            elif msa_paths is not None:
                entry["msa"] = str(msa_paths[server_idx])
                server_idx += 1
            if comp.get("cyclic"):
                entry["cyclic"] = True
            sequences.append({"protein": entry})
        elif ctype in ("dna", "rna"):
            entry = {"id": ids, "sequence": comp["sequence"]}
            if comp.get("cyclic"):
                entry["cyclic"] = True
            sequences.append({ctype: entry})
        else:
            entry = {"id": ids}
            if comp.get("smiles"):
                entry["smiles"] = comp["smiles"]
            else:
                entry["ccd"] = comp["ccd"]
            sequences.append({"ligand": entry})
    doc: dict[str, Any] = {"version": 1, "sequences": sequences}
    cons = []
    for con in spec["constraints"]:
        if con["type"] == "pocket":
            cons.append({"pocket": {"binder": con["binder"], "contacts": con["contacts"],
                                    "max_distance": con["max_distance"], "force": con["force"]}})
        else:
            cons.append({"contact": {"token1": con["token1"], "token2": con["token2"],
                                     "max_distance": con["max_distance"], "force": con["force"]}})
    if cons:
        doc["constraints"] = cons
    if spec.get("affinity_binder"):
        doc["properties"] = [{"affinity": {"binder": spec["affinity_binder"]}}]
    return yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)


# ------------------------------------------------------------------ run
PHASES = [
    (re.compile(r"Downloading the "), "download", "モデル・辞書をダウンロード中"),
    (re.compile(r"Checking input data\.|Processing \d+ inputs"), "preprocess", "入力を前処理中"),
    (re.compile(r"Calling MSA server|Generating MSA"), "msa", "MSA を検索中 (ColabFold サーバー)"),
    (re.compile(r"Running structure prediction"), "structure", "構造を予測中"),
    (re.compile(r"Predicting property: affinity"), "affinity", "結合親和性を予測中"),
]
PHASE_ORDER = ["starting", "download", "preprocess", "msa", "structure", "affinity"]


def _accelerator(override: str = "auto") -> str:
    acc = override if override != "auto" else get_settings().accelerator
    if acc != "auto":
        return acc
    try:
        import torch

        return "mps" if torch.backends.mps.is_available() else "cpu"
    except Exception:  # torch import failure is reported by boltz itself
        return "cpu"


def run_env(accelerator: str) -> dict[str, str]:
    """Environment for the Boltz subprocess.

    Ops without an MPS kernel run on the CPU instead of aborting (PyTorch's documented
    switch). Measured 2026-09-14 with it turned off: a 76-residue chain (35 s) and a
    260-residue complex with ligands and affinity (243 s) both completed, so nothing falls
    back today. Strict mode keeps it that way loudly — a future op that needs the CPU fails
    the job instead of quietly making every run three times slower.
    """
    s = get_settings()
    env = os.environ.copy()
    strict = s.mps_strict and accelerator == "mps"
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "0" if strict else "1"
    env["PYTHONUNBUFFERED"] = "1"
    if accelerator == "mps":
        # PyTorch's MPS allocator refuses an allocation past recommended_max_memory x ratio,
        # default 1.7. Measured here: with iogpu.wired_limit_mb at its default that ceiling is
        # 17.76 x 1.7 = 30.2 GB, and a 1,696-residue three-chain complex hit it after 16
        # minutes and lost everything. Nothing was wrong with the machine — past physical
        # memory macOS swaps and the job keeps going, about 300x slower but alive. Raising
        # the ratio (0 = no ceiling) hands that decision back to the person, who is told what
        # it costs in SSD writes before submitting rather than discovering it as an OOM.
        # Only the high watermark is set; leaving the low one at its 1.4 default keeps the
        # allocator releasing cached blocks under pressure. Verified on this machine that
        # HIGH=0.0 against the default LOW starts and allocates without complaint.
        env["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = f"{max(0.0, s.mps_memory_ratio):.2f}"
    return env


def run_prediction(job: dict[str, Any], job_dir: Path, set_phase: Callable[[str, str], None],
                   cancelled: Callable[[], bool], log_line: Callable[[str], None],
                   report: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    spec = normalize_spec(job["spec"])
    boltz_bin = resolve_boltz_bin()
    if not boltz_bin:
        raise PredictionError("boltz コマンドが見つかりません。scripts/setup.sh を実行してください")
    s = get_settings()
    input_dir = job_dir / "input"
    out_dir = job_dir / "out"
    input_dir.mkdir(parents=True, exist_ok=True)

    reuse = None
    msa_paths = None
    if s.reuse_msa_for_variants and job["spec"].get("reuse_msa", True):
        reuse = find_reusable_msa(spec)
        if reuse is not None:
            msa_paths = _write_reused_msa(reuse, _protein_entities(spec), input_dir / "msa")
            log_line(f"[oritatami] MSA を再利用: job {reuse['job_id']} (置換 {reuse['differences']} 箇所)")

    yaml_path = input_dir / f"{INPUT_NAME}.yaml"
    yaml_path.write_text(build_yaml(spec, msa_paths), "utf-8")
    needs_server = any(c["type"] == "protein" and c["msa"] == "server" for c in spec["components"]) and msa_paths is None

    p = spec["params"]
    accelerator = _accelerator(p.get("accelerator", "auto"))
    cmd = [
        boltz_bin, "predict", str(yaml_path),
        "--out_dir", str(out_dir),
        "--cache", str(Path(s.boltz_cache).expanduser()),
        "--accelerator", accelerator,
        "--output_format", "mmcif",
        "--diffusion_samples", str(p["diffusion_samples"]),
        "--recycling_steps", str(p["recycling_steps"]),
        "--sampling_steps", str(p["sampling_steps"]),
        "--override",
        "--no_write_full_pde",
    ]
    if needs_server:
        cmd += ["--use_msa_server", "--msa_server_url", s.msa_server_url]
    if p["use_potentials"]:
        cmd.append("--use_potentials")
    if p["seed"] is not None:
        cmd += ["--seed", str(p["seed"])]

    env = run_env(accelerator)
    if env["PYTORCH_ENABLE_MPS_FALLBACK"] == "0":
        log_line("[oritatami] MPS 厳格モード: CPU への切り替えが起きたらジョブを失敗させます")
    log_line("[oritatami] " + " ".join(cmd))

    log_path = job_dir / "boltz.log"
    started = time.time()
    timings: dict[str, float] = {}
    tracker = _PhaseTracker(set_phase, timings, started)
    tracker.enter("starting", "Boltz を起動中")
    with log_path.open("w", encoding="utf-8") as log_file:
        # The supervise shim leads Boltz's process group and stops it if this app exits abruptly.
        supervised = [sys.executable, "-m", "oritatami.engines.supervise", str(os.getpid()), "--", *cmd]
        proc = subprocess.Popen(supervised, stdout=log_file, stderr=subprocess.STDOUT, cwd=job_dir,
                                env=env, start_new_session=True)
        seen_pos = 0
        live_seen = 0.0
        try:
            while proc.poll() is None:
                if cancelled():
                    _terminate(proc)
                    raise Cancelled()
                time.sleep(0.7)
                seen_pos = _scan_log(log_path, seen_pos, tracker)
                if report is not None:
                    live_seen = _report_live(job_dir, live_seen, report)
        except BaseException:
            if proc.poll() is None:
                _terminate(proc)
            raise
        finally:
            # Nothing reads it once the process is gone, and a stale one left in the job
            # directory would be exported with the results as if it meant something.
            for name in (LIVE_MEMORY_NAME, LIVE_MEMORY_NAME + ".tmp"):
                (job_dir / name).unlink(missing_ok=True)
    _scan_log(log_path, seen_pos, tracker)
    tracker.finish()
    elapsed = time.time() - started
    log_text = log_path.read_text("utf-8", errors="replace")
    if proc.returncode != 0:
        raise _failure(f"boltz が終了コード {proc.returncode} で終了しました", log_text, proc.returncode, tracker.phase)
    failed = re.findall(r"Number of failed examples: (\d+)", log_text)
    if ((failed and any(int(x) > 0 for x in failed)) or re.search(r"structure prediction\(s\) failed", log_text)
            or re.search(r"\d+ input\(s\) failed", log_text)):
        raise _failure("boltz が予測に失敗しました", log_text, proc.returncode, tracker.phase)

    root = out_dir / f"boltz_results_{INPUT_NAME}"
    pred_dir = root / "predictions" / INPUT_NAME
    if not pred_dir.exists() or not list(pred_dir.glob(f"{INPUT_NAME}_model_*.cif")):
        raise _failure("予測結果の構造ファイルが見つかりません", log_text, proc.returncode, tracker.phase)
    if needs_server:
        _store_msa(job_dir, spec, root, job["id"])
    elif reuse is not None:
        _reanchor_msa(spec, reuse, job["id"])
    set_phase("collect", "結果を集計中")
    result = collect_results(spec, job_dir, pred_dir)
    if s.cleanup_intermediate:
        freed = cleanup_intermediate(job_dir)
        if freed:
            log_line(f"[oritatami] 中間ファイルを削除しました ({freed / 1e6:.1f} MB)")
    result["elapsed_sec"] = round(elapsed, 1)
    result["timings"] = {k: round(v, 1) for k, v in timings.items()}
    result["token_estimate"] = spec["token_estimate"]
    result["accelerator"] = accelerator
    # Written by the supervisor while the run was alive; the only measurement of what a
    # prediction of this size actually costs in memory on this machine.
    peak_file = job_dir / PEAK_MEMORY_NAME
    if peak_file.exists():
        try:
            result["peak_memory_gb"] = round(float(peak_file.read_text("utf-8").strip()), 2)
            # Runs before 2026-09-14 recorded resident size, which on Metal is unrelated to
            # what the job actually held (6.5 MB reported against a 30 GB real footprint).
            # Those rows stay in the database but the estimator ignores anything unstamped.
            result["peak_memory_method"] = "footprint"
        except (OSError, ValueError):
            pass
    result["msa"] = {
        "server": needs_server,
        "reused_from": reuse["job_id"] if reuse else None,
        "single_sequence_chains": [cid for c in spec["components"] if c.get("msa") == "single" for cid in c["chains"]],
    }
    result["normalized_spec"] = spec
    return result


def _report_live(job_dir: Path, seen: float, report: Callable[[dict[str, Any]], None]) -> float:
    """Forward the supervisor's memory sample to the UI, if it wrote a newer one.

    Boltz publishes no progress inside the diffusion phase — its own bar reads 1/1 for the
    whole run — so on a long job the only honest signals are elapsed time against the
    estimate and whether the machine is still resident or has fallen into swap. Reading a
    small file every 0.7 s is cheaper than sampling the process group from here, and the
    supervisor is already doing that work for the peak.
    """
    path = job_dir / LIVE_MEMORY_NAME
    try:
        raw = path.read_text("utf-8")
    except OSError:            # not written yet, or being rewritten between our open and read
        return seen
    try:
        sample = json.loads(raw)
    except json.JSONDecodeError:   # caught a partial write; the next tick gets a whole one
        return seen
    ts = float(sample.get("ts") or 0.0)
    if ts <= seen:
        return seen
    report({"memory": sample})
    return ts


def _terminate(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        proc.wait(timeout=5)


class _PhaseTracker:
    """Forwards Boltz phases to the job and measures how long each one took (for runtime estimates)."""

    def __init__(self, set_phase: Callable[..., None], timings: dict[str, float], started: float) -> None:
        self.set_phase = set_phase
        self.timings = timings
        self.phase = "starting"
        self.label = ""
        self.last_detail = ""
        self.since = started

    def enter(self, phase: str, label: str) -> None:
        now = time.time()
        if self.label:
            self.timings[self.phase] = self.timings.get(self.phase, 0.0) + (now - self.since)
        self.phase, self.label, self.since, self.last_detail = phase, label, now, ""
        self.set_phase(phase, label)

    def detail(self, extra: str) -> None:
        if extra != self.last_detail:
            self.last_detail = extra
            self.set_phase(self.phase, f"{self.label} · {extra}")

    def finish(self) -> None:
        self.timings[self.phase] = self.timings.get(self.phase, 0.0) + (time.time() - self.since)
        self.label = ""


_MSA_STATUS_RE = re.compile(r"Reason: (RATELIMIT|PENDING|RUNNING|UNKNOWN)")
_MSA_STATUS_JA = {"RATELIMIT": "サーバー混雑のため待機中", "PENDING": "サーバーの順番待ち", "RUNNING": "サーバーで検索中",
                  "UNKNOWN": "サーバーの応答待ち"}


def _scan_log(path: Path, pos: int, tracker: _PhaseTracker) -> int:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            f.seek(pos)
            chunk = f.read()
            pos = f.tell()
    except FileNotFoundError:
        return pos
    msa_status = None
    for line in chunk.replace("\r", "\n").splitlines():
        for regex, key, label in PHASES:
            if regex.search(line) and PHASE_ORDER.index(key) > PHASE_ORDER.index(tracker.phase):
                tracker.enter(key, label)
        if tracker.phase == "msa":
            m = _MSA_STATUS_RE.search(line)
            if m:
                msa_status = m.group(1)
    if msa_status:
        tracker.detail(_MSA_STATUS_JA[msa_status])
    return pos


def cleanup_intermediate(job_dir: Path) -> int:
    """Delete files only Boltz itself needs (preprocessed tensors, raw MSA search output, lightning logs).

    Predicted structures, confidences, PAE/pLDDT arrays, affinity and the MSA CSVs are kept.
    Returns the number of bytes freed.
    """
    root = job_dir / "out" / f"boltz_results_{INPUT_NAME}"
    targets = [root / "processed", root / "lightning_logs", *(root / "msa").glob("*_tmp_env")] if root.exists() else []
    freed = 0
    for target in targets:
        if not target.exists():
            continue
        for f in target.rglob("*"):
            if f.is_file():
                try:
                    freed += f.stat().st_size
                except OSError:
                    pass
        shutil.rmtree(target, ignore_errors=True)
    return freed


# ------------------------------------------------------------------ results
def _chain_table(spec: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for comp in spec["components"]:
        for cid in comp["chains"]:
            rows.append({
                "chain": cid,
                "type": comp["type"],
                "label": comp.get("label") or "",
                "length": len(comp.get("sequence", "")) if comp["type"] != "ligand" else None,
                "sequence": comp.get("sequence"),
                "smiles": comp.get("smiles"),
                "ccd": comp.get("ccd"),
                "mutations": comp.get("mutations") or [],
            })
    return rows


def _downsample(matrix: np.ndarray, max_size: int = 256) -> tuple[list[list[float]], int]:
    n = matrix.shape[0]
    if n <= max_size:
        return np.round(matrix, 2).tolist(), 1
    factor = int(np.ceil(n / max_size))
    m = n // factor * factor
    trimmed = matrix[:m, :m]
    pooled = trimmed.reshape(m // factor, factor, m // factor, factor).mean(axis=(1, 3))
    return np.round(pooled, 2).tolist(), factor


def collect_results(spec: dict[str, Any], job_dir: Path, pred_dir: Path) -> dict[str, Any]:
    models = []
    cifs = sorted(pred_dir.glob(f"{INPUT_NAME}_model_*.cif"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    chain_rows = _chain_table(spec)
    index_to_chain = {str(i): row["chain"] for i, row in enumerate(chain_rows)}
    for cif in cifs:
        idx = int(cif.stem.rsplit("_", 1)[1])
        conf_path = pred_dir / f"confidence_{INPUT_NAME}_model_{idx}.json"
        conf = json.loads(conf_path.read_text("utf-8")) if conf_path.exists() else {}
        # Boltz keys chains by index in input order; translate to chain ids for the UI.
        for key in ("chains_ptm", "chains_pae"):
            if isinstance(conf.get(key), dict):
                conf[key] = {index_to_chain.get(k, k): v for k, v in conf[key].items()}
        for key in ("pair_chains_iptm", "pair_chains_pae"):
            if isinstance(conf.get(key), dict):
                conf[key] = {
                    index_to_chain.get(k, k): {index_to_chain.get(k2, k2): v2 for k2, v2 in row.items()}
                    for k, row in conf[key].items()
                }
        plddt = structure.residue_bfactors(cif)
        lig_plddt = structure.ligand_bfactors(cif)
        models.append({
            "index": idx,
            "file": str(cif.relative_to(job_dir)),
            "confidence": conf,
            "plddt": plddt,
            "ligand_plddt": lig_plddt,
        })
    affinity = None
    aff_files = sorted(pred_dir.glob("affinity_*.json"))
    if aff_files:
        raw = json.loads(aff_files[0].read_text("utf-8"))
        value = raw.get("affinity_pred_value")
        affinity = {
            **raw,
            "binder": spec.get("affinity_binder"),
            "ic50_um": round(float(10 ** value), 4) if value is not None else None,
            "delta_g_kcal": round(float(-1.363 * (6 - value)), 2) if value is not None else None,
        }
    pae = None
    pae_path = pred_dir / f"pae_{INPUT_NAME}_model_0.npz"
    if pae_path.exists():
        with np.load(pae_path) as data:
            key = "pae" if "pae" in data else list(data.keys())[0]
            matrix = np.asarray(data[key], dtype=np.float32)
        values, factor = _downsample(matrix)
        segments = structure.token_segments(pred_dir / f"{INPUT_NAME}_model_0.cif")
        if not segments or segments[-1]["end"] != matrix.shape[0]:
            segments = None  # tokenisation differs (e.g. modified residues); draw the matrix unlabeled
        pae = {"size": int(matrix.shape[0]), "factor": factor, "matrix": values, "segments": segments}
    interfaces = None
    if models and len(chain_rows) > 1:
        try:
            interfaces = structure.contacts(job_dir / models[0]["file"])
        except Exception as exc:  # contacts are an extra; report instead of failing the job
            interfaces = {"error": f"接触解析に失敗: {exc}"}
    return {
        "models": models,
        "chains": chain_rows,
        "affinity": affinity,
        "pae": pae,
        "interfaces": interfaces,
    }
