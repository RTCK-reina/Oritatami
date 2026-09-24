"""FastAPI application: REST API + the built frontend."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import logging
import logging.handlers
import re
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import (
    __version__,
    assistant,
    autopilot,
    chem,
    estimate,
    exporter,
    function_risk,
    governor,
    gpu,
    history,
    llm,
    regime,
    sources,
    ssd,
    structure,
    system,
)
from . import pdb_watcher as _pdb_watcher_mod
from .config import (
    app_home,
    exports_dir,
    frontend_dist,
    get_settings,
    imports_dir,
    jobs_dir,
    resolve_boltz_bin,
    update_settings,
)
from .db import Database
from .engines import boltz, esm
from .handlers import register_all
from .jobs import JobManager, job_file
from .seq import SequenceError, apply_mutations, clean_sequence, diff_substitutions, parse_mutations

log = logging.getLogger("oritatami")


def _configure_logging() -> None:
    root = logging.getLogger("oritatami")
    if root.handlers:
        return
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    fh = logging.handlers.RotatingFileHandler(app_home() / "oritatami.log", maxBytes=5_000_000, backupCount=3,
                                              encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)


class State:
    db: Database
    jobs: JobManager


state = State()


def _idle_unloader(stop: threading.Event) -> None:
    while not stop.wait(60):
        esm.unload_if_idle(600)


_torch_probe: dict[str, Any] = {"done": False, "torch": None, "mps": False}


def _probe_torch() -> None:
    """Importing torch takes seconds on a cold start; do it once in the background so /api/health stays fast."""
    try:
        import torch

        _torch_probe.update(torch=torch.__version__, mps=bool(torch.backends.mps.is_available()))
    except Exception as exc:  # reported through /api/health as torch=None
        log.warning("PyTorch を読み込めません: %s", exc)
    finally:
        _torch_probe["done"] = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    _configure_logging()
    state.db = Database()
    state.jobs = JobManager(state.db)
    register_all(state.jobs)
    state.jobs.start()
    autopilot.start(state.jobs, state.db)
    _pdb_watcher_mod.start(state.jobs, state.db)
    stop = threading.Event()
    threading.Thread(target=_idle_unloader, args=(stop,), daemon=True, name="esm-idle").start()
    threading.Thread(target=_probe_torch, daemon=True, name="torch-probe").start()
    log.info("Oritatami %s started, home=%s", __version__, app_home())
    yield
    stop.set()
    state.jobs.stop()


app = FastAPI(title="Oritatami", version=__version__, lifespan=lifespan)


@app.middleware("http")
async def _cache_headers(request: Request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path.startswith("/assets/"):
        # Vite file names carry a content hash, so the WebView may keep them forever.
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    elif not path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


@app.exception_handler(SequenceError)
@app.exception_handler(chem.ChemError)
@app.exception_handler(ValueError)
async def _bad_request(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(sources.SourceError)
async def _source_error(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.exception_handler(llm.SafeguardError)
async def _safeguard_error(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc), "code": "safeguard"})


@app.exception_handler(llm.LlmError)
async def _llm_error(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(KeyError)
async def _not_found(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": f"見つかりません: {exc}"})


@app.exception_handler(esm.EsmUnavailable)
@app.exception_handler(esm.EsmBusy)
async def _esm_missing(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(PermissionError)
async def _forbidden(_: Request, exc: PermissionError) -> JSONResponse:
    return JSONResponse(status_code=403, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    """Anything not named above.

    Without this the frontend showed the words "Internal Server Error" and nothing else, which
    says only that the app is broken — not which part, and not whether it is worth retrying.
    The log keeps the traceback; the toast gets the sentence a person can act on.
    """
    log.exception("未処理の例外: %s %s", request.method, request.url.path)
    detail = f"{type(exc).__name__}: {exc}".strip()
    return JSONResponse(status_code=500, content={
        "detail": f"想定外のエラーです ({detail[:300]})。詳しくは ~/Library/Logs/Oritatami/ のログを確認してください",
        "code": "internal",
    })


# ------------------------------------------------------------------ health / settings
@app.get("/api/ping")
def ping() -> dict[str, Any]:
    return {"ok": True, "version": __version__}


def _boltz_version() -> str | None:
    from importlib import metadata

    for dist in ("boltz-community", "boltz"):
        try:
            return metadata.version(dist)
        except metadata.PackageNotFoundError:
            continue
    return None


@app.get("/api/health")
def health() -> dict[str, Any]:
    s = get_settings()
    cache = Path(s.boltz_cache).expanduser()
    return {
        "version": __version__,
        "home": str(app_home()),
        "boltz": {
            "bin": resolve_boltz_bin() or None,
            "version": _boltz_version(),
            "weights": (cache / "boltz2_conf.ckpt").exists(),
            "affinity_weights": (cache / "boltz2_aff.ckpt").exists(),
            "ccd": (cache / "mols").exists(),
        },
        "torch": _torch_probe["torch"],
        "torch_probed": _torch_probe["done"],
        "mps": _torch_probe["mps"],
        "llm": llm.status(),
        "esm": {**esm.status(), "cached": esm.is_cached()},
        "machine": system.machine(),
        "disk_free_gb": system.disk_free_gb(),
        "exports_dir": str(Path.home() / "Downloads" / "Oritatami"),
    }


@app.get("/api/settings")
def read_settings() -> dict[str, Any]:
    return get_settings().public()


@app.patch("/api/settings")
def patch_settings(patch: dict[str, Any]) -> dict[str, Any]:
    return update_settings(patch).public()


# ------------------------------------------------------------------ LLM
class PullBody(BaseModel):
    model: str


@app.get("/api/llm/status")
def llm_status() -> dict[str, Any]:
    return llm.status()


@app.post("/api/llm/start")
def llm_start() -> dict[str, Any]:
    """Start Ollama, fetching it first if this machine has none.

    The button that calls this is the consent: a 150 MB download does not start on its own,
    and once it has, the caller watches ``/api/llm/status`` for the progress.
    """
    return llm.ensure_server(install=True)


@app.post("/api/llm/install")
def llm_install() -> dict[str, Any]:
    return llm.start_install()


@app.post("/api/llm/pull")
def llm_pull(body: PullBody) -> dict[str, Any]:
    if not llm.server_up():
        started = llm.ensure_server()
        if not started.get("running"):
            raise llm.LlmError(started.get("error") or "Ollama を起動できません")
    return llm.start_pull(body.model)


# ------------------------------------------------------------------ sequences / chemistry
class SequenceBody(BaseModel):
    sequence: str
    type: str = "protein"


@app.post("/api/sequence/validate")
def validate_sequence(body: SequenceBody) -> dict[str, Any]:
    seq = clean_sequence(body.sequence, body.type)
    return {"sequence": seq, "length": len(seq)}


class MutateBody(BaseModel):
    sequence: str
    mutations: str | list[str]


@app.post("/api/sequence/mutate")
def mutate(body: MutateBody) -> dict[str, Any]:
    seq = clean_sequence(body.sequence, "protein")
    muts = parse_mutations(body.mutations)
    new = apply_mutations(seq, muts)
    return {"sequence": new, "mutations": [m.code for m in muts]}


class DiffBody(BaseModel):
    a: str
    b: str


@app.post("/api/sequence/diff")
def diff(body: DiffBody) -> dict[str, Any]:
    return {"substitutions": diff_substitutions(body.a.upper(), body.b.upper())}


class SmilesBody(BaseModel):
    smiles: str


@app.post("/api/chem/describe")
def chem_describe(body: SmilesBody) -> dict[str, Any]:
    info = chem.describe_smiles(body.smiles)
    info["svg"] = chem.smiles_svg(body.smiles)
    return info


@app.get("/api/chem/pubchem")
def chem_pubchem(name: str) -> dict[str, Any]:
    found = sources.pubchem_lookup(name)
    found["describe"] = chem.describe_smiles(found["smiles"])
    found["svg"] = chem.smiles_svg(found["smiles"])
    _remember_search("pubchem", name, [{"id": str(found.get("cid") or name),
                                        "title": found.get("name") or name}])
    return found


@app.get("/api/chem/ccd/{code}")
def chem_ccd(code: str) -> dict[str, Any]:
    valid = chem.validate_ccd(code)
    info = sources.ccd_info(valid) or {"ccd": valid}
    if info.get("smiles"):
        try:
            info["svg"] = chem.smiles_svg(info["smiles"])
        except chem.ChemError as exc:
            info["svg_error"] = str(exc)
    return info


# ------------------------------------------------------------------ databases / imports
@app.get("/api/uniprot/search")
def uniprot_search(q: str) -> list[dict[str, Any]]:
    hits = sources.uniprot_search(q)
    _remember_search("uniprot", q, [{"id": h.get("accession"), "title": h.get("name")} for h in hits])
    return hits


@app.get("/api/uniprot/{accession}")
def uniprot_entry(accession: str) -> dict[str, Any]:
    entry = sources.uniprot_entry(accession)
    _note_pick("uniprot", accession, entry.get("name") or accession)
    return entry


@app.get("/api/pdb/search")
def pdb_search(q: str) -> list[dict[str, Any]]:
    hits = sources.pdb_search(q)
    _remember_search("pdb", q, [{"id": h.get("id"), "title": h.get("title")} for h in hits])
    return hits


def _remember_search(source: str, query: str, hits: list[dict[str, Any]]) -> None:
    """History is a convenience; a database hiccup must not turn a search into an error."""
    try:
        state.db.record_search(source=source, query=query, hits=len(hits),
                               top=[h for h in hits if h.get("id")])
    except Exception as exc:  # noqa: BLE001
        log.warning("検索履歴の記録に失敗しました: %s", exc)


@app.get("/api/searches")
def list_searches(limit: int = 100, source: str | None = None) -> list[dict[str, Any]]:
    return state.db.list_searches(limit=max(1, min(limit, 500)), source=source)


@app.delete("/api/searches")
def clear_searches() -> dict[str, int]:
    return {"deleted": state.db.clear_searches()}


# ------------------------------------------------------------------ system
class NotifyBody(BaseModel):
    title: str = Field("Oritatami", max_length=120)
    message: str = Field(..., max_length=400)


@app.post("/api/notify")
def notify(body: NotifyBody) -> dict[str, Any]:
    return {"shown": system.notify(body.title, body.message)}


@app.get("/api/storage")
def storage() -> dict[str, Any]:
    return system.storage()


class CleanupBody(BaseModel):
    intermediate: bool = True
    aligned_older_than_days: float | None = 0
    delete_failed_jobs: bool = False


@app.post("/api/storage/cleanup")
def storage_cleanup(body: CleanupBody) -> dict[str, Any]:
    active = set(state.db.job_ids(("queued", "running")))
    result = system.cleanup(intermediate=body.intermediate, aligned_older_than_days=body.aligned_older_than_days,
                            skip_jobs=active)
    deleted = 0
    if body.delete_failed_jobs:
        for job_id in state.db.job_ids(("failed", "cancelled")):
            state.jobs.delete(job_id)
            deleted += 1
    result["jobs_deleted"] = deleted
    result["storage"] = system.storage()
    return result


class SaveFileBody(BaseModel):
    filename: str = Field(..., max_length=160)
    data_url: str = Field(..., max_length=80_000_000)
    reveal: bool = True


@app.post("/api/files/save")
def save_file(body: SaveFileBody) -> dict[str, Any]:
    """Save a data: URL (e.g. a viewer screenshot) into the exports folder."""
    m = re.match(r"^data:([\w/+.-]+);base64,(.*)$", body.data_url, re.DOTALL)
    if not m:
        raise ValueError("data URL (base64) ではありません")
    try:
        data = base64.b64decode(m.group(2), validate=True)
    except ValueError as exc:
        raise ValueError(f"base64 を復号できません: {exc}") from exc
    path = exporter.save_to_exports(data, body.filename)
    if body.reveal:
        exporter.reveal(path)
    return {"path": str(path)}


def _import_response(path: Path, title: str, source: dict[str, Any], plddt: bool = False) -> dict[str, Any]:
    summary = structure.summarize(path)
    return {
        "name": path.name,
        "url": f"/api/files/imports/{path.name}",
        "format": "pdb" if path.suffix.lower() in (".pdb", ".ent") else "mmcif",
        "title": title or summary.get("title") or path.name,
        "source": source,
        "plddt_in_bfactor": plddt,
        "summary": summary,
    }


class IdBody(BaseModel):
    id: str
    assembly: bool = False


@app.post("/api/import/pdb")
def import_pdb(body: IdBody) -> dict[str, Any]:
    got = sources.fetch_pdb(body.id, assembly=body.assembly)
    out = _import_response(got["path"], got["title"], got["source"])
    out["note"] = got.get("note")
    out["bytes"] = got.get("bytes")
    _note_pick("pdb", body.id, out["title"])
    return out


def _note_pick(source: str, item_id: str, title: str) -> None:
    try:
        state.db.note_search_pick(source=source, item_id=item_id, title=title)
    except Exception as exc:  # noqa: BLE001
        log.warning("検索履歴の更新に失敗しました: %s", exc)


@app.post("/api/import/afdb")
def import_afdb(body: IdBody) -> dict[str, Any]:
    got = sources.fetch_afdb(body.id)
    out = _import_response(got["path"], got["title"], got["source"], plddt=True)
    _note_pick("afdb", body.id, out["title"])
    return out


class UploadBody(BaseModel):
    filename: str
    content: str = Field(max_length=60_000_000)


@app.post("/api/import/upload")
def import_upload(body: UploadBody) -> dict[str, Any]:
    path = sources.import_uploaded(body.filename, body.content.encode("utf-8"))
    try:
        return _import_response(path, body.filename, {"db": "file", "id": body.filename})
    except Exception:
        path.unlink(missing_ok=True)
        raise


@app.get("/api/files/imports/{name}")
def import_file(name: str) -> FileResponse:
    root = imports_dir().resolve()
    path = (root / name).resolve()
    if path.parent != root or not path.exists():
        raise HTTPException(404, "ファイルがありません")
    return FileResponse(path, media_type="text/plain")


# ------------------------------------------------------------------ jobs
class PredictBody(BaseModel):
    spec: dict[str, Any]
    title: str | None = None
    parent_id: str | None = None
    origin: str = "user"


def _require_boltz() -> None:
    if not resolve_boltz_bin():
        raise ValueError("Boltz-2 がインストールされていません。ターミナルで scripts/setup.sh を実行してください")


def _queue_predict(raw_spec: dict[str, Any], title: str | None, parent_id: str | None, origin: str) -> dict[str, Any]:
    normalized = boltz.normalize_spec(raw_spec)  # reject bad input before queueing
    spec = dict(raw_spec)
    spec["components"] = normalized["components"]
    name = (title or normalized["name"]).strip() or "prediction"
    return state.jobs.submit("predict", spec, name, parent_id=parent_id, origin=origin)


@app.post("/api/jobs/predict")
def submit_predict(body: PredictBody) -> dict[str, Any]:
    _require_boltz()
    return _queue_predict(body.spec, body.title, body.parent_id, body.origin)


class Variant(BaseModel):
    chain: str
    mutations: list[str] = Field(..., min_length=1, max_length=20)


class BatchBody(BaseModel):
    spec: dict[str, Any]
    variants: list[Variant] = Field(..., min_length=1, max_length=20)
    parent_id: str | None = None
    origin: str = "user"


def _variant_spec(spec: dict[str, Any], variant: Variant) -> tuple[dict[str, Any], list[str]]:
    out = copy.deepcopy(spec)
    comp = next((c for c in out.get("components") or []
                 if c.get("type") == "protein" and variant.chain in (c.get("chains") or [])), None)
    if comp is None:
        raise ValueError(f"チェーン {variant.chain} のタンパク質がありません")
    muts = parse_mutations(variant.mutations)
    current = clean_sequence(comp.get("sequence", ""), "protein")
    mutated = apply_mutations(current, muts)
    base = comp.get("parent_sequence") or current
    comp["sequence"] = mutated
    comp["parent_sequence"] = base
    comp["mutations"] = diff_substitutions(base, mutated) or []
    return out, [m.code.split(":")[-1] for m in muts]


@app.post("/api/jobs/predict/batch")
def submit_batch(body: BatchBody) -> dict[str, Any]:
    """Queue one prediction per variant. Everything is validated first, so either all jobs are queued or none."""
    _require_boltz()
    base_name = (body.spec.get("workbench_name") or body.spec.get("name") or "variant").split(" + ")[0]
    prepared = []
    for v in body.variants:
        vspec, codes = _variant_spec(body.spec, v)
        boltz.normalize_spec(vspec)
        vspec["name"] = vspec["workbench_name"] = f"{base_name} + {'/'.join(codes)}"
        prepared.append(vspec)
    jobs = [_queue_predict(vs, vs["name"], body.parent_id, body.origin) for vs in prepared]
    return {"jobs": jobs}


class EstimateBody(BaseModel):
    spec: dict[str, Any]


@app.post("/api/estimate")
def estimate_runtime(body: EstimateBody) -> dict[str, Any]:
    normalized = boltz.normalize_spec(body.spec)
    server = any(c["type"] == "protein" and c.get("msa") == "server" for c in normalized["components"])
    reusable = server and get_settings().reuse_msa_for_variants and boltz.find_reusable_msa(normalized) is not None
    queued_ahead = sum(1 for j in state.db.list_jobs() if j["kind"] == "predict" and j["status"] in ("queued", "running"))
    result = estimate.estimate(normalized, server and not reusable, state.db.finished_predictions())
    result["msa_reuse"] = bool(reusable)
    result["queued_ahead"] = queued_ahead
    return result


@app.get("/api/jobs/eta")
def queue_eta() -> dict[str, Any]:
    """When the queue as a whole is expected to be empty.

    Per-job estimates existed only on the workbench, before submitting. Once a run is going the
    question is a different one — "can I leave" — and that needs the running job's remaining
    time plus everything behind it, as a clock time rather than a duration.
    """
    history = state.db.finished_predictions()
    now = time.time()
    total = 0.0
    counted, unknown = 0, 0
    parts: list[dict[str, Any]] = []
    for job in sorted((j for j in state.db.list_jobs(limit=2000)
                       if j["status"] in ("queued", "running")),
                      key=lambda j: j["created_at"]):
        seconds: float | None = None
        baseline: float | None = None
        detail: dict[str, Any] = {}
        normalized = None
        if job["kind"] == "predict":
            try:
                normalized = boltz.normalize_spec(job["spec"])
                server = any(c["type"] == "protein" and c.get("msa") == "server"
                             for c in normalized["components"])
                reusable = (server and get_settings().reuse_msa_for_variants
                            and boltz.find_reusable_msa(normalized) is not None)
                est = estimate.estimate(normalized, server and not reusable, history)
                seconds = float(est["seconds"])
                # ``remaining`` adds the paging the live footprint implies on top of its
                # baseline, so the predicted paging must not be inside the baseline too.
                baseline = seconds - float((est.get("breakdown") or {}).get("paging") or 0.0)
            except Exception:  # a spec we cannot normalise still occupies the queue
                seconds = None
        else:
            # ESM lanes are short and run beside the predictions, so they do not extend the wait
            seconds = 0.0
        if job["status"] == "running" and normalized is not None:
            # A running job is not a baseline any more: it has been measured. The remaining
            # time comes from what it has actually computed against what this size costs,
            # which is the only thing that stays true once the machine starts swapping.
            live = state.jobs.live(job["id"]) or {}
            started = live.get("started_at") or job.get("started_at") or now
            detail = estimate.remaining(normalized, now - started, baseline or 0.0, live, history)
            seconds = detail["seconds"]
        if seconds is None:
            unknown += 1
        else:
            total += seconds
            counted += 1
        parts.append({"id": job["id"], "title": job["title"], "status": job["status"],
                      "kind": job["kind"], "seconds": None if seconds is None else round(seconds),
                      **{k: detail[k] for k in ("basis", "progress", "efficiency", "overrun",
                                               "note", "regime", "user_share", "overage_gb",
                                               "swap_seconds")
                         if k in detail}})
    return {
        "jobs": len(parts),
        "counted": counted,
        "unknown": unknown,
        "seconds": round(total),
        # With an untimeable job in the queue the clock time is a lower bound, not a forecast.
        # Saying which it is costs one field and stops the number from being read as a promise.
        "complete": unknown == 0,
        "finish_at": now + total if parts and unknown == 0 else None,
        "at_least_until": now + total if parts and unknown else None,
        "now": now,
        "items": parts[:50],
    }


@app.get("/api/jobs/{job_id}/function_risk")
def job_function_risk(job_id: str) -> dict[str, Any]:
    """Whether this result looks like a working molecule or a broken one."""
    job = state.db.get_job(job_id)
    if job is None:
        raise KeyError(job_id)
    # Results predicted before the geometry check existed have no geometry block; measure it
    # now so old work gets the same warnings as new work.
    job = function_risk.ensure_geometry(state.db, job)
    return function_risk.assess(state.db, job)


@app.get("/api/jobs/{job_id}/protected_suggest")
def job_protected_suggest(job_id: str, chain: str | None = None) -> dict[str, Any]:
    """Residues worth putting off limits for this molecule, each with its reason."""
    job = state.db.get_job(job_id)
    if job is None:
        raise KeyError(job_id)
    if not chain:
        chains = [c.get("chain") for c in ((job.get("result") or {}).get("chains") or [])
                  if c.get("chain")]
        if not chains:
            comps = (job.get("spec") or {}).get("components") or []
            chains = [ch for c in comps for ch in (c.get("chains") or [])]
        chain = chains[0] if chains else "A"
    return function_risk.suggest_protected(state.db, job, chain)


class ScanBody(BaseModel):
    sequence: str
    chain: str | None = None
    label: str | None = None
    parent_id: str | None = None


@app.post("/api/jobs/scan")
def submit_scan(body: ScanBody) -> dict[str, Any]:
    seq = clean_sequence(body.sequence, "protein")
    title = f"変異スキャン {body.label or ''} {body.chain or ''}".strip()
    return state.jobs.submit("scan", {"sequence": seq, "chain": body.chain, "label": body.label}, title,
                             parent_id=body.parent_id)


class RefineBody(BaseModel):
    sequence: str
    label: str | None = None
    rounds: int = Field(6, ge=1, le=30)
    fraction: float = Field(0.1, gt=0, le=0.5)
    temperature: float = Field(1.0, gt=0, le=3.0)
    fixed_positions: list[int] = []
    seed: int | None = None
    origin: str = "user"


@app.post("/api/jobs/refine")
def submit_refine(body: RefineBody) -> dict[str, Any]:
    seq = clean_sequence(body.sequence, "protein")
    spec = body.model_dump()
    spec["sequence"] = seq
    return state.jobs.submit("refine", spec, f"ESM で配列を磨く {body.label or ''}".strip(), origin=body.origin)


def _summarize(j: dict[str, Any]) -> dict[str, Any]:
    """Job list rows: enough for the list, badges and variant tables, without sequences or matrices."""
    res = j.get("result")
    if j["kind"] == "predict":
        spec = j.get("spec") or {}
        j["spec"] = {
            "name": spec.get("name"),
            "workbench_name": spec.get("workbench_name"),
            "components": [{"type": c.get("type"), "label": c.get("label"), "chains": c.get("chains"),
                            "mutations": c.get("mutations") or []} for c in spec.get("components") or []],
            "affinity_binder": spec.get("affinity_binder"),
            # the job list badges the loop's own jobs with these
            "autopilot_experiment": spec.get("autopilot_experiment"),
            "autopilot_depth": spec.get("autopilot_depth"),
        }
        if res:
            m0 = res["models"][0] if res.get("models") else {}
            conf = dict(m0.get("confidence", {}))
            for key in ("pair_chains_iptm", "pair_chains_pae", "chains_pae"):
                conf.pop(key, None)
            plddt = [v for vals in (m0.get("plddt") or {}).values() for v in vals]
            j["result"] = {"confidence": conf, "affinity": res.get("affinity"), "elapsed_sec": res.get("elapsed_sec"),
                           "model_count": len(res.get("models", [])),
                           "mean_plddt": round(sum(plddt) / len(plddt), 2) if plddt else None}
    elif res and j["kind"] == "scan":
        j["result"] = {"pseudo_perplexity": res.get("pseudo_perplexity"), "chain": res.get("chain")}
    elif res and j["kind"] == "refine":
        j["result"] = {"pseudo_perplexity": res.get("pseudo_perplexity"),
                       "start_pseudo_perplexity": res.get("start_pseudo_perplexity")}
    return j


def _metrics(job: dict[str, Any]) -> dict[str, Any]:
    """Scalar scores of a finished prediction, flattened for ranking."""
    res = job.get("result") or {}
    models = res.get("models") or []
    m0 = models[0] if models else {}
    conf = m0.get("confidence") or {}
    plddt = [v for vals in (m0.get("plddt") or {}).values() for v in vals]
    aff = res.get("affinity") or {}

    def _num(v: Any) -> float | None:
        try:
            return round(float(v), 4) if v is not None else None
        except (TypeError, ValueError):
            return None

    return {
        "mean_plddt": round(sum(plddt) / len(plddt), 2) if plddt else None,
        "core_plddt": (round(autopilot.core_plddt(res), 2)
                       if autopilot.core_plddt(res) is not None else None),
        "confidence_score": _num(conf.get("confidence_score")),
        "ptm": _num(conf.get("ptm")),
        "iptm": _num(conf.get("iptm")),
        "complex_plddt": _num(conf.get("complex_plddt")),
        "affinity_ic50_um": _num(aff.get("ic50_um")),
        "affinity_delta_g": _num(aff.get("delta_g_kcal")),
    }


LEADERBOARD_METRICS = ("mean_plddt", "core_plddt", "confidence_score", "iptm", "ptm",
                       "complex_plddt")


def pareto_front(rows: list[dict[str, Any]], metric: str,
                 parent_of: dict[str, str | None]) -> tuple[set[str], dict[str, str]]:
    """Rows nothing else beats on both score and simplicity, within each lineage.

    Per lineage, not across the whole table: a carbonic anhydrase complex at 98 with no
    mutations dominates every ubiquitin variant ever made, which says nothing about
    either of them.
    """
    roots: dict[str, str] = {}

    def root_of(job_id: str) -> str:
        seen: set[str] = set()
        cur = job_id
        while parent_of.get(cur) and cur not in seen:
            seen.add(cur)
            cur = parent_of[cur]          # type: ignore[assignment]
        return cur

    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        roots[r["id"]] = root_of(r["id"])
        if r.get(metric) is not None:
            groups.setdefault(roots[r["id"]], []).append(r)

    front: set[str] = set()
    for members in groups.values():
        # A lineage of one is trivially its own front and says nothing about a trade-off.
        if len(members) < 2:
            continue
        for r in members:
            n, v = len(r["mutations"]), r[metric]
            if not any((len(o["mutations"]) <= n and o[metric] > v)
                       or (len(o["mutations"]) < n and o[metric] >= v)
                       for o in members if o is not r):
                front.add(r["id"])
    return front, roots


@app.get("/api/leaderboard")
def leaderboard(metric: str = "mean_plddt", limit: int = 200) -> dict[str, Any]:
    """Every successful prediction, scored and ranked, with the delta against its parent.

    Once the autopilot queues work on its own the bottleneck stops being "produce
    results" and becomes "find the ones that matter". This is that view: one row per
    prediction, sorted by the chosen metric, and — where the job came from a parent —
    how much better or worse it is than what it was derived from. The frontend also
    builds the lineage tree from ``parent_id`` in these same rows.
    """
    if metric not in LEADERBOARD_METRICS:
        raise HTTPException(400, f"指標は次のいずれかです: {', '.join(LEADERBOARD_METRICS)}")

    jobs_list = state.db.succeeded_predictions(limit=None)
    by_id = {j["id"]: j for j in jobs_list}
    scores = {j["id"]: _metrics(j) for j in jobs_list}

    rows: list[dict[str, Any]] = []
    for j in jobs_list:
        spec = j.get("spec") or {}
        mine = scores[j["id"]]
        parent_id = j.get("parent_id")
        parent = by_id.get(parent_id) if parent_id else None
        parent_scores = scores.get(parent_id) if parent_id else None

        deltas: dict[str, Any] = {}
        for key in LEADERBOARD_METRICS:
            a, b = mine.get(key), (parent_scores or {}).get(key)
            deltas[f"delta_{key}"] = round(a - b, 3) if (a is not None and b is not None) else None

        mutations: list[str] = []
        for c in spec.get("components") or []:
            mutations.extend(c.get("mutations") or [])

        rows.append({
            "id": j["id"],
            "title": j.get("title"),
            "origin": j.get("origin") or "user",
            "created_at": j.get("created_at"),
            "finished_at": j.get("finished_at"),
            "starred": bool(j.get("starred")),
            "parent_id": parent_id,
            "parent_title": parent.get("title") if parent else None,
            "autopilot_depth": int(spec.get("autopilot_depth", 0) or 0),
            "mutations": mutations,
            "elapsed_sec": (j.get("result") or {}).get("elapsed_sec"),
            "regime": regime.of(j.get("result")).short(),
            "regime_label": regime.of(j.get("result")).label(),
            # A delta against a parent computed under other conditions measures the
            # settings change, not the mutation. Marked so the table can grey it out.
            "comparable_to_parent": (
                None if parent is None
                else regime.comparable(regime.of(j.get("result")), regime.of(parent.get("result")))),
            **mine,
            **deltas,
        })

    # Rows missing the chosen metric sink to the bottom rather than sorting as zero.
    rows.sort(key=lambda r: (r.get(metric) is not None, r.get(metric) or 0.0), reverse=True)
    for i, r in enumerate(rows, 1):
        r["rank"] = i

    # The single column hides the trade-off it is made of: "+0.9 with sixteen mutations"
    # and "+0.7 with two" sort next to each other and are not the same result. The front
    # is the set where nothing else is both better-scoring and simpler.
    front, roots = pareto_front(rows, metric, {k: v.get("parent_id") for k, v in by_id.items()})
    pareto = sorted(front)
    for r in rows:
        r["pareto"] = r["id"] in front
        r["lineage_root"] = roots.get(r["id"])

    best = rows[0] if rows else None
    improved = [r for r in rows if (r.get(f"delta_{metric}") or 0) > 0]
    # Rank over everything, then cut. Cutting first would rank only the newest rows
    # against each other and call the best of those the leader.
    total, improved_total = len(rows), len(improved)
    # Deltas across a settings change are not improvements; counting them inflates the
    # only number anyone reads to judge whether the search is working.
    improved_total = len([r for r in improved if r.get("comparable_to_parent") is not False])
    rows = rows[:max(1, min(limit, 1000))]
    return {
        "metric": metric,
        "metrics": list(LEADERBOARD_METRICS),
        "count": total,
        "returned": len(rows),
        "improved_count": improved_total,
        "best_id": best["id"] if best else None,
        "conditions": regime.summarize([j.get("result") for j in jobs_list]),
        "pareto_count": len(pareto),
        "rows": rows,
    }


@app.get("/api/history")
def search_history() -> dict[str, Any]:
    """What the run has already learned: repeat noise, per-position outcomes, waste.

    Read-only over finished jobs. The heavy field (``known``, one entry per distinct
    sequence) is for the loop's own dedupe and is not sent to the UI.
    """
    data = history.get(state.db, lambda r: autopilot.metric_value(r, autopilot.resolve_metric(r)))
    out = {k: v for k, v in data.items() if k != "known"}
    out["distinct_experiments"] = len(data.get("known") or {})
    out["suggested_delta"] = history.suggested_delta(data)
    out["configured_delta"] = float(get_settings().autopilot_improvement_delta)
    positions = data.get("by_position") or {}
    out["worst_positions"] = sorted(
        ({"position": p, **row} for p, row in positions.items() if row["n"] >= 5),
        key=lambda r: r["mean"])[:12]
    out["by_position"] = positions
    subs = data.get("by_substitution") or {}
    out["repeated_substitutions"] = sorted(
        ({"code": k, **v} for k, v in subs.items() if v["n"] >= 3),
        key=lambda r: -r["n"])[:20]
    out.pop("by_substitution", None)
    # A strategy comparison needs hundreds of predictions before it means anything; say
    # how many, rather than letting a 20-row table look like an answer.
    rate = data.get("improved_rate") or 0.0
    out["strategy_power"] = {
        "improved_rate": rate,
        "per_arm_for_2x": history.required_samples(rate, 2.0),
        "per_arm_for_1_5x": history.required_samples(rate, 1.5),
    }
    return out


@app.get("/api/autopilot/status")
def autopilot_status() -> dict[str, Any]:
    """Budget, disk headroom and whether autonomous submissions are currently accepted."""
    return governor.status(state.db)


@app.get("/api/jobs")
def list_jobs(summary: bool = True, limit: int = 200) -> list[dict[str, Any]]:
    """Newest first. The default window is small on purpose; the UI asks for more."""
    jobs = state.jobs.list(limit=max(1, min(limit, 5000)))
    return [_summarize(j) for j in jobs] if summary else jobs


@app.get("/api/jobs/changes")
def job_changes(rev: int = 0, timeout: float = 25.0) -> dict[str, Any]:
    """Long poll: returns as soon as any job changes (or after `timeout` seconds) with the new revision."""
    current = state.jobs.wait_for_change(rev, min(max(timeout, 0.0), 55.0))
    return {"rev": current, "changed": current != rev}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    return state.jobs.describe(state.db.get_job(job_id))


class JobPatch(BaseModel):
    title: str | None = None
    starred: bool | None = None


@app.patch("/api/jobs/{job_id}")
def patch_job(job_id: str, body: JobPatch) -> dict[str, Any]:
    if state.db.get_job(job_id) is None:
        raise KeyError(job_id)
    values = {k: v for k, v in body.model_dump().items() if v is not None}
    if "starred" in values:
        values["starred"] = int(values["starred"])
    state.db.update_job(job_id, **values)
    state.jobs.touch()
    return state.jobs.describe(state.db.get_job(job_id))


class RetryBody(BaseModel):
    msa: str | None = Field(None, pattern="^(single)$")
    accelerator: str | None = Field(None, pattern="^(auto|mps|cpu)$")
    diffusion_samples: int | None = Field(None, ge=1, le=10)
    new_seed: bool = False


@app.post("/api/jobs/{job_id}/retry")
def retry_job(job_id: str, body: RetryBody) -> dict[str, Any]:
    """Queue the same input again, optionally with the changes a failure hint suggests."""
    job = state.db.get_job(job_id)
    if job is None:
        raise KeyError(job_id)
    spec = copy.deepcopy(job["spec"])
    notes = []
    if job["kind"] == "predict":
        _require_boltz()
        params = dict(spec.get("params") or {})
        if body.msa == "single":
            for comp in spec.get("components") or []:
                if comp.get("type") == "protein":
                    comp["msa"] = "single"
            notes.append("MSA なし")
        if body.accelerator:
            params["accelerator"] = body.accelerator
            notes.append(body.accelerator.upper())
        if body.diffusion_samples:
            params["diffusion_samples"] = body.diffusion_samples
            notes.append(f"サンプル {body.diffusion_samples}")
        if body.new_seed:
            params["seed"] = None
        spec["params"] = params
        title = re.sub(r" \(再実行[^)]*\)$", "", job["title"]) + f" (再実行{'・' + '・'.join(notes) if notes else ''})"
        return _queue_predict(spec, title, job["parent_id"], job["origin"])
    title = re.sub(r" \(再実行[^)]*\)$", "", job["title"]) + " (再実行)"
    return state.jobs.submit(job["kind"], spec, title, parent_id=job["parent_id"], origin=job["origin"])


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict[str, Any]:
    return state.jobs.cancel(job_id)


class ReorderBody(BaseModel):
    action: str = Field(..., pattern="^(top|up|down|bottom)$")


@app.post("/api/jobs/{job_id}/reorder")
def reorder_job(job_id: str, body: ReorderBody) -> dict[str, Any]:
    """Move a waiting job within its lane. Only the order changes; nothing is re-run."""
    return state.jobs.reorder(job_id, body.action)


class CancelQueuedBody(BaseModel):
    kind: str | None = Field(None, pattern="^(predict|scan|refine)$")
    # Off by default: an old client asking to clear the queue must not also kill the job that
    # has been running for forty minutes.
    include_running: bool = False


@app.post("/api/jobs/queue/cancel_all")
def cancel_queued_jobs(body: CancelQueuedBody) -> dict[str, Any]:
    ids = state.jobs.cancel_batch(body.kind, include_running=body.include_running)
    return {"cancelled": ids, "count": len(ids)}


@app.get("/api/system/gpu")
def gpu_state(fresh: bool = False) -> dict[str, Any]:
    """What Metal will let one process hold, and the sysctl behind it."""
    return gpu.state(fresh=fresh)


class GpuLimitBody(BaseModel):
    wired_limit_mb: int = Field(..., ge=0, le=1_000_000)


@app.get("/api/system/memory")
def system_memory() -> dict[str, Any]:
    """The app's own footprint across this session's jobs, and what torch is holding.

    Boltz takes its memory with it when its subprocess exits; what can grow unnoticed over a
    long batch is this process. Reported rather than acted on: the numbers say whether the
    growth is real before anything is done about it.
    """
    trend = system.memory_trend()
    torch_mod = sys.modules.get("torch")
    mps: dict[str, Any] | None = None
    if torch_mod is not None:
        try:
            if torch_mod.backends.mps.is_available():
                mps = {
                    "driver_allocated_gb": round(torch_mod.mps.driver_allocated_memory() / 1024**3, 2),
                    "in_use_gb": round(torch_mod.mps.current_allocated_memory() / 1024**3, 2),
                }
        except (AttributeError, RuntimeError):
            mps = None
    return {**trend, "torch_mps": mps, "growth_warn_gb": system.FOOTPRINT_GROWTH_WARN_GB}


@app.post("/api/system/memory/release")
def system_memory_release() -> dict[str, Any]:
    """Give back what is held but unused: torch's MPS pool, then the ESM-2 model if it is idle."""
    from .engines import esm

    freed = esm.release_cache()
    unloaded = esm.unload_if_unused()
    if unloaded:
        freed += esm.release_cache()
    llm.unload_model()
    return {"freed_gb": round(freed, 2), "esm_unloaded": unloaded,
            "footprint_gb": system.process_footprint_gb()}


@app.get("/api/system/ssd")
def ssd_wear(fresh: bool = False) -> dict[str, Any]:
    """SSD wear and what a swapping run costs it. ``available`` is False without smartmontools."""
    forecast = ssd.swap_write_forecast(fresh=fresh)
    return {**forecast, "available": forecast["wear"] is not None,
            "smartctl": ssd.smartctl_bin(), "applecare": get_settings().applecare}


@app.post("/api/system/gpu")
def gpu_set_limit(body: GpuLimitBody) -> dict[str, Any]:
    """Change iogpu.wired_limit_mb. macOS asks the user to authenticate; the app never sees it."""
    try:
        return gpu.apply(body.wired_limit_mb)
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict[str, Any]:
    state.jobs.delete(job_id)
    return {"deleted": job_id}


@app.get("/api/jobs/{job_id}/log", response_class=PlainTextResponse)
def job_log(job_id: str) -> str:
    # An unknown id used to return 200 with an empty body, so a mistyped or deleted job
    # showed an empty log pane instead of saying the job is gone.
    if state.db.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません")
    return state.jobs.log_tail(job_id)


@app.get("/api/jobs/{job_id}/files/{rel:path}")
def job_files(job_id: str, rel: str) -> FileResponse:
    path = job_file(job_id, rel)
    if not path.is_file():
        raise HTTPException(404, "ファイルがありません")
    return FileResponse(path, media_type="text/plain", filename=path.name if rel.endswith(".cif") else None)


def _finished_job(job_id: str) -> dict[str, Any]:
    job = state.db.get_job(job_id)
    if job is None:
        raise KeyError(job_id)
    if job["status"] in ("queued", "running"):
        raise ValueError("実行中のジョブは書き出せません。完了を待ってください")
    return job


@app.get("/api/jobs/{job_id}/export.zip")
def export_zip(job_id: str) -> Response:
    data, name = exporter.build_zip(_finished_job(job_id))
    return Response(data, media_type="application/zip",
                    headers={"Content-Disposition": f"attachment; filename=\"export.zip\"; filename*=UTF-8''{quote(name)}"})


@app.post("/api/jobs/{job_id}/export")
def export_to_folder(job_id: str) -> dict[str, Any]:
    """Write the zip into ~/Downloads/Oritatami and select it in Finder (reliable inside the native window)."""
    data, name = exporter.build_zip(_finished_job(job_id))
    path = exporter.save_to_exports(data, name)
    exporter.reveal(path)
    return {"path": str(path), "bytes": len(data)}


@app.get("/api/jobs/{job_id}/structure.pdb")
def structure_pdb(job_id: str, model: int = 0) -> Response:
    job = _finished_job(job_id)
    models = (job.get("result") or {}).get("models") or []
    match = next((m for m in models if m["index"] == model), None)
    if match is None:
        raise KeyError(f"model {model}")
    text = structure.to_pdb(job_file(job_id, match["file"]))
    name = f"{exporter.safe_name(job['title'])}_model{model}.pdb"

    # A non-text media type so WKWebView offers a save panel instead of rendering the file.
    return Response(text, media_type="chemical/x-pdb",
                    headers={"Content-Disposition": f"attachment; filename=\"model.pdb\"; filename*=UTF-8''{quote(name)}"})


@app.post("/api/jobs/{job_id}/reveal")
def reveal_job(job_id: str) -> dict[str, Any]:
    """Show the job directory in Finder (downloads are unreliable inside WKWebView)."""
    import subprocess

    path = jobs_dir() / job_id
    if not path.exists():
        raise KeyError(job_id)
    subprocess.run(["open", str(path)], check=True)
    return {"opened": str(path)}


# ------------------------------------------------------------------ comparison
class StructureRef(BaseModel):
    job_id: str | None = None
    model: int = 0
    import_name: str | None = None


def _ref_path(ref: StructureRef) -> Path:
    if ref.job_id:
        job = state.db.get_job(ref.job_id)
        if job is None or not job.get("result"):
            raise ValueError(f"ジョブ {ref.job_id} に結果がありません")
        models = job["result"].get("models") or []
        match = next((m for m in models if m["index"] == ref.model), None)
        if match is None:
            raise ValueError(f"モデル {ref.model} がありません")
        return job_file(ref.job_id, match["file"])
    if ref.import_name:
        path = (imports_dir() / ref.import_name).resolve()
        if path.parent != imports_dir().resolve() or not path.exists():
            raise ValueError("インポートした構造がありません")
        return path
    raise ValueError("比較対象が指定されていません")


class CompareBody(BaseModel):
    fixed: StructureRef
    moving: StructureRef


@app.post("/api/compare")
def compare(body: CompareBody) -> dict[str, Any]:
    fixed = _ref_path(body.fixed)
    moving = _ref_path(body.moving)
    key = hashlib.sha1(f"{fixed}|{fixed.stat().st_mtime}|{moving}|{moving.stat().st_mtime}".encode()).hexdigest()[:16]
    name = f"aligned_{key}.cif"
    out = imports_dir() / name
    cache = out.with_suffix(".json")
    if out.exists() and cache.exists():
        try:
            result = json.loads(cache.read_text("utf-8"))
            result["cached"] = True
            return result
        except ValueError:
            pass
    result = structure.superpose(fixed, moving, out)
    result["url"] = f"/api/files/imports/{name}"
    result["output"] = name
    cache.write_text(json.dumps(result), "utf-8")
    return result


@app.get("/api/jobs/{job_id}/autopilot")
def get_autopilot(job_id: str) -> dict:
    job = state.db.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません")
    path = job_file(job_id, "autopilot.json")
    if not path.exists():
        raise HTTPException(status_code=404, detail="このジョブの自動解析はまだありません")
    return json.loads(path.read_text("utf-8"))


# ------------------------------------------------------------------ PDB watcher

@app.get("/api/pdb_watcher/config")
def get_pdb_watcher_config() -> dict:
    return _pdb_watcher_mod.load_config(state.db)


class PdbWatcherConfigBody(BaseModel):
    enabled: bool | None = None
    max_per_poll: int | None = None
    min_seq_len: int | None = None
    max_seq_len: int | None = None


@app.post("/api/pdb_watcher/config")
def set_pdb_watcher_config(body: PdbWatcherConfigBody) -> dict:
    cfg = _pdb_watcher_mod.load_config(state.db)
    update = body.model_dump(exclude_none=True)
    cfg.update(update)
    _pdb_watcher_mod.save_config(state.db, cfg)
    return cfg


@app.post("/api/pdb_watcher/poll_now")
def pdb_watcher_poll_now() -> dict:
    """Trigger an immediate PDB poll in the background."""
    _pdb_watcher_mod.poll_now(state.jobs, state.db)
    return {"status": "polling"}


# ------------------------------------------------------------------ assistant
class AskBody(BaseModel):
    thread_id: str | None = None
    mode: str = "chat"
    message: str = ""
    workbench: dict[str, Any] = {}
    job_id: str | None = None
    scan_job_id: str | None = None
    focus_chain: str | None = None
    count: int = 3
    heavy: bool = False


@app.post("/api/assistant/ask")
def assistant_ask(body: AskBody) -> dict[str, Any]:
    job = state.db.get_job(body.job_id) if body.job_id else None
    if job is not None and (job["kind"] != "predict" or job["status"] != "succeeded"):
        job = None
    scan = None
    scan_chain = None
    if body.scan_job_id:
        sj = state.db.get_job(body.scan_job_id)
        if sj and sj["kind"] == "scan" and sj["status"] == "succeeded":
            scan = sj["result"]
            scan_chain = scan.get("chain")
    if not llm.server_up():
        started = llm.ensure_server()
        if not started.get("running"):
            raise llm.LlmError(started.get("error") or "Ollama が起動していません")
    heavy_model = get_settings().llm_model_heavy.strip()
    if body.heavy and not heavy_model:
        raise HTTPException(400, "じっくり答えるモデルが設定されていません。設定画面で指定してください")
    return assistant.ask(state.db, thread_id=body.thread_id, mode=body.mode, message=body.message,
                         workbench=body.workbench, job=job, scan=scan, scan_chain=scan_chain,
                         focus_chain=body.focus_chain, count=body.count,
                         model=heavy_model if body.heavy else None)


@app.get("/api/llm/calls")
def llm_calls(limit: int = 50, origin: str | None = None, full: bool = False) -> dict[str, Any]:
    if origin not in (None, "user", "autopilot"):
        raise HTTPException(400, "origin は user か autopilot です")
    return {"total": state.db.count_llm_calls(),
            "calls": state.db.list_llm_calls(limit=max(1, min(limit, 500)), origin=origin,
                                             with_messages=full)}


@app.post("/api/llm/calls/export")
def export_llm_calls() -> dict[str, Any]:
    """Write the whole log out as JSONL — one exchange per line, prompt included."""
    out = exports_dir() / f"llm-calls-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    tmp = out.with_suffix(".jsonl.part")
    n = 0
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            for row in state.db.iter_llm_calls():
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
        if n == 0:
            # Writing straight to the destination left an empty file behind every time the
            # log happened to be empty, which reads as "the export is broken".
            tmp.unlink(missing_ok=True)
            raise HTTPException(400, "書き出すやり取りがまだありません")
        tmp.replace(out)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return {"path": str(out), "count": n, "bytes": out.stat().st_size}


@app.get("/api/assistant/threads")
def list_threads() -> list[dict[str, Any]]:
    return state.db.list_threads()


@app.get("/api/assistant/threads/{tid}")
def get_thread(tid: str) -> dict[str, Any]:
    thread = state.db.get_thread(tid)
    if thread is None:
        raise KeyError(tid)
    return thread


@app.delete("/api/assistant/threads/{tid}")
def delete_thread(tid: str) -> dict[str, Any]:
    state.db.delete_thread(tid)
    return {"deleted": tid}


# ------------------------------------------------------------------ library
class LibraryBody(BaseModel):
    type: str
    name: str
    data: dict[str, Any]


@app.get("/api/library")
def library() -> list[dict[str, Any]]:
    return state.db.list_library()


@app.post("/api/library")
def add_library(body: LibraryBody) -> dict[str, Any]:
    if body.type not in ("protein", "dna", "rna", "ligand", "workbench"):
        raise ValueError("type が不正です")
    return state.db.add_library(body.type, body.name, body.data)


@app.delete("/api/library/{item_id}")
def delete_library(item_id: str) -> dict[str, Any]:
    state.db.delete_library(item_id)
    return {"deleted": item_id}


# ------------------------------------------------------------------ frontend
_dist = frontend_dist()
if (_dist / "assets").exists():
    app.mount("/assets", StaticFiles(directory=_dist / "assets"), name="assets")


@app.get("/{full_path:path}", include_in_schema=False)
def spa(full_path: str) -> FileResponse:
    if full_path.startswith("api/"):
        raise HTTPException(404, "見つかりません")
    candidate = (_dist / full_path).resolve()
    if full_path and candidate.is_file() and _dist.resolve() in candidate.parents:
        return FileResponse(candidate)
    index = _dist / "index.html"
    if not index.exists():
        raise HTTPException(503, "フロントエンドがビルドされていません (cd frontend && npm run build)")
    return FileResponse(index)
