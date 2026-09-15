"""Job handlers wired into the JobManager."""

from __future__ import annotations

from typing import Any

from .engines import boltz, esm
from .jobs import JobCancelled, JobContext, JobManager
from .seq import clean_sequence


def predict(ctx: JobContext) -> dict[str, Any]:
    return boltz.run_prediction(ctx.job, ctx.dir, ctx.set_phase, ctx.cancelled, ctx.log, ctx.report)


def scan(ctx: JobContext) -> dict[str, Any]:
    spec = ctx.job["spec"]
    seq = clean_sequence(spec["sequence"], "protein")
    ctx.set_phase("load", "ESM-2 を読み込み中")
    esm_status = esm.status()
    if not esm_status["loaded"]:
        ctx.log("ESM-2 model load (first use downloads ~2.5 GB from Hugging Face)")

    def progress(frac: float) -> None:
        if ctx.cancelled():
            raise JobCancelled()
        ctx.set_phase("scan", f"変異スキャン {frac:.0%}", progress=frac)

    result = esm.mutation_scan(seq, progress=progress)
    result["chain"] = spec.get("chain")
    result["label"] = spec.get("label")
    return result


def refine(ctx: JobContext) -> dict[str, Any]:
    spec = ctx.job["spec"]
    seq = clean_sequence(spec["sequence"], "protein")
    fixed = {int(p) - 1 for p in spec.get("fixed_positions") or []}
    ctx.set_phase("load", "ESM-2 を読み込み中")

    def progress(frac: float, label: str) -> None:
        ctx.set_phase("refine", label, progress=frac)

    result = esm.refine(
        seq,
        rounds=int(spec.get("rounds", 6)),
        fraction=float(spec.get("fraction", 0.1)),
        temperature=float(spec.get("temperature", 1.0)),
        fixed=fixed,
        seed=spec.get("seed"),
        progress=progress,
        cancelled=ctx.cancelled,
    )
    if ctx.cancelled():
        raise JobCancelled()
    result["label"] = spec.get("label")
    return result


def register_all(manager: JobManager) -> None:
    manager.register("predict", predict, lane="predict")
    manager.register("scan", scan, lane="esm")
    manager.register("refine", refine, lane="esm")
