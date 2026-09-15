"""What the run has already learned: repeat noise, tried sequences, per-position outcomes.

The loop spent 893 predictions and read none of them back. Three things were sitting in
that history, measured on this machine on 2026-09-14:

1. Reproducibility. 84 predictions (9.4%) recomputed a sequence that had already been
   computed under identical conditions — one hour of compute. Those repeats give the
   noise floor directly: sigma 0.190 core pLDDT with a reused MSA, 0.290 without,
   0.219 pooled. The autopilot calls +0.5 an improvement, which is 1.6 sigma on a
   *difference* of two measurements. Of 1012 regime-comparable parent/child pairs in
   the whole run, 13 cleared 0.5 and only 6 cleared 1.0.

2. Position. Grouping deltas by the position mutated gives a between-group spread of
   0.254 against 0.129 for shuffled labels (permutation test, p = 0.0025) — so which
   residue is touched genuinely predicts the outcome. No position has a positive mean;
   the useful signal is which ones are reliably worse. 23% of all attempts went to the
   six worst.

3. Direction. The mean delta is -0.643 and 4.3% of mutations improve anything. This is
   a search where almost every step is downhill, so not wasting steps matters more than
   picking winners.

Everything here is read-only over finished jobs, and every comparison is filtered through
``regime`` — mixing conditions was what made the older numbers wrong.
"""

from __future__ import annotations

import re
import statistics as st
import threading
import time
from typing import Any

from . import regime
from .db import Database

_MUT = re.compile(r"^([A-Z])(\d{1,4})([A-Z])$")

# Recomputing the whole history walks every finished job; the loop asks once per cycle.
_CACHE_TTL = 120.0
_cache: dict[str, Any] = {}
_lock = threading.Lock()


def _key(job: dict[str, Any], result: dict[str, Any] | None) -> tuple | None:
    """What makes two predictions the same experiment: the molecules and the conditions."""
    comps = ((result or {}).get("normalized_spec") or {}).get("components") or []
    if not comps:
        comps = (job.get("spec") or {}).get("components") or []
    seqs = tuple(sorted((c.get("sequence") or "").upper() for c in comps
                        if c.get("type") in ("protein", "dna", "rna") and c.get("sequence")))
    if not seqs:
        return None
    ligs = tuple(sorted(str(c.get("ccd") or c.get("smiles") or "") for c in comps
                        if c.get("type") == "ligand"))
    return (seqs, ligs, regime.of(result))


def sequence_key(sequences: list[str], ligands: list[str], reg: regime.Regime) -> tuple:
    """The same key, for a candidate that has not been submitted yet."""
    return (tuple(sorted(s.upper() for s in sequences if s)),
            tuple(sorted(str(x) for x in ligands)), reg)


def build(db: Database, score) -> dict[str, Any]:
    """Walk every finished prediction once. ``score`` maps a result to the metric."""
    jobs = {j["id"]: j for j in db.succeeded_predictions(limit=None)}
    results: dict[str, Any] = {}

    def result_of(job_id: str) -> dict[str, Any] | None:
        if job_id not in results:
            full = db.get_job(job_id)
            results[job_id] = (full or {}).get("result")
        return results[job_id]

    by_key: dict[tuple, list[float]] = {}
    best_by_key: dict[tuple, tuple[float, str]] = {}
    deltas: list[float] = []
    by_position: dict[int, list[float]] = {}
    by_substitution: dict[str, list[float]] = {}
    by_strategy: dict[str, list[float]] = {}
    skipped = 0

    for job in jobs.values():
        res = result_of(job["id"])
        value = score(res)
        key = _key(job, res)
        if key is not None and value is not None:
            by_key.setdefault(key, []).append(value)
            prev = best_by_key.get(key)
            if prev is None or value > prev[0]:
                best_by_key[key] = (value, job["id"])

        parent_id = job.get("parent_id")
        if not parent_id or parent_id not in jobs:
            continue
        parent_res = result_of(parent_id)
        child_value, parent_value = value, score(parent_res)
        if child_value is None or parent_value is None:
            continue
        if not regime.comparable(regime.of(res), regime.of(parent_res)):
            skipped += 1
            continue
        delta = child_value - parent_value
        deltas.append(delta)
        strategy = (job.get("spec") or {}).get("autopilot_strategy")
        if strategy:
            by_strategy.setdefault(str(strategy), []).append(delta)
        for code in ((job.get("spec") or {}).get("autopilot_mutation") or "").split("/"):
            m = _MUT.match(code.strip().upper())
            if not m:
                continue
            by_position.setdefault(int(m.group(2)), []).append(delta)
            by_substitution.setdefault(m.group(0), []).append(delta)

    # Repeats of one experiment are the only clean read on how much of a delta is noise.
    spread: dict[str, list[float]] = {}
    repeats = 0
    for key, values in by_key.items():
        if len(values) < 2:
            continue
        repeats += len(values) - 1
        mean = st.mean(values)
        spread.setdefault(key[2].msa, []).extend(v - mean for v in values)
    pooled = [d for vals in spread.values() for d in vals]

    return {
        "predictions": len(jobs),
        "pairs": len(deltas),
        "skipped_regime": skipped,
        "repeats": repeats,
        "mean_delta": round(st.mean(deltas), 4) if deltas else None,
        "improved_rate": round(sum(1 for d in deltas if d > 0) / len(deltas), 4) if deltas else None,
        "sigma": round(st.pstdev(pooled), 4) if len(pooled) > 1 else None,
        "sigma_by_msa": {k: round(st.pstdev(v), 4) for k, v in spread.items() if len(v) > 1},
        "by_position": {p: {"n": len(v), "mean": round(st.mean(v), 4)}
                        for p, v in sorted(by_position.items())},
        "by_strategy": {k: {"n": len(v), "mean": round(st.mean(v), 4),
                            "improved": round(sum(1 for d in v if d > 0) / len(v), 4)}
                        for k, v in by_strategy.items()},
        "by_substitution": {k: {"n": len(v), "mean": round(st.mean(v), 4),
                                "best": round(max(v), 4)}
                            for k, v in by_substitution.items()},
        "known": {k: v for k, v in best_by_key.items()},
        "built_at": time.time(),
    }


def get(db: Database, score) -> dict[str, Any]:
    """Cached ``build`` — the loop asks several times per cycle."""
    with _lock:
        cached = _cache.get("data")
        if cached and time.time() - cached["built_at"] < _CACHE_TTL:
            return cached
        data = build(db, score)
        _cache["data"] = data
        return data


def invalidate() -> None:
    with _lock:
        _cache.pop("data", None)


def position_prior(data: dict[str, Any], position: int, min_n: int = 5) -> float | None:
    """Mean delta seen at this position, or None when too little has been tried there."""
    row = (data.get("by_position") or {}).get(position)
    if not row or row["n"] < min_n:
        return None
    return float(row["mean"])


def detectable(data: dict[str, Any], delta: float) -> bool:
    """Is a delta this size distinguishable from recomputing the same sequence twice?

    The comparison is a difference of two measurements, so the relevant spread is
    sigma*sqrt(2); two of those is the usual bar for calling something real.
    """
    sigma = data.get("sigma")
    if not sigma:
        return True
    return abs(delta) >= 2 * sigma * 1.4142135623730951


def suggested_delta(data: dict[str, Any]) -> float | None:
    """The improvement threshold the measured noise actually supports."""
    sigma = data.get("sigma")
    if not sigma:
        return None
    return round(2 * sigma * 1.4142135623730951, 2)


def required_samples(rate: float, factor: float = 2.0) -> int | None:
    """Predictions per arm to tell ``rate`` from ``rate * factor`` (two-sided 5%, power 80%).

    Improvement is a coin flip that lands 1 time in 10, so a strategy comparison needs
    hundreds of predictions before it says anything. Worth knowing before spending days
    on one: at 78 s per prediction, 98 per arm is about four hours.
    """
    import math

    p1, p2 = rate, min(rate * factor, 0.999)
    if not 0 < p1 < 1 or p2 <= p1:
        return None
    h = 2 * math.asin(math.sqrt(p2)) - 2 * math.asin(math.sqrt(p1))
    if h == 0:
        return None
    return math.ceil((1.96 + 0.84) ** 2 / (h * h))
