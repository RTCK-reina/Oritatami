"""Governor: the single gate every *autonomous* job submission has to pass.

The autopilot loop and the PDB watcher can both queue predictions without a human
in the loop, and a prediction is expensive (minutes to tens of minutes of GPU time
and tens of MB on disk). Without a brake, one user job can fan out into a dozen
variants, and the PDB watcher adds more every six hours — the queue fills with
machine-made work and the user's own jobs end up behind all of it.

Three independent brakes, all configurable in settings:

* **fan-out** — how many variants a single analysed job may spawn
  (applied by the caller, see ``max_variants_per_job``)
* **daily budget** — autonomous predictions started per rolling 24 h
* **free disk** — refuse to start autonomous work when the disk is nearly full

``check()`` returns a decision rather than raising, so callers can log *why* they
backed off. Human-initiated submissions never go through here.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from . import system
from .config import get_settings
from .db import Database

log = logging.getLogger("oritatami.governor")

# Origins that count as machine-made work against the daily budget.
AUTONOMOUS_ORIGINS = ("autopilot_variant", "pdb_watch")

_DAY_SEC = 24 * 3600


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str = ""
    detail: dict[str, Any] | None = None

    def __bool__(self) -> bool:  # `if governor.check(db): ...`
        return self.allowed


def max_variants_per_job() -> int:
    return max(0, int(get_settings().autopilot_max_variants_per_job))


def max_depth() -> int:
    """Generations the chain may go. 0 means unlimited — run until switched off."""
    return max(0, int(get_settings().autopilot_max_depth))


def depth_exhausted(depth: int) -> bool:
    limit = max_depth()
    return limit != 0 and depth >= limit


def queued_autonomous(db: Database) -> int:
    """Autonomous predictions already waiting in the lane."""
    return sum(1 for j in db.list_jobs()
               if j["status"] == "queued" and j.get("origin") in AUTONOMOUS_ORIGINS)


def proposals_per_call() -> int:
    """How many mutations to ask for in one LLM call."""
    return max(1, min(8, int(get_settings().autopilot_proposals_per_call)))


def climb_patience() -> int:
    """Children one branch point may spend before the climb drops to the next-best. 0 = never."""
    return max(0, int(get_settings().autopilot_climb_patience))


def min_esm_llr() -> float:
    return float(get_settings().autopilot_min_esm_llr)


def enabled() -> bool:
    return bool(get_settings().autopilot_enabled)


def used_today(db: Database) -> int:
    return db.count_jobs_since(time.time() - _DAY_SEC, AUTONOMOUS_ORIGINS)


def check(db: Database, *, count: int = 1) -> Decision:
    """May *count* more autonomous prediction(s) be queued right now?"""
    s = get_settings()

    if not s.autopilot_enabled:
        return Decision(False, "autopilot_disabled", {})

    free_gb = system.disk_free_gb()
    if free_gb < s.autopilot_min_disk_gb:
        return Decision(False, "low_disk", {"free_gb": free_gb, "min_gb": s.autopilot_min_disk_gb})

    # A backlog means the lane cannot keep up; adding more only grows it. This is what
    # keeps an unlimited-depth loop from turning into an unbounded queue.
    queued = queued_autonomous(db)
    cap = int(s.autopilot_max_queued)
    if queued + count > cap:
        return Decision(False, "queue_full", {"queued": queued, "cap": cap})

    budget = int(s.autopilot_daily_budget)
    used = used_today(db)
    if budget and used + count > budget:
        return Decision(False, "daily_budget", {"used": used, "budget": budget})

    return Decision(True, "", {"used": used, "budget": budget, "free_gb": free_gb,
                               "queued": queued})


def describe_reason(d: Decision) -> str:
    """Human-readable Japanese explanation, for logs and the UI."""
    det = d.detail or {}
    if d.reason == "autopilot_disabled":
        return "オートパイロットが無効です"
    if d.reason == "low_disk":
        return (f"空き容量が不足しています "
                f"({det.get('free_gb')} GB < 下限 {det.get('min_gb')} GB)")
    if d.reason == "daily_budget":
        return (f"24時間あたりの自律ジョブ上限に達しました "
                f"({det.get('used')}/{det.get('budget')} 件)")
    if d.reason == "queue_full":
        return (f"待機中の自律ジョブが上限です "
                f"({det.get('queued')}/{det.get('cap')} 件)。処理が追いつくまで投入を止めます")
    return "許可"


def _noise(db: Database) -> dict[str, Any]:
    """How big a delta has to be before it means anything, measured from repeats.

    Predicting one sequence twice under identical conditions moves core pLDDT by sigma
    0.219 on this machine. The improvement threshold was set to 0.5 by hand, which is
    1.6 sigma on a difference of two measurements — so a run declares improvements it
    cannot distinguish from rerolling the dice.
    """
    try:
        from . import autopilot, history

        data = history.get(db, lambda r: autopilot.metric_value(r, autopilot.resolve_metric(r)))
    except Exception:  # noqa: BLE001 — status must never fail on a diagnostic
        return {}
    delta = float(get_settings().autopilot_improvement_delta)
    suggested = history.suggested_delta(data)
    return {
        "noise_sigma": data.get("sigma"),
        "noise_repeats": data.get("repeats"),
        "suggested_delta": suggested,
        "delta_below_noise": bool(suggested is not None and delta < suggested),
        "history_pairs": data.get("pairs"),
        "history_mean_delta": data.get("mean_delta"),
        "history_improved_rate": data.get("improved_rate"),
    }


def status(db: Database) -> dict[str, Any]:
    """Snapshot for GET /api/autopilot/status."""
    s = get_settings()
    used = used_today(db)
    free_gb = system.disk_free_gb()
    d = check(db)
    budget = int(s.autopilot_daily_budget)
    return {
        "enabled": s.autopilot_enabled,
        "daily_budget": budget,
        "used_today": used,
        "remaining_today": (max(0, budget - used) if budget else None),
        "max_depth": max_depth(),
        "queued": queued_autonomous(db),
        "max_queued": int(s.autopilot_max_queued),
        "max_variants_per_job": max_variants_per_job(),
        "strategy": s.autopilot_strategy,
        "selection": s.autopilot_selection,
        "protect_interfaces": s.autopilot_protect_interfaces,
        "experiment": s.autopilot_experiment,
        **_noise(db),
        "climb_patience": climb_patience(),
        "min_esm_llr": min_esm_llr(),
        "protected_residues": s.autopilot_protected_residues,
        "min_disk_gb": s.autopilot_min_disk_gb,
        "disk_free_gb": free_gb,
        "accepting": d.allowed,
        "blocked_reason": None if d.allowed else describe_reason(d),
    }
