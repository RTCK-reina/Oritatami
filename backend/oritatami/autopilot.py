"""Autonomous post-prediction analysis (Autopilot).

After a predict job succeeds, automatically runs AI analysis in the background:
  1. "explain" mode  — interprets the result and suggests next steps
  2. "mutations" mode — proposes stability/affinity mutations for each protein chain

Results are saved to <jobs_dir>/<job_id>/autopilot.json and exposed via
GET /api/jobs/<id>/autopilot.
"""

from __future__ import annotations

import copy
import json
import logging
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from . import assistant, governor, history, llm, regime, system
from .config import get_settings, jobs_dir
from .db import Database
from .engines import boltz, esm, mpnn
from .jobs import JobManager
from .seq import apply_mutations, clean_sequence, diff_substitutions, parse_mutations

log = logging.getLogger("oritatami.autopilot")

# Maximum number of protein chains to run mutation analysis on
_MAX_MUTATION_CHAINS = 2

# How often the watchdog checks that a continuous run is still moving, and how long the
# loop may sit idle before it is restarted. One analysis plus one prediction is a couple
# of minutes, so several minutes of total silence means the chain has died.
_WATCHDOG_PERIOD_SEC = 120.0
_WATCHDOG_IDLE_SEC = 420.0

_watchdog_stop = threading.Event()
# Jobs a restart already tried and failed to advance from, so the watchdog moves on to
# the next-best candidate instead of re-analysing the same one every couple of minutes.
_watchdog_tried: set[str] = set()

# Longest chain we will run a full ESM-2 mutation scan on before asking for mutations.
# The scan is what makes the model's proposals land on real residues (see _chain_scan),
# but it costs one masked forward pass per position, so it is capped.
_MAX_SCAN_LEN = 400

# Time to wait for Ollama to be ready before giving up (seconds)
_OLLAMA_WAIT = 180.0

# How deep the chain may go now lives in settings (autopilot_max_depth, 0 = unlimited)
# and is read through governor.depth_exhausted(). Continuous operation is that setting
# at 0 with a fan-out of 1: one lineage advancing a step at a time, bounded by the
# queue cap, the disk floor and the enabled switch rather than by a generation count.


def start(jobs: JobManager, db: Database) -> None:
    """Hook into the JobManager; called once from app lifespan."""

    def _on_complete(job: dict[str, Any]) -> None:
        if job.get("kind") != "predict":
            return
        if job.get("status") != "succeeded":
            return

        # Run in the callback thread (already background)
        _analyze(job, db, jobs)

    def _on_complete_notify(job: dict[str, Any]) -> None:
        if job.get("kind") != "predict" or job.get("status") != "succeeded":
            return
        try:
            _notify_if_improved(job, db)
        except Exception as exc:  # a notification must never break the pipeline
            log.debug("Autopilot: 改善通知に失敗しました: %s", exc)

    jobs.on_complete(_on_complete)
    # Separate listener so the (fast) comparison is not stuck behind the (slow) LLM run.
    jobs.on_complete(_on_complete_notify)

    _watchdog_stop.clear()
    threading.Thread(target=_watchdog, args=(jobs, db), daemon=True,
                     name="autopilot-watchdog").start()
    log.info("Autopilot: 構造予測完了時の自動解析を有効にしました (ウォッチドッグ %.0f 秒)",
             _WATCHDOG_IDLE_SEC)


# ── Improvement detection ─────────────────────────────────────────────────────

# Every score normalised onto the 0–100 pLDDT scale, so one threshold setting covers
# all of them: ipTM 0.62 → 62, mean pLDDT 88.5 → 88.5.
_METRIC_SCALE = {
    "mean_plddt": 1.0,
    "complex_plddt": 100.0,
    "confidence_score": 100.0,
    "iptm": 100.0,
    "ptm": 100.0,
}

_METRIC_LABEL = {
    "mean_plddt": "平均pLDDT",
    "complex_plddt": "複合体pLDDT",
    "confidence_score": "信頼度スコア",
    "iptm": "ipTM",
    "ptm": "pTM",
}


def mean_plddt(result: dict[str, Any] | None) -> float | None:
    """Mean per-residue pLDDT of the top model, or None when unavailable."""
    if not result:
        return None
    models = result.get("models") or []
    if not models:
        return None
    values = [v for arr in (models[0].get("plddt") or {}).values() for v in arr]
    return sum(values) / len(values) if values else None


CORE_TRIM_FRACTION = 0.10


def core_plddt(result: dict[str, Any] | None) -> float | None:
    """Mean per-residue pLDDT with the worst 10% of residues dropped.

    The plain mean is dominated by whatever the predictor is least sure about, which for
    most proteins is a flexible terminus or loop — and those are the cheapest residues to
    "improve". In one 491-prediction run the four C-terminal residues supplied a third of
    the total gain while the protein lost the very motif it needs to function. Trimming
    the tail of the distribution scores the part of the molecule that was actually folded.
    """
    if not result:
        return None
    models = result.get("models") or []
    if not models:
        return None
    values = sorted(v for arr in (models[0].get("plddt") or {}).values() for v in arr)
    if not values:
        return None
    drop = min(len(values) - 1, int(len(values) * CORE_TRIM_FRACTION))
    kept = values[drop:]
    return sum(kept) / len(kept)


def _top_model(result: dict[str, Any] | None) -> dict[str, Any]:
    models = (result or {}).get("models") or []
    return models[0] if models else {}


def metric_value(result: dict[str, Any] | None, metric: str) -> float | None:
    """A prediction's score for *metric*, normalised onto the 0–100 pLDDT scale."""
    if metric == "mean_plddt":
        return mean_plddt(result)
    if metric == "core_plddt":
        return core_plddt(result)
    raw = (_top_model(result).get("confidence") or {}).get(metric)
    try:
        return float(raw) * _METRIC_SCALE[metric] if raw is not None else None
    except (TypeError, ValueError, KeyError):
        return None


def resolve_metric(result: dict[str, Any] | None) -> str:
    """Which score to judge this prediction by, honouring the "auto" setting.

    A monomer has no interface, and Boltz reports ipTM 0 for one — judging it by ipTM
    would be meaningless. So "auto" means ipTM for multi-chain predictions and mean
    pLDDT otherwise.
    """
    configured = get_settings().autopilot_improvement_metric
    if configured != "auto":
        return configured
    chains = (_top_model(result).get("plddt") or {})
    ligands = (_top_model(result).get("ligand_plddt") or {})
    return "iptm" if (len(chains) + len(ligands)) > 1 else "mean_plddt"


def _notify_if_improved(job: dict[str, Any], db: Database) -> None:
    """Notify only when a variant actually beats the job it was derived from.

    An autonomous loop that announces every completion is noise; the one event worth
    interrupting someone for is "this variant is better than its parent". Fires from the
    backend, so it works with the window closed. Parent and child are compared on the
    same metric — resolved from the child, since a variant keeps its parent's chain
    composition.
    """
    s = get_settings()
    if not s.autopilot_notify_improvement:
        return
    parent_id = job.get("parent_id")
    if not parent_id:
        return
    parent = db.get_job(parent_id)
    if parent is None or parent.get("status") != "succeeded":
        return

    metric = resolve_metric(job.get("result"))
    child_score = metric_value(job.get("result"), metric)
    parent_score = metric_value(parent.get("result"), metric)
    if child_score is None or parent_score is None:
        return

    # Same sequence, different MSA: up to 1.9 pLDDT on this machine's own history — nearly
    # four times the improvement threshold. Comparing across regimes reports the settings
    # change as a discovery.
    child_reg, parent_reg = regime.of(job.get("result")), regime.of(parent.get("result"))
    if not regime.comparable(child_reg, parent_reg):
        log.info("Autopilot: %s と親は計算条件が違うので比較しません (%s)",
                 job["id"][:8], ", ".join(regime.differences(child_reg, parent_reg)) or "条件不明")
        return

    delta = child_score - parent_score
    if delta < float(s.autopilot_improvement_delta):
        return

    label = _METRIC_LABEL.get(metric, metric)
    scale = _METRIC_SCALE.get(metric, 1.0)
    # Report in the metric's own units, not the normalised ones.
    p_disp, c_disp, d_disp = parent_score / scale, child_score / scale, delta / scale
    digits = 1 if scale == 1.0 else 3
    log.info("Autopilot: %s が親を上回りました (%s %.*f → %.*f, +%.*f)",
             job["id"][:8], label, digits, p_disp, digits, c_disp, digits, d_disp)
    system.notify(
        "親を上回る変異体が見つかりました",
        f"{job.get('title') or job['id']}  {label} {p_disp:.{digits}f} → {c_disp:.{digits}f} "
        f"(+{d_disp:.{digits}f})",
        respect_setting=False,
    )


def _resumable(job: dict[str, Any]) -> bool:
    """Can a chain actually be continued from this job? It needs a protein to mutate."""
    spec = job.get("spec") or {}
    comps = ((spec.get("workbench") or {}).get("components")
             or spec.get("components") or [])
    return any(c.get("type") == "protein" and c.get("sequence") for c in comps)


def _ranked_jobs(db: Database, exclude: set[str] | None = None) -> list[dict[str, Any]]:
    """Finished predictions a chain can be resumed from, best first."""
    exclude = exclude or set()
    experiment = (get_settings().autopilot_experiment or "").strip()
    scored = []
    for j in db.ranking_rows():
        if j["id"] in exclude or not _resumable(j):
            continue
        if experiment and ((j.get("spec") or {}).get("autopilot_experiment") or "") != experiment:
            continue
        score = metric_value(j.get("result"), resolve_metric(j.get("result")))
        if score is not None:
            scored.append((score, j))
    scored.sort(key=lambda t: t[0], reverse=True)
    if scored:
        # The peak sets the bar every child is measured against, so a lone high score from
        # a different regime would hold the climb hostage to a settings change.
        top = regime.of(scored[0][1].get("result"))
        same = [(v, j) for v, j in scored
                if regime.comparable(regime.of(j.get("result")), top)]
        if len(same) < len(scored):
            log.info("Autopilot: 計算条件が頂点 (%s) と違う %d 件を候補から外しました",
                     top.label(), len(scored) - len(same))
        # Never hand back an empty list on the strength of a bookkeeping rule: with no
        # peak the climb has nothing to branch from and the cycle spins.
        scored = same or scored
    if experiment and not scored:
        log.warning("Autopilot: 実験名 '%s' に属する完了済みの予測がありません。"
                    "この名前で最初の予測を1件走らせるか、実験名を空にしてください", experiment)
    return [j for _, j in scored]


def _best_succeeded_predict(db: Database, exclude: set[str] | None = None) -> dict[str, Any] | None:
    """The highest-scoring finished prediction a chain can be resumed from."""
    ranked = _ranked_jobs(db, exclude)
    # Ranking reads a trimmed row; callers go on to analyse the job, so hand back the real one.
    return db.get_job(ranked[0]["id"]) if ranked else None


def _branch_point(db: Database, exclude: set[str] | None = None) -> dict[str, Any] | None:
    """Where the next attempt should branch from.

    The peak, until the climb has spent ``autopilot_climb_patience`` children on it
    without any of them beating it — then the next-best result, and so on. That is what
    lets the search reach two-mutation territory: a sequence that is already excellent
    (wild-type ubiquitin outscored all 72 single mutants tried against it) would
    otherwise hold the branch point forever. When every candidate is spent the peak is
    used again rather than stalling the loop.
    """
    patience = governor.climb_patience() if get_settings().autopilot_strategy == "climb" else 0
    ranked = _ranked_jobs(db, exclude)
    if not ranked:
        return None
    if patience:
        for job in ranked:
            if len(_tried_mutations(db, job["id"])) < patience:
                if job["id"] != ranked[0]["id"]:
                    log.info("Autopilot: 頂点 %s は %d 手試して超えられなかったので、次点 %s から探します",
                             (ranked[0].get("title") or "")[:28], patience,
                             (job.get("title") or "")[:28])
                return db.get_job(job["id"])
        log.info("Autopilot: 上位の候補をすべて試し切ったので頂点に戻ります")
    return db.get_job(ranked[0]["id"])


def _loop_is_idle(db: Database) -> bool:
    """Nothing running, nothing queued, and nothing finished recently."""
    jobs = db.list_jobs(limit=200)
    if any(j["status"] in ("queued", "running") for j in jobs):
        return False
    newest = max((j.get("finished_at") or 0) for j in jobs) if jobs else 0
    return (time.time() - newest) > _WATCHDOG_IDLE_SEC


def _watchdog(jobs: JobManager, db: Database) -> None:
    """Keep a continuous run going across a failed analysis.

    The chain advances only when an analysis yields a usable proposal, so one bad LLM
    reply — a truncated JSON, a refusal, Ollama restarting — ends it for good, silently.
    That is fine for a bounded depth but not for a loop asked to run until it is switched
    off. When everything has been quiet for a while, this re-analyses the best result so
    far, which both restarts the chain and pulls it back to the top of the hill instead
    of leaving it wherever the random walk wandered to.
    """
    while not _watchdog_stop.wait(_WATCHDOG_PERIOD_SEC):
        try:
            if not governor.enabled() or governor.max_depth() != 0:
                continue  # only guards continuous mode
            if not _loop_is_idle(db):
                continue
            decision = governor.check(db)
            if not decision:
                log.info("Autopilot ウォッチドッグ: 再開を見送りました — %s",
                         governor.describe_reason(decision))
                continue
            best = _branch_point(db, exclude=_watchdog_tried)
            if best is None:
                if _watchdog_tried:
                    # Everything has been tried once; allow another sweep rather than
                    # going quiet for good.
                    log.info("Autopilot ウォッチドッグ: 候補を一巡したので再試行対象を戻します")
                    _watchdog_tried.clear()
                continue
            out = jobs_dir() / best["id"] / "autopilot.json"
            log.warning("Autopilot ウォッチドッグ: %.0f 秒動きがないため最良ジョブ %s から再開します",
                        _WATCHDOG_IDLE_SEC, (best.get("title") or best["id"])[:40])
            out.unlink(missing_ok=True)   # let _analyze run again on it
            _watchdog_tried.add(best["id"])
            _analyze(best, db, jobs)
            if _loop_is_idle(db):
                log.warning("Autopilot ウォッチドッグ: %s からは再開できませんでした。次の候補を試します",
                            (best.get("title") or best["id"])[:40])
            else:
                _watchdog_tried.clear()   # moving again — every job is a candidate again
        except Exception as exc:
            log.warning("Autopilot ウォッチドッグ: %s", exc)


def _chain_scan(comp: dict[str, Any], job_id: str, chain: str) -> dict[str, Any] | None:
    """Run an ESM-2 mutation scan so the model has real, pre-validated substitutions.

    Left to itself the model reads positions off the low-pLDDT range list and guesses
    which residue is there, so most of its proposals name a wild-type residue the
    sequence does not have and get rejected. Every entry in a scan's
    ``top_substitutions`` is generated from the sequence itself, so handing them over
    turns the task from "invent a mutation code" into "pick from this list".

    Best effort: a scan failure or an over-long chain just means the model works from
    the numbered residue table alone.
    """
    seq = comp.get("sequence") or ""
    if not seq or len(seq) > _MAX_SCAN_LEN:
        if seq:
            log.info("Autopilot: チェーン %s は %d 残基のため ESM-2 スキャンを省略します",
                     chain, len(seq))
        return None
    try:
        started = time.time()
        scan = esm.mutation_scan(seq)
        log.info("Autopilot: %s — ESM-2 スキャン (チェーン %s, %d 残基, %.1f 秒)",
                 job_id, chain, len(seq), time.time() - started)
        return scan
    except Exception as exc:
        log.warning("Autopilot: チェーン %s の ESM-2 スキャンに失敗しました: %s", chain, exc)
        return None


def _analyze(job: dict[str, Any], db: Database, jobs: JobManager,
             spent: set[str] | None = None) -> None:
    job_id = job["id"]
    out_path = jobs_dir() / job_id / "autopilot.json"

    if out_path.exists():
        log.info("Autopilot: %s は既に解析済みです", job_id)
        return

    log.info("Autopilot: %s の自動解析を開始します", job_id)

    # Wait for Ollama
    deadline = time.time() + _OLLAMA_WAIT
    waited = False
    while time.time() < deadline:
        if llm.server_up():
            if waited:
                log.info("Autopilot: Ollama が応答しました。%s の解析を続けます", job_id)
            break
        if not waited:
            # Three minutes of silence looked like a hung loop; say what is being waited on.
            log.info("Autopilot: Ollama の起動を待っています (最大 %.0f 秒)", _OLLAMA_WAIT)
            waited = True
        time.sleep(5.0)
    else:
        log.warning("Autopilot: Ollama が起動していないため %s の解析をスキップします", job_id)
        return

    spec = job.get("spec") or {}
    # Jobs queued from the UI carry only `spec.components`; `workbench` is set by the
    # paths that build a spec themselves. Without this fallback the mutations pass finds
    # no chains, produces no proposals, and a continuous run cannot resume from such a
    # job — it just re-runs `explain` forever.
    workbench = spec.get("workbench") or {}
    if not (workbench.get("components") or []):
        comps = [{k: v for k, v in c.items() if k != "msa"}
                 for c in (spec.get("components") or [])]
        if comps:
            workbench = {"name": spec.get("workbench_name") or spec.get("name") or job.get("title"),
                         "components": comps}
    started_at = time.time()

    results: dict[str, Any] = {
        "job_id": job_id,
        "started_at": started_at,
        "analyses": [],
    }

    protein_comps = [c for c in (workbench.get("components") or []) if c.get("type") == "protein"]

    # One scan up front, shared by explain and by the first chain's mutation pass. Explain
    # is asked for proposals too, and without the scan its mutation codes name residues
    # the sequence does not have — the same failure the mutations pass had.
    lead_comp = protein_comps[0] if protein_comps else None
    lead_chain = ((lead_comp.get("chains") or ["A"])[0]) if lead_comp else None
    scans: dict[str, dict[str, Any] | None] = {}
    if lead_comp is not None:
        scans[lead_chain] = _chain_scan(lead_comp, job_id, lead_chain)

    # -- 1. Explain ----------------------------------------------------------
    # A second LLM call per generation, whose prose nothing reads: 27% of the wall
    # clock for the 25% of generations where it happens to out-propose the mutations
    # pass. Off, the loop runs about a third more generations in the same hour.
    if get_settings().autopilot_explain:
        try:
            log.info("Autopilot: %s — explain", job_id)
            lead_scan = scans.get(lead_chain) if lead_chain else None
            r = assistant.ask(
                db,
                thread_id=None,
                mode="explain",
                message="この予測結果を解説し、注目すべき点と次のステップを提案してください",
                workbench=workbench,
                job=job,
                scan=lead_scan,
                scan_chain=lead_chain if lead_scan else None,
                focus_chain=lead_chain,
                count=2,
                persist=False,
                verify_reply=False,
                origin="autopilot",
            )
            results["analyses"].append({
                "mode": "explain",
                "thread_id": r["thread_id"],
                "reply": r["reply"],
                "proposals": r["proposals"],
                "reply_issues": r.get("reply_issues") or [],
                "elapsed_sec": r.get("elapsed_sec"),
            })
        except Exception as exc:
            log.warning("Autopilot: explain 失敗 (%s): %s", job_id, exc)
            results["analyses"].append({"mode": "explain", "error": str(exc)})

    # -- 2. Mutations per protein chain ---------------------------------------
    for comp in protein_comps[:_MAX_MUTATION_CHAINS]:
        chains = comp.get("chains") or []
        chain = chains[0] if chains else "A"
        if chain not in scans:
            scans[chain] = _chain_scan(comp, job_id, chain)
        scan = scans[chain]
        try:
            log.info("Autopilot: %s — mutations (chain %s)", job_id, chain)
            # Say it in the prompt as well as filtering afterwards: a proposal that never
            # gets made costs nothing, one that gets rejected costs a generation's ideas.
            # The list goes into the residue table itself (see numbered_residues) because
            # a sentence alone was ignored by 28% of proposals.
            forbidden = sorted(_protected_positions(db, job, chain))
            note = ""
            if forbidden:
                shown = ", ".join(str(x) for x in forbidden[:20])
                note = (f" 次の残基番号は絶対に変更しないでください: {shown}"
                        + ("ほか" if len(forbidden) > 20 else "") + "。")
            r = assistant.ask(
                db,
                thread_id=None,
                mode="mutations",
                message="熱安定性・結合親和性・溶解性を改善する変異を提案してください" + note,
                workbench=workbench,
                job=job,
                scan=scan,
                scan_chain=chain if scan else None,
                focus_chain=chain,
                count=governor.proposals_per_call(),
                persist=False,
                verify_reply=False,
                origin="autopilot",
                fixed_positions=set(forbidden),
            )
            results["analyses"].append({
                "mode": "mutations",
                "chain": chain,
                "thread_id": r["thread_id"],
                "reply": r["reply"],
                "proposals": r["proposals"],
                "reply_issues": r.get("reply_issues") or [],
                "elapsed_sec": r.get("elapsed_sec"),
            })
        except Exception as exc:
            log.warning("Autopilot: mutations chain %s 失敗 (%s): %s", chain, job_id, exc)
            results["analyses"].append({"mode": "mutations", "chain": chain, "error": str(exc)})

    results["finished_at"] = time.time()
    results["elapsed_sec"] = round(results["finished_at"] - started_at, 1)

    # Hand the memory back. The analysis runs while the next prediction may already be
    # underway, and Ollama holds a model until its keep_alive expires — 6 GB of unified
    # memory sitting on top of a Boltz run that wants nearly all of it.
    try:
        llm.unload_model()
    except Exception as exc:  # never let bookkeeping kill the loop
        log.warning("Autopilot: LLM のメモリ解放に失敗しました: %s", exc)

    try:
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), "utf-8")
        log.info("Autopilot: %s 完了 (%.1f 秒)", job_id, results["elapsed_sec"])
    except OSError as exc:
        log.error("Autopilot: 結果の保存に失敗しました (%s): %s", job_id, exc)

    # Notify the frontend that something changed
    jobs.touch()

    # Auto-submit the best verified mutation proposals as new predict jobs
    _auto_submit_variants(job, results, jobs, db, spent=spent)


def _proposal_llr(proposal: dict[str, Any]) -> float | None:
    """Total ESM-2 log-likelihood ratio for a proposal, if it was scored.

    Higher is better: positive means ESM-2 considers the mutant more likely than the
    wild type at those positions. ``assistant._verify_mutations`` already computed
    this during the analysis, so using it costs nothing.
    """
    esm = proposal.get("esm")
    if not isinstance(esm, dict):
        return None
    value = esm.get("total_llr")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _collect_variant_candidates(job: dict[str, Any], results: dict[str, Any]) -> list[dict[str, Any]]:
    """Turn every verified mutation proposal into a ready-to-submit variant spec."""
    spec = job.get("spec") or {}
    depth = int(spec.get("autopilot_depth", 0) or 0)
    workbench = spec.get("workbench") or {}
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for analysis in results.get("analyses") or []:
        for proposal in analysis.get("proposals") or []:
            if proposal.get("status") not in ("ok", "warning"):
                continue
            apply = proposal.get("apply") or {}
            if apply.get("action") != "mutate":
                continue
            chain = apply.get("chain", "A")
            mutations: list[str] = apply.get("mutations") or []
            if not mutations:
                continue

            comp_src = next(
                (c for c in (workbench.get("components") or [])
                 if c.get("type") == "protein" and chain in (c.get("chains") or ["A"])),
                None,
            )
            if comp_src is None:
                comp_src = next(
                    (c for c in (spec.get("components") or []) if c.get("type") == "protein"),
                    None,
                )
            if comp_src is None:
                continue

            try:
                seq = clean_sequence(comp_src.get("sequence", ""), "protein")
                muts = parse_mutations(mutations)
                mutated = apply_mutations(seq, muts)
                if mutated == seq:
                    continue  # no-op proposal
                mut_str = "/".join(m.split(":")[-1] for m in mutations)
                key = (chain, mut_str)
                if key in seen:
                    continue  # the same mutation proposed by two analysis modes
                seen.add(key)

                base = comp_src.get("parent_sequence") or seq
                diff = diff_substitutions(base, mutated) or []

                vspec = copy.deepcopy(spec)
                base_name = (spec.get("workbench_name") or spec.get("name") or "variant").split(" + ")[0]
                vspec["name"] = vspec["workbench_name"] = f"{base_name} + {mut_str}"
                vspec["autopilot_depth"] = depth + 1
                vspec["autopilot_source"] = "autopilot_variant"
                # Stamped from the setting, not inherited. Copied from the parent it went
                # stale the moment the name changed, and every child kept carrying the old
                # tag — which _ranked_jobs then filtered out, leaving the loop with no peak
                # to climb from and nothing in the log to say why.
                settings = get_settings()
                exp = (settings.autopilot_experiment or "").strip()
                if exp:
                    vspec["autopilot_experiment"] = exp
                else:
                    vspec.pop("autopilot_experiment", None)
                # Which search produced this attempt. Unrecorded, the only way to compare
                # hill climb against random walk was to split the history by date — and
                # the two periods differ in sequence, depth and settings as well, so the
                # comparison measures everything at once. Stamped, an A/B is exact.
                vspec["autopilot_strategy"] = settings.autopilot_strategy
                if settings.autopilot_strategy == "climb":
                    vspec["autopilot_climb_patience"] = int(settings.autopilot_climb_patience)
                # What this attempt changed relative to its parent. A hill climb retries
                # the same parent with different candidates, so it has to know which ones
                # it already spent a prediction on.
                vspec["autopilot_mutation"] = mut_str
                vspec.pop("_retry_attempt", None)

                for group in (vspec.get("components") or [],
                              (vspec.get("workbench") or {}).get("components") or []):
                    for c in group:
                        if c.get("type") == "protein" and chain in (c.get("chains") or ["A"]):
                            c["sequence"] = mutated
                            c["parent_sequence"] = base
                            c["mutations"] = diff
                            break

                boltz.normalize_spec(vspec)
                candidates.append({
                    "chain": chain,
                    "mut_str": mut_str,
                    "llr": _proposal_llr(proposal),
                    "spec": vspec,
                })
            except Exception as exc:
                log.warning("Autopilot: バリアント '%s' を組み立てられませんでした: %s", mutations, exc)

    return candidates


_POS_IN_CODE = re.compile(r"(\d+)")


def _explicit_protected() -> set[int]:
    """Residue positions the user has put off limits, from "K63, G75, 76"-style text."""
    out: set[int] = set()
    for token in (get_settings().autopilot_protected_residues or "").replace(";", ",").split(","):
        m = _POS_IN_CODE.search(token)
        if m:
            out.add(int(m.group(1)))
    return out


def _lineage_root(db: Database, job: dict[str, Any]) -> dict[str, Any]:
    """The prediction this chain started from — the reference for what was ever ordered."""
    seen: set[str] = set()
    cur = job
    while cur.get("parent_id") and cur["parent_id"] not in seen:
        seen.add(cur["parent_id"])
        nxt = db.get_job(cur["parent_id"])
        if nxt is None:
            break
        cur = nxt
    return cur


def _interface_positions(job: dict[str, Any], chain: str) -> set[int]:
    """Residues this chain uses to touch another chain, as the prediction measured them.

    Boltz computes the contact list for every prediction and it was read only by the
    prose pass. For a complex it is the one part of the molecule whose job is known, so
    it belongs in the protected set unless the person is deliberately redesigning the
    interface.
    """
    result = job.get("result") or {}
    block = result.get("interfaces") or {}
    out: set[int] = set()
    for itf in (block.get("interfaces") or []):
        residues = itf.get("residues") or {}
        for code in residues.get(chain) or []:
            digits = "".join(ch for ch in str(code) if ch.isdigit())
            if digits:
                out.add(int(digits))
    return out


def _protected_positions(db: Database, job: dict[str, Any], chain: str) -> set[int]:
    """Positions no proposal may touch.

    Two sources: the list the user wrote, and every residue that was disordered in the
    lineage root. The second one is the general form of the failure this search hit —
    a residue the predictor is unsure about is a free +30 pLDDT to anyone willing to
    replace it, and those residues are disproportionately the functional ones (termini,
    linkers, catalytic loops).
    """
    protected = _explicit_protected()
    s = get_settings()
    if s.autopilot_protect_interfaces:
        contacts = _interface_positions(job, chain)
        if contacts:
            log.info("Autopilot: チェーン %s の界面残基 %d 個を保護します", chain, len(contacts))
        protected |= contacts
    if not s.autopilot_protect_disordered:
        return protected
    root = _lineage_root(db, job)
    models = ((root.get("result") or {}).get("models") or [])
    if not models:
        return protected
    per_res = (models[0].get("plddt") or {}).get(chain) or []
    protected |= {i for i, v in enumerate(per_res, 1)
                  if isinstance(v, (int, float)) and v < s.autopilot_disorder_plddt}
    return protected


def _touches_protected(mut_str: str, protected: set[int]) -> list[int]:
    return sorted({int(m) for m in _POS_IN_CODE.findall(mut_str)} & protected)


def _load_analysis(job_id: str) -> dict[str, Any] | None:
    """The stored analysis for a job, so the peak can be branched from again."""
    path = jobs_dir() / job_id / "autopilot.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _tried_mutations(db: Database, parent_id: str) -> set[str]:
    """Mutations already spent a prediction on from this parent, successful or not."""
    return {
        (j.get("spec") or {}).get("autopilot_mutation")
        for j in db.children_of(parent_id)
        if (j.get("spec") or {}).get("autopilot_mutation")
    }


def _ranked_candidates(job: dict[str, Any], results: dict[str, Any],
                       db: Database | None = None) -> list[dict[str, Any]]:
    """Verified proposals for this job, protected residues and ESM-2 floor applied, best first."""
    candidates = _collect_variant_candidates(job, results)

    if db is not None:
        by_chain: dict[str, set[int]] = {}
        allowed = []
        for c in candidates:
            chain = c["chain"]
            if chain not in by_chain:
                by_chain[chain] = _protected_positions(db, job, chain)
            hit = _touches_protected(c["mut_str"], by_chain[chain])
            if hit:
                log.info("Autopilot: '%s' は保護残基 %s に触るため除外しました",
                         c["mut_str"], ", ".join(str(h) for h in hit))
            else:
                allowed.append(c)
        candidates = allowed

    floor = governor.min_esm_llr()
    kept = [c for c in candidates if c["llr"] is None or c["llr"] >= floor]
    dropped = len(candidates) - len(kept)
    if dropped:
        log.info("Autopilot: ESM-2 スコアが下限 (%.2f) を下回る %d 件を除外しました", floor, dropped)

    if db is not None:
        kept = _drop_already_measured(db, kept)
        kept = _inverse_folding_check(job, kept)
        kept = _apply_history(db, kept)
    else:
        # A scored proposal always outranks an unscored one: a measured signal beats none.
        kept.sort(key=lambda c: (c["llr"] is not None, c["llr"] if c["llr"] is not None else 0.0),
                  reverse=True)
    return kept


def _backbone_of(job_id: str) -> Path | None:
    out = jobs_dir() / job_id / "out"
    if not out.exists():
        return None
    cifs = sorted(out.rglob("*_model_0.cif")) or sorted(out.rglob("*.cif"))
    return cifs[0] if cifs else None


def _inverse_folding_check(job: dict[str, Any], candidates: list[dict[str, Any]]
                           ) -> list[dict[str, Any]]:
    """Ask a scorer whose blind spot points the other way, before spending a prediction.

    The parent's structure is already on disk, so every candidate can be threaded onto it
    for a couple of seconds — against the 78 seconds a prediction costs. Measured over 480
    historical pairs, keeping the better half by this score lifts the share of attempts
    that beat their parent from 4.6% to 8.3%.
    """
    s = get_settings()
    if not s.mpnn_enabled or not candidates or not mpnn.available():
        return candidates
    parent_seq = None
    for comp in ((job.get("spec") or {}).get("components") or []):
        if comp.get("type") == "protein" and comp.get("sequence"):
            parent_seq = comp["sequence"]
            break
    cif = _backbone_of(job["id"])
    if cif is None or not parent_seq:
        return candidates

    usable, rest = [], []
    for c in candidates:
        seqs = [cc.get("sequence") for cc in (c["spec"].get("components") or [])
                if cc.get("type") == "protein" and cc.get("sequence")]
        if len(seqs) == 1 and len(seqs[0]) == len(parent_seq):
            usable.append((c, seqs[0]))
        else:
            rest.append(c)          # length changed or several chains: not threadable
    if not usable:
        return candidates

    try:
        with tempfile.TemporaryDirectory() as tmp:
            pdb = mpnn.backbone_pdb(cif, Path(tmp) / "parent.pdb")
            scores = mpnn.relative(pdb, parent_seq, [seq for _, seq in usable])
    except Exception as exc:  # noqa: BLE001 — a third opinion must not stop the loop
        log.info("Autopilot: 逆折り畳みの採点をとばしました (%s)", exc)
        return candidates

    veto = float(s.mpnn_veto)
    kept = []
    rejected = []
    for (c, _seq), value in zip(usable, scores, strict=False):
        c["mpnn"] = value
        if veto > 0 and value > veto:
            rejected.append(c)
        else:
            kept.append(c)
    if rejected:
        log.info("Autopilot: 逆折り畳みが骨格に合わないと判断した %d 件を却下しました (%s)",
                 len(rejected),
                 ", ".join(f"{c['mut_str']} {c['mpnn']:+.3f}" for c in rejected[:4]))
    if not kept:
        # Everything vetoed means the proposals were uniformly bad, not that the cycle
        # should stall; keep the least bad one and say so.
        best = min(rejected, key=lambda c: c["mpnn"])
        log.info("Autopilot: 全件却下だったので最もましな %s (%.3f) だけ残します",
                 best["mut_str"], best["mpnn"])
        kept = [best]
    return kept + rest


def _history(db: Database) -> dict[str, Any]:
    return history.get(db, lambda r: metric_value(r, resolve_metric(r)))


def _drop_already_measured(db: Database, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Never spend a prediction on a sequence that has already been predicted.

    Not the same as "already tried from this parent": two branches reach the same molecule
    by different routes, and the loop then paid for it twice. Measured over this machine's
    893 predictions, 84 of them (9.4%, one hour of compute) recomputed a sequence that had
    already been computed under identical conditions.
    """
    data = _history(db)
    known = data.get("known") or {}
    if not known:
        return candidates
    fresh, seen = [], []
    for c in candidates:
        spec = c["spec"]
        comps = spec.get("components") or []
        key = history.sequence_key(
            [cc.get("sequence") or "" for cc in comps
             if cc.get("type") in ("protein", "dna", "rna")],
            [str(cc.get("ccd") or cc.get("smiles") or "") for cc in comps
             if cc.get("type") == "ligand"],
            # A candidate has not run yet, so it will inherit whatever the settings say
            # now; compare against the regime the loop is currently producing.
            _current_regime(data),
        )
        hit = known.get(key)
        if hit:
            seen.append((c["mut_str"], hit[0]))
        else:
            fresh.append(c)
    if seen:
        log.info("Autopilot: 既に同じ配列を計算済みの %d 件を除外しました (%s)", len(seen),
                 ", ".join(f"{m} → {v:.2f}" for m, v in seen[:4]))
    return fresh


def _current_regime(data: dict[str, Any]) -> regime.Regime:
    """The conditions the next prediction will run under, as the settings stand."""
    s = get_settings()
    msa = "reused" if s.reuse_msa_for_variants else ("server" if s.msa_server_url else "single")
    return regime.Regime(int(s.diffusion_samples), int(s.recycling_steps),
                         int(s.sampling_steps), bool(s.use_potentials), msa)


def _apply_history(db: Database, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order by what this run has already learned, not by ESM-2 alone.

    Grouping past deltas by the position mutated separates them well beyond chance
    (permutation test p = 0.0025 over 1012 pairs), and no position has a positive mean —
    so the history says which residues are reliably worse, and that is what it is used
    for. ESM-2 stays the primary key; history breaks its ties and pushes known-bad
    positions down.
    """
    data = _history(db)
    if not (data.get("by_position")):
        candidates.sort(key=lambda c: (c["llr"] is not None,
                                       c["llr"] if c["llr"] is not None else 0.0), reverse=True)
        return candidates

    def prior(c: dict[str, Any]) -> float:
        seen = [history.position_prior(data, int(m)) for m in _POS_IN_CODE.findall(c["mut_str"])]
        seen = [v for v in seen if v is not None]
        return min(seen) if seen else 0.0

    def novelty(c: dict[str, Any]) -> int:
        """How many attempts this run has already spent on the positions being touched."""
        counts = data.get("by_position") or {}
        return max((counts.get(int(m), {}).get("n", 0)
                    for m in _POS_IN_CODE.findall(c["mut_str"])), default=0)

    for c in candidates:
        c["history_prior"] = round(prior(c), 4)
        c["position_tries"] = novelty(c)

    if get_settings().autopilot_selection == "explore":
        # Greedy ranking keeps returning to the positions it already understands: 23% of
        # this run's attempts went to six residues. Least-tried first turns the same
        # budget into coverage, with ESM-2 and the veto still filtering what is allowed.
        candidates.sort(key=lambda c: (c["position_tries"],
                                       -(c["llr"] if c["llr"] is not None else 0.0)))
        log.info("Autopilot: 試していない位置を優先します (先頭 %s, 既存 %d 手)",
                 candidates[0]["mut_str"], candidates[0]["position_tries"])
        return candidates
    candidates.sort(key=lambda c: (c["llr"] is not None,
                                   (c["llr"] if c["llr"] is not None else 0.0) + c["history_prior"],
                                   -float(c.get("mpnn") or 0.0)),
                    reverse=True)
    worst = [c for c in candidates if c["history_prior"] <= -1.0]
    if worst:
        log.info("Autopilot: 過去に平均 -1.0 以下だった位置に触る %d 件を後ろに回しました (%s)",
                 len(worst), ", ".join(c["mut_str"] for c in worst[:4]))
    return candidates


def _submit_candidates(jobs: JobManager, db: Database, parent: dict[str, Any],
                       candidates: list[dict[str, Any]], limit: int) -> int:
    """Queue up to *limit* untried candidates as children of *parent*."""
    tried = _tried_mutations(db, parent["id"])
    fresh = [c for c in candidates if c["mut_str"] not in tried]
    if len(fresh) < len(candidates):
        log.info("Autopilot: 試行済みの %d 件を除外しました", len(candidates) - len(fresh))
    chosen = fresh[:limit]
    if not chosen:
        return 0

    decision = governor.check(db, count=len(chosen))
    if not decision:
        log.warning("Autopilot: バリアント %d 件の投入を見送りました — %s",
                    len(chosen), governor.describe_reason(decision))
        return 0

    parent_depth = int((parent.get("spec") or {}).get("autopilot_depth", 0) or 0)
    submitted = 0
    for c in chosen:
        try:
            jobs.submit("predict", c["spec"], c["spec"]["name"],
                        parent_id=parent["id"], origin="autopilot_variant")
            submitted += 1
            score = "スコアなし" if c["llr"] is None else f"ESM {c['llr']:+.2f}"
            log.info("Autopilot: バリアント '%s' を投入しました (%s, depth %d→%d)",
                     c["mut_str"], score, parent_depth, parent_depth + 1)
        except Exception as exc:
            log.warning("Autopilot: バリアント '%s' の投入に失敗しました: %s", c["mut_str"], exc)
    if submitted:
        # New children change what "already measured" means on the next cycle.
        history.invalidate()
        jobs.touch()
    return submitted


_MAX_REANALYSIS_PER_CYCLE = 3


def _climb_from_peak(jobs: JobManager, db: Database, limit: int,
                     spent: set[str] | None = None) -> int:
    """A non-improving attempt is abandoned; try the next candidate from the branch point.

    That job's analysis already produced a ranked candidate list, so this costs no LLM
    time — it just spends the next-best idea. Only when that list is exhausted is the
    job re-analysed for fresh ones.

    ``spent`` bounds that re-analysis. Re-analysing calls back into here when it again
    finds nothing to submit, and with a strict protected-residue mask "nothing to submit"
    can be the permanent answer for a given parent — without the bound the loop re-asks
    the same question of the same job forever, at full GPU, submitting nothing.
    """
    spent = set() if spent is None else spent
    peak = _branch_point(db, exclude=spent)
    if peak is None:
        if spent:
            log.info("Autopilot: 候補を出せる分岐点が尽きました (%d 件試行)", len(spent))
        return 0
    results = _load_analysis(peak["id"])
    if results:
        n = _submit_candidates(jobs, db, peak, _ranked_candidates(peak, results, db), limit)
        if n:
            log.info("Autopilot: 頂点 %s から分岐しました", (peak.get("title") or "")[:40])
            return n
    if len(spent) >= _MAX_REANALYSIS_PER_CYCLE:
        log.warning("Autopilot: %d 件を解析し直しても投入できる候補が出ませんでした。"
                    "保護残基が厳しすぎないか確認してください", len(spent))
        return 0
    log.info("Autopilot: 頂点 %s の候補が尽きたため解析し直します", (peak.get("title") or "")[:40])
    spent.add(peak["id"])
    (jobs_dir() / peak["id"] / "autopilot.json").unlink(missing_ok=True)
    _analyze(peak, db, jobs, spent=spent)
    return 0


def _auto_submit_variants(job: dict[str, Any], results: dict[str, Any], jobs: JobManager,
                          db: Database, spent: set[str] | None = None) -> None:
    """Queue the *best few* mutation proposals as predict jobs — not all of them.

    Four independent brakes, because an unbounded fan-out is what turns this loop
    from useful into a queue that never drains:

    1. **depth** — how many generations the chain may go (``autopilot_max_depth``;
       0 means unlimited, for running until switched off).
    2. **ESM-2 floor** — proposals the language model scores below
       ``autopilot_min_esm_llr`` are dropped without spending a Boltz run on them.
    3. **fan-out cap** — the survivors are ranked by ESM-2 score and only the top
       ``autopilot_max_variants_per_job`` are kept.
    4. **governor** — the daily autonomous budget and the free-disk floor can veto
       the whole batch.
    """
    spec = job.get("spec") or {}
    depth = int(spec.get("autopilot_depth", 0) or 0)
    if governor.depth_exhausted(depth):
        log.info("Autopilot: %s は depth=%d で上限に達したためここで打ち切ります",
                 job["id"][:8], depth)
        return

    if not governor.enabled():
        log.info("Autopilot: 無効化されているためバリアントを投入しません")
        return

    limit = governor.max_variants_per_job()
    if limit <= 0:
        log.info("Autopilot: 1ジョブあたりの投入上限が 0 のためバリアントを投入しません")
        return

    if get_settings().autopilot_strategy == "walk":
        # Continue from whatever just finished, improvement or not.
        _submit_candidates(jobs, db, job, _ranked_candidates(job, results, db), limit)
        return

    # Hill climb: the chain only advances from the best result so far. Continuing from
    # the newest one instead is what let a 51-generation run drift 26 mutations away and
    # end up below where it started.
    peak = _branch_point(db)
    if peak is not None and peak["id"] != job["id"]:
        log.info("Autopilot: %s は %s を超えなかったので、%s から次の候補を試します",
                 (job.get("title") or job["id"])[:34], (peak.get("title") or "")[:34],
                 (peak.get("title") or "")[:34])
        _climb_from_peak(jobs, db, limit, spent)
        return

    if _submit_candidates(jobs, db, job, _ranked_candidates(job, results, db), limit) == 0:
        # This job is the peak but has nothing new to try; fall through to a re-analysis.
        _climb_from_peak(jobs, db, limit, spent)
