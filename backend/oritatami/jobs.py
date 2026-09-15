"""Job queues.

Jobs run in lanes, one job at a time per lane: ``predict`` (Boltz, needs most of the
unified memory) and ``esm`` (mutation scans and sequence refinement, ~3 GB). A scan
can therefore finish while a long prediction is running. Each job owns
``jobs/<id>/`` with a ``job.log``. State lives in SQLite; live phase/progress is also
kept in memory for cheap polling.
"""

from __future__ import annotations

import copy
import logging
import queue
import re
import shutil
import threading
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import llm
from .config import get_settings, jobs_dir
from .db import Database

log = logging.getLogger("oritatami.jobs")

JobHandler = Callable[["JobContext"], dict[str, Any]]

# Failures worth retrying on their own: the ColabFold MSA server and general
# connectivity are flaky, and an unattended run would otherwise sit dead until
# the user notices in the morning. Everything else (bad input, OOM, a missing
# dependency) would fail again identically, so it is left alone.
# Failures that a later attempt can plausibly survive. "msa" is the ColabFold server
# timing out or rate-limiting — the single failure in an 857-prediction overnight run was
# exactly that, and auto-retry sat it out because only "network" was listed. "download"
# is the Boltz weights mirror, same story. Everything else (oom, input, error) will fail
# again the same way, so retrying just burns the GPU.
RETRYABLE_ERROR_KINDS = ("network", "msa", "download")
RETRY_BASE_DELAY_SEC = 60.0


class JobCancelled(Exception):
    pass


class JobContext:
    def __init__(self, manager: JobManager, job: dict[str, Any]) -> None:
        self.manager = manager
        self.job = job
        self.dir = jobs_dir() / job["id"]
        self.dir.mkdir(parents=True, exist_ok=True)
        self._log = (self.dir / "job.log").open("a", encoding="utf-8")

    def log(self, line: str) -> None:
        self._log.write(f"{time.strftime('%H:%M:%S')} {line}\n")
        self._log.flush()

    def set_phase(self, phase: str, label: str, progress: float | None = None) -> None:
        self.manager._set_live(self.job["id"], phase=phase, label=label, progress=progress)
        self.manager.db.update_job(self.job["id"], phase=phase)
        self.log(f"[phase] {phase}: {label}")

    def report(self, values: dict[str, Any]) -> None:
        """Publish live detail (memory, swap) for the UI. Not persisted: it dies with the run."""
        self.manager._set_live(self.job["id"], **values)

    def cancelled(self) -> bool:
        return self.manager._cancel_flags.get(self.job["id"], threading.Event()).is_set()

    def close(self) -> None:
        self._log.close()


class JobManager:
    def __init__(self, db: Database) -> None:
        self.db = db
        self.handlers: dict[str, JobHandler] = {}
        self.lanes: dict[str, str] = {}
        self._queues: dict[str, queue.Queue[str]] = {}
        self._threads: list[threading.Thread] = []
        self._cancel_flags: dict[str, threading.Event] = {}
        self._live: dict[str, dict[str, Any]] = {}
        self._live_lock = threading.Lock()
        self._current: dict[str, str | None] = {}
        self._stop = threading.Event()
        self._completion_listeners: list[Callable[[dict[str, Any]], None]] = []
        self._retry_timers: list[threading.Timer] = []
        # Monotonic change counter for long-polling clients (/api/jobs/changes).
        self._rev = 1
        self._rev_cond = threading.Condition()

    def register(self, kind: str, handler: JobHandler, lane: str) -> None:
        self.handlers[kind] = handler
        self.lanes[kind] = lane
        self._queues.setdefault(lane, queue.Queue())
        self._current.setdefault(lane, None)

    def start(self) -> None:
        interrupted = self.db.mark_interrupted()
        if interrupted:
            log.warning("前回終了時に未完了だったジョブ %d 件を失敗扱いにしました", interrupted)
        for lane in self._queues:
            t = threading.Thread(target=self._run, args=(lane,), name=f"jobs-{lane}", daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self._stop.set()
        for t in list(self._retry_timers):
            t.cancel()
        self._retry_timers.clear()
        for job_id in list(self._current.values()):
            if job_id:
                self.cancel(job_id)
        with self._rev_cond:
            self._rev_cond.notify_all()


    def on_complete(self, fn: Callable[[dict[str, Any]], None]) -> None:
        """Register a callback called (in a background thread) whenever any job succeeds."""
        self._completion_listeners.append(fn)

    def _notify_complete(self, job_id: str) -> None:
        if not self._completion_listeners:
            return
        job = self.describe(self.db.get_job(job_id))
        for fn in list(self._completion_listeners):
            threading.Thread(target=fn, args=(job,), daemon=True,
                             name=f"complete-cb-{job_id[:8]}").start()

    # ---- change notification ------------------------------------------------
    @property
    def revision(self) -> int:
        return self._rev

    def touch(self) -> None:
        with self._rev_cond:
            self._rev += 1
            self._rev_cond.notify_all()

    def wait_for_change(self, since: int, timeout: float) -> int:
        """Block until the revision differs from `since` (or timeout / shutdown). Returns the current revision."""
        deadline = time.monotonic() + max(0.0, timeout)
        with self._rev_cond:
            while self._rev == since and not self._stop.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._rev_cond.wait(remaining)
            return self._rev

    def active_count(self) -> int:
        return sum(1 for j in self.db.list_jobs() if j["status"] in ("queued", "running"))

    # ---- public API ------------------------------------------------------
    def submit(self, kind: str, spec: dict[str, Any], title: str, *, parent_id: str | None = None,
               origin: str = "user") -> dict[str, Any]:
        if kind not in self.handlers:
            raise ValueError(f"未知のジョブ種別: {kind}")
        job = self.db.insert_job(kind=kind, title=title, spec=spec, parent_id=parent_id, origin=origin)
        self._cancel_flags[job["id"]] = threading.Event()
        self._queues[self.lanes[kind]].put(job["id"])
        self.touch()
        return self.describe(job)

    def cancel(self, job_id: str) -> dict[str, Any]:
        job = self.db.get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        if job["status"] == "queued":
            self.db.update_job(job_id, status="cancelled", phase="cancelled", finished_at=time.time())
            # Drop it from the lane as well. Leaving it in meant the worker woke for a job it
            # would only skip, and — the visible part — every still-queued job behind it counted
            # the cancelled ones in its "N 件待ち", so after a few cancellations the queue
            # positions the UI showed were fiction.
            self._forget(self.lanes.get(job["kind"]), [job_id])
        if job["status"] in ("queued", "running"):
            self._cancel_flags.setdefault(job_id, threading.Event()).set()
        self.touch()
        return self.describe(self.db.get_job(job_id))

    def cancel_batch(self, kind: str | None = None, *, include_running: bool = False) -> list[str]:
        """Cancel everything waiting — and, with ``include_running``, whatever is going too.

        One LLM proposal round can queue twenty variants of a sequence that turns out not to
        fit in memory; cancelling them one at a time is twenty round trips for a decision the
        user already made once. ``include_running`` is the other half of that: having decided
        the whole batch was wrong, leaving the one job that happens to be mid-flight — the
        longest of them, usually — is not what "stop" means to the person pressing it.

        A running job is only flagged here. It is the handler that notices, kills Boltz and
        writes the final status, so that a job is never recorded as cancelled while its
        process is still holding the GPU.
        """
        done: list[str] = []
        queued: list[str] = []
        for job in self.db.list_jobs():
            if kind and job["kind"] != kind:
                continue
            if job["status"] == "queued":
                self.db.update_job(job["id"], status="cancelled", phase="cancelled",
                                   finished_at=time.time())
                queued.append(job["id"])
            elif job["status"] == "running" and include_running:
                pass
            else:
                continue
            self._cancel_flags.setdefault(job["id"], threading.Event()).set()
            done.append(job["id"])
        for lane in set(self.lanes.values()):
            self._forget(lane, queued)
        if done:
            self.touch()
        return done

    def cancel_queued(self, kind: str | None = None) -> list[str]:
        """Waiting jobs only. Kept because "stop the queue" and "stop everything" are
        different decisions and the UI offers them separately."""
        return self.cancel_batch(kind, include_running=False)

    def reorder(self, job_id: str, action: str) -> dict[str, Any]:
        """Move a waiting job within its lane: top / up / down / bottom."""
        if action not in ("top", "up", "down", "bottom"):
            raise ValueError(f"未知の並べ替え: {action}")
        job = self.db.get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        if job["status"] != "queued":
            raise ValueError("待機中のジョブだけ並べ替えられます")
        lane = self.lanes.get(job["kind"])
        if lane is None:
            raise ValueError(f"この種別は並べ替えられません: {job['kind']}")
        q = self._queues[lane]
        with q.mutex:
            items = list(q.queue)
            if job_id not in items:
                raise ValueError("このジョブはもう順番待ちの列にありません")
            index = items.index(job_id)
            items.pop(index)
            target = {"top": 0, "bottom": len(items),
                      "up": max(0, index - 1), "down": min(len(items), index + 1)}[action]
            items.insert(target, job_id)
            q.queue.clear()
            q.queue.extend(items)
        self.touch()
        return self.describe(self.db.get_job(job_id))

    def _forget(self, lane: str | None, job_ids: list[str]) -> None:
        """Remove ids from a lane's waiting list without disturbing the queue's accounting."""
        if not lane or not job_ids or lane not in self._queues:
            return
        drop = set(job_ids)
        q = self._queues[lane]
        with q.mutex:
            keep = [i for i in q.queue if i not in drop]
            removed = len(q.queue) - len(keep)
            if not removed:
                return
            q.queue.clear()
            q.queue.extend(keep)
            # Queue.get() holds one unfinished_tasks / not_empty slot per put(); take back the
            # ones whose job will never be handed out, or join() would never return and the
            # worker would block on an empty deque it was told is non-empty.
            for _ in range(removed):
                q.unfinished_tasks = max(0, q.unfinished_tasks - 1)
            if not q.queue:
                q.all_tasks_done.notify_all()

    def delete(self, job_id: str) -> None:
        job = self.db.get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        if job["status"] in ("queued", "running"):
            raise ValueError("実行中・待機中のジョブは削除できません。先にキャンセルしてください")
        self.db.delete_job(job_id)
        with self._live_lock:
            self._live.pop(job_id, None)
        self._cancel_flags.pop(job_id, None)
        job_path = jobs_dir() / job_id
        if job_path.exists():
            shutil.rmtree(job_path, ignore_errors=True)
        self.touch()

    def live(self, job_id: str) -> dict[str, Any]:
        """Live phase/progress for one job, without loading the row."""
        with self._live_lock:
            return dict(self._live.get(job_id, {}))

    def describe(self, job: dict[str, Any] | None) -> dict[str, Any]:
        if job is None:
            raise KeyError("job")
        with self._live_lock:
            live = dict(self._live.get(job["id"], {}))
        out = dict(job)
        out["live"] = live
        if job["status"] == "queued" and job["kind"] in self.lanes:
            out["queue_position"] = self._queue_position(self.lanes[job["kind"]], job["id"])
        return out

    def list(self, limit: int = 200) -> list[dict[str, Any]]:
        return [self.describe(j) for j in self.db.list_jobs(limit=limit)]

    def log_tail(self, job_id: str, lines: int = 200) -> str:
        parts = []
        for name in ("job.log", "boltz.log"):
            path = jobs_dir() / job_id / name
            if path.exists():
                text = path.read_text("utf-8", errors="replace").replace("\r", "\n")
                parts.append(f"==== {name} ====\n" + "\n".join(
                    [ln for ln in text.splitlines() if ln.strip()][-lines:]))
        return "\n".join(parts)

    # ---- worker ------------------------------------------------------------
    def _queue_position(self, lane: str, job_id: str) -> int:
        q = self._queues[lane]
        with q.mutex:
            items = list(q.queue)
        return items.index(job_id) + 1 if job_id in items else 0

    def _predict_estimate(self, job: dict[str, Any]) -> float | None:
        """Seconds this prediction was expected to take, frozen at the moment it started.

        A running Boltz job has no progress to show: its own bar reads 1/1 for the whole
        diffusion phase, so from the outside a 40-minute job and a wedged one look the same.
        Elapsed time against what this machine did on comparable jobs is the only thing that
        separates them, and it has to be captured at the start — recomputing it later would
        move as finished jobs change the fit, and a target that drifts is worse than none.
        """
        from . import estimate as est
        from .engines import boltz

        try:
            spec = boltz.normalize_spec(job["spec"])
            server = any(c["type"] == "protein" and c.get("msa") == "server" for c in spec["components"])
            reusable = (server and get_settings().reuse_msa_for_variants
                        and boltz.find_reusable_msa(spec) is not None)
            return float(est.estimate(spec, server and not reusable, self.db.finished_predictions())["seconds"])
        except Exception:
            # An estimate is a label on a progress line, never a reason to fail a job that
            # would otherwise run. Logged rather than swallowed so a broken estimator is
            # visible in the log instead of silently showing nothing forever.
            log.warning("実行時間の見積もりを計算できませんでした (job %s)", job["id"], exc_info=True)
            return None

    def _set_live(self, job_id: str, **values: Any) -> None:
        with self._live_lock:
            self._live.setdefault(job_id, {}).update(values, updated_at=time.time())
        self.touch()

    def _run(self, lane: str) -> None:
        q = self._queues[lane]
        while not self._stop.is_set():
            try:
                job_id = q.get(timeout=1.0)
            except queue.Empty:
                continue
            job = self.db.get_job(job_id)
            if job is None or job["status"] != "queued":
                continue
            flag = self._cancel_flags.setdefault(job_id, threading.Event())
            if flag.is_set():
                continue
            self._current[lane] = job_id
            ctx = JobContext(self, job)
            started = time.time()
            self.db.update_job(job_id, status="running", started_at=started, phase="starting")
            self._set_live(job_id, phase="starting", label="開始", progress=None, started_at=started,
                           estimate_sec=self._predict_estimate(job) if job["kind"] == "predict" else None)
            try:
                if lane == "predict":
                    # Boltz needs most of unified memory (13 GB measured for a 260-residue
                    # complex with affinity); release the idle chat and ESM-2 models first.
                    # The flag keeps them from being pulled straight back in: while it is set
                    # every LLM request asks Ollama to drop the model as soon as it answers,
                    # instead of parking it in unified memory for the 15-minute keep_alive.
                    llm.set_heavy(True)
                    llm.unload_model()
                    from .engines import esm

                    if esm.unload_if_unused():
                        ctx.log("[oritatami] ESM-2 をメモリから解放しました")
                    # Even when the model has to stay (a scan is running, or it was never
                    # loaded), torch's MPS pool can be holding blocks nobody is using. Metal
                    # budgets its working set per process, so what this one keeps is taken
                    # out of what Boltz can have.
                    freed = esm.release_cache()
                    if freed >= 0.05:
                        ctx.log(f"[oritatami] Metal のキャッシュを {freed:.1f} GB 解放しました")
                result = self.handlers[job["kind"]](ctx)
                if ctx.cancelled():
                    raise JobCancelled()
                self.db.update_job(job_id, status="succeeded", result=result, phase="done",
                                   finished_at=time.time())
                self._set_live(job_id, phase="done", label="完了", progress=1.0)
                ctx.log(f"[done] {time.time() - started:.1f}s")
                self._notify_complete(job_id)
            except JobCancelled:
                self._finish_cancelled(ctx)
            except Exception as exc:
                from .engines.boltz import Cancelled

                if isinstance(exc, Cancelled) or ctx.cancelled():
                    self._finish_cancelled(ctx)
                else:
                    ctx.log("[error] " + traceback.format_exc())
                    log.error("job %s failed: %s", job_id, exc)
                    kind = getattr(exc, "kind", None) or classify_exception(exc)
                    self.db.update_job(job_id, status="failed", error=str(exc), error_kind=kind, phase="failed",
                                       finished_at=time.time())
                    self._set_live(job_id, phase="failed", label="失敗")
                    self._maybe_retry(job, kind)
            finally:
                if lane == "predict":
                    llm.set_heavy(False)
                from .engines import esm as _esm

                _esm.release_cache()          # give the blocks back between jobs, not at exit
                from . import system as _system

                _system.record_footprint()    # one syscall; this is how fragmentation is seen
                ctx.close()
                self._current[lane] = None

    def _maybe_retry(self, job: dict[str, Any], error_kind: str) -> None:
        """Re-queue a transiently failed job after an exponential backoff.

        A retry is a *new* job carrying ``_retry_attempt`` in its spec, so the failed
        attempt stays visible in the history instead of being silently overwritten.
        """
        s = get_settings()
        if not s.job_auto_retry or error_kind not in RETRYABLE_ERROR_KINDS:
            return
        max_retries = int(s.job_max_retries)
        spec = job.get("spec") or {}
        attempt = int(spec.get("_retry_attempt", 0) or 0)
        if attempt >= max_retries:
            log.info("job %s: 再試行上限 (%d 回) に達したため諦めます", job["id"][:8], max_retries)
            return

        next_attempt = attempt + 1
        delay = RETRY_BASE_DELAY_SEC * (2 ** attempt)
        new_spec = {**copy.deepcopy(spec), "_retry_attempt": next_attempt}
        base_title = re.sub(r" \(自動再試行 \d+/\d+\)$", "", job["title"])
        title = f"{base_title} (自動再試行 {next_attempt}/{max_retries})"

        def _fire() -> None:
            if self._stop.is_set():
                return
            try:
                self.submit(job["kind"], new_spec, title,
                            parent_id=job.get("parent_id"), origin=job.get("origin") or "user")
                log.info("job %s: 自動再試行を投入しました (%d/%d)", job["id"][:8], next_attempt, max_retries)
            except Exception as exc:
                log.warning("job %s の自動再試行を投入できませんでした: %s", job["id"][:8], exc)

        timer = threading.Timer(delay, _fire)
        timer.daemon = True
        timer.name = f"retry-{job['id'][:8]}"
        timer.start()
        with self._live_lock:
            self._retry_timers = [t for t in self._retry_timers if t.is_alive()]
            self._retry_timers.append(timer)
        log.info("job %s: %s のため %.0f 秒後に自動再試行します (%d/%d)",
                 job["id"][:8], error_kind, delay, next_attempt, max_retries)

    def _finish_cancelled(self, ctx: JobContext) -> None:
        self.db.update_job(ctx.job["id"], status="cancelled", phase="cancelled", finished_at=time.time())
        self._set_live(ctx.job["id"], phase="cancelled", label="キャンセル済み")
        ctx.log("[cancelled]")


def classify_exception(exc: BaseException) -> str:
    """Coarse failure class for exceptions that do not carry their own ``kind``."""
    text = f"{exc.__class__.__name__}: {exc}".lower()
    if "out of memory" in text or isinstance(exc, MemoryError):
        return "oom"
    if "esmunavailable" in text or "no module named" in text:
        return "missing_dependency"
    if any(k in text for k in ("connecterror", "connection", "timed out", "timeout", "name resolution", "ssl")):
        return "network"
    return "error"


def job_file(job_id: str, rel: str) -> Path:
    root = (jobs_dir() / job_id).resolve()
    path = (root / rel).resolve()
    if root not in path.parents and path != root:
        raise PermissionError("job directory の外は参照できません")
    return path
