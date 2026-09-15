"""SQLite persistence: jobs, the component library, chat threads, the LLM call log
and the search history."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from .config import app_home

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    title TEXT NOT NULL,
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL,
    spec TEXT NOT NULL,
    result TEXT,
    error TEXT,
    phase TEXT,
    parent_id TEXT,
    origin TEXT NOT NULL DEFAULT 'user',
    starred INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS jobs_created ON jobs(created_at DESC);
-- children_of() and the lineage view walk by parent; without this they scan the table.
CREATE INDEX IF NOT EXISTS jobs_parent ON jobs(parent_id);
-- ranking_rows() and the leaderboard read every finished prediction on each pass.
CREATE INDEX IF NOT EXISTS jobs_kind_status ON jobs(kind, status);

CREATE TABLE IF NOT EXISTS library (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    name TEXT NOT NULL,
    data TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS threads (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    messages TEXT NOT NULL
);

-- Every exchange with the model, the autopilot's included. The chat threads only hold
-- what a person asked for, so an overnight run used to leave no trace of the prompts that
-- produced its proposals — exactly the material needed to work out why the model misses.
CREATE TABLE IF NOT EXISTS llm_calls (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    model TEXT NOT NULL,
    mode TEXT NOT NULL,
    origin TEXT NOT NULL,
    job_id TEXT,
    thread_id TEXT,
    messages TEXT NOT NULL,
    raw TEXT NOT NULL,
    reply TEXT NOT NULL,
    proposals TEXT NOT NULL,
    reply_issues TEXT NOT NULL,
    corrected INTEGER NOT NULL DEFAULT 0,
    elapsed_sec REAL NOT NULL,
    error TEXT
);
CREATE INDEX IF NOT EXISTS llm_calls_created ON llm_calls(created_at DESC);

-- What was searched for and what came back, so a component found once can be found again
-- without remembering the accession.
CREATE TABLE IF NOT EXISTS searches (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    source TEXT NOT NULL,
    query TEXT NOT NULL,
    hits INTEGER NOT NULL,
    top TEXT NOT NULL,
    picked TEXT
);
CREATE INDEX IF NOT EXISTS searches_created ON searches(created_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS searches_key ON searches(source, query);

CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

JOB_STATUSES = ("queued", "running", "succeeded", "failed", "cancelled")


def new_id(prefix: str) -> str:
    return f"{prefix}_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}"


class Database:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (app_home() / "oritatami.sqlite3")
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=15)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=15000")
        self._conn.executescript(SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Additive column migrations for databases created by older versions."""
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(jobs)")}
        if "error_kind" not in cols:
            self._conn.execute("ALTER TABLE jobs ADD COLUMN error_kind TEXT")

    # ---- jobs -----------------------------------------------------------
    def insert_job(self, *, kind: str, title: str, spec: dict[str, Any], parent_id: str | None,
                   origin: str) -> dict[str, Any]:
        job_id = new_id("job")
        with self._lock:
            self._conn.execute(
                "INSERT INTO jobs (id, kind, status, title, created_at, spec, parent_id, origin, phase)"
                " VALUES (?, ?, 'queued', ?, ?, ?, ?, ?, 'queued')",
                (job_id, kind, title, time.time(), json.dumps(spec, ensure_ascii=False), parent_id, origin),
            )
            self._conn.commit()
        job = self.get_job(job_id)
        assert job is not None
        return job

    def update_job(self, job_id: str, **values: Any) -> None:
        if not values:
            return
        cols = []
        params: list[Any] = []
        for key, value in values.items():
            if key not in {"status", "started_at", "finished_at", "result", "error", "error_kind", "phase", "title",
                           "starred"}:
                raise ValueError(f"cannot update job column {key}")
            if key == "status" and value not in JOB_STATUSES:
                raise ValueError(f"bad status {value}")
            if key == "result" and value is not None:
                value = json.dumps(value, ensure_ascii=False)
            cols.append(f"{key} = ?")
            params.append(value)
        params.append(job_id)
        with self._lock:
            self._conn.execute(f"UPDATE jobs SET {', '.join(cols)} WHERE id = ?", params)
            self._conn.commit()

    @staticmethod
    def _job_row(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["spec"] = json.loads(d["spec"])
        d["result"] = json.loads(d["result"]) if d["result"] else None
        d["starred"] = bool(d["starred"])
        return d

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._job_row(row) if row else None

    def list_jobs(self, limit: int = 200) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._job_row(r) for r in rows]

    def finished_predictions(self, limit: int = 60) -> list[dict[str, Any]]:
        """Recent successful predictions (newest first), for runtime estimates."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM jobs WHERE kind = 'predict' AND status = 'succeeded' ORDER BY finished_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._job_row(r) for r in rows]

    def count_jobs_since(self, since: float, origins: tuple[str, ...], kind: str = "predict") -> int:
        """How many jobs of *kind* with one of *origins* were created after *since* (unix seconds)."""
        if not origins:
            return 0
        marks = ",".join("?" for _ in origins)
        with self._lock:
            row = self._conn.execute(
                f"SELECT COUNT(*) AS n FROM jobs WHERE kind = ? AND created_at >= ?"
                f" AND origin IN ({marks})",
                (kind, since, *origins),
            ).fetchone()
        return int(row["n"]) if row else 0

    def succeeded_predictions(self, limit: int | None = 500) -> list[dict[str, Any]]:
        """All successful predictions, newest first — the leaderboard and lineage source.

        ``limit=None`` returns every one. Anything that RANKS predictions must ask for
        all of them: a newest-first window silently drops the older rows, and the best
        result is usually an old one — the run that found ubiquitin's peak had it at
        row 1 of 200 within a day.
        """
        sql = ("SELECT * FROM jobs WHERE kind = 'predict' AND status = 'succeeded'"
               " ORDER BY created_at DESC")
        with self._lock:
            if limit is None:
                rows = self._conn.execute(sql).fetchall()
            else:
                rows = self._conn.execute(sql + " LIMIT ?", (limit,)).fetchall()
        return [self._job_row(r) for r in rows]

    def ranking_rows(self) -> list[dict[str, Any]]:
        """Every finished prediction, carrying only what ranking needs.

        A full result row is dominated by the PAE matrix — 76x76 floats for a small
        protein — which ranking never looks at. Pulling just the pLDDT and confidence
        halves the query and skips parsing the rest: at 550 predictions that is 0.43 s
        down to 0.2 s per pass, and the loop makes several passes per generation.
        """
        sql = ("SELECT id, title, parent_id, origin, spec, finished_at,"
               "       json_extract(result, '$.models[0].plddt')      AS plddt,"
               "       json_extract(result, '$.models[0].confidence') AS confidence,"
               # Two small scalars-and-objects more: without them every ranked row looks
               # like it ran under unknown conditions, and scores computed with and
               # without an MSA (7 pLDDT apart on this machine) rank in one column.
               "       json_extract(result, '$.normalized_spec.params') AS params,"
               "       json_extract(result, '$.msa')                    AS msa"
               " FROM jobs WHERE kind = 'predict' AND status = 'succeeded'"
               " ORDER BY created_at DESC")
        with self._lock:
            rows = self._conn.execute(sql).fetchall()
        out = []
        for r in rows:
            try:
                spec = json.loads(r["spec"]) if r["spec"] else {}
                model = {"plddt": json.loads(r["plddt"]) if r["plddt"] else {},
                         "confidence": json.loads(r["confidence"]) if r["confidence"] else {}}
                params = json.loads(r["params"]) if r["params"] else {}
                msa = json.loads(r["msa"]) if r["msa"] else {}
            except (TypeError, json.JSONDecodeError):
                continue
            out.append({"id": r["id"], "title": r["title"], "parent_id": r["parent_id"],
                        "origin": r["origin"], "spec": spec, "finished_at": r["finished_at"],
                        "result": {"models": [model], "msa": msa,
                                   "normalized_spec": {"params": params}}})
        return out

    def children_of(self, parent_id: str) -> list[dict[str, Any]]:
        """Every job derived from this one, however long ago."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM jobs WHERE parent_id = ? ORDER BY created_at", (parent_id,)
            ).fetchall()
        return [self._job_row(r) for r in rows]

    def job_ids(self, statuses: tuple[str, ...]) -> list[str]:
        marks = ",".join("?" for _ in statuses)
        with self._lock:
            rows = self._conn.execute(f"SELECT id FROM jobs WHERE status IN ({marks})", statuses).fetchall()
        return [r["id"] for r in rows]

    def delete_job(self, job_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            self._conn.commit()

    def mark_interrupted(self) -> int:
        """Jobs left queued/running by a previous process can never finish."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE jobs SET status = 'failed', phase = 'interrupted', finished_at = ?,"
                " error = COALESCE(error, 'アプリが終了したため中断されました')"
                " WHERE status IN ('queued', 'running')",
                (time.time(),),
            )
            self._conn.commit()
            return cur.rowcount

    # ---- library --------------------------------------------------------
    def add_library(self, type_: str, name: str, data: dict[str, Any]) -> dict[str, Any]:
        item_id = new_id("lib")
        with self._lock:
            self._conn.execute(
                "INSERT INTO library (id, type, name, data, created_at) VALUES (?, ?, ?, ?, ?)",
                (item_id, type_, name, json.dumps(data, ensure_ascii=False), time.time()),
            )
            self._conn.commit()
        return {"id": item_id, "type": type_, "name": name, "data": data}

    def list_library(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM library ORDER BY created_at DESC").fetchall()
        return [{**dict(r), "data": json.loads(r["data"])} for r in rows]

    def delete_library(self, item_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM library WHERE id = ?", (item_id,))
            self._conn.commit()

    # ---- assistant threads ---------------------------------------------
    def create_thread(self, title: str) -> dict[str, Any]:
        tid = new_id("thr")
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO threads (id, title, created_at, updated_at, messages) VALUES (?, ?, ?, ?, '[]')",
                (tid, title, now, now),
            )
            self._conn.commit()
        return {"id": tid, "title": title, "created_at": now, "updated_at": now, "messages": []}

    def get_thread(self, tid: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM threads WHERE id = ?", (tid,)).fetchone()
        if not row:
            return None
        return {**dict(row), "messages": json.loads(row["messages"])}

    def list_threads(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, title, created_at, updated_at FROM threads ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def save_thread(self, tid: str, messages: list[dict[str, Any]], title: str | None = None) -> None:
        with self._lock:
            if title is None:
                self._conn.execute(
                    "UPDATE threads SET messages = ?, updated_at = ? WHERE id = ?",
                    (json.dumps(messages, ensure_ascii=False), time.time(), tid),
                )
            else:
                self._conn.execute(
                    "UPDATE threads SET messages = ?, updated_at = ?, title = ? WHERE id = ?",
                    (json.dumps(messages, ensure_ascii=False), time.time(), title, tid),
                )
            self._conn.commit()

    def delete_thread(self, tid: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM threads WHERE id = ?", (tid,))
            self._conn.commit()


    # ── key-value store ───────────────────────────────────────────────────

    # ---- llm call log ---------------------------------------------------
    def record_llm_call(self, *, model: str, mode: str, origin: str, job_id: str | None,
                        thread_id: str | None, messages: list[dict[str, Any]], raw: str,
                        reply: str, proposals: list[dict[str, Any]], reply_issues: list[str],
                        corrected: bool, elapsed_sec: float, error: str | None = None,
                        keep: int = 5000) -> str:
        call_id = new_id("llm")
        with self._lock:
            self._conn.execute(
                "INSERT INTO llm_calls (id, created_at, model, mode, origin, job_id, thread_id,"
                " messages, raw, reply, proposals, reply_issues, corrected, elapsed_sec, error)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (call_id, time.time(), model, mode, origin, job_id, thread_id,
                 json.dumps(messages, ensure_ascii=False), raw, reply,
                 json.dumps(proposals, ensure_ascii=False),
                 json.dumps(reply_issues, ensure_ascii=False),
                 1 if corrected else 0, float(elapsed_sec), error))
            if keep > 0:
                # Prompts run 10-15 KB each and the loop makes two per generation, so an
                # unbounded log outgrows the jobs table within a week.
                self._conn.execute(
                    "DELETE FROM llm_calls WHERE id NOT IN"
                    " (SELECT id FROM llm_calls ORDER BY created_at DESC LIMIT ?)", (keep,))
            self._conn.commit()
        return call_id

    def list_llm_calls(self, limit: int = 100, origin: str | None = None,
                       with_messages: bool = False) -> list[dict[str, Any]]:
        cols = ("id, created_at, model, mode, origin, job_id, thread_id, reply, proposals,"
                " reply_issues, corrected, elapsed_sec, error")
        if with_messages:
            cols += ", messages, raw"
        sql = f"SELECT {cols} FROM llm_calls"
        args: list[Any] = []
        if origin:
            sql += " WHERE origin = ?"
            args.append(origin)
        sql += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            for key in ("proposals", "reply_issues", "messages"):
                if key in d and d[key]:
                    d[key] = json.loads(d[key])
            d["corrected"] = bool(d["corrected"])
            out.append(d)
        return out

    def count_llm_calls(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM llm_calls").fetchone()[0])

    def iter_llm_calls(self):
        """Oldest first, whole rows — for writing the log out as JSONL."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM llm_calls ORDER BY created_at ASC").fetchall()
        for r in rows:
            d = dict(r)
            for key in ("messages", "proposals", "reply_issues"):
                if d.get(key):
                    d[key] = json.loads(d[key])
            d["corrected"] = bool(d["corrected"])
            yield d

    # ---- search history --------------------------------------------------
    def record_search(self, *, source: str, query: str, hits: int,
                      top: list[dict[str, Any]], keep: int = 500) -> None:
        query = (query or "").strip()
        if not query:
            return
        with self._lock:
            self._conn.execute(
                "INSERT INTO searches (id, created_at, source, query, hits, top) VALUES (?,?,?,?,?,?)"
                " ON CONFLICT(source, query) DO UPDATE SET created_at=excluded.created_at,"
                " hits=excluded.hits, top=excluded.top",
                (new_id("search"), time.time(), source, query, hits,
                 json.dumps(top[:8], ensure_ascii=False)))
            if keep > 0:
                self._conn.execute(
                    "DELETE FROM searches WHERE id NOT IN"
                    " (SELECT id FROM searches ORDER BY created_at DESC LIMIT ?)", (keep,))
            self._conn.commit()

    def note_search_pick(self, *, source: str, item_id: str, title: str) -> None:
        """Remember what was actually added, on the most recent search that offered it."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, top FROM searches WHERE source = ? ORDER BY created_at DESC LIMIT 40",
                (source,)).fetchall()
            target = None
            for r in rows:
                try:
                    listed = json.loads(r["top"])
                except (TypeError, ValueError):
                    continue
                if any(str(h.get("id")) == item_id for h in listed):
                    target = r["id"]
                    break
            picked = json.dumps({"id": item_id, "title": title, "at": time.time()},
                                ensure_ascii=False)
            if target is not None:
                self._conn.execute("UPDATE searches SET picked = ? WHERE id = ?", (picked, target))
            else:
                # Imported by typing the accession straight in: still worth keeping.
                self._conn.execute(
                    "INSERT INTO searches (id, created_at, source, query, hits, top, picked)"
                    " VALUES (?,?,?,?,?,?,?)"
                    " ON CONFLICT(source, query) DO UPDATE SET created_at=excluded.created_at,"
                    " picked=excluded.picked",
                    (new_id("search"), time.time(), source, item_id, 1,
                     json.dumps([{"id": item_id, "title": title}], ensure_ascii=False), picked))
            self._conn.commit()

    def list_searches(self, limit: int = 100, source: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM searches"
        args: list[Any] = []
        if source:
            sql += " WHERE source = ?"
            args.append(source)
        sql += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["top"] = json.loads(d["top"]) if d["top"] else []
            d["picked"] = json.loads(d["picked"]) if d["picked"] else None
            out.append(d)
        return out

    def clear_searches(self) -> int:
        with self._lock:
            n = self._conn.execute("DELETE FROM searches").rowcount
            self._conn.commit()
        return n

    def kv_get(self, key: str) -> str | None:
        """Return the value for *key*, or None if not set."""
        with self._lock:
            row = self._conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def kv_set(self, key: str, value: str) -> None:
        """Upsert *key* → *value* in the kv table."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO kv (key, value) VALUES (?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
            self._conn.commit()
