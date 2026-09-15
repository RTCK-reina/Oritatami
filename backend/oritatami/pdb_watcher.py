"""PDB Watcher: periodically fetches newly released protein structures from RCSB PDB
and submits them as predict jobs for autonomous analysis.

Polling interval: 6 hours (configurable).
Config stored in the kv table under key "pdb_watcher_config".
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from . import governor
from .db import Database
from .engines import boltz
from .jobs import JobManager

log = logging.getLogger("oritatami.pdb_watcher")

POLL_INTERVAL_SEC = 6 * 3600        # 6 hours between polls
MAX_PER_POLL      = 5               # max new entries to queue per poll
MIN_SEQ_LEN       = 50              # shortest protein (aa) to accept
MAX_SEQ_LEN       = 500             # longest protein (aa) to accept
_SEARCH_URL       = "https://search.rcsb.org/rcsbsearch/v2/query"
_FASTA_URL        = "https://www.rcsb.org/fasta/entry/{pdb_id}/download"

_FETCH_ROWS        = 200            # entries pulled per search (backlog visibility)
_MAX_SCAN_PER_POLL = 40             # cap on FASTA downloads per poll
_SEEN_CAP          = 1000           # how many processed PDB ids to remember

_stop_event = threading.Event()
_poll_lock  = threading.Lock()


# ── Public API ────────────────────────────────────────────────────────────────

def start(jobs: JobManager, db: Database) -> None:
    """Start the background PDB watcher thread. Call once from app lifespan."""
    t = threading.Thread(target=_loop, args=(jobs, db), name="pdb-watcher", daemon=True)
    t.start()
    log.info("PDB Watcher: 起動しました (ポーリング間隔 %d 時間)", POLL_INTERVAL_SEC // 3600)


def poll_now(jobs: JobManager, db: Database) -> None:
    """Trigger an immediate poll in a background thread (for the API endpoint)."""
    threading.Thread(target=_poll_once, args=(jobs, db), daemon=True, name="pdb-poll-now").start()


# ── Internal ──────────────────────────────────────────────────────────────────

def _loop(jobs: JobManager, db: Database) -> None:
    # Small delay so app finishes starting up before the first network hit
    time.sleep(10.0)
    _poll_once(jobs, db)
    while not _stop_event.wait(POLL_INTERVAL_SEC):
        _poll_once(jobs, db)


def _poll_once(jobs: JobManager, db: Database) -> None:
    # Only one poll at a time: the 6h loop and a manual poll_now() can overlap.
    if not _poll_lock.acquire(blocking=False):
        log.info("PDB Watcher: 別のポーリングが実行中のためスキップします")
        return
    try:
        _poll_once_locked(jobs, db)
    finally:
        _poll_lock.release()


def _poll_once_locked(jobs: JobManager, db: Database) -> None:
    cfg = _load_config(db)
    if not cfg.get("enabled", True):
        log.debug("PDB Watcher: 無効化されているためスキップします")
        return

    since = cfg.get("last_checked") or _default_since()
    log.info("PDB Watcher: %s 以降の新着構造を検索します", since[:10])

    try:
        entries = _fetch_new_entries(since, max_results=_FETCH_ROWS)
    except Exception as exc:
        # Do NOT advance the cursor here: a transient network failure must not
        # silently skip everything released during this window.
        log.warning("PDB Watcher: RCSB 検索に失敗しました (次回に持ち越します): %s", exc)
        return

    seen = _load_seen(db)
    seen_set = set(seen)
    candidates = [e for e in entries if e not in seen_set]
    log.info("PDB Watcher: %d 件ヒット / 未処理 %d 件", len(entries), len(candidates))

    budget  = int(cfg.get("max_per_poll", MAX_PER_POLL))
    min_len = int(cfg.get("min_seq_len", MIN_SEQ_LEN))
    max_len = int(cfg.get("max_seq_len", MAX_SEQ_LEN))

    submitted = 0
    scanned   = 0
    processed: list[str] = []

    for pdb_id in candidates:
        if submitted >= budget or scanned >= _MAX_SCAN_PER_POLL:
            break
        # The daily autonomous budget and the free-disk floor apply here too: the
        # watcher is machine-made work competing with the user's own jobs.
        decision = governor.check(db, count=1)
        if not decision:
            log.warning("PDB Watcher: 投入を停止しました — %s", governor.describe_reason(decision))
            break
        scanned += 1
        # Mark as attempted up front so a permanently unreadable entry can never
        # stall the cursor forever.
        processed.append(pdb_id)
        try:
            seq = _fetch_sequence(pdb_id)
            if seq is None:
                log.debug("PDB Watcher: %s — 配列を取得できませんでした", pdb_id)
                continue
            if not (min_len <= len(seq) <= max_len):
                log.debug("PDB Watcher: %s — 長さ %d aa は範囲外 (%d–%d)",
                          pdb_id, len(seq), min_len, max_len)
                continue
            _submit_job(jobs, pdb_id, seq, cfg)
            submitted += 1
            time.sleep(1.0)  # be polite to the API
        except Exception as exc:
            log.warning("PDB Watcher: %s のジョブ投入に失敗しました: %s", pdb_id, exc)

    if processed:
        _save_seen(db, seen + processed)

    # The cursor advances on every *successful* poll, even when the window held more
    # entries than the per-poll budget. This watcher samples the newest releases; it is
    # not an exhaustive crawler, and RCSB publishes far more per week than the daily
    # budget allows. Holding the cursor back for the remainder would freeze it forever
    # and make the search window grow without bound. A failed search is different —
    # that path returns early above without touching the cursor, so a dropped
    # connection never silently skips a window.
    skipped = len(candidates) - scanned
    _save_config(db, {**cfg, "last_checked": _now_iso()})
    if skipped > 0:
        log.info("PDB Watcher: 今回の枠を超えた %d 件は見送りました (新しいものを優先)", skipped)

    log.info("PDB Watcher: %d 件を投入しました", submitted)


def _load_seen(db: Database) -> list[str]:
    raw = db.kv_get("pdb_watcher_seen")
    if not raw:
        return []
    try:
        v = json.loads(raw)
        return [str(x) for x in v] if isinstance(v, list) else []
    except Exception:
        return []


def _save_seen(db: Database, ids: list[str]) -> None:
    """Persist processed PDB ids, oldest → newest, capped at _SEEN_CAP."""
    db.kv_set("pdb_watcher_seen", json.dumps(ids[-_SEEN_CAP:], ensure_ascii=False))


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _default_since() -> str:
    """Yesterday at midnight UTC — used on first run."""
    d = datetime.now(UTC) - timedelta(days=1)
    return d.strftime("%Y-%m-%dT00:00:00Z")


def _fetch_new_entries(since: str, max_results: int = MAX_PER_POLL) -> list[str]:
    """Query RCSB for protein-only entries released after *since* (ISO date string)."""
    since_date = since[:10]  # RCSB accepts 'YYYY-MM-DD'
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_accession_info.initial_release_date",
                        "operator": "greater",
                        "negation": False,
                        "value": since_date,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.selected_polymer_entity_types",
                        "operator": "exact_match",
                        "negation": False,
                        "value": "Protein (only)",
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": max_results},
            "sort": [
                {"sort_by": "rcsb_accession_info.initial_release_date", "direction": "desc"}
            ],
        },
    }
    with httpx.Client(timeout=30.0) as client:
        r = client.post(_SEARCH_URL, json=query)
        r.raise_for_status()
        # RCSB answers 204 No Content (empty body) when the query matches nothing —
        # a normal "no new structures", not a failure. Parsing it as JSON would raise.
        if r.status_code == 204 or not r.content:
            return []
        payload = r.json()
    return [hit["identifier"] for hit in (payload.get("result_set") or [])]


def _fetch_sequence(pdb_id: str) -> str | None:
    """Download FASTA for a PDB entry and return the first protein sequence (uppercase)."""
    url = _FASTA_URL.format(pdb_id=pdb_id)
    with httpx.Client(timeout=20.0) as client:
        r = client.get(url)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        text = r.text

    # Parse FASTA: return sequence for the first entity only
    seq_lines: list[str] = []
    in_seq = False
    for line in text.splitlines():
        if line.startswith(">"):
            if in_seq:
                break  # second entity — stop
            in_seq = True
        elif in_seq:
            seq_lines.append(line.strip())

    seq = "".join(seq_lines).upper()
    # Accept only standard amino-acid letters (X allowed for unknown)
    if not seq or not re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWYX]+", seq):
        return None
    return seq


def _submit_job(jobs: JobManager, pdb_id: str, sequence: str, cfg: dict[str, Any]) -> None:
    components = [
        {
            "type": "protein",
            "label": pdb_id,
            "sequence": sequence,
            "chains": ["A"],
            "msa": "server",
        }
    ]
    workbench_components = [
        {
            "type": "protein",
            "label": pdb_id,
            "sequence": sequence,
            "chains": ["A"],
        }
    ]
    spec: dict[str, Any] = {
        "name": f"PDB {pdb_id}",
        "workbench_name": f"PDB {pdb_id}",
        "autopilot_source": "pdb_watch",
        "autopilot_depth": 0,
        "workbench": {
            "name": f"PDB {pdb_id}",
            "components": workbench_components,
        },
        "components": components,
    }
    boltz.normalize_spec(spec)
    jobs.submit("predict", spec, f"PDB {pdb_id}", origin="pdb_watch")
    log.info("PDB Watcher: %s (%d aa) をキューに追加しました", pdb_id, len(sequence))


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(db: Database) -> dict[str, Any]:
    raw = db.kv_get("pdb_watcher_config")
    if raw:
        try:
            return json.loads(raw)
        except Exception:
            pass
    return {
        "enabled": True,
        "max_per_poll": MAX_PER_POLL,
        "min_seq_len": MIN_SEQ_LEN,
        "max_seq_len": MAX_SEQ_LEN,
        "last_checked": None,
    }


def save_config(db: Database, cfg: dict[str, Any]) -> None:
    db.kv_set("pdb_watcher_config", json.dumps(cfg, ensure_ascii=False))


# private aliases for internal use
_load_config = load_config
_save_config  = save_config
