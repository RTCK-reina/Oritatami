"""Single running instance per data directory.

Two processes sharing one SQLite database and job directory would fight: the second
one marks the first one's running jobs as interrupted on startup, and both would run
Boltz at the same time on 24 GB of unified memory. The first process holds an
exclusive ``flock`` on ``<home>/instance.lock`` and writes its URL next to it; a later
launch finds the lock taken and simply opens a window on the existing server.
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import httpx

from .config import app_home


@dataclass
class InstanceLock:
    handle: IO[str]
    info_path: Path

    def publish(self, url: str, version: str) -> None:
        tmp = self.info_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"url": url, "pid": os.getpid(), "version": version, "started": time.time()}), "utf-8")
        tmp.replace(self.info_path)

    def release(self) -> None:
        try:
            self.info_path.unlink(missing_ok=True)
            fcntl.flock(self.handle, fcntl.LOCK_UN)
        finally:
            self.handle.close()


def _alive(url: str) -> bool:
    try:
        return httpx.get(url + "/api/ping", timeout=2.0).status_code == 200
    except httpx.HTTPError:
        return False


def _try_lock(path: Path) -> IO[str] | None:
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return handle
    except BlockingIOError:
        handle.close()
        return None


def acquire(wait_for_url: float = 20.0) -> InstanceLock | str:
    """Take the lock, or return the URL of the instance that already holds it.

    While waiting, the lock is retried as well: a previous instance that is still shutting
    down (quit and relaunched right away) releases it within a few seconds.
    """
    home = app_home()
    info_path = home / "instance.json"
    lock_path = home / "instance.lock"
    deadline = time.time() + wait_for_url
    while True:
        handle = _try_lock(lock_path)
        if handle is not None:
            return InstanceLock(handle=handle, info_path=info_path)
        try:
            url = json.loads(info_path.read_text("utf-8"))["url"]
            if _alive(url):
                return url
        except (OSError, ValueError, KeyError):
            pass
        if time.time() >= deadline:
            raise RuntimeError(
                "別の Oritatami が起動中ですが応答しません。アクティビティモニタで python / oritatami を終了してから"
                f"起動し直してください (ロック: {lock_path})"
            )
        time.sleep(0.4)
