"""Ollama client for the local chat model (Qwen by default)."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import httpx

from .config import app_home, get_settings

log = logging.getLogger("oritatami.llm")


class LlmError(RuntimeError):
    pass


class TruncatedError(LlmError):
    """The model hit the output budget and the reply is incomplete."""


class SafeguardError(LlmError):
    """Raised when the model refuses to answer (safety filter triggered)."""
    pass


# Patterns that indicate the model refused rather than answered
_REFUSAL_FRAGMENTS = (
    "申し訳", "お答えできません", "できかねます", "不適切", "倫理的",
    "お力になれません", "対応しておりません", "提供できません",
    "I cannot", "I'm unable", "I'm sorry, but", "I apologize",
    "I can't assist", "I can't help",
)


def _looks_like_refusal(text: str) -> bool:
    """Return True if the model output is a safety refusal rather than a real answer."""
    if len(text) > 600:          # real answers tend to be longer
        return False
    low = text.lower()
    return any(f.lower() in low for f in _REFUSAL_FRAGMENTS)


def _url(path: str) -> str:
    return get_settings().ollama_url.rstrip("/") + path


def server_up() -> bool:
    try:
        r = httpx.get(_url("/api/version"), timeout=2.0)
        return r.status_code == 200
    except httpx.HTTPError:
        return False


_serve_proc: subprocess.Popen | None = None

# The tarball the project ships with, and the one the app fetches for itself when the bundle
# was built without it. Flat archive: `ollama`, `llama-server` and the runner libraries side by
# side, which is why it is extracted into a directory of its own rather than onto PATH.
OLLAMA_URL = "https://github.com/ollama/ollama/releases/latest/download/ollama-darwin.tgz"
OLLAMA_DOWNLOAD_MB = 153


def bundled_ollama_dir() -> Path | None:
    """Where the .app keeps its copy, if this is a bundled build.

    The launcher exports the path; a source checkout has none, and then only the downloaded
    copy and whatever is installed on the machine are left.
    """
    env = os.environ.get("ORITATAMI_OLLAMA_DIR")
    if env and (Path(env) / "ollama").exists():
        return Path(env)
    return None


def _downloaded_dir() -> Path:
    return app_home() / "ollama"


def _usable(path: Path | str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    if p.exists() and p.stat().st_size > 1_000_000:
        try:
            p.chmod(p.stat().st_mode | 0o111)
        except OSError:
            pass
        return str(p)
    return None


def find_ollama() -> str | None:
    """The Ollama binary this app should use, best first.

    The bundled copy comes first on purpose: it is the version the app was built and tested
    against, and on a machine with no Ollama at all it is the only one there is. A person who
    wants their own build can name it in the settings, and the model store is ``~/.ollama``
    either way, so nothing is duplicated by preferring one over the other.
    """
    explicit = (get_settings().ollama_bin or "").strip()
    if explicit:
        return _usable(explicit) or explicit  # named by hand: report it even if it is wrong
    bundled = bundled_ollama_dir()
    if bundled:
        found = _usable(bundled / "ollama")
        if found:
            return found
    found = _usable(_downloaded_dir() / "ollama")
    if found:
        return found
    found = _usable(shutil.which("ollama"))
    if found:
        return found
    for candidate in ("/opt/homebrew/bin/ollama", "/usr/local/bin/ollama",
                      str(app_home() / "bin" / "ollama"),
                      "/Applications/Ollama.app/Contents/Resources/ollama"):
        found = _usable(candidate)
        if found:
            return found
    return None


def ollama_source(path: str | None = None) -> str:
    """Where the binary in use came from, for the setup screen."""
    path = path or find_ollama() or ""
    if not path:
        return "none"
    bundled = bundled_ollama_dir()
    if bundled and path.startswith(str(bundled)):
        return "bundled"
    if path.startswith(str(_downloaded_dir())):
        return "downloaded"
    return "system"


# ---- installing Ollama itself (background, with progress) -------------------
_install_lock = threading.Lock()
_install: dict[str, Any] = {"active": False}


def install_state() -> dict[str, Any]:
    with _install_lock:
        return dict(_install)


def start_install() -> dict[str, Any]:
    """Fetch Ollama into the app's own folder, reporting progress as it goes.

    Extraction goes through the system ``tar`` rather than Python's tarfile: the archive
    carries the binaries' code-signature metadata, and a copy that loses it is killed on sight
    by macOS. Nothing is installed system-wide — this folder belongs to the app.
    """
    with _install_lock:
        if _install.get("active"):
            return dict(_install)
        _install.clear()
        _install.update({"active": True, "status": "接続しています", "completed": 0,
                         "total": OLLAMA_DOWNLOAD_MB * 1_000_000, "error": None,
                         "started_at": time.time(), "path": None})

    def worker() -> None:
        target = _downloaded_dir()
        tmp = target.with_suffix(".part.tgz")
        try:
            if sys.platform != "darwin":
                raise LlmError("このプラットフォーム向けの自動取得には対応していません")
            target.mkdir(parents=True, exist_ok=True)
            with httpx.stream("GET", OLLAMA_URL, follow_redirects=True,
                              timeout=httpx.Timeout(900.0, connect=20.0)) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length") or 0)
                with _install_lock:
                    _install["status"] = "ダウンロード中"
                    if total:
                        _install["total"] = total
                done = 0
                with tmp.open("wb") as fh:
                    for chunk in resp.iter_bytes(chunk_size=1 << 18):
                        fh.write(chunk)
                        done += len(chunk)
                        with _install_lock:
                            _install["completed"] = done
            with _install_lock:
                _install["status"] = "展開中"
            proc = subprocess.run(["/usr/bin/tar", "xzf", str(tmp), "-C", str(target)],
                                  capture_output=True, text=True, timeout=600)
            if proc.returncode != 0:
                raise LlmError(f"展開に失敗しました: {proc.stderr.strip()[:200]}")
            binary = target / "ollama"
            if not binary.exists():
                raise LlmError("展開した中に ollama が見つかりませんでした")
            binary.chmod(binary.stat().st_mode | 0o111)
            with _install_lock:
                _install["status"] = "success"
                _install["path"] = str(binary)
            log.info("Ollama を取得しました: %s", binary)
        except Exception as exc:
            log.warning("Ollama の取得に失敗しました: %s", exc)
            with _install_lock:
                _install["error"] = str(exc)
                _install["status"] = "failed"
        finally:
            tmp.unlink(missing_ok=True)
            with _install_lock:
                _install["active"] = False
                _install["finished_at"] = time.time()

    threading.Thread(target=worker, name="ollama-install", daemon=True).start()
    return install_state()


def ensure_server(wait_sec: float = 15.0, install: bool = False) -> dict[str, Any]:
    """Start `ollama serve` if nothing answers on the configured URL. Returns what happened.

    With ``install`` the missing binary is fetched first — which takes long enough that the
    caller gets ``installing`` back and is expected to watch the progress rather than block.
    """
    global _serve_proc
    if server_up():
        return {"running": True, "started": False, "binary": find_ollama(),
                "source": ollama_source()}
    binary = find_ollama()
    if binary is None:
        state = install_state()
        if state.get("active"):
            return {"running": False, "started": False, "installing": True, "install": state}
        if install:
            log.info("Ollama が見つかりません。アプリ用にダウンロードします")
            return {"running": False, "started": False, "installing": True,
                    "install": start_install()}
        return {"running": False, "started": False,
                "error": "ollama が見つかりません。「用意する」で自動的に取得できます"}
    log_file = (app_home() / "ollama-serve.log").open("a")
    try:
        _serve_proc = subprocess.Popen([binary, "serve"], stdout=log_file, stderr=subprocess.STDOUT,
                                       start_new_session=True)
    except OSError as exc:
        # a wrong architecture, a broken download, a path that is not executable
        return {"running": False, "started": False, "binary": binary,
                "error": f"{binary} を起動できませんでした ({exc})。設定で場所を指定するか、取得し直してください"}
    deadline = time.time() + wait_sec
    while time.time() < deadline:
        if server_up():
            return {"running": True, "started": True, "binary": binary, "source": ollama_source(binary)}
        if _serve_proc.poll() is not None:
            return {"running": False, "started": False,
                    "error": f"ollama serve が終了しました (code {_serve_proc.returncode})。{app_home() / 'ollama-serve.log'} を確認してください"}
        time.sleep(0.5)
    return {"running": False, "started": True, "error": "ollama serve の起動待ちがタイムアウトしました"}


def stop_started_server() -> None:
    if _serve_proc is not None and _serve_proc.poll() is None:
        _serve_proc.terminate()


def list_models() -> list[dict[str, Any]]:
    try:
        r = httpx.get(_url("/api/tags"), timeout=5.0)
    except httpx.HTTPError as exc:
        raise LlmError(f"Ollama に接続できません ({exc})") from exc
    r.raise_for_status()
    return [{"name": m["name"], "size": m.get("size"), "family": m.get("details", {}).get("family"),
             "parameters": m.get("details", {}).get("parameter_size"),
             "quantization": m.get("details", {}).get("quantization_level")}
            for m in r.json().get("models", [])]


def status() -> dict[str, Any]:
    s = get_settings()
    binary = find_ollama()
    where = {"binary": binary, "source": ollama_source(binary), "install": install_state(),
             "download_mb": OLLAMA_DOWNLOAD_MB}
    if not server_up():
        return {"server": False, "model": s.llm_model, "model_available": False, "models": [],
                "heavy_model": s.llm_model_heavy, "heavy_available": False, **where}
    models = list_models()
    names = {m["name"] for m in models}
    wanted = s.llm_model if ":" in s.llm_model else f"{s.llm_model}:latest"
    heavy = s.llm_model_heavy.strip()
    heavy_wanted = (heavy if ":" in heavy else f"{heavy}:latest") if heavy else ""
    return {"server": True, "model": s.llm_model, "model_available": wanted in names, "models": models,
            "heavy_model": heavy, "heavy_available": bool(heavy) and heavy_wanted in names,
            "pull": pull_state(), **where}


# ---- model pull (background) ----------------------------------------------
_pull_lock = threading.Lock()
_pull: dict[str, Any] = {"active": False}


def pull_state() -> dict[str, Any]:
    with _pull_lock:
        return dict(_pull)


def start_pull(model: str) -> dict[str, Any]:
    with _pull_lock:
        if _pull.get("active"):
            return dict(_pull)
        _pull.clear()
        _pull.update({"active": True, "model": model, "status": "starting", "completed": 0, "total": 0,
                      "error": None, "started_at": time.time()})

    def worker() -> None:
        try:
            with httpx.stream("POST", _url("/api/pull"), json={"model": model, "stream": True},
                              timeout=httpx.Timeout(None, connect=10.0)) as resp:
                if resp.status_code >= 400:
                    raise LlmError(f"pull が HTTP {resp.status_code} を返しました: {resp.read().decode(errors='replace')}")
                for line in resp.iter_lines():
                    if not line:
                        continue
                    ev = json.loads(line)
                    with _pull_lock:
                        if "error" in ev:
                            _pull["error"] = ev["error"]
                        _pull["status"] = ev.get("status", _pull["status"])
                        if ev.get("total"):
                            _pull["total"] = ev["total"]
                            _pull["completed"] = ev.get("completed", 0)
            with _pull_lock:
                if not _pull.get("error"):
                    _pull["status"] = "success"
        except Exception as exc:  # surfaced to the UI through pull_state()
            with _pull_lock:
                _pull["error"] = str(exc)
        finally:
            with _pull_lock:
                _pull["active"] = False
                _pull["finished_at"] = time.time()

    threading.Thread(target=worker, name="ollama-pull", daemon=True).start()
    return pull_state()




def ensure_model(model: str | None = None, wait_sec: float = 900.0) -> dict[str, Any]:
    """Pull *model* in the background if it is not already available. Returns a status dict."""
    s = get_settings()
    model = model or s.llm_model
    if not server_up():
        return {"ok": False, "error": "Ollama サーバーが起動していません"}
    try:
        models = list_models()
    except LlmError as exc:
        return {"ok": False, "error": str(exc)}
    wanted = model if ":" in model else f"{model}:latest"
    if wanted in {m["name"] for m in models}:
        return {"ok": True, "already_present": True}
    log.info("モデル %s をダウンロードします", model)
    start_pull(model)
    deadline = time.time() + wait_sec
    while time.time() < deadline:
        state = pull_state()
        if not state.get("active"):
            ok = state.get("status") == "success"
            if not ok:
                log.warning("モデルのダウンロードに失敗しました: %s", state.get("error"))
            else:
                log.info("モデル %s のダウンロード完了", model)
            return {"ok": ok, "pull": state}
        time.sleep(3.0)
    return {"ok": False, "error": f"モデルダウンロードがタイムアウトしました ({wait_sec}s)"}

def loaded_models() -> list[str]:
    """Model names Ollama currently holds in memory."""
    try:
        r = httpx.get(_url("/api/ps"), timeout=5.0)
        r.raise_for_status()
        return [m.get("name") or m.get("model") or "" for m in r.json().get("models", [])]
    except (httpx.HTTPError, ValueError):
        return []


def _release(name: str) -> None:
    try:
        httpx.post(_url("/api/generate"), json={"model": name, "keep_alive": 0}, timeout=20.0)
    except httpx.HTTPError as exc:
        # Not fatal: the model stays loaded and Ollama frees it on its own keep_alive timer.
        log.warning("Ollama のモデル解放に失敗しました (%s): %s", name, exc)


# While a structure prediction is running, Boltz wants most of unified memory. Ollama used to
# reload the chat model behind its back — the autopilot analyses a finished job while the next
# one is already predicting — and then hold it for the 15-minute keep_alive. Under that flag
# every request asks Ollama to drop the model as soon as it has answered.
_heavy = threading.Event()


def set_heavy(active: bool) -> None:
    """Mark that a memory-hungry job (a prediction) is running."""
    if active:
        _heavy.set()
    else:
        _heavy.clear()


def heavy_active() -> bool:
    return _heavy.is_set()


def keep_alive_for(explicit: float | str | None = None) -> float | str:
    """How long Ollama should hold the model after answering."""
    if explicit is not None:
        return explicit
    return 0 if _heavy.is_set() else "15m"


def unload_model(name: str | None = None) -> None:
    """Release model memory before a heavy Boltz run.

    Everything resident, not just the configured model: once a second (larger) model can be
    picked per request, unloading only ``llm_model`` leaves the big one sitting in unified
    memory while Boltz asks for 13 GB of it.

    Ollama frees asynchronously, so this waits (briefly) for /api/ps to actually go quiet
    rather than reporting success the moment the request returns.
    """
    if not server_up():
        return
    targets = [name] if name else loaded_models() or [get_settings().llm_model]
    for m in targets:
        if m:
            _release(m)
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        still = [m for m in loaded_models() if m]
        if not still:
            return
        time.sleep(0.5)
    still = [m for m in loaded_models() if m]
    if still:
        log.warning("Ollama がまだ %s を保持しています。予測とメモリを奪い合う可能性があります",
                    ", ".join(still))


def make_room_for(name: str) -> None:
    """Evict other models before loading this one.

    A 6.6 GB model on a 15-minute keep_alive plus a 14 GB one is 20.6 GB on a 24 GB machine:
    past the GPU residency limit, so the new model lands split across CPU and GPU and decodes
    several times slower. Measured on qwen3.5:27b: 17%/83% CPU/GPU and no answer in 15 minutes.
    """
    others = [m for m in loaded_models() if m and m != name
              and m.split(":")[0] != name.split(":")[0]]
    if not others:
        return
    log.info("%s を載せる前に %s を解放します", name, ", ".join(others))
    for m in others:
        _release(m)


# Context window. Measured prompt sizes: a bare 76-residue chain is ~900 tokens, one
# UniProt-annotated 1100-residue chain ~5,200, and a four-chain annotated complex
# ~10,400 — 63 % of a 16,384 window before any reply or chat history. The LLM panel
# also replays up to twelve past messages, so a long conversation about a large complex
# could overflow, and an overflow is silent: llama.cpp drops the OLDEST tokens, which is
# the system prompt and the workbench context. The model keeps answering, without the
# rules or the sequence. Doubling the window costs 0.6 GB and, measured, nothing in
# speed (44.7 tok/s at 16k, 32k, 64k and 128k alike).
NUM_CTX = 32768

# Hard ceiling on generated tokens. Without one, output is bounded only by the window —
# and in thinking mode that is the only thing that stops a runaway: measured, 2 of 10
# trivial questions generated tokens until the window was exhausted (405 s and 409 s,
# 16,304 tokens each) and returned no answer at all. A wider window would have made
# those run proportionally longer. This cap makes such a case fail in about 90 seconds
# instead, while leaving ample room for real replies (the longest legitimate one
# measured was ~1,500 tokens).
# Raised from 4096 after a continuous run died on it: by generation 51 the chain had
# accumulated 26 mutations, the mutations-mode reply grew past the cap and was cut off
# mid-string, so the JSON would not parse and the loop ended silently. Still far below
# the window, so a thinking-mode runaway is bounded at roughly three minutes.
NUM_PREDICT = 8192

# Warn when an assembled prompt is close to the window. Rough: measured ~1.6 characters
# per token for this mix of Japanese prose and residue sequences.
_CHARS_PER_TOKEN = 1.6


# Academic context injected on retry when the model's safety filter fires.
# Placed as the first system message so it sets the frame before any other content.
_ACADEMIC_PREAMBLE = {
    "role": "system",
    "content": (
        "これは学術・教育目的の計算構造生物学ソフトウェアです。"
        "分子動力学・タンパク質工学・創薬研究の標準的な計算手法を扱います。"
        "アミノ酸置換・配位子結合・配列設計はすべて構造予測エンジンで数値検証されます。"
        "科学的な回答を提供してください。"
    ),
}


def _post_chat(payload: dict[str, Any], timeout: float) -> str:
    """Low-level Ollama /api/chat call. Returns the content string.

    Truncation is reported as its own error. Ollama simply stops at the token budget,
    leaving a half-written reply; downstream that surfaces as an unparseable JSON blob
    and reads like the model misbehaved, when the real cause is the cap.
    """
    asked = payload.get("model") or get_settings().llm_model
    try:
        r = httpx.post(_url("/api/chat"), json=payload,
                       timeout=httpx.Timeout(timeout, connect=10.0))
    except httpx.HTTPError as exc:
        raise LlmError(f"Ollama への問い合わせに失敗しました ({exc})") from exc
    if r.status_code == 404:
        raise LlmError(f"モデル {asked} が Ollama にありません。設定画面からダウンロードしてください")
    if r.status_code >= 400:
        raise LlmError(f"Ollama が HTTP {r.status_code} を返しました: {r.text[:500]}")
    data = r.json()
    if data.get("done_reason") == "length":
        raise TruncatedError(
            f"応答が出力上限 ({NUM_PREDICT} トークン) に達して途中で打ち切られました。"
            f"作業台が大きいか、思考モードが有効になっている可能性があります")
    return data.get("message", {}).get("content", "")


# ---- chat ---------------------------------------------------------------------
def chat(messages: list[dict[str, str]], *, schema: dict[str, Any] | None = None,
         temperature: float | None = None, timeout: float = 600.0,
         model: str | None = None, keep_alive: float | str | None = None) -> str:
    s = get_settings()
    name = model or s.llm_model

    def _build_payload(msgs: list[dict[str, str]]) -> dict[str, Any]:
        p: dict[str, Any] = {
            "model": name,
            "messages": msgs,
            "stream": False,
            "think": s.llm_think,
            "options": {"temperature": s.llm_temperature if temperature is None else temperature,
                        "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT},
            "keep_alive": keep_alive_for(keep_alive),
        }
        if schema is not None:
            p["format"] = schema
        return p

    if name != s.llm_model:
        make_room_for(name)
    prompt_chars = sum(len(m.get("content") or "") for m in messages)
    est_tokens = prompt_chars / _CHARS_PER_TOKEN
    if est_tokens > NUM_CTX * 0.7:
        # Not fatal, but the user should be able to find out why an answer went vague:
        # past the window the oldest messages are dropped without any error.
        log.warning("プロンプトが約 %.0f トークンとコンテキスト窓 (%d) に迫っています。"
                    "超えると古い部分 (システムプロンプトや作業台の内容) が黙って捨てられます",
                    est_tokens, NUM_CTX)
    content = _post_chat(_build_payload(messages), timeout)

    # Detect safety-filter refusals and retry once with academic framing injected.
    if content and _looks_like_refusal(content):
        log.warning("モデルが安全フィルタで拒否しました。学術コンテキストを補足してリトライします")
        # Insert the academic preamble as the very first message.
        retry_messages = [_ACADEMIC_PREAMBLE] + list(messages)
        content = _post_chat(_build_payload(retry_messages), timeout)
        if content and _looks_like_refusal(content):
            raise SafeguardError(
                "モデルの安全フィルタが応答を拒否しました。"
                "質問の言い回しを変えるか、別のモードをお試しください。"
            )

    return content
