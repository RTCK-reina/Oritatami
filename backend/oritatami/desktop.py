"""Desktop launcher: API server in a background thread + a native window (pywebview/WKWebView).

    oritatami            # native window
    oritatami --browser  # open in the default browser instead
    oritatami --serve    # server only (for development)

Only one instance runs per data directory. Launching again while it runs opens another
window (or browser tab) on the running server instead of starting a second one.
"""

from __future__ import annotations

import argparse
import logging
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

import httpx
import uvicorn

from . import __version__, instance, llm

WINDOW_KW = dict(width=1680, height=1040, min_size=(1100, 720), background_color="#0d1117", text_select=True)
LOCALIZATION = {
    "global.quitConfirmation": "実行中または待機中のジョブがあります。終了すると中断されます。終了しますか？",
    "global.quit": "終了する",
    "global.cancel": "キャンセル",
    "global.saveFile": "ファイルを保存",
}


log = logging.getLogger("oritatami.desktop")


def _mac_branding() -> None:
    """Name and icon for the menu bar and Dock (the process is a plain Python binary, not an .app bundle)."""
    if sys.platform != "darwin":
        return
    try:
        import AppKit
        from Foundation import NSBundle

        info = NSBundle.mainBundle().infoDictionary()
        if info is not None:
            info["CFBundleName"] = "Oritatami"
        icon = Path(__file__).with_name("assets") / "icon-1024.png"
        if icon.exists():
            image = AppKit.NSImage.alloc().initWithContentsOfFile_(str(icon))
            AppKit.NSApplication.sharedApplication().setApplicationIconImage_(image)
    except Exception as exc:  # cosmetic only
        log.info("メニューバー名・アイコンを設定できませんでした: %s", exc)


def _alert(title: str, message: str) -> None:
    """Show a fatal error when there is no terminal to print it to (started from Oritatami.app)."""
    if sys.platform == "darwin":
        def q(text: str) -> str:
            return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'

        subprocess.run(["osascript", "-e", f"display alert {q(title)} message {q(message)} as critical"], check=False)


def _free_port(preferred: int) -> int:
    for port in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
            return s.getsockname()[1]
    raise RuntimeError("空いているポートがありません")


def _wait_ready(url: str, server_thread: threading.Thread, timeout: float = 120.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not server_thread.is_alive():
            raise RuntimeError("API サーバーが起動直後に終了しました。~/Library/Application Support/Oritatami/oritatami.log を確認してください")
        try:
            if httpx.get(url + "/api/ping", timeout=2.0).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    raise RuntimeError("API サーバーの起動待ちがタイムアウトしました")


def _active_jobs(url: str) -> int:
    try:
        jobs = httpx.get(url + "/api/jobs", timeout=3.0).json()
        return sum(1 for j in jobs if j.get("status") in ("queued", "running"))
    except (httpx.HTTPError, ValueError):
        return 0


def _open_window(url: str, debug: bool, guard_close: bool) -> None:
    import webview

    _mac_branding()
    webview.settings["ALLOW_DOWNLOADS"] = True  # export buttons open a native save panel
    window = webview.create_window("Oritatami", url, **WINDOW_KW)
    if guard_close:
        def on_closing(window) -> None:  # pywebview passes the window only to a parameter named "window"
            # Ask only when closing would interrupt work (read at close time by pywebview).
            window.confirm_close = _active_jobs(url) > 0

        window.events.closing += on_closing
    webview.start(debug=debug, localization=LOCALIZATION)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="oritatami", description=f"Oritatami {__version__}")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--browser", action="store_true", help="既定のブラウザで開く")
    mode.add_argument("--serve", action="store_true", help="サーバーのみ起動")
    parser.add_argument("--port", type=int, default=47823)
    parser.add_argument("--debug", action="store_true", help="WebView の開発者ツールを有効にする")
    parser.add_argument("--version", action="version", version=f"Oritatami {__version__}")
    parser.add_argument("--app", action="store_true", help=argparse.SUPPRESS)  # started from Oritatami.app
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except Exception as exc:
        if args.app:
            _alert("Oritatami を起動できませんでした", f"{exc}\n\nログ: ~/Library/Logs/Oritatami/launcher.log")
        raise


def _run(args: argparse.Namespace) -> int:
    try:
        lock = instance.acquire()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        if args.app:
            _alert("Oritatami を起動できませんでした", str(exc))
        return 1
    if isinstance(lock, str):
        url = lock
        print(f"Oritatami は既に起動しています: {url}")
        if args.serve:
            return 1
        if args.browser:
            webbrowser.open(url)
        else:
            _open_window(url, args.debug, guard_close=False)
        return 0

    try:
        port = _free_port(args.port) if not args.serve else args.port
        url = f"http://127.0.0.1:{port}"

        from .app import app

        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", timeout_graceful_shutdown=3)
        server = uvicorn.Server(config)

        def _ensure_llm() -> None:
            # Model first, then the server: llama-server needs a GGUF to start, and a
            # missing one is pulled here in the background.
            if llm.ensure_server(wait_sec=60.0).get("model_missing"):
                llm.ensure_model()
                llm.ensure_server(wait_sec=60.0)

        if args.serve:
            threading.Thread(target=_ensure_llm, name="llm-start", daemon=True).start()
            threading.Thread(target=lambda: (_wait_ready(url, threading.main_thread()), lock.publish(url, __version__)),
                             daemon=True).start()
            print(f"Oritatami API: {url}")
            server.run()
            return 0

        thread = threading.Thread(target=server.run, name="uvicorn", daemon=True)
        thread.start()
        threading.Thread(target=_ensure_llm, name="llm-start", daemon=True).start()
        _wait_ready(url, thread)
        lock.publish(url, __version__)

        try:
            if args.browser:
                webbrowser.open(url)
                print(f"Oritatami: {url}  (Ctrl+C で終了)")
                while thread.is_alive():
                    time.sleep(0.5)
            else:
                _open_window(url, args.debug, guard_close=True)
        except KeyboardInterrupt:
            pass
        finally:
            server.should_exit = True
            thread.join(timeout=20)
            llm.stop_started_server()
        return 0
    finally:
        lock.release()


if __name__ == "__main__":
    sys.exit(main())
