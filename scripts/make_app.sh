#!/usr/bin/env bash
# Oritatami.app を作る
#
#   ./scripts/make_app.sh [出力先フォルダ]           既定: ~/Applications (自己完結型)
#   ./scripts/make_app.sh --link [出力先フォルダ]     このフォルダの .venv を参照する軽い版
#   ./scripts/make_app.sh --no-ollama [出力先]       Ollama を同梱しない (約 150 MB 小さくなる)
#
# 既定では Python 本体・依存ライブラリ・画面・バックエンドをすべて .app の中に入れます
# (約 1.6 GB)。作ったあとはこのプロジェクトフォルダを移動・削除しても、.app 自体を別の
# 場所へ動かしても動きます。--link は従来どおりこのフォルダの .venv を指すだけの版で、
# 軽い代わりにフォルダを動かすと壊れます。
set -euo pipefail

MODE=standalone
WITH_OLLAMA=1
while :; do
    case "${1:-}" in
        --link) MODE=link; shift ;;
        --no-ollama) WITH_OLLAMA=0; shift ;;
        *) break ;;
    esac
done

REPO="$(cd "$(dirname "$0")/.." && pwd -P)"
DEST="${1:-$HOME/Applications}"
APP="$DEST/Oritatami.app"
RES="$APP/Contents/Resources"
PY="$REPO/.venv/bin/python"

[ -x "$PY" ] || { echo "先に ./scripts/setup.sh を実行してください (.venv がありません)" >&2; exit 1; }
[ -f "$REPO/frontend/dist/index.html" ] || {
    echo "先に画面をビルドしてください (cd frontend && npm run build)" >&2; exit 1; }
VERSION="$("$PY" -c 'import oritatami; print(oritatami.__version__)')"
PYTAG="$("$PY" -c 'import sys; print("python%d.%d" % sys.version_info[:2])')"

mkdir -p "$DEST"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$RES"

# the icon goes into the bundle (Finder/Launchpad) and next to the code (Dock icon at runtime)
"$PY" "$REPO/scripts/make_icon.py" "$REPO/backend/oritatami/assets" >/dev/null
cp "$REPO/backend/oritatami/assets/Oritatami.icns" "$RES/Oritatami.icns"

if [ "$MODE" = standalone ]; then
    echo "自己完結型で作ります (Python と依存ライブラリを .app の中に入れます)…"
    # 1. the interpreter the environment was built from
    BASE_BIN="$(awk -F'= *' '/^home/ {print $2; exit}' "$REPO/.venv/pyvenv.cfg")"
    BASE_ROOT="$(cd "$BASE_BIN/.." && pwd -P)"
    [ -x "$BASE_ROOT/bin/$PYTAG" ] || { echo "Python 本体が見つかりません: $BASE_ROOT" >&2; exit 1; }
    ditto "$BASE_ROOT" "$RES/runtime"

    # 2. the environment, re-pointed at the copy above
    ditto "$REPO/.venv" "$RES/venv"
    rm -f "$RES/venv/bin/python" "$RES/venv/bin/python3" "$RES/venv/bin/$PYTAG"
    ln -s "../../runtime/bin/$PYTAG" "$RES/venv/bin/$PYTAG"
    ln -s "$PYTAG" "$RES/venv/bin/python3"
    ln -s "$PYTAG" "$RES/venv/bin/python"
    # The interpreter finds its standard library through the symlink above, but this file is
    # the written record of where the environment came from — left alone it still names the
    # uv install outside the bundle. The launcher rewrites it again if the .app is moved.
    {
        echo "home = $RES/runtime/bin"
        echo "implementation = CPython"
        echo "version_info = $("$PY" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
        echo "include-system-site-packages = false"
    } > "$RES/venv/pyvenv.cfg"
    # activate/activate.fish/... are for an interactive shell and name the old path; inside an
    # .app nobody sources them, and leaving them would leave a trail back to this folder.
    rm -f "$RES/venv/bin"/activate*

    # Console scripts are written with the absolute path of the environment that built them,
    # so every one of them would still point into this folder. `env` plus a PATH set by the
    # launcher keeps them working wherever the bundle ends up — which is what lets Boltz run
    # after this project folder is gone.
    for f in "$RES/venv/bin"/*; do
        [ -f "$f" ] || continue
        head -c 2 "$f" | grep -q '^#!' || continue
        /usr/bin/sed -i '' "1s|^#!.*/python[0-9.]*$|#!/usr/bin/env $PYTAG|" "$f" || true
    done

    # 3. the app itself, as a normal installed package rather than a link back here
    SITE="$RES/venv/lib/$PYTAG/site-packages"
    rm -rf "$SITE/oritatami" "$SITE"/__editable__*oritatami* "$SITE"/__editable___oritatami*
    ditto "$REPO/backend/oritatami" "$SITE/oritatami"
    find "$SITE/oritatami" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
    ditto "$REPO/frontend/dist" "$RES/frontend/dist"

    # 4. Ollama itself. The LLM half of the app is useless without it, and a fresh Mac has none;
    # bundling it is what makes "open the app and it works" true. Kept as its own folder because
    # the archive is flat — the binary looks for llama-server and the runners beside itself.
    if [ "$WITH_OLLAMA" = 1 ]; then
        # The archive is kept, not the unpacked tree: 153 MB instead of 502 MB, and unpacking
        # it again costs a couple of seconds.
        OLLAMA_TGZ="$REPO/.cache/ollama-darwin.tgz"
        mkdir -p "$REPO/.cache"
        if [ ! -s "$OLLAMA_TGZ" ]; then
            echo "Ollama をダウンロードします (約 150 MB)…"
            /usr/bin/curl -fsSL --retry 2 -o "$OLLAMA_TGZ.part" \
                "https://github.com/ollama/ollama/releases/latest/download/ollama-darwin.tgz" \
                && mv "$OLLAMA_TGZ.part" "$OLLAMA_TGZ" \
                || { rm -f "$OLLAMA_TGZ.part"
                     echo "警告: Ollama を取得できませんでした。.app からは「用意する」で後から取得できます" >&2; }
        else
            echo "Ollama を同梱します (キャッシュ: $OLLAMA_TGZ)…"
        fi
        if [ -s "$OLLAMA_TGZ" ]; then
            mkdir -p "$RES/ollama"
            # bsdtar keeps the code-signature metadata; a copy without it is killed on launch
            /usr/bin/tar xzf "$OLLAMA_TGZ" -C "$RES/ollama"
            chmod +x "$RES/ollama/ollama" 2>/dev/null || true
        fi
    fi

    MISSING_MSG="この .app が壊れています。プロジェクトフォルダで scripts/make_app.sh を実行し直してください。"
else
    MISSING_MSG="Python 環境が見つかりません: $REPO/.venv — フォルダを移動した場合は scripts/setup.sh を実行し直してください。"
fi

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>Oritatami</string>
    <key>CFBundleDisplayName</key><string>Oritatami</string>
    <key>CFBundleIdentifier</key><string>app.rtck.oritatami</string>
    <key>CFBundleVersion</key><string>$VERSION</string>
    <key>CFBundleShortVersionString</key><string>$VERSION</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleExecutable</key><string>Oritatami</string>
    <key>CFBundleIconFile</key><string>Oritatami</string>
    <key>LSMinimumSystemVersion</key><string>13.0</string>
    <key>NSHighResolutionCapable</key><true/>
    <key>LSApplicationCategoryType</key><string>public.app-category.education</string>
</dict>
</plist>
PLIST

if [ "$MODE" = standalone ]; then
cat > "$APP/Contents/MacOS/Oritatami" <<'LAUNCHER'
#!/bin/bash
# Oritatami launcher (generated by scripts/make_app.sh — self-contained build)
# Everything it needs is inside this bundle; the paths are worked out from where the bundle
# actually is, so moving or renaming the .app does not break it.
set -u
HERE="$(cd "$(dirname "$0")" && pwd -P)"
RES="$(cd "$HERE/../Resources" && pwd -P)"
PYTAG="__PYTAG__"
VENV="$RES/venv"
LOG_DIR="$HOME/Library/Logs/Oritatami"
mkdir -p "$LOG_DIR"

if [ ! -x "$VENV/bin/$PYTAG" ]; then
  /usr/bin/osascript -e 'display alert "Oritatami を起動できません" message "__MISSING__" as critical'
  exit 1
fi

# The environment records where its interpreter lives. Rewrite it when the bundle has moved
# (or was built somewhere else), so the copy inside this bundle is the one that gets used.
CFG="$VENV/pyvenv.cfg"
WANT="home = $RES/runtime/bin"
if [ -w "$CFG" ] && ! /usr/bin/grep -qxF "$WANT" "$CFG" 2>/dev/null; then
  {
    echo "$WANT"
    echo "implementation = CPython"
    echo "include-system-site-packages = false"
  } > "$CFG" 2>/dev/null || true
fi

# console scripts (boltz) resolve their interpreter through PATH
export PATH="$VENV/bin:$RES/runtime/bin:/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:/usr/bin:/bin"
export ORITATAMI_FRONTEND_DIST="$RES/frontend/dist"
# Ollama lives in the bundle too when it was built with it; the app falls back to fetching
# its own copy into the data folder when this path is missing.
[ -x "$RES/ollama/ollama" ] && export ORITATAMI_OLLAMA_DIR="$RES/ollama"
# exec keeps this process as the app, so clicking the Dock icon again brings the window forward
exec "$VENV/bin/$PYTAG" -m oritatami --app >>"$LOG_DIR/launcher.log" 2>&1
LAUNCHER
/usr/bin/sed -i '' "s|__PYTAG__|$PYTAG|; s|__MISSING__|$MISSING_MSG|" "$APP/Contents/MacOS/Oritatami"
else
cat > "$APP/Contents/MacOS/Oritatami" <<LAUNCHER
#!/bin/bash
# Oritatami launcher (generated by scripts/make_app.sh — linked build)
export PATH="/opt/homebrew/bin:/usr/local/bin:\$HOME/.local/bin:/usr/bin:/bin"
LOG_DIR="\$HOME/Library/Logs/Oritatami"
mkdir -p "\$LOG_DIR"
if [ ! -x "$PY" ]; then
  /usr/bin/osascript -e 'display alert "Oritatami を起動できません" message "$MISSING_MSG" as critical'
  exit 1
fi
exec "$PY" -m oritatami --app >>"\$LOG_DIR/launcher.log" 2>&1
LAUNCHER
fi

chmod +x "$APP/Contents/MacOS/Oritatami"
touch "$APP"  # refresh the Finder / Launchpad icon cache

if [ "$MODE" = standalone ]; then
    # Prove the claim rather than make it: nothing the launcher uses may point back at this folder.
    LEFTOVERS="$(/usr/bin/grep -rl --binary-files=without-match -- "$REPO" \
        "$APP/Contents/MacOS" "$RES/venv/bin" "$RES/venv/pyvenv.cfg" 2>/dev/null | head -5 || true)"
    if [ -n "$LEFTOVERS" ]; then
        echo "警告: .app の中にこのフォルダへの参照が残っています:" >&2
        echo "$LEFTOVERS" >&2
    fi
    OLLAMA_NOTE="Ollama 同梱なし"
    [ -x "$RES/ollama/ollama" ] && OLLAMA_NOTE="Ollama 同梱"
    echo "作成しました: $APP ($(du -sh "$APP" | cut -f1), 自己完結型, $OLLAMA_NOTE)"
else
    echo "作成しました: $APP ($REPO/.venv を参照)"
fi
