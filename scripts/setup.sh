#!/usr/bin/env bash
# Oritatami セットアップ (macOS / Apple Silicon)
#
#   ./scripts/setup.sh              対話しながらセットアップ
#   ./scripts/setup.sh --yes        すべて既定の答え (はい) で進める
#   ./scripts/setup.sh --dev        テスト・lint 用の開発ツールも入れる
#
# 何度実行しても安全です (済んでいる手順は飛ばします)。
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

ASSUME_YES=0
DEV=0
for arg in "$@"; do
  case "$arg" in
    --yes|-y) ASSUME_YES=1 ;;
    --dev) DEV=1 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "不明なオプション: $arg" >&2; exit 2 ;;
  esac
done

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*"; }
fail() { printf '  \033[31m✗\033[0m %s\n' "$*" >&2; exit 1; }
ask() {  # ask "質問" -> 0 (はい) / 1 (いいえ)。既定ははい。端末から実行していないときは --yes がなければ「いいえ」
  if [ "$ASSUME_YES" = 1 ]; then return 0; fi
  if [ ! -t 0 ]; then return 1; fi
  local reply
  read -r -p "  $1 [Y/n] " reply
  [[ -z "$reply" || "$reply" =~ ^[Yy] ]]
}
have() { command -v "$1" >/dev/null 2>&1; }

bold "Oritatami セットアップ"
if [ "$(uname -s)" != "Darwin" ]; then
  warn "macOS 以外では動作確認していません (ネイティブウィンドウと GPU 計算は macOS 前提です)"
fi
if [ "$(uname -m)" != "arm64" ]; then
  warn "Apple Silicon 以外では GPU (MPS) が使えず、構造予測がとても遅くなります"
fi
export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$PATH"

# ---------------------------------------------------------------- 1. uv + Python 3.12
bold "[1/5] Python 環境"
if ! have uv; then
  if have brew; then
    ask "Python 環境の管理に uv を Homebrew で入れますか？" || fail "uv が必要です (https://docs.astral.sh/uv/)"
    brew install uv
  else
    ask "uv を公式インストーラで入れますか？ (Homebrew がないため)" || fail "uv が必要です (https://docs.astral.sh/uv/)"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
fi
ok "uv $(uv --version | awk '{print $2}')"
if [ ! -x .venv/bin/python ]; then
  uv venv --python 3.12 .venv
fi
ok "Python $(.venv/bin/python -c 'import platform; print(platform.python_version())') (.venv)"

# ---------------------------------------------------------------- 2. Python packages
bold "[2/5] 計算エンジンとアプリ本体 (Boltz-2, PyTorch, ESM-2 など。初回は数分)"
if [ "$DEV" = 1 ]; then
  uv pip install --python .venv/bin/python -e ".[dev]"
else
  uv pip install --python .venv/bin/python -e .
fi
.venv/bin/python - <<'PY'
import importlib.util, sys
missing = [m for m in ("boltz", "torch", "transformers", "fastapi", "gemmi", "rdkit", "webview") if importlib.util.find_spec(m) is None]
if missing:
    sys.exit("読み込めないパッケージ: " + ", ".join(missing))
import torch
print("  \033[32m✓\033[0m PyTorch", torch.__version__, "/ GPU (MPS):", "使えます" if torch.backends.mps.is_available() else "使えません (CPU で計算します)")
PY

# ---------------------------------------------------------------- 3. frontend
bold "[3/5] 画面 (フロントエンド) のビルド"
needs_build=1
if [ -f frontend/dist/index.html ]; then
  newest_src="$(find frontend/src frontend/index.html frontend/package.json -type f -newer frontend/dist/index.html | head -1)"
  [ -z "$newest_src" ] && needs_build=0
fi
if [ "$needs_build" = 1 ]; then
  if ! have npm; then
    if have brew && ask "ビルドに Node.js が必要です。Homebrew で入れますか？"; then
      brew install node
    else
      fail "Node.js (npm) が必要です: https://nodejs.org/ から入れてから再実行してください"
    fi
  fi
  (cd frontend && { [ -d node_modules ] && [ node_modules -nt package-lock.json ] || npm ci --no-audit --no-fund; } && npm run build)
  ok "ビルドしました (frontend/dist)"
else
  ok "ビルド済み (変更なし)"
fi

# ---------------------------------------------------------------- 4. models
bold "[4/5] モデル"
if [ -f "$HOME/.boltz/boltz2_conf.ckpt" ] && [ -d "$HOME/.boltz/mols" ]; then
  ok "Boltz-2 の重み: ダウンロード済み"
elif ask "Boltz-2 の重みと化学辞書 (約 6 GB) を今ダウンロードしますか？ (いいえなら初回の予測時に自動で取得)"; then
  .venv/bin/python -c "from pathlib import Path; from boltz.main import download_boltz2; p = Path.home() / '.boltz'; p.mkdir(exist_ok=True); download_boltz2(p)"
  ok "Boltz-2 の重み: ダウンロードしました"
else
  warn "Boltz-2 の重み: 初回の予測時にダウンロードします"
fi

if ! have llama-server; then
  if have brew && ask "AI アシスタント (LLM) 用に llama.cpp を Homebrew で入れますか？"; then
    brew install llama.cpp
  else
    warn "llama-server がありません。アプリの設定画面からも取得できます (予測・変異スコアは使えます)"
  fi
fi
model="$(.venv/bin/python -c 'from oritatami.config import get_settings; print(get_settings().llm_model)' 2>/dev/null || echo gemma3:4b)"
if .venv/bin/python -c 'import sys; from oritatami import llm; sys.exit(0 if llm.resolve_model(sys.argv[1]) else 1)' "$model" 2>/dev/null; then
  ok "LLM モデル $model: ダウンロード済み"
elif ask "LLM モデル $model (数 GB) をダウンロードしますか？"; then
  if .venv/bin/python -c 'import sys; from oritatami import llm; sys.exit(0 if llm.ensure_model(sys.argv[1]).get("ok") else 1)' "$model"; then
    ok "LLM モデル $model: ダウンロードしました"
  else
    warn "LLM モデルのダウンロードに失敗しました — アプリの設定画面からも試せます"
  fi
else
  warn "LLM モデルはアプリの設定画面からもダウンロードできます"
fi

# ---------------------------------------------------------------- 5. app bundle
bold "[5/5] アプリとして起動できるようにする"
if ask "Oritatami.app を ~/Applications に作りますか？ (Launchpad や Dock から起動できます)"; then
  "$REPO/scripts/make_app.sh" "$HOME/Applications"
else
  warn "あとで ./scripts/make_app.sh で作れます"
fi

echo
bold "セットアップ完了"
cat <<EOF
  起動:
    open ~/Applications/Oritatami.app         (アプリを作った場合)
    $REPO/.venv/bin/oritatami                 (ネイティブウィンドウ)
    $REPO/.venv/bin/oritatami --browser       (ブラウザで開く)
EOF
