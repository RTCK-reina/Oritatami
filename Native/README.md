# Oritatami — Native macOS App (SwiftUI)

ターミナルなしでダブルクリック起動できるネイティブ Mac アプリです。

## 構成

```
Native/
├── Package.swift                    # Swift Package Manager 定義
├── build-app.sh                     # .app バンドルビルドスクリプト
└── Sources/Oritatami/
    ├── App/
    │   ├── OrItatamiApp.swift       # @main エントリーポイント
    │   └── BackendManager.swift     # Python バックエンドサブプロセス管理
    ├── API/
    │   ├── APIClient.swift          # FastAPI クライアント (URLSession)
    │   └── Models.swift             # Swift データモデル
    └── Views/
        ├── ContentView.swift        # メインウィンドウ (3ペインレイアウト)
        ├── WorkbenchView.swift      # コンポーネントエディタ + ジョブ一覧
        ├── ViewerView.swift         # WKWebView → Mol* 3D ビューア
        └── AssistantView.swift      # AI アシスタント (Qwen3)
```

## ビルド方法

### 前提条件

- macOS 14 Sonoma 以上
- Xcode 15 以上（`xcodebuild` / `swift build` が使えること）
- oritatami バックエンドが `pip install -e .` でインストール済み
- frontend が `cd frontend && npm run build` でビルド済み

### コマンド

```bash
cd Native

# デバッグビルド（開発用）
./build-app.sh

# リリースビルド（配布用・最適化あり）
./build-app.sh --release

# 起動
open Oritatami.app
```

## アーキテクチャ

```
┌─────────────────────────────────────────────────────┐
│  Oritatami.app (SwiftUI)                            │
│                                                     │
│  ┌───────────────┐  ┌──────────────┐  ┌──────────┐ │
│  │ WorkbenchView │  │  ViewerView  │  │Assistant │ │
│  │               │  │  (WKWebView) │  │  View    │ │
│  │ - コンポーネント │  │  viewer.html │  │          │ │
│  │   エディタ     │  │  Mol* 5.x    │  │  Qwen3   │ │
│  │ - ジョブ一覧   │  │              │  │          │ │
│  └───────┬───────┘  └──────┬───────┘  └────┬─────┘ │
│          │                 │               │        │
│          └─────────────────┴───────────────┘        │
│                     ↓ URLSession                    │
│             BackendManager (subprocess)             │
└──────────────────────┬──────────────────────────────┘
                       │ HTTP :47823
           ┌───────────▼───────────────┐
           │  Python FastAPI backend   │
           │  oritatami --serve        │
           │                           │
           │  Boltz-2 / ESM-2 / Qwen  │
           └───────────────────────────┘
```

## 開発メモ

- `BackendManager` は起動時に `oritatami --serve` をサブプロセスとして起動し、
  `/api/health` が 200 を返すまでポーリングします
- `ViewerView` は WKWebView で `/viewer.html` をロードし、
  `window.postMessage` / `webkit.messageHandlers.viewer` で Swift と通信します
- 生物学的安全フィルター（SafeguardError）は API 422 + `code: "safeguard"` で
  返り、AssistantView が黄色バナーで表示します
