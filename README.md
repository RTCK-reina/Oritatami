# Oritatami

*A Mac app for folding proteins on your own machine: Boltz-2 structure prediction, ESM-2
mutation scoring and a local LLM that proposes what to try next, in one window. Everything
runs locally — only MSA search and database lookups leave the machine. Japanese UI.*

タンパク質の立体構造を Mac の中で計算して、眺めて、いじって遊ぶためのアプリです。
配列を並べて予測し、変異を入れて比べ、ローカル LLM に次の一手を提案させる、という流れを 1 つの画面で行えます。

- **構造予測**: Boltz-2 (タンパク質・DNA/RNA・薬などの低分子・金属イオンの複合体、結合親和性)
- **変異の手がかり**: ESM-2 による全 1 残基置換のスコア (変異スキャン)、配列の「天然らしさ」の改良
- **AI アシスタント**: ローカルの LLM (Ollama、既定は `qwen3.5:9b`) が変異・結合相手・新しい配列を提案。提案は配列との照合、UniProt / PubChem / PDB 化学辞書、RDKit、ESM-2 で検証してから表示
- **3D ビューア**: Mol* を内蔵。pLDDT 色分け、表面・原子表示、ポケット、変異残基の強調、2 構造の重ね合わせ (RMSD)、4K 画像

計算は手元の Mac で行います。外部に送られるのは、MSA 検索 (ColabFold 公開サーバー) に使うタンパク質配列と、UniProt・RCSB PDB・AlphaFold DB・PubChem への検索語だけです。

---

## 動作環境

- macOS 13 以降、Apple Silicon。動作確認は M5 Pro / メモリ 24 GB (メモリが多いほど大きな複合体を扱えます)
- 空きディスク 20 GB 程度 (Boltz-2 の重み約 6 GB、ESM-2 約 2.5 GB、LLM 約 7 GB)
- インターネット接続 (初回のモデル取得、MSA 検索、データベース検索)

## インストール

```bash
cd Oritatami
./scripts/setup.sh
```

対話しながら次を行います。何度実行しても安全です (済んでいる手順は飛ばします)。

- Python 3.12 の仮想環境 (`.venv`) を uv で作り、Boltz-2・PyTorch・ESM-2 などを入れる
- 画面 (フロントエンド) をビルドする (Node.js がなければ Homebrew で入れるか尋ねます)
- Boltz-2 の重みを先にダウンロードするか尋ねる (いいえなら初回の予測時に自動取得)
- Ollama と LLM モデル (`qwen3.5:9b`) を入れるか尋ねる
- `~/Applications/Oritatami.app` を作るか尋ねる

すべて「はい」で進めるときは `./scripts/setup.sh --yes`、テストや lint の道具も入れるときは `--dev` を付けます。

## 起動

- **アプリ**: Launchpad / Spotlight から「Oritatami」を開く (`./scripts/make_app.sh` で作り直せます)
- **ターミナル**: `.venv/bin/oritatami` (ネイティブウィンドウ) / `.venv/bin/oritatami --browser` (ブラウザ)

同じデータフォルダを使う Oritatami は 1 つだけ動きます。起動中にもう一度開くと、動いているものにウィンドウをつなぎます。
計算中にウィンドウを閉じようとすると確認が出ます。

## 使い方

最初は作業台に例が並んでいます。「はじめての構造予測 (ユビキチン)」を押すと、分子の準備から計算まで自動で進み、約 1 分で 3D 表示と結果が出ます。

自分の分子で試すときは、左の作業台に「＋ タンパク質」(UniProt 検索)・「＋ 配列を貼る」(FASTA)・「＋ リガンド・薬」(PubChem / SMILES / CCD)・「＋ PDB から」で分子を並べ、**構造を予測する** (⌘↩) を押します。予測の前に、これまでの実績から推定した所要時間とメモリの目安が表示されます。

結果パネルでは、信頼度の数値を平易な文章でも読めます。数値の意味は各項目の「?」か、ヘルプ (?) の用語集で確認できます。

変異で遊ぶ流れの例:

- 結果の「作業台で改変」→ 配列の残基をクリックしてアミノ酸を選ぶ (ESM-2 スキャン済みならスコア付き) → 予測
- 「変異スキャン (ESM-2)」→ ヒートマップのセルをクリックで変異を入れる、または「ESM の上位 N 個をまとめて予測」
- LLM に「変異を提案」→ 検証済みの案を「適用して予測」、複数の案は「まとめて予測」
- 同じ元から作った結果は「比較」タブに並び、pLDDT・pTM・ipTM・結合確率の差と、「重ねる」で構造の違い (RMSD) を確認できます

失敗したジョブには原因に応じた立て直し方が出ます (メモリ不足なら「サンプル 1 で再実行」「CPU で再実行」、MSA サーバーの不調なら「MSA なしで再実行」など)。

結果は「書き出し」から zip (mmCIF・PDB・スコア・入力・ログ・README) で保存できます。

### キーボードショートカット

| キー | 動作 |
| --- | --- |
| ⌘K | コマンド検索 (ほぼすべての操作と例・過去のジョブを開ける) |
| ⌘↩ | 作業台の構造を予測する (LLM の入力欄では送信) |
| ⌘I | 分子を追加 |
| ⌘1 / ⌘2 | 左側を作業台 / ジョブ一覧に切り替え |
| ⌘B / ⌘⇧B | 左パネル / LLM パネルの表示切り替え |
| ⌘, | 設定・準備状況 |
| F | 3D ビューアを大きく表示 / 戻す |
| R | 3D の視点をリセット |
| ? | 使い方と用語 |

パネルの境目はドラッグで幅・高さを変えられ (ダブルクリックで元に戻る)、配置とテーマ (ライト / ダーク / システム) は次回も引き継がれます。

## データの保存場所

| 内容 | 場所 |
| --- | --- |
| ジョブ・会話履歴・ライブラリ・設定 | `~/Library/Application Support/Oritatami/` |
| 書き出したファイル・画像 | `~/Downloads/Oritatami/` (書き込めない場合はアプリのデータフォルダ内 `exports/`) |
| Boltz-2 の重み・化学辞書 | `~/.boltz/` |
| ESM-2 の重み | `~/.cache/huggingface/hub/` |
| アプリ起動時のログ | `~/Library/Logs/Oritatami/launcher.log` |

設定 → ストレージから、予測結果を残したまま中間ファイルを削除できます (既定では予測完了時に自動削除)。
データフォルダは環境変数 `ORITATAMI_HOME` で変更できます。

## 困ったとき

- **準備状況の確認**: 設定 (⌘,) の「準備状況」に、Boltz-2・重み・GPU・ESM-2・Ollama・LLM モデル・空きディスクの状態と、足りないものの直し方が出ます
- **予測が遅い / 止まって見える**: 結果パネルの「ログを表示」で進行を確認できます。初回は重みのダウンロードで時間がかかり、「MSA 検索」は公開サーバーが混んでいると数分待つことがあります
- **メモリ不足**: 使用量はトークン数 (残基数 + リガンド重原子数) が増えると急に増えます。予測前の目安表示に「足りなくなる可能性」と出たら、コピー数を減らすか、長いタンパク質を必要なドメインだけに切り出してください
- **アプリが起動しない**: `~/Library/Logs/Oritatami/launcher.log` と `~/Library/Application Support/Oritatami/oritatami.log` を確認してください。`.venv` を作り直すときは `./scripts/setup.sh` を再実行します

---

## 開発

```
backend/oritatami/
  app.py          FastAPI (REST API と画面の配信)
  desktop.py      起動 (サーバースレッド + pywebview ウィンドウ、単一インスタンス)
  instance.py     データフォルダごとの排他ロック
  jobs.py         ジョブキュー (予測レーンと ESM レーン、変更通知の long-poll)
  handlers.py     ジョブ種別ごとの処理
  engines/boltz.py  Boltz-2 CLI の実行・進行解析・失敗分類・結果集計
  engines/esm.py    ESM-2 の変異スコア・配列改良
  assistant.py    LLM への文脈構築と提案の検証
  estimate.py     所要時間・メモリの推定 (実績で較正)
  exporter.py     zip 書き出し
  structure.py    gemmi による構造解析 (pLDDT・界面・重ね合わせ・PDB 変換)
  sources.py      UniProt / RCSB PDB / AlphaFold DB / PubChem
  system.py       マシン情報・通知・ストレージ
frontend/src/
  App.tsx         レイアウト・ショートカット・コマンドパレット
  store.tsx       状態とサーバー同期
  components/     各パネル・ダイアログ
  viewer/         Mol* ラッパー
scripts/          setup.sh / make_app.sh / make_icon.py
tests/            pytest
```

```bash
./scripts/setup.sh --dev
.venv/bin/python -m pytest              # バックエンドのテスト
.venv/bin/ruff check backend tests scripts
cd frontend && npm run build            # 型チェック + ビルド (frontend/dist)
cd frontend && npm run dev              # 開発サーバー (API は 127.0.0.1:47823 へ転送)
.venv/bin/oritatami --serve             # API サーバーのみ
```

### API

すべて `http://127.0.0.1:47823/api` 以下、JSON。エラーは `{"detail": "日本語の説明"}` と HTTP ステータス (400 入力不正 / 404 なし / 502 外部 DB / 503 Ollama・ESM 不可)。

| メソッド | パス | 内容 |
| --- | --- | --- |
| GET | `/ping` | 生存確認 `{ok, version}` |
| GET | `/health` | 準備状況 (Boltz・重み・GPU・ESM・Ollama・マシン・空きディスク) |
| GET / PATCH | `/settings` | 設定の取得・部分更新 |
| POST | `/estimate` | `{spec}` → 所要時間 `{seconds, low, high, breakdown, basis, memory}` |
| POST | `/jobs/predict` | `{spec, title?, parent_id?, origin?}` → ジョブ |
| POST | `/jobs/predict/batch` | `{spec, variants: [{chain, mutations}], parent_id?}` → `{jobs}` (全件検証してから一括投入) |
| POST | `/jobs/scan` / `/jobs/refine` | ESM-2 変異スキャン / 配列改良 |
| GET | `/jobs` | 一覧 (配列を省いた要約) |
| GET | `/jobs/changes?rev=&timeout=` | 変更があるまで待つ long-poll `{rev, changed}` |
| GET / PATCH / DELETE | `/jobs/{id}` | 詳細 / 名前・お気に入り / 削除 |
| POST | `/jobs/{id}/cancel` | キャンセル |
| POST | `/jobs/{id}/retry` | `{msa?: "single", accelerator?, diffusion_samples?, new_seed?}` で再実行 |
| GET | `/jobs/{id}/export.zip` | 書き出し (添付ファイル) |
| POST | `/jobs/{id}/export` | ダウンロードフォルダへ書き出して Finder で表示 |
| GET | `/jobs/{id}/structure.pdb?model=` | PDB 形式 |
| GET | `/jobs/{id}/log` / `/jobs/{id}/files/{path}` | ログ / ジョブ内ファイル |
| POST | `/compare` | 2 構造の重ね合わせ `{rmsd, matched_ca, deviations, url}` |
| POST | `/assistant/ask` | LLM への依頼 (`mode`: chat / mutations / complex / design / explain) |
| GET / DELETE | `/assistant/threads[/{id}]` | 会話履歴 |
| GET / POST / DELETE | `/library[/{id}]` | 保存した分子・作業台 |
| GET | `/uniprot/search`, `/uniprot/{acc}`, `/pdb/search` | データベース検索 |
| POST | `/import/pdb`, `/import/afdb`, `/import/upload` | 構造の取り込み |
| POST / GET | `/chem/describe`, `/chem/pubchem`, `/chem/ccd/{code}` | 低分子の情報・構造式 SVG |
| GET / POST | `/storage`, `/storage/cleanup` | 使用容量 / 不要ファイルの削除 |
| POST | `/files/save`, `/notify` | 画像の保存 / macOS 通知 |
| GET / POST | `/llm/status`, `/llm/start`, `/llm/pull` | Ollama の状態・起動・モデル取得 |

予測の入力 (`spec`) の形:

```json
{
  "name": "CA2 + Zn + アセタゾラミド",
  "components": [
    {"type": "protein", "chains": ["A"], "label": "CA2", "sequence": "MSHHWGYGKH...", "msa": "server"},
    {"type": "ligand", "chains": ["B"], "label": "亜鉛", "ccd": "ZN"},
    {"type": "ligand", "chains": ["C"], "label": "AZM", "smiles": "CC(=O)Nc1nnc(s1)S(N)(=O)=O"}
  ],
  "affinity_binder": "C",
  "params": {"diffusion_samples": 1, "recycling_steps": 3, "sampling_steps": 200, "use_potentials": false, "seed": null, "accelerator": "auto"}
}
```

## 使っているもの

Boltz-2 (boltz-community)、ESM-2 (Meta AI)、Mol*、Ollama と Qwen、FastAPI、pywebview、RDKit、gemmi、Biopython、React、Vite。各モデル・ライブラリはそれぞれのライセンスに従います。
予測値はすべて計算による推定で、実験値の代わりにはなりません。

© 2026 RTCK

---

## リリースの .app について

[Releases](../../releases) に macOS 用の `.app` を置いています。どちらも Python・依存ライブラリ・
画面・バックエンドをすべてバンドル内に持つ自己完結型で、プロジェクトフォルダが無くても、
`.app` を別の場所に移しても動きます。

| 配布物 | 中身 | 初回起動 |
| --- | --- | --- |
| `Oritatami.app.zip` | Ollama も同梱 | そのまま使えます |
| `Oritatami-no-ollama.app.zip` | Ollama は含まない | 設定 → 準備状況の「用意する」でアプリが取得します (約 150 MB) |

どちらも Boltz-2 の重み (約 6 GB)・ESM-2 (約 2.5 GB)・LLM モデル (約 6 GB) は初回に必要な
ぶんだけダウンロードします。署名も公証もしていないので、初回は Finder で右クリック →「開く」
から起動してください。

ソースから動かす場合は上の「インストール」の手順 (`./scripts/setup.sh`) を使ってください。
`.app` を作り直すときは `./scripts/make_app.sh` (Ollama を省くなら `--no-ollama`) です。

## ライセンス

MIT License — [LICENSE](LICENSE) を参照してください。

同梱している第三者のソフトウェアはそれぞれのライセンスに従います (Ollama: MIT、Boltz: MIT、
PyTorch: BSD-3-Clause、Python: PSF、ESM-2 の重み: MIT)。
