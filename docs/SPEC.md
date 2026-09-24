# Oritatami 仕様書

対象バージョン 0.2.3 / 2026-09-15

この文書は Oritatami の設計と外部契約を記述する。読者は本アプリを改修する開発者、および HTTP API
を直接叩く利用者を想定する。UI の操作手順は README に譲り、ここでは「何がどう動くと決まっているか」
だけを書く。

数値のうち「実測」と明記したものは、この開発機（Apple M5 Pro / 24 GB / macOS）で取得した値である。
機種が変われば絶対値は変わるが、大小関係の判断材料としては使える。

---

## 1. 目的と設計方針

タンパク質の構造予測（Boltz-2）、変異の評価（ESM-2）、逆折り畳み（ProteinMPNN）、および局所 LLM
による提案を、1 台の Mac の中で完結させるデスクトップアプリである。外部に送るのは MSA 検索
（ColabFold サーバー）と、UniProt / PDB / PubChem の参照だけで、配列以外のデータは外に出ない。

設計上の決定事項は以下の 5 点。

**1.1 計算は必ず別プロセスに出す。** Boltz は `boltz predict` を supervisor 経由で起動する。アプリが
強制終了しても GPU を掴んだままの孤児プロセスが残らず、逆にアプリ側のクラッシュが計算を巻き込まない。

**1.2 スコアは正しさを意味しない。** pLDDT・pTM・ipTM はいずれも予測器の自信であって、分子として
成立しているかとは独立である。したがって「機能を壊していないか」（§8.1）と「形が物理的に可能か」
（§8.2）を別系統で検査し、スコアの隣に並べて表示する。

**1.3 自律動作には必ず上限を置く。** 自律ループと PDB ウォッチャーは人間の承認なしにジョブを投入
できるため、全ての自律投入は governor（§7.3）を通す。人間が直接押した投入は governor を通らない。

**1.4 LLM の出力は検証してから通す。** 提案は作業台の実データと照合し、通らないものは落とす。ただし
残基名が正しく位置だけ誤っているケースは、その残基が実在する位置へ移して通し、移したことを明示する
（§6.3）。

**1.5 外部ファイルシステムに依存しない。** リリースされる .app は Python ランタイム・依存ライブラリ・
フロントエンドのビルド成果物・（同梱版は）llama-server バイナリを全て内部に持つ。開発リポジトリや外部の
仮想環境を参照しない（§13）。

---

## 2. プロセス構成

```
Oritatami.app
  └─ python -m oritatami --app            アプリ本体（1 プロセス）
       ├─ FastAPI + uvicorn                127.0.0.1 の動的ポート
       ├─ pywebview ウィンドウ              フロントエンド（React）を表示
       ├─ JobManager                        2 レーン（predict / esm）のワーカースレッド
       ├─ Autopilot ウォッチドッグ           予測完了を監視して自律解析を起動
       └─ PDB ウォッチャー                   6 時間ごとに新着構造を検索
            │
            ├─(subprocess)→ supervise → boltz predict     構造予測
            ├─(subprocess)→ llama-server                  局所 LLM
            └─(subprocess)→ proteinmpnn                    逆折り畳み
```

起動しているインスタンスは `instance.json`（`{url, pid, version, started}`）と `instance.lock` で
一意性を担保する。二重起動時は既存ウィンドウを前面に出して終了する。

ESM-2 のみアプリと同一プロセス内（torch + transformers）で動く。これは 650M パラメータのモデルを
スキャンのたびに読み直すコストを避けるためで、代わりに予測ジョブの直前に明示的な解放を行う（§9.2）。

---

## 3. モジュール構成

### 3.1 バックエンド（`backend/oritatami/`）

| モジュール | 責務 |
| --- | --- |
| `app.py` | HTTP API 定義、起動・終了処理、例外ハンドラ |
| `jobs.py` | ジョブキュー、レーン、ワーカー、キャンセル、再試行、ライブ状態 |
| `handlers.py` | ジョブ種別と実行関数の対応付け |
| `db.py` | SQLite スキーマとアクセス（jobs / library / threads / llm_calls / searches） |
| `config.py` | 設定モデル、永続化、マイグレーション、データディレクトリの解決 |
| `engines/boltz.py` | 予測仕様の正規化、YAML 生成、Boltz 起動、MSA 再利用、結果収集 |
| `engines/supervise.py` | 子プロセスの監視、親の死亡検知、ピークメモリ計測 |
| `engines/esm.py` | ESM-2 の読み込み・変異スキャン・配列磨き・メモリ解放 |
| `engines/mpnn.py` | ProteinMPNN による逆折り畳みスコア |
| `structure.py` | gemmi による構造解析（鎖・配列・pLDDT・接触・重ね合わせ・形状検査） |
| `seq.py` | 配列の検証、変異の解析・適用・修復 |
| `assistant.py` | LLM への文脈構築、構造化提案の取得、提案の検証 |
| `autopilot.py` | 自律ループ（解析 → 提案 → 変異投入） |
| `governor.py` | 自律投入のゲート（日次予算・キュー長・空きディスク） |
| `function_risk.py` | 保護すべき残基の推定、結果の機能・形状リスク評価 |
| `estimate.py` | 所要時間とピークメモリの推定（履歴からの回帰） |
| `regime.py` | 実行レジームの判定（メモリ状況に対する事前警告） |
| `gpu.py` | Metal のワーキングセット上限の読み書き |
| `ssd.py` | SSD の摩耗量と、スワップを伴う実行の書き込み見積もり |
| `system.py` | 機種情報、通知、ディスク使用量、自プロセスのメモリ推移 |
| `llm.py` | llama-server の解決・起動・自己取得・モデル管理 |
| `sources.py` | UniProt / PDB / AFDB / PubChem の取得 |
| `chem.py` | SMILES と CCD の検証、記述子 |
| `exporter.py` | 結果一式の zip 書き出し |
| `history.py` | 検索履歴 |
| `pdb_watcher.py` | PDB 新着の定期取得と自動投入 |
| `desktop.py` | pywebview ウィンドウ、メニュー、通知 |
| `instance.py` | 単一インスタンス制御 |

### 3.2 フロントエンド（`frontend/src/`）

React 19 + TypeScript + Vite 8。依存は `molstar` / `react` / `react-dom` の 3 つだけで、状態管理
ライブラリは使わない。状態は `store.tsx` の React Context 1 つに集約し、サーバーとの同期は
`api.ts` 経由でのみ行う。構造表示は Mol\* 5.11 を `viewer.html` に隔離した iframe で動かす
（ビューアのメモリをアプリ本体から切り離すため）。

`api.ts` は全ての変更系リクエストを in-flight マップで重複排除し、完了後 900 ms のクールダウンを
かける。これは二重押し・三重押しで同一ジョブが複数投入される事故を防ぐためで、読み取り専用の
POST（`/api/estimate`, `/api/sequence/*`, `/api/chem/*`, `/api/compare`）は対象外として無言で
合流させる。

---

## 4. データモデル

### 4.1 SQLite スキーマ

データベースは `<データディレクトリ>/oritatami.sqlite3`。`journal_mode=WAL`、`synchronous=NORMAL`、
`busy_timeout=15000`。

```sql
jobs(id TEXT PK, kind TEXT, status TEXT, title TEXT,
     created_at REAL, started_at REAL, finished_at REAL,
     spec TEXT /*JSON*/, result TEXT /*JSON*/, error TEXT, error_kind TEXT, -- error_kind は起動時マイグレーションで追加
     phase TEXT, parent_id TEXT, origin TEXT DEFAULT 'user', starred INTEGER DEFAULT 0)
  INDEX jobs_created(created_at DESC), jobs_parent(parent_id), jobs_kind_status(kind, status)

library(id TEXT PK, type TEXT, name TEXT, data TEXT /*JSON*/, created_at REAL)

threads(id TEXT PK, title TEXT, created_at REAL, updated_at REAL, messages TEXT /*JSON*/)

llm_calls(id TEXT PK, created_at REAL, model TEXT, mode TEXT, origin TEXT,
          job_id TEXT, thread_id TEXT, messages TEXT, raw TEXT, reply TEXT,
          proposals TEXT, reply_issues TEXT, corrected INTEGER, elapsed_sec REAL, error TEXT)
  INDEX llm_calls_created(created_at DESC)

searches(id TEXT PK, created_at REAL, source TEXT, query TEXT, hits INTEGER, top TEXT, picked TEXT)
```

ID は `<接頭辞>_YYYYMMDD-HHMMSS_<hex6>`（`job` / `lib` / `thr` / `llm`）。時刻順に並び、かつ衝突しない。

`status` は `queued | running | succeeded | failed | cancelled`。
`kind` は `predict | scan | refine`。
`origin` は `user | autopilot_variant | pdb_watch | qwen` ほか。自律判定に使う（§7.3）。

### 4.2 予測仕様（workbench spec）

`POST /api/jobs/predict` の `spec` と、結果に保存される `normalized_spec` の形。正規化は
`boltz.normalize_spec()` が行い、不正な入力は日本語のメッセージ付き 400 になる。

```jsonc
{
  "name": "prediction",                  // 省略時 "prediction"
  "components": [                        // 1〜60 個。0 個は 400
    {
      "type": "protein",                 // protein | dna | rna | ligand
      "chains": ["A", "B"],              // 省略時は自動採番（A..Z のあと A0..H9、計 106 個）
      "copies": 2,                       // 1〜12。chains 指定時はその長さが優先
      "label": "PCNA",
      "sequence": "MFEAR...",            // protein/dna/rna。大文字の標準残基のみ
      "msa": "server",                   // protein のみ。server | single
      "cyclic": false,                   // 環状。true は結果を壊しうるため UI で警告する
      "smiles": "CC(=O)O",               // ligand。smiles と ccd はどちらか一方のみ
      "ccd": "ATP",
      "source":  {"db": "uniprot", "id": "P12004"},   // 任意。機能注釈の照合に使う
      "mutations": ["K48R"],             // 任意。由来の記録
      "parent_sequence": "MFEAR...",     // 任意。変異前の配列
      "notes": "..."                     // 任意
    }
  ],
  "affinity_binder": "C",                // 任意。ligand のチェーン ID。copies=1 かつ protein が必要
  "constraints": [
    {"type": "pocket",  "binder": "C", "contacts": [["A", 94]], "max_distance": 6.0, "force": false},
    {"type": "contact", "token1": ["A", 10], "token2": ["B", 20], "max_distance": 6.0, "force": false}
  ],
  "params": {
    "diffusion_samples": 1,              // 1〜10
    "recycling_steps": 4,                // 1〜10
    "sampling_steps": 200,               // 10〜500
    "use_potentials": false,             // 物理補正（§8.3）
    "seed": null,                        // null = Boltz 既定
    "accelerator": "auto"                // auto | mps | cpu
  },
  "token_estimate": 261                  // 正規化が計算して埋める（読み取り専用）
}
```

トークン数は「タンパク質・核酸は 1 残基 = 1、リガンドは重原子 1 個 = 1（CCD は 30 と仮定）」で
コピー数を掛けた合計。所要時間とピークメモリの推定はこの値を入力とする。

### 4.3 予測結果

```jsonc
{
  "models": [
    {"index": 0, "file": "out/.../complex_model_0.cif",
     "confidence": {"confidence_score": 0.93, "complex_plddt": 0.936, "ptm": 0.87, "iptm": 0.7,
                    "chains_ptm": {"A": 0.8}, "pair_chains_iptm": {"A": {"B": 0.7}}, ...},
     "plddt": {"A": [93.1, 95.0, ...]},          // 残基ごと（0–100）
     "ligand_plddt": {"C": 88.4}}
  ],
  "chains": [{"chain": "A", "type": "protein", "label": "", "length": 76,
              "sequence": "...", "smiles": null, "ccd": null, "mutations": []}],
  "affinity": {"affinity_pred_value": 1.2, "affinity_probability_binary": 0.8,
               "binder": "C", "ic50_um": 15.8, "delta_g_kcal": -6.5},
  "pae": {"size": 261, "factor": 2, "matrix": [[...]], "segments": [{"chain": "A", "kind": "polymer",
                                                                    "start": 0, "end": 261}]},
  "interfaces": {"cutoff": 4.5, "interfaces": [{"chains": ["A", "B"],
                                                "residues": {"A": ["K48"], "B": ["D21"]},
                                                "contact_pairs": 37}]},
  "geometry": { /* §8.2 */ },
  "elapsed_sec": 33.4,
  "timings": {"preprocess": 2.1, "msa": 0.0, "structure": 28.9},
  "token_estimate": 76,
  "accelerator": "mps",
  "peak_memory_gb": 5.2,
  "peak_memory_method": "footprint",
  "msa": {"server": false, "reused_from": "job_...", "single_sequence_chains": []},
  "normalized_spec": { /* §4.2 */ }
}
```

PAE 行列は 256×256 を超える場合に平均プーリングで縮小し、`factor` に縮小率を入れる。生の行列を
そのまま返すと 1,700 残基で 90 MB を超えるため。

---

## 5. ジョブのライフサイクル

### 5.1 レーンとキュー

レーンは 2 本。`predict`（Boltz）と `esm`（ESM-2 のスキャンと磨き）。各レーンは 1 本のワーカー
スレッドが FIFO で処理する。予測は 1 度に 1 件しか走らない。ESM レーンは予測と並走するが、
所要時間が桁違いに短いためキュー全体の待ち時間には加算しない（§5.5）。

待機中のジョブは `POST /api/jobs/{id}/reorder`（`top | up | down | bottom`）で並べ替えられる。
実行中のジョブは並べ替えられない。

### 5.2 フェーズ

進捗は Boltz の標準出力を正規表現で拾って段階に変換する。

| phase | 表示 | 検出条件 |
| --- | --- | --- |
| `starting` | 開始 | ジョブ開始時 |
| `download` | モデル・辞書をダウンロード中 | `Downloading the ` |
| `preprocess` | 入力を前処理中 | `Checking input data.` / `Processing N inputs` |
| `msa` | MSA を検索中 | `Calling MSA server` / `Generating MSA` |
| `structure` | 構造を予測中 | `Running structure prediction` |
| `affinity` | 結合親和性を予測中 | `Predicting property: affinity` |
| `collect` | 結果を集計中 | 予測プロセス終了後 |
| `done` / `failed` | 完了 / 失敗 | — |

拡散の内部には進捗がない。Boltz 自身のプログレスバーは最初から最後まで 1/1 のままなので、
`structure` フェーズの「止まっているのか、まだ大きいだけなのか」は supervisor が書き出す
メモリ推移（§9.1）でしか判断できない。

### 5.3 キャンセル

`POST /api/jobs/{id}/cancel`。待機中ならキューから外す。実行中ならキャンセルフラグを立て、
supervisor 経由でプロセスグループに SIGTERM、8 秒後に SIGKILL。Boltz はデータローダーの
ワーカーを fork するため、子 1 個ではなくプロセスグループが正しい停止単位である。

`POST /api/jobs/queue/cancel_all` で種別を指定した一括キャンセル。`include_running` で実行中を
含めるかを選ぶ。

### 5.4 自動再試行

一過性の失敗（ネットワーク、MSA サーバー）は `job_auto_retry` が有効なら指数バックオフで
再投入する。再試行は**新しいジョブ**として作られ、`spec._retry_attempt` に回数が入る。失敗した
試行は履歴に残る（上書きしない）。上限は `job_max_retries`（既定 2）。

### 5.5 キュー全体の終了見込み

`GET /api/jobs/eta`。待機中のジョブは履歴から回帰した所要時間、**実行中のジョブは実測**を使う。

実行中のジョブに「経過時間を引いた残り」を使うのは、推定どおりに進んでいる間しか正しくない。
1,278 トークンのジョブが 26 分の推定に対して 162 分走っていた実例では、引き算が負になって下限の
15 秒に張り付き、「もうすぐ終わる」と表示し続けていた。

**CPU 時間は進捗の代わりにならない。** 計算は GPU がやるので、CPU はほぼ一定の付き添いコストに
なる。この機械で測った 3 点（いずれもメモリに収まる範囲）:

| トークン | 実時間 | CPU 時間 | CPU/実時間 | ピーク | user 率 |
| --- | --- | --- | --- | --- | --- |
| 76 | 36.2 秒 | 20.0 秒 | 0.553 | 5.53 GB | 0.855 |
| 304 | 116.3 秒 | 27.0 秒 | 0.232 | 7.81 GB | 0.731 |
| 608 | 388.6 秒 | 40.3 秒 | 0.104 | 14.75 GB | 0.642 |

実時間が 10 倍になる間に CPU 時間は 2 倍にしかならない。「消費 CPU ÷ 必要 CPU」を進捗にすると
開始 1 分で 50 % を指してから止まる。**CPU/実時間の比も健康状態の指標にならない**: 正常な 608
トークンが 0.104、スワップで溺れていた 1,278 トークンが 0.21 で、**遅い方が高い**。

CPU 時間が明確に語るのは「何に使われたか」だけである。正常な実行は user 時間が 64〜86 %（サイズが
大きいほど GPU に寄るので下がる）、最後までページングしていた実行は **1.6 %** だった。40 倍離れて
いるので、`THRASH_USER_SHARE = 0.25` を境にどちらの領域かを判定する。

したがって残り時間は実時間モデルから出し、**収まらない分のページング時間を足す**（§5.6）。実行中は
予測ピークではなく実測フットプリントを使うので、想定より膨らめば推定も伸びる。

`basis` が答えの出どころを示す。`swap` は履歴 + ページング、`baseline` は履歴のみ、`unknown` は
正直に答えられない場合で、このとき `seconds` は null になる。

```jsonc
{"jobs": 5, "counted": 4, "unknown": 1, "seconds": 1820,
 "complete": false,              // false なら合計は下限であって予測ではない
 "finish_at": null,              // complete のときだけ入る
 "at_least_until": 1789441000.0, // 推定できなかったジョブがあるとき
 "now": 1789439180.0,
 "items": [{"id": "job_...", "title": "...", "status": "running", "kind": "predict",
            "seconds": null, "basis": "unknown", "progress": null, "efficiency": 0.004,
            "overrun": true, "note": "ほとんど計算が進んでいません（スワップ待ち）…"}]}
```

### 5.6 物理メモリを超える実行の時間

Boltz は working set が収まらなくなっても失敗しない。ページングしながら、メモリの速度ではなく
SSD の速度で進み続ける。1,278 トークンの予測がその状態にある間に実測した値は、16 KB ページで
30 秒あたり swap-in 563,094・swap-out 497,311、すなわち **約 565 MB/s**。常駐時のページ流量は
97 GB/s なので、**170 倍**の差になる。40 分の見積もりのジョブが 3 時間半経っても終わらなかった
理由はこれである。

かかる時間は「収まらない量 × working set を何周するか ÷ スワップ速度」で出す。

**収まらない量** = 回帰したピーク − 収まる容量。収まる容量は搭載メモリから macOS のぶん 4 GB を
引いた値（24 GB の機械で 20 GB）。Metal のワーキングセット上限ではない。あれは確保の可否を決める
もので、ここで起きたのは拒否ではなくページングである。

**周回数** = `diffusion_samples × sampling_steps + 80`。同じ実行から逆算した: 212 分 × 565 MB/s
= 7.2 TB、超過 25.6 GB で割って 280 周。`sampling_steps` が 200、リサイクル 4 のときの値なので、
拡散 1 ステップにつき 1 周 + トランク側の固定費という形に置いた。

**この係数の根拠は実測 1 件だけである。** 「40 分」を「4 時間」に変えるには十分で、下 2 桁を
信じるには足りない。

検算（この機械の全履歴を入力として）:

| ジョブ | トークン | 推定 | うちページング | 超過 | 実際 |
| --- | --- | --- | --- | --- | --- |
| 較正 76 | 76 | 1 分 | 0 | 0 | 36 秒 |
| 較正 304 | 304 | 3 分 | 0 | 0 | 116 秒 |
| 較正 608 | 608 | 9 分 | 0 | 0 | 389 秒 |
| ヌクレオソーム | 1,278 | **312 分** | 272 分 | 32.3 GB | 212 分で未完了 |
| Cas9 複合体 | 1,508 | 482 分 | 426 分 | 50.6 GB | 未実行 |
| フェリチン 12 量体 | 2,196 | 1,157 分 | 1,040 分 | 123.6 GB | 未実行 |

投入前の目安（`POST /api/estimate`）にもこの項を含める。`breakdown.paging` に秒数、
`memory.overage_gb` / `capacity_gb` / `passes` / `traffic_tb` に内訳が入る。**投入自体は止めない。**
止めるかどうかは人が決めることで、アプリの仕事は「40 分ではなく 5 時間だ」と先に言うことである。

---

## 6. 予測パイプライン

### 6.1 MSA の再利用

変異体は親と配列がほとんど同じなので、MSA を取り直す意味がない。`reuse_msa_for_variants` が
有効なとき、`find_reusable_msa()` が過去のジョブから「置換数が配列長の一定割合以内」の MSA を
探して再利用する。再利用時は ColabFold サーバーへのアクセスが発生しない。

### 6.2 実行

```
python -m oritatami.engines.supervise <親PID> -- \
  boltz predict <input>.yaml --out_dir <job>/out --cache ~/.boltz \
       --accelerator mps --output_format mmcif \
       --diffusion_samples N --recycling_steps N --sampling_steps N \
       --override --no_write_full_pde \
       [--use_msa_server --msa_server_url ...] [--use_potentials] [--seed N]
```

環境変数は `run_env()` が組み立てる。`PYTORCH_ENABLE_MPS_FALLBACK` は `mps_strict` の裏返しで、
0 にすると Metal カーネルのない演算が CPU に落ちた時点でジョブが失敗する。既定は 1（フォール
バックを許す）。黙って 3 倍遅くなるより落ちてほしい場合に 0 にする。

**精度について。** Boltz は MPS と CPU では常に float32 で動く（`main.py` が
`if accelerator in ("mps","cpu"): precision = 32`）。bf16-mixed は CUDA のみ。したがって半精度に
起因する数値破綻はこの構成では発生せず、精度を選ぶ設定も存在しない。

### 6.3 変異提案の修復

LLM が出す変異コードは、残基名が正しく位置が誤っているケースが多い（先頭メチオニンの有無、
UniProt 番号と construct 番号のずれ）。`seq.repair_mutations()` は以下の順で処理する。

1. 野生型残基が一致するものはそのまま通す。
2. 一致しないものが 2 件以上あり、同一のオフセット k（±12 以内）で全て説明できる場合、その
   オフセットを適用する。共有オフセットは証拠であり、最近傍探索は推測なので、前者を優先する。
3. それ以外は、前後 12 残基以内でその残基が実在する最も近い位置へ移す。
4. 12 残基以内に該当残基がなければ落とし、実際にそこにある残基名を添えて報告する。
5. 修復の結果 2 件が同じ位置に着地した場合、置換先が同じならまとめ、異なれば 2 件目を落とす。

修復した提案は `status: "warning"` のままで、`repaired` 配列に `"S257P → S252P (位置 257 は A、
最も近い S は 252)"` の形で理由が入る。適用される変異は元の提案と別物なので、黙って通さない。

---

## 7. 自律ループ

### 7.1 流れ

```
予測完了 → ウォッチドッグが検知
  → 結果を LLM に説明させる（autopilot_explain）
  → 変異提案を要求（autopilot_proposals_per_call 件）
  → 提案を検証・修復（§6.3）、ESM-2 で採点
  → ProteinMPNN で拒否権（mpnn_veto）
  → 禁止リストに触れるものを除外
  → governor を通過したものを予測ジョブとして投入
```

### 7.2 禁止リスト（対象分子ごとに生成）

pLDDT を最大化する探索は、予測器が自信を持てない部分＝機能を担う部分を削るのが最も安上がりに
なる。ユビキチンでの実測では、491 回の自律試行のうち 91 % が C 末端 G76 を、94 % が K63 を
書き換えていた。スコアは上がり、ユビキチンではなくなっていた。

`function_risk.suggest_protected()` は次の根拠から保護候補を生成し、それぞれに理由を付けて返す。

| 根拠 | 重み | 出典 |
| --- | --- | --- |
| 活性部位・結合部位・金属結合・ジスルフィド | critical | UniProt 注釈 |
| DNA 結合・ジンクフィンガー・翻訳後修飾・糖鎖付加 | high | UniProt 注釈 |
| 機能部位・モチーフ・シグナル・プロペプチド・膜貫通 | medium | UniProt 注釈 |
| 他鎖との界面残基 | high | Boltz の接触解析 |
| 系統の起点で pLDDT が閾値未満 | high | 起点ジョブの pLDDT |
| ESM-2 がどの置換も不自然と見る位置（下位 10 %） | high | 変異スキャン |
| N 末端 / C 末端（末尾 2 残基） | medium | 経験則 |

12 残基を超える幅の注釈は medium に落とす（25 残基の膜貫通ヘリックスは小さいタンパク質の
4 分の 1 を占めるため）。既定でチェックが入るのは critical と high。保護対象が配列長の 40 % を
超えると警告を出す。

### 7.3 governor

自律投入（`origin` が `autopilot_variant` または `pdb_watch`）は全て以下を通る。人間が直接押した
投入は通らない。

- **ファンアウト**: 1 ジョブあたりの変異数 `autopilot_max_variants_per_job`（既定 2）
- **深さ**: `autopilot_max_depth`（既定 1、0 で無制限）
- **日次予算**: ローリング 24 時間あたりの自律予測数 `autopilot_daily_budget`（既定 20、0 で無制限）
- **キュー長**: 同時に待機できる自律ジョブ数 `autopilot_max_queued`（既定 8）
- **空きディスク**: `autopilot_min_disk_gb`（既定 20 GB）を下回ったら受け付けない
- **実験タグ**: `autopilot_experiment` が設定されている間は、同じタグを持つ予測からしか分岐しない

`check()` は例外ではなく決定を返すので、呼び出し側が「なぜ手を引いたか」をログに残せる。

---

## 8. 結果の健全性検査

### 8.1 機能リスク（`GET /api/jobs/{id}/function_risk`）

| kind | severity | 条件 |
| --- | --- | --- |
| `empty` | critical | 成功扱いなのに構造モデルが無い |
| `nonfinite` | critical | 座標または pLDDT が NaN / inf（§8.2） |
| `chain_break` | high | 主鎖が途切れている（§8.2） |
| `clash` | high / medium | 原子が重なっている（§8.2） |
| `hit` | critical / high | 変異が保護対象残基（§7.2）に当たっている |
| `disorder_gain` | high | pLDDT の上がり幅の 60 % 以上が、親で乱れていた残基の書き換えによる |

`level` は critical があれば `danger`、それ以外に所見があれば `warn`、無ければ `ok`。
所見は severity 順に並べて返す。

`disorder_gain` は「平均が上がったか」ではなく「どこで上がったか」を見る。柔らかい末端だけを
整えて回る探索は、スコアが上がり続けながら分子としては劣化するため。

### 8.2 形状検査（`structure.geometry_check()`）

信頼度スコアは座標が使い物になるかを答えない。NaN の残基にも pLDDT は付き、原子が重なっていても
両方「自信あり」と出る。したがってファイルそのものを測る。予測完了時に第 1 モデルに対して実行し、
結果の `geometry` に保存する。1 モデルあたり約 2 ms。

```jsonc
{
  "model_index": 0,
  "atoms": 601,
  "nonfinite_atoms": 0,
  "nonfinite_examples": [],
  "other_models_nonfinite": [{"model_index": 2, "nonfinite_atoms": 4}],
  "clashes": 1,
  "severe_clashes": 0,
  "clashscore": 1.7,
  "worst_clashes": [{"a": "A/LYS6/CB", "b": "A/GLN41/CA", "dist": 2.26, "overlap": 0.53}],
  "chain_breaks": [{"chain": "A", "after": "I30", "before": "Q31", "dist": 22.1, "limit": 5.0}],
  "vdw_tolerance": 0.4
}
```

**衝突の定義。** 隣接しない残基に属する重原子の対について、距離が
`r1 + r2 − 0.4 Å`（水素結合・ハロゲン結合が成立しうる N/O/S/F 同士はさらに −0.6 Å）を下回るものを 1 件と数える。
0.4 Å は MolProbity と同じ許容幅。0.6 Å の上乗せが無いと、塩橋が全て衝突として計上される
（実測: 上乗せ無しではこの機械の 973 モデル中 141 件が該当し、中身はほぼ全てアルギニン–
アスパラギン酸対の 2.0–2.2 Å だった）。金属配位（2.0–2.2 Å）とジスルフィド（2.05 Å）は結合なので
除外する。`severe_clashes` は食い込みが 0.8 Å 以上のもの。`clashscore` は 1000 原子あたりの件数。

**主鎖の断裂。** 連続する残基の Cα 間距離がタンパク質で 5.0 Å、核酸（C1'）で 9.0 Å を超えたものを
断裂とする。B-DNA の一本鎖に沿った C1' 原子は約 5.4 Å 離れているため、核酸に protein の閾値を
当てると正しく組み上がった二重らせんが全て断裂扱いになる。

**NaN を検出した場合、衝突検査は行わない。** 座標に NaN があると近傍探索の結果が意味を持たないため。

0.2.1 以前に計算した結果には `geometry` が無い。`function_risk` を初めて開いた時点で測定し、
ジョブの結果に書き戻す。

### 8.3 物理補正（`use_potentials`）

Boltz の steering potentials。既定は **オフ**。この機械での実測（961 件、全て補正オフ）は以下。

| 指標 | 値 |
| --- | --- |
| 衝突ゼロの割合 | 60 % |
| clashscore 中央値 / 75 % 点 / 90 % 点 | 0.0 / 1.7 / 4.9 |
| 深い重なりを 1 か所以上含むもの | 44 件（4.6 %） |
| clashscore 10 以上 | 17 件（1.8 %） |
| 警告対象の合計 | 61 件（6.3 %） |

補正オンの実測データはこの機械には存在しない。Boltz の公式 FAQ も「衝突を減らすために有効化
できる」とだけ述べ、精度への影響も計算コストも数値を公開していない。したがって「オンにすると
どれだけ遅くなるか」は未確認である。

運用としては、既定オフのまま、§8.2 の警告が出た結果と、リガンド複合体・親和性予測のように
形状の妥当性が結論に直結するものを補正つきで再実行する形を推奨する。根拠は 3 点。

1. 6 割の結果は補正なしで既に衝突ゼロであり、残り 4 割の大半も clashscore 5 未満に収まる。
   全件に一律のコストを払って直すのは 6.3 % のためである。
2. 自律ループの選抜は pLDDT / ipTM で行われる。衝突の解消はこれらの指標をほとんど動かさないため、
   補正をオンにしても探索の進み方は改善しない。遅くなる分だけ世代数が減る。
3. どの結果に補正が要るかは、0.2.1 以降は警告として自動的に特定される。全件に保険をかける理由が
   なくなった。

---

## 9. メモリ管理

Apple Silicon では CPU と GPU が同じ物理メモリを共有するが、Metal が 1 プロセスに渡す量には
別途上限がある（`recommendedMaxWorkingSetSize`）。24 GB のこの機械では既定 17.76 GB。

### 9.1 計測

計測対象は RSS ではなく **physical footprint**。Metal のテンソルは IOAccelerator の確保に載るため
RSS に現れない。実測では 1,696 残基・3 鎖の予測で、同一時点に `ps` が 6.5 MB、カーネルが 27 GB
（生涯ピーク 30 GB）を報告した。RSS はこの値の過小評価ではなく、無関係な数字である。

supervisor が `proc_pid_rusage` の `ri_lifetime_max_phys_footprint` を読み、ジョブディレクトリに
`peak_memory.txt`（ピーク GB）、`cpu_seconds.txt`（消費 CPU 秒）、`live_memory.json`（3 秒ごとの
現在値・スワップ量・空きディスク・消費 CPU 秒・実効効率）を書く。ピークは所要時間の推定と同様、
履歴から回帰してトークン数から予測する。

`cpu_seconds.txt` には合計と user のぶんを空白区切りで書く。合計だけでは、その実行が計算して
いたのかページングしていたのかが分からないため（§5.5）。

CPU 時間は `ri_user_time` と `ri_system_time`（スロット 2 と 3）から取る。**この 2 つの単位は
ナノ秒ではなく mach absolute time** で、この機械では 1 tick = 125/3 ns。ナノ秒として読むと約 42 分の 1 に
なり、それらしい値に見えてしまうので、`mach_timebase_info` を読んで換算する（`ps -o time=` との
突き合わせで検証済み: 0.8e9 + 48.6e9 tick = 2,058 秒、ps の表示と一致）。

ユーザー時間とシステム時間の比もそのまま診断になる。正常な計算はユーザー時間が主だが、スワップ
待ちのジョブはカーネルのページフォルト処理に落ちるため、実測でシステム時間が 98 % を占めた。

### 9.2 解放

予測ジョブの直前に、LLM のモデルを降ろし（`keep_alive=0`）、ESM-2 が使用中でなければ降ろす。
使用中で降ろせない場合も、torch の MPS アロケータが抱えている未使用ブロックは返す
（`esm.release_cache()` → `torch.mps.empty_cache()`、実測で 1 回 1.0 GB）。torch が未 import の
場合は何もしない。import 自体が解放量より高くつくため。

同じ解放を全ジョブの終了時にも行う。

### 9.3 自プロセスの推移

Boltz は別プロセスなので終了時に全て返す。連続実行で太りうるのはアプリ本体だけなので、
ジョブ終了ごと（成功・失敗の両方）に自プロセスの footprint を記録する。
`GET /api/system/memory` が推移を返し、最初と最後の差が 4 GB 以上なら `climbing: true` を立てる。

```jsonc
{"current_gb": 0.36, "samples": 12, "first_gb": 0.34, "last_gb": 0.41,
 "growth_gb": 0.07, "climbing": false, "growth_warn_gb": 4.0,
 "series": [{"job": 1, "gb": 0.34}],
 "torch_mps": {"driver_allocated_gb": 0.0, "in_use_gb": 0.0}}
```

### 9.4 Metal の上限変更

`POST /api/system/gpu` が `sysctl iogpu.wired_limit_mb` を変更する。macOS 自身の認証ダイアログ
（`do shell script ... with administrator privileges`）を出すため、パスワードはアプリを経由しない。
搭載メモリから 4 GB を引いた値を上限とする。再起動で既定に戻る。Metal はプロセスが Metal デバイスを
開いた時点の値を読むので、反映にはアプリの再起動が要る。

---

## 10. 局所 LLM（llama.cpp）

アプリは推論を `llama-server` のプロセスとして自分で管理する（Ollama のような常駐
デーモンは持たない）。1 プロセスが 1 モデルを面倒見るので、モデル切替は再起動、メモリ解放は
プロセス終了。会話は OpenAI 互換の `POST /v1/chat/completions` で、構造化出力は
`response_format` の JSON スキーマで強制する（組めないテンプレートでは `json_object` に落とす）。
ランタイムは上流 llama.cpp のピン留め版（b11158）。Ollama 同梱の `llama-server` は gemma3・
qwen3.5 のチャットテンプレートを組めない旧フォークなので使わない。

### 10.1 バイナリの解決順

1. 設定の `ollama_bin`（明示指定）
2. .app に同梱されたもの（`ORITATAMI_OLLAMA_DIR`）
3. このアプリがダウンロードしたもの（`<データディレクトリ>/ollama`）
4. マシンにインストールされているもの（PATH）

`GET /api/llm/status` が `source` として `bundled | downloaded | system | none` のいずれかを返す。
どれも無い場合は `POST /api/llm/install` で上流 llama.cpp のリリースアーカイブ
（`llama-b11158-bin-macos-arm64.tar.gz`、約 12 MB）をバックグラウンドで取得する。
システム全体には何もインストールせず、押されない限り何も始まらない。展開には
`/usr/bin/tar` を使う（コード署名を保つため）。

### 10.2 モデルの取得とメモリの譲り渡し

モデルは平文の GGUF ファイルで、解決順は 明示パス → `<データディレクトリ>/models` →
`~/.ollama` の既存ストア（manifest→blob 参照。Ollama で pull 済みのモデルはそのまま使える）。
ダウンロードは `name:tag` の表記を保ったまま、よく使うモデルは Hugging Face の無認可 GGUF
（`_KNOWN_MODELS` 表）から直接取り、それ以外は `registry.ollama.ai` の manifest→blob に
フォールバックする。落とした GGUF はロード前にヘッダ検査し、Ollama フォーク専用の変換
（gemma3 の layer_norm イプシロン欠落、qwen35 の rope セクション数不一致）は「非互換」と
判定して再取得を促す — 上流 `llama-server` はそれらをロードできない。

予測ジョブが走っている間は `heavy` フラグが立ち、全ての LLM リクエストが回答し終えた時点で
サーバーを止める（従来の `keep_alive=0` と同じ契約）。通常利用では 15 分のアイドル経過で
モデルを降ろす。`unload_model()` はサーバーが静かになるまで最大 15 秒待つ。

### 10.3 モデル能力への適応

GGUF ヘッダを直接読んで `capabilities`（`tokenizer.chat_template` 中の思考スイッチの有無）と
学習コンテキスト長（`*.context_length`）を得る — サーバー未起動でも判定できる:

- `thinking` に非対応のモデル（Gemma 系など）では `llm_think` が有効でも `enable_thinking` を
  送らない。設定画面にも非対応の警告が出る
- 起動するコンテキスト窓 (`--ctx-size`) は、既定 32,768 とプロンプト見積もりの大きい方を
  モデルの学習窓で頭打ちにする。学習窓を超えた KV キャッシュはメモリを食うだけで品質は上がらない
- 別タグのモデルを読み込む際は同居中のモデルを全て解放する（`make_room_for` = プロセス再起動）。
  同ファミリでも別サイズは別モデルであり、9B を残したまま 27B を載せると実測で CPU/GPU に
  分割配置され応答不能になった
- 出力が JSON として読めないときは一度だけ、壊れた出力を見せてスキーマに従う JSON だけを
  出力し直させる。直らなければ従来通りエラーになる
- `raise_exception` を内蔵するテンプレート（Gemma の発言順チェックなど）では llama-server が
  JSON スキーマの文法を組めず 400 を返す。その応答を検知したら `json_object` モードに
  切り替えてやり直す — 妥当な JSON であること自体は強制できる

---

## 11. API コントラクト

全て `127.0.0.1` のみ。認証は無い（ローカル専用）。エラーは `{"detail": "<日本語の説明>"}` を返し、
分類がある場合のみ `code` が付く。

| 例外 | ステータス | code |
| --- | --- | --- |
| `ValueError` / `SequenceError` / `ChemError` | 400 | — |
| `KeyError`（対象が無い） | 404 | — |
| `llm.SafeguardError` | 422 | `safeguard` |
| `sources.SourceError`（外部 API） | 502 | — |
| `llm.LlmError` | 503 | — |
| 未捕捉 | 500 | `internal` |

### 11.1 ジョブ

| メソッド | パス | 引数 | 返り値 |
| --- | --- | --- | --- |
| `POST` | `/api/jobs/predict` | `{spec, title?, parent_id?, origin?}` | ジョブ概要 |
| `POST` | `/api/jobs/predict/batch` | `{spec, variants:[{chain,mutations}], parent_id?, origin?}` | 投入したジョブの配列 |
| `POST` | `/api/jobs/scan` | `{sequence, chain?, label?, parent_id?}` | ジョブ概要 |
| `POST` | `/api/jobs/refine` | `{sequence, label?, rounds=6, fraction=0.1, temperature=1.0, fixed_positions=[], seed?, origin?}` | ジョブ概要 |
| `GET` | `/api/jobs` | `summary?`, `limit?` | ジョブ一覧 |
| `GET` | `/api/jobs/changes` | `rev?`, `timeout?` | ロングポーリング。変化があるまで待つ |
| `GET` | `/api/jobs/eta` | — | §5.5 |
| `GET` | `/api/jobs/{id}` | — | ジョブ全体（spec + result） |
| `PATCH` | `/api/jobs/{id}` | `{title?, starred?}` | 更新後のジョブ |
| `DELETE` | `/api/jobs/{id}` | — | `{deleted}` |
| `POST` | `/api/jobs/{id}/cancel` | — | ジョブ概要 |
| `POST` | `/api/jobs/{id}/retry` | `{msa?, accelerator?, diffusion_samples?, new_seed?}` | 新しいジョブ |
| `POST` | `/api/jobs/{id}/reorder` | `{action: top\|up\|down\|bottom}` | ジョブ概要 |
| `POST` | `/api/jobs/queue/cancel_all` | `{kind?, include_running=false}` | `{cancelled, count}` |
| `GET` | `/api/jobs/{id}/log` | — | 実行ログ（text/plain） |
| `GET` | `/api/jobs/{id}/files/{rel}` | — | ジョブディレクトリ内のファイル |
| `GET` | `/api/jobs/{id}/structure.pdb` | `model?` | PDB 形式に変換して返す |
| `GET` | `/api/jobs/{id}/export.zip` | — | 結果一式 |
| `POST` | `/api/jobs/{id}/export` | — | ダウンロードフォルダに書き出す |
| `POST` | `/api/jobs/{id}/reveal` | — | Finder で開く |
| `GET` | `/api/jobs/{id}/function_risk` | — | §8.1 |
| `GET` | `/api/jobs/{id}/protected_suggest` | `chain?` | §7.2 |
| `GET` | `/api/jobs/{id}/autopilot` | — | この結果に対する自律解析の記録 |

### 11.2 作業台の材料

| メソッド | パス | 引数 |
| --- | --- | --- |
| `GET` | `/api/uniprot/search` | `q` |
| `GET` | `/api/uniprot/{accession}` | — |
| `GET` | `/api/pdb/search` | `q` |
| `POST` | `/api/import/pdb` | `{id, assembly=false}` |
| `POST` | `/api/import/afdb` | `{id}` |
| `POST` | `/api/import/upload` | `{filename, content}` |
| `GET` | `/api/chem/pubchem` | `name` |
| `GET` | `/api/chem/ccd/{code}` | — |
| `POST` | `/api/chem/describe` | `{smiles}` |
| `POST` | `/api/sequence/validate` | `{sequence, type=protein}` |
| `POST` | `/api/sequence/mutate` | `{sequence, mutations}` |
| `POST` | `/api/sequence/diff` | `{a, b}` |
| `POST` | `/api/estimate` | `{spec}` → 下記 |
| `POST` | `/api/compare` | `{fixed, moving}` → 重ね合わせと RMSD |

`/api/estimate` の返り値。投入前に「待てる時間か」「載るメモリか」を判断するためのもので、
所要時間は履歴からの回帰（`basis: "history"`、履歴 2 件未満なら `"default"`）。

```jsonc
{"seconds": 420, "low": 280, "high": 630,
 "breakdown": {"startup": 20, "msa": 0, "structure": 400, "affinity": 0},
 "basis": "history", "samples": 137, "tokens": 261, "needs_msa_search": false,
 "memory": {"peak_gb": 12.4, "total_gb": 24.0, "level": "warn", "basis": "history",
            "samples": 137, "beyond_physical": false, "applecare": false},
 "msa_reuse": true, "queued_ahead": 2}
```

### 11.3 LLM と自律

| メソッド | パス | 引数 |
| --- | --- | --- |
| `POST` | `/api/assistant/ask` | `{thread_id?, mode=chat, message, workbench, job_id?, scan_job_id?, focus_chain?, count=3, heavy=false}` |
| `GET` | `/api/assistant/threads` | — |
| `GET`/`DELETE` | `/api/assistant/threads/{tid}` | — |
| `GET` | `/api/llm/status` | — |
| `POST` | `/api/llm/start` | —（必要なら取得も行う） |
| `POST` | `/api/llm/install` | — |
| `POST` | `/api/llm/pull` | `{model}` |
| `GET` | `/api/llm/calls` | `limit?`, `origin?`, `full?` |
| `POST` | `/api/llm/calls/export` | — |
| `GET` | `/api/autopilot/status` | — |

`mode` は `chat | mutations | complex | design | explain`。提案は以下の形で返る。

```jsonc
{"type": "mutation_set", "title": "...", "rationale": "...",
 "status": "ok | warning | invalid",
 "issues": ["..."], "repaired": ["S257P → S252P (位置 257 は A、最も近い S は 252)"],
 "chain": "A", "mutations": ["K48R"], "rejected": ["..."],
 "esm": {"total_llr": -8.6, "per_mutation": [{"mutation": "K48R", "llr": -8.6}]},
 "apply": {"action": "mutate", "chain": "A", "mutations": ["K48R"]}}
```

`type` は `mutation_set | add_ligand | add_protein | add_nucleic | new_protein`。
`apply` が null でない提案だけが適用可能。

### 11.4 システム・設定

| メソッド | パス | 用途 |
| --- | --- | --- |
| `GET` | `/api/health` | バージョン、Boltz の所在と重みの有無、データディレクトリ |
| `GET` | `/api/ping` | 生存確認 |
| `GET`/`PATCH` | `/api/settings` | 設定の取得・更新（§12） |
| `GET` | `/api/storage` | ジョブ・インポート・MSA キャッシュ・Boltz キャッシュの使用量 |
| `POST` | `/api/storage/cleanup` | `{intermediate=true, aligned_older_than_days=0, delete_failed_jobs=false}` |
| `GET`/`POST` | `/api/system/gpu` | §9.4 |
| `GET` | `/api/system/memory` | §9.3 |
| `POST` | `/api/system/memory/release` | 保持中のメモリを今すぐ返す |
| `GET` | `/api/system/ssd` | SSD の摩耗とスワップ実行時の書き込み見積もり |
| `GET` | `/api/leaderboard` | `metric?`（`mean_plddt` / `core_plddt` / `confidence_score` / `iptm` / `ptm` / `complex_plddt`）, `limit?`（既定 200） |
| `GET`/`POST` | `/api/library` ・ `DELETE /api/library/{id}` | 作業台の保存と読み出し |
| `GET`/`DELETE` | `/api/searches` | 検索履歴 |
| `GET` | `/api/history` | 検索履歴（集計） |
| `GET`/`POST` | `/api/pdb_watcher/config` ・ `POST /api/pdb_watcher/poll_now` | PDB ウォッチャー |
| `POST` | `/api/notify` | macOS 通知 |
| `POST` | `/api/files/save` ・ `GET /api/files/imports/{name}` | ファイルの保存・取得 |

---

## 12. 設定

`<データディレクトリ>/settings.json`。`PATCH /api/settings` は差分を受け取り、値の型と範囲を
検証してから書き戻す。真偽値の文字列は `"true"/"yes"/"on"/"1"` と `"false"/"no"/"off"/"0"/""`
だけを受け付け、それ以外は 400 にする（かつて `"false"` が True として通っていた）。

`settings_revision` は既定値の変更を既存の設定ファイルに一度だけ反映するための版番号。
現在の値は 1（0 は 1 サンプル / 4 リサイクル / 200 ステップの軽量プリセット導入前）。

主要な項目は §7.3、§8.3、§9、§10 に記述したものに加え、以下。

| キー | 既定 | 意味 |
| --- | --- | --- |
| `llm_model` | `gemma3:4b` | 常用モデル |
| `llm_model_heavy` | `""` | 「じっくり答える」用。空なら機能を出さない。自律ループでは使わない |
| `llm_log_limit` | 5000 | LLM 呼び出しログの保持件数。0 で無効 |
| `esm_model` | `facebook/esm2_t33_650M_UR50D` | 変異スコアリングのモデル |
| `mps_strict` | false | true で CPU フォールバック時にジョブを失敗させる |
| `mps_memory_ratio` | 0.0 | MPS アロケータの上限倍率。0 で無制限 |
| `mpnn_enabled` / `mpnn_veto` | true / 0.06 | 逆折り畳みによる拒否権 |
| `autopilot_selection` | `greedy` | `greedy` か `explore`（試行の少ない位置を優先） |
| `autopilot_improvement_metric` | `auto` | 複数鎖なら ipTM、単鎖なら平均 pLDDT |
| `autopilot_improvement_delta` | 2.0 | 改善とみなす差（pLDDT 換算） |
| `autopilot_min_esm_llr` | −15.0 | ESM-2 スコアの下限。選抜ではなく極端な裾の切り落とし |
| `applecare` | false | SSD 摩耗警告の文面だけを変える。動作には影響しない |

---

## 13. ファイル配置

### 13.1 データディレクトリ

`~/Library/Application Support/Oritatami`（`ORITATAMI_HOME` で上書き可）。

```
oritatami.sqlite3      ジョブ・ライブラリ・スレッド・LLM ログ・検索履歴
settings.json          設定
instance.json / .lock  単一インスタンス制御
oritatami.log          アプリのログ
llama-server.log       起動した llama-server のログ
jobs/<job_id>/         入力 YAML、MSA、Boltz の出力、実行ログ、メモリ計測
imports/               取り込んだ構造と重ね合わせ結果
msa-cache/             再利用可能な MSA
ollama/                自己取得したランタイム（llama-server。同梱版・システム版がない場合）
```

Boltz の重みは `~/.boltz`（`boltz_cache`）。構造の重みが約 2 GB、親和性の重みと化学
辞書を含めた全体で約 6 GB。書き出し先の既定は
`~/Downloads/Oritatami`（`ORITATAMI_EXPORT_DIR`）。

### 13.2 .app の構成

```
Oritatami.app/Contents/
  MacOS/Oritatami          ランチャー（bash）
  Resources/
    runtime/               CPython 3.12（uv 由来）
    venv/                  依存ライブラリ一式＋ oritatami パッケージ
    frontend/dist/         ビルド済みフロントエンド
    ollama/                同梱版のみ
    Oritatami.icns
```

ランチャーは自分の位置から全てのパスを解決し、`pyvenv.cfg` の `home` がバンドル内を指していない
場合は書き換える。したがって .app を移動・改名しても壊れない。`ORITATAMI_FRONTEND_DIST` と
`ORITATAMI_OLLAMA_DIR` を輸出してから `python -m oritatami --app` を exec する。

ビルドは `scripts/make_app.sh`。`--no-ollama` で同梱を省く。`--link` は開発用（venv を参照する）。
ビルド後にバンドル内をリポジトリのパスで grep し、外部参照が残っていないことを確認する。
署名は行っていないため、初回起動は右クリック →「開く」が必要である。

---

## 14. 既知の制約

- **未署名・未公証。** Apple Developer Program が必要なため実施していない。Gatekeeper の回避操作が
  初回に要る。
- **物理補正オンの実測が無い。** §8.3 の推奨は、補正オフ側 961 件の実測と、オン側について公開
  情報が無いという事実に基づく判断であり、A/B 実測ではない。
- **CUDA 経路は未検証。** Boltz 側では bf16-mixed が使われるが、この構成では通らない。
- **環状ペプチド（`cyclic`）** は Boltz 側の対応が限定的で、オンにすると結果が壊れることがある。
  オフは無害なので、UI では有効化時に警告する。
- **リガンドのトークン数** は CCD の場合 30 と仮定しており、実際の重原子数とは異なる。推定時間と
  推定メモリの誤差要因になる。
- **ESM-2 はアプリと同一プロセス。** スキャン中は解放できないため、その間に始まる予測ジョブは
  ESM-2 のぶんだけ狭いメモリで走る。

---

## 15. 変更履歴

| バージョン | 変更 |
| --- | --- |
| 0.2.3 | 物理メモリ超過分からページング時間を推定して所要時間に加算、領域判定を user 時間の割合に変更 |
| 0.2.2 | 実行中ジョブの残り時間を実測ベースに（CPU 秒の記録）、推定不能なキューを下限として表示 |
| 0.2.1 | 形状検査（NaN・衝突・主鎖断裂）と結果警告、変異提案の位置修復、ジョブ間の Metal キャッシュ解放と自プロセスのメモリ推移記録 |
| 0.2.0 | 二重押し防止、自律ループの設定画面と禁止リスト自動生成、キュー全体の終了見込み、自己完結型 .app、Ollama の同梱と自己取得 |
