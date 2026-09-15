"""Paths and user-editable settings.

Settings live in ``<home>/settings.json``. Every key has a default here, so a
missing or partial file is valid; unknown keys in the file are ignored.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import threading
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

APP_NAME = "Oritatami"
log = logging.getLogger("oritatami.config")

# Scores the autopilot can use to decide a variant beat its parent. "auto" picks per job.
IMPROVEMENT_METRICS = ("auto", "mean_plddt", "core_plddt", "iptm", "ptm",
                      "confidence_score", "complex_plddt")

# How the autonomous loop moves through sequence space.
#   climb — always branch from the best result so far. A non-improving attempt is
#           abandoned and the next-ranked candidate from the peak is tried instead.
#   walk  — always continue from the newest result. Explores more widely, but measured
#           over 51 generations it drifted downhill: best 93.37 at generation 3,
#           91.43 by generation 51, 26 mutations away from wild type.
AUTOPILOT_STRATEGIES = ("climb", "walk")


def app_home() -> Path:
    env = os.environ.get("ORITATAMI_HOME")
    if env:
        base = Path(env).expanduser()
    else:
        base = Path.home() / "Library" / "Application Support" / APP_NAME
    base.mkdir(parents=True, exist_ok=True)
    return base


def jobs_dir() -> Path:
    p = app_home() / "jobs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def imports_dir() -> Path:
    p = app_home() / "imports"
    p.mkdir(parents=True, exist_ok=True)
    return p


def msa_cache_dir() -> Path:
    p = app_home() / "msa-cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def exports_dir() -> Path:
    """Where exported bundles and screenshots go: ~/Downloads/Oritatami, or the app home if that is not writable."""
    env = os.environ.get("ORITATAMI_EXPORT_DIR")
    candidates = [Path(env).expanduser()] if env else []
    candidates += [Path.home() / "Downloads" / APP_NAME, app_home() / "exports"]
    for p in candidates:
        try:
            p.mkdir(parents=True, exist_ok=True)
            probe = p / ".write-test"
            probe.write_text("ok", "utf-8")
            probe.unlink()
            return p
        except OSError:
            continue
    raise RuntimeError("書き出し先のフォルダを作成できません")


def repo_root() -> Path:
    # backend/oritatami/config.py -> repo root
    return Path(__file__).resolve().parents[2]


def frontend_dist() -> Path:
    env = os.environ.get("ORITATAMI_FRONTEND_DIST")
    if env:
        return Path(env)
    return repo_root() / "frontend" / "dist"


@dataclass
class Settings:
    # Local LLM (Ollama)
    ollama_url: str = "http://127.0.0.1:11434"
    # Empty = find it: the copy inside the .app, then the one this app downloaded, then
    # whatever is installed on the machine.
    ollama_bin: str = ""
    llm_model: str = "qwen3.5:9b"
    # Optional second model, used only when the person presses じっくり答える. Empty = the
    # button stays off. It is never used by the autopilot: a slower model there would cut
    # the number of variants the loop gets through, for prose nobody reads.
    llm_model_heavy: str = ""
    # 0 disables the log entirely; otherwise the newest N calls are kept (~12 KB each).
    llm_log_limit: int = 5000
    llm_think: bool = False
    llm_temperature: float = 0.6
    # Structure prediction (Boltz-2)
    boltz_bin: str = ""  # empty = the boltz next to the running interpreter
    boltz_cache: str = str(Path.home() / ".boltz")
    accelerator: str = "auto"  # auto | mps | cpu
    # Off: an op with no Metal kernel silently runs on the CPU. On: it fails the job, so a
    # regression shows up as an error instead of as a run that is quietly three times slower.
    mps_strict: bool = False
    # Ceiling for the MPS allocator, as a multiple of what Metal recommends (20.0 GB here
    # with iogpu.wired_limit_mb=20480, 17.76 GB at the default). PyTorch's own default is
    # 1.7, and that is exactly where a 1,696-residue three-chain complex died: 17.76 x 1.7
    # = 30.2 GB, the job asked for more, and the allocator refused rather than letting
    # macOS swap. Past physical memory the machine does not fail, it slows — so the refusal
    # threw away 16 minutes of work to avoid something survivable. 0 removes the ceiling.
    # The cost is not free and is not hidden: a job that swaps writes about 190 MB/s to the
    # SSD for as long as it runs, and the workbench says so before it is submitted.
    mps_memory_ratio: float = 0.0
    # Not a knob, a fact the app cannot look up: whether this machine is under AppleCare+.
    # It changes nothing about how anything runs — it only decides how bluntly the swap
    # warning is worded. Without cover, a worn-out SSD is a logic board at full price,
    # because Apple Silicon storage is soldered. With cover it is merely arguable: the
    # contract excludes "通常の消耗、または通常の経年劣化に起因する故障" and Apple publishes no
    # write limit to test that against, so nobody can say in advance which side wear falls on.
    applecare: bool = False
    # Inverse folding as a veto on the obviously destructive. Measured: it ranks I44A,
    # L67D, G75C and G76C worst and the benign surface changes best — the ordering the
    # pLDDT and ESM-2 both get wrong. Off means the loop keeps their shared blind spot.
    # greedy: take the best-looking candidate. explore: prefer positions the run has
    # barely tried. With 4.3% of attempts improving anything, the greedy choice keeps
    # re-testing the same handful of positions — 23% of all attempts went to six of them.
    autopilot_selection: str = "greedy"   # greedy | explore
    # Boltz already computes which residues touch another chain, and nothing but the prose
    # pass ever read it. In a complex they are the one part whose job is known.
    autopilot_protect_interfaces: bool = True
    mpnn_enabled: bool = True
    # Reject a candidate whose inverse-folding score is worse than the parent's by more
    # than this. 0 disables the veto and leaves only the re-ranking.
    mpnn_veto: float = 0.06
    msa_server_url: str = "https://api.colabfold.com"
    diffusion_samples: int = 1
    recycling_steps: int = 4
    # Bumped when a default changes in a way an existing settings.json should follow once.
    # 0 = written before the light first-pass preset (1 sample / 4 recycles / 200 steps).
    settings_revision: int = 1
    sampling_steps: int = 200
    use_potentials: bool = False
    reuse_msa_for_variants: bool = True
    cleanup_intermediate: bool = True  # delete Boltz's processed/ and raw MSA search files after a job succeeds
    # Protein language model (mutation scoring / sequence refinement)
    esm_model: str = "facebook/esm2_t33_650M_UR50D"
    esm_device: str = "auto"  # auto | mps | cpu
    # Desktop integration
    notify_on_finish: bool = True  # macOS notification when a job finishes while the window is in the background
    # Autopilot: guards on the autonomous predict → analyse → mutate loop
    autopilot_enabled: bool = True
    # Fan-out per analysed job. With an unlimited depth this is the difference between a
    # walk and an explosion: at 1 the loop advances one lineage step at a time and the
    # queue stays flat, at 2 each generation doubles it and the queue never drains,
    # because the predict lane runs one job at a time.
    autopilot_max_variants_per_job: int = 2
    # How many generations deep the chain may go. 0 = unlimited, for running the loop
    # continuously until it is switched off; pair that with a fan-out of 1.
    autopilot_max_depth: int = 1
    autopilot_strategy: str = "climb"            # climb | walk — see AUTOPILOT_STRATEGIES
    # Name of the search currently running. When set, the loop may only branch from
    # predictions carrying the same tag, so a new experiment does not silently continue
    # the previous one's lineage — or adopt a one-off job someone ran by hand, which is
    # exactly what happened the first time a second run was started.
    autopilot_experiment: str = ""
    # Hill climb only: how many children one branch point may spend before the search
    # gives up on it and drops to the next-best result. Without it the climb sticks to
    # single mutations of one sequence forever whenever that sequence is already good —
    # ubiquitin's wild type outscored all 72 variants tried against it, so nothing ever
    # moved. 0 = never give up (a pure hill climb).
    autopilot_climb_patience: int = 8
    autopilot_daily_budget: int = 20             # autonomous predict jobs per rolling 24 h; 0 = unlimited
    # Hard stop on how many autonomous jobs may sit queued at once. Even with a fan-out
    # of 1 a burst (the PDB watcher, a manual batch) can stack work up; past this the
    # governor stops accepting until the lane drains.
    autopilot_max_queued: int = 8
    autopilot_min_disk_gb: float = 20.0          # refuse autonomous submits below this much free disk
    # ESM-2 log-likelihood-ratio floor — deliberately loose, and it is NOT the selection
    # mechanism: ranking picks the winners. Two measurements say a tight floor does more
    # harm than good. (1) The scale is protein-dependent — over every single mutant of
    # ubiquitin the median is -7.0 and only 2.4 % score above zero, so anything near
    # zero rejects almost everything. (2) Against Boltz, ESM only correlates weakly
    # (Pearson r = 0.34 over 9 ubiquitin mutants): a -10 floor would have rejected the
    # best-scoring mutant of that set (Q40V, LLR -10.0, +0.8 pLDDT) while letting the one
    # genuinely destructive mutant through (L67R, LLR -9.7, -9.4 pLDDT). So the default
    # only trims the extreme tail; raise it if you want a stricter pre-filter.
    # The loop asks the LLM twice per generation: once to explain the result, once for
    # mutations. Measured over 7 generations the explain call is 27% of the wall clock
    # (29.5 s of 111 s) and its prose is not stored anywhere; across 526 analyses it
    # supplied the top-scoring candidate 25% of the time, so turning it off trades a
    # little proposal diversity for roughly a third more generations per hour.
    #
    # Buying that diversity back by raising proposals_per_call does NOT pay: measured,
    # 3 -> 6 proposals took the mutations call from 39.5 s to 177 s. Twice the proposals
    # cost four and a half times the time, because a longer reply is more constrained
    # tokens and the schema decoder is the slow part. Leave this at 3.
    autopilot_explain: bool = True
    autopilot_proposals_per_call: int = 3
    autopilot_min_esm_llr: float = -15.0
    # Residues the search may not touch, as positions or codes ("K63, G75, G76").
    # Left open, a pLDDT-driven search deletes exactly the residues that make a protein
    # work: 491 predictions of ubiquitin mutated the essential C-terminal G76 in 91% of
    # them and the K63 signalling lysine in 94%, because a flexible tail is where the
    # cheapest confidence lives. The score went up; the protein stopped being ubiquitin.
    autopilot_protected_residues: str = ""
    # Same hole, found automatically: any residue that is disordered in the lineage root
    # is cheap confidence, so the search is not allowed to buy it.
    autopilot_protect_disordered: bool = True
    autopilot_disorder_plddt: float = 70.0
    autopilot_notify_improvement: bool = True    # notify when a variant beats its parent
    # Which score decides "better than the parent". "auto" uses ipTM for multi-chain
    # predictions (interface quality is the point of a complex) and mean pLDDT otherwise.
    autopilot_improvement_metric: str = "auto"
    # Gain that counts as an improvement, on the 0–100 pLDDT scale. Metrics that live on
    # 0–1 (ipTM, pTM, confidence) are converted, so 2.0 means +2 pLDDT or +0.02 ipTM.
    autopilot_improvement_delta: float = 2.0
    # Automatic retry of transient failures (network / MSA server)
    job_auto_retry: bool = True
    job_max_retries: int = 2

    def public(self) -> dict[str, Any]:
        return asdict(self)


_lock = threading.Lock()
_settings: Settings | None = None


def _settings_path() -> Path:
    return app_home() / "settings.json"


def _coerce(name: str, value: Any, default: Any) -> Any:
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            # Anything unrecognised has to be an error, not a silent True. The old code
            # returned `text in (...)`, so "false", "off" and "はい" all switched the
            # setting ON — the two that read as "no" being the dangerous ones.
            low = value.strip().lower()
            if low in ("1", "true", "yes", "on"):
                return True
            if low in ("0", "false", "no", "off", ""):
                return False
            raise ValueError(f"{label(name)} には true か false を指定してください (受け取った値: {value!r})")
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value)
        raise ValueError(f"{label(name)} には true か false を指定してください")
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    if isinstance(default, str):
        return _clean_text(name, value)
    raise TypeError(f"unsupported setting type for {name}")


# Free-text settings had no bound at all: a megabyte of control characters went straight
# into settings.json, and a model name of "<script>alert(1)</script>" silently broke every
# assistant call until someone noticed. Length and printability are checked for all of
# them; the ones whose shape is known are checked against it too.
# Errors from here are shown to the user verbatim in the settings dialog, so they have
# to name the field the way the dialog labels it — not by its internal key.
_LABELS = {
    "ollama_bin": "Ollama の場所",
    "diffusion_samples": "既定のサンプル数", "recycling_steps": "既定のリサイクル",
    "sampling_steps": "既定の拡散ステップ", "llm_model": "使うモデル",
    "llm_model_heavy": "じっくり答えるときのモデル",
    "llm_log_limit": "やり取りの保存件数",
    "mps_strict": "CPU へ落ちたら失敗させる",
    "mps_memory_ratio": "GPU メモリの上限倍率",
    "applecare": "AppleCare+ に加入している",
    "autopilot_selection": "候補の選び方",
    "autopilot_protect_interfaces": "界面の残基を保護する",
    "mpnn_enabled": "逆折り畳みで確認する",
    "mpnn_veto": "逆折り畳みの却下ライン",
    "llm_temperature": "温度", "ollama_url": "Ollama URL",
    "msa_server_url": "MSA サーバー", "esm_model": "ESM-2 のモデル",
    "boltz_cache": "キャッシュ (重み・化学辞書)",
    "autopilot_protected_residues": "変更を禁止する残基",
    "autopilot_experiment": "実験名",
    "autopilot_max_variants_per_job": "1ジョブあたりの変異体数",
    "autopilot_max_depth": "世代の上限", "autopilot_max_queued": "待機ジョブの上限",
    "autopilot_daily_budget": "24時間あたりの自律ジョブ上限",
    "autopilot_min_disk_gb": "空き容量の下限", "autopilot_min_esm_llr": "ESM-2 スコアの下限",
    "autopilot_climb_patience": "山登りの我慢の手数",
    "autopilot_proposals_per_call": "1回に出させる提案数",
    "autopilot_disorder_plddt": "乱れた残基のしきい値",
    "autopilot_improvement_delta": "改善とみなす差", "job_max_retries": "再試行の回数",
}


def label(name: str) -> str:
    return _LABELS.get(name, name)
MAX_SETTING_CHARS = 2000
# Blanking one of these does not disable a feature, it breaks it: an empty model name
# makes every call fail with "model is required" and nothing says why.
_REQUIRED_TEXT = frozenset({"llm_model", "esm_model", "ollama_url", "msa_server_url"})
_MODEL_NAMES = frozenset({"llm_model", "esm_model"})
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_SETTING_SHAPES = {
    "llm_model": (re.compile(r"^[A-Za-z0-9._:@/+-]{1,200}$"),
                  "モデル名に使える文字は英数字と . _ : @ / + - です"),
    "ollama_url": (re.compile(r"^https?://[^\s]{1,300}$"), "http:// か https:// で始まる URL にしてください"),
    "msa_server_url": (re.compile(r"^https?://[^\s]{1,300}$"), "http:// か https:// で始まる URL にしてください"),
    "esm_model": (re.compile(r"^[A-Za-z0-9._:@/+-]{1,200}$"), "モデル名に使える文字は英数字と . _ : @ / + - です"),
    "llm_model_heavy": (re.compile(r"^([A-Za-z0-9._:@/+-]{1,200})?$"),
                        "モデル名に使える文字は英数字と . _ : @ / + - です (空欄で無効)"),
    "autopilot_protected_residues": (re.compile(r"^[A-Za-z0-9,;\s]{0,500}$"),
                                     "残基の指定は K63, G76 のように英数字とカンマで書いてください"),
    "autopilot_experiment": (re.compile(r"^[\w .:@/+-]{0,80}$"), "実験名は80文字以内の英数字・かな漢字で指定してください"),
}


def _clean_text(name: str, value: Any) -> str:
    text = str(value)
    if len(text) > MAX_SETTING_CHARS:
        raise ValueError(f"{label(name)} が長すぎます ({len(text)} 文字 / 上限 {MAX_SETTING_CHARS})")
    if _CONTROL.search(text):
        raise ValueError(f"{label(name)} に制御文字が含まれています")
    text = text.strip()
    if name in _REQUIRED_TEXT and not text:
        raise ValueError(f"{label(name)} は空にできません")
    shape = _SETTING_SHAPES.get(name)
    if shape is not None and text and not shape[0].match(text):
        raise ValueError(f"{label(name)}: {shape[1]}")
    if name in _MODEL_NAMES and text:
        # These are handed to a loader that also accepts local paths, so a value shaped
        # like ../../../etc/passwd would be read off disk rather than rejected.
        if text.startswith(("/", "~")) or ".." in text.split("/"):
            raise ValueError(f"{label(name)}: パスではなくモデル名を指定してください")
    return text


def _write_settings(s: Settings) -> None:
    """Atomically persist the settings file. Failure is not fatal: the values stay in memory."""
    try:
        tmp = _settings_path().with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(s), indent=2, ensure_ascii=False), "utf-8")
        tmp.replace(_settings_path())
    except OSError as exc:
        log.warning("settings.json を書けませんでした: %s", exc)


def get_settings() -> Settings:
    global _settings
    with _lock:
        if _settings is None:
            s = Settings()
            path = _settings_path()
            if path.exists():
                try:
                    data = json.loads(path.read_text("utf-8"))
                    if not isinstance(data, dict):
                        raise ValueError("top level is not an object")
                except (json.JSONDecodeError, ValueError, OSError) as exc:
                    # A broken settings file must not keep the app from starting: keep a copy and use defaults.
                    backup = path.with_name(f"settings.broken-{int(time.time())}.json")
                    try:
                        path.replace(backup)
                    except OSError:
                        pass
                    log.warning("settings.json を読めないため既定値で起動します (%s)。元のファイル: %s", exc, backup)
                    data = {}
                defaults = Settings()
                for f in fields(Settings):
                    if f.name in data:
                        try:
                            setattr(s, f.name, _coerce(f.name, data[f.name], getattr(defaults, f.name)))
                        except (TypeError, ValueError):
                            log.warning("settings.json の %s の値 %r が不正なため既定値を使います", f.name, data[f.name])
                if _migrate(s, data):
                    _write_settings(s)
            _settings = s
        return _settings


def _migrate(s: Settings, data: dict[str, Any]) -> bool:
    """Carry a changed default into a settings file written before it. Runs once.

    Only a value still sitting on the *old* default is moved: a number the user chose
    themselves is left alone, and the revision marker keeps it from being moved twice.
    """
    if int(data.get("settings_revision", 0)) >= 1:
        return False
    changed = False
    if int(data.get("recycling_steps", 3)) == 3:
        s.recycling_steps = 4
        changed = True
        log.info("既定のリサイクルを 3 から 4 に更新しました (初回プリセットの変更)")
    s.settings_revision = 1
    return changed or True


def update_settings(patch: dict[str, Any]) -> Settings:
    global _settings
    current = get_settings()
    defaults = Settings()
    names = {f.name for f in fields(Settings)}
    unknown = set(patch) - names
    if unknown:
        raise ValueError(f"知らない設定項目です: {', '.join(sorted(unknown))}")
    with _lock:
        new = Settings(**asdict(current))
        for key, value in patch.items():
            setattr(new, key, _coerce(key, value, getattr(defaults, key)))
        if new.accelerator not in ("auto", "mps", "cpu"):
            raise ValueError("計算デバイスは 自動 / GPU (MPS) / CPU のいずれかです")
        if new.esm_device not in ("auto", "mps", "cpu"):
            raise ValueError("ESM-2 のデバイスは 自動 / GPU (MPS) / CPU のいずれかです")
        for key in ("diffusion_samples", "recycling_steps", "sampling_steps"):
            if getattr(new, key) < 1:
                raise ValueError(f"{label(key)} は 1 以上にしてください")
        if new.autopilot_max_variants_per_job < 0:
            raise ValueError("1ジョブあたりの変異体数は 0 以上にしてください")
        if new.autopilot_daily_budget < 0:
            raise ValueError("24時間あたりの自律ジョブ上限は 0 以上にしてください (0 で無制限)")
        if new.autopilot_max_depth < 0:
            raise ValueError("世代の上限は 0 以上にしてください (0 で無制限)")
        if new.autopilot_strategy not in AUTOPILOT_STRATEGIES:
            raise ValueError("探索のしかたは 山登り (climb) か 酔歩 (walk) です")
        if new.autopilot_climb_patience < 0:
            raise ValueError("山登りの我慢の手数は 0 以上にしてください (0 で降りない)")
        if not 1 <= new.autopilot_proposals_per_call <= 8:
            raise ValueError("1回に出させる提案数は 1〜8 です")
        if not 0.0 <= new.autopilot_disorder_plddt <= 100.0:
            raise ValueError("乱れた残基のしきい値は 0〜100 です")
        if new.autopilot_max_queued < 1:
            raise ValueError("待機ジョブの上限は 1 以上にしてください")
        if new.autopilot_min_disk_gb < 0:
            raise ValueError("空き容量の下限は 0 以上にしてください")
        if new.autopilot_selection not in ("greedy", "explore"):
            raise ValueError("候補の選び方は 良さそうな順 / 試していない位置優先 のどちらかです")
        if not 0.0 <= new.mps_memory_ratio <= 8.0:
            raise ValueError("GPU メモリの上限倍率は 0〜8 です (0 で上限なし)")
        if not 0.0 <= new.mpnn_veto <= 5.0:
            raise ValueError("逆折り畳みの却下ラインは 0〜5 です (0 で却下しない)")
        if not 0 <= new.llm_log_limit <= 1_000_000:
            raise ValueError("やり取りの保存件数は 0〜1000000 です (0 で保存しない)")
        if not 0 <= new.job_max_retries <= 10:
            raise ValueError("再試行の回数は 0〜10 です")
        if new.autopilot_improvement_metric not in IMPROVEMENT_METRICS:
            raise ValueError(
                f"autopilot_improvement_metric must be one of: {', '.join(IMPROVEMENT_METRICS)}")
        _write_settings(new)
        # Without this line a setting can change under a long unattended run — the autopilot
        # switching itself off, say — and the log gives no hint that anything was touched.
        changed = [f"{label(k)}: {getattr(current, k)!r} → {getattr(new, k)!r}"
                   for k in sorted(patch) if getattr(current, k) != getattr(new, k)]
        if changed:
            log.info("設定を変更しました — %s", " / ".join(changed))
        _settings = new
        return new


def resolve_boltz_bin() -> str:
    s = get_settings()
    if s.boltz_bin:
        return s.boltz_bin
    import sys

    candidate = Path(sys.executable).parent / "boltz"
    if candidate.exists():
        return str(candidate)
    found = shutil.which("boltz")
    if found:
        return found
    return ""
