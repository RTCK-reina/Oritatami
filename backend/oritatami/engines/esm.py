"""ESM-2 protein language model: zero-shot mutation scoring and sequence refinement.

Scores are masked-marginal log-likelihood ratios (Meier et al., 2021):
``LLR(i, a) = log p(x_i = a | x_-i) - log p(x_i = wt | x_-i)``. Positive values mean the
language model finds the substitution at least as "natural" as the wild type; it is a
plausibility signal, not a measured stability or activity.
"""

from __future__ import annotations

import math
import random
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

from ..config import get_settings
from ..seq import AMINO_ACIDS, Mutation, apply_mutations

MAX_WINDOW = 1022  # ESM-2 positional limit (1024 incl. BOS/EOS)


class EsmUnavailable(RuntimeError):
    kind = "missing_dependency"


class EsmBusy(RuntimeError):
    """The model is in use by a long job (e.g. a mutation scan) and the caller did not want to wait."""


class _Model:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self.name = ""
        self.last_used = 0.0
        self.aa_token_ids: list[int] = []

    def load(self) -> None:
        s = get_settings()
        with self.lock:
            if self.model is not None and self.name == s.esm_model:
                self.last_used = time.time()
                return
            try:
                import torch
                from transformers import AutoTokenizer, EsmForMaskedLM
            except ImportError as exc:
                raise EsmUnavailable(f"ESM に必要なライブラリを読み込めません: {exc}") from exc
            device = s.esm_device
            if device == "auto":
                device = "mps" if torch.backends.mps.is_available() else "cpu"
            tok = AutoTokenizer.from_pretrained(s.esm_model)
            model = EsmForMaskedLM.from_pretrained(s.esm_model)
            model.eval().to(device)
            self.model, self.tokenizer, self.device, self.name = model, tok, device, s.esm_model
            self.aa_token_ids = [tok.convert_tokens_to_ids(a) for a in AMINO_ACIDS]
            self.last_used = time.time()

    def unload(self) -> None:
        with self.lock:
            if self.model is None:
                return
            self.model = None
            self.tokenizer = None
            import gc

            gc.collect()
            try:
                import torch

                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except ImportError:
                pass

    def status(self) -> dict[str, Any]:
        return {"loaded": self.model is not None, "model": self.name or get_settings().esm_model,
                "device": self.device if self.model is not None else None}


_M = _Model()


def status() -> dict[str, Any]:
    return _M.status()


def unload() -> None:
    _M.unload()


def unload_if_idle(idle_sec: float = 600) -> None:
    if _M.model is not None and time.time() - _M.last_used > idle_sec and _M.lock.acquire(blocking=False):
        try:
            _M.unload()
        finally:
            _M.lock.release()


def release_cache() -> float:
    """Hand Metal back the blocks torch is holding but not using. Returns the GB freed.

    Torch's MPS allocator keeps freed blocks in its own pool rather than returning them, so a
    scan that peaked at 3 GB goes on occupying that much as far as the rest of the machine is
    concerned — including the Boltz subprocess that starts next, which gets its own Metal
    working-set budget out of the same unified memory. Unloading the model is the heavy
    version of this and is not always possible (a scan may be mid-flight); emptying the cache
    is always safe and needs no lock.

    Deliberately does nothing when torch has not been imported: importing it costs seconds
    and hundreds of megabytes, which is the opposite of the point.
    """
    torch = sys.modules.get("torch")
    if torch is None:
        return 0.0
    try:
        if not torch.backends.mps.is_available():
            return 0.0
        # driver_allocated_memory is what Metal has handed this process, cache included;
        # current_allocated_memory counts only live tensors and does not move when the cache
        # is emptied, so it would always report 0 freed.
        before = torch.mps.driver_allocated_memory()
        torch.mps.empty_cache()
        after = torch.mps.driver_allocated_memory()
    except (AttributeError, RuntimeError):
        return 0.0
    return max(0.0, (before - after) / 1024**3)


def unload_if_unused() -> bool:
    """Free the model now unless a scan/refine is using it. Returns True if memory was released."""
    if _M.model is None or not _M.lock.acquire(blocking=False):
        return False
    try:
        _M.unload()
        return True
    finally:
        _M.lock.release()


def is_cached() -> bool:
    """Whether the configured model's weights are already in the Hugging Face cache (no download needed)."""
    name = get_settings().esm_model
    import os
    from pathlib import Path

    hub = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub")
    return (hub / f"models--{name.replace('/', '--')}").is_dir()


def _window(length: int, pos: int) -> tuple[int, int]:
    if length <= MAX_WINDOW:
        return 0, length
    start = max(0, min(pos - MAX_WINDOW // 2, length - MAX_WINDOW))
    return start, start + MAX_WINDOW


def _masked_log_probs(seq: str, positions: list[int], batch_size: int = 8,
                      progress: Callable[[float], None] | None = None,
                      wait: float | None = None) -> dict[int, list[float]]:
    """log p(a | context with position masked) for each 0-based position -> 20 values (AMINO_ACIDS order).

    `wait` bounds how long to wait for a model that another job is using (None = wait indefinitely).
    """
    import torch

    if not _M.lock.acquire(timeout=-1 if wait is None else wait):
        raise EsmBusy("ESM-2 は別の計算 (変異スキャンなど) で使用中です")
    try:
        _M.load()
    finally:
        _M.lock.release()
    with _M.lock:
        model, tok, device = _M.model, _M.tokenizer, _M.device
        assert model is not None and tok is not None
        out: dict[int, list[float]] = {}
        # group positions by window so each batch shares one context length
        groups: dict[tuple[int, int], list[int]] = {}
        for p in positions:
            groups.setdefault(_window(len(seq), p), []).append(p)
        done = 0
        with torch.no_grad():
            for (ws, we), plist in groups.items():
                sub = seq[ws:we]
                ids = tok(sub, return_tensors="pt")["input_ids"][0]
                for b in range(0, len(plist), batch_size):
                    chunk = plist[b:b + batch_size]
                    batch = ids.repeat(len(chunk), 1)
                    for row, p in enumerate(chunk):
                        batch[row, p - ws + 1] = tok.mask_token_id  # +1 for BOS
                    logits = model(input_ids=batch.to(device)).logits.float()
                    for row, p in enumerate(chunk):
                        lp = torch.log_softmax(logits[row, p - ws + 1], dim=-1)
                        out[p] = lp[_M.aa_token_ids].cpu().tolist()
                    done += len(chunk)
                    if progress:
                        progress(done / len(positions))
        _M.last_used = time.time()
    return out


def mutation_scan(seq: str, progress: Callable[[float], None] | None = None) -> dict[str, Any]:
    """Full single-substitution landscape: L x 20 LLR matrix plus summary statistics."""
    seq = seq.upper()
    started = time.time()
    lps = _masked_log_probs(seq, list(range(len(seq))), progress=progress)
    matrix = []
    wt_logp = []
    for i, wt in enumerate(seq):
        row = lps[i]
        w = row[AMINO_ACIDS.index(wt)]
        wt_logp.append(w)
        matrix.append([round(v - w, 3) for v in row])
    pppl = math.exp(-sum(wt_logp) / len(wt_logp))
    ranked = []
    for i, row in enumerate(matrix):
        for j, v in enumerate(row):
            if AMINO_ACIDS[j] != seq[i]:
                ranked.append((v, f"{seq[i]}{i + 1}{AMINO_ACIDS[j]}"))
    ranked.sort(reverse=True)
    # per-position tolerance: mean LLR of all substitutions (higher = more tolerant)
    tolerance = [round(sum(v for j, v in enumerate(row) if AMINO_ACIDS[j] != seq[i]) / 19, 3)
                 for i, row in enumerate(matrix)]
    return {
        "model": get_settings().esm_model,
        "alphabet": AMINO_ACIDS,
        "sequence": seq,
        "matrix": matrix,
        "pseudo_perplexity": round(pppl, 3),
        "top_substitutions": [{"mutation": m, "llr": round(v, 3)} for v, m in ranked[:40]],
        "position_tolerance": tolerance,
        "elapsed_sec": round(time.time() - started, 2),
    }


def score_mutations(seq: str, mutations: list[Mutation], wait: float | None = None) -> dict[str, Any]:
    """Additive masked-marginal score of a mutation set against `seq` (the wild type).

    ``llr`` is the raw log-likelihood ratio against the wild-type residue: negative means
    ESM-2 finds the mutant less likely than what is already there. Its scale depends on
    how constrained the protein is — in a highly conserved one like ubiquitin the median
    single mutant sits near -7 — so a raw threshold does not transfer between targets.
    ``rank`` normalises that: where the mutant residue places among the 20 amino acids at
    its own position (1 = the residue ESM-2 considers most likely there), together with
    the residue it would have picked. Both come from log-probabilities already computed
    here, so they cost nothing extra.
    """
    seq = seq.upper()
    apply_mutations(seq, mutations)  # validates positions and WT residues
    positions = sorted({m.position - 1 for m in mutations})
    lps = _masked_log_probs(seq, positions, wait=wait)
    parts = []
    total = 0.0
    ranks = []
    for m in mutations:
        row = lps[m.position - 1]
        llr = row[AMINO_ACIDS.index(m.mt)] - row[AMINO_ACIDS.index(m.wt)]
        total += llr
        order = sorted(range(len(AMINO_ACIDS)), key=lambda j: -row[j])
        rank = order.index(AMINO_ACIDS.index(m.mt)) + 1
        ranks.append(rank)
        parts.append({
            "mutation": m.code,
            "llr": round(llr, 3),
            "rank": rank,
            "of": len(AMINO_ACIDS),
            "best": AMINO_ACIDS[order[0]],
        })
    return {
        "total_llr": round(total, 3),
        "mean_llr": round(total / len(mutations), 3) if mutations else None,
        "worst_rank": max(ranks) if ranks else None,
        "per_mutation": parts,
    }


def pseudo_perplexity(seq: str, progress: Callable[[float], None] | None = None, wait: float | None = None) -> float:
    seq = seq.upper()
    lps = _masked_log_probs(seq, list(range(len(seq))), progress=progress, wait=wait)
    total = sum(lps[i][AMINO_ACIDS.index(a)] for i, a in enumerate(seq))
    return round(math.exp(-total / len(seq)), 3)


def refine(seq: str, *, rounds: int = 6, fraction: float = 0.1, temperature: float = 1.0,
           fixed: set[int] | None = None, seed: int | None = None,
           progress: Callable[[float, str], None] | None = None,
           cancelled: Callable[[], bool] | None = None) -> dict[str, Any]:
    """Iterative masked resampling (Gibbs-style) toward sequences ESM-2 finds more natural.

    Each round masks the least-likely unfixed positions (plus a few random ones), samples
    replacements at `temperature`, and keeps the round only if pseudo-perplexity does not
    get worse. Returns the trajectory so the UI can show what changed.
    """
    rng = random.Random(seed)
    seq = seq.upper()
    fixed = fixed or set()
    history = []
    lps = _masked_log_probs(seq, list(range(len(seq))))
    current_pppl = math.exp(-sum(lps[i][AMINO_ACIDS.index(a)] for i, a in enumerate(seq)) / len(seq))
    history.append({"round": 0, "sequence": seq, "pseudo_perplexity": round(current_pppl, 3), "changed": []})
    for r in range(1, rounds + 1):
        if cancelled and cancelled():
            break
        candidates = [i for i in range(len(seq)) if i not in fixed]
        if not candidates:
            break
        k = max(1, int(len(candidates) * fraction))
        worst = sorted(candidates, key=lambda i: lps[i][AMINO_ACIDS.index(seq[i])])[: max(1, k * 2 // 3)]
        rest = [i for i in candidates if i not in worst]
        picks = sorted(set(worst + rng.sample(rest, min(len(rest), k - len(worst)))))
        proposal = list(seq)
        for i in picks:
            logits = [v / max(temperature, 1e-3) for v in lps[i]]
            mx = max(logits)
            weights = [math.exp(v - mx) for v in logits]
            proposal[i] = rng.choices(AMINO_ACIDS, weights=weights)[0]
        new_seq = "".join(proposal)
        if cancelled and cancelled():
            break
        new_lps = _masked_log_probs(new_seq, list(range(len(new_seq))))
        new_pppl = math.exp(-sum(new_lps[i][AMINO_ACIDS.index(a)] for i, a in enumerate(new_seq)) / len(new_seq))
        changed = [f"{seq[i]}{i + 1}{new_seq[i]}" for i in picks if seq[i] != new_seq[i]]
        accepted = new_pppl <= current_pppl
        if accepted:
            seq, lps, current_pppl = new_seq, new_lps, new_pppl
        history.append({"round": r, "sequence": seq, "pseudo_perplexity": round(current_pppl, 3),
                        "changed": changed if accepted else [], "accepted": accepted,
                        "proposal_pseudo_perplexity": round(new_pppl, 3)})
        if progress:
            progress(r / rounds, f"ラウンド {r}/{rounds}  PPPL {current_pppl:.2f}")
    return {"sequence": seq, "pseudo_perplexity": round(current_pppl, 3), "history": history,
            "start_pseudo_perplexity": history[0]["pseudo_perplexity"]}
