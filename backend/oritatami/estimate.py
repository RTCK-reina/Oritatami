"""Runtime and memory estimates for a prediction, calibrated on this machine's own finished jobs.

Boltz's cost grows with the token count (residues + ligand heavy atoms), diffusion samples,
sampling steps and recycling. Each finished job records how long every phase took; the
structure phase is fitted as ``load + k * work(spec)`` over those records, while MSA search,
start-up and affinity use medians because they do not scale with the input the same way.
Estimates therefore improve as the user runs predictions. Everything here is a rough guide
and the UI labels it as such.
"""

from __future__ import annotations

import os
import statistics
import subprocess
from functools import lru_cache
from typing import Any

from .config import get_settings

# The same headroom the Metal limit leaves for macOS; one number, one definition.
from .gpu import HEADROOM_GB

# Defaults from jobs measured on an M5 Pro (24 GB), per phase:
#   76-residue monomer:            start-up+preprocess 4 s, MSA 2 s, structure 30 s
#   303-token complex with ligand: start-up+preprocess 4 s, MSA 5 s, structure 68 s, affinity 92 s
# The structure phase includes loading the weights (~25 s), so it is modelled as load + k * work.
DEFAULT_LOAD = 25.0
DEFAULT_K = 4.5  # seconds per work unit on top of loading
DEFAULT_MSA = 15.0
DEFAULT_OVERHEAD = 5.0
DEFAULT_AFFINITY = 90.0


def work_units(tokens: int, params: dict[str, Any]) -> float:
    """Relative cost of the structure phase beyond loading: pair representations scale ~N²."""
    size = (max(10, tokens) / 100.0) ** 2
    samples = 0.6 + 0.4 * max(1, int(params.get("diffusion_samples", 1)))
    steps = 0.4 + 0.6 * (max(10, int(params.get("sampling_steps", 200))) / 200.0)
    recycle = 0.5 + 0.5 * ((max(1, int(params.get("recycling_steps", 4))) + 1) / 4.0)
    return size * samples * steps * recycle


def _fit_structure(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Least-squares fit of structure_seconds = load + k * work. Falls back to defaults when underdetermined."""
    if not points:
        return DEFAULT_LOAD, DEFAULT_K
    ws = [w for w, _ in points]
    if len(points) >= 2 and max(ws) >= 2 * min(ws):
        mw = statistics.fmean(ws)
        mt = statistics.fmean(t for _, t in points)
        var = sum((w - mw) ** 2 for w in ws)
        k = sum((w - mw) * (t - mt) for w, t in points) / var if var else DEFAULT_K
        load = mt - k * mw
        if k > 0 and 5 <= load <= 90:
            return load, k
    # One size only: keep the default load and scale k to match the observations.
    ks = [(t - DEFAULT_LOAD) / w for w, t in points if t > DEFAULT_LOAD]
    return DEFAULT_LOAD, statistics.median(ks) if ks else DEFAULT_K


def _median(values: list[float], default: float) -> float:
    values = [v for v in values if v > 0]
    return statistics.median(values) if values else default


@lru_cache(maxsize=1)
def memory_gb() -> float | None:
    try:
        out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=2, check=True)
        return int(out.stdout.strip()) / 1024**3
    except (OSError, ValueError, subprocess.SubprocessError):
        try:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
        except (ValueError, OSError, AttributeError):
            return None


# Peak unified memory (physical footprint, not resident size). Quadratic in tokens because the
# pair representations dominate. The constants are a least-squares fit over three runs measured
# on this machine on 2026-09-14, single-sequence MSA, diffusion_samples 1, recycling 3,
# sampling 200:
#
#       76 tokens    5.52 GB     55 s
#      609 tokens   14.81 GB    296 s
#     1198 tokens   31.0  GB    did not finish in 80 minutes (peak plateaued, no OOM)
#
# fit: peak = 6.70 + 1.45 * (tokens/290)^2   (+23% / -12% / +1% against those three)
#
# The previous constants (3.5 + 7.0x) came from one unverified observation and were five times
# too steep: they put 609 tokens at 34 GB, which would have shown a red warning for a job that
# actually runs in five minutes. Erring high is not free — a warning nobody believes is the
# same as no warning.
#
# Adding a real MSA cost 1.7 GB at 609 tokens (14.81 -> 16.48 with 1,135 sequences), which is
# inside the spread of this fit, so MSA depth is not modelled separately.
DEFAULT_MEM_BASE = 6.7
DEFAULT_MEM_K = 1.45
# Affinity is still anchored to the single old observation (~13 GB at 260 tokens with a ligand),
# which has never been re-measured with the footprint reader. It is deliberately pessimistic;
# the first footprint-stamped affinity run will replace it through the history fit.
DEFAULT_MEM_K_AFFINITY = 7.8
# Plausible range for a fitted slope, in GB per unit of (tokens/290)^2. Measured 1.45 here
# without affinity; the affinity anchor implies about 7.8. Past 12 a 290-token job would cost
# more than 18 GB over its baseline, which is four times anything seen on this machine, so a
# fit that lands there is bad data rather than a discovery. A band relative to the prior was
# tried first and broke the moment the prior was corrected downwards: an honest history with
# affinity sat six times above a 1.45 prior and was thrown away.
MIN_MEM_K = 0.3
MAX_MEM_K = 12.0


def _peak_memory(tokens: int, affinity: bool,
                 history: list[dict[str, Any]]) -> tuple[float, str, int]:
    """Peak GB for this size, fitted from measured runs when there are any."""
    default_k = DEFAULT_MEM_K_AFFINITY if affinity else DEFAULT_MEM_K
    points: list[tuple[float, float]] = []
    for job in history:
        res = job.get("result") or {}
        measured = res.get("peak_memory_gb")
        t = res.get("token_estimate") or (res.get("normalized_spec") or {}).get("token_estimate")
        # Only footprint-based readings are comparable. The first version of the supervisor
        # summed resident size, which on Metal misses every GPU allocation: the same process
        # read 6.5 MB resident and 30 GB of physical footprint at the same instant. Those
        # rows are not low, they are meaningless, so they are dropped rather than corrected.
        if res.get("peak_memory_method") != "footprint":
            continue
        if not measured or not t:
            continue
        points.append(((int(t) / 290.0) ** 2, float(measured)))
    if not points:
        return DEFAULT_MEM_BASE + default_k * (tokens / 290.0) ** 2, "default", 0

    xs = [x for x, _ in points]
    my = statistics.fmean(y for _, y in points)

    # The slope is only identifiable if the measured sizes actually differ. Fitting it from a
    # narrow cluster and then extrapolating is how a handful of small jobs turned into a
    # 48 GB/unit slope that flagged every ordinary prediction as dangerous: three runs at 51 and
    # 76 tokens (2.6-4.9 GB measured) predicted 50 GB at 290 tokens, four times the ~13 GB that
    # size actually costs. So below a 2x spread in tokens (4x in size, which is quadratic) keep
    # the prior slope and let the measurements move only the offset.
    spread = len(points) >= 3 and max(xs) >= 4 * min(xs)
    if not spread:
        base = statistics.median(y - default_k * x for x, y in points)
        return max(0.5, base) + default_k * (tokens / 290.0) ** 2, "history", len(points)

    # peak = base + k * size, least squares, with the slope held inside the range any real
    # configuration here has produced (see MIN_MEM_K / MAX_MEM_K).
    mx = statistics.fmean(xs)
    var = sum((x - mx) ** 2 for x in xs)
    k = (sum((x - mx) * (y - my) for x, y in points) / var) if var else default_k
    base = my - k * mx
    if not (MIN_MEM_K <= k <= MAX_MEM_K) or not 0.5 <= base <= 12.0:
        base = max(0.5, statistics.median(y - default_k * x for x, y in points))
        k = default_k
    return base + k * (tokens / 290.0) ** 2, "history", len(points)


def estimate(spec: dict[str, Any], needs_msa_search: bool, history: list[dict[str, Any]]) -> dict[str, Any]:
    tokens = int(spec["token_estimate"])
    params = spec["params"]
    affinity = bool(spec.get("affinity_binder"))
    points: list[tuple[float, float]] = []
    msa_times, overheads, aff_times = [], [], []
    same_device = params.get("accelerator", "auto")
    for job in history:
        res = job.get("result") or {}
        nspec = res.get("normalized_spec") or {}
        if not nspec or res.get("elapsed_sec") is None:
            continue
        if same_device != "auto" and res.get("accelerator") and res["accelerator"] != same_device:
            continue
        t_tokens = res.get("token_estimate") or nspec.get("token_estimate")
        if not t_tokens:
            continue
        w = work_units(int(t_tokens), nspec.get("params") or {})
        timings = res.get("timings") or {}
        if timings.get("structure"):
            points.append((w, float(timings["structure"])))
            if timings.get("msa"):
                msa_times.append(timings["msa"])
            if timings.get("affinity"):
                aff_times.append(timings["affinity"])
            overheads.append(sum(v for key, v in timings.items() if key not in ("structure", "msa", "affinity")))
        else:
            # Older results only have the total: attribute the remainder to the structure phase.
            rest = res["elapsed_sec"] - DEFAULT_OVERHEAD - (DEFAULT_MSA if (res.get("msa") or {}).get("server") else 0)
            if nspec.get("affinity_binder"):
                rest -= DEFAULT_AFFINITY
            if rest > 0:
                points.append((w, rest))
    load, k = _fit_structure(points)
    structure = load + k * work_units(tokens, params)
    msa = _median(msa_times, DEFAULT_MSA) if needs_msa_search else 0.0
    overhead = _median(overheads, DEFAULT_OVERHEAD)
    aff = _median(aff_times, DEFAULT_AFFINITY) if affinity else 0.0
    seconds = overhead + msa + structure + aff
    spread = 1.35 if len(points) >= 3 else 1.8
    peak, mem_basis, mem_samples = _peak_memory(tokens, affinity, history)
    total = memory_gb()
    # The structure fit is built from runs that fit in memory, so it describes a job at full
    # speed. If this one is not going to fit, the paging it will do is the larger half of the
    # answer and belongs in the same number — not in a warning beside it.
    penalty = swap_penalty(peak, params, total)
    seconds += penalty["seconds"]
    level = "ok"
    if total:
        # Past physical memory the machine does not fail, it slows by about 300x (measured:
        # 97 GB/s of page traffic while resident, 296 MB/s once swapping). So the line that
        # matters is "will this fit", not "will this crash" — and it has to leave room for
        # everything else that is running.
        if peak > total * 0.75:
            level = "danger"
        elif peak > total * 0.55:
            level = "caution"
    # What swapping costs the drive, attached only where we are actually predicting it. The
    # MPS allocator no longer refuses the allocation (see mps_memory_ratio), so this is the
    # only place the price gets named before the job is submitted: a soldered SSD taking
    # ~190 MB/s of writes for however long the run lasts.
    # Past physical memory the run is not "tight", it is swapping from end to end: the
    # 190 MB/s of page-out measured at 40 GB on a 24 GB machine is what it pays for every
    # second it is alive. Below the line a job merely brushes swap; above it the SSD is the
    # price of the whole run, and that is a different warning.
    #
    # The warning says so and stops there. Putting live terabyte-per-day and percent-of-life
    # figures on the submit button was tried and was worse: numbers that move invite reading
    # today's against yesterday's, when the only decision here is go or do not go. The figures
    # still exist, in settings, where someone who wants them can go and look — and keeping
    # them out of here also keeps smartctl off a path that runs on every keystroke.
    beyond_physical = bool(total and peak > total)
    return {
        "seconds": round(seconds),
        "low": round(seconds / spread),
        "high": round(seconds * spread),
        "breakdown": {"startup": round(overhead), "msa": round(msa), "structure": round(structure),
                      "affinity": round(aff), "paging": round(penalty["seconds"])},
        "basis": "history" if len(points) >= 2 else "default",
        "samples": len(points),
        "tokens": tokens,
        "needs_msa_search": needs_msa_search,
        "memory": {"peak_gb": round(peak, 1), "total_gb": round(total, 1) if total else None,
                   "level": level, "basis": mem_basis, "samples": mem_samples,
                   "beyond_physical": beyond_physical, "applecare": get_settings().applecare,
                   "capacity_gb": penalty["capacity_gb"], "overage_gb": penalty["overage_gb"],
                   "paging_sec": round(penalty["seconds"]), "passes": penalty["passes"],
                   "traffic_tb": penalty["traffic_tb"]},
    }


# ------------------------------------------------------------------ past physical memory
# Boltz does not fail when the working set stops fitting — it pages, and the run keeps going at
# the speed of the SSD instead of the speed of memory. Measured on this machine while a
# 1,278-token prediction was in that state: 563,094 swap-ins and 497,311 swap-outs per 30 s at
# 16 KB a page, so about 565 MB/s of traffic, against 97 GB/s of page traffic while resident.
# The ratio is 170x, which is why a 40-minute job was still running after three and a half
# hours.
#
# The cost is therefore (how much does not fit) x (how many times the run walks its working
# set) / (how fast swap moves). The first two are the parts worth arguing about:
#
#   * what does not fit — the fitted peak minus what can stay resident, which is installed
#     memory less the few GB macOS needs. Not the Metal working-set limit: that governs
#     whether an allocation is refused, and what happened here was paging, not a refusal.
#   * how many walks — anchored on that same run. 212 minutes at 565 MB/s is 7.2 TB of
#     traffic; against a 25.6 GB overage that is 280 passes, with sampling_steps at 200 and
#     recycling at 4. So one pass per diffusion step plus a fixed cost for the trunk.
#
# ONE observation stands behind PASSES_BASE. It is enough to turn "40 分" into "4 時間 20 分",
# which is the difference that matters, and not enough to trust the second digit.
SWAP_MB_PER_SEC = 565.0
RESIDENT_MB_PER_SEC = 97.0 * 1024
PASSES_BASE = 80
# Below this share of user time the run is not computing, it is waiting on the kernel to move
# pages. Measured: healthy runs sit at 0.86 (76 tokens), 0.73 (304) and 0.64 (608) — the share
# falls with size because the GPU does more of the work — while the thrashing 1,278-token run
# sat at 0.016. There is a factor of 40 between the two regimes, so the line can go anywhere
# sensible between them.
THRASH_USER_SHARE = 0.25


def resident_capacity_gb(total_gb: float | None = None) -> float | None:
    """How much of installed memory a prediction can actually keep resident."""
    total = total_gb if total_gb is not None else memory_gb()
    return max(1.0, total - HEADROOM_GB) if total else None


def swap_penalty(peak_gb: float, params: dict[str, Any],
                 total_gb: float | None = None) -> dict[str, Any]:
    """Seconds this run loses to paging, given the memory it is expected to need."""
    capacity = resident_capacity_gb(total_gb)
    if capacity is None or peak_gb <= capacity:
        return {"overage_gb": 0.0, "seconds": 0.0, "passes": 0, "traffic_tb": 0.0,
                "capacity_gb": capacity}
    overage = peak_gb - capacity
    passes = int(max(1, int(params.get("diffusion_samples", 1)))
                 * max(10, int(params.get("sampling_steps", 200))) + PASSES_BASE)
    traffic_mb = passes * overage * 1024.0
    # The difference between the two speeds, as asked: the pages would have been read anyway,
    # just 170x faster. Keeping the resident term in makes the formula say what it means, even
    # though it changes the answer by well under a percent.
    seconds = traffic_mb / SWAP_MB_PER_SEC - traffic_mb / RESIDENT_MB_PER_SEC
    return {"overage_gb": round(overage, 1), "seconds": round(seconds), "passes": passes,
            "traffic_tb": round(traffic_mb / 1024**2, 2), "capacity_gb": round(capacity, 1)}


# ------------------------------------------------------------------ live: what is left
# A prediction reports no progress from inside the diffusion phase, and the obvious substitute
# does not work either: CPU time is not a progress meter here, because the GPU does the
# arithmetic. Measured across three calibration runs on this machine — 76, 304 and 608 tokens —
# wall time went 36 s -> 116 s -> 389 s while CPU time went 20 s -> 27 s -> 40 s. The CPU is
# nearly a constant, so "CPU consumed / CPU expected" would read 50 % a minute in and then
# crawl.
#
# What CPU time does say, unambiguously, is which regime the run is in (see THRASH_USER_SHARE).
# So the remaining time comes from the wall-clock model, and the live measurement decides
# whether the paging term belongs in it — using the footprint the run has actually reached
# rather than the one that was predicted for it.
def regime(live: dict[str, Any] | None) -> str:
    """"resident" | "paging" | "unknown", from the live sample."""
    live = live or {}
    share = live.get("user_share")
    footprint = live.get("footprint_gb")
    total = live.get("total_gb")
    if isinstance(share, (int, float)) and (live.get("cpu_sec") or 0) >= 5:
        return "paging" if share < THRASH_USER_SHARE else "resident"
    capacity = resident_capacity_gb(total)
    if isinstance(footprint, (int, float)) and capacity:
        return "paging" if footprint > capacity else "resident"
    return "unknown"


def remaining(spec: dict[str, Any], elapsed: float, baseline: float,
              live: dict[str, Any] | None, history: list[dict[str, Any]]) -> dict[str, Any]:
    """How much longer a running prediction needs, and on what grounds.

    ``seconds`` is None when there is no honest answer — a job that has already run past
    everything the model can account for is a job nobody can time, and saying so is more use
    than a number that keeps resetting.
    """
    live = live or {}
    state = regime(live)
    peak = live.get("peak_gb") or live.get("footprint_gb")
    penalty = (swap_penalty(float(peak), spec["params"], live.get("total_gb"))
               if peak else {"overage_gb": 0.0, "seconds": 0.0, "passes": 0, "traffic_tb": 0.0,
                             "capacity_gb": resident_capacity_gb(live.get("total_gb"))})
    total = baseline + penalty["seconds"]
    out: dict[str, Any] = {
        "seconds": None,
        "basis": "unknown",
        "regime": state,
        "progress": None,
        "user_share": live.get("user_share"),
        "efficiency": live.get("efficiency"),
        "cpu_seconds": live.get("cpu_sec"),
        "overage_gb": penalty["overage_gb"],
        "swap_seconds": penalty["seconds"],
        "total_estimate": round(total),
        "overrun": elapsed > total,
        "note": "",
    }
    if elapsed <= total:
        out["seconds"] = round(total - elapsed)
        out["basis"] = "swap" if penalty["seconds"] else "baseline"
        out["progress"] = round(min(0.999, elapsed / total), 3) if total > 0 else None
        if penalty["seconds"]:
            out["note"] = (f"物理メモリを {penalty['overage_gb']} GB 超過する見込みで、"
                           f"その分のページングに約 {penalty['seconds'] / 60:.0f} 分かかります")
        return out
    out["note"] = (f"スワップ込みの推定 {total / 60:.0f} 分に対して実測 {elapsed / 60:.0f} 分。"
                   "この規模の完走実績がないため、残りを推定できません")
    if state == "paging":
        out["note"] += "（user 時間の割合が %.1f%% しかなく、計算ではなくページングに使われています）" % (
            100 * (live.get("user_share") or 0))
    return out
