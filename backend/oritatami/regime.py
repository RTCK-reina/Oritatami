"""The conditions a prediction was computed under, and whether two are comparable.

Boltz writes a pLDDT, not a property of the molecule: change how the prediction was run
and the same sequence scores differently. Measured over this machine's 893 finished
predictions, grouped by how the MSA was obtained:

    reused MSA   n=724   core pLDDT 96.21   sd  0.55
    server MSA   n= 88              95.29   sd  4.64
    single seq   n= 75              89.26   sd 12.65

Seven sequences happen to have been run under more than one regime; the same molecule
moved by up to 1.9 points between them. The autopilot calls anything above 0.5 an
improvement, so a regime change alone can manufacture — or hide — three improvements'
worth of signal. The leaderboard was ranking all 893 in one column.

Nothing here changes how a prediction runs. It reads what the run already recorded
(``result.normalized_spec.params`` and ``result.msa``) and makes the mixing visible.
"""

from __future__ import annotations

from typing import Any, NamedTuple

# Fields that change the number Boltz reports for an unchanged sequence.
_PARAM_KEYS = ("diffusion_samples", "recycling_steps", "sampling_steps", "use_potentials")

MSA_LABELS = {
    "reused": "MSA 使い回し",
    "server": "MSA サーバー取得",
    "single": "単一配列 (MSA なし)",
    "none": "MSA 不明",
}


class Regime(NamedTuple):
    """How a prediction was computed. Two predictions are comparable when these match."""

    diffusion_samples: int | None
    recycling_steps: int | None
    sampling_steps: int | None
    use_potentials: bool | None
    msa: str

    def label(self) -> str:
        parts = [MSA_LABELS.get(self.msa, self.msa)]
        if self.diffusion_samples not in (None, 1):
            parts.append(f"サンプル {self.diffusion_samples}")
        if self.recycling_steps not in (None, 3):
            parts.append(f"リサイクル {self.recycling_steps}")
        if self.sampling_steps not in (None, 200):
            parts.append(f"ステップ {self.sampling_steps}")
        if self.use_potentials:
            parts.append("ポテンシャル")
        return " · ".join(parts)

    def short(self) -> str:
        """One token for a table cell."""
        return {"reused": "再利用", "server": "サーバー", "single": "単一", "none": "?"}.get(
            self.msa, self.msa)


UNKNOWN = Regime(None, None, None, None, "none")


def msa_mode(result: dict[str, Any] | None) -> str:
    m = (result or {}).get("msa") or {}
    if m.get("reused_from"):
        return "reused"
    if m.get("server"):
        return "server"
    if m.get("single_sequence_chains"):
        return "single"
    return "none"


def of(result: dict[str, Any] | None) -> Regime:
    """Read the regime a finished prediction ran under."""
    if not result:
        return UNKNOWN
    params = ((result.get("normalized_spec") or {}).get("params")) or {}
    return Regime(
        diffusion_samples=params.get("diffusion_samples"),
        recycling_steps=params.get("recycling_steps"),
        sampling_steps=params.get("sampling_steps"),
        use_potentials=None if params.get("use_potentials") is None else bool(params["use_potentials"]),
        msa=msa_mode(result),
    )


def comparable(a: Regime, b: Regime) -> bool:
    """Can these two scores be put side by side?

    Equality, including two equally unrecorded regimes — the failure this guards against
    is MIXING, and predictions that both recorded nothing are no worse off than before
    the guard existed. A known regime against an unknown one is not comparable: that is
    exactly the case where a settings change hides in the gap.
    """
    return a == b


def differences(a: Regime, b: Regime) -> list[str]:
    """What differs, in words, for a message a person reads."""
    out = []
    if a.msa != b.msa:
        out.append(f"MSA ({MSA_LABELS.get(b.msa, b.msa)} → {MSA_LABELS.get(a.msa, a.msa)})")
    labels = {"diffusion_samples": "サンプル数", "recycling_steps": "リサイクル",
              "sampling_steps": "拡散ステップ", "use_potentials": "ポテンシャル"}
    for key in _PARAM_KEYS:
        av, bv = getattr(a, key), getattr(b, key)
        if av != bv:
            out.append(f"{labels[key]} ({bv} → {av})")
    return out


def summarize(results: list[dict[str, Any] | None]) -> dict[str, Any]:
    """Which regimes a set of predictions spans — for a leaderboard header."""
    counts: dict[Regime, int] = {}
    for r in results:
        reg = of(r)
        counts[reg] = counts.get(reg, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: -kv[1])
    return {
        "mixed": len(counts) > 1,
        "regimes": [{"label": reg.label(), "short": reg.short(), "msa": reg.msa, "count": n}
                    for reg, n in ordered],
    }
