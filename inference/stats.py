"""
inference/stats.py
------------------
Paired statistics for comparing two models on the same patches.

Why paired, and why an effect floor
===================================
Every model in the grid sees the *same* test patches, so the comparison is
paired: the right quantity is the per-patch difference, not the difference of
means. Pairing removes patch-to-patch variance, which on HSI is enormous (a
shadowed patch is hard for everyone) and would otherwise swamp the model effect.

With 3,084 test patches, essentially ANY non-zero difference reaches p < 0.05.
So significance alone certifies nothing. Every comparison therefore reports a
minimum meaningful effect alongside the p-value, and a result that clears
significance but not the effect floor is labelled `significant_but_negligible`
rather than being claimed as a win. Thresholds come from
`inference/preregistration.yaml`, never from this file.

Nothing here is HSI-specific; it operates on paired arrays of per-patch metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np


@dataclass
class PairedResult:
    metric: str
    model_a: str
    model_b: str
    n: int
    mean_a: float
    mean_b: float
    delta: float                 # mean(a - b)
    ci_low: float
    ci_high: float
    p_value: float
    p_holm: float | None
    cliffs_delta: float
    significant: bool
    meaningful: bool
    verdict: str

    def as_row(self) -> dict:
        return asdict(self)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def paired_bootstrap_ci(a: np.ndarray, b: np.ndarray, resamples: int = 10000,
                        alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """
    Percentile bootstrap CI for mean(a - b), resampling PATCHES (not values).

    Resampling the paired differences preserves the pairing — resampling a and b
    independently would reintroduce exactly the patch-difficulty variance the
    pairing exists to remove, and inflate the interval.
    """
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    n = d.size
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    rng = _rng(seed)
    idx = rng.integers(0, n, size=(resamples, n))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(d.mean()), float(lo), float(hi)


def paired_permutation_test(a: np.ndarray, b: np.ndarray, resamples: int = 10000,
                            seed: int = 0) -> float:
    """
    Two-sided exact-style permutation test on the paired differences.

    Under the null "the two models are interchangeable on a given patch", the
    sign of each paired difference is exchangeable. So the null distribution is
    built by flipping signs at random, not by shuffling labels across patches.
    """
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    n = d.size
    if n == 0:
        return float("nan")
    obs = abs(d.mean())
    rng = _rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(resamples, n))
    null = np.abs((signs * d).mean(axis=1))
    # +1 in numerator and denominator: a permutation test can never report p=0,
    # since the observed statistic is itself one of the possible arrangements.
    return float((np.sum(null >= obs) + 1) / (resamples + 1))


def cliffs_delta(a: np.ndarray, b: np.ndarray, max_n: int = 2000,
                 seed: int = 0) -> float:
    """
    Cliff's delta: P(a > b) - P(a < b), a non-parametric effect size in [-1, 1].

    Reported next to the mean difference because it is scale-free — useful when
    comparing an effect measured in dB against one measured in radians. Exact
    computation is O(n^2), so large inputs are subsampled.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0:
        return float("nan")
    if a.size > max_n:
        rng = _rng(seed)
        sel = rng.choice(a.size, size=max_n, replace=False)
        a, b = a[sel], b[sel]
    diff = a[:, None] - b[None, :]
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / diff.size)


def holm_correct(p_values: Sequence[float]) -> list[float]:
    """
    Holm-Bonferroni step-down adjustment.

    Holm rather than plain Bonferroni: it controls the same family-wise error
    rate but is uniformly more powerful, which matters because the model-pair
    family within a dataset has 6 members (4 models choose 2) and Bonferroni
    would cost real sensitivity for no gain in rigour.
    """
    p = np.asarray(p_values, dtype=np.float64)
    m = p.size
    if m == 0:
        return []
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=np.float64)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        running = max(running, val)          # enforce monotonicity
        adjusted[i] = min(1.0, running)
    return [float(x) for x in adjusted]


def compare(metric: str, model_a: str, model_b: str,
            a: np.ndarray, b: np.ndarray, *,
            min_effect: float, higher_is_better: bool,
            cfg: dict, seed: int = 0) -> PairedResult:
    """
    One paired comparison, fully decided against the preregistered rules.

    `min_effect` is the preregistered minimum meaningful difference for this
    metric; `higher_is_better` orients the verdict (PSNR/SSIM up, SAM/MSE down).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    delta, lo, hi = paired_bootstrap_ci(
        a, b, cfg.get("bootstrap_resamples", 10000), cfg.get("alpha", 0.05), seed)
    p = paired_permutation_test(a, b, cfg.get("permutation_resamples", 10000), seed)
    cd = cliffs_delta(a, b, seed=seed)

    significant = p < cfg.get("alpha", 0.05)
    meaningful = abs(delta) >= min_effect
    better = (delta > 0) if higher_is_better else (delta < 0)

    if not significant:
        verdict = "no_difference"
    elif not meaningful:
        verdict = "significant_but_negligible"
    else:
        verdict = f"{model_a}_better" if better else f"{model_b}_better"

    return PairedResult(
        metric=metric, model_a=model_a, model_b=model_b, n=int(a.size),
        mean_a=float(a.mean()), mean_b=float(b.mean()),
        delta=delta, ci_low=lo, ci_high=hi,
        p_value=p, p_holm=None, cliffs_delta=cd,
        significant=significant, meaningful=meaningful, verdict=verdict,
    )


def apply_holm(results: list[PairedResult], cfg: dict) -> list[PairedResult]:
    """
    Apply Holm within a family and re-decide `significant`/`verdict`.

    The family is every pairwise comparison of one metric within one dataset —
    that is the set of tests you would look across before making a claim, so
    that is the set the correction must cover.
    """
    if not results:
        return results
    adj = holm_correct([r.p_value for r in results])
    alpha = cfg.get("alpha", 0.05)
    for r, pa in zip(results, adj):
        r.p_holm = pa
        r.significant = pa < alpha
        if not r.significant:
            r.verdict = "no_difference"
        elif not r.meaningful:
            r.verdict = "significant_but_negligible"
    return results
