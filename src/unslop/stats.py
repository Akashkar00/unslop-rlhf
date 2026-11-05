"""Bootstrap CIs, length-controlled win rate, and agreement statistics."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class CI:
    point: float
    lo: float
    hi: float

    def as_dict(self) -> dict:
        return asdict(self)

    def excludes(self, value: float) -> bool:
        return self.lo > value or self.hi < value


def bootstrap_ci(
    n: int, statistic: Callable[[np.ndarray], float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> CI:
    """Percentile bootstrap. ``statistic`` receives an index array and resamples
    paired arrays itself, so per-prompt pairing is preserved."""
    if n == 0:
        nan = float("nan")
        return CI(nan, nan, nan)
    rng = np.random.default_rng(seed)
    point = float(statistic(np.arange(n)))
    boots = np.array([statistic(rng.integers(0, n, n)) for _ in range(n_boot)])
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return CI(point, float(lo), float(hi))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 1e-2, iters: int = 100) -> np.ndarray:
    """One-feature logistic regression by Newton's method. ``y`` may be fractional
    (ties = 0.5). The small L2 term keeps the fit finite under perfect separation."""
    X = np.column_stack([np.ones_like(x, dtype=float), np.asarray(x, dtype=float)])
    y = np.asarray(y, dtype=float)
    w = np.zeros(2)
    for _ in range(iters):
        p = _sigmoid(X @ w)
        grad = X.T @ (p - y) + l2 * w
        hess = X.T @ (X * (p * (1 - p))[:, None]) + l2 * np.eye(2)
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < 1e-9:
            break
    return w


def length_controlled_win_rate(scores: Sequence[float], len_a: Sequence[float], len_b: Sequence[float]) -> float:
    """P(A beats B) when both responses have the same length.

    Fits win ~ log(len_a / len_b) and reads off the intercept. A simplified
    version of the AlpacaEval 2 LC idea: if A only wins because it is longer,
    this goes to 0.5 while the raw win rate stays high.
    """
    la = np.maximum(np.asarray(len_a, dtype=float), 1.0)
    lb = np.maximum(np.asarray(len_b, dtype=float), 1.0)
    b0, _ = fit_logistic(np.log(la / lb), np.asarray(scores, dtype=float))
    return float(_sigmoid(b0))


def to_label(score: float) -> str:
    return "x" if score > 0.5 else "y" if score < 0.5 else "tie"


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    if len(a) != len(b) or not a:
        return float("nan")
    a, b = np.asarray(a), np.asarray(b)
    p_obs = float(np.mean(a == b))
    cats = np.union1d(a, b)
    p_exp = float(sum(np.mean(a == c) * np.mean(b == c) for c in cats))
    return 1.0 if p_exp == 1.0 else (p_obs - p_exp) / (1 - p_exp)
