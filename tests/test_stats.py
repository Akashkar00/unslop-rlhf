import numpy as np
import pytest

from unslop.stats import bootstrap_ci, cohen_kappa, length_controlled_win_rate, to_label


def test_bootstrap_ci_brackets_the_mean():
    x = np.random.default_rng(0).normal(0.6, 0.2, 400)
    ci = bootstrap_ci(len(x), lambda i: x[i].mean(), n_boot=500)
    assert ci.lo < ci.point < ci.hi
    assert ci.lo < 0.6 < ci.hi


def test_bootstrap_ci_empty_is_nan():
    assert np.isnan(bootstrap_ci(0, lambda i: 0.0).point)


def test_lc_win_rate_removes_pure_length_effect():
    # A wins only because it is longer, and it usually is longer: high raw win
    # rate, but at equal length it should be a coin flip.
    rng = np.random.default_rng(1)
    log_ratio = rng.normal(1.0, 1.0, 3000)
    wins = (rng.random(3000) < 1 / (1 + np.exp(-3 * log_ratio))).astype(float)
    len_b = np.full(3000, 200.0)
    len_a = len_b * np.exp(log_ratio)
    assert wins.mean() > 0.75
    assert length_controlled_win_rate(wins, len_a, len_b) == pytest.approx(0.5, abs=0.05)


def test_lc_win_rate_keeps_real_quality_effect():
    rng = np.random.default_rng(2)
    n = 3000
    len_a = len_b = np.full(n, 150.0)
    wins = (rng.random(n) < 0.7).astype(float)
    assert length_controlled_win_rate(wins, len_a, len_b) == pytest.approx(0.7, abs=0.03)


def test_lc_win_rate_survives_perfect_separation():
    assert 0.5 < length_controlled_win_rate([1, 1, 1, 1], [100] * 4, [100] * 4) < 1.0


def test_cohen_kappa():
    assert cohen_kappa(["x", "y", "x", "y"], ["x", "y", "x", "y"]) == pytest.approx(1.0)
    assert cohen_kappa(["x", "x", "y", "y"], ["x", "y", "x", "y"]) == pytest.approx(0.0)
    assert cohen_kappa(["x", "x"], ["x", "x"]) == 1.0


def test_to_label():
    assert [to_label(s) for s in (1.0, 0.75, 0.5, 0.25, 0.0)] == ["x", "x", "tie", "y", "y"]
