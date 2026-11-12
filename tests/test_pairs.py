import numpy as np
import pandas as pd
import pytest

from pairs.core import fit_pair, half_life, hedge, pair_returns, positions, zscore
from pairs.walk import candidate_pairs, market_beta, metrics, sharpe_ci, walk_forward, windows


def ou(rng, n, theta=0.1, sigma=0.02):
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = s[t - 1] * (1 - theta) + rng.normal(0, sigma)
    return s


def make_pair(rng, n, beta=1.5, theta=0.1):
    log_b = np.log(100) + np.cumsum(rng.normal(0.0003, 0.015, n))
    log_a = 0.4 + beta * log_b + ou(rng, n, theta)
    return log_a, log_b


def test_hedge_and_half_life_recover_truth():
    rng = np.random.default_rng(0)
    log_a, log_b = make_pair(rng, 5000, beta=1.5, theta=0.1)
    alpha, beta = hedge(log_a, log_b)
    assert beta == pytest.approx(1.5, abs=0.02)
    hl = half_life(log_a - alpha - beta * log_b)
    assert hl == pytest.approx(-np.log(2) / np.log(0.9), rel=0.15)
    assert half_life(np.cumsum(np.ones(50))) == np.inf


def test_cointegration_test_separates_pairs_from_independent_walks():
    rng = np.random.default_rng(1)
    log_a, log_b = make_pair(rng, 500)
    assert fit_pair(log_a, log_b).pvalue < 0.01
    walks = [np.cumsum(rng.normal(0, 0.015, (500, 2)), axis=0) for _ in range(60)]
    rejected = np.mean([fit_pair(w[:, 0], w[:, 1]).pvalue < 0.05 for w in walks])
    assert rejected < 0.2


def test_position_rule_by_hand():
    z = np.array([0.0, 2.1, 1.0, 0.4, -2.5, -1.0, -0.5, 3.0, 4.5, 2.5, 0.0])
    assert positions(z).tolist() == [0, -1, -1, 0, 1, 1, 0, -1, 0, 0, 0]
    # a jump straight past the stop is not an entry
    assert positions(np.array([0.0, 5.0, 3.0])).tolist() == [0, 0, -1]


def test_pair_returns_lag_costs_and_sign():
    ret_a = np.array([0.0, 0.02, 0.01, -0.03])
    ret_b = np.array([0.0, 0.01, 0.02, 0.01])
    target = np.array([1.0, 1.0, 0.0, 0.0])
    r = pair_returns(ret_a, ret_b, target, beta=1.0, cost=0.001)
    # day 0: enter, pay cost. days 1, 2: hold (a - b) / 2. day 2: exit cost. day 3: flat
    assert np.allclose(r, [-0.001, 0.005, -0.005 - 0.001, 0.0])
    assert np.allclose(pair_returns(ret_a, ret_b, -target, 1.0, 0.0), [0, -0.005, 0.005, 0])


def test_windows_do_not_overlap_and_formation_precedes_trading():
    w = list(windows(1000, 500, 120))
    assert w[0] == (0, 500, 620) and w[-1][2] == 1000
    assert all(a[2] == b[1] for a, b in zip(w, w[1:]))
    assert all(t0 - f0 == 500 for f0, t0, _ in w)


def frame(rng, n=1600):
    idx = pd.bdate_range("2015-01-01", periods=n)
    a1, b1 = make_pair(rng, n, beta=1.0, theta=0.08)
    a2, b2 = make_pair(rng, n, beta=0.7, theta=0.08)
    noise = np.log(100) + np.cumsum(rng.normal(0, 0.015, (n, 2)), axis=0)
    data = np.exp(np.column_stack([a1, b1, a2, b2, noise]))
    return pd.DataFrame(data, index=idx, columns=["A1", "B1", "A2", "B2", "N1", "N2"])


def test_walk_forward_profits_on_cointegrated_pairs_and_ignores_the_future():
    rng = np.random.default_rng(2)
    p = frame(rng)
    pairs = candidate_pairs({"x": ["A1", "B1"], "y": ["A2", "B2"], "z": ["N1", "N2"]})
    r, tab = walk_forward(p, pairs, formation=500, trading=120, top=2, cost=0.0005)
    assert r.index[0] == p.index[500] and r.index.is_unique and len(r) == 1100
    assert metrics(r)["sharpe"] > 1.0
    assert set(tab["a"]) <= {"A1", "A2"}
    # changing prices after day 1000 must not change returns before it
    q = p.copy()
    q.iloc[1000:] *= np.linspace(1, 3, 600)[:, None] * np.array([1, 2, 1, 2, 1, 2])
    r2, _ = walk_forward(q, pairs, formation=500, trading=120, top=2, cost=0.0005)
    assert np.allclose(r.iloc[:480], r2.iloc[:480])


def test_costs_only_reduce_returns():
    rng = np.random.default_rng(3)
    p = frame(rng)
    pairs = [("A1", "B1"), ("A2", "B2")]
    free, _ = walk_forward(p, pairs, formation=500, trading=120, top=2, cost=0.0)
    paid, _ = walk_forward(p, pairs, formation=500, trading=120, top=2, cost=0.002)
    assert (paid <= free + 1e-15).all() and paid.sum() < free.sum()


def test_market_beta_and_sharpe_ci():
    rng = np.random.default_rng(4)
    idx = pd.bdate_range("2020-01-01", periods=3000)
    m = pd.Series(rng.normal(0.0004, 0.01, 3000), index=idx)
    r = 0.3 * m + rng.normal(0.0005, 0.004, 3000)
    assert market_beta(r, m)[0] == pytest.approx(0.3, abs=0.03)
    lo, hi, p0 = sharpe_ci(r, n_boot=400)
    assert lo < metrics(r)["sharpe"] < hi and lo > 0 and p0 < 0.01
