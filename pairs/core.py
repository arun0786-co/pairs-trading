"""Cointegration tests, spread construction and the trading rule for one pair."""

from dataclasses import dataclass

import numpy as np
from statsmodels.tsa.stattools import coint


@dataclass
class Fit:
    alpha: float
    beta: float  # hedge ratio: log A = alpha + beta * log B + spread
    mean: float
    std: float
    pvalue: float  # Engle-Granger test, null = no cointegration
    half_life: float  # days for half of a spread deviation to decay


def hedge(log_a: np.ndarray, log_b: np.ndarray) -> tuple[float, float]:
    """OLS of log A on log B. Returns (alpha, beta)."""
    x = np.column_stack([np.ones_like(log_b), log_b])
    alpha, beta = np.linalg.lstsq(x, log_a, rcond=None)[0]
    return float(alpha), float(beta)


def half_life(spread: np.ndarray) -> float:
    """Half-life from an AR(1) fit: d s_t = c + theta * s_{t-1} + e_t."""
    lag, diff = spread[:-1], np.diff(spread)
    x = np.column_stack([np.ones_like(lag), lag])
    theta = np.linalg.lstsq(x, diff, rcond=None)[0][1]
    if theta >= 0 or theta <= -1:
        return np.inf
    return float(-np.log(2) / np.log1p(theta))


def fit_pair(log_a: np.ndarray, log_b: np.ndarray) -> Fit:
    alpha, beta = hedge(log_a, log_b)
    spread = log_a - alpha - beta * log_b
    pvalue = float(coint(log_a, log_b, trend="c", autolag="aic")[1])
    return Fit(alpha, beta, float(spread.mean()), float(spread.std(ddof=1)), pvalue, half_life(spread))


def zscore(log_a, log_b, fit: Fit) -> np.ndarray:
    """Spread in formation-period standard deviations. Uses formation parameters only."""
    return (log_a - fit.alpha - fit.beta * log_b - fit.mean) / fit.std


def positions(z: np.ndarray, entry=2.0, exit=0.5, stop=4.0) -> np.ndarray:
    """Target position in the spread decided at each close: +1 long A short B, -1 the reverse.

    Enter when |z| passes entry, close when z comes back inside exit, and close for the rest
    of the period when |z| passes stop (the relationship is treated as broken).
    """
    pos = np.zeros(len(z))
    cur, dead = 0.0, False
    for t, zt in enumerate(z):
        if dead:
            cur = 0.0
        elif cur == 0.0:
            if entry <= abs(zt) < stop:
                cur = -np.sign(zt)
        elif abs(zt) >= stop:
            cur, dead = 0.0, True
        elif cur * zt >= -exit:  # long spread exits once z >= -exit, short once z <= exit
            cur = 0.0
        pos[t] = cur
    return pos


def pair_returns(ret_a, ret_b, target, beta, cost=0.001, carry_in=0.0) -> np.ndarray:
    """Daily return on gross capital for one pair.

    target[t] is decided at close t and held over day t + 1. Capital of 1 is split
    1 / (1 + beta) in A and beta / (1 + beta) in B on opposite sides. cost is charged on
    the traded notional of both legs. carry_in is the position held coming into day 0.
    """
    held = np.concatenate([[carry_in], target[:-1]])
    gross = (ret_a - beta * ret_b) / (1 + beta)
    traded = np.abs(np.diff(np.concatenate([[carry_in], target])))
    # the trade at close t is paid for on day t; it changes the return from day t + 1
    return held * gross - cost * traded
