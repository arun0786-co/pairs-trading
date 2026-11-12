"""Walk-forward pairs trading: pick pairs on a formation window, trade them on the next window."""

from itertools import combinations

import numpy as np
import pandas as pd

from .core import fit_pair, pair_returns, positions, zscore

TRADING_DAYS = 252


def candidate_pairs(sectors: dict) -> list[tuple[str, str]]:
    return [p for names in sectors.values() for p in combinations(names, 2)]


def windows(n: int, formation: int, trading: int):
    """(formation start, formation end = trading start, trading end) index triples."""
    t = formation
    while t < n:
        yield t - formation, t, min(t + trading, n)
        t += trading


def walk_forward(prices: pd.DataFrame, pairs, formation=504, trading=126, top=5,
                 max_p=0.05, hl_range=(2.0, 60.0), entry=2.0, exit=0.5, stop=4.0,
                 cost=0.001, select="coint", seed=0, cache=None):
    """Returns (daily portfolio returns, table of every traded pair-period).

    select = "coint": the `top` pairs with the lowest Engle-Granger p-value that pass the
    filters. select = "all": every pair with a positive hedge ratio, no test (control).
    select = "random": `top` pairs at random among those with a positive hedge ratio.
    Capital is split evenly over `top` slots (or over all pairs for "all"); empty slots earn 0.
    cache is an optional dict that keeps formation fits between calls on the same prices.
    """
    logp = np.log(prices)
    rets = prices.pct_change().fillna(0.0)
    rng = np.random.default_rng(seed)
    daily, rows = [], []
    for f0, t0, t1 in windows(len(prices), formation, trading):
        form, trade = logp.iloc[f0:t0], logp.iloc[t0:t1]
        fits = {}
        for a, b in pairs:
            key = (a, b, f0, t0)
            if cache is not None and key in cache:
                fit = cache[key]
            else:
                fit = fit_pair(form[a].to_numpy(), form[b].to_numpy())
                if cache is not None:
                    cache[key] = fit
            if fit.beta > 0:
                fits[(a, b)] = fit
        if select == "coint":
            ok = [k for k, f in fits.items()
                  if f.pvalue < max_p and hl_range[0] <= f.half_life <= hl_range[1]]
            chosen = sorted(ok, key=lambda k: fits[k].pvalue)[:top]
            slots = top
        elif select == "random":
            keys = list(fits)
            chosen = [keys[i] for i in rng.choice(len(keys), min(top, len(keys)), replace=False)]
            slots = top
        elif select == "all":
            chosen, slots = list(fits), max(len(fits), 1)
        else:
            raise ValueError(select)

        port = np.zeros(t1 - t0)
        for a, b in chosen:
            fit = fits[(a, b)]
            z = zscore(trade[a].to_numpy(), trade[b].to_numpy(), fit)
            target = positions(z, entry, exit, stop)
            target[-1] = 0.0  # flat at the end of the trading window
            r = pair_returns(rets[a].iloc[t0:t1].to_numpy(), rets[b].iloc[t0:t1].to_numpy(),
                             target, fit.beta, cost)
            port += r / slots
            # was the pair still cointegrated in the window it was traded on?
            later = fit_pair(trade[a].to_numpy(), trade[b].to_numpy())
            held = np.concatenate([[0.0], target[:-1]])
            rows.append({
                "start": prices.index[t0], "end": prices.index[t1 - 1], "a": a, "b": b,
                "beta": fit.beta, "form_p": fit.pvalue, "half_life": fit.half_life,
                "trade_p": later.pvalue, "entries": int(np.sum((np.diff(target, prepend=0) != 0) & (target != 0))),
                "stopped": bool(np.any((held != 0) & (np.abs(z) >= stop))),
                "return": float(np.prod(1 + r) - 1),
            })
        daily.append(pd.Series(port, index=prices.index[t0:t1]))
    table = pd.DataFrame(rows)
    return pd.concat(daily), table


def metrics(r: pd.Series) -> dict:
    eq = (1 + r).cumprod()
    sd = r.std()
    return {
        "cagr": float(eq.iloc[-1] ** (TRADING_DAYS / len(r)) - 1),
        "vol": float(sd * np.sqrt(TRADING_DAYS)),
        "sharpe": float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0,
        "max_dd": float((1 - eq / eq.cummax()).max()),
    }


def market_beta(r: pd.Series, market: pd.Series) -> tuple[float, float]:
    """Slope and correlation of strategy returns against market returns."""
    m = market.reindex(r.index).fillna(0.0)
    c = np.cov(r, m)
    return float(c[0, 1] / c[1, 1]), float(np.corrcoef(r, m)[0, 1])


def sharpe_ci(r, n_boot=2000, block=20.0, level=0.95, seed=0):
    """Stationary bootstrap interval for the Sharpe ratio, and the share of resamples below 0."""
    x = np.asarray(r)
    n = len(x)
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for i in range(n_boot):
        new = rng.random(n) < 1.0 / block
        new[0] = True
        first = np.flatnonzero(new)
        bid = np.cumsum(new) - 1
        idx = (rng.integers(0, n, len(first))[bid] + np.arange(n) - first[bid]) % n
        s = x[idx]
        out[i] = s.mean() / s.std(ddof=1) * np.sqrt(TRADING_DAYS)
    lo, hi = np.quantile(out, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi), float(np.mean(out <= 0))
