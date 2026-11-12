"""Run the walk-forward study for a config and write a markdown report with figures."""

import sys
import tomllib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .core import fit_pair, positions, zscore
from .walk import candidate_pairs, market_beta, metrics, sharpe_ci, walk_forward

ACCENT, DARK, GREY, LIGHT = "#b3261e", "#222222", "#8a8a8a", "#d9d9d9"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight",
                     "text.color": DARK, "axes.labelcolor": DARK, "axes.edgecolor": DARK,
                     "xtick.color": DARK, "ytick.color": DARK})


def load_prices(tickers, start, end, cache_dir="data", refresh=False) -> pd.DataFrame:
    cache = Path(cache_dir) / f"prices_{start}_{end}.csv"
    if cache.exists() and not refresh:
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        if set(tickers) <= set(df.columns):
            return clean(df[list(tickers)])
    import yfinance as yf

    df = yf.download(list(tickers), start=start, end=end, auto_adjust=True, progress=False)["Close"]
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache)
    return clean(df[list(tickers)])


def clean(df: pd.DataFrame, max_move: float = 0.4) -> pd.DataFrame:
    df = df.dropna(how="any")
    worst = df.pct_change().abs().max()
    bad = worst[worst > max_move]
    if len(bad):
        raise ValueError(f"daily move above {max_move:.0%}, likely bad data: {bad.to_dict()}")
    return df


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def name(t):
    return t.replace(".NS", "")


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def plot_equity(curves: dict, path):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for (label, r), c in zip(curves.items(), [ACCENT, DARK, GREY]):
        ax.plot((1 + r).cumprod(), color=c, lw=1.2, label=label)
    ax.axhline(1, color=LIGHT, lw=0.8)
    ax.set_ylabel("growth of 1")
    ax.set_title("out of sample equity, after costs", loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def plot_example(prices, row, formation, cfg, path):
    a, b = row.a, row.b
    t0 = prices.index.get_loc(row.start)
    t1 = prices.index.get_loc(row.end) + 1
    logp = np.log(prices[[a, b]])
    form, span = logp.iloc[t0 - formation:t0], logp.iloc[t0 - formation:t1]
    fit = fit_pair(form[a].to_numpy(), form[b].to_numpy())
    z = pd.Series(zscore(span[a].to_numpy(), span[b].to_numpy(), fit), index=span.index)
    target = positions(z.iloc[formation:].to_numpy(), cfg["entry"], cfg["exit"], cfg["stop"])
    fig, (u, v) = plt.subplots(2, 1, figsize=(7, 4.6), sharex=True)
    norm = prices[[a, b]].iloc[t0 - formation:t1]
    norm = norm / norm.iloc[0] * 100
    u.plot(norm[a], color=DARK, lw=1.1, label=name(a))
    u.plot(norm[b], color=GREY, lw=1.1, label=name(b))
    u.set_ylabel("price (start = 100)")
    u.legend(frameon=False, fontsize=8)
    u.set_title(f"{name(a)} and {name(b)}: formation window, then trading window", loc="left")
    v.plot(z, color=DARK, lw=0.9)
    for level in (cfg["entry"], -cfg["entry"]):
        v.axhline(level, color=GREY, lw=0.7, ls="--")
    v.axhline(0, color=LIGHT, lw=0.7)
    trade_idx = span.index[formation:]
    v.fill_between(trade_idx, z.min(), z.max(), where=target != 0, color=ACCENT, alpha=0.18, lw=0)
    for ax in (u, v):
        ax.axvline(row.start, color=ACCENT, lw=0.9)
    v.set_ylabel("spread z-score")
    loc = matplotlib.dates.AutoDateLocator(maxticks=7)
    v.xaxis.set_major_locator(loc)
    v.xaxis.set_major_formatter(matplotlib.dates.ConciseDateFormatter(loc))
    fig.savefig(path)
    plt.close(fig)


def run(config_path="config.toml", out_dir="results", refresh=False, log=print):
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sectors = cfg["sectors"]
    tickers = [t for names in sectors.values() for t in names]
    allp = load_prices(tickers + [cfg["market"]], cfg["start"], cfg["end"], refresh=refresh)
    prices, market = allp[tickers], allp[cfg["market"]].pct_change().fillna(0.0)
    pairs = candidate_pairs(sectors)
    kw = dict(formation=cfg["formation"], trading=cfg["trading"], top=cfg["top"], max_p=cfg["max_p"],
              hl_range=tuple(cfg["half_life"]), entry=cfg["entry"], exit=cfg["exit"], stop=cfg["stop"],
              cost=cfg["cost"], cache={})
    seed, n_boot = cfg.get("seed", 0), cfg.get("n_boot", 2000)
    log(f"{len(prices)} days, {len(tickers)} stocks, {len(pairs)} candidate pairs")

    r, tab = walk_forward(prices, pairs, select="coint", **kw)
    r_all, tab_all = walk_forward(prices, pairs, select="all", **kw)
    r_free, _ = walk_forward(prices, pairs, select="coint", **{**kw, "cost": 0.0})
    rand = [walk_forward(prices, pairs, select="random", seed=s, **kw)[0] for s in range(cfg.get("random_runs", 20))]
    rand_sharpes = np.array([metrics(x)["sharpe"] for x in rand])
    log("walk forward done")

    def row(label, x):
        m = metrics(x)
        lo, hi, _ = sharpe_ci(x, n_boot=n_boot, seed=seed)
        beta, corr = market_beta(x, market)
        return [label, pct(m["cagr"]), pct(m["vol"]), f"{m['sharpe']:.2f}", f"{lo:.2f} to {hi:.2f}",
                pct(m["max_dd"]), f"{beta:+.2f}"]

    n_periods = tab_all["start"].nunique()
    md = ["# Pairs trading on NSE stocks", "",
          f"{len(tickers)} stocks in {len(sectors)} sectors, {len(pairs)} same-sector pairs. Every "
          f"{cfg['trading']} trading days the pairs are tested for cointegration on the previous "
          f"{cfg['formation']} days, the best {cfg['top']} are traded for the next {cfg['trading']} "
          f"days, then the process repeats. {n_periods} trading windows, {r.index[0].date()} to "
          f"{r.index[-1].date()}. Enter at {cfg['entry']} standard deviations, exit at {cfg['exit']}, "
          f"stop at {cfg['stop']}. {cfg['cost'] * 1e4:.0f} basis points per leg per trade. Returns are "
          f"on gross capital (long plus short), with capital split evenly over {cfg['top']} slots.", "",
          "## Out of sample result", "",
          table(["", "return / yr", "vol / yr", "Sharpe", "Sharpe 95% interval", "max drawdown", "beta to Nifty"],
                [row(f"cointegrated pairs, top {cfg['top']}", r),
                 row("same, before costs", r_free),
                 row("all pairs, no test", r_all)]), "",
          f"Control: {len(rand)} runs that pick {cfg['top']} same-sector pairs at random each window "
          f"have Sharpe ratios from {rand_sharpes.min():.2f} to {rand_sharpes.max():.2f}, mean "
          f"{rand_sharpes.mean():.2f}. {int((rand_sharpes >= metrics(r)['sharpe']).sum())} of "
          f"{len(rand)} match or beat the cointegration selection.", "",
          "![equity](equity.png)", ""]
    plot_equity({f"cointegrated pairs, top {cfg['top']}": r, "all pairs, no test": r_all,
                 "cointegrated pairs before costs": r_free}, out / "equity.png")

    # does cointegration persist?
    traded = tab[tab["entries"] > 0]
    still = (tab["trade_p"] < 0.05).mean()
    base = (tab_all["trade_p"] < 0.05).mean()
    form_rate = (tab_all["form_p"] < 0.05).mean()
    md += ["## Does cointegration last?", "",
           f"Across all windows {pct(form_rate)} of candidate pairs test as cointegrated at the 5% "
           f"level in the formation window. Of the {len(tab)} pair-windows selected, "
           f"{pct(still)} still test as cointegrated in the window they were traded in. For pairs "
           f"picked without any test the figure is {pct(base)}.", "",
           table(["selected pair-windows", "with at least one trade", "profitable", "hit the stop",
                  "mean return when traded", "median half-life (days)"],
                 [[len(tab), len(traded), f"{int((traded['return'] > 0).sum())} ({pct((traded['return'] > 0).mean(), 0)})",
                   f"{int(tab['stopped'].sum())} ({pct(tab['stopped'].mean(), 0)})",
                   pct(traded["return"].mean(), 2), f"{tab['half_life'].median():.0f}"]]), "",
           "Split by whether the stop was hit:", "",
           table(["", "pair-windows", "mean return", "share profitable"],
                 [[lab, len(g), pct(g["return"].mean(), 2), pct((g["return"] > 0).mean(), 0)]
                  for lab, g in (("stop not hit", traded[~traded["stopped"]]), ("stop hit", traded[traded["stopped"]]))
                  if len(g)]), ""]

    # by year
    yr = pd.DataFrame({"pairs": r, "nifty": market.reindex(r.index)}).groupby(r.index.year).apply(
        lambda d: (1 + d).prod() - 1)
    md += ["## By year", "", table(["year", "pairs strategy", "Nifty 50"],
                                   [[y, pct(v.pairs), pct(v.nifty)] for y, v in yr.iterrows()]), ""]

    # which pairs
    by_pair = tab.assign(pair=tab["a"].map(name) + " / " + tab["b"].map(name)).groupby("pair").agg(
        windows=("return", "size"), total=("return", "sum")).sort_values("windows", ascending=False)
    md += ["## Pairs selected most often", "",
           table(["pair", "windows selected", "sum of window returns"],
                 [[p, int(v.windows), pct(v.total)] for p, v in by_pair.head(10).iterrows()]), ""]

    best = traded.loc[traded["return"].idxmax()]
    worst = traded.loc[traded["return"].idxmin()]
    plot_example(prices, best, cfg["formation"], cfg, out / "example_best.png")
    plot_example(prices, worst, cfg["formation"], cfg, out / "example_worst.png")
    md += ["## Best and worst pair-window", "",
           f"Best: {name(best.a)} / {name(best.b)} from {best.start.date()}, {pct(best['return'])}. "
           f"Worst: {name(worst.a)} / {name(worst.b)} from {worst.start.date()}, {pct(worst['return'])}. "
           f"The red line marks the end of the formation window; shading marks days with a position.", "",
           "![best](example_best.png)", "", "![worst](example_worst.png)", ""]

    # sensitivity
    md += ["## Sensitivity", ""]
    rows = []
    for label, change in [("base case", {}), ("entry 1.5", {"entry": 1.5}), ("entry 2.5", {"entry": 2.5}),
                          ("no stop", {"stop": 99.0}), ("top 3", {"top": 3}), ("top 10", {"top": 10}),
                          ("formation 252 days", {"formation": 252}), ("trading 63 days", {"trading": 63}),
                          ("cost 20 bp", {"cost": 0.002})]:
        x = r if not change else walk_forward(prices, pairs, select="coint", **{**kw, **change})[0]
        m = metrics(x)
        rows.append([label, pct(m["cagr"]), f"{m['sharpe']:.2f}", pct(m["max_dd"])])
    md += [table(["setting", "return / yr", "Sharpe", "max drawdown"], rows), ""]

    (out / "report.md").write_text("\n".join(md) + "\n")
    tab.to_csv(out / "selected_pairs.csv", index=False, float_format="%.5f")
    log(f"wrote {out / 'report.md'}")


if __name__ == "__main__":
    run(*sys.argv[1:2])
