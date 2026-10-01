# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: blackrock-intraday
#     language: python
#     name: blackrock-intraday
# ---

# %% [markdown]
# # Backtest — each asset on its own, then the combined book
#
# Trades the sign of `dtw_dir`, gated and sized as set in section 0, on the test years
# (2021-2026). One section per asset compares the strategy at three cost levels against
# that asset's buy-and-hold; section 4 compares two multi-asset books, equities (ES + NQ) and
# combined (ES + NQ + ZN), against risk-weighted buy-and-holds of the same assets. Every
# statistic reported is from the test years.
#
# **Units.** All P&L is in basis points of one contract's notional, summed rather than
# compounded, on a daily grid of trading sessions with zero on days with no trade.
# Buy-and-hold skips the price jump at each contract roll. The strategy is in the market
# only ~30 minutes around releases, so its exposure is far smaller than buy-and-hold's —
# compare Sharpe and max drawdown, not total return.
#
# **The test years are run once.** `EVAL_YEARS` is the period scored; the gate and sizing
# constants use only earlier years.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent if "__file__" in globals() else "."))
from config import (ALL_ASSETS as ASSETS, BOOK_SYMBOLS, CLEAN_BARS_FMT, COST_TICKS_PER_SIDE, GATE_GRID, MAX_POSITION, NY,
                    OUT, PANEL, SESSION_SHIFT, SIZING_GRID, TEST_YEARS_SPAN, TRAIN_YEARS_SPAN)

pd.set_option("display.width", 200)

EVAL_YEARS = TEST_YEARS_SPAN
FIT_YEARS = (TRAIN_YEARS_SPAN[0], EVAL_YEARS[0] - 1)   # everything before the scored period
TUNE_COST = "quarter tick"                             # cost level the gate and sizing are tuned at
symbols = BOOK_SYMBOLS
print(f"scored: {EVAL_YEARS[0]}-{EVAL_YEARS[1]}   gate/sizing tuned on: "
      f"{TRAIN_YEARS_SPAN[0]}-{TRAIN_YEARS_SPAN[1]}   constants from: {FIT_YEARS[0]}-{FIT_YEARS[1]}")


def session_of(ts: pd.Series) -> pd.Series:
    """The trading session a timestamp belongs to (18:00-17:00 ET, dated by the close)."""
    return (ts.dt.tz_convert(NY) + SESSION_SHIFT).dt.normalize().dt.tz_localize(None)


def in_years(days: pd.DatetimeIndex, span: tuple[int, int]) -> pd.DatetimeIndex:
    return days[(days.year >= span[0]) & (days.year <= span[1])]


panel = pd.read_parquet(PANEL)
games = panel[panel.admitted & panel.dtw_dir.notna()].copy()
bars = {}
for s in symbols:
    b = pd.read_parquet(CLEAN_BARS_FMT.format(symbol=s), columns=["close", "contract", "session_date"])
    b.index = pd.to_datetime(b.index, utc=True)
    bars[s] = b.sort_index()


def buy_and_hold(s: str) -> pd.Series:
    """Daily bps from holding one contract, skipping the jump at each roll."""
    b = bars[s]
    r = np.log(b.close).diff() * 1e4
    r[b.contract.ne(b.contract.shift())] = 0.0
    return r.fillna(0.0).groupby(pd.DatetimeIndex(b.session_date)).sum()


bnh = {s: buy_and_hold(s) for s in symbols}

# %% [markdown]
# ## The trading rule
#
# * **Direction.** The sign of `dtw_dir`.
# * **Gate.** Trade a game only if its `dtw_dir` is below the `q` quantile or above the
#   `1 − q` quantile of that asset's admitted games in the fitting years.
# * **Size.** Either 1 contract (`flat`), or `median(dtw_disp) / dtw_disp` capped at
#   `MAX_POSITION` (`inverse_disp`): take less when the neighbours disagreed about the move.
#
# One position per entry minute: when two admitted families print together their positions
# are averaged, so no minute is traded twice. Costs are charged per side on the contracts
# traded, in ticks converted to bps at the entry price.


# %%
def make_rule(fit: pd.DataFrame, q: float, sizing: str) -> dict:
    return {"q": q, "sizing": sizing, "tail_lo": fit.dtw_dir.quantile(q),
            "tail_hi": fit.dtw_dir.quantile(1 - q), "disp_ref": fit.dtw_disp.median()}


def trades_for(s: str, span: tuple[int, int], rule: dict) -> pd.DataFrame:
    g = games[(games.symbol == s) & games.year.between(*span)].copy()
    in_tail = (g.dtw_dir <= rule["tail_lo"]) | (g.dtw_dir >= rule["tail_hi"])
    size = (1.0 if rule["sizing"] == "flat"
            else np.clip(rule["disp_ref"] / g.dtw_disp, 0.0, MAX_POSITION))
    g["position"] = np.where(in_tail, np.sign(g.dtw_dir) * size, 0.0)
    t = (g[g.position != 0].groupby("entry_ts")
         .agg(position=("position", "mean"), fwd_ret=("fwd_ret", "first")).reset_index())
    t = t[t.position != 0]
    t["session"] = session_of(t.entry_ts)
    tick_bps = ASSETS[s]["tick"] / bars[s].close.reindex(t.entry_ts).to_numpy() * 1e4
    t["gross"] = t.position * t.fwd_ret * 1e4
    for name, ticks in COST_TICKS_PER_SIDE.items():
        t[name] = t.gross - t.position.abs() * 2 * ticks * tick_bps
    return t


def daily_pnl(t: pd.DataFrame, days: pd.DatetimeIndex, col: str) -> pd.Series:
    return t.groupby("session")[col].sum().reindex(days).fillna(0.0)


def sharpe(daily: pd.Series) -> float:
    sd = daily.std(ddof=1)
    return float(daily.mean() / sd * np.sqrt(252)) if sd else np.nan


# %% [markdown]
# ## 0. Gate and sizing — one pass on the training years
#
# Every (`q`, sizing) in `GATE_GRID` x `SIZING_GRID` is scored on 2013-2017 at a quarter
# tick per side, with the gate thresholds and `disp_ref` taken from those same training
# games. The pair with the best Sharpe averaged across the assets is chosen, then refit on
# every year before the scored period. Only the chosen rule is printed: this notebook reports
# test-year statistics only.

# %%
rows = []
for q in GATE_GRID:
    for sizing in SIZING_GRID:
        for s in symbols:
            fit = games[(games.symbol == s) & games.year.between(*TRAIN_YEARS_SPAN)]
            t = trades_for(s, TRAIN_YEARS_SPAN, make_rule(fit, q, sizing))
            d = daily_pnl(t, in_years(bnh[s].index, TRAIN_YEARS_SPAN), TUNE_COST)
            rows.append({"q": q, "sizing": sizing, "asset": s, "trades": len(t),
                         "sharpe": sharpe(d), "bps/trade": t[TUNE_COST].mean()})
tune = pd.DataFrame(rows)
grid = tune.pivot_table(index=["q", "sizing"], columns="asset", values="sharpe")[symbols]
grid["pooled"] = grid.mean(axis=1)
best_q, best_sizing = grid.pooled.idxmax()
print(f"chosen: q = {best_q} (trade the outer {2 * best_q:.0%} of games), sizing = {best_sizing}")

rules = {s: make_rule(games[(games.symbol == s) & games.year.between(*FIT_YEARS)], best_q, best_sizing)
         for s in symbols}
print(pd.DataFrame(rules).T.to_string())

# %% [markdown]
# **Findings:** The gate and sizing are fixed at the original ES book's rule (outer 60%,
# inverse `dtw_disp`), so nothing is chosen here. The cut-offs and `disp_ref` for each asset
# come from its 2013-2020 games.

# %% [markdown]
# ## Trades in the scored period

# %%
trades = {s: trades_for(s, EVAL_YEARS, rules[s]) for s in symbols}
for s in symbols:
    print(f"{s}: {len(trades[s]):,} trades in {EVAL_YEARS[0]}-{EVAL_YEARS[1]}")

# %% [markdown]
# ## Metrics
#
# Return, vol and Sharpe are annualised over 252 sessions. Beta is against the asset's own
# buy-and-hold (multi-asset books: against their risk-weighted buy-and-hold). Win rates count
# trades whose P&L at that cost level is positive; for buy-and-hold, days that closed up.


# %%
def metrics(daily: pd.Series, bench: pd.Series, t: pd.DataFrame | None = None,
            col: str | None = None) -> dict:
    ann = daily.mean() * 252 / 100
    vol = daily.std(ddof=1) * np.sqrt(252) / 100
    eq = daily.cumsum()
    out = {"total %": daily.sum() / 100, "ann return %": ann, "ann vol %": vol,
           "sharpe": ann / vol if vol else np.nan,
           "max DD %": float((eq - eq.cummax()).min()) / 100,
           "beta": float(np.cov(daily, bench)[0, 1] / bench.var()) if bench.var() else np.nan}
    if t is None:                                   # buy-and-hold
        out |= {"trades": np.nan, "win rate": float((daily > 0).mean()), "% long": 1.0,
                "% short": 0.0, "win rate long": float((daily > 0).mean()), "win rate short": np.nan}
    else:
        long, short = t[t.position > 0], t[t.position < 0]
        out |= {"trades": len(t), "win rate": float((t[col] > 0).mean()),
                "% long": len(long) / len(t), "% short": len(short) / len(t),
                "win rate long": float((long[col] > 0).mean()),
                "win rate short": float((short[col] > 0).mean())}
    return out


def eval_days(s: str) -> pd.DatetimeIndex:
    return in_years(bnh[s].index, EVAL_YEARS)


def asset_table(s: str) -> pd.DataFrame:
    days = eval_days(s)
    bench = bnh[s].reindex(days)
    rows = {}
    for col in COST_TICKS_PER_SIDE:
        daily = daily_pnl(trades[s], days, col)
        rows[f"strategy, {col}"] = metrics(daily, bench, trades[s], col)
    rows["buy and hold"] = metrics(bench, bench)
    return pd.DataFrame(rows).T


tables = {}

# %% [markdown]
# ## 1. ES — E-mini S&P 500

# %%
tables["ES"] = asset_table("ES")
print(tables["ES"].round(3).to_string())

# %% [markdown]
# **Findings:** ES, 2021-2026, 1,202 trades: Sharpe 0.78 gross, 0.64 at a quarter tick, 0.49
# at a half tick, against 0.69 for buy-and-hold. Max drawdown −5.0% at a quarter tick against
# −31.1%. Win rate 50.0%, 53% of trades long, beta 0.007.

# %% [markdown]
# ## 2. NQ — E-mini Nasdaq-100

# %%
tables["NQ"] = asset_table("NQ")
print(tables["NQ"].round(3).to_string())

# %% [markdown]
# **Findings:** NQ, 1,141 trades: Sharpe 0.14 gross, 0.11 at a quarter tick, 0.08 at a half
# tick, against 0.58 for buy-and-hold. Max drawdown −8.2% against −48.3%. Win rate 49.5%.
# Positive but weak: the ES setting carries to NQ only in part.

# %% [markdown]
# ## 3. ZN — 10-year Treasury note

# %%
tables["ZN"] = asset_table("ZN")
print(tables["ZN"].round(3).to_string())
# Extra assets (DTW_EXTRA=YM): the same table for each.
for s in [s for s in symbols if s not in tables]:
    tables[s] = asset_table(s)
    print(f"\n{s}:\n" + tables[s].round(3).to_string())

# %% [markdown]
# **Findings:** ZN, 1,945 trades: Sharpe 0.34 gross but −0.97 at a quarter tick and −2.24 at a
# half tick, against −0.69 for buy-and-hold. Max drawdown −8.0% at a quarter tick. Win rate 46.9%,
# 45% of trades long. Before costs the direction is slightly right; a quarter tick of ZN is large
# next to its 30-minute moves, so costs take all of it and more.

# %% [markdown]
# ## 4. Equities (ES + NQ) and combined (ES + NQ + ZN) vs risk-weighted buy-and-holds
#
# **Inverse vol with a cost gate, walk-forward.** Each day, an asset's strategy weight is
#
# `1 / trailing vol of the underlying`, or 0 if the asset fails the cost gate,
#
# normalised to sum to one within the book, so every book is in average-asset bps.
#
# * **Cost gate** (monthly): an asset is traded only while a round trip costs under
#   `COST_RATIO_MAX` of its median |30-minute move| on admitted games over the past year. The
#   round trip is at the cost level being reported and the price at the month start. It asks
#   whether costs are small next to the moves being traded, not whether past trades made money.
# * **Trailing vol** (daily): the sd of the underlying's daily return over the previous
#   `VOL_SESSIONS` sessions. An asset gets less weight when its market is more volatile.
#
# Chosen on 2016-2020 among eleven dynamic schemes in `experiments/weighting.py`; the gate at 10%
# sits between ES and NQ (1-7% every year 2014-2020) and ZN (13-21%). The weights use only data
# before the day they apply to. Each cost row below is the book a trader facing that cost would
# have run, so at zero cost every asset passes the gate. The checks in section 5 use the
# quarter-tick book. The buy-and-hold benchmark is weighted 1 / trailing vol. Trade counts and
# win rates pool the trades the book actually takes (weight above zero).

# %%
COST_RATIO_MAX = 0.10
VOL_SESSIONS = 60


def span_days(m: list[str], span: tuple[int, int]) -> pd.DatetimeIndex:
    days = in_years(bnh[m[0]].index, span)
    for s in m[1:]:
        days = days.union(in_years(bnh[s].index, span))
    return days


def cost_ratio(s: str, days: pd.DatetimeIndex, ticks: float) -> pd.Series:
    """Round trip at `ticks` per side over the median |30-minute move| of the asset's admitted
    games in the year before each day's month."""
    month = days.to_period("M")
    g = games[games.symbol == s]
    by_month = {}
    for p in month.unique():
        start = pd.Timestamp(p.start_time, tz="UTC")
        past = g[(g.entry_ts < start) & (g.entry_ts >= start - pd.DateOffset(years=1))]
        price = bars[s].close.iloc[bars[s].index.searchsorted(start) - 1]
        by_month[p] = 2 * ticks * ASSETS[s]["tick"] / price / past.fwd_ret.abs().median()
    return pd.Series([by_month[p] for p in month], index=days)


test_days = span_days(symbols, EVAL_YEARS)
ratio = {c: {} for c in COST_TICKS_PER_SIDE}
vol = {}
for s in symbols:
    for c, ticks in COST_TICKS_PER_SIDE.items():
        ratio[c][s] = cost_ratio(s, test_days, ticks)
    vol[s] = bnh[s].rolling(VOL_SESSIONS).std().shift(1).reindex(test_days).ffill()
raw_strat = {c: pd.DataFrame({s: (ratio[c][s] < COST_RATIO_MAX) / vol[s] for s in symbols})
             for c in COST_TICKS_PER_SIDE}
raw_bnh = pd.DataFrame({s: 1 / vol[s] for s in symbols})


def book_weights(m: list[str], days: pd.DatetimeIndex, raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """Daily weights of the assets in `m` (the quarter-tick book unless `raw` is given); a single
    asset is always weight 1."""
    if len(m) == 1:
        return pd.DataFrame(1.0, index=days, columns=m)
    w = (raw_strat[TUNE_COST] if raw is None else raw).reindex(days)[m]
    return w.div(w.sum(axis=1), axis=0).fillna(0.0)


def book_daily(ts: dict, days: pd.DatetimeIndex, col: str, raw: pd.DataFrame | None = None) -> pd.Series:
    w = book_weights(list(ts), days, raw)
    return sum(w[s] * daily_pnl(t, days, col) for s, t in ts.items())


def bench_daily(m: list[str], days: pd.DatetimeIndex) -> pd.Series:
    w = book_weights(m, days, raw_bnh)
    return sum(w[s] * bnh[s].reindex(days).fillna(0.0) for s in m)


def live(ts: dict, days: pd.DatetimeIndex, raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """The trades the book takes on `days`: those whose asset has weight above zero that day."""
    w = book_weights(list(ts), days, raw)
    return pd.concat([t[t.session.isin(days) & (w[s].reindex(t.session).to_numpy() > 0)]
                      for s, t in ts.items()])


print(f"round trip at a {TUNE_COST} / median |30-min move|, mean by year (gate: < {COST_RATIO_MAX:.0%}):")
print(pd.DataFrame(ratio[TUNE_COST]).groupby(test_days.year).mean().round(3).to_string())

MULTI = {"equities": ["ES", "NQ"], "combined": [s for s in symbols if s in ("ES", "NQ", "ZN")]}
if "YM" in symbols:                     # DTW_EXTRA=YM run: the three equity indices together
    MULTI["equities + YM"] = ["ES", "NQ", "YM"]
for name, m in MULTI.items():
    days = span_days(m, EVAL_YEARS)
    print(f"\n{name} ({' + '.join(m)}): mean strategy weight by year, by cost level")
    print(pd.concat({c: book_weights(m, days, raw_strat[c]).groupby(days.year).mean()
                     for c in COST_TICKS_PER_SIDE}, axis=1).round(2).to_string())
    bench = bench_daily(m, days)
    ts = {s: trades[s] for s in m}
    rows = {}
    for col in COST_TICKS_PER_SIDE:
        raw = raw_strat[col]
        rows[f"{name}, {col}"] = metrics(book_daily(ts, days, col, raw), bench, live(ts, days, raw), col)
    rows["risk-weighted buy and hold"] = metrics(bench, bench)
    tables[name] = pd.DataFrame(rows).T
    print(tables[name].round(3).to_string())

# %% [markdown]
# **Findings:** Equities (ES + NQ), 2,343 trades, about 58% ES / 42% NQ in every year: Sharpe
# 0.66 gross, 0.54 at a quarter tick, 0.42 at a half tick, against 0.65 for its buy-and-hold;
# return 7.4%, max drawdown −5.3% against −38.2%, beta 0.003. Combined (ES + NQ + ZN): 0.76 gross
# (ZN passes the gate at zero cost and takes 51-66% of the weight; max drawdown −2.2%), 0.40 at a
# quarter tick and 0.41 at a half tick, against 0.24 for buy-and-hold. At a quarter tick ZN's cost
# ratio is 10-17% and it fails the gate except in part of 2023, when bigger rate moves brought
# it to 10.1% on average and it took 21% of that year's weight; at a half tick it never passes,
# which is why the combined book scores slightly higher at the higher cost.

# %% [markdown]
# ## 5. Checks on the result
#
# Five checks on the scored period, each on ES, NQ, ZN, equities and the combined book: a leak test, a
# random-sign test, an always-long control, a cost curve, and robustness to the gate, the
# cut-offs and the sizing. They share one function, `book`, which rebuilds the trades from a
# panel under any rule; with the book's own rule it reproduces section 1-4's trades exactly.

# %%
RNG = np.random.default_rng(0)


def book(panel_games: pd.DataFrame, s: str, span: tuple[int, int], q: float = best_q,
         sizing: str = best_sizing, cutoffs: str = "fixed") -> pd.DataFrame:
    """One position per entry minute in `span`. Cut-offs and the dtw_disp reference come from
    FIT_YEARS ("fixed", as backtest.py) or from every earlier year ("walk-forward")."""
    g = panel_games[panel_games.symbol == s].copy()
    if cutoffs == "fixed":
        fit = g[g.year.between(*FIT_YEARS)]
        g["lo"], g["hi"], g["ref"] = fit.dtw_dir.quantile(q), fit.dtw_dir.quantile(1 - q), fit.dtw_disp.median()
    else:
        for c in ("lo", "hi", "ref"):
            g[c] = np.nan
        for y in g.year.unique():
            prior = g[(g.year >= FIT_YEARS[0]) & (g.year < y)]
            if len(prior) >= 50:
                m = g.year == y
                g.loc[m, "lo"], g.loc[m, "hi"] = prior.dtw_dir.quantile(q), prior.dtw_dir.quantile(1 - q)
                g.loc[m, "ref"] = prior.dtw_disp.median()
    g = g[g.year.between(*span) & g.ref.notna()]
    size = 1.0 if sizing == "flat" else np.clip(g.ref / g.dtw_disp, 0.0, MAX_POSITION)
    g["position"] = np.where((g.dtw_dir <= g.lo) | (g.dtw_dir >= g.hi), np.sign(g.dtw_dir) * size, 0.0)
    t = (g[g.position != 0].groupby("entry_ts")
         .agg(position=("position", "mean"), fwd_ret=("fwd_ret", "first"), year=("year", "first"),
              dtw_dir=("dtw_dir", "first"))
         .reset_index())
    t = t[t.position != 0]
    t["session"] = session_of(t.entry_ts)
    t["price"] = bars[s].close.reindex(t.entry_ts).to_numpy()
    t["tick_bps"] = ASSETS[s]["tick"] / t.price * 1e4
    t["gross"] = t.position * t.fwd_ret * 1e4
    return t


def net(t: pd.DataFrame, s: str, ticks_per_side: float, commission_rt: float = 0.0) -> pd.Series:
    """Net bps at a cost in ticks per side plus a round-trip commission in dollars."""
    per_contract = 2 * ticks_per_side * t.tick_bps + commission_rt / (t.price * ASSETS[s]["point_value"]) * 1e4
    return t.gross - t.position.abs() * per_contract


def book_sharpe(ts: dict, span: tuple[int, int], col: str = "q") -> float:
    """Sharpe of one asset (one key) or of the risk-weighted book of several."""
    return sharpe(book_daily(ts, span_days(list(ts), span), col))


base = {s: book(games, s, EVAL_YEARS) for s in symbols}
for s in symbols:
    base[s]["q"] = net(base[s], s, COST_TICKS_PER_SIDE[TUNE_COST])
    assert np.allclose(base[s].set_index("entry_ts").q, trades[s].set_index("entry_ts")[TUNE_COST]), s
BOOKS = {**{s: [s] for s in symbols}, **MULTI}
print("book() reproduces sections 1-4 at a quarter tick:",
      {name: round(book_sharpe({s: base[s] for s in m}, EVAL_YEARS), 3) for name, m in BOOKS.items()})


# %% [markdown]
# **Findings:** `book` reproduces sections 1-4 trade for trade (asserted): ES 0.635, NQ 0.110,
# ZN −0.965, equities 0.540, combined 0.396 at a quarter tick.

# %% [markdown]
# ### 5a. Leak test
#
# `features.py` rerun with the path running 5 minutes **past** entry
# (`LEAK_TEST_MINUTES=5 uv run python combined/features.py`, written to
# `features_leaked.parquet`). A path that sees part of the outcome should score far better. If
# the real and leaked books scored alike, the timing plumbing would be suspect.

# %%
from config import PANEL_LEAKED

if PANEL_LEAKED.exists():
    leaked_panel = pd.read_parquet(PANEL_LEAKED)
    leaked_games = leaked_panel[leaked_panel.admitted & leaked_panel.dtw_dir.notna()]
    leak = {s: book(leaked_games, s, EVAL_YEARS) for s in symbols}
    rows = {}
    for name, m in BOOKS.items():
        for label, src in (("real", base), ("leaked", leak)):
            ts = {s: src[s].assign(q=net(src[s], s, COST_TICKS_PER_SIDE[TUNE_COST])) for s in m}
            allt = live(ts, span_days(m, EVAL_YEARS))
            rows[(name, label)] = {"trades": len(allt),
                                   "hit rate": float((np.sign(allt.position) * allt.fwd_ret > 0).mean()),
                                   "sharpe, quarter tick": book_sharpe(ts, EVAL_YEARS)}
    print(pd.DataFrame(rows).T.round(3).to_string())
else:
    print("no leaked panel: run  LEAK_TEST_MINUTES=5 uv run python combined/features.py")


# %% [markdown]
# **Findings:** Letting the path see 5 minutes past entry lifts the hit rate from 50.0% to
# 59.9% on ES, 49.5% to 59.5% on NQ and 46.9% to 55.2% on ZN, and Sharpe to 2.7-3.6 on every
# book. The real book shows none of that, so the path ends where it should and the result is
# not a timing leak.

# %% [markdown]
# ### 5b. Random-sign test
#
# Keep every trade, its size, its timing and its cost, and draw each trade's direction at
# random. The p-value is the share of 5,000 random books whose Sharpe at a quarter tick is at
# least the real book's. It asks whether the direction adds anything beyond being in the market
# at these times with these sizes.

# %%
N_NULL = 5000


def null_sharpes(ts: dict, span: tuple[int, int], n: int) -> np.ndarray:
    days = span_days(list(ts), span)
    wts = book_weights(list(ts), days)
    parts = []
    for s, t in ts.items():
        cost = (t.gross - t.q).to_numpy()
        di = days.get_indexer(t.session)
        parts.append((di, np.abs(t.gross.to_numpy()), cost, wts[s].to_numpy()[di]))
    out = np.empty(n)
    for i in range(n):
        daily = np.zeros(len(days))
        for di, g_abs, cost, w in parts:
            sign = RNG.choice([-1.0, 1.0], size=len(g_abs))
            daily += np.bincount(di, weights=w * (sign * g_abs - cost), minlength=len(days))
        out[i] = daily.mean() / daily.std(ddof=1) * np.sqrt(252)
    return out


rows = {}
for name, m in BOOKS.items():
    ts = {s: base[s] for s in m}
    obs = book_sharpe(ts, EVAL_YEARS)
    nul = null_sharpes(ts, EVAL_YEARS, N_NULL)
    rows[name] = {"sharpe": obs, "null median": np.median(nul), "null 95th pct": np.quantile(nul, 0.95),
                  "p-value": float((nul >= obs).mean())}
print(pd.DataFrame(rows).T.round(3).to_string())

# %% [markdown]
# **Findings:** ES's direction beats random signs on the same trades: Sharpe 0.635 against a
# null median of −0.13, p = 0.041 over 5,000 draws. Equities is at the edge (p = 0.064); the
# combined book (p = 0.11), NQ (p = 0.37) and ZN (p = 0.21) are not distinguishable from
# random.

# %% [markdown]
# ### 5c. Always-long control
#
# The same trades and sizes, always long. A release window may simply tend to rise; if the
# always-long book earned as much as the real one, the edge would be exposure, not direction.

# %%
rows = {}
for name, m in BOOKS.items():
    ts = {}
    for s in m:
        t = base[s].copy()
        t["gross"] = t.position.abs() * t.fwd_ret * 1e4
        t["q"] = t.gross - (base[s].gross - base[s].q)
        ts[s] = t
    days = span_days(m, EVAL_YEARS)
    long_daily = book_daily(ts, days, "q")
    real_daily = book_daily({s: base[s] for s in m}, days, "q")
    rows[name] = {"real total %": real_daily.sum() / 100, "real sharpe": sharpe(real_daily),
                  "always-long total %": long_daily.sum() / 100, "always-long sharpe": sharpe(long_daily)}
print(pd.DataFrame(rows).T.round(3).to_string())

# %% [markdown]
# **Findings:** Always long in the same windows at the same sizes loses on every book: ES −2.4%
# (Sharpe −0.14), equities −1.8%, combined −2.0%, ZN −7.4%, against +10.9% (ES), +7.4%
# (equities) and +5.4% (combined) for the real book. On the equity books the P&L comes from the
# direction; on ZN the direction does no better than always long.

# %% [markdown]
# ### 5d. Cost curve
#
# Sharpe as the cost per side rises from zero to a full tick, and at Reza's base case (2 ticks
# round trip + $4.50 commission). Break-even is the cost per side at which the book's total
# net P&L is zero.

# %%
COST_CURVE = {"0": (0.0, 0.0), "1/8 tick": (0.125, 0.0), "1/4 tick": (0.25, 0.0),
              "1/2 tick": (0.5, 0.0), "3/4 tick": (0.75, 0.0), "1 tick": (1.0, 0.0),
              "1 tick + $4.50 rt": (1.0, 4.5)}
rows = {}
for name, m in BOOKS.items():
    row = {}
    for label, (tk, cm) in COST_CURVE.items():
        ts = {s: base[s].assign(q=net(base[s], s, tk, cm)) for s in m}
        row[label] = book_sharpe(ts, EVAL_YEARS)
    w = book_weights(m, span_days(m, EVAL_YEARS))
    tw = {s: w[s].reindex(base[s].session).to_numpy() for s in m}
    gross = sum((tw[s] * base[s].gross).sum() for s in m)
    per_tick = sum((tw[s] * base[s].position.abs() * 2 * base[s].tick_bps).sum() for s in m)
    row["break-even, ticks per side"] = gross / per_tick
    rows[name] = row
print(pd.DataFrame(rows).T.round(3).to_string())

# %% [markdown]
# **Findings:** ES breaks even at 1.36 ticks per side, NQ at 1.10 and equities at 1.35; the
# combined book at 0.95 and ZN at 0.07 (quarter-tick weights at every cost). Equities keeps 0.42
# at half a tick, 0.17 at a full tick and 0.06 at 1 tick + $4.50 round trip. A quarter-tick
# round trip on ZN is 10-17% of its median 30-minute move, against 1-2% on ES and under 1% on NQ.

# %% [markdown]
# ### 5e. Robustness — gate width, cut-offs and sizing
#
# The book's rule is one point in a small grid: gate q ∈ {0.2, 0.3, 0.4} (trade the outer
# 40%, 60%, 80%), cut-offs fixed from 2013-2020 or recomputed each year from all earlier years,
# and flat or inverse-`dtw_disp` sizing. The DTW setting is held fixed; how the result moves
# with k is in the README.

# %%
rows = []
for q in (0.2, 0.3, 0.4):
    for cut in ("fixed", "walk-forward"):
        for sizing in ("flat", "inverse_disp"):
            ts = {s: book(games, s, EVAL_YEARS, q, sizing, cut) for s in symbols}
            for s in symbols:
                ts[s]["q"] = net(ts[s], s, COST_TICKS_PER_SIDE[TUNE_COST])
            row = {"q": q, "cut-offs": cut, "sizing": sizing}
            for name, m in BOOKS.items():
                row[name] = book_sharpe({s: ts[s] for s in m}, EVAL_YEARS)
            rows.append(row)
grid = pd.DataFrame(rows).set_index(["q", "cut-offs", "sizing"])
print(f"Sharpe at a {TUNE_COST}, {EVAL_YEARS[0]}-{EVAL_YEARS[1]} (book's rule: q = {best_q}, fixed, {best_sizing}):")
print(grid.round(3).to_string())
print(f"\nshare of the 12 rules with Sharpe > 0: " + ", ".join(f"{c} {(grid[c] > 0).mean():.0%}" for c in grid))


# %% [markdown]
# **Findings:** ES and equities are positive in all 12 gate × cut-off × sizing rules (ES 0.26
# to 0.68, equities 0.09 to 0.71); the combined book in 9 of 12 (−0.03 to 0.54); NQ in 7 of 12
# (−0.19 to +0.38); ZN in none (−1.36 to −0.75). Inverse-`dtw_disp` sizing beats flat in every
# ES, ZN, equities and combined row. The widest gate (q = 0.4) scores highest on the equity
# books, but switching to it now would be choosing on the test period, so the book keeps
# q = 0.3.

# %% [markdown]
# ### 5f. Year by year — each asset, equities and the combined book
#
# The book's own rule, each test year, at a quarter tick; 2026 is a partial year.
# Return is the year's total in bps / 100 (multi-asset books in average-asset units), vol and
# Sharpe are annualised, beta is against the book's buy-and-hold. Long / short shares and win
# rates count trades; a win is a trade with positive P&L after costs.

# %%
YEARS = range(EVAL_YEARS[0], EVAL_YEARS[1] + 1)
YEARLY_COLS = {"total %": "return %", "ann vol %": "vol %", "sharpe": "sharpe", "max DD %": "max DD %",
               "beta": "beta", "trades": "trades", "win rate": "win rate", "% long": "% long",
               "% short": "% short", "win rate long": "win long", "win rate short": "win short"}
yearly = {}
for name, m in BOOKS.items():
    ts = {s: base[s] for s in m}
    rows = {}
    for y in YEARS:
        days = span_days(m, (y, y))
        rows[y] = metrics(book_daily(ts, days, "q"), bench_daily(m, days), live(ts, days), "q")
    yearly[name] = pd.DataFrame(rows).T[list(YEARLY_COLS)].rename(columns=YEARLY_COLS)
    print(f"\n{name}, by year, quarter tick:")
    print(yearly[name].round(3).to_string())

# %% [markdown]
# **Findings:** Equities gains in 5 of 6 test years (Sharpe 0.71, −0.87, 0.70, 0.78, 1.04, 1.58;
# 2022 the loss, −2.5%, max drawdown −3.8%). The combined book is the same except in 2023, when
# ZN passed the cost gate and lost (−0.23 against equities' 0.70). ES is positive in 5 of 6
# years, NQ in 3, ZN in none. Win rates are 45-56% in every book-year, so the equity edge comes
# from winners being larger, not more frequent. Equities goes long 47-55% of the time and wins
# more often long than short in every year except 2022 and 2025; ZN leans short (50-60% of
# trades), and its shorts win more often in 4 of 6 years. Beta is under 0.06 in every book-year.

# %% [markdown]
# ### 5g. Information ratio — against the market, and as IC × breadth
#
# Three ways to put an IR on the book:
#
# * **IR vs buy-and-hold**: mean / sd of (book − buy-and-hold). Shown for completeness; the book
#   is in the market about 2% of the time with no beta, so this mostly measures being short
#   the market and is the wrong benchmark.
# * **Appraisal ratio**: regress the book's daily P&L on buy-and-hold; alpha / residual vol,
#   annualised. The market-neutral IR, with the alpha's t-stat.
# * **Fundamental law**: IR ≈ IC × √(bets per year), with IC the rank correlation of the trade
#   direction with the next 30-minute return on traded games. How much of the Sharpe the
#   per-trade edge and the number of bets explain.

# %%
from scipy import stats

span_years = len(in_years(bnh[symbols[0]].index, EVAL_YEARS)) / 252
rows = {}
for name, m in BOOKS.items():
    days = span_days(m, EVAL_YEARS)
    daily = book_daily({s: base[s] for s in m}, days, "q")
    bench = bench_daily(m, days)
    X = np.c_[np.ones(len(days)), bench.to_numpy()]
    coef, *_ = np.linalg.lstsq(X, daily.to_numpy(), rcond=None)
    resid_sd = (daily.to_numpy() - X @ coef).std(ddof=2)
    active = daily - bench
    allt = live({s: base[s] for s in m}, days)
    ic_sign = stats.spearmanr(np.sign(allt.position), allt.fwd_ret).correlation
    bets = len(allt) / span_years
    rows[name] = {"sharpe": sharpe(daily), "beta": coef[1],
                  "appraisal ratio": coef[0] / resid_sd * np.sqrt(252),
                  "alpha t": coef[0] / (resid_sd / np.sqrt(len(days))),
                  "IR vs buy-and-hold": active.mean() / active.std(ddof=1) * np.sqrt(252),
                  "IC, dtw_dir": stats.spearmanr(allt.dtw_dir, allt.fwd_ret).correlation,
                  "IC, direction": ic_sign, "bets / year": bets,
                  "IC x sqrt(bets)": ic_sign * np.sqrt(bets)}
print(pd.DataFrame(rows).T.round(3).to_string())

# %% [markdown]
# **Findings:** ES's appraisal ratio is 0.61 (beta 0.007, alpha t 1.41), equities' 0.53 (alpha
# t 1.22) and the combined book's 0.39 (alpha t 0.92): the market-neutral IR is about the
# Sharpe. The IR against buy-and-hold is about −0.57 for ES, NQ and equities, which measures
# being out of a rising market, not skill. By the fundamental law, ES's direction IC of 0.027
# over 225 bets a year implies an IR of about 0.41; the realised 0.61 is above that, partly from
# sizing and partly luck. NQ's direction IC is −0.002: no edge. ZN's is +0.016 on 364 bets a
# year, but its appraisal ratio is −0.97 (alpha t −2.25): the edge is below its cost.

# %%
pd.concat(tables, names=["book", "row"]).to_csv(OUT / "backtest_summary.csv")
print(f"wrote backtest_summary.csv  (scored {EVAL_YEARS[0]}-{EVAL_YEARS[1]})")
