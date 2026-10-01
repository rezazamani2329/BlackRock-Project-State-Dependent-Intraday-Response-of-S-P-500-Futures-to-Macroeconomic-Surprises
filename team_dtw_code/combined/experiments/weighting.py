"""Dynamic weighting schemes for the equities (ES + NQ) and combined (ES + NQ + ZN) books.

Every scheme is walk-forward: a day's weights use only data from before that day. The daily
volatility inputs update daily; the P&L-based inputs (trailing Sharpe, strategy vol, typical
move) refresh at each month start. Weights are normalised to sum to one within the book; a day
on which every asset has zero weight is flat.

    equal            1 / N
    inv_vol          1 / underlying vol (60 sessions)
    inv_vol_gate     1 / underlying vol, if the asset's trailing 3-year net Sharpe > 0
    inv_vol_gate_exp 1 / underlying vol, if its net Sharpe since 2013 > 0
    inv_move_gate    1 / median |30-min move| on admitted games over the past year, gated (3y)
                     (the old static equal-risk weight, made dynamic)
    inv_pnl_vol_gate 1 / sd of the asset's own daily strategy P&L over the past year, gated (3y)
    sharpe_3y        max(trailing 3-year net Sharpe, 0) / underlying vol  (backtest.py now)
    sharpe_exp       max(net Sharpe since 2013, 0) / underlying vol
    mean_var_3y      max(mean, 0) / variance of the asset's daily strategy P&L over 3 years
    inv_vol_cost     1 / underlying vol, if a round trip costs under COST_RATIO_MAX of the
                     asset's median |30-min move| over the past year (a structural cost gate:
                     no P&L estimate, and it opens when the market's moves get bigger)
    inv_move_cost    1 / median |30-min move| over the past year, with the same cost gate

Costs are a quarter tick per side; "net" means after that cost. The trades are backtest.py's:
the book's rule, with cut-offs fitted on 2013-2020, so the pre-2021 P&L these schemes are scored
on uses in-sample cut-offs. That affects every scheme alike.

Selection uses 2016-2020 only (2013-2015 is the history the trailing windows need): the mean
of the equities and combined books' Sharpe over 2016-2020, among schemes whose combined book
can trade (some asset has weight) on at least MIN_INVESTED of days. The book only ever holds
positions in release windows; a gate that zeroes every weight skips those trades, which avoids
losses but is not a book to run, and its Sharpe rests on a handful of days. The test years are read once, by
`--report`, for the chosen scheme and backtest.py's current one.

Run: uv run python combined/experiments/weighting.py            # score on 2016-2020, choose
     uv run python combined/experiments/weighting.py --report   # then 2021-2026 for the choice

Findings (2026-09-29): every gate built on trailing P&L (3-year or expanding Sharpe, mean /
variance) zeroes every weight, so the book skips its event trades, on 59-85% of 2016-2020 days, because the direction lost on every
asset in 2013-2017; they are not runnable books. Among invested schemes, inv_vol_cost scores
best on 2016-2020 (0.37 on both books; equal weights 0.36 / 0.18, ungated inverse vol 0.37 /
−0.47 because it gives ZN 63%). The cost ratio ran 4-7% on ES, 1-3% on NQ and 13-21% on ZN in
every year 2014-2020, so any cut-off from about 8% to 13% makes the same choice. On 2021-2026,
read once: inv_vol_cost equities 0.53, combined 0.40 (ZN's ratio fell under 10% for part of
2023 and it lost there: combined −0.23 against equities 0.69 that year), against 0.34 / 0.34 for
sharpe_3y, backtest.py's current scheme.
"""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BACKTEST = HERE.parent / "backtest.py"

# backtest.py's data, rule and trades, without its books or checks
src = BACKTEST.read_text()
src = src[:src.index("# ## 4. Equities")]
ns = {"__file__": str(BACKTEST), "__name__": "__main__"}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(src, str(BACKTEST), "exec"), ns)
bnh, rules, games, symbols = ns["bnh"], ns["rules"], ns["games"], ns["symbols"]
trades_for, daily_pnl, sharpe, in_years = ns["trades_for"], ns["daily_pnl"], ns["sharpe"], ns["in_years"]
COST = ns["TUNE_COST"]

HIST = (2013, 2026)
SELECT = (2016, 2020)
TEST = (2021, 2026)
BOOKS = {"equities": ["ES", "NQ"], "combined": ["ES", "NQ", "ZN"]}
VOL_SESSIONS = 60
EDGE_YEARS = 3
CURRENT = "sharpe_3y"
COST_RATIO_MAX = 0.10          # ES ~3%, NQ <1%, ZN ~19% of the median move
MIN_INVESTED = 0.75
bars = ns["bars"]
ASSETS, TICKS = ns["ASSETS"], ns["COST_TICKS_PER_SIDE"][COST]

days = in_years(bnh[symbols[0]].index, HIST)
for s in symbols[1:]:
    days = days.union(in_years(bnh[s].index, HIST))
pnl = pd.DataFrame({s: daily_pnl(trades_for(s, HIST, rules[s]), in_years(bnh[s].index, HIST), COST)
                    for s in symbols}).reindex(days)          # NaN where the asset has no session
vol = pd.DataFrame({s: bnh[s].rolling(VOL_SESSIONS).std().shift(1) for s in symbols}).reindex(days).ffill()

# month-start inputs from strictly earlier data
months = days.to_period("M")
rows = []
for p in months.unique():
    start = p.start_time
    row = {"month": p}
    for s in symbols:
        d = pnl[s].dropna()
        d3 = d[(d.index >= start - pd.DateOffset(years=EDGE_YEARS)) & (d.index < start)]
        dx = d[d.index < start]
        d1 = d[(d.index >= start - pd.DateOffset(years=1)) & (d.index < start)]
        g = games[(games.symbol == s) & (games.entry_ts < pd.Timestamp(start, tz="UTC"))
                  & (games.entry_ts >= pd.Timestamp(start - pd.DateOffset(years=1), tz="UTC"))]
        row |= {(s, "sr3"): sharpe(d3) if len(d3) > 60 else np.nan,
                (s, "srx"): sharpe(dx) if len(dx) > 60 else np.nan,
                (s, "pnl_vol1"): d1.std() if len(d1) > 60 else np.nan,
                (s, "mv3"): d3.mean() / d3.var() if len(d3) > 60 and d3.var() > 0 else np.nan,
                (s, "move1"): g.fwd_ret.abs().median() * 1e4 if len(g) >= 20 else np.nan}
        i = bars[s].index.searchsorted(pd.Timestamp(start, tz="UTC")) - 1
        rt_bps = 2 * TICKS * ASSETS[s]["tick"] / bars[s].close.iloc[i] * 1e4 if i >= 0 else np.nan
        row[(s, "cost_ratio")] = rt_bps / row[(s, "move1")]
    rows.append(row)
monthly = pd.DataFrame(rows).set_index("month")
monthly.columns = pd.MultiIndex.from_tuples(monthly.columns)
m_in = monthly.reindex(months)
m_in.index = days


def inp(s: str, key: str) -> pd.Series:
    return m_in[(s, key)]


def gate(s: str, key: str = "sr3") -> pd.Series:
    return (inp(s, key) > 0).astype(float)


SCHEMES = {
    "equal": lambda s: pd.Series(1.0, index=days),
    "inv_vol": lambda s: 1 / vol[s],
    "inv_vol_gate": lambda s: gate(s) / vol[s],
    "inv_vol_gate_exp": lambda s: gate(s, "srx") / vol[s],
    "inv_move_gate": lambda s: gate(s) / inp(s, "move1"),
    "inv_pnl_vol_gate": lambda s: gate(s) / inp(s, "pnl_vol1"),
    "sharpe_3y": lambda s: inp(s, "sr3").clip(lower=0) / vol[s],
    "sharpe_exp": lambda s: inp(s, "srx").clip(lower=0) / vol[s],
    "mean_var_3y": lambda s: inp(s, "mv3").clip(lower=0),
    "inv_vol_cost": lambda s: (inp(s, "cost_ratio") < COST_RATIO_MAX) / vol[s],
    "inv_move_cost": lambda s: (inp(s, "cost_ratio") < COST_RATIO_MAX) / inp(s, "move1"),
}


def weights(scheme: str, m: list[str]) -> pd.DataFrame:
    raw = pd.DataFrame({s: SCHEMES[scheme](s) for s in m}).fillna(0.0)
    return raw.div(raw.sum(axis=1), axis=0).fillna(0.0)


def book(scheme: str, m: list[str]) -> pd.Series:
    return (weights(scheme, m) * pnl[m].fillna(0.0)).sum(axis=1)


def span(x: pd.Series | pd.DataFrame, y: tuple[int, int]):
    return x[(x.index.year >= y[0]) & (x.index.year <= y[1])]


def score(years: tuple[int, int], by_year: bool = False) -> pd.DataFrame:
    out = {}
    for sc in SCHEMES:
        row = {}
        for name, m in BOOKS.items():
            b = book(sc, m)
            row[(name, f"{years[0]}-{years[1]}")] = sharpe(span(b, years))
            if by_year:
                for y in range(years[0], years[1] + 1):
                    row[(name, y)] = sharpe(span(b, (y, y)))
        w = span(weights(sc, BOOKS["combined"]), years)
        row[("combined", "% days tradable")] = float((w.sum(axis=1) > 0).mean())
        out[sc] = row
    return pd.DataFrame(out).T


def mean_weights(scheme: str, years: tuple[int, int]) -> pd.Series:
    w = span(weights(scheme, BOOKS["combined"]), years)
    return w.mean().round(2)


pd.set_option("display.width", 200)
sel = score(SELECT, by_year=True)
sel["select"] = sel[[("equities", "2016-2020"), ("combined", "2016-2020")]].mean(axis=1)
sel = sel.sort_values("select", ascending=False)
eligible = sel[sel[("combined", "% days tradable")] >= MIN_INVESTED]
print(f"Sharpe at a {COST} per side, {SELECT[0]}-{SELECT[1]} (selection years only):")
print(sel.round(2).to_string())
print(f"\nmean combined-book weight, {SELECT[0]}-{SELECT[1]}:")
print(pd.DataFrame({sc: mean_weights(sc, SELECT) for sc in SCHEMES}).T.loc[sel.index].to_string())
chosen = eligible.index[0]
ratio = m_in.xs("cost_ratio", axis=1, level=1)
print("\nround-trip cost / median |30-min move|, by year (month-start values, mean):")
print(span(ratio, (2014, SELECT[1])).groupby(span(ratio, (2014, SELECT[1])).index.year).mean().round(3).to_string())
print(f"\nchosen on {SELECT[0]}-{SELECT[1]} (invested >= {MIN_INVESTED:.0%}): {chosen}")

if "--report" in sys.argv:
    test = score(TEST, by_year=True).loc[[chosen, CURRENT]]
    print(f"\nSharpe at a {COST} per side, test {TEST[0]}-{TEST[1]}, chosen and current only:")
    print(test.round(2).to_string())
    print(f"\nmean combined-book weight, {TEST[0]}-{TEST[1]}:")
    print(pd.DataFrame({sc: mean_weights(sc, TEST) for sc in (chosen, CURRENT)}).T.to_string())
