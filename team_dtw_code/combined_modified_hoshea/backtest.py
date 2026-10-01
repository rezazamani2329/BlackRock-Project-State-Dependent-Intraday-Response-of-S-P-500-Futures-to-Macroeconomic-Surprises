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
# # Backtest — 读本目录 `output/features.parquet`
#
# 从 `combined/backtest.py` 拷过来，只改 import 路径。先跑完
# `hyperparam_search.py` 和 `build_panel.py`，再跑这个。

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent if "__file__" in globals() else "."))
from config import (ASSETS, CLEAN_BARS_FMT, COST_TICKS_PER_SIDE, GATE_GRID, MAX_POSITION, NY,
                    OUT, PANEL, SESSION_SHIFT, SIZING_GRID, TEST_YEARS_SPAN, TRAIN_YEARS_SPAN)

pd.set_option("display.width", 200)

EVAL_YEARS = TEST_YEARS_SPAN
FIT_YEARS = (TRAIN_YEARS_SPAN[0], EVAL_YEARS[0] - 1)
TUNE_COST = "quarter tick"
symbols = list(ASSETS)
print(f"scored: {EVAL_YEARS[0]}-{EVAL_YEARS[1]}   gate/sizing tuned on: "
      f"{TRAIN_YEARS_SPAN[0]}-{TRAIN_YEARS_SPAN[1]}   constants from: {FIT_YEARS[0]}-{FIT_YEARS[1]}")


def session_of(ts: pd.Series) -> pd.Series:
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
    b = bars[s]
    r = np.log(b.close).diff() * 1e4
    r[b.contract.ne(b.contract.shift())] = 0.0
    return r.fillna(0.0).groupby(pd.DatetimeIndex(b.session_date)).sum()


bnh = {s: buy_and_hold(s) for s in symbols}

# %% [markdown]
# ## The trading rule
#
# * **Direction.** The sign of `dtw_dir`.
# * **Gate.** Trade only if `dtw_dir` is outside `[q, 1−q]` of the fitting years.
# * **Size.** 1 contract (`flat`) or `median(dtw_disp) / dtw_disp` capped at `MAX_POSITION`.

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
# ## 0. Gate and sizing — training years only

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
print(f"Sharpe at a {TUNE_COST} per side, train {TRAIN_YEARS_SPAN[0]}-{TRAIN_YEARS_SPAN[1]}:")
print(grid.round(3).to_string())
best_q, best_sizing = grid.pooled.idxmax()
print(f"\nchosen: q = {best_q} (trade the outer {2 * best_q:.0%} of games), sizing = {best_sizing}")

rules = {s: make_rule(games[(games.symbol == s) & games.year.between(*FIT_YEARS)], best_q, best_sizing)
         for s in symbols}
print(pd.DataFrame(rules).T.to_string())

# %% [markdown]
# **Findings:** 等跑完 chosen panel 后再填。

# %%
trades = {s: trades_for(s, EVAL_YEARS, rules[s]) for s in symbols}
for s in symbols:
    print(f"{s}: {len(trades[s]):,} trades in {EVAL_YEARS[0]}-{EVAL_YEARS[1]}")


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
    if t is None:
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
# ## 1–3. 分品种

# %%
for s in symbols:
    tables[s] = asset_table(s)
    print(f"\n{s}")
    print(tables[s].round(3).to_string())

# %% [markdown]
# **Findings:** 等跑完后再填各品种数字。

# %% [markdown]
# ## 4. Combined book

# %%
fit_days = {s: in_years(bnh[s].index, FIT_YEARS) for s in symbols}
w_strat = pd.Series({s: 1 / (games[(games.symbol == s) & games.year.between(*FIT_YEARS)]
                             .fwd_ret.abs().median() * 1e4) for s in symbols})
w_bnh = pd.Series({s: 1 / bnh[s].reindex(fit_days[s]).std() for s in symbols})
w_strat, w_bnh = w_strat / w_strat.sum(), w_bnh / w_bnh.sum()
print(pd.DataFrame({"strategy weight": w_strat, "buy-and-hold weight": w_bnh}).round(3).to_string())

days = eval_days(symbols[0])
for s in symbols[1:]:
    days = days.union(eval_days(s))
bench = sum(w_bnh[s] * bnh[s].reindex(days).fillna(0.0) for s in symbols)
all_trades = pd.concat(trades.values())
rows = {}
for col in COST_TICKS_PER_SIDE:
    daily = sum(w_strat[s] * daily_pnl(trades[s], days, col) for s in symbols)
    rows[f"combined, {col}"] = metrics(daily, bench, all_trades, col)
rows["equal-risk buy and hold"] = metrics(bench, bench)
tables["combined"] = pd.DataFrame(rows).T
print(tables["combined"].round(3).to_string())

# %% [markdown]
# **Findings:** 等跑完后再填 combined 数字。

# %%
pd.concat(tables, names=["book", "row"]).to_csv(OUT / "backtest_summary.csv")
print(f"wrote backtest_summary.csv  (scored {EVAL_YEARS[0]}-{EVAL_YEARS[1]})")
