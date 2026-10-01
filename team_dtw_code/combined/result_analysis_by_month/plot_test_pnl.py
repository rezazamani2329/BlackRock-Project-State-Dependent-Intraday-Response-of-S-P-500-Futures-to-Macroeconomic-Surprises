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
# # Test-period P&L — strategy vs always-long, by month
#
# Visual check of `combined/backtest.py` on the **test years (2021–2026)**, one asset at a
# time. The trading rule is copied from that notebook (not re-tuned here):
#
# - sign of `dtw_dir`, outer 60% gate (`q = 0.3`), inverse-`dtw_disp` size, cap 2
# - cut-offs and `disp_ref` from 2013–2020, per asset
# - net of a **quarter tick** per side
#
# **Always-long** keeps the same entry minutes and sizes, but the position is always
# `+|size|`. If that curve tracks the strategy, the edge is being in the release window,
# not the DTW direction.
#
# Units match `backtest.py`: daily P&L in bps of one contract, summed (not compounded),
# zeros on sessions with no trade. Cumulative plots are that sum ÷ 100 (percent of notional).

# %%
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

COMBINED = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path("combined").resolve()
HERE = Path(__file__).resolve().parent if "__file__" in globals() else COMBINED / "result_analysis_by_month"
sys.path.insert(0, str(COMBINED))

from config import (ASSETS, BOOK_SYMBOLS, CLEAN_BARS_FMT, COST_TICKS_PER_SIDE, MAX_POSITION,
                    NY, PANEL, SESSION_SHIFT, TEST_YEARS_SPAN, TRAIN_YEARS_SPAN)

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight", "savefig.facecolor": "white"})

FIG = HERE / "figures"
OUT = HERE / "output"
FIG.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

EVAL_YEARS = TEST_YEARS_SPAN
FIT_YEARS = (TRAIN_YEARS_SPAN[0], EVAL_YEARS[0] - 1)
COST = "quarter tick"
Q, SIZING = 0.3, "inverse_disp"
SYMBOLS = list(BOOK_SYMBOLS)
COLORS = {"strategy": "#1f4e79", "always-long": "#b0b0b0"}


def session_of(ts: pd.Series) -> pd.Series:
    return (ts.dt.tz_convert(NY) + SESSION_SHIFT).dt.normalize().dt.tz_localize(None)


def in_years(days: pd.DatetimeIndex, span: tuple[int, int]) -> pd.DatetimeIndex:
    return days[(days.year >= span[0]) & (days.year <= span[1])]


def make_rule(fit: pd.DataFrame) -> dict:
    return {"tail_lo": fit.dtw_dir.quantile(Q), "tail_hi": fit.dtw_dir.quantile(1 - Q),
            "disp_ref": fit.dtw_disp.median()}


def trades_for(games: pd.DataFrame, bars: dict, s: str, span: tuple[int, int], rule: dict) -> pd.DataFrame:
    g = games[(games.symbol == s) & games.year.between(*span)].copy()
    in_tail = (g.dtw_dir <= rule["tail_lo"]) | (g.dtw_dir >= rule["tail_hi"])
    size = np.clip(rule["disp_ref"] / g.dtw_disp, 0.0, MAX_POSITION)
    g["position"] = np.where(in_tail, np.sign(g.dtw_dir) * size, 0.0)
    t = (g[g.position != 0].groupby("entry_ts")
         .agg(position=("position", "mean"), fwd_ret=("fwd_ret", "first")).reset_index())
    t = t[t.position != 0]
    t["session"] = session_of(t.entry_ts)
    tick_bps = ASSETS[s]["tick"] / bars[s].close.reindex(t.entry_ts).to_numpy() * 1e4
    t["tick_bps"] = tick_bps
    t["gross"] = t.position * t.fwd_ret * 1e4
    t["net"] = t.gross - t.position.abs() * 2 * COST_TICKS_PER_SIDE[COST] * tick_bps
    t["long_gross"] = t.position.abs() * t.fwd_ret * 1e4
    t["long_net"] = t.long_gross - t.position.abs() * 2 * COST_TICKS_PER_SIDE[COST] * tick_bps
    return t


def daily_pnl(t: pd.DataFrame, days: pd.DatetimeIndex, col: str) -> pd.Series:
    return t.groupby("session")[col].sum().reindex(days).fillna(0.0)


def sharpe(daily: pd.Series) -> float:
    sd = daily.std(ddof=1)
    return float(daily.mean() / sd * np.sqrt(252)) if sd else np.nan


# %%
panel = pd.read_parquet(PANEL)
games = panel[panel.admitted & panel.dtw_dir.notna()].copy()
bars = {}
for s in SYMBOLS:
    b = pd.read_parquet(CLEAN_BARS_FMT.format(symbol=s), columns=["close", "contract", "session_date"])
    b.index = pd.to_datetime(b.index, utc=True)
    bars[s] = b.sort_index()

session_days = {}
for s in SYMBOLS:
    session_days[s] = pd.DatetimeIndex(pd.to_datetime(bars[s].session_date).unique()).sort_values()
    session_days[s] = in_years(session_days[s], EVAL_YEARS)

trades, daily = {}, {}
for s in SYMBOLS:
    rule = make_rule(games[(games.symbol == s) & games.year.between(*FIT_YEARS)])
    t = trades_for(games, bars, s, EVAL_YEARS, rule)
    trades[s] = t
    days = session_days[s]
    daily[s] = pd.DataFrame({
        "strategy": daily_pnl(t, days, "net"),
        "always-long": daily_pnl(t, days, "long_net"),
    })
    print(f"{s}: {len(t):,} trades  {EVAL_YEARS[0]}-{EVAL_YEARS[1]}  "
          f"rule lo/hi/ref = {rule['tail_lo']:.5f} / {rule['tail_hi']:.5f} / {rule['disp_ref']:.5f}")


# %% [markdown]
# ## 1. Cumulative P&L on the test years
#
# Strategy (navy) vs always-long in the same windows (grey). One panel per asset.

# %%
fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6), sharex=False)
for ax, s in zip(axes, SYMBOLS):
    d = daily[s]
    ax.plot(d.index, d["strategy"].cumsum() / 100, color=COLORS["strategy"], lw=1.8, label="DTW strategy")
    ax.plot(d.index, d["always-long"].cumsum() / 100, color=COLORS["always-long"], lw=1.5,
            ls="--", label="Always long, same windows")
    ax.axhline(0, color="0.4", lw=0.6)
    ax.set_title(s)
    ax.set_ylabel("cumulative % of notional")
    ax.legend(frameon=False, fontsize=10)
fig.suptitle(f"Test {EVAL_YEARS[0]}–{EVAL_YEARS[1]}, net of {COST} per side", y=1.02, fontsize=14)
fig.tight_layout()
fig.savefig(FIG / "pnl_curves_test.png", dpi=150)
plt.show()

summary = []
for s in SYMBOLS:
    d = daily[s]
    eq = d["strategy"].cumsum()
    eq_l = d["always-long"].cumsum()
    summary.append({
        "asset": s, "trades": len(trades[s]),
        "strategy total %": d["strategy"].sum() / 100,
        "strategy Sharpe": sharpe(d["strategy"]),
        "strategy max DD %": float((eq - eq.cummax()).min()) / 100,
        "always-long total %": d["always-long"].sum() / 100,
        "always-long Sharpe": sharpe(d["always-long"]),
        "always-long max DD %": float((eq_l - eq_l.cummax()).min()) / 100,
    })
summary = pd.DataFrame(summary).set_index("asset")
print(summary.round(3).to_string())
summary.to_csv(OUT / "test_summary.csv")


# %% [markdown]
# ## 2. Monthly returns
#
# Each bar is the sum of that month's session P&L, in percent of notional. The heatmap is
# year × calendar month; the grouped bars put the strategy next to always-long.

# %%
def monthly_pct(daily_s: pd.Series) -> pd.Series:
    return daily_s.groupby(daily_s.index.to_period("M")).sum() / 100


rows = []
for s in SYMBOLS:
    for kind in ("strategy", "always-long"):
        m = monthly_pct(daily[s][kind])
        for per, val in m.items():
            rows.append({"asset": s, "book": kind, "month": per.to_timestamp(),
                         "year": per.year, "calendar_month": per.month, "return_%": val})
monthly = pd.DataFrame(rows)
monthly.to_csv(OUT / "monthly_returns.csv", index=False)
print(f"wrote {len(monthly)} month-rows -> {OUT / 'monthly_returns.csv'}")

# %%
fig, axes = plt.subplots(2, 3, figsize=(16.5, 7.2))
months = list(range(1, 13))
month_labs = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
vmax = monthly["return_%"].abs().quantile(0.98)
for row, kind in enumerate(("strategy", "always-long")):
    for col, s in enumerate(SYMBOLS):
        ax = axes[row, col]
        sub = monthly[(monthly.asset == s) & (monthly.book == kind)]
        mat = (sub.pivot(index="year", columns="calendar_month", values="return_%")
               .reindex(columns=months))
        sns.heatmap(mat, ax=ax, cmap="RdBu", center=0, vmin=-vmax, vmax=vmax,
                    cbar=col == 2, xticklabels=month_labs, yticklabels=True)
        ax.set_title(f"{s} · {kind}")
        ax.set_xlabel("")
        ax.set_ylabel("")
fig.suptitle("Monthly return (% of notional)", y=1.01, fontsize=14)
fig.tight_layout()
fig.savefig(FIG / "monthly_returns_heatmap.png", dpi=150)
plt.show()

# %%
fig, axes = plt.subplots(3, 1, figsize=(16.5, 10.5), sharex=True)
width = pd.Timedelta(days=8)
for ax, s in zip(axes, SYMBOLS):
    sub = monthly[monthly.asset == s].copy()
    strat = sub[sub.book == "strategy"].set_index("month")["return_%"]
    long = sub[sub.book == "always-long"].set_index("month")["return_%"]
    idx = strat.index.union(long.index).sort_values()
    strat, long = strat.reindex(idx).fillna(0.0), long.reindex(idx).fillna(0.0)
    ax.bar(idx - width / 2, strat.to_numpy(), width=width, color=COLORS["strategy"],
           label="DTW strategy", alpha=0.9)
    ax.bar(idx + width / 2, long.to_numpy(), width=width, color=COLORS["always-long"],
           label="Always long, same windows", alpha=0.9)
    ax.axhline(0, color="0.4", lw=0.6)
    ax.set_ylabel(f"{s}\nmonthly %")
    ax.legend(frameon=False, loc="upper left", fontsize=10)
axes[-1].set_xlabel("")
fig.suptitle(f"Monthly P&L, test {EVAL_YEARS[0]}–{EVAL_YEARS[1]}, {COST} per side", y=1.01, fontsize=14)
fig.tight_layout()
fig.savefig(FIG / "monthly_returns_bars.png", dpi=150)
plt.show()


# %% [markdown]
# ## 3. Month-level summary

# %%
def month_stats(s: str, kind: str) -> dict:
    r = monthly[(monthly.asset == s) & (monthly.book == kind)]["return_%"]
    return {"n months": int(r.notna().sum()), "hit rate": float((r > 0).mean()),
            "mean %": float(r.mean()), "median %": float(r.median()),
            "best %": float(r.max()), "worst %": float(r.min()),
            "best month": str(monthly.loc[r.idxmax(), "month"].date())[:7],
            "worst month": str(monthly.loc[r.idxmin(), "month"].date())[:7]}


stats = pd.concat({(s, k): pd.Series(month_stats(s, k))
                   for s in SYMBOLS for k in ("strategy", "always-long")}, axis=1).T
print(stats.to_string())
stats.to_csv(OUT / "monthly_stats.csv")

by_year = (monthly.groupby(["asset", "book", "year"])["return_%"].sum()
           .unstack("year").reindex(pd.MultiIndex.from_product([SYMBOLS, ["strategy", "always-long"]])))
print("\ncalendar-year sum of monthly %:")
print(by_year.round(2).to_string())
by_year.to_csv(OUT / "yearly_from_months.csv")


# %% [markdown]
# **Findings:** Test 2021–2026, quarter tick, 66 months. ES 1,187 trades: strategy **+8.0%,
# Sharpe 0.43**, always-long **−5.5%, −0.31**. NQ 1,082 trades: **+11.7%, 0.53** vs
# always-long **−12.8%, −0.57**. ZN 1,896 trades: **−6.7%, −0.86**, almost the same as
# always-long (−7.5%, −0.90). Monthly hit rate is 53% on both equity books and 38% on ZN.
# ES’s best / worst months are 2025-02 (+2.5%) and 2023-01 (−2.5%); NQ’s are 2025-08
# (+3.0%) and 2022-05 (−4.7%). By year the equity edge is concentrated in 2021 and 2025
# (ES +4.0 / +6.5, NQ +3.3 / +11.0); 2022 is the loss year on both. ZN is negative every
# calendar year. On ES and NQ the direction, not the window, is what makes the P&L; on ZN
# it is not.
