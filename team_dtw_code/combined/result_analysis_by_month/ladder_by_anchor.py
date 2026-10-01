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
# # Ladder split by anchor — P&L, similarity, direction
#
# Same test book as `plot_test_pnl.py` (2021–2026, outer 60% gate from 2013–2020,
# inverse-`dtw_disp`, quarter tick). Trades are then split by ladder rung
# `T+{1,31,61,91,121}`. The gate is **not** re-fit per rung, so these are the
# book's own trades on each step, not five separately tuned books.
#
# 1. Cumulative net P&L vs always-long in the same windows.
# 2. Monthly mean book-weighted neighbour similarity, plus the test-period mean.
# 3. Direction hit rate over the whole test: \(\mathrm{sign}(\texttt{dtw\_dir})\)
#    vs the 30-minute return.

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

from config import (ANCHORS, ASSETS, BOOK_SYMBOLS, CLEAN_BARS_FMT, COST_TICKS_PER_SIDE,
                    MAX_POSITION, NY, PANEL, SESSION_SHIFT, TEST_YEARS_SPAN, TRAIN_YEARS_SPAN)

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight", "savefig.facecolor": "white"})

FIG = HERE / "figures"
OUT = HERE / "output"
FIG.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

EVAL_YEARS = TEST_YEARS_SPAN
FIT_YEARS = (TRAIN_YEARS_SPAN[0], EVAL_YEARS[0] - 1)
COST = "quarter tick"
Q = 0.3
SYMBOLS = list(BOOK_SYMBOLS)
RUNGS = list(ANCHORS)
COLORS = {"strategy": "#1f4e79", "always-long": "#b0b0b0"}


def session_of(ts: pd.Series) -> pd.Series:
    return (ts.dt.tz_convert(NY) + SESSION_SHIFT).dt.normalize().dt.tz_localize(None)


def in_years(days: pd.DatetimeIndex, span: tuple[int, int]) -> pd.DatetimeIndex:
    return days[(days.year >= span[0]) & (days.year <= span[1])]


def sharpe(daily: pd.Series) -> float:
    sd = daily.std(ddof=1)
    return float(daily.mean() / sd * np.sqrt(252)) if sd else np.nan


def daily_pnl(t: pd.DataFrame, days: pd.DatetimeIndex, col: str) -> pd.Series:
    return t.groupby("session")[col].sum().reindex(days).fillna(0.0)


def trades_for(games: pd.DataFrame, bars: dict, s: str) -> pd.DataFrame:
    g = games[(games.symbol == s)].copy()
    fit = g[g.year.between(*FIT_YEARS)]
    lo, hi, ref = fit.dtw_dir.quantile(Q), fit.dtw_dir.quantile(1 - Q), fit.dtw_disp.median()
    cur = g[g.year.between(*EVAL_YEARS)].copy()
    in_tail = (cur.dtw_dir <= lo) | (cur.dtw_dir >= hi)
    size = np.clip(ref / cur.dtw_disp, 0.0, MAX_POSITION)
    cur["position"] = np.where(in_tail, np.sign(cur.dtw_dir) * size, 0.0)
    t = (cur[cur.position != 0]
         .groupby("entry_ts")
         .agg(position=("position", "mean"), fwd_ret=("fwd_ret", "first"),
              dtw_dir=("dtw_dir", "first"), anchor=("anchor", "first"))
         .reset_index())
    t = t[t.position != 0]
    t["session"] = session_of(t.entry_ts)
    tick_bps = ASSETS[s]["tick"] / bars[s].close.reindex(t.entry_ts).to_numpy() * 1e4
    t["net"] = t.position * t.fwd_ret * 1e4 - t.position.abs() * 2 * COST_TICKS_PER_SIDE[COST] * tick_bps
    t["long_net"] = (t.position.abs() * t.fwd_ret * 1e4
                     - t.position.abs() * 2 * COST_TICKS_PER_SIDE[COST] * tick_bps)
    t["symbol"] = s
    t["correct"] = np.sign(t.dtw_dir) * t.fwd_ret > 0
    return t


# %%
panel = pd.read_parquet(PANEL)
panel["entry_ts"] = pd.to_datetime(panel["entry_ts"], utc=True)
games = panel[panel.admitted & panel.dtw_dir.notna()].copy()

bars, days = {}, {}
for s in SYMBOLS:
    b = pd.read_parquet(CLEAN_BARS_FMT.format(symbol=s), columns=["close", "contract", "session_date"])
    b.index = pd.to_datetime(b.index, utc=True)
    bars[s] = b.sort_index()
    sess = pd.DatetimeIndex(pd.to_datetime(b.session_date).unique()).sort_values()
    days[s] = in_years(sess, EVAL_YEARS)

trades = pd.concat([trades_for(games, bars, s) for s in SYMBOLS], ignore_index=True)
print(trades.groupby(["symbol", "anchor"]).size().unstack("anchor").reindex(SYMBOLS)[RUNGS].to_string())

sim_path = OUT / "trade_weighted_similarity.csv"
sim = pd.read_csv(sim_path, parse_dates=["entry_ts", "session"])
sim["entry_ts"] = pd.to_datetime(sim["entry_ts"], utc=True)
trades = trades.merge(sim[["symbol", "entry_ts", "weighted_similarity", "weighted_distance"]],
                      on=["symbol", "entry_ts"], how="left")
print(f"similarity coverage: {trades.weighted_similarity.notna().mean():.1%}")


# %% [markdown]
# ## 1. Cumulative P&L vs always-long, by rung

# %%
fig, axes = plt.subplots(3, 5, figsize=(18.5, 9.2), sharex=False)
pnl_rows = []
for i, s in enumerate(SYMBOLS):
    for j, a in enumerate(RUNGS):
        ax = axes[i, j]
        t = trades[(trades.symbol == s) & (trades.anchor == a)]
        d_s = daily_pnl(t, days[s], "net")
        d_l = daily_pnl(t, days[s], "long_net")
        ax.plot(d_s.index, d_s.cumsum() / 100, color=COLORS["strategy"], lw=1.5, label="DTW")
        ax.plot(d_l.index, d_l.cumsum() / 100, color=COLORS["always-long"], lw=1.2,
                ls="--", label="Always long")
        ax.axhline(0, color="0.4", lw=0.5)
        if i == 0:
            ax.set_title(f"T+{a}")
        if j == 0:
            ax.set_ylabel(f"{s}\ncum %")
        if i == 0 and j == 4:
            ax.legend(frameon=False, fontsize=8)
        eq = d_s.cumsum()
        pnl_rows.append({
            "asset": s, "anchor": a, "trades": len(t),
            "strategy total %": d_s.sum() / 100, "strategy Sharpe": sharpe(d_s),
            "strategy max DD %": float((eq - eq.cummax()).min()) / 100,
            "always-long total %": d_l.sum() / 100, "always-long Sharpe": sharpe(d_l),
        })
fig.suptitle(f"Test {EVAL_YEARS[0]}–{EVAL_YEARS[1]}, quarter tick, split by ladder rung",
             y=1.02, fontsize=14)
fig.tight_layout()
fig.savefig(FIG / "pnl_curves_by_anchor.png", dpi=150)
plt.show()

pnl = pd.DataFrame(pnl_rows)
pnl.to_csv(OUT / "pnl_by_anchor.csv", index=False)
print(pnl.round(3).to_string(index=False))


# %% [markdown]
# ## 2. Weighted similarity by rung — monthly and full test

# %%
trades["month"] = trades.session.dt.to_period("M").dt.to_timestamp()
monthly_sim = (trades.dropna(subset=["weighted_similarity"])
               .groupby(["symbol", "anchor", "month"], as_index=False)
               .agg(n_trades=("weighted_similarity", "size"),
                    mean_weighted_similarity=("weighted_similarity", "mean"),
                    mean_weighted_distance=("weighted_distance", "mean")))
monthly_sim.to_csv(OUT / "monthly_similarity_by_anchor.csv", index=False)

test_sim = (trades.dropna(subset=["weighted_similarity"])
            .groupby(["symbol", "anchor"], as_index=False)
            .agg(n_trades=("weighted_similarity", "size"),
                 mean_weighted_similarity=("weighted_similarity", "mean"),
                 median_weighted_similarity=("weighted_similarity", "median"),
                 mean_weighted_distance=("weighted_distance", "mean")))
test_sim.to_csv(OUT / "test_similarity_by_anchor.csv", index=False)
print("test-period mean weighted similarity:")
print(test_sim.pivot(index="symbol", columns="anchor", values="mean_weighted_similarity")
      .reindex(SYMBOLS)[RUNGS].round(3).to_string())

fig, axes = plt.subplots(3, 5, figsize=(18.5, 8.6), sharex=True, sharey="row")
for i, s in enumerate(SYMBOLS):
    for j, a in enumerate(RUNGS):
        ax = axes[i, j]
        sub = monthly_sim[(monthly_sim.symbol == s) & (monthly_sim.anchor == a)].sort_values("month")
        ax.plot(sub.month, sub.mean_weighted_similarity, color="#1f4e79", lw=1.4)
        ax.axhline(test_sim[(test_sim.symbol == s) & (test_sim.anchor == a)]
                   .mean_weighted_similarity.iloc[0], color="0.5", ls=":", lw=1)
        if i == 0:
            ax.set_title(f"T+{a}")
        if j == 0:
            ax.set_ylabel(f"{s}\nsim")
fig.suptitle("Monthly mean weighted similarity (dotted = test-period mean)", y=1.02, fontsize=14)
fig.tight_layout()
fig.savefig(FIG / "monthly_similarity_by_anchor.png", dpi=150)
plt.show()


# %% [markdown]
# ## 3. Direction hit rate over the test period

# %%
dir_rows = []
for s in SYMBOLS:
    for a in RUNGS:
        t = trades[(trades.symbol == s) & (trades.anchor == a)]
        m = t.fwd_ret != 0
        tt = t[m]
        long, short = tt[tt.dtw_dir > 0], tt[tt.dtw_dir < 0]
        dir_rows.append({
            "asset": s, "anchor": a, "trades": len(tt),
            "hit_rate": float(tt.correct.mean()) if len(tt) else np.nan,
            "hit_rate_long": float((long.fwd_ret > 0).mean()) if len(long) else np.nan,
            "hit_rate_short": float((short.fwd_ret < 0).mean()) if len(short) else np.nan,
            "pct_long": float((tt.dtw_dir > 0).mean()) if len(tt) else np.nan,
            "corr_dir_ret": float(tt.dtw_dir.corr(tt.fwd_ret)) if len(tt) > 5 else np.nan,
        })
direction = pd.DataFrame(dir_rows)
direction.to_csv(OUT / "direction_by_anchor.csv", index=False)
print(direction.round(3).to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4), sharey=True)
x = np.arange(len(RUNGS))
for ax, s in zip(axes, SYMBOLS):
    sub = direction[direction.asset == s].set_index("anchor").loc[RUNGS]
    ax.bar(x, sub.hit_rate, color="#1f4e79", width=0.7)
    ax.axhline(0.5, color="0.4", ls="--", lw=0.8)
    ax.set_xticks(x, [f"T+{a}" for a in RUNGS])
    ax.set_title(s)
    ax.set_ylim(0.35, 0.65)
    for xi, v, n in zip(x, sub.hit_rate, sub.trades):
        ax.text(xi, v + 0.008, f"{v:.1%}\n{int(n)}", ha="center", va="bottom", fontsize=8)
axes[0].set_ylabel("direction hit rate")
fig.suptitle("Test-period sign(dtw_dir) vs 30-minute return, by rung", y=1.03, fontsize=14)
fig.tight_layout()
fig.savefig(FIG / "direction_hit_by_anchor.png", dpi=150)
plt.show()


# %% [markdown]
# **Findings:** The edge is concentrated on the first rungs. **T+1** is the only step
# that clearly beats always-long on both equity books: ES +9.0% / Sharpe 0.93 (always-long
# +0.9%), NQ +9.7% / 0.86 (always-long +2.3%). Hit rates there are 55% and 54%. By
# **T+121** ES is −6.9% / −1.06 with hit rate 43%; NQ is flat-to-negative. Similarity
# barely moves across rungs (ES 0.48–0.54, NQ 0.48–0.54, ZN 0.44–0.48), so later
# windows are not “worse matches” — they are worse *direction*. ZN is negative or
# noise on every rung. This matches the idea that surprise/reaction content lives
# near T+1, and the later ladder is leftover path.
