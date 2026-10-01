"""
dtw_book_with_ym.py
===================
Figures for the DTW paper with YM added to the DTW book. Run after the DTW_EXTRA=YM runs of
features.py and backtest.py, which write to combined/output/with_YM/:

    DTW_EXTRA=YM uv run python combined/features.py
    DTW_EXTRA=YM LEAK_TEST_MINUTES=5 uv run python combined/features.py
    DTW_EXTRA=YM uv run python combined/backtest.py > combined/output/with_YM/backtest_log.txt
    DTW_EXTRA=YM uv run python combined/dtw_book_with_ym.py

Writes to combined/output/with_YM/:
    sharpe_by_cost_with_ym.png       Sharpe by cost level against buy-and-hold, each book
    pnl_curves_by_anchor_with_ym.png cumulative P&L by ladder rung against always long
    pnl_by_anchor.csv                the numbers behind the second figure
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("DTW_EXTRA", "YM")
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path("combined").resolve()
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (ALL_ASSETS, ANCHORS, BOOK_SYMBOLS, CLEAN_BARS_FMT, COST_TICKS_PER_SIDE,
                    MAX_POSITION, NY, OUT, PANEL, SESSION_SHIFT, TEST_YEARS_SPAN, TRAIN_YEARS_SPAN)

EVAL_YEARS = TEST_YEARS_SPAN
FIT_YEARS = (TRAIN_YEARS_SPAN[0], EVAL_YEARS[0] - 1)
Q = 0.3
COST = "quarter tick"
SYMBOLS = ["ES", "NQ", "YM", "ZN"]
NAVY, ORANGE = "#1F3A5F", "#C8702A"

# ---------------------------------------------------------------- figure 5: Sharpe by cost
summ = pd.read_csv(OUT / "backtest_summary.csv", index_col=[0, 1])
BOOKS = [("ES", "ES"), ("NQ", "NQ"), ("YM", "YM"), ("equities", "ES + NQ"), ("equities + YM", "ES + NQ + YM")]
vals = {k: [] for k in ["Gross", "Quarter tick", "Half tick", "Buy-and-hold"]}
for key, _ in BOOKS:
    t = summ.loc[key]
    pre = "strategy" if key in ALL_ASSETS else key
    vals["Gross"].append(t.loc[f"{pre}, gross", "sharpe"])
    vals["Quarter tick"].append(t.loc[f"{pre}, quarter tick", "sharpe"])
    vals["Half tick"].append(t.loc[f"{pre}, half tick", "sharpe"])
    vals["Buy-and-hold"].append(t.loc["buy and hold" if key in ALL_ASSETS else "risk-weighted buy and hold", "sharpe"])
fig, ax = plt.subplots(figsize=(9.5, 3.4))
x = np.arange(len(BOOKS)); w = 0.19
cc = [NAVY, "#4F6D91", "#8FA6C1", ORANGE]
for i, (k, v) in enumerate(vals.items()):
    b = ax.bar(x + (i - 1.5) * w, v, w, color=cc[i], label=k)
    for r in b:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width() / 2, h + (0.01 if h >= 0 else -0.06), f"{h:.2f}",
                ha="center", fontsize=7)
ax.axhline(0, color="black", lw=0.6)
ax.set_xticks(x, [lab for _, lab in BOOKS])
ax.set_ylabel(f"Sharpe ratio, {EVAL_YEARS[0]}–{EVAL_YEARS[1]}")
ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper right")
lo = min(min(v) for v in vals.values())
ax.set_ylim(min(0, lo - 0.1), max(max(v) for v in vals.values()) + 0.25)
plt.tight_layout()
plt.savefig(OUT / "sharpe_by_cost_with_ym.png", dpi=200)
plt.close()
print(pd.DataFrame(vals, index=[lab for _, lab in BOOKS]).round(2).to_string())


# ---------------------------------------------------------------- figure 6: P&L by rung
def session_of(ts):
    return (ts.dt.tz_convert(NY) + SESSION_SHIFT).dt.normalize().dt.tz_localize(None)


def sharpe(d):
    sd = d.std(ddof=1)
    return float(d.mean() / sd * np.sqrt(252)) if sd else np.nan


panel = pd.read_parquet(PANEL)
panel["entry_ts"] = pd.to_datetime(panel["entry_ts"], utc=True)
games = panel[panel.admitted & panel.dtw_dir.notna()].copy()
bars, days = {}, {}
for s in SYMBOLS:
    b = pd.read_parquet(CLEAN_BARS_FMT.format(symbol=s), columns=["close", "contract", "session_date"])
    b.index = pd.to_datetime(b.index, utc=True)
    bars[s] = b.sort_index()
    sess = pd.DatetimeIndex(pd.to_datetime(b.session_date).unique()).sort_values()
    days[s] = sess[(sess.year >= EVAL_YEARS[0]) & (sess.year <= EVAL_YEARS[1])]


def trades_for(s):
    """The book's rule (as backtest.py), keeping each trade's rung."""
    g = games[games.symbol == s].copy()
    fit = g[g.year.between(*FIT_YEARS)]
    lo, hi, ref = fit.dtw_dir.quantile(Q), fit.dtw_dir.quantile(1 - Q), fit.dtw_disp.median()
    cur = g[g.year.between(*EVAL_YEARS)].copy()
    size = np.clip(ref / cur.dtw_disp, 0.0, MAX_POSITION)
    cur["position"] = np.where((cur.dtw_dir <= lo) | (cur.dtw_dir >= hi), np.sign(cur.dtw_dir) * size, 0.0)
    t = (cur[cur.position != 0].groupby("entry_ts")
         .agg(position=("position", "mean"), fwd_ret=("fwd_ret", "first"),
              dtw_dir=("dtw_dir", "first"), anchor=("anchor", "first")).reset_index())
    t = t[t.position != 0]
    t["session"] = session_of(t.entry_ts)
    tick_bps = ALL_ASSETS[s]["tick"] / bars[s].close.reindex(t.entry_ts).to_numpy() * 1e4
    cost = t.position.abs() * 2 * COST_TICKS_PER_SIDE[COST] * tick_bps
    t["net"] = t.position * t.fwd_ret * 1e4 - cost
    t["long_net"] = t.position.abs() * t.fwd_ret * 1e4 - cost
    t["hit"] = np.sign(t.position) * t.fwd_ret > 0
    return t


fig, axes = plt.subplots(len(SYMBOLS), len(ANCHORS), figsize=(16, 2.6 * len(SYMBOLS)))
rows = []
for i, s in enumerate(SYMBOLS):
    tr = trades_for(s)
    for j, a in enumerate(ANCHORS):
        ax = axes[i, j]
        t = tr[tr.anchor == a]
        d_s = t.groupby("session").net.sum().reindex(days[s]).fillna(0.0)
        d_l = t.groupby("session").long_net.sum().reindex(days[s]).fillna(0.0)
        ax.plot(d_s.index, d_s.cumsum() / 100, color=NAVY, lw=1.3, label="DTW book")
        ax.plot(d_l.index, d_l.cumsum() / 100, color="0.6", lw=1.0, ls="--", label="Always long")
        ax.axhline(0, color="0.4", lw=0.5)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.set_title(f"Entry T+{a}", fontsize=10)
        if j == 0:
            ax.set_ylabel(f"{s}\ncumulative %", fontsize=9)
        ax.text(0.03, 0.92, f"Sharpe {sharpe(d_s):.2f}", transform=ax.transAxes, fontsize=7.5, va="top")
        rows.append({"asset": s, "anchor": a, "trades": len(t), "strategy total %": d_s.sum() / 100,
                     "strategy Sharpe": sharpe(d_s), "hit rate": t.hit.mean(),
                     "always-long total %": d_l.sum() / 100, "always-long Sharpe": sharpe(d_l)})
axes[0, -1].legend(frameon=False, fontsize=7, loc="lower left")
plt.tight_layout()
plt.savefig(OUT / "pnl_curves_by_anchor_with_ym.png", dpi=170)
plt.close()
pnl = pd.DataFrame(rows)
pnl.to_csv(OUT / "pnl_by_anchor.csv", index=False)
print(pnl.round(3).to_string(index=False))
print(f"wrote figures and pnl_by_anchor.csv to {OUT}")
