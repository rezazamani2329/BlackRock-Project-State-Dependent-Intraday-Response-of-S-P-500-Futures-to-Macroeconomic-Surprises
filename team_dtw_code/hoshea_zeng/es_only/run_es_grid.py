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
# # ES-only event book — three construction axes
#
# Jack's traded ES book (sign of `dtw_dir`, skip the middle 40%, inverse-`dtw_disp`,
# quarter tick, expanding walk-forward from 2018) with the three knobs he did not
# sweep on the headline Sharpe:
#
# 1. **waiting time** `w ∈ {1, 2, 5}` — ladder `T+w, T+w+30, …, T+w+120`
# 2. **path lookback** `path_pre ∈ {30, 45, 60}` — DTW path length = `path_pre + w`
# 3. **neighbour weights** — seven aggregations of the k = 15 nearest
#
# `k = 15`, lookback = 3 years, `TAIL_Q = 0.30` stay fixed so the grid is those three
# axes only. One asset: ES.

# %%
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path("hoshea_zeng/es_only").resolve()
sys.path.insert(0, str(HERE))

from config import (
    ASSETS,
    CLEAN_BARS_FMT,
    COST_TICKS,
    FIG,
    HOLD_MIN,
    K_NEIGHBOURS,
    LOOKBACK_YEARS,
    MAX_POSITION,
    NY,
    OOS_START_YEAR,
    OUT,
    PATH_PRE_GRID_MIN,
    SESSION_SHIFT,
    SIZING,
    TAIL_Q,
    TICK,
    TRAIN_YEARS_SPAN,
    VALID_YEARS_SPAN,
    WAIT_GRID_MIN,
    WEIGHTING_GRID,
    anchors_for_wait,
    path_len_minutes,
)
from engine import (
    admit_games,
    build_games,
    dtw_block,
    impact_by_year,
    load_families,
    load_prices,
    score_span,
)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 11,
    "axes.titlesize": 13,
    "figure.dpi": 120,
})
NAVY, GOLD, BLUE, ORANGE = "#003262", "#FDB515", "#3B7EA1", "#C4820E"

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 80)

RESULTS = OUT / "es_only_grid.csv"


def session_of(ts: pd.Series) -> pd.Series:
    return (ts.dt.tz_convert(NY) + SESSION_SHIFT).dt.normalize().dt.tz_localize(None)


def load_es_bars() -> pd.DataFrame:
    b = pd.read_parquet(CLEAN_BARS_FMT.format(symbol="ES"),
                        columns=["close", "contract", "session_date"])
    b.index = pd.to_datetime(b.index, utc=True)
    return b.sort_index()


def buy_and_hold_daily(bars: pd.DataFrame) -> pd.Series:
    r = np.log(bars.close).diff() * 1e4
    r[bars.contract.ne(bars.contract.shift())] = 0.0
    return r.fillna(0.0).groupby(pd.DatetimeIndex(bars.session_date)).sum()


def make_rule(fit: pd.DataFrame) -> dict:
    return {
        "tail_lo": fit.dtw_dir.quantile(TAIL_Q),
        "tail_hi": fit.dtw_dir.quantile(1 - TAIL_Q),
        "disp_ref": fit.dtw_disp.median(),
    }


def trades_year(g: pd.DataFrame, bars: pd.DataFrame, rule: dict) -> pd.DataFrame:
    in_tail = (g.dtw_dir <= rule["tail_lo"]) | (g.dtw_dir >= rule["tail_hi"])
    size = np.clip(rule["disp_ref"] / g.dtw_disp, 0.0, MAX_POSITION)
    g = g.copy()
    g["position"] = np.where(in_tail, np.sign(g.dtw_dir) * size, 0.0)
    t = (g[g.position != 0].groupby("entry_ts")
         .agg(position=("position", "mean"), fwd_ret=("fwd_ret", "first")).reset_index())
    t = t[t.position != 0]
    if t.empty:
        return t
    t["session"] = session_of(t.entry_ts)
    tick_bps = TICK / bars.close.reindex(t.entry_ts).to_numpy() * 1e4
    t["gross"] = t.position * t.fwd_ret * 1e4
    t["net"] = t.gross - t.position.abs() * 2 * COST_TICKS * tick_bps
    t["long_net"] = t.fwd_ret * 1e4 - 2 * COST_TICKS * tick_bps
    return t


def walk_forward(panel: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    g = panel[panel.admitted & panel.dtw_dir.notna()].copy()
    parts = []
    for y in range(OOS_START_YEAR, int(g.year.max()) + 1):
        fit = g[g.year < y]
        te = g[g.year == y]
        if len(fit) < 80 or te.empty:
            continue
        parts.append(trades_year(te, bars, make_rule(fit)))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def daily_col(t: pd.DataFrame, days: pd.DatetimeIndex, col: str) -> pd.Series:
    if t.empty:
        return pd.Series(0.0, index=days)
    return t.groupby("session")[col].sum().reindex(days).fillna(0.0)


def metrics(daily: pd.Series, bench: pd.Series | None = None) -> dict:
    sd = daily.std(ddof=1)
    eq = daily.cumsum()
    out = {
        "total_%": daily.sum() / 100,
        "vol_%": float(sd * np.sqrt(252) / 100) if sd else np.nan,
        "sharpe": (daily.mean() / sd * np.sqrt(252)) if sd else np.nan,
        "max_dd_%": float((eq - eq.cummax()).min()) / 100,
    }
    if bench is not None and bench.var():
        out["beta"] = float(np.cov(daily.reindex(bench.index).fillna(0.0), bench)[0, 1] / bench.var())
    else:
        out["beta"] = np.nan
    return out


def evaluate(panel: pd.DataFrame, bars: pd.DataFrame, bnh: pd.Series) -> dict:
    t = walk_forward(panel, bars)
    days = bnh.index[(bnh.index.year >= OOS_START_YEAR)]
    net = daily_col(t, days, "net")
    gross = daily_col(t, days, "gross")
    long_ = daily_col(t, days, "long_net")
    m_net = metrics(net, bnh.reindex(days).fillna(0.0))
    m_gross = metrics(gross, bnh.reindex(days).fillna(0.0))
    m_long = metrics(long_, bnh.reindex(days).fillna(0.0))
    n_years = max(days.year.nunique(), 1)
    ic_tr = score_span(panel, panel[["dtw_dir", "dtw_disp"]], TRAIN_YEARS_SPAN)
    ic_va = score_span(panel, panel[["dtw_dir", "dtw_disp"]], VALID_YEARS_SPAN)
    return {
        "n_trades": int(len(t)),
        "trades_per_year": len(t) / n_years,
        "net_sharpe": m_net["sharpe"],
        "net_total_%": m_net["total_%"],
        "net_vol_%": m_net["vol_%"],
        "net_max_dd_%": m_net["max_dd_%"],
        "net_beta": m_net["beta"],
        "gross_sharpe": m_gross["sharpe"],
        "gross_total_%": m_gross["total_%"],
        "long_sharpe": m_long["sharpe"],
        "long_total_%": m_long["total_%"],
        "train_dir_ic": ic_tr["dir_ic"],
        "valid_dir_ic": ic_va["dir_ic"],
        "train_disp_ic": ic_tr["disp_ic"],
        "valid_disp_ic": ic_va["disp_ic"],
        "train_hit": ic_tr["hit"],
        "valid_hit": ic_va["hit"],
        "_net": net,
        "_long": long_,
        "_gross": gross,
    }


# %% [markdown]
# ## Run the grid
#
# Nine DTW constructions (wait × path_pre) × seven weightings. Distances are computed
# once per construction.

# %%
def run_grid() -> pd.DataFrame:
    pxs, contracts = load_prices()
    fam = load_families()
    bars = load_es_bars()
    bnh = buy_and_hold_daily(bars)
    years = sorted({t.year for ts in fam.values() for t in ts})
    impact = impact_by_year(pxs["ES"], fam, years)
    print(f"ES impact screen: {len(impact):,} family-years")

    rows, curves = [], {}
    n_jobs = len(WAIT_GRID_MIN) * len(PATH_PRE_GRID_MIN)
    job = 0
    for wait in WAIT_GRID_MIN:
        for pre in PATH_PRE_GRID_MIN:
            job += 1
            plen = path_len_minutes(pre, wait)
            t0 = time.time()
            print(f"[{job}/{n_jobs}] wait={wait} pre={pre} path={plen}  "
                  f"anchors={anchors_for_wait(wait)}", flush=True)
            g = admit_games(build_games(pxs["ES"], contracts["ES"], fam, wait, pre), impact)
            print(f"         {len(g):,} games, {int(g.admitted.sum()):,} admitted", flush=True)
            feats = dtw_block(g, g.admitted.to_numpy(),
                              [K_NEIGHBOURS], [LOOKBACK_YEARS], WEIGHTING_GRID)
            g_nop = g.drop(columns=["path"])
            for scheme in WEIGHTING_GRID:
                f = feats[K_NEIGHBOURS, LOOKBACK_YEARS, scheme]
                panel = pd.concat([g_nop, f], axis=1)
                ev = evaluate(panel, bars, bnh)
                key = (wait, pre, scheme)
                curves[key] = {k: ev.pop(k) for k in ("_net", "_long", "_gross")}
                rows.append({
                    "wait_min": wait,
                    "path_pre_min": pre,
                    "path_len": plen,
                    "weighting": scheme,
                    "k": K_NEIGHBOURS,
                    "lookback_years": LOOKBACK_YEARS,
                    "q": TAIL_Q,
                    **ev,
                })
            print(f"         {len(WEIGHTING_GRID)} weightings in {time.time() - t0:.0f}s", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS, index=False)
    pd.to_pickle(curves, OUT / "es_only_curves.pkl")
    print(f"wrote {RESULTS}  {len(out)} rows")
    return out, curves, bnh


# %%
if RESULTS.exists():
    grid = pd.read_csv(RESULTS)
    if grid["net_vol_%"].median() > 20:
        grid["net_vol_%"] = grid["net_vol_%"] / 100
        grid.to_csv(RESULTS, index=False)
    curves = pd.read_pickle(OUT / "es_only_curves.pkl") if (OUT / "es_only_curves.pkl").exists() else {}
    print(f"loaded {RESULTS.name}  {len(grid)} rows")
    bars = load_es_bars()
    bnh = buy_and_hold_daily(bars)
else:
    grid, curves, bnh = run_grid()


# %% [markdown]
# ## Tables

# %%
def is_baseline(r) -> bool:
    return (r.wait_min == 1) and (r.path_pre_min == 30) and (r.weighting == "inv_dist_recency")


show = ["wait_min", "path_pre_min", "path_len", "weighting",
        "net_sharpe", "net_total_%", "net_vol_%", "net_max_dd_%", "n_trades",
        "train_dir_ic", "valid_dir_ic", "train_disp_ic", "valid_disp_ic"]
print("Jack-style baseline (wait=1, path_pre=30, inv_dist_recency):")
print(grid[grid.apply(is_baseline, axis=1)][show].round(4).to_string(index=False))
print("\ntop 10 by OOS net Sharpe (walk-forward 2018–2026):")
print(grid.sort_values("net_sharpe", ascending=False).head(10)[show].round(4).to_string(index=False))
print("\nbottom 5 by OOS net Sharpe:")
print(grid.sort_values("net_sharpe").head(5)[show].round(4).to_string(index=False))

print("\nmean net Sharpe by wait × path_pre:")
print(grid.pivot_table(index="wait_min", columns="path_pre_min", values="net_sharpe").round(3))
print("\nmean net Sharpe by weighting:")
print(grid.groupby("weighting")["net_sharpe"].mean().sort_values(ascending=False).round(3).to_string())
print("\nmean valid dir IC / disp IC by wait × path_pre:")
print(grid.pivot_table(index="wait_min", columns="path_pre_min", values="valid_dir_ic").round(3))
print(grid.pivot_table(index="wait_min", columns="path_pre_min", values="valid_disp_ic").round(3))


# %% [markdown]
# ## Figures

# %%
def savefig(fig, name: str) -> None:
    p = FIG / name
    fig.savefig(p, bbox_inches="tight")
    print("wrote", p)


heat = grid.pivot_table(index="wait_min", columns="path_pre_min", values="net_sharpe")
fig, ax = plt.subplots(figsize=(6.2, 3.8))
im = ax.imshow(heat.values, cmap="RdYlGn", vmin=-0.4, vmax=0.8, aspect="auto")
ax.set_xticks(range(len(heat.columns)), [str(c) for c in heat.columns])
ax.set_yticks(range(len(heat.index)), [str(i) for i in heat.index])
ax.set_xlabel("path lookback (min)")
ax.set_ylabel("waiting time (min)")
ax.set_title("ES OOS net Sharpe  ·  mean over weightings")
for i, w in enumerate(heat.index):
    for j, p in enumerate(heat.columns):
        ax.text(j, i, f"{heat.loc[w, p]:.2f}", ha="center", va="center", color="black", fontsize=11)
fig.colorbar(im, ax=ax, fraction=0.046)
savefig(fig, "sharpe_heatmap_wait_path.png")
plt.close(fig)

sub = grid[(grid.wait_min == 1) & (grid.path_pre_min == 30)].sort_values("net_sharpe")
fig, ax = plt.subplots(figsize=(7.2, 3.6))
colors = [GOLD if w == "inv_dist_recency" else BLUE for w in sub.weighting]
ax.barh(sub.weighting, sub.net_sharpe, color=colors)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("OOS net Sharpe")
ax.set_title("Weighting schemes  ·  wait=1, 31-point path (Jack's construction)")
savefig(fig, "sharpe_by_weighting_baseline_path.png")
plt.close(fig)

if curves:
    base_key = (1, 30, "inv_dist_recency")
    best = grid.sort_values("net_sharpe", ascending=False).iloc[0]
    best_key = (int(best.wait_min), int(best.path_pre_min), best.weighting)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for key, label, color, ls in [
        (base_key, "Jack baseline (w=1, 31pt, 1/d×recency)", NAVY, "-"),
        (best_key, f"best Sharpe  w={best_key[0]} pre={best_key[1]} {best_key[2]}", GOLD, "-"),
    ]:
        if key in curves:
            ax.plot(curves[key]["_net"].cumsum() / 100, label=label, color=color, lw=2, ls=ls)
    if base_key in curves:
        ax.plot(curves[base_key]["_long"].cumsum() / 100,
                label="Always long, same windows (net)", color=ORANGE, lw=1.5, ls="--")
    ax.set_ylabel("cumulative %  (one ES contract)")
    ax.set_title("Walk-forward 2018–2026  ·  quarter tick")
    ax.legend(frameon=False, loc="upper left")
    savefig(fig, "equity_baseline_vs_best.png")
    plt.close(fig)

print("\nPPT-style row for the Jack baseline:")
b = grid[grid.apply(is_baseline, axis=1)].iloc[0]
print(pd.Series({
    "Net total %": b["net_total_%"], "Net vol %": b["net_vol_%"], "Net Sharpe": b["net_sharpe"],
    "Net max DD %": b["net_max_dd_%"], "Beta": b["net_beta"], "Trades/yr": b["trades_per_year"],
    "Gross Sharpe": b["gross_sharpe"], "Gross total %": b["gross_total_%"],
}).round(3).to_string())


# %% [markdown]
# **Findings:** Walk-forward 2018–2026, ES only, Jack's rule (k=15, lookback 3y,
# middle 40% gated, inverse-`dtw_disp`, quarter tick). The Jack baseline
# (wait=1, 31-point path, `inv_dist_recency`) prints **1,669 trades (~185/yr),
# net Sharpe 0.53, total +14.3%, max DD −4.93%** — same trade count and drawdown
# as the PPT, Sharpe a little lower (PPT 0.61). The only combos that look as
# good or better keep that **1-minute, 31-point path**: `inv_dist_sq_recency`
# reaches net Sharpe **0.61** / +16.2%. Stretching the path to 45–60 minutes or
# waiting 2–5 minutes **hurts** OOS Sharpe (wait=1 × 30-min path averages 0.51
# across weights; wait=5 × 45-min path averages −0.38). `dtw_disp` IC stays
# positive on every cell; `dtw_dir` IC is still negative in train and only
# positive on the original 31-point path in validation. So the prettier ES-only
# number is the original construction (optionally `1/d² × recency`), not a new
# waiting time or a longer DTW path.

