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
# # Monthly weighted DTW match similarity (test trades)
#
# For every **executed** test-period trade, rebuild the k-neighbour match used to form
# `dtw_dir` and record how similar those neighbours were.
#
# Weights are the book's: \(w_i \propto (1/(d_i+\varepsilon))\exp(-\mathrm{age}_i/730)\),
# then \(\sum w_i = 1\). Similarity of neighbour \(i\) is the inv-distance term
# \(s_i = 1/(d_i+\varepsilon)\). The trade-level number is the same weighted average
# \(\sum w_i s_i\). Months average those trade-level numbers (equal weight per trade).
#
# Neighbour pool, k, lookback and band match `features.py` / `config.py`. Paths are
# rebuilt; admission and the tail gate come from the saved panel so the trade list
# matches `plot_test_pnl.py`.

# %%
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dtaidistance import dtw

COMBINED = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path("combined").resolve()
HERE = Path(__file__).resolve().parent if "__file__" in globals() else COMBINED / "result_analysis_by_month"
sys.path.insert(0, str(COMBINED))

from config import (ANCHORS, BLOOMBERG, BOOK_SYMBOLS, CLEAN_BARS_FMT, DTW_GRID,
                    DTW_HALFLIFE_DAYS, DTW_MAX_POOL, DTW_WARP_MINUTES, DTW_WEIGHTS,
                    ENTRY_LAG_MIN, FAMILY_OVERLAP, HOLD_MIN, LEAK_TEST_MINUTES,
                    MAX_POSITION, MIN_NEIGHBOURS, NY, PANEL, PATH_FREQ_MIN, PATH_PRE_MIN,
                    SESSION_SHIFT, TEST_YEARS_SPAN, TRAIN_YEARS_SPAN)

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight", "savefig.facecolor": "white"})

FIG = HERE / "figures"
OUT = HERE / "output"
FIG.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

EVAL_YEARS = TEST_YEARS_SPAN
FIT_YEARS = (TRAIN_YEARS_SPAN[0], EVAL_YEARS[0] - 1)
Q = 0.3
K = DTW_GRID["k"][0]
LOOKBACK = DTW_GRID["lookback_years"][0]
EPS = 1e-6
SYMBOLS = list(BOOK_SYMBOLS)


class Px:
    def __init__(self, s: pd.Series):
        self.ts = s.index.values.astype("datetime64[ns]").astype(np.int64)
        self.v = s.to_numpy(float)

    def at(self, stamps) -> np.ndarray:
        q = pd.DatetimeIndex(stamps).tz_convert("UTC").values.astype("datetime64[ns]").astype(np.int64)
        pos = np.searchsorted(self.ts, q)
        pc = np.clip(pos, 0, len(self.ts) - 1)
        out = np.full(len(q), np.nan)
        hit = (pos < len(self.ts)) & (self.ts[pc] == q)
        out[hit] = self.v[pc[hit]]
        return out


def session_of(ts: pd.Series) -> pd.Series:
    return (ts.dt.tz_convert(NY) + SESSION_SHIFT).dt.normalize().dt.tz_localize(None)


def load_lines() -> pd.DataFrame:
    b = pd.concat([pd.read_excel(BLOOMBERG, sheet_name=s)
                   for s in pd.ExcelFile(BLOOMBERG).sheet_names], ignore_index=True)
    b.columns = [c.split(".")[-1] if "DROPNA" in str(c) else c for c in b.columns]
    b = b[b.COUNTRY_NAME.astype(str).str.contains("United States", na=False)]
    b["RELEASE_DATE"] = pd.to_datetime(b.RELEASE_DATE, errors="coerce")
    b = b.dropna(subset=["RELEASE_DATE", "RELEASE_TIME", "EVENT_NAME"])
    t = pd.to_datetime(b.RELEASE_TIME.astype(str), format="%H:%M:%S", errors="coerce")
    b, t = b[t.notna()].copy(), t[t.notna()]
    local = (b.RELEASE_DATE.dt.normalize() + pd.to_timedelta(t.dt.hour, "h")
             + pd.to_timedelta(t.dt.minute, "m"))
    b["ts"] = local.dt.tz_localize(NY, ambiguous=True, nonexistent="shift_forward").dt.tz_convert("UTC")
    b["clock_et"] = b.ts.dt.tz_convert(NY).dt.strftime("%H:%M")
    return b[["ts", "clock_et", "EVENT_NAME"]].rename(columns={"EVENT_NAME": "line"})


def assign_families(lines: pd.DataFrame) -> dict[str, str]:
    stamps = {ln: set(g.ts) for ln, g in lines.groupby("line") if len(g) >= 12}
    names = sorted(stamps, key=lambda n: -len(stamps[n]))
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(names):
        for c in names[i + 1:]:
            if find(a) == find(c):
                continue
            shared = len(stamps[a] & stamps[c])
            if shared and shared / min(len(stamps[a]), len(stamps[c])) >= FAMILY_OVERLAP:
                parent[find(c)] = find(a)
    groups: dict[str, list[str]] = {}
    for n in names:
        groups.setdefault(find(n), []).append(n)
    out = {}
    for members in groups.values():
        best = max(members, key=lambda m: len(stamps[m]))
        clock = lines.loc[lines.line == best, "clock_et"].mode().iloc[0]
        for m in members:
            out[m] = f"{best} @ {clock}"
    return out


def build_games(px: Px, contract: Px, fam_stamps: dict) -> pd.DataFrame:
    steps = np.arange(-PATH_PRE_MIN, LEAK_TEST_MINUTES + 1)
    recs = []
    for f, ts in fam_stamps.items():
        T = pd.DatetimeIndex(sorted(ts))
        has_print = np.isfinite(px.at(T)) & np.isfinite(px.at(T + pd.Timedelta(minutes=ENTRY_LAG_MIN)))
        for a in ANCHORS:
            entry = T + pd.Timedelta(minutes=a)
            grid = entry.values[:, None] + (steps * 60_000_000_000).astype("timedelta64[ns]")
            path = px.at(pd.DatetimeIndex(grid.ravel()).tz_localize("UTC")).reshape(len(T), len(steps))
            p0 = px.at(entry)
            p1 = px.at(entry + pd.Timedelta(minutes=HOLD_MIN))
            same_contract = (contract.at(entry - pd.Timedelta(minutes=PATH_PRE_MIN))
                             == contract.at(entry + pd.Timedelta(minutes=HOLD_MIN)))
            ok = (np.isfinite(path).all(1) & np.isfinite(p0) & np.isfinite(p1) & has_print
                  & same_contract)
            sd = path.std(1, keepdims=True)
            sd[sd < 1e-12] = 1.0
            z = (path - path.mean(1, keepdims=True)) / sd
            for i in np.where(ok)[0]:
                recs.append(dict(
                    family=f, event_ts=T[i], anchor=int(a),
                    clock_et=T[i].tz_convert(NY).strftime("%H:%M"),
                    entry_ts=entry[i],
                    year=int(T[i].year),
                    path=z[i]))
    return pd.DataFrame(recs)


def traded_keys(panel: pd.DataFrame, s: str) -> pd.DataFrame:
    g = panel[(panel.symbol == s) & panel.admitted & panel.dtw_dir.notna()].copy()
    fit = g[g.year.between(*FIT_YEARS)]
    lo, hi = fit.dtw_dir.quantile(Q), fit.dtw_dir.quantile(1 - Q)
    ref = fit.dtw_disp.median()
    cur = g[g.year.between(*EVAL_YEARS)].copy()
    in_tail = (cur.dtw_dir <= lo) | (cur.dtw_dir >= hi)
    size = np.clip(ref / cur.dtw_disp, 0.0, MAX_POSITION)
    cur["position"] = np.where(in_tail, np.sign(cur.dtw_dir) * size, 0.0)
    t = (cur[cur.position != 0]
         .groupby("entry_ts", as_index=False)
         .agg(position=("position", "mean"), anchor=("anchor", "first"),
              family=("family", "first"), clock_et=("clock_et", "first"),
              event_ts=("event_ts", "first")))
    return t[t.position != 0]


def neighbour_stats(games: pd.DataFrame, query_mask: np.ndarray) -> pd.DataFrame:
    """One row per queried game: weighted similarity / distance of the k neighbours."""
    win = max(1, round(DTW_WARP_MINUTES / PATH_FREQ_MIN))
    n = len(games)
    wsim = np.full(n, np.nan)
    wdist = np.full(n, np.nan)
    kn = np.zeros(n, dtype=int)
    for _, grp in games.groupby(["clock_et", "anchor"], sort=False):
        idx = grp.index.to_numpy()
        qpos = np.where(query_mask[idx])[0]
        if len(qpos) == 0:
            continue
        t0 = grp.entry_ts.values.astype("datetime64[ns]").astype(np.int64)
        paths = [np.asarray(p, dtype=np.double) for p in grp.path]
        for pos in qpos:
            age = (t0[pos] - t0[:pos]) / 86_400_000_000_000.0
            el = np.where((age > 0) & (age <= 365 * LOOKBACK))[0]
            if DTW_MAX_POOL and len(el) > DTW_MAX_POOL:
                el = el[-DTW_MAX_POOL:]
            if len(el) < MIN_NEIGHBOURS:
                continue
            qp = paths[pos]
            d = np.array([dtw.distance_fast(qp, paths[j], window=win) for j in el], dtype=float)
            m = np.isfinite(d)
            el, d = el[m], d[m]
            if len(el) < MIN_NEIGHBOURS:
                continue
            order = np.argsort(d)[:K]
            dsel, asel = d[order], age[el[order]]
            inv = 1.0 / (dsel + EPS)
            w = inv if DTW_WEIGHTS == "inv_dist" else np.ones(len(dsel))
            w = w * np.exp(-asel / DTW_HALFLIFE_DAYS)
            w = w / w.sum()
            gi = idx[pos]
            wsim[gi] = float(np.dot(w, inv))
            wdist[gi] = float(np.dot(w, dsel))
            kn[gi] = len(dsel)
    return pd.DataFrame({"weighted_similarity": wsim, "weighted_distance": wdist,
                         "n_neighbours": kn}, index=games.index)


# %%
print("loading panel + calendar + bars")
panel = pd.read_parquet(PANEL)
panel["event_ts"] = pd.to_datetime(panel["event_ts"], utc=True)
panel["entry_ts"] = pd.to_datetime(panel["entry_ts"], utc=True)
lines = load_lines()
fam = assign_families(lines)
lines["family"] = lines.line.map(fam)
lines = lines.dropna(subset=["family"])
fam_stamps = {f: sorted(set(g.ts)) for f, g in lines.groupby("family")}

trade_rows = []
for s in SYMBOLS:
    print(f"\n=== {s}: rebuild paths and neighbour matches ===")
    b = pd.read_parquet(CLEAN_BARS_FMT.format(symbol=s), columns=["close", "contract"]).sort_index()
    b.index = pd.to_datetime(b.index, utc=True)
    g = build_games(Px(b.close), Px(b.contract.astype(float)), fam_stamps)
    g["entry_ts"] = pd.to_datetime(g["entry_ts"], utc=True)
    g["event_ts"] = pd.to_datetime(g["event_ts"], utc=True)
    g = g.sort_values("entry_ts").reset_index(drop=True)
    keys = traded_keys(panel, s)
    keys["entry_ts"] = pd.to_datetime(keys["entry_ts"], utc=True)
    mark = g.merge(keys[["entry_ts", "position"]], on="entry_ts", how="left")
    query = mark["position"].notna().to_numpy()
    print(f"  games {len(g):,}  traded queries {int(query.sum()):,}")
    stats = neighbour_stats(g, query)
    hit = mark.loc[query].join(stats)
    hit = hit[hit.weighted_similarity.notna()].copy()
    hit = (hit.groupby("entry_ts", as_index=False)
           .agg(anchor=("anchor", "first"), clock_et=("clock_et", "first"),
                family=("family", lambda v: "+".join(sorted(set(v)))),
                position=("position", "first"),
                weighted_similarity=("weighted_similarity", "mean"),
                weighted_distance=("weighted_distance", "mean"),
                n_neighbours=("n_neighbours", "mean")))
    hit["symbol"] = s
    hit["session"] = session_of(hit.entry_ts)
    trade_rows.append(hit[["symbol", "entry_ts", "session", "anchor", "clock_et", "family",
                           "position", "weighted_similarity", "weighted_distance", "n_neighbours"]])
    print(f"  scored trades {len(hit):,}  "
          f"mean weighted sim {hit.weighted_similarity.mean():.3f}")

trades = pd.concat(trade_rows, ignore_index=True)
trades.to_csv(OUT / "trade_weighted_similarity.csv", index=False)

trades["month"] = trades.session.dt.to_period("M").dt.to_timestamp()
monthly = (trades.groupby(["symbol", "month"], as_index=False)
           .agg(n_trades=("weighted_similarity", "size"),
                mean_weighted_similarity=("weighted_similarity", "mean"),
                median_weighted_similarity=("weighted_similarity", "median"),
                mean_weighted_distance=("weighted_distance", "mean"),
                mean_n_neighbours=("n_neighbours", "mean")))
monthly.to_csv(OUT / "monthly_weighted_similarity.csv", index=False)
print(f"\nwrote {OUT / 'monthly_weighted_similarity.csv'}  ({len(monthly)} month-rows)")
print(monthly.groupby("symbol")["mean_weighted_similarity"].describe().round(3).to_string())


# %% [markdown]
# ## Monthly average weighted similarity
#
# Higher = the k neighbours used that month were closer (after the book's 1/d × recency
# weights). This is match quality, not directional accuracy.

# %%
fig, axes = plt.subplots(3, 1, figsize=(16.5, 10.5), sharex=True)
for ax, s in zip(axes, SYMBOLS):
    sub = monthly[monthly.symbol == s].sort_values("month")
    ax.plot(sub.month, sub.mean_weighted_similarity, color="#1f4e79", lw=1.8, marker="o",
            ms=3.5, label="Mean weighted similarity")
    ax.set_ylabel(f"{s}\nweighted sim")
    ax2 = ax.twinx()
    ax2.bar(sub.month, sub.n_trades, width=pd.Timedelta(days=20), color="0.75",
            alpha=0.45, label="Trades")
    ax2.set_ylabel("trades")
    ax.grid(True, axis="y", alpha=0.4)
    if s == "ES":
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left", fontsize=10)
fig.suptitle("Test trades: monthly average of book-weighted neighbour similarity  "
             r"($\sum w_i / (d_i+\varepsilon)$)", y=1.01, fontsize=14)
fig.tight_layout()
fig.savefig(FIG / "monthly_weighted_similarity.png", dpi=150)
plt.show()

fig, ax = plt.subplots(figsize=(16.5, 4.8))
for s, c in zip(SYMBOLS, ["#1f4e79", "#c45c26", "#2e7d4f"]):
    sub = monthly[monthly.symbol == s].sort_values("month")
    ax.plot(sub.month, sub.mean_weighted_similarity, lw=1.8, marker="o", ms=3, label=s, color=c)
ax.set_ylabel("mean weighted similarity")
ax.set_title("ES / NQ / ZN on the same scale")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIG / "monthly_weighted_similarity_overlay.png", dpi=150)
plt.show()


# %% [markdown]
# **Findings:** Test trades, k = 15, same 1/d × recency weights as `dtw_dir`. Mean
# weighted similarity is **0.51 on ES and NQ** and **0.46 on ZN**. Weakest months:
# ES 2026-06 (0.41), NQ 2021-10 (0.39), ZN 2021-06 (0.38). Strongest: ES 2023-10
# (0.61), NQ 2024-06 (0.62), ZN 2022-06 (0.54). 2022 H1 is not a poor-match window
# (ES 0.51, NQ 0.50, ZN 0.48 vs other months 0.51 / 0.51 / 0.46). The equity
# drawdown is not “no close neighbours”; matches were as close as usual, and the
# path-to-return map is what broke.
