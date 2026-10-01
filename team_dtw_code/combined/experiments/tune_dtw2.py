"""Second DTW tuning round: what the match is made on and how neighbours become a signal.
ES and NQ, 2013-2020 only (the test years are never loaded).

Holds config.py's setting fixed (band 10, 8-year lookback, no pool cap, no recency decay,
uniform weights) and varies:

    path        "zprice"  z-scored price level (features.py)
                "zret"    z-scored 1-minute log returns: the rhythm of the move, not its level
                "bps"     price change from the path's start in bps, not z-scored: keeps size
    length      15, 30, 60 minutes before entry
    pool        "clock"   same clock slot and anchor, any family (features.py)
                "family"  same release family and anchor only
    k           20, 40, 80
    aggregate   "mean"    mean neighbour forward return (features.py)
                "median"  median neighbour forward return
                "vote"    share of neighbours that went up, minus one half
                "volnorm" mean of each neighbour's forward return / its own pre-entry vol

dtw_disp is always the sd of the k neighbours' forward returns. Each variant is scored by
rank IC and by trading: gate on the outer 2q of the aggregate (walk-forward thresholds from
earlier years), q in {0.3, 0.4, 0.5 = trade all}, inverse-dtw_disp sizing, quarter tick per
side, Sharpe on the daily session grid. Train is scored 2014-2017, validation 2018-2020.

Run: uv run python combined/experiments/tune_dtw2.py

Findings (2026-09-27): nothing here beats config.py's setting. Across the 216 variants, train
and validation IC correlate at 0.07; with ~700 unique games per period an IC's standard error
is ~0.04, so the best train ICs (+0.08 to +0.10) are no better than the best of 216 noise
signals. On average, z-scored returns do a little better than z-scored price, and the
raw-bps path and the family-only pool do worse (family pool NQ validation Sharpe −0.73).
The current setting (z-scored price, 30 minutes, clock pool, k 40, mean) is one of 20
variants with positive IC on ES and NQ in both periods (+0.05 / +0.05 ES, +0.05 / +0.04 NQ,
train / valid), and the steadiest of them. Median and vote raise train IC but fall in validation.
"""
from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dtaidistance import dtw
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from config import (ASSETS, BOOK_SYMBOLS, CLEAN_BARS_FMT, DTW_GRID, DTW_WARP_MINUTES, MAX_POSITION,
                    MIN_NEIGHBOURS, NY, OUT, PANEL, SESSION_SHIFT, VALID_YEARS_SPAN)
from util import Px

LAST_YEAR = VALID_YEARS_SPAN[1]
PERIODS = {"train": (2014, 2017), "valid": VALID_YEARS_SPAN}
IC_PERIODS = {"train": (2013, 2017), "valid": VALID_YEARS_SPAN}
LOOKBACK = DTW_GRID["lookback_years"][0]
COST_TICKS = 0.25
TUNE = OUT / "tune"
TUNE.mkdir(exist_ok=True)

PATHS = ["zprice", "zret", "bps"]
LENGTHS = [15, 30, 60]
POOLS = ["clock", "family"]
KS = [20, 40, 80]
AGGS = ["mean", "median", "vote", "volnorm"]
GATE_QS = [0.3, 0.4, 0.5]


def load(s: str):
    panel = pd.read_parquet(PANEL)
    g = panel[(panel.symbol == s) & (panel.year <= LAST_YEAR)].sort_values("entry_ts").reset_index(drop=True)
    b = pd.read_parquet(CLEAN_BARS_FMT.format(symbol=s), columns=["close", "contract", "session_date"]).sort_index()
    b.index = pd.to_datetime(b.index, utc=True)
    days = pd.DatetimeIndex(pd.to_datetime(b.session_date).unique()).sort_values()
    return g, Px(np.log(b.close)), Px(b.contract.astype(float)), days


def make_paths(g: pd.DataFrame, lpx: Px, ct: Px, length: int, kind: str):
    steps = np.arange(-length, 1) * 60_000_000_000
    t = g.entry_ts.values.astype("datetime64[ns]").astype(np.int64)[:, None] + steps
    stamps = pd.DatetimeIndex(t.ravel().astype("datetime64[ns]")).tz_localize("UTC")
    lp = lpx.at(stamps).reshape(len(g), -1)
    c = ct.at(stamps).reshape(len(g), -1)
    ok = np.isfinite(lp).all(1) & (c == c[:, [-1]]).all(1)
    r = np.diff(lp, axis=1)
    vol = r.std(1)
    if kind == "zprice":
        x = np.exp(lp)                 # price level, as features.py z-scores it
    elif kind == "zret":
        x = r
    else:
        x = (lp - lp[:, [0]]) * 1e4
    if kind != "bps":
        sd = x.std(1, keepdims=True)
        sd[sd < 1e-12] = 1.0
        x = (x - x.mean(1, keepdims=True)) / sd
    return x, ok, vol


def features(g: pd.DataFrame, x: np.ndarray, ok: np.ndarray, vol: np.ndarray, pool: str) -> np.ndarray:
    """[game, k, agg, (signal, disp)] for admitted games; NaN where not computed."""
    adm = g.admitted.to_numpy()
    out = np.full((len(g), len(KS), len(AGGS), 2), np.nan)
    key = ["clock_et", "anchor"] if pool == "clock" else ["family", "anchor"]
    rets = g.fwd_ret.to_numpy(float)
    t_all = g.entry_ts.values.astype("datetime64[ns]").astype(np.int64)
    for _, grp in g.groupby(key, sort=False):
        idx = grp.index.to_numpy()
        idx = idx[ok[idx]]
        qpos = np.where(adm[idx])[0]
        if len(qpos) == 0:
            continue
        t0, r, v = t_all[idx], rets[idx], vol[idx]
        paths = [np.asarray(p, dtype=np.double) for p in x[idx]]
        hi = int(qpos.max()) + 1
        D = dtw.distance_matrix_fast(paths[:hi], window=DTW_WARP_MINUTES, use_pruning=True,
                                     block=((0, hi), (int(qpos.min()), hi)), compact=False, parallel=True)
        for pos in qpos:
            age = (t0[pos] - t0[:pos]) / 86_400_000_000_000.0
            el = np.where((age > 0) & (age <= 365 * LOOKBACK))[0]
            d = D[pos, el]
            m = np.isfinite(d)
            el, d = el[m], d[m]
            if len(el) < MIN_NEIGHBOURS:
                continue
            order = el[np.argsort(d, kind="stable")]
            for ki, k in enumerate(KS):
                sel = order[:k]
                rr = r[sel]
                disp = rr.std()
                vv = np.where(v[sel] > 0, v[sel], np.nan)
                sig = {"mean": rr.mean(), "median": np.median(rr), "vote": (rr > 0).mean() - 0.5,
                       "volnorm": np.nanmean(rr / vv)}
                for ai, a in enumerate(AGGS):
                    out[idx[pos], ki, ai] = sig[a], disp
    return out


def score(s: str, g: pd.DataFrame, F: np.ndarray, days: pd.DatetimeIndex, variant: dict) -> list[dict]:
    keep = g.admitted.to_numpy() & ~g.entry_ts.duplicated().to_numpy()
    g, F = g[keep].reset_index(drop=True), F[keep]
    year = g.year.to_numpy()
    ret = g.fwd_ret.to_numpy()
    sess = (g.entry_ts.dt.tz_convert(NY) + SESSION_SHIFT).dt.normalize().dt.tz_localize(None)
    b = pd.read_parquet(CLEAN_BARS_FMT.format(symbol=s), columns=["close"])
    b.index = pd.to_datetime(b.index, utc=True)
    cost = 2 * COST_TICKS * ASSETS[s]["tick"] / b.close.reindex(g.entry_ts).to_numpy() * 1e4
    per = {}
    for name, (y0, y1) in PERIODS.items():
        dd = days[(days.year >= y0) & (days.year <= y1)]
        m = (year >= y0) & (year <= y1)
        per[name] = (m, dd.get_indexer(sess[m]), len(dd))
    rows = []
    for (ki, k), (ai, a) in itertools.product(enumerate(KS), enumerate(AGGS)):
        sig, disp = F[:, ki, ai, 0], F[:, ki, ai, 1]
        have = np.isfinite(sig) & np.isfinite(disp) & (disp > 0)
        row = dict(variant, symbol=s, k=k, agg=a, games=int(have.sum()))
        for name, (y0, y1) in IC_PERIODS.items():
            m = have & (year >= y0) & (year <= y1)
            row[f"IC {name}"] = stats.spearmanr(sig[m], ret[m])[0] if m.sum() > 30 else np.nan
        lo, hi, ref = (np.full(len(sig), np.nan) for _ in range(3))
        for q in GATE_QS:
            for y in sorted(set(year)):
                prior = have & (year < y)
                if prior.sum() >= 50:
                    lo[year == y], hi[year == y] = np.quantile(sig[prior], [q, 1 - q])
                    ref[year == y] = np.median(disp[prior])
            gate = have & np.isfinite(ref)
            if q < 0.5:
                gate &= (sig <= lo) | (sig >= hi)
            size = np.clip(ref / np.where(disp > 0, disp, np.nan), 0, MAX_POSITION)
            pos = np.nan_to_num(np.where(gate, np.sign(sig) * size, 0.0))
            net = pos * ret * 1e4 - np.abs(pos) * cost
            for name, (m, di, nd) in per.items():
                tr = m & (pos != 0)
                daily = np.bincount(di[pos[m] != 0], weights=net[tr], minlength=nd)
                row[f"q{q} {name}"] = daily.mean() / daily.std(ddof=1) * np.sqrt(252)
                row[f"q{q} {name} n"] = int(tr.sum())
        rows.append(row)
    return rows


if __name__ == "__main__":
    rows = []
    for s in BOOK_SYMBOLS:
        g, lpx, ct, days = load(s)
        for kind, length, pool in itertools.product(PATHS, LENGTHS, POOLS):
            t = time.time()
            x, ok, vol = make_paths(g, lpx, ct, length, kind)
            F = features(g, x, ok, vol, pool)
            variant = dict(path=kind, length=length, pool=pool)
            if variant == dict(path="zprice", length=30, pool="clock"):
                ref = F[:, KS.index(40), AGGS.index("mean"), 0]
                m = g.admitted.to_numpy()
                err = np.nanmax(np.abs(ref[m] - g.dtw_dir.to_numpy()[m]))
                print(f"  {s} check vs features.parquet dtw_dir at k=40, mean: max diff {err:.2e}")
            rows += score(s, g, F, days, variant)
            print(f"{s} {kind:6s} {length:2d}m {pool:6s} {time.time() - t:5.1f}s", flush=True)
    res = pd.DataFrame(rows)
    res.to_parquet(TUNE / "results2.parquet")
    print(f"wrote {TUNE / 'results2.parquet'}  {len(res):,} rows")
