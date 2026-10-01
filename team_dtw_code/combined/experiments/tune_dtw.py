"""In-depth tune of the DTW features, gate and sizing. ES and NQ, 2013-2020 only.

Stage 1 recomputes dtw_dir / dtw_disp for every DTW setting from one set of distance
matrices per warping band, exactly as features.py does (checked against the panel):

    band (Sakoe-Chiba, minutes)  3, 5, 10
    lookback years               2, 3, 5, 8
    pool cap                     600 most recent (features.py), or none
    dedupe                       one neighbour per entry minute, or every family's copy
    k                            5, 10, 15, 25, 40, 60
    recency half-life (days)     365, 730, 1460, none
    weights                      1/distance (features.py), or uniform

Stage 2 trades every setting through a gate and a sizing rule, quarter tick per side,
positions one per entry minute, Sharpe on the daily session grid:

    gate variable   dtw_dir, or dtw_dir / dtw_disp
    gate q          0 (trade all), 0.1, 0.2, 0.3, 0.4 -- trade outside the [q, 1-q] quantiles
    thresholds      walk-forward: each year's cut-offs from all earlier admitted games
    sizing          flat, inverse dtw_disp (capped at 2), inverse sqrt dtw_disp (capped at 2)

Train is scored 2014-2017 (2013 has no earlier games for the thresholds), validation 2018-2020.
The test years are never loaded. Writes output/tune/ (gitignored).

Run: uv run python combined/experiments/tune_dtw.py

Findings (2026-09-27, quarter tick, inverse-dtw_disp sizing unless noted):
- The old ES book's settings (band 5, lookback 3, cap 600, k 15, half-life 730, 1/distance,
  outer-60% gate) score ES -0.76 / +0.62 and NQ -0.66 / +0.94 (train / valid Sharpe). 93% of
  all 138k rule settings lose on train, 73% win on valid, and the two correlate at only 0.15
  across settings: the direction signal's regime, not the tuning, sets the sign.
- Flipping dtw_dir's sign on its trailing record (100-1000 games) fixes train and breaks valid.
- dtw_disp tunes cleanly (train/valid rank of disp IC correlate at 0.67): large k, one neighbour
  per entry minute and a 1-year half-life lift disp IC from 0.19 / 0.13 to ~0.30 / 0.18. Using
  that better dispersion for sizing does not raise Sharpe; sizing moves it by about +-0.2.
- Only 16 of 2,304 DTW settings have positive dir IC on ES and NQ in both periods; all use an
  8-year lookback, no pool cap and no recency decay. Band 10, lookback 8, no cap, no decay,
  k 40, uniform weights: outer 60% gate ES +0.60 / +0.79, NQ 0.00 / +1.26; outer 80% gate
  ES +0.58 / +0.36, NQ +0.24 / +1.21, with ~30% more trades than the old book. ES is positive
  in 5 of 7 years (old book 3 of 7), NQ 5 of 6 (old book 3 of 6). Caveat: picked by looking at
  validation as well as train, so 2021-2026 is the only clean check left for it.
"""
from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dtaidistance import dtw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from config import (ASSETS, CLEAN_BARS_FMT, MAX_POSITION, MIN_NEIGHBOURS, NY, OUT, PANEL,
                    PATH_PRE_MIN, SESSION_SHIFT, VALID_YEARS_SPAN)
from util import Px

SYMBOLS = ["ES", "NQ"]
LAST_YEAR = VALID_YEARS_SPAN[1]
PERIODS = {"train": (2014, 2017), "valid": VALID_YEARS_SPAN}
COST_TICKS = 0.25
TUNE = OUT / "tune"
TUNE.mkdir(exist_ok=True)

BANDS = [3, 5, 10]
LOOKBACKS = [2, 3, 5, 8]
POOL_CAPS = [600, None]
DEDUPE = [False, True]
KS = [5, 10, 15, 25, 40, 60]
HALFLIVES = [365, 730, 1460, np.inf]
WEIGHTS = ["inv_dist", "uniform"]
CONTEXTS = list(itertools.product(LOOKBACKS, POOL_CAPS, DEDUPE))
INNER = list(itertools.product(KS, HALFLIVES, WEIGHTS))
BASELINE = dict(band=5, lookback=3, pool=600, dedupe=False, k=15, halflife=730, weights="inv_dist")


# ---------------------------------------------------------------- stage 1: features
def load(s: str):
    panel = pd.read_parquet(PANEL)
    g = panel[(panel.symbol == s) & (panel.year <= LAST_YEAR)].sort_values("entry_ts").reset_index(drop=True)
    b = pd.read_parquet(CLEAN_BARS_FMT.format(symbol=s), columns=["close"]).sort_index()
    b.index = pd.to_datetime(b.index, utc=True)
    px = Px(b.close)
    steps = np.arange(-PATH_PRE_MIN, 1) * 60_000_000_000
    grid = g.entry_ts.values.astype("datetime64[ns]").astype(np.int64)[:, None] + steps
    path = px.at(pd.DatetimeIndex(grid.ravel().astype("datetime64[ns]")).tz_localize("UTC")).reshape(len(g), -1)
    assert np.isfinite(path).all(), "a panel game has a missing path minute"
    sd = path.std(1, keepdims=True)
    sd[sd < 1e-12] = 1.0
    g["entry_px"] = path[:, -1]
    return g, (path - path.mean(1, keepdims=True)) / sd


def features(s: str) -> tuple[pd.DataFrame, np.ndarray, list]:
    """Returns the admitted games and an array [game, band, context, inner, (dir, disp)]."""
    g, z = load(s)
    q_idx = np.where(g.admitted.to_numpy())[0]
    out = np.full((len(q_idx), len(BANDS), len(CONTEXTS), len(INNER), 2), np.nan)
    row_of = {gi: r for r, gi in enumerate(q_idx)}
    for _, grp in g.groupby(["clock_et", "anchor"], sort=False):
        idx = grp.index.to_numpy()
        qpos = np.where(g.admitted.to_numpy()[idx])[0]
        if len(qpos) == 0:
            continue
        t0 = grp.entry_ts.values.astype("datetime64[ns]").astype(np.int64)
        rets = grp.fwd_ret.to_numpy(float)
        first = ~pd.Series(t0).duplicated().to_numpy()      # first copy of each entry minute
        paths = [np.asarray(p, dtype=np.double) for p in z[idx]]
        hi = int(qpos.max()) + 1
        for bi, band in enumerate(BANDS):
            D = dtw.distance_matrix_fast(paths[:hi], window=band, use_pruning=True,
                                         block=((0, hi), (int(qpos.min()), hi)), compact=False,
                                         parallel=True)
            for pos in qpos:
                age = (t0[pos] - t0[:pos]) / 86_400_000_000_000.0
                r = row_of[idx[pos]]
                for ci, (lb, cap, dd) in enumerate(CONTEXTS):
                    ok = (age > 0) & (age <= 365 * lb)
                    if dd:
                        ok &= first[:pos]
                    el = np.where(ok)[0]
                    if cap:
                        el = el[-cap:]
                    d = D[pos, el]
                    m = np.isfinite(d)
                    el, d = el[m], d[m]
                    if len(el) < MIN_NEIGHBOURS:
                        continue
                    order = np.argsort(d, kind="stable")
                    for ii, (k, hl, wt) in enumerate(INNER):
                        sel, dsel = el[order[:k]], d[order[:k]]
                        w = (1.0 / (dsel + 1e-6)) if wt == "inv_dist" else np.ones(len(sel))
                        if np.isfinite(hl):
                            w = w * np.exp(-age[sel] / hl)
                        w = w / w.sum()
                        rr = rets[sel]
                        f = float(np.dot(w, rr))
                        out[r, bi, ci, ii] = f, float(np.sqrt(np.dot(w, (rr - f) ** 2)))
    return g.iloc[q_idx].reset_index(drop=True), out


def setting_index(band, lookback, pool, dedupe, k, halflife, weights):
    return (BANDS.index(band), CONTEXTS.index((lookback, pool, dedupe)),
            INNER.index((k, halflife, weights)))


# ---------------------------------------------------------------- stage 2: trade
GATE_VARS = ["dir", "dir/disp"]
GATE_QS = [0.0, 0.1, 0.2, 0.3, 0.4]
SIZINGS = ["flat", "inv_disp", "inv_sqrt_disp"]


def trade_all(s: str, games: pd.DataFrame, F: np.ndarray) -> pd.DataFrame:
    # one row per entry minute: simultaneous families share the path, pool and so the features
    keep = ~games.entry_ts.duplicated().to_numpy()
    g = games[keep].reset_index(drop=True)
    F = F[keep]
    sess = (g.entry_ts.dt.tz_convert(NY) + SESSION_SHIFT).dt.normalize().dt.tz_localize(None)
    b = pd.read_parquet(CLEAN_BARS_FMT.format(symbol=s), columns=["session_date"])
    all_days = pd.DatetimeIndex(pd.to_datetime(b.session_date).unique()).sort_values()
    year = g.year.to_numpy()
    ret_bps = g.fwd_ret.to_numpy() * 1e4
    cost_bps = 2 * COST_TICKS * ASSETS[s]["tick"] / g.entry_px.to_numpy() * 1e4
    per = {}
    for name, (y0, y1) in PERIODS.items():
        days = all_days[(all_days.year >= y0) & (all_days.year <= y1)]
        m = (year >= y0) & (year <= y1)
        per[name] = (m, days.get_indexer(sess[m]), len(days))
    years = sorted(set(year))
    rows = []
    n_b, n_c, n_i = F.shape[1:4]
    for bi, ci, ii in itertools.product(range(n_b), range(n_c), range(n_i)):
        d, disp = F[:, bi, ci, ii, 0], F[:, bi, ci, ii, 1]
        have = np.isfinite(d) & np.isfinite(disp) & (disp > 0)
        base = {"band": BANDS[bi], "lookback": CONTEXTS[ci][0], "pool": CONTEXTS[ci][1] or 0,
                "dedupe": CONTEXTS[ci][2], "k": INNER[ii][0], "halflife": INNER[ii][1],
                "weights": INNER[ii][2]}
        for gv in GATE_VARS:
            x = d if gv == "dir" else d / np.where(disp > 0, disp, np.nan)
            for q in GATE_QS:
                gate = have.copy()
                if q > 0:
                    lo, hi_ = np.full(len(x), np.nan), np.full(len(x), np.nan)
                    for y in years:              # cut-offs from every earlier year
                        prior = have & (year < y)
                        if prior.sum() < 50:
                            continue
                        lo[year == y], hi_[year == y] = np.nanquantile(x[prior], [q, 1 - q])
                    gate &= (x <= lo) | (x >= hi_)
                for sz in SIZINGS:
                    if sz == "flat":
                        size = np.ones(len(x))
                    else:
                        ref = np.full(len(x), np.nan)
                        for y in years:
                            prior = have & (year < y)
                            if prior.sum() >= 50:
                                ref[year == y] = np.median(disp[prior])
                        ratio = ref / disp if sz == "inv_disp" else np.sqrt(ref / disp)
                        size = np.clip(ratio, 0, MAX_POSITION)
                    pos = np.where(gate, np.sign(x) * size, 0.0)
                    pos = np.nan_to_num(pos)
                    net = pos * ret_bps - np.abs(pos) * cost_bps
                    row = dict(base, gate_var=gv, q=q, sizing=sz)
                    for name, (m, di, nd) in per.items():
                        tr = m & (pos != 0)
                        daily = np.bincount(di[pos[m] != 0], weights=net[tr], minlength=nd)
                        sdv = daily.std(ddof=1)
                        row[f"{name} sharpe"] = daily.mean() / sdv * np.sqrt(252) if sdv else np.nan
                        row[f"{name} trades"] = int(tr.sum())
                        row[f"{name} bps/trade"] = net[tr].mean() if tr.any() else np.nan
                    rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    for s in SYMBOLS:
        t = time.time()
        feat_path = TUNE / f"features_{s}.npz"
        if feat_path.exists():
            z = np.load(feat_path, allow_pickle=True)
            games, F = pd.read_parquet(TUNE / f"games_{s}.parquet"), z["F"]
        else:
            games, F = features(s)
            np.savez_compressed(feat_path, F=F)
            games.to_parquet(TUNE / f"games_{s}.parquet")
        # check: the baseline cell of features.py's chosen setting reproduces the panel
        bi, ci, ii = setting_index(5, 3, 600, False, 10, 730, "inv_dist")
        err = np.nanmax(np.abs(F[:, bi, ci, ii, 0] - games.dtw_dir.to_numpy()))
        print(f"{s}: {len(games):,} admitted games, features in {time.time() - t:.0f}s, "
              f"max |dtw_dir - panel| at k=10, lb=3 = {err:.2e}")
        t = time.time()
        res = trade_all(s, games, F)
        res.to_parquet(TUNE / f"results_{s}.parquet")
        print(f"{s}: {len(res):,} rule settings traded in {time.time() - t:.0f}s")
