"""Local tune around the original ES book's setting, gate fixed. ES and NQ.

The gate is fixed at backtest.py's rule: trade the outer 60% of dtw_dir (q = 0.3), cut-offs
from 2013-2020 admitted games. Everything else moves in small steps around config.py:

    band 3, 4, 5, 6, 7            lookback 2, 3, 4 years       pool cap 400, 600, 800, none
    dedupe no / yes               k 10, 12, 15, 18, 20, 25     half-life 365, 730, 1095, 1460 days
    weights 1/distance, uniform   sizing flat, inverse dtw_disp, inverse sqrt dtw_disp

Scored like backtest.py: quarter tick per side, one position per entry minute, Sharpe on the
daily session grid, per period (train 2013-2017, valid 2018-2020, fit 2013-2020, test
2021-2026). Selection uses only the 2013-2020 columns: the pooled ES + NQ Sharpe on 2013-2020,
averaged over the setting and its one-step neighbours so a lone spike does not win. The test
column is read once, for the chosen setting and the baseline, by `--report`.

Run: uv run python combined/experiments/tune_local.py            # features + results
     uv run python combined/experiments/tune_local.py --report   # choose on 2013-2020, then test

Findings (2026-09-27): the pick on 2013-2020 (band 5, lookback 3, no cap, k 10, half-life 365,
1/distance, inverse dtw_disp) scores pooled 0.51 on 2013-2020 against the baseline's 0.11, but
on 2021-2026 it is ES 0.22 / NQ 0.19 against the baseline's ES 0.61 / NQ 0.11. Config was not
changed. Across all 17,280 settings the 2013-2020 score does carry to 2021-2026 on ES (Spearman
0.46; top decile averages ES 0.47 on test, bottom decile −0.04) but not on NQ (0.03). So on ES
the ranking is informative on average while the single best setting is noise; on NQ tuning
has nothing to find. Smaller k, a shorter half-life, band 5-7 and 1/distance weights score
better in-sample; deduping neighbours scores worse.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import tune_dtw as T
from config import (ASSETS, BOOK_SYMBOLS, CLEAN_BARS_FMT, MAX_POSITION, NY, SESSION_SHIFT,
                    TEST_YEARS_SPAN, TRAIN_YEARS_SPAN, VALID_YEARS_SPAN)

T.LAST_YEAR = TEST_YEARS_SPAN[1]
T.BANDS = [3, 4, 5, 6, 7]
T.CONTEXTS = list(itertools.product([2, 3, 4], [400, 600, 800, None], [False, True]))
T.INNER = list(itertools.product([10, 12, 15, 18, 20, 25], [365, 730, 1095, 1460],
                                 ["inv_dist", "uniform"]))
GATE_Q = 0.3
FIT = (TRAIN_YEARS_SPAN[0], VALID_YEARS_SPAN[1])
PERIODS = {"train": TRAIN_YEARS_SPAN, "valid": VALID_YEARS_SPAN, "fit": FIT, "test": TEST_YEARS_SPAN}
SIZINGS = ["flat", "inv_disp", "inv_sqrt_disp"]
BASELINE = dict(band=5, lookback=3, pool=600, dedupe=False, k=15, halflife=730,
                weights="inv_dist", sizing="inv_disp")
DIMS = {"band": T.BANDS, "lookback": [2, 3, 4], "pool": [400, 600, 800, 0], "dedupe": [False, True],
        "k": [10, 12, 15, 18, 20, 25], "halflife": [365, 730, 1095, 1460],
        "weights": ["inv_dist", "uniform"], "sizing": SIZINGS}


def trade(s: str, games: pd.DataFrame, F: np.ndarray) -> pd.DataFrame:
    keep = ~games.entry_ts.duplicated().to_numpy()
    g, F = games[keep].reset_index(drop=True), F[keep]
    b = pd.read_parquet(CLEAN_BARS_FMT.format(symbol=s), columns=["session_date"])
    all_days = pd.DatetimeIndex(pd.to_datetime(b.session_date).unique()).sort_values()
    sess = (g.entry_ts.dt.tz_convert(NY) + SESSION_SHIFT).dt.normalize().dt.tz_localize(None)
    year = g.year.to_numpy()
    ret = g.fwd_ret.to_numpy() * 1e4
    cost = 2 * T.COST_TICKS * ASSETS[s]["tick"] / g.entry_px.to_numpy() * 1e4
    fit = (year >= FIT[0]) & (year <= FIT[1])
    per = {}
    for name, (y0, y1) in PERIODS.items():
        days = all_days[(all_days.year >= y0) & (all_days.year <= y1)]
        m = (year >= y0) & (year <= y1)
        per[name] = (m, days.get_indexer(sess[m]), len(days))
    rows = []
    for bi, ci, ii in itertools.product(range(len(T.BANDS)), range(len(T.CONTEXTS)), range(len(T.INNER))):
        d, disp = F[:, bi, ci, ii, 0], F[:, bi, ci, ii, 1]
        have = np.isfinite(d) & np.isfinite(disp) & (disp > 0)
        lo, hi = np.quantile(d[have & fit], [GATE_Q, 1 - GATE_Q])
        ref = np.median(disp[have & fit])
        gate = have & ((d <= lo) | (d >= hi))
        lb, cap, dd = T.CONTEXTS[ci]
        k, hl, wt = T.INNER[ii]
        for sz in SIZINGS:
            ratio = ref / np.where(have, disp, np.nan)
            size = {"flat": np.ones(len(d)), "inv_disp": np.clip(ratio, 0, MAX_POSITION),
                    "inv_sqrt_disp": np.clip(np.sqrt(ratio), 0, MAX_POSITION)}[sz]
            pos = np.nan_to_num(np.where(gate, np.sign(d) * size, 0.0))
            net = pos * ret - np.abs(pos) * cost
            row = dict(symbol=s, band=T.BANDS[bi], lookback=lb, pool=cap or 0, dedupe=dd, k=k,
                       halflife=hl, weights=wt, sizing=sz)
            for name, (m, di, nd) in per.items():
                tr = m & (pos != 0)
                daily = np.bincount(di[pos[m] != 0], weights=net[tr], minlength=nd)
                row[name] = daily.mean() / daily.std(ddof=1) * np.sqrt(252)
                row[f"{name} n"] = int(tr.sum())
            rows.append(row)
    return pd.DataFrame(rows)


def report() -> None:
    r = pd.read_parquet(T.TUNE / "results_local.parquet")
    keys = list(DIMS)
    w = r.pivot_table(index=keys, columns="symbol", values=["train", "valid", "fit", "test"])
    w.columns = [f"{a} {b}" for a, b in w.columns]
    w["fit pooled"] = w[[f"fit {s}" for s in BOOK_SYMBOLS]].mean(axis=1)
    # neighbourhood score: the setting and every setting one step away in one dimension
    score = w["fit pooled"]
    idx = score.index
    pos = {k: {v: i for i, v in enumerate(vals)} for k, vals in DIMS.items()}
    lookup = score.to_dict()
    nb = []
    for key in idx:
        vals = [lookup[key]]
        for di, k in enumerate(keys):
            i = pos[k][key[di]]
            for j in (i - 1, i + 1):
                if 0 <= j < len(DIMS[k]):
                    alt = list(key)
                    alt[di] = DIMS[k][j]
                    vals.append(lookup.get(tuple(alt), np.nan))
        nb.append(np.nanmean(vals))
    w["neighbourhood"] = nb
    base_key = tuple(BASELINE[k] for k in keys)
    fitcols = [f"{p} {s}" for s in BOOK_SYMBOLS for p in ("train", "valid", "fit")]
    print("How the fit-period Sharpe (pooled ES + NQ, 2013-2020) moves with each dimension:")
    for k in keys:
        print(f"\n{k}:\n" + w.groupby(level=k)[fitcols + ["fit pooled"]].mean().round(3).to_string())
    top = w.sort_values("neighbourhood", ascending=False).head(10)
    print("\nTop 10 by neighbourhood score (2013-2020 only):")
    print(top[fitcols + ["fit pooled", "neighbourhood"]].round(3).to_string())
    chosen = top.index[0]
    print("\nbaseline rank by neighbourhood:", int((w["neighbourhood"] > w.loc[base_key, "neighbourhood"]).sum()) + 1,
          "of", len(w))
    out = w.loc[[base_key, chosen], fitcols + ["fit pooled", "neighbourhood"]
                + [f"test {s}" for s in BOOK_SYMBOLS]]
    out.index = ["baseline", "chosen"]
    print("\nBaseline vs chosen -- test 2021-2026 read once:")
    print(out.round(3).T.to_string())
    print("\nchosen:", dict(zip(keys, chosen)))


if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
        sys.exit()
    res = []
    for s in BOOK_SYMBOLS:
        games, F = T.features(s)
        bi, ci, ii = T.setting_index(5, 3, 600, False, 15, 730, "inv_dist")
        err = np.nanmax(np.abs(F[:, bi, ci, ii, 0] - games.dtw_dir.to_numpy()))
        print(f"{s}: {len(games):,} admitted games; baseline vs features.parquet max diff {err:.2e}", flush=True)
        res.append(trade(s, games, F))
        del F
    pd.concat(res).to_parquet(T.TUNE / "results_local.parquet")
    print("wrote results_local.parquet")
