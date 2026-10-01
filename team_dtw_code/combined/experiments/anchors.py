"""Hoshea's multi-anchor structure, applied to the impact-screened book.

The book trades one window per release: enter T+1, exit T+31. Hoshea's DTW event game
instead placed several anchors around each release and scored a 30-minute forward window
at each. That is the single change most likely to fix this book's real weakness, which is
not edge per trade (ours is 2.5x the scanner's) but **trade count** -- 57/year against the
scanner's 987, and Sharpe scales with sqrt(count).

Design, generalising Hoshea's offsets into a non-overlapping ladder:

    entry at T+1, T+31, T+61, T+91, T+121   (five 30-minute windows, back to back)
    path  = the 31 minutes of z-scored PRICE ending exactly at entry
    target= log(P(entry+30) / P(entry))
    pool  = every prior release at the same (release clock slot, anchor), any family

Matching is conditioned on the anchor as well as the clock, per Hoshea's `offset_min`:
the reaction at T+1 is a structurally different object from the drift at T+121, and
pooling them would blur both.

T+1 is the existing book. Anchors 2-5 are new and carry no news content -- Jack's sweep
shows the surprise is spent by T+30 -- so DTW has to carry them alone. That is the test.

Run: uv run python combined/experiments/anchors.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dtaidistance import dtw
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # combined/
from config import (
    CLEAN_BARS, COST_BPS_PER_SIDE, HOLD_MIN, K_NEIGHBOURS, LOOKBACK_YEARS, MAX_POOL,
    MIN_NEIGHBOURS, NY, OUT, PANEL, PATH_PRE_MIN, RECENCY_HALFLIFE_DAYS, TRAIN_YEARS,
    WARP_MINUTES,
)
from util import Px

pd.set_option("display.width", 200)

ANCHORS = [1, 31, 61, 91, 121]          # entry offsets from the release, minutes
PATH_LEN = PATH_PRE_MIN + 1             # 31 prices ending at entry


def build(px: Px, rel: pd.DataFrame) -> pd.DataFrame:
    """One row per (family, release, anchor). Paths end exactly at entry."""
    recs = []
    steps = np.arange(-PATH_PRE_MIN, 1)
    for (fam, clock), g in rel.groupby(["family", "clock_et"], sort=False):
        T = pd.DatetimeIndex(sorted(g.event_ts))
        for a in ANCHORS:
            entry = T + pd.Timedelta(minutes=a)
            grid = entry.values[:, None] + (steps * 60_000_000_000).astype("timedelta64[ns]")
            path = px.at(pd.DatetimeIndex(grid.ravel()).tz_localize("UTC")).reshape(len(T), PATH_LEN)
            p0 = px.at(entry)
            p1 = px.at(entry + pd.Timedelta(minutes=HOLD_MIN))
            ok = np.isfinite(path).all(1) & np.isfinite(p0) & np.isfinite(p1)
            sd = path.std(1, keepdims=True)
            sd[sd < 1e-12] = 1.0
            z = (path - path.mean(1, keepdims=True)) / sd
            for i in np.where(ok)[0]:
                recs.append(dict(family=fam, event_ts=T[i], clock_et=clock, anchor=a,
                                 entry_ts=entry[i], year=int(T[i].year),
                                 fwd_ret=float(np.log(p1[i] / p0[i])), path=z[i]))
    return pd.DataFrame(recs).sort_values("entry_ts").reset_index(drop=True)


def dtw_factor(games: pd.DataFrame, query: np.ndarray) -> pd.DataFrame:
    """Distances only for the games we might trade; the pool is everything."""
    win = max(1, round(WARP_MINUTES / 1))
    out = {k: np.full(len(games), np.nan) for k in ("dtw_dir", "dtw_agree", "dtw_dist")}
    for _, grp in games.groupby(["clock_et", "anchor"], sort=False):
        idx = grp.index.to_numpy()
        t0 = grp.entry_ts.values.astype("datetime64[ns]").astype(np.int64)
        rets = grp.fwd_ret.to_numpy(float)
        paths = [np.asarray(p, dtype=np.double) for p in grp.path]
        want = np.where(query[idx])[0]
        for pos in want:
            age = (t0[pos] - t0[:pos]) / 86_400_000_000_000.0
            el = np.where((age > 0) & (age <= 365 * LOOKBACK_YEARS))[0]
            if len(el) < MIN_NEIGHBOURS:
                continue
            el = el[-MAX_POOL:]
            d = np.array([dtw.distance_fast(paths[pos], paths[j], window=win, use_pruning=True)
                          for j in el])
            m = np.isfinite(d)
            el, d = el[m], d[m]
            if len(el) < MIN_NEIGHBOURS:
                continue
            o = np.argsort(d)[:K_NEIGHBOURS]
            sel, dsel = el[o], d[o]
            a = (t0[pos] - t0[sel]) / 86_400_000_000_000.0
            w = (1.0 / (dsel + 1e-6)) * np.exp(-a / RECENCY_HALFLIFE_DAYS)
            w /= w.sum()
            r = rets[sel]
            f = float(np.dot(w, r))
            i = idx[pos]
            out["dtw_dir"][i] = f
            out["dtw_agree"][i] = float((np.sign(r) == np.sign(f)).mean())
            out["dtw_dist"][i] = float(dsel.mean())
    return pd.DataFrame(out, index=games.index)


def score(g: pd.DataFrame, pos: np.ndarray, label: str, years: float) -> dict:
    gross = pos * g.fwd_ret.to_numpy() * 1e4
    net = gross - np.abs(pos) * 2 * COST_BPS_PER_SIDE
    cl = pd.Series(net).groupby(g.event_ts.dt.date.to_numpy()).mean()
    t = cl.mean() / (cl.std(ddof=1) / np.sqrt(len(cl))) if len(cl) > 2 else np.nan
    sd = net.std(ddof=1)
    return dict(book=label, n=len(g), per_yr=round(len(g) / years),
                ic=np.corrcoef(g.dtw_dir, g.fwd_ret)[0, 1],
                hit=float((gross > 0).mean()), gross=gross.mean(), net=net.mean(),
                sharpe=(net.mean() / sd) * np.sqrt(len(g) / years) if sd else np.nan, t=t)


def main() -> None:
    close = pd.read_parquet(CLEAN_BARS)["close"]
    close.index = pd.to_datetime(close.index, utc=True)
    close = close[~close.index.duplicated()].sort_index()
    px = Px(close)

    panel = pd.read_parquet(PANEL)
    rel = panel[["family", "event_ts", "clock_et", "admitted", "year"]].drop_duplicates()
    print(f"{len(rel):,} releases, {rel.admitted.sum():,} admitted")

    games = build(px, rel)
    adm = rel[rel.admitted].set_index(["family", "event_ts"]).index
    games["admitted"] = pd.MultiIndex.from_frame(games[["family", "event_ts"]]).isin(adm)
    print(f"{len(games):,} anchor-games ({len(ANCHORS)} anchors), "
          f"{int(games.admitted.sum()):,} tradeable", flush=True)

    print("computing DTW (admitted queries only, full pool) ...", flush=True)
    games = pd.concat([games, dtw_factor(games, games.admitted.to_numpy())], axis=1)
    games = games.drop(columns=["path"])
    games.to_parquet(OUT / "anchor_panel.parquet")

    tr = games[games.admitted & games.dtw_dir.notna()].copy()
    first_oos = tr.year.min() + TRAIN_YEARS
    oos = tr[tr.year >= first_oos]
    yrs = oos.year.nunique()
    print(f"\nout-of-sample from {first_oos}: {len(oos):,} anchor-games over {yrs} years")

    print("\n===== PER ANCHOR (sign of dtw_dir, net of half a tick per side) =====")
    rows = [score(g, np.sign(g.dtw_dir), f"T+{a} only", yrs)
            for a, g in oos.groupby("anchor")]
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\n===== CUMULATIVE — does adding anchors help? =====")
    rows = []
    for k in range(1, len(ANCHORS) + 1):
        sub = oos[oos.anchor.isin(ANCHORS[:k])]
        rows.append(score(sub, np.sign(sub.dtw_dir),
                          f"T+1..T+{ANCHORS[k - 1]} ({k} anchors)", yrs))
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\n===== per year, all anchors =====")
    allg = oos.copy()
    allg["net"] = np.sign(allg.dtw_dir) * allg.fwd_ret * 1e4 - 2 * COST_BPS_PER_SIDE
    print(allg.groupby("year").net.agg(n="size", total="sum", per_trade="mean").round(2).to_string())

    print("\n===== null: random signs, all anchors =====")
    rng = np.random.default_rng(0)
    real = allg.net.mean()
    draws = np.array([(rng.choice([-1.0, 1.0], len(allg)) * allg.fwd_ret.to_numpy() * 1e4
                       - 2 * COST_BPS_PER_SIDE).mean() for _ in range(3000)])
    print(f"real {real:+.3f} bps | null {draws.mean():+.3f} sd {draws.std():.3f} | "
          f"P(null >= real) = {(draws >= real).mean():.4f}")
    print("\nwrote anchor_panel.parquet")


if __name__ == "__main__":
    main()
