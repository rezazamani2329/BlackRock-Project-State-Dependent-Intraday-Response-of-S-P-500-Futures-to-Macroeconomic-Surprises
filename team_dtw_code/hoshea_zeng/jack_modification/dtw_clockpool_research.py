"""DTW event games with clock-slot neighbour pooling — 2016-2021 select / 2022-2026 confirm.

Why this design. Hoshea's grid found `fomc_jolts_pce_claims` beat `core_major` on every
cell. Jack read that as pool density and retracted the PCE result when same-event matching
reversed it. Both are consistent with a rule neither tested: the right neighbour pool is
*the clock slot*, not the event type. PCE prints at 08:30 alongside Claims, CPI, NFP, PPI,
GDP, Retail Sales, Import/Export and ECI -- 1,573 releases since 2016 against PCE's 131.
Matching a PCE path against that library is what Hoshea's winning group did by accident.

So: same construction, but neighbours are every prior release sharing the clock slot and
the anchor offset, whatever the event. Hyperparameters are chosen on 2016-2021 and the
choice is confirmed on 2022-2026, per the requested split.

Run: uv run python jack/dtw_clockpool_research.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd
from dtaidistance import dtw
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PRICE = ROOT / "data/processed/es_1min_clean.parquet"
EVENTS = ROOT / "data/processed/macro_event_calendar_expanded_2010_2026.parquet"
OUT = ROOT / "jack/data"; OUT.mkdir(exist_ok=True)
NY = ZoneInfo("America/New_York")

OFFSETS = [-60, -30, 0, 30, 60]      # anchor T0 relative to release
PATH_MIN, HOLD_MIN = 60, 30          # path [T0,T0+60]; entry T0+60; exit T0+90
WARMUP_START = "2014-01-01"          # neighbours may reach back here; scoring starts 2016
SCORE_START, SPLIT, = 2016, 2022
WARP_MINUTES = 15                    # Sakoe-Chiba tolerance, converted per frequency
RECENCY_HL_DAYS = 365 * 2
GRID_FREQ = [3, 5]
GRID_LOOKBACK = [2, 3]
GRID_K = [15, 25]
MIN_HISTORY = 20
TICK_BPS = 0.5                       # one ES tick; round trip ~1 bp


def zscore(x):
    sd = float(np.std(x))
    return np.zeros_like(x) if sd < 1e-12 else (x - float(np.mean(x))) / sd


def build_games(close: pd.Series, events: pd.DataFrame, freq: int) -> pd.DataFrame:
    rows = []
    for row in events.itertuples(index=False):
        for off in OFFSETS:
            t0 = row.timestamp_utc + pd.Timedelta(minutes=off)
            idx = pd.date_range(t0, t0 + pd.Timedelta(minutes=PATH_MIN), freq=f"{freq}min", tz="UTC")
            px = close.reindex(idx)
            if px.isna().any():
                continue
            lr = np.diff(np.log(px.to_numpy(float)))
            if len(lr) == 0 or not np.isfinite(lr).all():
                continue
            entry = t0 + pd.Timedelta(minutes=PATH_MIN)
            lab = close.reindex([entry, entry + pd.Timedelta(minutes=HOLD_MIN)])
            if lab.isna().any():
                continue
            rows.append({"event_type": row.event_type, "event_ts": row.timestamp_utc,
                         "offset_min": off, "t0_utc": t0, "entry_utc": entry,
                         "clock_et": t0.tz_convert(NY).strftime("%H:%M"),
                         "fwd": float(np.log(lab.iloc[1]) - np.log(lab.iloc[0])),
                         "path": zscore(lr)})
    return pd.DataFrame(rows).sort_values("t0_utc").reset_index(drop=True)


def factors_for_freq(games: pd.DataFrame, freq: int) -> dict:
    """One distance matrix per (clock, offset) group; slice it for every (lookback, k)."""
    win = max(1, round(WARP_MINUTES / freq))
    res = {(lb, k): np.full(len(games), np.nan) for lb in GRID_LOOKBACK for k in GRID_K}
    nbr = np.full(len(games), np.nan)
    for (clock, off), grp in games.groupby(["clock_et", "offset_min"], sort=False):
        idx = grp.index.to_numpy()
        if len(idx) < MIN_HISTORY + 1:
            continue
        series = [np.asarray(p, dtype=np.double) for p in grp["path"]]
        D = dtw.distance_matrix_fast(series, window=win, compact=False)
        D = np.asarray(D)
        D[np.isinf(D)] = np.nan
        # symmetrise (distance_matrix_fast fills upper triangle only)
        D = np.where(np.isnan(D), D.T, D)
        t0 = grp["t0_utc"].values.astype("datetime64[ns]").astype(np.int64)
        rets = grp["fwd"].to_numpy(float)
        days = (t0[:, None] - t0[None, :]) / 86_400_000_000_000.0     # age of j seen from i, days
        for pos in range(len(idx)):
            age = days[pos]
            for lb in GRID_LOOKBACK:
                elig = np.where((age > 0) & (age <= 365 * lb) & np.isfinite(D[pos]))[0]
                if lb == GRID_LOOKBACK[0]:
                    nbr[idx[pos]] = len(elig)
                if len(elig) < MIN_HISTORY:
                    continue
                d = D[pos, elig]
                order = np.argsort(d)
                for k in GRID_K:
                    sel = elig[order[:min(k, len(elig))]]
                    w = (1.0 / (D[pos, sel] + 1e-6)) * np.exp(-age[sel] / RECENCY_HL_DAYS)
                    res[(lb, k)][idx[pos]] = float(np.dot(w, rets[sel]) / w.sum())
    return res, nbr


def score(sub: pd.DataFrame, fcol="dtw_factor") -> dict:
    s = sub.dropna(subset=[fcol, "fwd"])
    if len(s) < 40:
        return {}
    x, y = s[fcol].to_numpy(), s["fwd"].to_numpy() * 1e4
    sign_pnl = np.sign(x) * y
    sized = (x / np.std(x)) * y
    cl = pd.DataFrame({"p": sign_pnl, "c": s["event_ts"].dt.date}).groupby("c").p.mean()
    t = cl.mean() / (cl.std(ddof=1) / np.sqrt(len(cl))) if len(cl) > 2 else np.nan
    return dict(n=len(s), ic=np.corrcoef(x, y)[0, 1], sic=stats.spearmanr(x, y).correlation,
                hit=float((sign_pnl > 0).mean()), bps=float(sign_pnl.mean()),
                net=float(sign_pnl.mean() - 2 * TICK_BPS), t=float(t),
                sized=float(sized.mean()))


def main():
    close = pd.read_parquet(PRICE)["close"]
    close.index = pd.to_datetime(close.index, utc=True)
    close = close[~close.index.duplicated()].sort_index()
    ev = pd.read_parquet(EVENTS)
    ev = ev[ev.timestamp_utc >= pd.Timestamp(WARMUP_START, tz="UTC")].sort_values("timestamp_utc")
    print(f"{len(ev)} releases from {WARMUP_START}", flush=True)

    panels = []
    for freq in GRID_FREQ:
        g = build_games(close, ev, freq)
        print(f"freq {freq}min: {len(g)} games", flush=True)
        res, nbr = factors_for_freq(g, freq)
        g["n_nbr"] = nbr
        for (lb, k), vals in res.items():
            p = g.drop(columns=["path"]).copy()
            p["dtw_factor"] = vals
            p["freq"], p["lookback"], p["k"] = freq, lb, k
            panels.append(p)
        print(f"  done freq {freq}", flush=True)

    panel = pd.concat(panels, ignore_index=True)
    panel["yr"] = panel.t0_utc.dt.year
    panel = panel[panel.yr >= SCORE_START]
    panel.to_parquet(OUT / "dtw_clockpool_panel.parquet")

    # ---- grid: select on 2016-2021, confirm on 2022-2026. Exclude offset -60 (entry at
    # the release instant, i.e. trading the jump) from the headline; reported separately.
    rows = []
    for (f, lb, k), g in panel.groupby(["freq", "lookback", "k"]):
        gt = g[g.offset_min > -60]
        sel, con = score(gt[gt.yr < SPLIT]), score(gt[gt.yr >= SPLIT])
        if not sel or not con:
            continue
        rows.append(dict(freq=f, lookback=lb, k=k,
                         **{f"sel_{a}": b for a, b in sel.items()},
                         **{f"con_{a}": b for a, b in con.items()}))
    grid = pd.DataFrame(rows).sort_values("sel_ic", ascending=False)
    grid.to_csv(OUT / "dtw_clockpool_grid.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n===== GRID (offsets > -60), selected on 2016-2021 =====")
    print(grid[["freq", "lookback", "k", "sel_n", "sel_ic", "sel_bps", "sel_t",
                "con_n", "con_ic", "con_sic", "con_hit", "con_bps", "con_net",
                "con_t", "con_sized"]].round(4).to_string(index=False))
    if len(grid) > 1:
        print(f"\nrank corr select->confirm IC: {stats.spearmanr(grid.sel_ic, grid.con_ic).correlation:+.3f}")

    best = grid.iloc[0]
    print(f"\n===== BEST BY SELECTION: freq={best.freq} lookback={best.lookback} k={best.k} =====")
    b = panel[(panel.freq == best.freq) & (panel.lookback == best.lookback) & (panel.k == best.k)]
    for lab, sub in [("2016-2021", b[b.yr < SPLIT]), ("2022-2026", b[b.yr >= SPLIT])]:
        print(f"\n--- {lab} ---")
        for off in sorted(b.offset_min.unique()):
            s = score(sub[sub.offset_min == off])
            if s:
                print(f"  offset {off:+4d}  n={s['n']:5d} ic={s['ic']:+.4f} sic={s['sic']:+.4f} "
                      f"hit={s['hit']:.3f} bps={s['bps']:+6.2f} net={s['net']:+6.2f} t={s['t']:+.2f} sized={s['sized']:+6.2f}")
        tr = sub[sub.offset_min > -60]
        for ev_t in sorted(tr.event_type.unique()):
            s = score(tr[tr.event_type == ev_t])
            if s:
                print(f"  {ev_t[:24]:24s} n={s['n']:5d} ic={s['ic']:+.4f} sic={s['sic']:+.4f} "
                      f"hit={s['hit']:.3f} bps={s['bps']:+6.2f} net={s['net']:+6.2f} t={s['t']:+.2f} sized={s['sized']:+6.2f}")
    print("\nwrote dtw_clockpool_panel.parquet / dtw_clockpool_grid.csv")


if __name__ == "__main__":
    main()
