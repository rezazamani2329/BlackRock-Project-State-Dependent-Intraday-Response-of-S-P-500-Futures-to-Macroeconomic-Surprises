"""Decouple what we TRADE from what we MATCH AGAINST.

The impact screen admits ten high-impact families, and the DTW factor built on that
universe is flat: best-by-training cell confirms at +0.07 bps net, t = 0.27. The reason
is visible in the construction. High-impact releases are monthly, so admitting only them
leaves ~430 paths in the 08:30 library. Hoshea's winning configuration had ~1,800,
because weekly Initial Jobless Claims was in his pool -- and his own per-event table
shows Claims contributes no signal of its own. It was never a tradeable event. It was
the library.

So: trade the admitted ten, but match against *every* US release at the same clock slot
and offset, high impact or not. Neighbours contribute their forward return whether or not
we would ever trade them. Pool density and trading impact become separate choices.

Run: uv run python jack/dtw_dense_pool.py
"""
from __future__ import annotations
import json
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd
from dtaidistance import dtw
from scipy import stats

import importlib.util
spec = importlib.util.spec_from_file_location("ib", Path(__file__).with_name("dtw_impact_book.py"))
ib = importlib.util.module_from_spec(spec); spec.loader.exec_module(ib)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "jack/data"
NY = ZoneInfo("America/New_York")
TRAIN_START, SPLIT = 2016, 2022
TRADE_OFFSETS = [-30, 0, 30, 60]
MAX_POOL = 600                      # most recent candidates inside the lookback
GRID_FREQ, GRID_LOOKBACK, GRID_K = [3, 5], [2, 3], [15, 25]
MIN_HISTORY, WARP_MINUTES, RECENCY_HL_DAYS, TICK_BPS = 20, 15, 730, 0.5


def main():
    close = pd.read_parquet(ib.PRICE)["close"]
    close.index = pd.to_datetime(close.index, utc=True)
    close = close[~close.index.duplicated()].sort_index()
    px = ib.Px(close)

    lines = ib.load_lines()
    fam = ib.assign_families(lines)
    lines["family"] = lines.line.map(fam)
    lines = lines.dropna(subset=["family"])
    fam_stamps = {f: sorted(set(g.ts)) for f, g in lines.groupby("family")}

    screen = ib.impact_screen(px, fam_stamps)
    admitted = set(screen.head(ib.TOP_K).family)
    print(f"{len(fam_stamps)} families in the pool; {len(admitted)} admitted for trading")
    all_fams = pd.DataFrame({"family": list(fam_stamps)})
    news = ib.news_panel(lines, fam)

    panels = []
    for freq in GRID_FREQ:
        g = ib.build_games(px, all_fams, fam_stamps, freq)
        g["admitted"] = g.family.isin(admitted)
        print(f"freq {freq}: {len(g)} pool games, {int(g.admitted.sum())} tradeable", flush=True)
        res = {(lb, k): np.full(len(g), np.nan) for lb in GRID_LOOKBACK for k in GRID_K}
        win = max(1, round(WARP_MINUTES / freq))
        maxlb = max(GRID_LOOKBACK)
        for _, grp in g.groupby(["clock_et", "offset_min"], sort=False):
            gi = grp.index.to_numpy()
            t0 = grp["t0_utc"].values.astype("datetime64[ns]").astype(np.int64)
            rets = grp["fwd"].to_numpy(float)
            paths = [np.asarray(p, dtype=np.double) for p in grp["path"]]
            adm = grp["admitted"].to_numpy()
            for pos in np.where(adm)[0]:
                age = (t0[pos] - t0[:pos]) / 86_400_000_000_000.0
                el = np.where((age > 0) & (age <= 365 * maxlb))[0]
                if len(el) < MIN_HISTORY:
                    continue
                el = el[-MAX_POOL:]
                d = np.array([dtw.distance_fast(paths[pos], paths[j], window=win,
                                                use_pruning=True) for j in el])
                ok = np.isfinite(d)
                el, d = el[ok], d[ok]
                if len(el) < MIN_HISTORY:
                    continue
                a = (t0[pos] - t0[el]) / 86_400_000_000_000.0
                order = np.argsort(d)
                for lb in GRID_LOOKBACK:
                    m = order[a[order] <= 365 * lb]
                    if len(m) < MIN_HISTORY:
                        continue
                    for k in GRID_K:
                        s = m[:min(k, len(m))]
                        w = (1.0 / (d[s] + 1e-6)) * np.exp(-a[s] / RECENCY_HL_DAYS)
                        res[(lb, k)][gi[pos]] = float(np.dot(w, rets[s]) / w.sum())
        base = g.drop(columns=["path"])
        for (lb, k), v in res.items():
            p = base.copy(); p["dtw_factor"] = v
            p["freq"], p["lookback"], p["k"] = freq, lb, k
            panels.append(p[p.admitted])
        print(f"  freq {freq} done", flush=True)

    panel = pd.concat(panels, ignore_index=True)
    panel["yr"] = panel.t0_utc.dt.year
    panel = panel[(panel.yr >= TRAIN_START) & panel.dtw_factor.notna()]
    panel = panel.merge(news.rename(columns={"ts": "event_ts"}), on=["family", "event_ts"], how="left")
    panel.to_parquet(OUT / "dtw_dense_panel.parquet")
    print(f"panel {len(panel)} rows")

    trade = panel[panel.offset_min.isin(TRADE_OFFSETS)]
    rows = []
    for (f, lb, k), gg in trade.groupby(["freq", "lookback", "k"]):
        s, c = ib.score(gg[gg.yr < SPLIT], "dtw_factor"), ib.score(gg[gg.yr >= SPLIT], "dtw_factor")
        rows.append(dict(freq=f, lookback=lb, k=k, sel_n=s["n"], sel_ic=s["ic"], sel_gross=s["gross"],
                         sel_t=s["t"], con_n=c["n"], con_ic=c["ic"], con_hit=c["hit"],
                         con_gross=c["gross"], con_net=c["net"], con_t=c["t"]))
    grid = pd.DataFrame(rows).sort_values("sel_ic", ascending=False)
    grid.to_csv(OUT / "dtw_dense_grid.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n===== DENSE-POOL GRID (selected on 2016-2021) =====")
    print(grid.round(4).to_string(index=False))
    print(f"rank corr select->confirm IC: {stats.spearmanr(grid.sel_ic, grid.con_ic).correlation:+.3f}")

    best = grid.iloc[0]
    b = trade[(trade.freq == best.freq) & (trade.lookback == best.lookback) & (trade.k == best.k)]
    print(f"\nBEST CELL freq={int(best.freq)} lb={int(best.lookback)} k={int(best.k)} — per family & offset")
    for lab, sub in [("train", b[b.yr < SPLIT]), ("TEST ", b[b.yr >= SPLIT])]:
        for key in ["family", "offset_min"]:
            for v, gg in sub.groupby(key):
                if len(gg) < 40:
                    continue
                s = ib.score(gg, "dtw_factor")
                print(f"  {lab} {str(v)[:34]:34s} n={s['n']:4d} ic={s['ic']:+.4f} hit={s['hit']:.3f} "
                      f"gross={s['gross']:+6.2f} net={s['net']:+6.2f} t={s['t']:+.2f}")
    print("\nwrote dtw_dense_panel.parquet / dtw_dense_grid.csv")


if __name__ == "__main__":
    main()
