"""DTW on the reaction path — match the print itself, enter at T+1.

Every DTW design in this project so far ends its path at T0+60 and opens the position
there. For the anchors that matter that is 30 to 120 minutes after the release, and
Jack's entry sweep shows the news drift is spent by T+30 (the next clean bar is t = +0.72).
So the existing games enter after the tradeable window has closed, which is why blending
a news surprise into them adds nothing at any weight.

This design moves the path to where the information is:

    path   [T-PRE, T+1]  -- the pre-release hour PLUS the first minute of the reaction
    entry  T+1           -- executable; the same lag that survives Jack's sweep
    exit   T+31          -- a 30-minute hold

The path now contains the announcement reaction itself, so the matcher is asked a
different and much better question: given the market jumped like *this* off a release at
this clock slot, what happened over the next half hour? That is the tape's own version of
a surprise, it needs no Bloomberg consensus, and it is available one minute after the
print. Neighbours are drawn from every US release at the same clock slot (the dense pool),
since pool density is what made Hoshea's configuration work.

Two normalisations are tested because the earlier work turned on exactly this choice:
  zscore   per-path z, shape only -- what every directional design used, all of which failed
  volscale divided by prevailing volatility, so the size of the jump survives -- dropping
           the z-normalisation is what made Jack's magnitude DTW feature work

Run: uv run python jack/dtw_reaction_path.py
"""
from __future__ import annotations
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
ENTRY_LAG, HOLD = 1, 30
PRE_GRID = [30, 60]
NORMS = ["zscore", "volscale"]
FREQ = 1
MAX_POOL, MIN_HISTORY = 600, 20
GRID_LOOKBACK, GRID_K = [2, 3], [15, 25]
WARP_MIN, RECENCY_HL, TICK = 5, 730, 0.5


def main():
    close = pd.read_parquet(ib.PRICE)["close"]
    close.index = pd.to_datetime(close.index, utc=True)
    close = close[~close.index.duplicated()].sort_index()
    px = ib.Px(close)
    # prevailing volatility: stdev of 1-min log returns over the 4 hours before the path
    lr = np.log(close).diff()
    vol4h = lr.rolling(240, min_periods=60).std()
    vpx = ib.Px(vol4h.dropna())

    lines = ib.load_lines()
    fam = ib.assign_families(lines)
    lines["family"] = lines.line.map(fam)
    lines = lines.dropna(subset=["family"])
    fam_stamps = {f: sorted(set(g.ts)) for f, g in lines.groupby("family")}
    screen = ib.impact_screen(px, fam_stamps)
    admitted = set(screen.head(ib.TOP_K).family)
    news = ib.news_panel(lines, fam)
    print(f"{len(fam_stamps)} families pooled, {len(admitted)} admitted", flush=True)

    out_rows = []
    panels = []
    for pre in PRE_GRID:
        steps = np.arange(-pre, ENTRY_LAG + 1, FREQ)
        recs = []
        for f, ts in fam_stamps.items():
            T = pd.DatetimeIndex(sorted(ts))
            grid = T.values[:, None] + (steps * 60_000_000_000).astype("timedelta64[ns]")
            pth = px.at(pd.DatetimeIndex(grid.ravel()).tz_localize("UTC")).reshape(len(T), len(steps))
            ent = px.at(T + pd.Timedelta(minutes=ENTRY_LAG))
            ext = px.at(T + pd.Timedelta(minutes=ENTRY_LAG + HOLD))
            v = vpx.at(T - pd.Timedelta(minutes=pre))
            ok = np.isfinite(pth).all(1) & np.isfinite(ent) & np.isfinite(ext) & np.isfinite(v) & (v > 0)
            r = np.diff(np.log(pth), axis=1)
            sd = r.std(1, keepdims=True); sd[sd < 1e-12] = 1
            z = (r - r.mean(1, keepdims=True)) / sd
            vs = r / v[:, None]
            for i in np.where(ok)[0]:
                recs.append(dict(family=f, event_ts=T[i], clock_et=T[i].tz_convert(NY).strftime("%H:%M"),
                                 entry_utc=T[i] + pd.Timedelta(minutes=ENTRY_LAG),
                                 fwd=float(np.log(ext[i] / ent[i])), vol=float(v[i]),
                                 admitted=f in admitted, zscore=z[i], volscale=vs[i]))
        g = pd.DataFrame(recs).sort_values("event_ts").reset_index(drop=True)
        print(f"pre={pre}: {len(g)} pool games, {int(g.admitted.sum())} tradeable", flush=True)

        for norm in NORMS:
            res = {(lb, k): np.full(len(g), np.nan) for lb in GRID_LOOKBACK for k in GRID_K}
            win = max(1, round(WARP_MIN / FREQ))
            for _, grp in g.groupby("clock_et", sort=False):
                t0 = grp["event_ts"].values.astype("datetime64[ns]").astype(np.int64)
                rets = grp["fwd"].to_numpy(float)
                paths = [np.asarray(p, dtype=np.double) for p in grp[norm]]
                gi = grp.index.to_numpy(); adm = grp["admitted"].to_numpy()
                for pos in np.where(adm)[0]:
                    age = (t0[pos] - t0[:pos]) / 86_400_000_000_000.0
                    el = np.where((age > 0) & (age <= 365 * max(GRID_LOOKBACK)))[0][-MAX_POOL:]
                    if len(el) < MIN_HISTORY:
                        continue
                    d = np.array([dtw.distance_fast(paths[pos], paths[j], window=win,
                                                    use_pruning=True) for j in el])
                    m = np.isfinite(d); el, d = el[m], d[m]
                    if len(el) < MIN_HISTORY:
                        continue
                    a = (t0[pos] - t0[el]) / 86_400_000_000_000.0
                    order = np.argsort(d)
                    for lb in GRID_LOOKBACK:
                        mm = order[a[order] <= 365 * lb]
                        if len(mm) < MIN_HISTORY:
                            continue
                        for k in GRID_K:
                            s = mm[:min(k, len(mm))]
                            w = (1.0 / (d[s] + 1e-6)) * np.exp(-a[s] / RECENCY_HL)
                            res[(lb, k)][gi[pos]] = float(np.dot(w, rets[s]) / w.sum())
            base = g.drop(columns=NORMS)
            for (lb, k), v in res.items():
                p = base.copy(); p["dtw_factor"] = v
                p["pre"], p["norm"], p["lookback"], p["k"] = pre, norm, lb, k
                panels.append(p[p.admitted & p.dtw_factor.notna()])
            print(f"  pre={pre} norm={norm} done", flush=True)

    panel = pd.concat(panels, ignore_index=True)
    panel["yr"] = panel.event_ts.dt.year
    panel = panel[panel.yr >= TRAIN_START]
    panel = panel.merge(news.rename(columns={"ts": "event_ts"}), on=["family", "event_ts"], how="left")
    panel.to_parquet(OUT / "dtw_reaction_panel.parquet")

    rows = []
    for (pre, norm, lb, k), gg in panel.groupby(["pre", "norm", "lookback", "k"]):
        s, c = ib.score(gg[gg.yr < SPLIT], "dtw_factor"), ib.score(gg[gg.yr >= SPLIT], "dtw_factor")
        rows.append(dict(pre=pre, norm=norm, lookback=lb, k=k, sel_n=s["n"], sel_ic=s["ic"],
                         sel_gross=s["gross"], sel_t=s["t"], con_n=c["n"], con_ic=c["ic"],
                         con_hit=c["hit"], con_gross=c["gross"], con_net=c["net"], con_t=c["t"]))
    grid = pd.DataFrame(rows).sort_values("sel_ic", ascending=False)
    grid.to_csv(OUT / "dtw_reaction_grid.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n===== REACTION-PATH GRID (selected on 2016-2021) =====")
    print(grid.round(4).to_string(index=False))
    print(f"\nrank corr select->confirm IC: {stats.spearmanr(grid.sel_ic, grid.con_ic).correlation:+.3f}")
    print("\nmean by normalisation:")
    print(grid.groupby("norm")[["sel_ic", "con_ic", "con_gross", "con_t"]].mean().round(4).to_string())
    best = grid.iloc[0]
    b = panel[(panel.pre == best.pre) & (panel["norm"] == best["norm"]) &
              (panel.lookback == best.lookback) & (panel.k == best.k)]
    print(f"\nBEST pre={int(best.pre)} norm={best['norm']} lb={int(best.lookback)} k={int(best.k)} — per family")
    for lab, sub in [("train", b[b.yr < SPLIT]), ("TEST ", b[b.yr >= SPLIT])]:
        for v, gg in sub.groupby("family"):
            if len(gg) < 30:
                continue
            s = ib.score(gg, "dtw_factor")
            print(f"  {lab} {str(v)[:36]:36s} n={s['n']:4d} ic={s['ic']:+.4f} hit={s['hit']:.3f} "
                  f"gross={s['gross']:+6.2f} net={s['net']:+6.2f} t={s['t']:+.2f}")
    print("\nwrote dtw_reaction_panel.parquet / dtw_reaction_grid.csv")


if __name__ == "__main__":
    main()
