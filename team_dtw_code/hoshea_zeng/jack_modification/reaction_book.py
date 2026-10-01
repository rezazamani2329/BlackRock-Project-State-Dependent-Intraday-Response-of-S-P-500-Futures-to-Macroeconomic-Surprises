"""The book: reaction-path DTW + news surprise at T+1, sized, one trade per timestamp.

Entry T+1, exit T+31 -- the window where Jack's sweep shows the news drift still has
content (-5.64 bps/sigma, t -2.68 for CPI) and where the reaction-path DTW factor is
measurable. This is the first design in the project where both signals are available at
the same executable instant, so it is the first place a blend can be tested honestly.

Everything is fitted on 2016-2021: the news direction signs, the standardisation
statistics, the blend weight, the sizing scale. 2022-2026 is read once.

Run: uv run python jack/reaction_book.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "jack/data"
SPLIT, TICK = 2022, 0.5
FEATURES = OUT / "features.parquet"
rng = np.random.default_rng(0)


def clustered_t(p, c):
    m = pd.DataFrame({"p": p, "c": c}).groupby("c").p.mean()
    return float(m.mean() / (m.std(ddof=1) / np.sqrt(len(m)))) if len(m) > 2 else np.nan


def collapse(d, sigcol):
    """One position per release timestamp: average the signals of simultaneous families."""
    g = d.groupby("event_ts").agg(sig=(sigcol, "mean"), fwd=("fwd", "first"),
                                  vol=("vol", "first"), entry_utc=("entry_utc", "first"))
    return g[g.sig != 0].reset_index()


def report(t, pos, label, years):
    gross = pos * t.fwd.to_numpy() * 1e4
    net = gross - np.abs(pos) * 2 * TICK
    d = pd.DataFrame({"p": net, "d": t.entry_utc.dt.date}).groupby("d").p.sum()
    sh = d.mean() / d.std(ddof=1) * np.sqrt(252) if d.std(ddof=1) > 0 else np.nan
    return dict(book=label, n=len(t), per_yr=round(len(t) / years, 1),
                gross=round(gross.mean(), 2), net=round(net.mean(), 2),
                bps_yr=round(net.sum() / years, 0), hit=round(float((gross > 0).mean()), 3),
                sharpe=round(float(sh), 2), t=round(clustered_t(net, t.event_ts.dt.date), 2))


def main():
    p = pd.read_parquet(OUT / "dtw_reaction_panel.parquet")
    p["yr"] = p.event_ts.dt.year
    # average the DTW factor over all 16 configurations: no cell is selected, so nothing
    # is tuned. The grid showed hyperparameter ranking does not persist (rank corr -0.17),
    # which makes the average the honest estimator rather than the best cell.
    key = ["family", "event_ts", "entry_utc", "fwd", "vol", "yr", "news_raw", "clock_et"]
    a = (p.groupby(key, dropna=False)
           .dtw_factor.mean().rename("dtw_factor").reset_index())
    print(f"{len(a)} games ({a.event_ts.nunique()} release timestamps), "
          f"{a.news_raw.notna().mean():.1%} carry a consensus")

    tr, te = a[a.yr < SPLIT].copy(), a[a.yr >= SPLIT].copy()
    yrs_tr, yrs_te = tr.yr.nunique(), te.yr.nunique()

    # news direction sign from the TRAIN jump regression only
    signs = {}
    for f, g in tr.dropna(subset=["news_raw"]).groupby("family"):
        u = g.drop_duplicates("event_ts")
        if len(u) > 25:
            signs[f] = float(np.sign(np.polyfit(u.news_raw, u.fwd, 1)[0]))
    for d in (tr, te):
        d["news_z"] = (d.news_raw * d.family.map(signs)).fillna(0.0)

    st = tr.groupby("family")[["dtw_factor", "news_z"]].agg(["mean", "std"])
    for d in (tr, te):
        for c in ["dtw_factor", "news_z"]:
            m, s = d.family.map(st[(c, "mean")]), d.family.map(st[(c, "std")])
            d[c + "_s"] = ((d[c] - m) / s).fillna(0.0).clip(-4, 4)

    print("\n===== DIRECTION at T+1, one trade per timestamp, net of a tick each way =====")
    rows = []
    for w in [1.0, 0.75, 0.5, 0.25, 0.0]:
        for lab, d, y in [("train 16-21", tr, yrs_tr), ("TEST 22-26", te, yrs_te)]:
            d = d.copy(); d["blend"] = w * d.dtw_factor_s + (1 - w) * d.news_z_s
            t = collapse(d, "blend")
            rows.append(dict(w_dtw=w, period=lab, **report(t, np.sign(t.sig), "sign", y)))
    print(pd.DataFrame(rows).drop(columns="book").to_string(index=False))

    print("\n===== SIZING (equal blend, w_dtw = 0.5) =====")
    rows = []
    for lab, d, y in [("train 16-21", tr, yrs_tr), ("TEST 22-26", te, yrs_te)]:
        d = d.copy(); d["blend"] = 0.5 * d.dtw_factor_s + 0.5 * d.news_z_s
        t = collapse(d, "blend")
        scale = np.std(collapse(tr.assign(blend=0.5 * tr.dtw_factor_s + 0.5 * tr.news_z_s), "blend").sig)
        volref = np.median(collapse(tr.assign(blend=tr.dtw_factor_s), "blend").vol)
        vn = (t.vol / volref).to_numpy()
        prop = np.clip(t.sig.to_numpy() / scale, -3, 3)
        for nm, pos in [("fixed sign", np.sign(t.sig)), ("proportional", prop),
                        ("inverse-vol", np.sign(t.sig) / vn), ("proportional / vol", prop / vn),
                        ("proportional * vol", prop * vn)]:
            rows.append(dict(period=lab, **report(t, np.asarray(pos), nm, y)))
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n===== PER YEAR (w_dtw = 0.5, fixed sign) =====")
    a2 = pd.concat([tr, te]); a2["blend"] = 0.5 * a2.dtw_factor_s + 0.5 * a2.news_z_s
    t = collapse(a2, "blend")
    t["yr"] = t.event_ts.dt.year
    pnl = np.sign(t.sig) * t.fwd * 1e4 - 2 * TICK
    for y, g in pd.DataFrame({"yr": t.yr, "p": pnl}).groupby("yr"):
        print(f"  {y}  n={len(g):4d}  net={g.p.mean():+6.2f} bps  total={g.p.sum():+7.0f}  hit={(g.p>0).mean():.3f}")

    print("\n===== NULL: is the test-period blend distinguishable from a coin flip? =====")
    d = te.copy(); d["blend"] = 0.5 * d.dtw_factor_s + 0.5 * d.news_z_s
    t = collapse(d, "blend")
    real = (np.sign(t.sig) * t.fwd * 1e4 - 2 * TICK).mean()
    draws = np.array([(rng.choice([-1.0, 1.0], len(t)) * t.fwd * 1e4 - 2 * TICK).mean()
                      for _ in range(5000)])
    print(f"  real net {real:+.3f} bps;  random-sign null mean {draws.mean():+.3f}, "
          f"sd {draws.std():.3f};  P(null >= real) = {(draws >= real).mean():.3f}")


if __name__ == "__main__":
    main()
