"""Two questions about the multi-anchor result, before anyone builds on it.

A. DECOMPOSITION. The ladder scored Sharpe 0.431 against the fitted book's 0.159, but two
   things changed at once: more anchors, AND a plain sign() rule replacing the fitted
   model plus E[r]/E|r| sizing. The trivial-benchmark table already showed the model
   destroys ~5x of value on its own, so the 2x2 below separates the two effects.

B. FRAGILITY. Two near-identical constructions -- 31 vs 32 path points, pool keyed on
   (clock, anchor) vs clock -- correlate 0.849 and return +0.745 vs +1.650 bps on the same
   520 trades. One bar of path length roughly doubles the P&L. If Sharpe ranges widely
   across constructions that ought to be interchangeable, the headline is one draw from
   that spread rather than an estimate of anything. Sweeps path length x pooling key.

Run: uv run python combined/experiments/anchor_robustness.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from dtaidistance import dtw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # combined/
from config import (
    CLEAN_BARS, COST_BPS_PER_SIDE, HOLD_MIN, K_NEIGHBOURS, LOOKBACK_YEARS, MAX_POOL,
    MAX_POSITION, MIN_NEIGHBOURS, OUT, PANEL, RECENCY_HALFLIFE_DAYS, TRAIN_YEARS,
    WARP_MINUTES,
)
from util import Px

pd.set_option("display.width", 200)

ANCHORS = [1, 31, 61, 91, 121]
DIR_FEATURES = ["dtw_dir", "news_z", "jump_z", "dtw_x_agree", "dtw_x_news"]
MAG_FEATURES = ["vix_level", "log_rv4h", "seas_exp", "impact"]
PRE_SWEEP = [20, 30, 40]
KEY_SWEEP = [("clock_et", "anchor"), ("clock_et",), ("family", "anchor")]


def daily_stats(sub: pd.DataFrame, pos: np.ndarray, span: pd.DatetimeIndex) -> dict:
    gross = pos * sub.fwd_ret.to_numpy() * 1e4
    net = gross - np.abs(pos) * 2 * COST_BPS_PER_SIDE
    d = pd.Series(net).groupby(sub.date.to_numpy()).sum().reindex(span).fillna(0.0)
    ann, vol = d.mean() * 252 / 100, d.std(ddof=1) * np.sqrt(252) / 100
    eq = d.cumsum()
    return dict(n=len(sub), per_yr=round(len(sub) / sub.year.nunique()),
                net_bps=net.mean(), total_pct=net.sum() / 100,
                sharpe=ann / vol if vol else np.nan,
                max_dd=float((eq - eq.cummax()).min()) / 100,
                hit=float((gross > 0).mean()))


def walk_forward(panel: pd.DataFrame) -> pd.DataFrame:
    """Expanding-window refit, one year at a time. Same protocol as backtest.py."""
    first = panel.year.min() + TRAIN_YEARS
    out = []
    for y in sorted(panel.year.unique()):
        if y < first:
            continue
        tr, te = panel[panel.year < y], panel[panel.year == y]
        if len(tr) < 80 or te.empty:
            continue
        Xd = sm.add_constant(tr[DIR_FEATURES], has_constant="add")
        d = sm.OLS(tr.fwd_ret * 1e4, Xd).fit()
        Xm = sm.add_constant(tr[MAG_FEATURES], has_constant="add")
        m = sm.GLM((tr.fwd_ret * 1e4).abs().clip(lower=0.1), Xm,
                   family=sm.families.Gamma(sm.families.links.Log())).fit()
        t = te.copy()
        t["pred"] = d.predict(sm.add_constant(te[DIR_FEATURES], has_constant="add"))
        t["pred_abs"] = np.clip(m.predict(sm.add_constant(te[MAG_FEATURES], has_constant="add")),
                                1e-6, None)
        out.append(t)
    return pd.concat(out) if out else pd.DataFrame()


def build_paths(px: Px, rel: pd.DataFrame, pre: int) -> pd.DataFrame:
    steps = np.arange(-pre, 1)
    recs = []
    for (fam, clock), g in rel.groupby(["family", "clock_et"], sort=False):
        T = pd.DatetimeIndex(sorted(g.event_ts))
        for a in ANCHORS:
            entry = T + pd.Timedelta(minutes=a)
            grid = entry.values[:, None] + (steps * 60_000_000_000).astype("timedelta64[ns]")
            path = px.at(pd.DatetimeIndex(grid.ravel()).tz_localize("UTC")).reshape(len(T), len(steps))
            p0, p1 = px.at(entry), px.at(entry + pd.Timedelta(minutes=HOLD_MIN))
            ok = np.isfinite(path).all(1) & np.isfinite(p0) & np.isfinite(p1)
            sd = path.std(1, keepdims=True)
            sd[sd < 1e-12] = 1.0
            z = (path - path.mean(1, keepdims=True)) / sd
            for i in np.where(ok)[0]:
                recs.append(dict(family=fam, event_ts=T[i], clock_et=clock, anchor=a,
                                 entry_ts=entry[i], year=int(T[i].year),
                                 fwd_ret=float(np.log(p1[i] / p0[i])), path=z[i]))
    return pd.DataFrame(recs).sort_values("entry_ts").reset_index(drop=True)


def dtw_on(games: pd.DataFrame, query: np.ndarray, key: tuple) -> np.ndarray:
    win = max(1, round(WARP_MINUTES / 1))
    out = np.full(len(games), np.nan)
    for _, grp in games.groupby(list(key), sort=False):
        idx = grp.index.to_numpy()
        t0 = grp.entry_ts.values.astype("datetime64[ns]").astype(np.int64)
        rets = grp.fwd_ret.to_numpy(float)
        paths = [np.asarray(p, dtype=np.double) for p in grp.path]
        for pos in np.where(query[idx])[0]:
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
            out[idx[pos]] = float(np.dot(w / w.sum(), rets[sel]))
    return out


def main() -> None:
    close = pd.read_parquet(CLEAN_BARS)["close"]
    close.index = pd.to_datetime(close.index, utc=True)
    close = close[~close.index.duplicated()].sort_index()
    px = Px(close)
    dpx = close.groupby(close.index.tz_convert("America/New_York").normalize().tz_localize(None)).last()

    feat = pd.read_parquet(PANEL)
    rel_cols = ["family", "event_ts", "news_z", "jump_z", "log_rv4h", "vix_level",
                "seas_exp", "impact"]
    rel_feat = feat[rel_cols].drop_duplicates(["family", "event_ts"])

    # ---------------- A. decomposition -------------------------------------
    a = pd.read_parquet(OUT / "anchor_panel.parquet")
    a = a[a.admitted & a.dtw_dir.notna()].merge(rel_feat, on=["family", "event_ts"], how="left")
    a["news_z"] = a.news_z.fillna(0.0)
    a["jump_z"] = a.jump_z.clip(-10, 10).fillna(0.0)
    a["dtw_x_agree"] = a.dtw_dir * (a.dtw_agree - 0.5)
    a["dtw_x_news"] = a.dtw_dir * a.news_z
    a = a.dropna(subset=DIR_FEATURES + MAG_FEATURES)
    a["date"] = a.entry_ts.dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    span = dpx.index[(dpx.index >= a.date.min()) & (dpx.index <= a.date.max())]

    print("===== A. DECOMPOSITION — is it the anchors, or dropping the model? =====")
    rows = []
    for nlab, ancs in [("1 anchor", [1]), ("5 anchors", ANCHORS)]:
        sub = a[a.anchor.isin(ancs)]
        oos_sign = sub[sub.year >= sub.year.min() + TRAIN_YEARS]
        rows.append(dict(anchors=nlab, rule="sign(dtw_dir)",
                         **daily_stats(oos_sign, np.sign(oos_sign.dtw_dir.to_numpy()), span)))
        wf = walk_forward(sub)
        if not wf.empty:
            pos = np.clip(wf.pred / wf.pred_abs, -MAX_POSITION, MAX_POSITION).to_numpy()
            rows.append(dict(anchors=nlab, rule="fitted model (sized)",
                             **daily_stats(wf, pos, span)))
            rows.append(dict(anchors=nlab, rule="fitted model, sign only",
                             **daily_stats(wf, np.sign(wf.pred.to_numpy()), span)))
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    # ---------------- B. fragility sweep -----------------------------------
    print("\n===== B. FRAGILITY — path length x pooling key, full ladder =====", flush=True)
    rel = feat[["family", "event_ts", "clock_et", "admitted"]].drop_duplicates()
    adm_idx = rel[rel.admitted].set_index(["family", "event_ts"]).index
    rows = []
    for pre in PRE_SWEEP:
        g = build_paths(px, rel, pre)
        g["admitted"] = pd.MultiIndex.from_frame(g[["family", "event_ts"]]).isin(adm_idx)
        q = g.admitted.to_numpy()
        for key in KEY_SWEEP:
            g2 = g.copy()
            g2["dtw_dir"] = dtw_on(g2, q, key)
            s = g2[g2.admitted & g2.dtw_dir.notna()].copy()
            s["date"] = s.entry_ts.dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
            s = s[s.year >= s.year.min() + TRAIN_YEARS]
            if len(s) < 200:
                continue
            rows.append(dict(pre_min=pre, pool="+".join(key),
                             **daily_stats(s, np.sign(s.dtw_dir.to_numpy()), span)))
            print(f"  pre={pre} pool={'+'.join(key):18s} "
                  f"sharpe {rows[-1]['sharpe']:+.3f}  net {rows[-1]['net_bps']:+.3f}", flush=True)
    sweep = pd.DataFrame(rows)
    sweep.to_csv(OUT / "anchor_robustness.csv", index=False)
    print("\n" + sweep.round(3).to_string(index=False))
    print(f"\nSharpe across {len(sweep)} interchangeable constructions: "
          f"min {sweep.sharpe.min():+.3f}  median {sweep.sharpe.median():+.3f}  "
          f"max {sweep.sharpe.max():+.3f}  sd {sweep.sharpe.std():.3f}")
    print(f"fraction positive: {(sweep.sharpe > 0).mean():.0%}")


if __name__ == "__main__":
    main()
