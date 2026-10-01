"""Can the consensus surprise and the DTW features be combined? ES and NQ, train + validation only.

Joins a causal surprise score onto the admitted games in `output/features.parquet` and asks:

1. Coverage: how many admitted games have a surprise at all.
2. Per anchor, does the surprise rank the next 30 minutes, and is it independent of `dtw_dir`?
3. Does `dtw_disp` change how much the surprise earns (terciles of dtw_disp)?
4. Five simple trading arms at T+1, quarter tick per side, train and validation Sharpe.

The surprise follows Reza's `surprise_universe` (z = (actual - survey) / sd of the line's
earlier surprises, capped at +-5; family score x = mean over its lines of sign * z), with two
changes so it exists on the training years: families are this folder's (features.py), and each
line's sign is the sign of cov(raw surprise, print-minute jump) on releases from 2010 up to the
start of the year, needing 24 releases. Test years (2021-2026) are never loaded.

Run: uv run python combined/experiments/surprise_dtw.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "Reza"))
from config import (ASSETS, BLOOMBERG, CLEAN_BARS_FMT, FAMILY_OVERLAP, MAX_POSITION, NY, PANEL,
                    SESSION_SHIFT, TRAIN_YEARS_SPAN, VALID_YEARS_SPAN)
from util import Px
import src.surprise_universe as su

pd.set_option("display.width", 200)

SYMBOLS = ["ES", "NQ"]
PERIODS = {"train": TRAIN_YEARS_SPAN, "valid": VALID_YEARS_SPAN}
LAST_YEAR = VALID_YEARS_SPAN[1]
COST_TICKS = 0.25                  # per side
MIN_SIGN_N = 24


# ---------------------------------------------------------------- families (as features.py)
def assign_families(lines: pd.DataFrame) -> dict[str, str]:
    """features.py's grouping, copied: importing features.py would re-run the pipeline."""
    stamps = {ln: set(g.ts_utc) for ln, g in lines.groupby("event_name") if len(g) >= 12}
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
        clock = lines.loc[lines.event_name == best, "clock_et"].mode().iloc[0]
        for m in members:
            out[m] = f"{best} @ {clock}"
    return out


rows = su.load_bloomberg(BLOOMBERG)
rows = rows[rows.ts_utc.dt.year <= LAST_YEAR]
rows["family"] = rows.event_name.map(assign_families(rows))
surp = su.standardise_surprises(rows.dropna(subset=["family"]))   # z, raw; causal

panel = pd.read_parquet(PANEL)
games = panel[panel.admitted & panel.dtw_dir.notna() & panel.symbol.isin(SYMBOLS)
              & panel.year.between(TRAIN_YEARS_SPAN[0], LAST_YEAR)].copy()
assert not (set(games.family) - set(rows.family)), "family names do not match features.py"


# ---------------------------------------------------------------- surprise score per market
def family_scores(s: str) -> pd.DataFrame:
    b = pd.read_parquet(CLEAN_BARS_FMT.format(symbol=s), columns=["close", "contract"]).sort_index()
    b.index = pd.to_datetime(b.index, utc=True)
    px, ct = Px(np.log(b.close)), Px(b.contract.astype(float))
    T = pd.DatetimeIndex(surp.ts_utc)
    one = pd.Timedelta(minutes=1)
    d = surp.copy()
    d["jump"] = np.where(ct.at(T) == ct.at(T - one), px.at(T) - px.at(T - one), np.nan)
    d["year"] = d.ts_utc.dt.tz_convert(NY).dt.year
    out = []
    for y in range(TRAIN_YEARS_SPAN[0], LAST_YEAR + 1):
        h = d[(d.year < y)].dropna(subset=["raw", "jump"])
        h = h[h.groupby("ticker").ticker.transform("size") >= MIN_SIGN_N]
        sign = np.sign(h.groupby("ticker")[["raw", "jump"]].apply(lambda g: np.cov(g.raw, g.jump)[0, 1]))
        cur = d[d.year == y].dropna(subset=["z"]).copy()
        cur["sz"] = cur.ticker.map(sign) * cur.z
        out.append(cur.dropna(subset=["sz"]).groupby(["family", "ts_utc"]).sz.mean())
    return pd.concat(out).rename("x").reset_index().rename(columns={"ts_utc": "event_ts"}).assign(symbol=s)


x = pd.concat([family_scores(s) for s in SYMBOLS])
games = games.merge(x, on=["symbol", "family", "event_ts"], how="left")
games["period"] = np.where(games.year <= TRAIN_YEARS_SPAN[1], "train", "valid")


def ic(a, b) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    return stats.spearmanr(a[m], b[m])[0] if m.sum() > 10 else np.nan


# ---------------------------------------------------------------- 1. coverage
print("1. Admitted games at T+1, and the share with a surprise score")
a1 = games[games.anchor == 1]
cov_tab = a1.groupby(["symbol", "period"]).agg(games=("x", "size"), with_x=("x", lambda v: v.notna().mean()))
print(cov_tab.round(2).to_string())
print("\nshare with a surprise, by family (T+1, both periods):")
print(a1.groupby(["family", "symbol"]).x.apply(lambda v: v.notna().mean()).unstack().round(2).to_string())

# ---------------------------------------------------------------- 2. IC by anchor
print("\n2. Rank IC with the next 30-minute return, games with a surprise only")
ic_rows = []
for (s, per, a), g in games.dropna(subset=["x"]).groupby(["symbol", "period", "anchor"]):
    ic_rows.append({"symbol": s, "period": per, "anchor": a, "n": len(g),
                    "IC surprise": ic(g.x.values, g.fwd_ret.values),
                    "IC dtw_dir": ic(g.dtw_dir.values, g.fwd_ret.values),
                    "corr(surprise, dtw_dir)": ic(g.x.values, g.dtw_dir.values),
                    "hit sign(x)": float((np.sign(g.x) * g.fwd_ret > 0).mean())})
print(pd.DataFrame(ic_rows).set_index(["symbol", "period", "anchor"]).round(3).to_string())

# ---------------------------------------------------------------- 3. dtw_disp terciles
print("\n3. T+1: gross bps per trade of sign(surprise), by dtw_disp tercile (cut on train)")
a1 = a1.dropna(subset=["x"]).copy()
a1["edge_bps"] = np.sign(a1.x) * a1.fwd_ret * 1e4
terc = []
for s in SYMBOLS:
    g = a1[a1.symbol == s].copy()
    cuts = g[g.period == "train"].dtw_disp.quantile([1 / 3, 2 / 3]).to_numpy()
    g["disp tercile"] = np.digitize(g.dtw_disp, cuts) + 1
    t = g.groupby(["period", "disp tercile"]).agg(n=("edge_bps", "size"), bps=("edge_bps", "mean"),
                                                   abs_move=("fwd_ret", lambda v: np.abs(v).mean() * 1e4))
    terc.append(t.assign(symbol=s).set_index("symbol", append=True))
print(pd.concat(terc).reorder_levels(["symbol", "period", "disp tercile"]).round(2).to_string())


# ---------------------------------------------------------------- 4. trading arms at T+1
def load_close(s: str) -> tuple[pd.Series, pd.DatetimeIndex]:
    b = pd.read_parquet(CLEAN_BARS_FMT.format(symbol=s), columns=["close", "session_date"])
    b.index = pd.to_datetime(b.index, utc=True)
    return b.close, pd.DatetimeIndex(pd.to_datetime(b.session_date).unique()).sort_values()


def arm_positions(g: pd.DataFrame, ref: dict, arm: str) -> pd.Series:
    inv = np.clip(ref["disp"] / g.dtw_disp, 0, MAX_POSITION)
    pro = np.clip(g.dtw_disp / ref["disp"], 0, MAX_POSITION)
    return {"DTW dir, flat": np.sign(g.dtw_dir),
            "surprise, flat": np.sign(g.x),
            "surprise, inverse disp": np.sign(g.x) * inv,
            "surprise, scaled with disp": np.sign(g.x) * pro,
            "surprise + dtw_dir blend": np.sign(g.x / ref["x_sd"] + g.dtw_dir / ref["dir_sd"])}[arm]


ARMS = ["DTW dir, flat", "surprise, flat", "surprise, inverse disp", "surprise, scaled with disp",
        "surprise + dtw_dir blend"]
print(f"\n4. Trading arms at T+1, games with a surprise, {COST_TICKS} tick per side "
      "(positions averaged per entry minute, Sharpe on the daily session grid)")
res = []
for s in SYMBOLS:
    g_all = a1[a1.symbol == s].copy()
    tr = g_all[g_all.period == "train"]
    ref = {"disp": tr.dtw_disp.median(), "x_sd": tr.x.std(), "dir_sd": tr.dtw_dir.std()}
    close, days = load_close(s)
    for arm in ARMS:
        g_all["position"] = arm_positions(g_all, ref, arm)
        t = (g_all[g_all.position != 0].groupby("entry_ts")
             .agg(position=("position", "mean"), fwd_ret=("fwd_ret", "first"), period=("period", "first"))
             .reset_index())
        t = t[t.position != 0]
        t["session"] = (t.entry_ts.dt.tz_convert(NY) + SESSION_SHIFT).dt.normalize().dt.tz_localize(None)
        tick_bps = ASSETS[s]["tick"] / close.reindex(t.entry_ts).to_numpy() * 1e4
        t["net"] = t.position * t.fwd_ret * 1e4 - t.position.abs() * 2 * COST_TICKS * tick_bps
        for per, span in PERIODS.items():
            tp = t[t.period == per]
            d = tp.groupby("session").net.sum().reindex(days[(days.year >= span[0]) & (days.year <= span[1])]).fillna(0.0)
            res.append({"symbol": s, "arm": arm, "period": per, "trades": len(tp),
                        "net bps/trade": tp.net.mean(),
                        "sharpe": d.mean() / d.std(ddof=1) * np.sqrt(252) if d.std() else np.nan})
res = pd.DataFrame(res).pivot_table(index=["symbol", "arm"], columns="period",
                                    values=["trades", "net bps/trade", "sharpe"])
print(res.reindex(ARMS, level="arm").round(3).to_string())
