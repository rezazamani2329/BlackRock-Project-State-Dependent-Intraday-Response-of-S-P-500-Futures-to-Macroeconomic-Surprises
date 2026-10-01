"""Rank macro releases by *event*, not by release line, and size the top-K basket.

`jack/news_admission_rule.py` ranks 93 Bloomberg release lines by jump response. That
ranking is misleading to read directly: six of its top eight are CPI MoM, CPI YoY, Core
CPI MoM, Core CPI YoY, CPI Index NSA and Core CPI Index SA, which are six views of one
number printed at one instant. "Top 8 lines" is really three distinct events.

This groups release lines into families automatically — two lines belong to the same
family when they share most of their release timestamps, which is a fact about the
calendar and needs no hand-maintained mapping — then ranks families, so "the top 10
events" means what it sounds like.

Within a family the member surprises are averaged into one direction-aligned signal, and
the family trades once per release. Families are ranked on 2012-2021 by the R-squared of
the announcement jump on that combined signal; the basket is then traded on 2022-2026.

Run: uv run python jack/event_family_screen.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ES_FILE = ROOT / "data" / "processed" / "es_1min_clean.parquet"
BLOOMBERG_FILE = ROOT / "data" / "raw" / "Bloomberg Economic Releases.xlsx"
OUT = ROOT / "jack" / "data"

WINSOR_SIGMA = 5.0
TICK_BPS = 0.5
MIN_OBS_SELECT = 30
SPLIT = pd.Timestamp("2022-01-01", tz="UTC")
START = pd.Timestamp("2012-01-01", tz="UTC")
DRIFT_MINUTES = 30
OVERLAP_THRESHOLD = 0.80   # share of timestamps two lines must share to be one family

es = pd.read_parquet(ES_FILE, columns=["open"])
es = es[~es.index.duplicated(keep="first")].sort_index()
ES_IDX, ES_OPEN = es.index, es["open"].to_numpy()


def px(ts: pd.Timestamp, tol=pd.Timedelta(minutes=5)) -> float:
    pos = ES_IDX.searchsorted(ts)
    if pos >= len(ES_IDX) or abs(ES_IDX[pos] - ts) > tol:
        return np.nan
    return ES_OPEN[pos]


def load_bloomberg() -> pd.DataFrame:
    sheets = pd.ExcelFile(BLOOMBERG_FILE)
    frames = []
    for sheet in sheets.sheet_names:
        part = pd.read_excel(BLOOMBERG_FILE, sheet_name=sheet)
        part.columns = [c.split(".")[-1] if "DROPNA" in str(c) else c for c in part.columns]
        frames.append(part)
    b = pd.concat(frames, ignore_index=True)
    b["RELEASE_DATE"] = pd.to_datetime(b.RELEASE_DATE, errors="coerce")
    b["RELEASE_TIME"] = b.RELEASE_TIME.astype(str)
    b = b.dropna(subset=["RELEASE_DATE"])
    return b[b.SURVEY_MEDIAN.notna() & b.ACTUAL.notna()]


def build_line_panel(b: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (name, time), g in b.groupby(["EVENT_NAME", "RELEASE_TIME"]):
        if len(g) < 60 or not time[:2].isdigit():
            continue
        g = g.sort_values("RELEASE_DATE").copy()
        g["surprise"] = g.ACTUAL - g.SURVEY_MEDIAN
        scale = g.surprise.expanding().std().shift(1)
        g["z"] = (g.surprise / scale).clip(-WINSOR_SIGMA, WINSOR_SIGMA)
        g = g[np.isfinite(g.z) & (scale > 0) & (g.z != 0)]
        if len(g) < 40:
            continue
        stamp = pd.to_datetime(g.RELEASE_DATE.dt.strftime("%Y-%m-%d") + " " + time)
        ts = (stamp.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
              .dt.tz_convert("UTC"))
        for r in g.assign(ts=ts).dropna(subset=["ts"]).itertuples(index=False):
            if r.ts < START:
                continue
            p0, p1 = px(r.ts), px(r.ts + pd.Timedelta(minutes=1))
            p2 = px(r.ts + pd.Timedelta(minutes=1 + DRIFT_MINUTES))
            if not np.isfinite([p0, p1, p2]).all():
                continue
            rows.append({"line": f"{name} @ {time[:5]}", "ts": r.ts, "z": r.z,
                         "jump": 1e4 * np.log(p1 / p0), "drift": 1e4 * np.log(p2 / p1)})
    return pd.DataFrame(rows)


def assign_families(panel: pd.DataFrame) -> dict[str, str]:
    """Union-find over release lines that share most of their release timestamps."""
    stamps = {line: set(g.ts) for line, g in panel.groupby("line")}
    lines = sorted(stamps, key=lambda l: -len(stamps[l]))
    parent = {l: l for l in lines}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(lines):
        for bl in lines[i + 1:]:
            if find(a) == find(bl):
                continue
            shared = len(stamps[a] & stamps[bl])
            if shared / min(len(stamps[a]), len(stamps[bl])) >= OVERLAP_THRESHOLD:
                parent[find(bl)] = find(a)
    # name each family after its longest-history member
    names = {}
    for l in lines:
        root = find(l)
        names.setdefault(root, root)
    return {l: names[find(l)] for l in lines}


def family_signal(panel: pd.DataFrame, signs: dict[str, int]) -> pd.DataFrame:
    """One row per (family, timestamp): the direction-aligned average surprise."""
    p = panel.copy()
    p["aligned_z"] = p.line.map(signs) * p.z
    return (p.groupby(["family", "ts"])
             .agg(signal=("aligned_z", "mean"), jump=("jump", "first"),
                  drift=("drift", "first"), legs=("line", "size"))
             .reset_index())


def r2(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < MIN_OBS_SELECT:
        return np.nan
    c = np.corrcoef(x, y)[0, 1]
    return float(c ** 2) if np.isfinite(c) else np.nan


def basket(trades: pd.DataFrame) -> dict:
    """Collapse simultaneous families into one position, then score."""
    t = (trades.assign(w=lambda d: np.sign(d.signal))
                .groupby("ts").agg(side=("w", "mean"), drift=("drift", "first")).reset_index())
    t = t[t.side != 0]
    pnl = np.sign(t.side) * t.drift - TICK_BPS
    years = (t.ts.max() - t.ts.min()).days / 365.25
    tpy = len(pnl) / years
    return {"trades": len(pnl), "trades_yr": round(tpy, 1), "bps": round(pnl.mean(), 2),
            "bps_yr": round(pnl.mean() * tpy, 1), "hit_%": round(100 * (pnl > 0).mean(), 1),
            "sharpe": round((pnl.mean() * tpy) / (pnl.std(ddof=1) * np.sqrt(tpy)), 2),
            "t": round(pnl.mean() / (pnl.std(ddof=1) / np.sqrt(len(pnl))), 2)}


def main() -> None:
    panel = build_line_panel(load_bloomberg())
    fam = assign_families(panel)
    panel["family"] = panel.line.map(fam)
    print(f"release lines: {panel.line.nunique()}  ->  event families: {panel.family.nunique()}")

    sel_lines = panel[panel.ts < SPLIT]
    # each line's reaction sign, from its own jump regression on the selection period
    signs = {}
    for line, g in sel_lines.groupby("line"):
        if len(g) < MIN_OBS_SELECT:
            signs[line] = 1
            continue
        signs[line] = int(np.sign(np.polyfit(g.z, g.jump, 1)[0])) or 1

    sig = family_signal(panel, signs)
    sel, con = sig[sig.ts < SPLIT], sig[sig.ts >= SPLIT]

    rank = []
    for family, g in sel.groupby("family"):
        gc = con[con.family == family]
        if len(g) < MIN_OBS_SELECT or len(gc) < 12:
            continue
        pnl_sel = (np.sign(g.signal) * g.drift).mean()
        pnl_con = (np.sign(gc.signal) * gc.drift).mean()
        rank.append({"family": family, "legs": int(g.legs.median()),
                     "n_sel": len(g), "n_con": len(gc),
                     "jump_r2": r2(g.signal.to_numpy(), g.jump.to_numpy()),
                     "sel_drift": pnl_sel, "con_drift": pnl_con})
    rank = pd.DataFrame(rank).dropna(subset=["jump_r2"]).sort_values("jump_r2", ascending=False)
    rank = rank.reset_index(drop=True)
    rank.index += 1
    rank.to_csv(OUT / "event_family_screen.csv")

    print("\n" + "=" * 108)
    print("EVENT FAMILIES RANKED BY JUMP RESPONSE (2012-2021 only)")
    print("=" * 108)
    view = rank.head(20).copy()
    view["gap_to_next"] = view.jump_r2.diff(-1)
    print(view.round(3).to_string())

    print("\n" + "=" * 108)
    print("BASKET SIZE SWEEP — chosen on 2012-2021, traded on 2022-2026")
    print("=" * 108)
    rows = []
    for K in range(1, min(21, len(rank)) + 1):
        keep = rank.head(K).family.tolist()
        s = basket(con[con.family.isin(keep)])
        s_in = basket(sel[sel.family.isin(keep)])
        rows.append({"K": K, "newest_family": rank.iloc[K - 1].family,
                     "jump_r2": round(rank.iloc[K - 1].jump_r2, 3),
                     **s, "in_sample_bps": s_in["bps"], "in_sample_sharpe": s_in["sharpe"]})
    sweep = pd.DataFrame(rows)
    print(sweep.to_string(index=False))
    sweep.to_csv(OUT / "event_family_sweep.csv", index=False)

    print("\n" + "=" * 108)
    print("THE TOP-10 BASKET, per family, 2022-2026")
    print("=" * 108)
    keep = rank.head(10).family.tolist()
    print(f"  {'family':<46} {'legs':>5} {'n':>4} {'bps':>8} {'hit':>7} {'t':>7}")
    for family in keep:
        g = con[con.family == family]
        pnl = np.sign(g.signal) * g.drift - TICK_BPS
        print(f"  {family:<46} {int(g.legs.median()):>5} {len(pnl):>4} {pnl.mean():>+8.2f} "
              f"{(pnl > 0).mean():>6.1%} {pnl.mean()/(pnl.std(ddof=1)/np.sqrt(len(pnl))):>+7.2f}")

    top10 = con[con.family.isin(keep)]
    t = (top10.assign(w=lambda d: np.sign(d.signal))
               .groupby("ts").agg(side=("w", "mean"), drift=("drift", "first")).reset_index())
    t = t[t.side != 0]
    t["pnl_bps"] = np.sign(t.side) * t.drift - TICK_BPS
    t["year"] = pd.DatetimeIndex(t.ts).year
    print("\n  year by year:")
    for year, g in t.groupby("year"):
        print(f"    {year}  trades {len(g):>3}  {g.pnl_bps.mean():>+7.2f} bps/trade  "
              f"{g.pnl_bps.sum():>+8.1f} bps total  hit {(g.pnl_bps > 0).mean():>5.1%}")
    t.to_parquet(OUT / "event_family_top10_trades.parquet", index=False)
    print(f"\nsaved -> {OUT/'event_family_screen.csv'}, {OUT/'event_family_sweep.csv'}")


if __name__ == "__main__":
    main()
