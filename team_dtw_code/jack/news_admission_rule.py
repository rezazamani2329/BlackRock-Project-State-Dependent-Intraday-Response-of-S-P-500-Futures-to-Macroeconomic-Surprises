"""A release-admission rule built on information, not on past P&L.

`jack/event_tradeability_rule.py` shows that picking events by their past directional
performance is actively harmful: across 12 event types, the rank correlation between
2012-2021 performance and 2022-2026 performance is -0.31 on bps and -0.50 on IC, and a
basket chosen that way underperforms the basket it rejected. Any strategy whose universe
is selected on past returns inherits that.

So select on something else. A release is worth trading only if the market demonstrably
*reacts* to it, and that reaction is measurable, stable, and has nothing to do with
whether the drift afterwards was profitable:

    jump_R2 = R^2 of ( log open(T+1) / open(T) )  on  the standardised surprise

That is a property of the information — does this number move the tape at all? — and it
is estimable from the selection period alone. The rule is then: admit the top-K releases
by jump_R2, trade the drift on every admitted release, never look at drift P&L when
choosing. This script fits the rule on <=2021 and reports 2022-2026.

Two controls run alongside it:
  * selecting on past *drift* P&L instead, which is the trap above;
  * the complement basket, so "admitted beats rejected" is testable rather than assumed.

Run: uv run python jack/news_admission_rule.py
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
MIN_OBS_SELECT = 30           # releases need this much history before 2022 to be judged
SPLIT = pd.Timestamp("2022-01-01", tz="UTC")
START = pd.Timestamp("2012-01-01", tz="UTC")
DRIFT_MINUTES = 30
TOP_K = 8

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


def build_release_panel(b: pd.DataFrame) -> pd.DataFrame:
    """One row per (release, date): standardised surprise, jump, drift."""
    rows = []
    for (name, time), g in b.groupby(["EVENT_NAME", "RELEASE_TIME"]):
        g = g.sort_values("RELEASE_DATE").copy()
        if len(g) < 60 or not time[:2].isdigit():
            continue
        g["surprise"] = g.ACTUAL - g.SURVEY_MEDIAN
        scale = g.surprise.expanding().std().shift(1)
        g["z"] = (g.surprise / scale).clip(-WINSOR_SIGMA, WINSOR_SIGMA)
        g = g[np.isfinite(g.z) & (scale > 0) & (g.z != 0)]
        if len(g) < 40:
            continue
        stamp = pd.to_datetime(g.RELEASE_DATE.dt.strftime("%Y-%m-%d") + " " + time)
        ts = (stamp.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
              .dt.tz_convert("UTC"))
        g = g.assign(ts=ts).dropna(subset=["ts"])
        for r in g.itertuples(index=False):
            if r.ts < START:
                continue
            p0 = px(r.ts)
            p1 = px(r.ts + pd.Timedelta(minutes=1))
            p2 = px(r.ts + pd.Timedelta(minutes=1 + DRIFT_MINUTES))
            if not np.isfinite([p0, p1, p2]).all():
                continue
            rows.append({"release": f"{name} @ {time[:5]}", "ts": r.ts, "z": r.z,
                         "jump": 1e4 * np.log(p1 / p0), "drift": 1e4 * np.log(p2 / p1)})
    return pd.DataFrame(rows)


def jump_r2(g: pd.DataFrame) -> float:
    if len(g) < MIN_OBS_SELECT:
        return np.nan
    c = np.corrcoef(g.z, g.jump)[0, 1]
    return float(c ** 2) if np.isfinite(c) else np.nan


def signed_drift(g: pd.DataFrame, sign: int) -> np.ndarray:
    return (sign * np.sign(g.z) * g.drift).to_numpy()


def collapse_to_trades(panel: pd.DataFrame, releases: list[str], signs: dict[str, int]) -> pd.DataFrame:
    """One trade per timestamp, not per release.

    CPI MoM, CPI YoY, Core CPI MoM, Core CPI YoY, CPI Index NSA and Core CPI Index SA all
    print at 08:30 on the same morning and are six views of one number. Treating them as
    six trades would six-count a single position and inflate the t-statistic accordingly.
    Releases firing at the same instant are averaged into one direction-aligned signal and
    traded once.
    """
    sub = panel[panel.release.isin(releases)].copy()
    if sub.empty:
        return sub.assign(pnl_bps=[], signal=[])
    sub["aligned_z"] = sub.release.map(signs) * sub.z
    out = (sub.groupby("ts")
              .agg(signal=("aligned_z", "mean"), drift=("drift", "first"), k=("release", "size"))
              .reset_index())
    out = out[out.signal != 0]
    out["pnl_bps"] = np.sign(out.signal) * out.drift - TICK_BPS
    return out


def basket_stats(panel: pd.DataFrame, releases: list[str], signs: dict[str, int], label: str) -> dict:
    t = collapse_to_trades(panel, releases, signs)
    if len(t) < 25:
        return {"basket": label, "releases": len(releases), "trades": len(t)}
    pnl = t.pnl_bps.to_numpy()
    years = (t.ts.max() - t.ts.min()).days / 365.25
    tpy = len(pnl) / years
    return {"basket": label, "releases": len(releases), "trades": len(pnl),
            "trades_yr": round(tpy, 1), "bps_trade": round(pnl.mean(), 2),
            "bps_year": round(pnl.mean() * tpy, 1), "hit_%": round(100 * (pnl > 0).mean(), 1),
            "sharpe": round((pnl.mean() * tpy) / (pnl.std(ddof=1) * np.sqrt(tpy)), 2),
            "t": round(pnl.mean() / (pnl.std(ddof=1) / np.sqrt(len(pnl))), 2)}


def main() -> None:
    b = load_bloomberg()
    panel = build_release_panel(b)
    print(f"releases with >=40 usable observations: {panel.release.nunique()}")
    print(f"release-days 2012 onward: {len(panel):,}\n")

    sel = panel[panel.ts < SPLIT]
    con = panel[panel.ts >= SPLIT]

    # --- properties measured on the selection period only ---
    props = []
    for release, g in sel.groupby("release"):
        if len(g) < MIN_OBS_SELECT:
            continue
        gc = con[con.release == release]
        if len(gc) < 12:
            continue
        # sign of the reaction is part of the information, not of the P&L:
        # it comes from the jump regression, which is measurable and never traded.
        beta_jump = np.polyfit(g.z, g.jump, 1)[0]
        sign = int(np.sign(beta_jump)) or 1
        props.append({"release": release, "n_sel": len(g), "n_con": len(gc),
                      "jump_r2": jump_r2(g), "jump_beta": beta_jump, "sign": sign,
                      "abs_jump": g.jump.abs().mean(),
                      "sel_drift_bps": signed_drift(g, sign).mean(),
                      "con_drift_bps": signed_drift(gc, sign).mean()})
    p = pd.DataFrame(props).dropna(subset=["jump_r2"]).sort_values("jump_r2", ascending=False)
    signs = dict(zip(p.release, p.sign))
    p.to_csv(OUT / "news_admission_rule.csv", index=False)

    print("=" * 112)
    print("RELEASES RANKED BY JUMP RESPONSE, measured on 2012-2021 only")
    print("=" * 112)
    print(p.head(15).round(3).to_string(index=False))

    print("\n" + "=" * 112)
    print("DOES THE ADMISSION PROPERTY PREDICT OUT-OF-SAMPLE DRIFT? (2022-2026)")
    print("=" * 112)
    print(f"  releases judged: {len(p)}")
    for prop in ("jump_r2", "abs_jump", "sel_drift_bps"):
        print(f"    rank corr({prop:<14}, 2022-26 drift bps) = "
              f"{p[prop].corr(p.con_drift_bps, method='spearman'):+.3f}")

    print("\n" + "=" * 112)
    print(f"BASKETS — chosen on 2012-2021, traded on 2022-2026 (drift T+1 to T+{DRIFT_MINUTES})")
    print("=" * 112)
    by_r2 = p.nlargest(TOP_K, "jump_r2").release.tolist()
    by_pnl = p.nlargest(TOP_K, "sel_drift_bps").release.tolist()
    rest_r2 = [r for r in p.release if r not in by_r2]
    rows = [
        basket_stats(con, by_r2, signs, f"admitted: top-{TOP_K} by jump response"),
        basket_stats(con, rest_r2, signs, "rejected: everything else"),
        basket_stats(con, by_pnl, signs, f"control: top-{TOP_K} by past drift P&L"),
        basket_stats(con, p.release.tolist(), signs, "all releases"),
    ]
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n  in-sample values of the same baskets, for reference (2012-2021):")
    rows = [basket_stats(sel, by_r2, signs, f"admitted: top-{TOP_K} by jump response"),
            basket_stats(sel, rest_r2, signs, "rejected: everything else"),
            basket_stats(sel, by_pnl, signs, f"control: top-{TOP_K} by past drift P&L")]
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n" + "=" * 112)
    print("THE ADMITTED BASKET AS ACTUAL TRADES — one per timestamp, 2022-2026")
    print("=" * 112)
    trades = collapse_to_trades(con, by_r2, signs)
    trades["clock"] = pd.DatetimeIndex(trades.ts).tz_convert("America/New_York").strftime("%H:%M")
    print(f"  {'clock slot':<12} {'legs':>5} {'trades':>7} {'bps':>8} {'hit':>7} {'t':>7}")
    for clock, g in trades.groupby("clock"):
        if len(g) < 10:
            continue
        pnl = g.pnl_bps.to_numpy()
        print(f"  {clock:<12} {g.k.median():>5.0f} {len(pnl):>7} {pnl.mean():>+8.2f} "
              f"{(pnl > 0).mean():>6.1%} {pnl.mean()/(pnl.std(ddof=1)/np.sqrt(len(pnl))):>+7.2f}")
    print(f"  {'ALL':<12} {'':>5} {len(trades):>7} {trades.pnl_bps.mean():>+8.2f} "
          f"{(trades.pnl_bps > 0).mean():>6.1%} "
          f"{trades.pnl_bps.mean()/(trades.pnl_bps.std(ddof=1)/np.sqrt(len(trades))):>+7.2f}")

    print("\n  year by year:")
    trades["year"] = pd.DatetimeIndex(trades.ts).year
    for year, g in trades.groupby("year"):
        print(f"    {year}  trades {len(g):>3}  {g.pnl_bps.mean():>+7.2f} bps/trade  "
              f"{g.pnl_bps.sum():>+8.1f} bps total  hit {(g.pnl_bps > 0).mean():>5.1%}")

    trades.to_parquet(OUT / "news_admission_trades.parquet", index=False)
    print(f"\nsaved -> {OUT/'news_admission_rule.csv'}")


if __name__ == "__main__":
    main()
