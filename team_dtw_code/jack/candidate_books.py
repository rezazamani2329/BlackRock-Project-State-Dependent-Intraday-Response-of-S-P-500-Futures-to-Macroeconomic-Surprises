"""Sizing up the candidate strategies with real P&L, not estimates.

Two books can be built today from work that already exists and has been checked:

  Book A  "event specialist"  — the CPI surprise overlay (jack/macro_surprise.py) plus
          the PCE path factor (hoshea_zeng, re-scored out of sample in
          jack/hoshea_grid_holdout.py). Two mechanisms, both independently confirmed.

  Book B  "pooled event ranker" — Hoshea's DTW factor across every event in the winning
          group, trading only the extreme quintiles. More trades, thinner edge each.

Both are sign trades on one ES contract, net of a full tick. Sharpe is annualised by
realised trade frequency, which is the right convention for event-driven books but is
not comparable to a daily-bar Sharpe: the capital is idle between releases.

Run: uv run python jack/candidate_books.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "jack" / "data" / "hoshea_holdout_best_panel.parquet"
CPI = ROOT / "jack" / "data" / "cpi_surprise.parquet"
TICK_BPS = 0.5
Z_FILTER = 0.5  # the CPI overlay's surprise-size filter


def stats(trades: pd.DataFrame, label: str) -> dict:
    """trades: columns ts, pnl_bps (already net)."""
    t = trades.sort_values("ts")
    pnl = t.pnl_bps.to_numpy()
    years = (t.ts.max() - t.ts.min()).days / 365.25
    tpy = len(pnl) / years
    equity = np.cumprod(1 + pnl / 1e4)
    dd = (equity / np.maximum.accumulate(equity) - 1).min()
    ann = pnl.mean() * tpy
    sharpe = ann / (pnl.std(ddof=1) * np.sqrt(tpy)) if pnl.std() > 0 else np.nan
    return {"book": label, "n": len(pnl), "yrs": round(years, 1), "trades_yr": round(tpy, 1),
            "bps_trade": round(pnl.mean(), 2), "bps_year": round(ann, 1),
            "hit_%": round(100 * (pnl > 0).mean(), 1), "sharpe": round(sharpe, 2),
            "max_dd_%": round(100 * dd, 2), "t": round(pnl.mean() / (pnl.std(ddof=1) / np.sqrt(len(pnl))), 2)}


def clustered_t(t: pd.DataFrame, cluster: str) -> float:
    m, n = t.pnl_bps.mean(), len(t)
    resid = t.groupby(cluster).pnl_bps.apply(lambda s: (s - m).sum())
    return float(m / (np.sqrt((resid ** 2).sum()) / n))


def build_cpi_leg() -> pd.DataFrame:
    cpi = pd.read_parquet(CPI)
    cpi = cpi[cpi.surprise.abs() > Z_FILTER]
    pnl = -np.sign(cpi.surprise) * cpi.drift_bps - TICK_BPS
    return pd.DataFrame({"ts": cpi.ts.to_numpy(), "pnl_bps": pnl.to_numpy(),
                         "leg": "CPI surprise", "cluster": cpi.ts.to_numpy()})


def build_pce_leg(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel[(panel.event_type == "PCE") & panel.dtw_factor.notna()]
    pnl = np.sign(p.dtw_factor) * p.future_logret * 1e4 - TICK_BPS
    return pd.DataFrame({"ts": p.t0_utc.to_numpy(), "pnl_bps": pnl.to_numpy(),
                         "leg": "PCE path", "cluster": p.event_ts.to_numpy()})


def build_pooled_leg(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel[panel.dtw_factor.notna()].copy()
    p["q"] = pd.qcut(p.dtw_factor, 5, labels=False)
    e = p[p.q.isin([0, 4])]
    pnl = np.where(e.q == 4, 1, -1) * e.future_logret * 1e4 - TICK_BPS
    return pd.DataFrame({"ts": e.t0_utc.to_numpy(), "pnl_bps": pnl,
                         "leg": "pooled Q1/Q5", "cluster": e.event_ts.to_numpy()})


def main() -> None:
    panel = pd.read_parquet(PANEL)
    cpi_leg, pce_leg = build_cpi_leg(), build_pce_leg(panel)
    book_a = pd.concat([cpi_leg, pce_leg], ignore_index=True)
    book_b = build_pooled_leg(panel)

    print("=" * 104)
    print("CANDIDATE BOOKS — one ES contract per trade, net of a full tick")
    print("=" * 104)
    rows = [stats(cpi_leg, "  CPI leg only"), stats(pce_leg, "  PCE leg only"),
            stats(book_a, "BOOK A  CPI + PCE"), stats(book_b, "BOOK B  pooled Q1/Q5")]
    print(pd.DataFrame(rows).to_string(index=False))

    print("\nclustered t on bps/trade (releases, not anchors, are the independent unit):")
    for label, t in (("CPI leg", cpi_leg), ("PCE leg", pce_leg),
                     ("BOOK A", book_a), ("BOOK B", book_b)):
        print(f"  {label:<10} naive t {t.pnl_bps.mean()/(t.pnl_bps.std(ddof=1)/np.sqrt(len(t))):>+5.2f}"
              f"   clustered t {clustered_t(t, 'cluster'):>+5.2f}")

    print("\ncorrelation between the two legs of Book A, aligned by month:")
    m_cpi = cpi_leg.set_index("ts").pnl_bps.resample("ME").sum()
    m_pce = pce_leg.set_index("ts").pnl_bps.resample("ME").sum()
    both = pd.concat([m_cpi.rename("cpi"), m_pce.rename("pce")], axis=1).fillna(0.0)
    print(f"  {both.cpi.corr(both.pce):+.3f}   (months with both active: "
          f"{((m_cpi.reindex(both.index).notna()) & (m_pce.reindex(both.index).notna())).sum()})")

    print("\n" + "=" * 104)
    print("YEAR BY YEAR, bps on one contract's notional")
    print("=" * 104)
    yearly = {}
    for label, t in (("CPI leg", cpi_leg), ("PCE leg", pce_leg), ("BOOK A", book_a), ("BOOK B", book_b)):
        g = t.copy(); g["yr"] = pd.DatetimeIndex(g.ts).year
        yearly[label] = g.groupby("yr").pnl_bps.sum().round(0)
    print(pd.DataFrame(yearly).fillna(0).astype(int).to_string())

    print("\n" + "=" * 104)
    print("SPLIT: Hoshea's confirmation boundary (2022) applied to both books")
    print("=" * 104)
    for label, t in (("BOOK A", book_a), ("BOOK B", book_b)):
        for period, sub in (("<2022", t[pd.DatetimeIndex(t.ts) < "2022-01-01"]),
                            (">=2022", t[pd.DatetimeIndex(t.ts) >= "2022-01-01"])):
            s = stats(sub, f"{label} {period}")
            print(f"  {s['book']:<18} n={s['n']:>4} {s['bps_trade']:>+6.2f} bps/trade  "
                  f"{s['bps_year']:>+7.1f} bps/yr  Sharpe {s['sharpe']:>5.2f}  "
                  f"clustered t {clustered_t(sub, 'cluster'):>+5.2f}")


if __name__ == "__main__":
    main()
