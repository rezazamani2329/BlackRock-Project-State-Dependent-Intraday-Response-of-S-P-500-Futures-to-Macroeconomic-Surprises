"""Does combining the two information sources beat either alone — generally?

Every result in this project so far is one team member's signal on the events where it
happened to work. The question a combined strategy actually rests on is whether the two
*independent* sources of directional information add to each other across the whole
event universe, not on a hand-picked pair of releases.

  source 1  path shape   — Hoshea's DTW event-game factor, same-event matching,
                           one fixed parameter set, computed over all 13 event types by
                           `jack/event_tradeability_rule.py`.
  source 2  news         — the signed standardised Bloomberg surprise, entered at a lag
                           so it is executable, built as in `jack/macro_surprise.py`.

They are independent by construction: one is the tape, the other is the consensus error.
Three models are fitted on identical games, walk-forward by year on an expanding window,
and scored on the same out-of-sample rows:

  path only | news only | both

Features are standardised within event type using training rows only, so no event's scale
leaks across the split and the pooled fit is not dominated by whichever release happens to
be weekly. Trading rule is the same for all three: long the top quintile of the fitted
prediction, short the bottom, one contract, net of a full tick.

Run: uv run python jack/combined_event_model.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "jack" / "data" / "event_tradeability_panel.parquet"
BLOOMBERG_FILE = ROOT / "data" / "raw" / "Bloomberg Economic Releases.xlsx"
OUT = ROOT / "jack" / "data"
WINSOR_SIGMA = 5.0
TICK_BPS = 0.5
FIRST_TEST_YEAR = 2016

# calendar event type -> the Bloomberg releases that carry its surprise
CONSENSUS_NAMES = {
    "CPI": ["CPI MoM", "Core CPI MoM"],
    "NFP": ["Change in Nonfarm Payrolls"],
    "PCE": ["Core PCE Price Index MoM"],
    "GDP": ["GDP Annualized QoQ"],
    "PPI": ["PPI Final Demand MoM"],
    "Retail Sales": ["Retail Sales Advance MoM"],
    "Industrial Production": ["Industrial Production MoM"],
    "Initial Jobless Claims": ["Initial Jobless Claims"],
    "JOLTS": ["JOLTS Job Openings"],
    "Import/Export Prices": ["Import Price Index MoM"],
    "Employment Cost Index": ["Employment Cost Index"],
}
# sign that turns a positive surprise into a long ES position, from the economics
NEWS_SIGN = {"CPI": -1, "PCE": -1, "PPI": -1, "Import/Export Prices": -1,
             "Employment Cost Index": -1, "NFP": +1, "GDP": +1, "Retail Sales": +1,
             "Industrial Production": +1, "JOLTS": +1, "Initial Jobless Claims": -1}


def load_surprises() -> pd.DataFrame:
    sheets = pd.ExcelFile(BLOOMBERG_FILE)
    frames = []
    for sheet in sheets.sheet_names:
        part = pd.read_excel(BLOOMBERG_FILE, sheet_name=sheet)
        part.columns = [c.split(".")[-1] if "DROPNA" in str(c) else c for c in part.columns]
        frames.append(part)
    b = pd.concat(frames, ignore_index=True)
    b["RELEASE_DATE"] = pd.to_datetime(b.RELEASE_DATE, errors="coerce")
    b = b.dropna(subset=["RELEASE_DATE"])
    b = b[b.SURVEY_MEDIAN.notna() & b.ACTUAL.notna()]

    rows = []
    for event_type, names in CONSENSUS_NAMES.items():
        parts = []
        for name in names:
            g = b[b.EVENT_NAME == name].sort_values("RELEASE_DATE").copy()
            if len(g) < 40:
                continue
            g["surprise"] = g.ACTUAL - g.SURVEY_MEDIAN
            scale = g.surprise.expanding().std().shift(1)  # prior releases only
            g["z"] = (g.surprise / scale).clip(-WINSOR_SIGMA, WINSOR_SIGMA)
            g = g[np.isfinite(g.z) & (scale > 0)]
            parts.append(g.groupby(g.RELEASE_DATE.dt.normalize()).z.mean())
        if not parts:
            continue
        z = pd.concat(parts, axis=1).mean(axis=1)  # components of one release get averaged
        rows.append(pd.DataFrame({"release_date": z.index, "news_z": z.to_numpy() * NEWS_SIGN[event_type],
                                  "event_type": event_type}))
    return pd.concat(rows, ignore_index=True)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, feats: list[str]) -> np.ndarray:
    """Pooled OLS with an intercept. Features are already standardised within event type."""
    X = np.column_stack([np.ones(len(train))] + [train[f].to_numpy() for f in feats])
    y = train.future_logret.to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    Xt = np.column_stack([np.ones(len(test))] + [test[f].to_numpy() for f in feats])
    return Xt @ beta


def standardise_within_event(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]):
    """Scale each feature by its training-period mean/sd inside each event type."""
    train, test = train.copy(), test.copy()
    for col in cols:
        stats = train.groupby("event_type")[col].agg(["mean", "std"])
        for frame in (train, test):
            mu = frame.event_type.map(stats["mean"])
            sd = frame.event_type.map(stats["std"]).replace(0, np.nan)
            frame[col] = ((frame[col] - mu) / sd).fillna(0.0)
    return train, test


def clustered_t(pnl: np.ndarray, cluster: np.ndarray) -> float:
    m, n = pnl.mean(), len(pnl)
    resid = pd.Series(pnl - m).groupby(cluster).sum()
    return float(m / (np.sqrt((resid ** 2).sum()) / n))


def evaluate(preds: pd.DataFrame, label: str) -> dict:
    p = preds.dropna(subset=["pred"])
    q = pd.qcut(p.pred, 5, labels=False, duplicates="drop")
    ext = p[q.isin([0, q.max()])].copy()
    side = np.where(q[q.isin([0, q.max()])] == q.max(), 1, -1)
    gross = side * ext.future_logret.to_numpy() * 1e4
    net = gross - TICK_BPS
    years = (p.t0_utc.max() - p.t0_utc.min()).days / 365.25
    tpy = len(net) / years
    return {"model": label, "n_scored": len(p),
            "ic": round(float(p.pred.corr(p.future_logret)), 4),
            "n_trades": len(net), "trades_yr": round(tpy, 1),
            "bps_trade": round(net.mean(), 2), "bps_year": round(net.mean() * tpy, 1),
            "hit_%": round(100 * (net > 0).mean(), 1),
            "sharpe": round((net.mean() * tpy) / (net.std(ddof=1) * np.sqrt(tpy)), 2),
            "t_clust": round(clustered_t(net, ext.event_ts.to_numpy()), 2)}


def main() -> None:
    panel = pd.read_parquet(PANEL)
    panel = panel.dropna(subset=["dtw_factor"]).copy()
    panel["release_date"] = pd.DatetimeIndex(panel.event_ts).tz_convert("UTC").normalize().tz_localize(None)
    panel["year"] = pd.DatetimeIndex(panel.t0_utc).year

    news = load_surprises()
    panel = panel.merge(news, on=["release_date", "event_type"], how="left")
    have = panel.news_z.notna()
    print(f"games with a DTW factor: {len(panel):,}")
    print(f"  of which also carry a consensus surprise: {have.sum():,} ({have.mean():.0%})")
    print(f"  event types with news: {sorted(panel.loc[have, 'event_type'].unique())}\n")

    # Games with no consensus still trade on path alone; news_z = 0 is the neutral prior.
    panel["news_z"] = panel.news_z.fillna(0.0)
    panel["has_news"] = have.astype(float)

    specs = {"path only": ["dtw_factor"], "news only": ["news_z"], "both": ["dtw_factor", "news_z"]}
    frames = {k: [] for k in specs}
    years = sorted(y for y in panel.year.unique() if y >= FIRST_TEST_YEAR)
    for year in years:
        train_raw = panel[panel.year < year]
        test_raw = panel[panel.year == year]
        if len(train_raw) < 300 or len(test_raw) == 0:
            continue
        train, test = standardise_within_event(train_raw, test_raw, ["dtw_factor", "news_z"])
        for label, feats in specs.items():
            t = test.copy()
            t["pred"] = fit_predict(train, test, feats)
            frames[label].append(t)

    print("=" * 108)
    print("WALK-FORWARD, all event types pooled. Trade the extreme quintiles, net of a full tick.")
    print("=" * 108)
    results = {k: pd.concat(v, ignore_index=True) for k, v in frames.items() if v}
    print(pd.DataFrame([evaluate(v, k) for k, v in results.items()]).to_string(index=False))

    print("\n" + "=" * 108)
    print("SAME MODELS, restricted to games that carry a consensus surprise")
    print("=" * 108)
    rows = []
    for k, v in results.items():
        sub = v[v.has_news > 0]
        if len(sub) > 200:
            rows.append(evaluate(sub, k))
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n" + "=" * 108)
    print("THE COMBINED MODEL, per event type")
    print("=" * 108)
    both = results["both"]
    print(f"  {'event':<24} {'n':>5} {'IC':>8} {'trades':>7} {'bps':>7} {'hit':>7} {'t_clust':>8}")
    for event_type, g in both.groupby("event_type"):
        if len(g) < 120:
            continue
        s = evaluate(g, event_type)
        print(f"  {event_type:<24} {s['n_scored']:>5} {s['ic']:>+8.4f} {s['n_trades']:>7} "
              f"{s['bps_trade']:>+7.2f} {s['hit_%']:>6.1f}% {s['t_clust']:>+8.2f}")

    print("\n" + "=" * 108)
    print("YEAR BY YEAR, the combined model")
    print("=" * 108)
    rows = []
    for year, g in both.groupby("year"):
        if len(g) < 60:
            continue
        s = evaluate(g, str(year))
        rows.append({"year": year, "n": s["n_scored"], "ic": s["ic"],
                     "trades": s["n_trades"], "bps_trade": s["bps_trade"], "hit_%": s["hit_%"]})
    print(pd.DataFrame(rows).to_string(index=False))

    both.to_parquet(OUT / "combined_event_predictions.parquet", index=False)
    print(f"\nsaved -> {OUT/'combined_event_predictions.parquet'}")


if __name__ == "__main__":
    main()
