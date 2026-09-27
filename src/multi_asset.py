"""
multi_asset.py
==============

Notebook 07: the surprise strategy (Notebook 05) and the DTW signal
(Notebook 06) applied to ES, NQ and ZN, then combined into one portfolio.

Every market goes through exactly the same pipeline:

1. 1-minute bars -> roll-adjusted log price (no roll jump inside a window).
2. Price measures at every release (jump, 30-minute trade window, 4-hour vol).
3. Impact screen, per market and per year, on history only: each market
   reacts to different releases (bonds care about different data than stocks).
4. Line signs from the jump regression, per market: a strong payrolls print
   pushes ES up but ZN (bond prices) down, so signs cannot be shared.
5. Strategies: CPI surprise model, multi-release naive surprise, DTW only,
   surprise + DTW agreement.

Data: ES uses this project's raw file (so ES reproduces Notebook 05 exactly);
NQ and ZN use Jack Duncan's cleaned files from the team repo
(blackrock-intraday/data/processed/futures_1min_clean_v3_{NQ,ZN}.parquet).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import state_conditioning as sc
from . import surprise_universe as su
from . import dtw_signal as ds

CPI = "CPI"

# contract specifications (Jack Duncan, combined/config.py)
SPECS = {
    "ES": dict(tick=0.25, point_value=50.0, name="E-mini S&P 500"),
    "NQ": dict(tick=0.25, point_value=20.0, name="E-mini Nasdaq-100"),
    "ZN": dict(tick=1 / 64, point_value=1000.0, name="10-year T-note"),
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_bars(path, kind: str) -> pd.DataFrame:
    """Return a DataFrame indexed by UTC minute with [close, instrument_id].

    kind = 'project' : this project's raw Databento file (instrument_id column)
    kind = 'team_v3' : Jack Duncan's cleaned file (contract column, filled minutes)
    """
    if kind == "project":
        return sc.load_es_minute(path)
    raw = pd.read_parquet(path, columns=["close", "contract"])
    raw = raw.rename(columns={"contract": "instrument_id"})
    return sc.prepare_es_minute(raw)


def adjusted(bars: pd.DataFrame) -> pd.DataFrame:
    adj, _ = sc.build_adjusted_log_price(bars)
    return adj


# ---------------------------------------------------------------------------
# One market, end to end
# ---------------------------------------------------------------------------
def run_market(sym, adj, rows, fam, surp, *, hist_start, eval_start, eval_end, eval_years,
               k_families=10, min_train=24, threshold=1.0, ticks_rt=2.0, commission_rt=4.5,
               dtw_params=None, verbose=True):
    """Full Notebook 05 + 06 pipeline for one market. Returns a dict."""
    spec = SPECS[sym]
    ts_all = rows.loc[rows["ts_utc"] >= pd.Timestamp(hist_start, tz="UTC"), "ts_utc"].unique()
    px = su.price_measures(adj, ts_all)
    px["cost_bps"] = su.cost_bps(px["pre_close"], ticks_rt, commission_rt, spec["tick"], spec["point_value"])

    screens, admitted, signs, hist = {}, {}, {}, {}
    for y in eval_years:
        screens[y] = su.impact_screen(rows, fam, px, hist_start, f"{y}-01-01")
        admitted[y] = set(screens[y].head(k_families)["family"]) if len(screens[y]) else set()
        signs[y] = su.line_signs(surp, px, hist_start, f"{y}-01-01")
        hist[y] = su.history_panel(surp, fam, px, admitted[y] | {CPI}, signs[y], hist_start, f"{y + 1}-01-01")

    fams_by_year = {y: pd.DataFrame({"family": sorted(admitted[y] | {CPI})}) for y in eval_years}
    panel = su.family_signal_panel(surp, fam, px, fams_by_year, signs, eval_years)
    panel["year"] = panel["ts_utc"].dt.tz_convert("America/New_York").dt.year
    panel["admitted"] = [f in admitted[y] for f, y in zip(panel["family"], panel["year"])]
    panel = panel[(panel["ts_utc"] >= pd.Timestamp(eval_start, tz="UTC")) &
                  (panel["ts_utc"] <= pd.Timestamp(eval_end, tz="UTC"))]

    def pos(sub, model):
        return su.build_positions(sub, hist, model, "w1", min_train, threshold)

    cpi_p = panel[panel["family"] == CPI]
    all_p = panel[panel["admitted"]]
    trades = {
        "CPI · surprise model": su.to_trades(pos(cpi_p, "surprise"), "w1", "CPI · surprise model"),
        "All · naive": su.to_trades(pos(all_p, "naive"), "w1", "All · naive"),
    }

    # ---- DTW (Notebook 06 specification)
    dp = dict(lookback_years=3, max_pool=600, k=15, band=5, half_life_years=2.0, gate_middle=0.40)
    dp.update(dtw_params or {})
    cal = (rows[rows["ts_utc"] >= pd.Timestamp(hist_start, tz="UTC")]
           .groupby("ts_utc")["clock_et"].first().reset_index().sort_values("ts_utc").reset_index(drop=True))
    cal = cal.merge(px[["ts_utc", "w1"]], on="ts_utc", how="left")
    paths = ds.zscore_rows(ds.extract_paths(adj, cal["ts_utc"]))
    qmask = (cal["ts_utc"] >= pd.Timestamp(eval_start, tz="UTC") - pd.DateOffset(years=2)).to_numpy()
    sig = ds.compute_dtw_signal(cal["ts_utc"], cal["clock_et"].to_numpy(), paths, cal["w1"].to_numpy(),
                                query_mask=qmask, lookback_years=dp["lookback_years"], max_pool=dp["max_pool"],
                                k=dp["k"], band=dp["band"], half_life_years=dp["half_life_years"],
                                progress_every=0)
    gates = ds.gate_thresholds(sig, eval_years, window_years=3, middle=dp["gate_middle"])

    def attach(t):
        t = t.merge(sig[["ts_utc", "dtw_dir", "dtw_score"]], left_on="timestamp_utc",
                    right_on="ts_utc", how="left").drop(columns="ts_utc")
        y = t["timestamp_utc"].dt.tz_convert("America/New_York").dt.year
        lo = y.map(lambda v: gates[v][0]); hi = y.map(lambda v: gates[v][1])
        s = t["dtw_score"]
        t["dtw_pos"] = pd.Series(np.where((s < lo) | (s > hi), np.sign(s), 0.0)).fillna(0.0).to_numpy()
        return t

    base_all = attach(trades["All · naive"])
    d = base_all.copy(); d["position"] = d["dtw_pos"]
    trades["All · DTW only"] = ds.recost(d, "All · DTW only")
    a = base_all.copy(); a["position"] = np.where(np.sign(a["dtw_dir"]) == np.sign(a["position"]), a["position"], 0.0)
    trades["All · surprise + DTW agree"] = ds.recost(a, "All · surprise + DTW agree")
    base_cpi = attach(trades["CPI · surprise model"])
    c = base_cpi.copy(); c["position"] = np.where(np.sign(c["dtw_dir"]) == np.sign(c["position"]), c["position"], 0.0)
    trades["CPI · surprise + DTW agree"] = ds.recost(c, "CPI · surprise + DTW agree")

    for name, t in trades.items():
        t["market"] = sym
        t["strategy"] = name

    if verbose:
        print(f"{sym}: {len(px):,} release timestamps, {px['w1'].notna().sum():,} with a trade return, "
              f"avg cost {px['cost_bps'].median():.2f} bps, DTW for {sig['dtw_dir'].notna().sum():,} releases")
    return dict(sym=sym, px=px, screens=screens, admitted=admitted, signs=signs, hist=hist,
                panel=panel, trades=trades, dtw=sig, gates=gates)


def with_costs(res, ticks_rt, commission_rt=4.5, min_train=24, threshold=1.0):
    """Re-run the surprise strategies of one market at a different cost level."""
    spec = SPECS[res["sym"]]
    p = res["panel"].copy()
    p["cost_bps"] = su.cost_bps(p["pre_close"], ticks_rt, commission_rt, spec["tick"], spec["point_value"])
    out = {}
    out["CPI · surprise model"] = su.to_trades(
        su.build_positions(p[p["family"] == CPI], res["hist"], "surprise", "w1", min_train, threshold),
        "w1", "CPI · surprise model")
    out["All · naive"] = su.to_trades(
        su.build_positions(p[p["admitted"]], res["hist"], "naive", "w1", min_train, threshold),
        "w1", "All · naive")
    return out


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------
def risk_weights(results: dict, start, end) -> pd.Series:
    """Equal-risk weights from HISTORY ONLY: inverse standard deviation of each
    market's 30-minute trade return over release windows in [start, end),
    normalised so ES = 1."""
    vol = {}
    for sym, r in results.items():
        px = r["px"]
        m = (px["ts_utc"] >= pd.Timestamp(start, tz="UTC")) & (px["ts_utc"] < pd.Timestamp(end, tz="UTC"))
        vol[sym] = px.loc[m, "w1"].std()
    w = 1.0 / pd.Series(vol)
    return w / w.get("ES", w.iloc[0])


def portfolio(results: dict, strategy: str, weights: pd.Series, name: str) -> pd.DataFrame:
    """Combine one strategy across markets, each scaled by its risk weight."""
    parts = []
    for sym, r in results.items():
        t = r["trades"][strategy].copy()
        t["position"] = t["position"] * weights[sym]
        parts.append(ds.recost(t, name).assign(market=sym))
    return pd.concat(parts).sort_values("timestamp_utc").reset_index(drop=True)
