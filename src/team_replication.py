"""
team_replication.py
===================

Replication of Jack Duncan's ES DTW event strategy (his slide), from its
published specification, so it can be priced under the same costs as the
other team strategies in Notebook 08.

Specification (Jack's slide "Dynamic Time Warping" / "When do we trade"):
* 8 release families: NFP, CPI, ISM Manufacturing, ISM Services, FOMC Minutes,
  Retail Sales, Core PCE QoQ (prints with GDP), Wholesale Inventories.
* Five back-to-back trades after each print: entries T+1, T+31, T+61, T+91,
  T+121, each held 30 minutes.
* DTW: z-scored price path over the 31 minutes ending exactly at entry,
  matched (Sakoe-Chiba band 5) against every prior release at the same clock
  slot and anchor, last 3 years, 600 most recent; k = 15 nearest, weighted by
  1/distance x recency (2-year half-life).
  direction = weighted mean of neighbours' returns; dispersion = weighted SD.
* Rule-based: skip the middle 40% of the historical direction range
  (walk-forward); otherwise trade the sign and exit after 30 minutes.
* Size inversely by dispersion (median dispersion / dispersion, capped at 2
  contracts, as in Jack's combined/config.py).

Differences from Jack's own code that cannot be removed here: his cleaned
data file and his exact family-grouping of Bloomberg lines. The replication
is therefore close, not identical, and is reported next to his slide numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import dtw_signal as ds

JACK_FAMILIES = ["Employment Report (NFP)", "CPI", "ISM Manufacturing", "ISM Services",
                 "FOMC Meeting Minutes", "Retail Sales", "GDP", "Wholesale Inventories"]
# note: Core PCE QoQ (GDPCPCEC Index) prints inside the GDP release, so its family is "GDP"
# anchor Bloomberg tickers for Jack's 8 releases (used in addition to the family names,
# so the selection does not depend on how a family happens to be named)
JACK_TICKERS = {"NFP TCH Index", "CPI CHNG Index", "NAPMPMI Index", "NAPMNMI Index", "FEDMMINU Index",
                "RSTAMOM Index", "GDPCPCEC Index", "GDP CQOQ Index", "MWINCHNG Index"}
JACK_ANCHORS = [0, 30, 60, 90, 120]      # entry = close of bar T+a  (= T+a+1 minutes)
HOLD = 30


def jack_ladder_book(rows: pd.DataFrame, fam: pd.DataFrame, es_adj: pd.DataFrame,
                     hist_start="2016-01-01", eval_years=range(2018, 2027),
                     lookback_years=3, max_pool=600, k=15, band=5, half_life=2.0,
                     gate_middle=0.40, gate_window_years=3, max_position=2.0,
                     families=JACK_FAMILIES, verbose=True) -> pd.DataFrame:
    """Trade log (one row per release timestamp x anchor) with columns
    entry_time, position, ret_bps, price, market, anchor, families."""
    fam_map = fam.set_index("ticker")["family"]
    r = rows[rows["ts_utc"] >= pd.Timestamp(hist_start, tz="UTC")].copy()
    r["family"] = r["ticker"].map(fam_map)
    minutes = r["event_name"].astype(str).str.contains("Minutes", case=False) & \
              r["event_name"].astype(str).str.contains("FOMC", case=False)
    r.loc[minutes, "family"] = "FOMC Meeting Minutes"
    r.loc[r["ticker"].isin(["GDPCPCEC Index", "GDP CQOQ Index"]), "family"] = "GDP"
    r.loc[r["ticker"].isin(JACK_TICKERS) & ~r["family"].isin(families), "family"] = \
        r.loc[r["ticker"].isin(JACK_TICKERS) & ~r["family"].isin(families), "ticker"]
    cal = (r.groupby("ts_utc").agg(clock=("clock_et", "first"),
                                    fams=("family", lambda s: sorted(set(s.dropna()))))
             .reset_index().sort_values("ts_utc").reset_index(drop=True))
    keep = set(families) | JACK_TICKERS
    cal["traded"] = cal["fams"].map(lambda f: any(x in keep for x in f))
    cal["fam_label"] = cal["fams"].map(lambda f: "+".join(x for x in f if x in keep))
    if verbose:
        found = sorted({x for f in cal.loc[cal["traded"], "fams"] for x in f if x in keep})
        print(f"Jack families found: {found}")
        print(f"release timestamps: {len(cal):,}; in Jack's 8 families: {int(cal['traded'].sum()):,}")

    idx = es_adj.index.as_unit("ns") if hasattr(es_adj.index, "as_unit") else es_adj.index
    t_ns = idx.asi8
    lp = es_adj["logp_adj"].to_numpy()
    close = es_adj["close"].to_numpy()
    T = pd.DatetimeIndex(cal["ts_utc"])
    T = T.as_unit("ns") if hasattr(T, "as_unit") else T

    def at(label, arr):
        x = T.asi8 + label * 60 * 10**9
        pos = np.searchsorted(t_ns, x, side="right") - 1
        ok = (pos >= 0) & ((x - t_ns[np.clip(pos, 0, None)]) <= 5 * 60 * 10**9)
        return np.where(ok, arr[np.clip(pos, 0, None)], np.nan)

    qmask = cal["traded"].to_numpy()
    out = []
    for a in JACK_ANCHORS:
        ret = (at(a + HOLD, lp) - at(a, lp)) * 1e4
        price = at(a, close)
        paths = ds.zscore_rows(ds.extract_paths(es_adj, cal["ts_utc"], end_offset_min=a))
        # library: every release at the same clock slot; the query uses the same anchor
        sig = ds.compute_dtw_signal(cal["ts_utc"], cal["clock"].to_numpy(), paths, ret, query_mask=qmask,
                                    lookback_years=lookback_years, max_pool=max_pool, k=k, band=band,
                                    half_life_years=half_life, progress_every=0, min_gap_min=a + HOLD + 1)
        g = pd.DataFrame({"ts_utc": cal["ts_utc"], "dir": sig["dtw_dir"], "disp": sig["dtw_disp"],
                          "ret_bps": ret, "price": price, "traded": qmask, "families": cal["fam_label"]})
        g = g[g["traded"] & g["dir"].notna()].copy()
        g["year"] = g["ts_utc"].dt.tz_convert("America/New_York").dt.year
        # walk-forward gate on the direction, and inverse-dispersion sizing, per anchor
        pos = np.zeros(len(g))
        for y in eval_years:
            hist = g[(g["year"] < y) & (g["year"] >= y - gate_window_years)]
            cur = (g["year"] == y).to_numpy()
            if len(hist) < 50 or not cur.any():
                continue
            lo, hi = hist["dir"].quantile(0.5 - gate_middle / 2), hist["dir"].quantile(0.5 + gate_middle / 2)
            med_disp = hist["disp"].median()
            d = g.loc[cur, "dir"].to_numpy()
            size = np.minimum(med_disp / g.loc[cur, "disp"].replace(0, np.nan).to_numpy(), max_position)
            pos[cur] = np.where((d < lo) | (d > hi), np.sign(d) * np.nan_to_num(size, nan=0.0), 0.0)
        g["position"] = pos
        g["anchor"] = a
        g["entry_time"] = g["ts_utc"] + pd.Timedelta(minutes=a + 1)
        out.append(g)
        if verbose:
            print(f"anchor T+{a + 1:>3d}: {int((g['position'] != 0).sum()):,} trades")
    book = pd.concat(out, ignore_index=True)
    book["market"] = "ES"
    return book[["entry_time", "position", "ret_bps", "price", "market", "anchor", "families"]]
