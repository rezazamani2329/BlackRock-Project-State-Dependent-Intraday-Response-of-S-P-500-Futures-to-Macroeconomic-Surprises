"""
dtw_signal.py
=============

Dynamic Time Warping (DTW) pattern signal for Notebook 06, following the
specification of the peer DTW strategy (Jack Duncan, Hoshea Zeng):

* Query path: the ES price path over the 31 minutes ending at entry, i.e. from
  T-30 to T+1 (the close of the first post-release bar). It contains the
  first minute of the market's reaction, as in Jack's "reaction path" design.
  The path is z-scored (shape only), as in the peer strategy.
* Library: earlier releases at the SAME clock time (e.g. all 08:30 releases),
  from the last 3 years, at most the 600 most recent, and only releases whose
  30-minute outcome was already known at time T (release time < T - 30 min).
* Matching: DTW distance with a Sakoe-Chiba band of 5 minutes; the k = 15
  nearest paths, weighted by 1/distance x recency (2-year half-life).
* Output: dtw_dir  = weighted mean of the neighbours' T+1 -> T+30 return (bps)
          dtw_disp = weighted standard deviation of those returns (bps)
          dtw_score = dtw_dir / dtw_disp (scale-free direction)

Pure numpy: no compiled DTW library is required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PATH_LEN = 32          # log prices at bar labels T-31 ... T  (prices at T-30 ... T+1)


def extract_paths(es_adj: pd.DataFrame, timestamps, stale_min: int = 5,
                  end_offset_min: int = 0):
    """Matrix (N, 32) of log prices at the close of bars T-31 ... T, relative
    to the first point. Rows with a missing or stale bar are NaN.
    end_offset_min shifts the whole path: the path ends at bar T + end_offset_min
    (used for the later windows of the ladder test)."""
    idx = es_adj.index
    idx = idx.as_unit("ns") if hasattr(idx, "as_unit") else idx
    t_ns = idx.asi8
    lp = es_adj["logp_adj"].to_numpy()
    T = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    T = T.as_unit("ns") if hasattr(T, "as_unit") else T
    offs = (np.arange(-(PATH_LEN - 1), 1) + end_offset_min) * 60 * 10**9   # -31 ... 0 (+ offset)
    labels = T.asi8[:, None] + offs[None, :]
    pos = np.searchsorted(t_ns, labels.ravel(), side="right").reshape(labels.shape) - 1
    posc = np.clip(pos, 0, None)
    ok = (pos >= 0) & ((labels - t_ns[posc]) <= stale_min * 60 * 10**9)
    P = lp[posc]
    P = P - P[:, :1]
    P[~ok.all(axis=1)] = np.nan
    return P


def zscore_rows(P):
    m = np.nanmean(P, axis=1, keepdims=True)
    s = np.nanstd(P, axis=1, keepdims=True)
    s[s == 0] = np.nan
    Z = (P - m) / s
    Z[np.isnan(s[:, 0])] = 0.0
    return Z


def dtw_batch(q: np.ndarray, C: np.ndarray, band: int = 5) -> np.ndarray:
    """DTW distances between one query q (n,) and K candidates C (K, n),
    squared-error cost, Sakoe-Chiba band. Vectorised over candidates."""
    n = len(q)
    K = C.shape[0]
    D = np.full((K, n + 1, n + 1), np.inf)
    D[:, 0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(1, i - band), min(n, i + band) + 1):
            cost = (q[i - 1] - C[:, j - 1]) ** 2
            D[:, i, j] = cost + np.minimum(np.minimum(D[:, i - 1, j], D[:, i, j - 1]), D[:, i - 1, j - 1])
    return np.sqrt(D[:, n, n])


def compute_dtw_signal(ts, clock, paths, outcome, query_mask=None,
                       lookback_years: float = 3.0, max_pool: int = 600,
                       k: int = 15, band: int = 5, half_life_years: float = 2.0,
                       min_pool: int = 30, progress_every: int = 1000,
                       min_gap_min: int = 30) -> pd.DataFrame:
    """DTW direction and dispersion for every release timestamp.

    ts        : sorted DatetimeIndex (UTC) of release timestamps
    clock     : array of 'HH:MM' ET clock slots, aligned with ts
    paths     : (N, 32) z-scored paths (NaN rows are skipped)
    outcome   : (N,) the T+1 -> T+30 return in bps (the neighbours' 'answer')
    query_mask: which rows to compute (default: all)
    min_gap_min: a neighbour is used only if its release was at least this many
                 minutes before T, so its outcome was already known (30 for
                 the first window; longer for later windows of the ladder)
    """
    ts = pd.DatetimeIndex(ts)
    t_ns = ts.asi8
    N = len(ts)
    valid = ~np.isnan(paths).any(axis=1) & np.isfinite(outcome)
    qmask = np.ones(N, bool) if query_mask is None else np.asarray(query_mask)
    look = int(lookback_years * 365.25 * 86400 * 1e9)
    thirty = min_gap_min * 60 * 10**9
    year_ns = 365.25 * 86400 * 1e9
    clock = np.asarray(clock)
    by_clock = {c: np.flatnonzero(clock == c) for c in np.unique(clock)}

    out_dir = np.full(N, np.nan)
    out_disp = np.full(N, np.nan)
    out_pool = np.zeros(N, int)
    out_d1 = np.full(N, np.nan)
    done = 0
    for i in np.flatnonzero(qmask):
        if np.isnan(paths[i]).any():
            continue
        cand = by_clock[clock[i]]
        cand = cand[(t_ns[cand] < t_ns[i] - thirty) & (t_ns[cand] >= t_ns[i] - look) & valid[cand]]
        if len(cand) > max_pool:
            cand = cand[-max_pool:]
        out_pool[i] = len(cand)
        if len(cand) < max(min_pool, k):
            continue
        d = dtw_batch(paths[i], paths[cand], band)
        nn = np.argsort(d)[:k]
        c = cand[nn]
        age = (t_ns[i] - t_ns[c]) / year_ns
        w = (1.0 / (d[nn] + 1e-9)) * 0.5 ** (age / half_life_years)
        w = w / w.sum()
        y = outcome[c]
        mu = np.sum(w * y)
        out_dir[i] = mu
        out_disp[i] = np.sqrt(np.sum(w * (y - mu) ** 2))
        out_d1[i] = d[nn][0]
        done += 1
        if progress_every and done % progress_every == 0:
            print(f"  DTW computed for {done:,} releases ...")
    res = pd.DataFrame({"ts_utc": ts, "clock": clock, "dtw_dir": out_dir,
                        "dtw_disp": out_disp, "pool": out_pool, "nn_dist": out_d1})
    res["dtw_score"] = res["dtw_dir"] / res["dtw_disp"].replace(0, np.nan)
    return res


def gate_thresholds(sig: pd.DataFrame, years, window_years: int = 3,
                    middle: float = 0.40) -> dict:
    """Walk-forward rank gate (peer strategy): for each year, the percentile
    cut-offs of dtw_score over the previous `window_years`. A release is traded
    only if its score is outside the middle `middle` share of that range."""
    out = {}
    lo_q, hi_q = 0.5 - middle / 2, 0.5 + middle / 2
    for y in years:
        a = pd.Timestamp(f"{y - window_years}-01-01", tz="UTC")
        b = pd.Timestamp(f"{y}-01-01", tz="UTC")
        s = sig.loc[(sig["ts_utc"] >= a) & (sig["ts_utc"] < b), "dtw_score"].dropna()
        out[y] = (s.quantile(lo_q), s.quantile(hi_q), len(s)) if len(s) >= 50 else (np.nan, np.nan, len(s))
    return out


def recost(t: pd.DataFrame, name: str) -> pd.DataFrame:
    """Recompute P&L columns of a trade log after positions change."""
    t = t.copy()
    t["gross_bps"] = t["position"] * t["ret_bps"].fillna(0)
    t["tc_bps"] = t["position"].abs() * t["cost_bps"].fillna(0)
    t["net_bps"] = t["gross_bps"] - t["tc_bps"]
    t["strategy"] = name
    return t


# ---------------------------------------------------------------------------
# Ladder test: several consecutive 30-minute windows after each release
# ---------------------------------------------------------------------------
LADDER = [(0, 29), (29, 59), (59, 89), (89, 119), (119, 149)]   # (entry bar, exit bar) after T
"""Window 1 = close of bar T -> close of bar T+29 (price T+1 -> T+30), as in
Notebooks 04-06; window k enters where window k-1 exits. Five windows cover
roughly T+1 -> T+150, like the peer strategy's T+1, T+31, ..., T+121 ladder."""


def ladder_returns(es_adj: pd.DataFrame, timestamps, windows=LADDER, stale_min: int = 5):
    """Returns (bps) of each ladder window for each release timestamp."""
    idx = es_adj.index
    idx = idx.as_unit("ns") if hasattr(idx, "as_unit") else idx
    t_ns = idx.asi8
    lp = es_adj["logp_adj"].to_numpy()
    T = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    T = T.as_unit("ns") if hasattr(T, "as_unit") else T

    def at(label):
        ts = T.asi8 + label * 60 * 10**9
        pos = np.searchsorted(t_ns, ts, side="right") - 1
        ok = (pos >= 0) & ((ts - t_ns[np.clip(pos, 0, None)]) <= stale_min * 60 * 10**9)
        return np.where(ok, lp[np.clip(pos, 0, None)], np.nan)

    out = pd.DataFrame({"ts_utc": T})
    for k, (a, b) in enumerate(windows, start=1):
        out[f"r{k}"] = (at(b) - at(a)) * 1e4
    return out
