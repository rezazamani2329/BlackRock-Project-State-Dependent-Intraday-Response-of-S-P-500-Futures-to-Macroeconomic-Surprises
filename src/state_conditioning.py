"""
state_conditioning.py
=====================

Helper functions for Notebook 03 — Pre-Event State Conditioning.

Research question
-----------------
Does the pre-event market state change how ES futures respond to the SAME
macroeconomic surprise?

Design principles (no look-ahead)
---------------------------------
1. Every state variable for an event at time T uses ES data strictly before T
   (last input bar = the T-1 minute bar, the same "pre_close" used in NB02).
2. Daily features come from the last completed trading day before the event
   date (the 16:00 ET close of day D-1).
3. State variables are converted to percentiles against a TRAILING one-year
   reference distribution that ends before the event date. No full-sample
   z-scores or full-sample quantiles are used anywhere.
4. ES.c.0 is an unadjusted continuous front-month series, so multi-day
   features are computed on a roll-adjusted log price (1-minute returns across
   an instrument_id change are set to zero).

Only numpy / pandas / matplotlib (scipy optional) are required.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:  # t-distribution p-values when scipy is available
    from scipy import stats as _st
except Exception:  # pragma: no cover
    _st = None

ET = "America/New_York"
HORIZONS = [1, 5, 15, 30, 60]
DRIFT_HORIZONS = [5, 15, 30, 60]

# ---------------------------------------------------------------------------
# 1. Signed surprise definition
# ---------------------------------------------------------------------------
# One headline signal per family, signed so that x > 0 means "good news for
# equities" under the economic prior established in Notebook 02:
#   hotter inflation  -> ES lower      (sign = -1)
#   stronger payrolls -> ES higher     (sign = +1)
# The sign is fixed by theory / NB02, NOT re-estimated here.
FAMILY_SIGNAL = {
    "CPI": ("headline_cpi", -1.0),
    "PCE": ("headline_pce", -1.0),
    "NFP": ("nonfarm_payrolls", +1.0),
}


def add_signed_surprise(events: pd.DataFrame) -> pd.DataFrame:
    """Add `x` (signed headline surprise) and `abs_x` to the event table."""
    out = events.copy()
    out["x"] = np.nan
    for fam, (col, sign) in FAMILY_SIGNAL.items():
        m = out["event_type"].eq(fam)
        out.loc[m, "x"] = sign * out.loc[m, col]
    out["abs_x"] = out["x"].abs()
    return out


# ---------------------------------------------------------------------------
# 2. ES minute data and roll adjustment
# ---------------------------------------------------------------------------
def load_es_minute(path) -> pd.DataFrame:
    """Load raw ES 1-minute parquet -> DataFrame indexed by UTC timestamp
    with columns [close, instrument_id]."""
    raw = pd.read_parquet(path, columns=["close", "instrument_id"])
    return prepare_es_minute(raw)


def prepare_es_minute(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    if "timestamp_utc" in df.columns:
        df = df.set_index("timestamp_utc")
    df.index = _ns_index(pd.to_datetime(df.index, utc=True))
    df.index.name = "timestamp_utc"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])
    if "instrument_id" not in df.columns:
        df["instrument_id"] = 0
    return df[["close", "instrument_id"]]


def build_adjusted_log_price(es: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Roll-adjusted log price.

    r_t = log(C_t / C_{t-1}), set to 0 when instrument_id changes (roll).
    logp_adj = log(C_0) + cumsum(r).  Within a contract, differences of
    logp_adj equal true log returns; across a roll the artificial calendar
    spread jump is removed.

    Returns (es_adj, roll_table).
    """
    logc = np.log(es["close"].to_numpy(dtype=float))
    inst = es["instrument_id"].to_numpy()
    r = np.diff(logc, prepend=logc[0])
    roll = np.r_[False, inst[1:] != inst[:-1]]
    r_clean = np.where(roll, 0.0, r)
    logp_adj = logc[0] + np.cumsum(r_clean)

    out = es.copy()
    out["logp_adj"] = logp_adj
    out["r1"] = r_clean
    rolls = pd.DataFrame(
        {
            "timestamp_utc": es.index[roll],
            "from_instrument": inst[np.r_[roll[1:], False]],
            "to_instrument": inst[roll],
            "raw_jump_bps": r[roll] * 1e4,
        }
    )
    return out, rolls


# ---------------------------------------------------------------------------
# 3. Daily close series and daily state features
# ---------------------------------------------------------------------------
def build_daily_close(es_adj: pd.DataFrame, min_rth_bars: int = 300) -> pd.DataFrame:
    """Daily close = last bar labeled 09:30–15:59 ET (i.e. the 16:00 ET close).

    Days with fewer than `min_rth_bars` regular-hours bars (holidays, early
    closes with missing data) are dropped.
    """
    et = es_adj.index.tz_convert(ET)
    mins = et.hour * 60 + et.minute
    rth = (mins >= 9 * 60 + 30) & (mins <= 15 * 60 + 59)
    sub = es_adj.loc[rth, ["logp_adj"]].copy()
    sub["date_et"] = et[rth].date
    sub["ts"] = sub.index
    g = sub.groupby("date_et")
    daily = pd.DataFrame(
        {
            "logp_close": g["logp_adj"].last(),
            "close_ts_utc": g["ts"].last(),
            "n_rth_bars": g.size(),
        }
    )
    daily = daily[daily["n_rth_bars"] >= min_rth_bars].copy()
    daily.index = pd.to_datetime(daily.index)
    daily.index.name = "date_et"
    return daily


def compute_daily_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Daily state features known at the 16:00 ET close of each date.

    rv_20d     : annualized realized vol of daily close-to-close returns (%)
    mom_5d     : 5-day log return (%)
    mom_20d    : 20-day log return (%)
    dd_60d     : distance from the 60-day high close (%, <= 0)
    ma50_gap   : log distance from the 50-day moving average (%)
    """
    lp = daily["logp_close"]
    ret = lp.diff()
    f = pd.DataFrame(index=daily.index)
    f["rv_20d"] = ret.rolling(20, min_periods=18).std() * np.sqrt(252) * 100
    f["mom_5d"] = (lp - lp.shift(5)) * 100
    f["mom_20d"] = (lp - lp.shift(20)) * 100
    f["dd_60d"] = (lp - lp.rolling(60, min_periods=50).max()) * 100
    f["ma50_gap"] = (lp - lp.rolling(50, min_periods=45).mean()) * 100
    f["daily_close_ts_utc"] = daily["close_ts_utc"]
    return f


# ---------------------------------------------------------------------------
# 4. Intraday pre-event features at an arbitrary cutoff time T
# ---------------------------------------------------------------------------
INTRADAY_FEATURES = ["rv_pre_1d", "rv_pre_60m", "ret_pre_60m", "overnight_ret"]


def intraday_features_at(
    es_adj: pd.DataFrame,
    daily: pd.DataFrame,
    cutoffs_utc,
    rv_window_bars: int = 1380,
    max_stale_min: int = 5,
) -> pd.DataFrame:
    """Compute intraday state features using ONLY bars with timestamp < T.

    rv_pre_1d    : realized vol over the last ~23 trading hours (1380 bars),
                   from 5-minute returns, in bps (not annualized)
    rv_pre_60m   : realized vol of 1-minute returns over the last 60 bars (bps)
    ret_pre_60m  : log return over the last 60 bars (bps)
    overnight_ret: log return from prior-day 16:00 ET close to the T-1 bar (bps)

    `asof_utc` is the timestamp of the last bar used; it is always < T.
    If the last available bar is more than `max_stale_min` minutes before T
    the row is NaN.
    """
    idx = _ns_index(es_adj.index).asi8  # ns since epoch
    lp = es_adj["logp_adj"].to_numpy()
    r1 = es_adj["r1"].to_numpy()
    cut = _ns_index(pd.DatetimeIndex(pd.to_datetime(cutoffs_utc, utc=True)))

    pos = np.searchsorted(idx, cut.asi8, side="left") - 1  # last bar < T
    rows = []
    for T, p in zip(cut, pos):
        rec = {"cutoff_utc": T, "asof_utc": pd.NaT}
        rec.update({k: np.nan for k in INTRADAY_FEATURES})
        if p < rv_window_bars:
            rows.append(rec)
            continue
        asof = pd.Timestamp(idx[p], tz="UTC")
        if (T - asof) > pd.Timedelta(minutes=max_stale_min):
            rows.append(rec)
            continue
        rec["asof_utc"] = asof
        w = lp[p - rv_window_bars : p + 1]
        r5 = np.diff(w[::-1][::5][::-1])  # 5-bar returns ending at bar p
        rec["rv_pre_1d"] = np.sqrt(np.sum(r5**2)) * 1e4
        rec["rv_pre_60m"] = np.sqrt(np.sum(r1[p - 59 : p + 1] ** 2)) * 1e4
        rec["ret_pre_60m"] = (lp[p] - lp[p - 60]) * 1e4
        rows.append(rec)
    out = pd.DataFrame(rows)

    # overnight return vs the last daily close strictly before T's ET date
    dc = daily[["logp_close", "close_ts_utc"]].copy()
    dc["date_et"] = _ns(dc.index)
    dc = dc.reset_index(drop=True)
    tmp = out[["cutoff_utc"]].copy()
    tmp["date_et"] = _ns(tmp["cutoff_utc"].dt.tz_convert(ET).dt.date)
    tmp["order"] = np.arange(len(tmp))
    tmp = tmp.sort_values("date_et")
    m = pd.merge_asof(
        tmp, dc.sort_values("date_et"), on="date_et",
        allow_exact_matches=False, direction="backward",
    ).sort_values("order")
    lp_asof = np.full(len(out), np.nan)
    ok = out["asof_utc"].notna().to_numpy()
    lp_asof[ok] = lp[pos[ok]]
    out["overnight_ret"] = (lp_asof - m["logp_close"].to_numpy()) * 1e4
    out["prev_close_ts_utc"] = m["close_ts_utc"].to_numpy()
    return out


# ---------------------------------------------------------------------------
# 5. Causal (trailing) percentiles
# ---------------------------------------------------------------------------
def trailing_percentile(
    values: pd.Series,
    value_dates: pd.Series,
    ref: pd.Series,
    window_days: int = 365,
    min_obs: int = 150,
) -> np.ndarray:
    """Percentile of each value within `ref` observations dated in
    [date - window_days, date)  (strictly before the event date).

    ref : Series indexed by date (daily reference distribution).
    """
    ref = ref.dropna().sort_index()
    rd = ref.index.values.astype("datetime64[ns]")
    rv = ref.to_numpy()
    out = np.full(len(values), np.nan)
    for i, (v, d) in enumerate(zip(values.to_numpy(), pd.to_datetime(value_dates))):
        if not np.isfinite(v):
            continue
        d64 = np.datetime64(d, "ns")
        lo = np.searchsorted(rd, d64 - np.timedelta64(window_days, "D"), "left")
        hi = np.searchsorted(rd, d64, "left")
        w = rv[lo:hi]
        if len(w) < min_obs:
            continue
        out[i] = (np.sum(w < v) + 0.5 * np.sum(w == v)) / len(w)
    return out


def build_intraday_reference(
    es_adj: pd.DataFrame, daily: pd.DataFrame, clock_time_et: str
) -> pd.DataFrame:
    """Same intraday features computed at `clock_time_et` on EVERY trading
    date. This is the reference distribution for events released at that
    clock time (e.g. '08:30' or '10:00')."""
    dates = daily.index
    cut = [
        pd.Timestamp(f"{d.date()} {clock_time_et}", tz=ET).tz_convert("UTC")
        for d in dates
    ]
    f = intraday_features_at(es_adj, daily, cut)
    f.index = dates
    return f


# ---------------------------------------------------------------------------
# 6. Full event state table
# ---------------------------------------------------------------------------
DAILY_STATE = ["rv_20d", "mom_5d", "mom_20d", "dd_60d", "ma50_gap"]


def build_event_states(
    events: pd.DataFrame,
    es_adj: pd.DataFrame,
    daily: pd.DataFrame,
    daily_feats: pd.DataFrame,
    window_days: int = 365,
) -> pd.DataFrame:
    """Attach raw state variables + causal percentiles to each event."""
    ev = events.sort_values("timestamp_utc").reset_index(drop=True).copy()
    ev["date_et"] = pd.to_datetime(ev["timestamp_utc"].dt.tz_convert(ET).dt.date)
    ev["clock_et"] = ev["timestamp_utc"].dt.tz_convert(ET).dt.strftime("%H:%M")

    # --- daily features from the last trading day strictly before date_et
    dfe = daily_feats.reset_index()          # column 'date_et' = feature date
    dfe["feat_date_et"] = dfe["date_et"]
    dfe["date_et"] = _ns(dfe["date_et"])
    ev["date_et"] = _ns(ev["date_et"])
    ev = pd.merge_asof(ev, dfe.sort_values("date_et"), on="date_et",
                       allow_exact_matches=False, direction="backward")

    # --- intraday features at the event time
    intr = intraday_features_at(es_adj, daily, ev["timestamp_utc"])
    for c in INTRADAY_FEATURES:
        ev[c] = intr[c].to_numpy()
    for c in ["asof_utc", "prev_close_ts_utc"]:
        ev[c] = pd.to_datetime(intr[c].reset_index(drop=True), utc=True)

    # --- causal percentiles: daily features vs their own trailing history
    for c in DAILY_STATE:
        ev[f"{c}_pct"] = trailing_percentile(
            ev[c], ev["date_et"], daily_feats[c], window_days
        )

    # --- intraday features vs same-clock-time reference distribution
    for c in INTRADAY_FEATURES:
        ev[f"{c}_pct"] = np.nan
    for clk, grp in ev.groupby("clock_et"):
        ref = build_intraday_reference(es_adj, daily, clk)
        for c in INTRADAY_FEATURES:
            ev.loc[grp.index, f"{c}_pct"] = trailing_percentile(
                grp[c], grp["date_et"], ref[c], window_days
            )

    # --- surprise history: previous signed surprise of the same family
    ev["prev_x"] = ev.groupby("event_type")["x"].shift(1)
    # prev_x is already a z-score; map to a pseudo-percentile via the normal
    # CDF so it lives on the same 0-1 scale as the other states
    ev["prev_x_pct"] = 0.5 * (1 + _erf(ev["prev_x"].to_numpy() / np.sqrt(2)))
    return ev


def _ns_index(ix):
    """DatetimeIndex in nanosecond resolution (pandas >= 2 may use us/s)."""
    return ix.as_unit("ns") if hasattr(ix, "as_unit") else ix


def _ns(x):
    """Force datetime64[ns] (merge_asof needs identical key dtypes)."""
    return pd.to_datetime(x).astype("datetime64[ns]")


def _erf(a):
    try:
        from scipy.special import erf
        return erf(a)
    except Exception:  # pragma: no cover
        return np.vectorize(__import__("math").erf)(a)


def add_state_buckets(ev: pd.DataFrame, states) -> pd.DataFrame:
    """Low / Mid / High tercile labels from CAUSAL percentiles (fixed 1/3, 2/3
    cut-offs, so the bucket is known before the release)."""
    out = ev.copy()
    for s in states:
        p = out[f"{s}_pct"]
        out[f"{s}_bucket"] = pd.cut(
            p, [-0.001, 1 / 3, 2 / 3, 1.001], labels=["Low", "Mid", "High"]
        )
    return out


# ---------------------------------------------------------------------------
# 7. Post-release drift (tradeable) returns
# ---------------------------------------------------------------------------
def add_drift_returns(ev: pd.DataFrame, es_adj: pd.DataFrame,
                      horizons=DRIFT_HORIZONS) -> pd.DataFrame:
    """Drift after the first minute: log(P_{T+h-1} / P_T), in bps.

    NB02 returns start at the T-1 close and include the instantaneous jump,
    which cannot be captured by a strategy that reacts to the surprise. The
    drift return starts at the close of the first post-release bar (label T),
    which is the earliest realistic entry. Requires exact bars (as in NB02).
    """
    out = ev.copy()
    lp = es_adj["logp_adj"]
    t0 = out["timestamp_utc"]
    p0 = lp.reindex(t0).to_numpy()
    for h in horizons:
        ph = lp.reindex(t0 + pd.Timedelta(minutes=h - 1)).to_numpy()
        out[f"drift_{h}m_bps"] = (ph - p0) * 1e4
    return out


# ---------------------------------------------------------------------------
# 8. OLS with HC3 standard errors (numpy only)
# ---------------------------------------------------------------------------
@dataclass
class OLSResult:
    params: pd.Series
    se: pd.Series
    t: pd.Series
    p: pd.Series
    n: int
    r2: float


def ols_hc3(y, X: pd.DataFrame) -> OLSResult:
    """OLS with HC3 heteroskedasticity-robust standard errors.

    Events are non-overlapping (one row per release, returns <= 60 min), so
    serial correlation in residuals is not a concern; HC3 is the recommended
    small-sample robust estimator.
    """
    y = np.asarray(y, dtype=float)
    Xv = X.to_numpy(dtype=float)
    n, k = Xv.shape
    XtX_inv = np.linalg.pinv(Xv.T @ Xv)
    beta = XtX_inv @ Xv.T @ y
    e = y - Xv @ beta
    h = np.einsum("ij,jk,ik->i", Xv, XtX_inv, Xv)
    w = e**2 / np.clip(1 - h, 1e-8, None) ** 2
    cov = XtX_inv @ (Xv.T * w) @ Xv @ XtX_inv
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    t = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    df = max(n - k, 1)
    if _st is not None:
        p = 2 * _st.t.sf(np.abs(t), df)
    else:  # normal approximation
        p = 2 * (1 - 0.5 * (1 + _erf(np.abs(t) / np.sqrt(2))))
    sst = np.sum((y - y.mean()) ** 2)
    r2 = 1 - np.sum(e**2) / sst if sst > 0 else np.nan
    cols = X.columns
    return OLSResult(pd.Series(beta, cols), pd.Series(se, cols),
                     pd.Series(t, cols), pd.Series(p, cols), n, r2)


def bh_fdr(pvals) -> np.ndarray:
    """Benjamini–Hochberg adjusted p-values (q-values)."""
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    m = len(pv)
    if m == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    qq = np.empty(m)
    qq[order] = np.clip(ranked, 0, 1)
    q[ok] = qq
    return q


# ---------------------------------------------------------------------------
# 9. Research tests
# ---------------------------------------------------------------------------
def _sample(ev, family, ycol, state=None):
    cols = ["x", ycol, "event_type"] + ([f"{state}_pct"] if state else [])
    d = ev if family == "ALL" else ev[ev["event_type"].eq(family)]
    return d[cols + ["timestamp_utc"]].dropna()


def bucket_betas(ev, states, families, ycols, min_n: int = 15) -> pd.DataFrame:
    """Surprise beta within each Low/Mid/High state bucket:
         ret = a + b * x   estimated separately per bucket."""
    rows = []
    for s in states:
        for fam in families:
            for y in ycols:
                d = _sample(ev, fam, y, s)
                d = d.assign(bucket=ev.loc[d.index, f"{s}_bucket"])
                for b in ["Low", "Mid", "High"]:
                    db = d[d["bucket"].eq(b)]
                    if len(db) < min_n:
                        continue
                    X = pd.DataFrame({"const": 1.0, "x": db["x"]})
                    r = ols_hc3(db[y], X)
                    rows.append(dict(state=s, family=fam, target=y, bucket=b,
                                     n=r.n, beta=r.params["x"], se=r.se["x"],
                                     t=r.t["x"], mean_abs_ret=db[y].abs().mean()))
    return pd.DataFrame(rows)


def interaction_design(d: pd.DataFrame, state: str, pooled: bool) -> pd.DataFrame:
    """ret = a + b*x + c*s + d*(x*s),  s = pct - 0.5 (centered).

    b : surprise beta at the MEDIAN state
    d : change in beta moving from the 0th to the 100th state percentile
    Pooled version: family-specific intercepts and slopes, common c and d.
    """
    s = d[f"{state}_pct"] - 0.5
    X = pd.DataFrame(index=d.index)
    if pooled:
        for fam in sorted(d["event_type"].unique()):
            m = d["event_type"].eq(fam).astype(float)
            X[f"const_{fam}"] = m
            X[f"x_{fam}"] = m * d["x"]
    else:
        X["const"] = 1.0
        X["x"] = d["x"]
    X["s"] = s
    X["x_s"] = d["x"] * s
    return X


def _perm_pvalue(d, s, pooled, ycol, t_obs, n_perm, rng):
    """Permutation p-value for |t(x_s)|: the state percentile is shuffled
    across events within each family, which destroys any state-surprise
    link while keeping each family's surprise/return distribution intact."""
    y = d[ycol].to_numpy(float)
    pct = d[f"{s}_pct"].to_numpy(float)
    fam = d["event_type"].to_numpy()
    groups = [np.flatnonzero(fam == f) for f in np.unique(fam)]
    base = interaction_design(d, s, pooled)
    ix_s = base.columns.get_loc("s")
    ix_xs = base.columns.get_loc("x_s")
    Xb = base.to_numpy(float)
    xv = d["x"].to_numpy(float)
    exceed = 0
    for _ in range(n_perm):
        pp = pct.copy()
        for g in groups:
            pp[g] = pct[rng.permutation(g)]
        X = Xb.copy()
        X[:, ix_s] = pp - 0.5
        X[:, ix_xs] = xv * (pp - 0.5)
        t = _t_col(y, X, ix_xs)
        exceed += abs(t) >= t_obs
    return (exceed + 1) / (n_perm + 1)


def _t_col(y, X, j):
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    w = e**2 / np.clip(1 - h, 1e-8, None) ** 2
    v = (XtX_inv @ (X.T * w) @ X @ XtX_inv)[j, j]
    return b[j] / np.sqrt(v) if v > 0 else 0.0


def interaction_tests(ev, states, families, ycols, n_perm: int = 0,
                      perm_targets=None, seed: int = 7) -> pd.DataFrame:
    """Interaction regression for every (state, family, target).

    If n_perm > 0, a permutation p-value is added for targets in
    `perm_targets` (default: all targets)."""
    rng = np.random.default_rng(seed)
    perm_targets = set(ycols if perm_targets is None else perm_targets)
    rows = []
    for s in states:
        for fam in families:
            for y in ycols:
                d = _sample(ev, fam, y, s)
                if len(d) < 30:
                    continue
                pooled = fam == "ALL"
                X = interaction_design(d, s, pooled)
                r = ols_hc3(d[y], X)
                row = dict(state=s, family=fam, target=y, n=r.n, r2=r.r2,
                           b_x=r.params.get("x", np.nan),
                           c_state=r.params["s"], t_state=r.t["s"],
                           d_interact=r.params["x_s"], se_interact=r.se["x_s"],
                           t_interact=r.t["x_s"], p_interact=r.p["x_s"])
                if n_perm and y in perm_targets:
                    row["p_perm"] = _perm_pvalue(d, s, pooled, y,
                                                 abs(r.t["x_s"]), n_perm, rng)
                rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out["q_bh"] = bh_fdr(out["p_interact"])
    return out


def magnitude_tests(ev, states, families, ycols) -> pd.DataFrame:
    """Does the state predict the SIZE of the move?
         |ret| = a + g*|x| + h*s     (s = centered percentile)"""
    rows = []
    for s in states:
        for fam in families:
            for y in ycols:
                d = _sample(ev, fam, y, s)
                if len(d) < 30:
                    continue
                X = pd.DataFrame({"const": 1.0, "abs_x": d["x"].abs(),
                                  "s": d[f"{s}_pct"] - 0.5})
                r = ols_hc3(d[y].abs(), X)
                rows.append(dict(state=s, family=fam, target=y, n=r.n, r2=r.r2,
                                 g_abs_x=r.params["abs_x"], h_state=r.params["s"],
                                 t_state=r.t["s"], p_state=r.p["s"]))
    out = pd.DataFrame(rows)
    if len(out):
        out["q_bh"] = bh_fdr(out["p_state"])
    return out


def subperiod_stability(ev, states, families, ycols, split_date="2021-01-01"):
    """Re-estimate the interaction in two subperiods; report sign agreement."""
    split = pd.Timestamp(split_date, tz="UTC")
    parts = {"early": ev[ev["timestamp_utc"] < split],
             "late": ev[ev["timestamp_utc"] >= split]}
    res = {k: interaction_tests(v, states, families, ycols)
           for k, v in parts.items()}
    key = ["state", "family", "target"]
    cols = key + ["n", "d_interact", "t_interact"]
    m = res["early"][cols].merge(res["late"][cols], on=key,
                                 suffixes=("_early", "_late"))
    m["same_sign"] = np.sign(m["d_interact_early"]) == np.sign(m["d_interact_late"])
    return m


def select_states(inter: pd.DataFrame, stab: pd.DataFrame, primary_targets,
                  q_max: float = 0.10, p_perm_max: float = 0.05,
                  min_horizon_agree: float = 0.6) -> pd.DataFrame:
    """Pre-registered selection rule for which (state, family) pairs go into
    Notebook 04. A pair passes when, on a primary target:
      1. BH-adjusted q < q_max (full sample)
      2. permutation p < p_perm_max
      3. interaction has the same sign in both subperiods
      4. the interaction sign agrees on >= min_horizon_agree of all targets
    """
    key = ["state", "family", "target"]
    m = inter.merge(stab[key + ["same_sign"]], on=key, how="left")
    agree = (
        m.groupby(["state", "family"])["d_interact"]
        .apply(lambda v: max((v > 0).mean(), (v < 0).mean()))
        .rename("horizon_sign_agreement").reset_index()
    )
    prim = m[m["target"].isin(primary_targets)].merge(agree, on=["state", "family"])
    prim["pass_fdr"] = prim["q_bh"] < q_max
    prim["pass_perm"] = prim.get("p_perm", pd.Series(np.nan, index=prim.index)) < p_perm_max
    prim["pass_stability"] = prim["same_sign"].fillna(False).astype(bool)
    prim["pass_horizons"] = prim["horizon_sign_agreement"] >= min_horizon_agree
    prim["selected"] = prim[["pass_fdr", "pass_perm", "pass_stability",
                             "pass_horizons"]].all(axis=1)
    return prim.sort_values(["selected", "q_bh"], ascending=[False, True])
