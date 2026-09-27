"""
surprise_universe.py
====================

Helper functions for Notebook 05: scaling the surprise strategy from CPI to
many US macro releases, on ES only.

Pipeline
--------
1. Read every US release line from the Bloomberg workbook (one line = one
   Bloomberg ticker, e.g. "CPI CHNG Index").
2. Group lines that print at the same instant into release FAMILIES
   (e.g. CPI MoM, Core CPI MoM, CPI YoY ... -> one "CPI" family), using
   Jack Duncan's rule: lines sharing >= 80% of their release timestamps.
3. Causally standardised surprise per line: (actual - consensus) divided by
   the standard deviation of that line's PREVIOUS surprises, winsorised at
   +/- 5 sigma (Aaryen Mehta's winsorisation).
4. ES price measures for every release timestamp: the instant jump
   (not tradeable), the 30-minute tradeable drift, the next 30 minutes, the
   total 30-minute reaction and the 4-hour pre-release volatility.
5. Impact screen (Jack Duncan): once a year, rank families by how much
   bigger the ES reaction is on their release than at the same clock slot
   when a DIFFERENT family is releasing; admit the top K. History only.
6. Direction: each line's sign comes from the regression of the instant jump
   on its surprise, estimated on history only (never on drift P&L). The
   family signal x is the average of sign * z across its lines, so x > 0
   always means "good news for ES".

Everything that is estimated uses information strictly before the year (or
the release) in which it is used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ET = "America/New_York"

BLOOMBERG_FIELDS = ["COUNTRY_NAME", "RELEASE_DATE", "RELEASE_TIME", "EVENT_NAME",
                    "PERIOD", "SURVEY_MEDIAN", "ACTUAL", "PRIOR", "REVISION",
                    "SCALING_FACTOR"]


# ---------------------------------------------------------------------------
# 1. Bloomberg workbook
# ---------------------------------------------------------------------------
def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    ren = {}
    for c in df.columns:
        for k in BLOOMBERG_FIELDS:
            if str(c).endswith("." + k):
                ren[c] = k.lower()
    df = df.rename(columns=ren)
    other = [c for c in df.columns if c not in ren.values() and c != "ID"]
    if other:
        df = df.rename(columns={other[-1]: "ticker"})
    return df


def load_bloomberg(path) -> pd.DataFrame:
    """All yearly sheets -> one row per (ticker, release timestamp)."""
    sheets = pd.read_excel(path, sheet_name=None)
    df = pd.concat([_clean_columns(d) for d in sheets.values()], ignore_index=True)
    df = df[df["ticker"].notna()].copy()
    df = df[df["release_time"].map(lambda v: hasattr(v, "hour"))].copy()
    naive = pd.to_datetime(df["release_date"].dt.strftime("%Y-%m-%d") + " "
                           + df["release_time"].astype(str), errors="coerce")
    df = df[naive.notna()].copy()
    naive = naive[naive.notna()]
    df["ts_et"] = naive.dt.tz_localize(ET, ambiguous="NaT", nonexistent="NaT")
    df = df[df["ts_et"].notna()].copy()
    df["ts_utc"] = df["ts_et"].dt.tz_convert("UTC")
    df["clock_et"] = df["ts_et"].dt.strftime("%H:%M")
    for c in ["actual", "survey_median"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = (df.sort_values(["ticker", "ts_utc"])
            .drop_duplicates(["ticker", "ts_utc"], keep="last")
            .reset_index(drop=True))
    return df[["ticker", "event_name", "ts_et", "ts_utc", "clock_et", "period",
               "survey_median", "actual"]]


FAMILY_NAMES = {
    "CPI CHNG Index": "CPI",
    "NFP TCH Index": "Employment Report (NFP)",
    "PCE CMOM Index": "Personal Income / PCE",
    "GDP CQOQ Index": "GDP",
    "RSTAMOM Index": "Retail Sales",
    "FDIDFDMO Index": "PPI",
    "NAPMPMI Index": "ISM Manufacturing",
    "NAPMNMI Index": "ISM Services",
    "FDTR Index": "FOMC Rate Decision",
    "INJCJC Index": "Jobless Claims",
    "JOLTTOTL Index": "JOLTS",
    "SPCS20SM Index": "Case-Shiller 20-City",
    "MTIBCHNG Index": "Business Inventories",
    "MWINCHNG Index": "Wholesale Inventories",
    "DGNOCHNG Index": "Durable Goods",
    "CONSSENT Index": "U. of Michigan Sentiment",
}

# ---------------------------------------------------------------------------
# 2. Release families
# ---------------------------------------------------------------------------
def build_families(rows: pd.DataFrame, start="2013-01-01", min_releases: int = 30,
                   overlap: float = 0.8) -> pd.DataFrame:
    """Group tickers whose release timestamps overlap by >= `overlap`, measured
    against the LARGER of the two sets (so a monthly survey that happens to
    print on a Thursday is not merged into weekly jobless claims).

    Returns one row per ticker: ticker, family, family_name, n_releases, clock_et.
    """
    r = rows[rows["ts_utc"] >= pd.Timestamp(start, tz="UTC")]
    sets = r.groupby("ticker")["ts_utc"].apply(lambda s: set(s.astype("int64")))
    sets = sets[sets.map(len) >= min_releases]
    tickers = list(sets.index)
    parent = {t: t for t in tickers}

    def find(t):
        while parent[t] != t:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t

    # candidate pairs share at least one timestamp
    ts_to_t = {}
    for t, s in sets.items():
        for v in s:
            ts_to_t.setdefault(v, []).append(t)
    seen = set()
    for group in ts_to_t.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if (a, b) in seen:
                    continue
                seen.add((a, b))
                inter = len(sets[a] & sets[b])
                if inter / max(len(sets[a]), len(sets[b])) >= overlap:
                    parent[find(a)] = find(b)

    fam = pd.DataFrame({"ticker": tickers})
    fam["root"] = fam["ticker"].map(find)
    info = r.groupby("ticker").agg(event_name=("event_name", lambda s: s.mode().iat[0]),
                                   n_releases=("ts_utc", "nunique"),
                                   clock_et=("clock_et", lambda s: s.mode().iat[0]),
                                   has_survey=("survey_median", lambda s: s.notna().mean()))
    fam = fam.join(info, on="ticker")
    # family name: the most-released line with a survey, else the most-released line
    def name_of(g):
        g2 = g.sort_values(["has_survey", "n_releases"], ascending=False)
        return g2["event_name"].iat[0]
    names = fam.groupby("root").apply(name_of)
    # readable names for the well-known releases (keyed by an anchor ticker)
    anchor = fam[fam["ticker"].isin(FAMILY_NAMES)].set_index("root")["ticker"].map(FAMILY_NAMES)
    anchor = anchor[~anchor.index.duplicated()]
    names.update(anchor)
    fam["family"] = fam["root"].map(names)
    fam["n_lines"] = fam.groupby("root")["ticker"].transform("size")
    return fam.drop(columns="root")


# ---------------------------------------------------------------------------
# 3. Causal surprises
# ---------------------------------------------------------------------------
def standardise_surprises(rows: pd.DataFrame, min_history: int = 24,
                          winsor: float = 5.0) -> pd.DataFrame:
    """z = (actual - consensus) / std of the line's PREVIOUS raw surprises."""
    s = rows.dropna(subset=["actual", "survey_median"]).copy()
    s = s.sort_values(["ticker", "ts_utc"])
    s["raw"] = s["actual"] - s["survey_median"]
    g = s.groupby("ticker")["raw"]
    prev_std = g.transform(lambda v: v.shift(1).expanding(min_periods=min_history).std())
    s["z"] = (s["raw"] / prev_std.replace(0, np.nan)).clip(-winsor, winsor)
    return s


# ---------------------------------------------------------------------------
# 4. ES price measures at release timestamps
# ---------------------------------------------------------------------------
def price_measures(es_adj: pd.DataFrame, timestamps, stale_min: int = 5,
                   vol_bars: int = 240) -> pd.DataFrame:
    """For each release timestamp T (UTC), in bps:

    jump      = P(T+1) - P(T)        close of bar T vs close of bar T-1; the
                                      instant reaction, NOT tradeable
    w1        = P(T+30) - P(T+1)     entry at the close of the first
                                      post-release bar, exit 30 min after T
                                      (identical to NB04's tr_L0_H30)
    w2        = P(T+60) - P(T+30)    the next 30 minutes (second window)
    react30   = P(T+30) - P(T)       total 30-minute reaction (impact screen)
    vol_4h    = std of 1-minute returns over the 240 bars before T (bps)
    pre_close = raw price at T-1 (for costs)
    """
    idx = es_adj.index
    idx = idx.as_unit("ns") if hasattr(idx, "as_unit") else idx
    t_ns = idx.asi8
    lp = es_adj["logp_adj"].to_numpy()
    r1 = es_adj["r1"].to_numpy()
    close = es_adj["close"].to_numpy()
    T = pd.DatetimeIndex(pd.to_datetime(pd.Series(timestamps).unique(), utc=True)).sort_values()
    T = T.as_unit("ns") if hasattr(T, "as_unit") else T
    stale = stale_min * 60 * 10**9

    def at(label_min):
        """log price at the close of the bar labelled T + label_min (as-of)."""
        ts = (T + pd.Timedelta(minutes=label_min)).asi8
        pos = np.searchsorted(t_ns, ts, side="right") - 1
        ok = (pos >= 0) & ((ts - t_ns[np.clip(pos, 0, None)]) <= stale)
        return np.where(ok, lp[np.clip(pos, 0, None)], np.nan), pos, ok

    p_pre, pos_pre, ok_pre = at(-1)
    p0, _, _ = at(0)
    p29, _, _ = at(29)
    p59, _, _ = at(59)

    # 4-hour pre-release volatility from cumulative sums (bars strictly before T)
    c1 = np.r_[0.0, np.cumsum(r1)]
    c2 = np.r_[0.0, np.cumsum(r1 ** 2)]
    hi = pos_pre + 1
    lo = np.clip(hi - vol_bars, 0, None)
    n = hi - lo
    mean = (c1[hi] - c1[lo]) / np.maximum(n, 1)
    var = (c2[hi] - c2[lo]) / np.maximum(n, 1) - mean ** 2
    vol = np.sqrt(np.clip(var, 0, None)) * np.sqrt(n / np.maximum(n - 1, 1)) * 1e4
    vol = np.where(ok_pre & (n >= vol_bars // 2), vol, np.nan)

    return pd.DataFrame({
        "ts_utc": T,
        "jump": (p0 - p_pre) * 1e4,
        "w1": (p29 - p0) * 1e4,
        "w2": (p59 - p29) * 1e4,
        "react30": (p29 - p_pre) * 1e4,
        "vol_4h": vol,
        "pre_close": np.where(ok_pre, close[np.clip(pos_pre, 0, None)], np.nan),
    })


# ---------------------------------------------------------------------------
# 5. Impact screen (yearly, history only)
# ---------------------------------------------------------------------------
def impact_screen(rows: pd.DataFrame, fam: pd.DataFrame, px: pd.DataFrame,
                  hist_start, year_start, min_n: int = 30,
                  survey_coverage: float = 0.5) -> pd.DataFrame:
    """Marginal impact of each family, measured on [hist_start, year_start).

    impact = mean |react30| on the family's releases
           / mean |react30| at the same clock slot when some OTHER family
             releases and this one does not
    Families need >= min_n releases and >= min_n control releases, and at
    least `survey_coverage` of their releases must carry a consensus (so the
    family has a surprise to trade).
    """
    lo, hi = pd.Timestamp(hist_start, tz="UTC"), pd.Timestamp(year_start, tz="UTC")
    r = rows.merge(fam[["ticker", "family"]], on="ticker")
    r = r[(r["ts_utc"] >= lo) & (r["ts_utc"] < hi)]
    move = px.set_index("ts_utc")["react30"].abs()
    ev = (r.groupby(["family", "ts_utc"])
            .agg(clock=("clock_et", "first"), has_survey=("survey_median", lambda s: s.notna().any()))
            .reset_index())
    ev["move"] = ev["ts_utc"].map(move)
    ev = ev.dropna(subset=["move"])
    out = []
    for f, g in ev.groupby("family"):
        slot = g["clock"].mode().iat[0]
        own = g[g["clock"] == slot]
        others = ev[(ev["clock"] == slot) & (ev["family"] != f)]
        ctrl = others[~others["ts_utc"].isin(own["ts_utc"])].drop_duplicates("ts_utc")
        if len(own) < min_n or len(ctrl) < min_n:
            continue
        a, b = own["move"].to_numpy(), ctrl["move"].to_numpy()
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        out.append(dict(family=f, clock=slot, n=len(a), n_control=len(b),
                        impact=a.mean() / b.mean(), t=(a.mean() - b.mean()) / se,
                        survey_share=own["has_survey"].mean()))
    out = pd.DataFrame(out)
    if len(out):
        out = out[out["survey_share"] >= survey_coverage]
        out = out.sort_values("impact", ascending=False).reset_index(drop=True)
        out["rank"] = np.arange(1, len(out) + 1)
    return out


# ---------------------------------------------------------------------------
# 6. Line signs and family signals
# ---------------------------------------------------------------------------
def line_signs(surp: pd.DataFrame, px: pd.DataFrame, hist_start, year_start,
               min_n: int = 24) -> pd.Series:
    """Sign of the slope of the instant jump on each line's z, history only.
    +1: a positive surprise has pushed ES up; -1: it has pushed ES down."""
    lo, hi = pd.Timestamp(hist_start, tz="UTC"), pd.Timestamp(year_start, tz="UTC")
    d = surp[(surp["ts_utc"] >= lo) & (surp["ts_utc"] < hi)].dropna(subset=["z"])
    d = d.merge(px[["ts_utc", "jump"]], on="ts_utc").dropna(subset=["jump"])
    out = {}
    for t, g in d.groupby("ticker"):
        if len(g) < min_n or g["z"].std() == 0:
            continue
        b = np.polyfit(g["z"], g["jump"], 1)[0]
        out[t] = np.sign(b) if b != 0 else np.nan
    return pd.Series(out, name="sign", dtype=float)


def family_signal_panel(surp, fam, px, screen_by_year: dict, signs_by_year: dict,
                        years) -> pd.DataFrame:
    """One row per (admitted family, release) in the given years:
    x = mean over the family's lines of sign * z (x > 0 = good news for ES)."""
    s = surp.merge(fam[["ticker", "family"]], on="ticker")
    s["year"] = s["ts_utc"].dt.tz_convert(ET).dt.year
    rows = []
    for y in years:
        admitted = set(screen_by_year[y]["family"]) if len(screen_by_year[y]) else set()
        sg = signs_by_year[y]
        d = s[(s["year"] == y) & s["family"].isin(admitted)].dropna(subset=["z"]).copy()
        d["sign"] = d["ticker"].map(sg)
        d = d.dropna(subset=["sign"])
        d["sz"] = d["sign"] * d["z"]
        g = (d.groupby(["family", "ts_utc"])
               .agg(x=("sz", "mean"), n_lines=("sz", "size"))
               .reset_index())
        rows.append(g)
    panel = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return panel.merge(px, on="ts_utc", how="left")


def history_panel(surp, fam, px, families, signs, hist_start, end):
    """Same signal for EARLIER releases (training data for the surprise model),
    using the signs of the current year."""
    lo, hi = pd.Timestamp(hist_start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    s = surp.merge(fam[["ticker", "family"]], on="ticker")
    d = s[(s["ts_utc"] >= lo) & (s["ts_utc"] < hi) & s["family"].isin(families)].dropna(subset=["z"]).copy()
    d["sign"] = d["ticker"].map(signs)
    d = d.dropna(subset=["sign"])
    d["sz"] = d["sign"] * d["z"]
    g = d.groupby(["family", "ts_utc"]).agg(x=("sz", "mean")).reset_index()
    return g.merge(px, on="ts_utc", how="left")


# ---------------------------------------------------------------------------
# 7. Strategies
# ---------------------------------------------------------------------------
def cost_bps(pre_close, ticks_rt=2.0, commission_rt=4.5, tick=0.25, mult=50.0):
    return (ticks_rt * tick + commission_rt / mult) / np.asarray(pre_close, float) * 1e4


def build_positions(panel: pd.DataFrame, hist: dict, model: str, target: str,
                    min_train: int = 24, thr_mult: float = 1.0,
                    cost_col: str = "cost_bps") -> pd.DataFrame:
    """Per (family, release) position for one sizing model.

    model:
      'naive'        position = sign(x)                                  (flat)
      'surprise'     E[r] = b_f * x, b_f re-fitted on the family's earlier
                     releases; trade sign(E[r]) if |E[r]| > thr_mult * cost (flat)
      'signal_vol'   position = clip(x, -3, 3) * vol_4h / median(past vol_4h),
                     then divided by the average |position| of earlier releases
                     so it averages about one unit (Jack Duncan's sizing)
    `hist[year]` is the history panel used to fit b_f for that year.
    """
    p = panel.sort_values("ts_utc").copy()
    p["position"] = 0.0
    p["pred"] = np.nan
    if model == "naive":
        p["position"] = np.sign(p["x"]).fillna(0.0)
        return p
    if model == "surprise":
        for i, row in p.iterrows():
            y = row["year"]
            h = hist[y]
            h = h[(h["family"] == row["family"]) & (h["ts_utc"] < row["ts_utc"])].dropna(subset=["x", target])
            if len(h) < min_train or h["x"].std() == 0:
                continue
            b = np.polyfit(h["x"], h[target], 1)[0]
            pred = b * row["x"]
            p.at[i, "pred"] = pred
            if abs(pred) > thr_mult * row[cost_col]:
                p.at[i, "position"] = float(np.sign(pred))
        return p
    if model == "signal_vol":
        raw = []
        for i, row in p.iterrows():
            h = hist[row["year"]]
            past_vol = h.loc[h["ts_utc"] < row["ts_utc"], "vol_4h"].dropna()
            if len(past_vol) < min_train or not np.isfinite(row["vol_4h"]):
                raw.append(np.nan)
                continue
            raw.append(np.clip(row["x"], -3, 3) * row["vol_4h"] / past_vol.median())
        p["raw_pos"] = raw
        # normalise to ~1 unit on average using EARLIER positions only
        scale = p["raw_pos"].abs().expanding().mean().shift(1)
        first = p["raw_pos"].abs().dropna()
        scale = scale.fillna(first.iloc[:12].mean() if len(first) else 1.0)
        p["position"] = (p["raw_pos"] / scale).fillna(0.0).clip(-4, 4)
        return p
    raise ValueError(model)


def to_trades(p: pd.DataFrame, target: str, strategy: str) -> pd.DataFrame:
    """Aggregate family positions to one position per timestamp (simultaneous
    families average), charge costs, and return a trade log in the format of
    walk_forward.py (so its performance functions can be reused)."""
    g = (p.groupby("ts_utc")
           .agg(position=("position", "mean"), ret_bps=(target, "first"),
                cost_bps=("cost_bps", "first"), families=("family", lambda s: "+".join(sorted(s))),
                x=("x", "mean"))
           .reset_index().rename(columns={"ts_utc": "timestamp_utc"}))
    g["eligible"] = g["ret_bps"].notna()
    g["gross_bps"] = g["position"] * g["ret_bps"].fillna(0)
    g["tc_bps"] = g["position"].abs() * g["cost_bps"].fillna(0)
    g["net_bps"] = g["gross_bps"] - g["tc_bps"]
    g["event_type"] = g["families"]
    g["strategy"] = strategy
    return g


# ---------------------------------------------------------------------------
# 8. Extra statistics
# ---------------------------------------------------------------------------
def random_sign_pvalue(trades: pd.DataFrame, monthly_fn, start, end,
                       n: int = 2000, seed: int = 5):
    """P(Sharpe of a random-sign book >= observed Sharpe). Each traded
    position keeps its size but gets a random direction (Jack Duncan's null)."""
    rng = np.random.default_rng(seed)
    obs_m = monthly_fn(trades, start, end)
    obs = obs_m.mean() / obs_m.std(ddof=1) * np.sqrt(12) if obs_m.std(ddof=1) > 0 else np.nan
    t = trades.copy()
    null = np.empty(n)
    for k in range(n):
        flip = rng.choice([-1.0, 1.0], size=len(t))
        t["net_bps"] = (np.abs(t["position"]) * flip * t["ret_bps"].fillna(0)
                        - t["position"].abs() * t["cost_bps"].fillna(0))
        m = monthly_fn(t, start, end)
        sd = m.std(ddof=1)
        null[k] = m.mean() / sd * np.sqrt(12) if sd > 0 else np.nan
    return obs, float(np.nanmean(null >= obs))


def per_family_stats(p: pd.DataFrame, target: str) -> pd.DataFrame:
    d = p[p["position"] != 0].copy()
    d["net"] = d["position"] * d[target] - d["position"].abs() * d["cost_bps"]
    g = d.groupby("family")["net"]
    out = pd.DataFrame({"trades": g.size(), "hit_rate": g.apply(lambda v: (v > 0).mean()),
                        "avg_net_bps": g.mean(),
                        "t_stat": g.apply(lambda v: v.mean() / (v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 2 and v.std() > 0 else np.nan),
                        "total_net_pct": g.sum() / 100})
    return out.sort_values("total_net_pct", ascending=False)
