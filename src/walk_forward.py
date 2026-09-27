"""
walk_forward.py
===============

Helper functions for Notebook 04: walk-forward trading strategy and
performance evaluation.

Trading logic
-------------
For every macro release at time T:

1. The surprise x is known at the release (T). The strategy enters at the
   close of the first post-release bar (label T, i.e. ~1 minute after the
   release) plus an optional extra latency of `entry_lag` minutes.
2. It exits at the close of bar T + h - 1 (same convention as NB02/NB03).
3. The trade direction and size come from a model estimated ONLY on earlier
   releases of the same family (expanding or rolling window). No parameter
   is ever fitted on data that includes the event being traded.

Models (per family)
-------------------
naive         position = sign(x). No estimation (economic prior only).
surprise      E[r] = b * x                      (b re-estimated each event)
state         E[r] = (b + d * s) * x            s = state percentile - 0.5
state_filter  surprise model, but trade only if state percentile > 1/3
adaptive      at each event, pick the state with the largest |t| on the
              x*s interaction in the TRAINING data; use it only if |t| >= t_min,
              otherwise fall back to the surprise model. This removes the
              selection bias of choosing the state from the full-sample NB03.

A trade is taken when the model's expected surprise-driven return exceeds
`threshold_mult` x the round-trip transaction cost.

Only numpy / pandas are required.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .state_conditioning import ols_hc3

PRIMARY_STATES = ["rv_20d", "rv_pre_1d", "mom_20d", "dd_60d",
                  "overnight_ret", "prev_x"]

ES_TICK = 0.25          # index points
ES_MULTIPLIER = 50.0    # USD per index point


# ---------------------------------------------------------------------------
# 1. Trade returns from 1-minute data
# ---------------------------------------------------------------------------
def trade_col(lag: int, hold: int) -> str:
    return f"tr_L{lag}_H{hold}"


def compute_trade_returns(events: pd.DataFrame, es_adj: pd.DataFrame,
                          entry_lags=(0, 1, 2, 5),
                          exit_minutes=(5, 15, 30, 60)) -> pd.DataFrame:
    """Add trade-return columns `tr_L{lag}_H{h}` in bps.

    entry price = close of bar labeled T + lag   (lag 0 = first post-release bar)
    exit price  = close of bar labeled T + h - 1
    Prices use the roll-adjusted log price; if a bar is missing the last
    available close at or before that label is used (as-of price).
    tr_L0_H{h} is identical to NB03's drift_{h}m_bps.
    """
    out = events.copy()
    idx = es_adj.index
    idx = idx.as_unit("ns") if hasattr(idx, "as_unit") else idx
    t_ns = idx.asi8
    lp = es_adj["logp_adj"].to_numpy()
    T = pd.DatetimeIndex(pd.to_datetime(out["timestamp_utc"], utc=True))
    T = T.as_unit("ns") if hasattr(T, "as_unit") else T

    def px(minutes):
        ts = (T + pd.Timedelta(minutes=minutes)).asi8
        pos = np.searchsorted(t_ns, ts, side="right") - 1
        val = lp[np.clip(pos, 0, None)]
        stale = (ts - t_ns[np.clip(pos, 0, None)]) > 5 * 60 * 1e9  # > 5 min old
        return np.where((pos < 0) | stale, np.nan, val)

    for L in entry_lags:
        p_in = px(L)
        for h in exit_minutes:
            if h - 1 <= L:
                continue
            out[trade_col(L, h)] = (px(h - 1) - p_in) * 1e4
    return out


# ---------------------------------------------------------------------------
# 2. Transaction costs
# ---------------------------------------------------------------------------
def round_trip_cost_bps(pre_close, slippage_ticks_rt: float = 2.0,
                        commission_rt_usd: float = 4.5):
    """Round-trip cost in bps of notional for one ES contract.

    slippage_ticks_rt : total ticks lost entering + exiting (bid-ask + impact).
                        Spreads widen around releases, so 2 ticks round trip
                        (1 per side) is the baseline; sensitivity goes to 8.
    commission_rt_usd : exchange + broker fees per contract, round trip.
    """
    pts = slippage_ticks_rt * ES_TICK + commission_rt_usd / ES_MULTIPLIER
    return pts / np.asarray(pre_close, dtype=float) * 1e4


# ---------------------------------------------------------------------------
# 3. Strategy specification
# ---------------------------------------------------------------------------
@dataclass
class StrategySpec:
    name: str
    model: str                       # naive | surprise | state | state_filter | adaptive
    state: str = "rv_20d"
    threshold_mult: float = 1.0      # trade if |E[r]| > threshold_mult * cost
    window: int | None = None        # None = expanding; int = last N events
    min_train: int = 36              # events of the same family before trading
    t_min: float = 2.0               # adaptive: minimum |t| to use a state
    filter_pct: float = 1 / 3        # state_filter: trade only above this pct
    candidates: list = field(default_factory=lambda: list(PRIMARY_STATES))


def _lstsq(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b


def _fit_surprise(past, target):
    X = np.column_stack([np.ones(len(past)), past["x"].to_numpy()])
    a, b = _lstsq(X, past[target].to_numpy())
    return b


def _fit_state(past, target, state):
    p = past.dropna(subset=[f"{state}_pct"])
    if len(p) < 20:
        return None
    s = p[f"{state}_pct"] - 0.5
    X = pd.DataFrame({"const": 1.0, "x": p["x"], "s": s, "x_s": p["x"] * s})
    r = ols_hc3(p[target], X)
    return dict(b=r.params["x"], d=r.params["x_s"], t_d=r.t["x_s"], n=r.n)


# ---------------------------------------------------------------------------
# 4. Walk-forward engine
# ---------------------------------------------------------------------------
def run_walk_forward(ev: pd.DataFrame, spec: StrategySpec, target: str,
                     families=("CPI", "PCE", "NFP"),
                     slippage_ticks_rt: float = 2.0,
                     commission_rt_usd: float = 4.5) -> pd.DataFrame:
    """Run one strategy through time. Returns one row per event."""
    rows = []
    for fam in families:
        d = (ev[ev["event_type"].eq(fam)]
             .sort_values("timestamp_utc").reset_index(drop=True))
        cost = round_trip_cost_bps(d["pre_close"], slippage_ticks_rt,
                                   commission_rt_usd)
        for i in range(len(d)):
            row = d.iloc[i]
            rec = dict(timestamp_utc=row["timestamp_utc"], event_type=fam,
                       strategy=spec.name, x=row["x"], ret_bps=row[target],
                       cost_bps=cost[i], pred_bps=np.nan, position=0.0,
                       eligible=False, b=np.nan, d=np.nan, t_d=np.nan,
                       used_state=None, state_pct=np.nan, n_train=0)
            past = d.iloc[:i].dropna(subset=["x", target])
            if spec.window:
                past = past.tail(spec.window)
            rec["n_train"] = len(past)

            if (not np.isfinite(row["x"]) or not np.isfinite(row[target])
                    or len(past) < spec.min_train):
                rows.append(rec)
                continue
            rec["eligible"] = True
            x = row["x"]

            if spec.model == "naive":
                rec["pred_bps"] = np.nan  # no model: trade the sign of x
                rec["position"] = float(np.sign(x))
                rows.append(rec)
                continue

            b_s = _fit_surprise(past, target)
            rec["b"] = b_s
            pred = b_s * x

            if spec.model == "state":
                sp = row.get(f"{spec.state}_pct", np.nan)
                f = _fit_state(past, target, spec.state)
                if f is not None and np.isfinite(sp):
                    s = sp - 0.5
                    pred = (f["b"] + f["d"] * s) * x
                    rec.update(b=f["b"], d=f["d"], t_d=f["t_d"],
                               used_state=spec.state, state_pct=sp)

            elif spec.model == "state_filter":
                sp = row.get(f"{spec.state}_pct", np.nan)
                rec.update(used_state=spec.state, state_pct=sp)
                if not (np.isfinite(sp) and sp > spec.filter_pct):
                    pred = 0.0

            elif spec.model == "adaptive":
                best = None
                for st in spec.candidates:
                    sp = row.get(f"{st}_pct", np.nan)
                    if not np.isfinite(sp):
                        continue
                    f = _fit_state(past, target, st)
                    if f is None or not np.isfinite(f["t_d"]):
                        continue
                    if best is None or abs(f["t_d"]) > abs(best[1]["t_d"]):
                        best = (st, f, sp)
                if best is not None and abs(best[1]["t_d"]) >= spec.t_min:
                    st, f, sp = best
                    pred = (f["b"] + f["d"] * (sp - 0.5)) * x
                    rec.update(b=f["b"], d=f["d"], t_d=f["t_d"],
                               used_state=st, state_pct=sp)
                else:
                    rec["used_state"] = "none"

            elif spec.model != "surprise":
                raise ValueError(f"unknown model {spec.model}")

            rec["pred_bps"] = pred
            if abs(pred) > spec.threshold_mult * cost[i]:
                rec["position"] = float(np.sign(pred))
            rows.append(rec)

    out = pd.DataFrame(rows).sort_values("timestamp_utc").reset_index(drop=True)
    out["gross_bps"] = out["position"] * out["ret_bps"].fillna(0)
    out["tc_bps"] = out["position"].abs() * out["cost_bps"]
    out["net_bps"] = out["gross_bps"] - out["tc_bps"]
    return out


# ---------------------------------------------------------------------------
# 5. Performance metrics
# ---------------------------------------------------------------------------
def _eval_slice(trades, start, end):
    t = trades[trades["eligible"]].copy()
    ts = pd.to_datetime(t["timestamp_utc"], utc=True)
    m = (ts >= pd.Timestamp(start, tz="UTC")) & (ts <= pd.Timestamp(end, tz="UTC"))
    return t[m.to_numpy()]


def monthly_returns(trades, start, end, col="net_bps") -> pd.Series:
    """Monthly return on one unit of notional (decimal). Months without a
    trade count as 0 so the Sharpe ratio reflects the true calendar."""
    t = _eval_slice(trades, start, end)
    months = pd.period_range(pd.Timestamp(start), pd.Timestamp(end), freq="M")
    per = pd.to_datetime(t["timestamp_utc"]).dt.tz_convert(None).dt.to_period("M")
    return (t.groupby(per)[col].sum() / 1e4).reindex(months, fill_value=0.0)


def performance(trades, start, end, label=None) -> dict:
    t = _eval_slice(trades, start, end)
    tr = t[t["position"] != 0]
    mret = monthly_returns(trades, start, end)
    mret_g = monthly_returns(trades, start, end, "gross_bps")
    cum = t["net_bps"].cumsum()
    dd = cum - cum.cummax()
    wins, losses = tr["net_bps"][tr["net_bps"] > 0], tr["net_bps"][tr["net_bps"] < 0]
    n = len(tr)
    sd = tr["net_bps"].std(ddof=1) if n > 1 else np.nan
    ann = lambda s: s.mean() * 12 / max(s.std(ddof=1) * np.sqrt(12), 1e-12)
    return {
        "strategy": label or (t["strategy"].iloc[0] if len(t) else ""),
        "events": len(t),
        "trades": n,
        "trade_rate": n / len(t) if len(t) else np.nan,
        "hit_rate": (tr["net_bps"] > 0).mean() if n else np.nan,
        "avg_gross_bps": tr["gross_bps"].mean() if n else np.nan,
        "avg_cost_bps": tr["tc_bps"].mean() if n else np.nan,
        "avg_net_bps": tr["net_bps"].mean() if n else np.nan,
        "total_net_pct": t["net_bps"].sum() / 100,
        "ann_return_pct": mret.mean() * 12 * 100,
        "ann_vol_pct": mret.std(ddof=1) * np.sqrt(12) * 100,
        "sharpe_net": ann(mret),
        "sharpe_gross": ann(mret_g),
        "max_dd_pct": dd.min() / 100 if len(dd) else np.nan,
        "t_stat_trade": tr["net_bps"].mean() / (sd / np.sqrt(n)) if n > 1 and sd > 0 else np.nan,
        "profit_factor": wins.sum() / abs(losses.sum()) if len(losses) and losses.sum() != 0 else np.nan,
        "avg_win_bps": wins.mean() if len(wins) else np.nan,
        "avg_loss_bps": losses.mean() if len(losses) else np.nan,
        "total_cost_pct": t["tc_bps"].sum() / 100,
    }


def perf_table(trade_sets: dict, start, end) -> pd.DataFrame:
    return pd.DataFrame([performance(t, start, end, k)
                         for k, t in trade_sets.items()]).set_index("strategy")


def bootstrap_sharpe(trades, start, end, n_boot=5000, seed=11, block=3):
    """Moving-block bootstrap of the monthly Sharpe ratio: returns
    (5%, 50%, 95%) percentiles and P(Sharpe <= 0)."""
    m = monthly_returns(trades, start, end).to_numpy()
    sh = _block_boot(m[None, :], n_boot, seed, block)[0]
    return np.nanpercentile(sh, [5, 50, 95]), float(np.mean(sh <= 0))


def bootstrap_sharpe_diff(trades_a, trades_b, start, end, n_boot=5000,
                          seed=11, block=3):
    """Sharpe(B) - Sharpe(A) with the SAME resampled months for both
    (paired block bootstrap). Returns (observed diff, 5/50/95 pct, P(diff<=0))."""
    ma = monthly_returns(trades_a, start, end).to_numpy()
    mb = monthly_returns(trades_b, start, end).to_numpy()
    sa, sb = _block_boot(np.vstack([ma, mb]), n_boot, seed, block)
    diff = sb - sa
    obs = _sharpe(mb) - _sharpe(ma)
    return obs, np.nanpercentile(diff, [5, 50, 95]), float(np.mean(diff <= 0))


def _sharpe(m):
    s = np.std(m, ddof=1)
    return np.mean(m) * 12 / (s * np.sqrt(12)) if s > 0 else np.nan


def _block_boot(M, n_boot, seed, block):
    rng = np.random.default_rng(seed)
    k, n = M.shape
    nb = int(np.ceil(n / block))
    out = np.full((k, n_boot), np.nan)
    for j in range(n_boot):
        starts = rng.integers(0, n - block + 1, nb)
        ix = (starts[:, None] + np.arange(block)).ravel()[:n]
        S = M[:, ix]
        sd = S.std(axis=1, ddof=1)
        out[:, j] = np.where(sd > 0, S.mean(axis=1) * np.sqrt(12) / np.where(sd > 0, sd, 1), np.nan)
    return out


def paired_trade_test(trades_a, trades_b, start, end):
    """Paired t-test on per-event net P&L difference (B - A), same events."""
    a = _eval_slice(trades_a, start, end).set_index("timestamp_utc")["net_bps"]
    b = _eval_slice(trades_b, start, end).set_index("timestamp_utc")["net_bps"]
    d = (b - a).dropna()
    n = len(d)
    sd = d.std(ddof=1)
    t = d.mean() / (sd / np.sqrt(n)) if sd > 0 else np.nan
    return dict(n_events=n, mean_diff_bps=d.mean(), t_stat=t,
                events_where_different=int((d != 0).sum()))


def drawdown_series(trades, start, end):
    t = _eval_slice(trades, start, end)
    cum = t.set_index("timestamp_utc")["net_bps"].cumsum() / 100
    return cum, cum - cum.cummax()


# ---------------------------------------------------------------------------
# 6. Peer-comparable (daily) metrics
# ---------------------------------------------------------------------------
# These reproduce the presentation format used by the DTW peer project:
# Total %, annualised vol, Sharpe, max drawdown and beta, all computed from a
# DAILY return series for one contract (constant notional, not compounded),
# with buy-and-hold ES and an "always long in the same windows" benchmark.

def always_long(trades: pd.DataFrame, name: str = None) -> pd.DataFrame:
    """Exposure-matched benchmark: go LONG in exactly the windows where the
    strategy traded (same entry, exit and costs). The gap between the strategy
    and this benchmark isolates the value of getting the DIRECTION right."""
    t = trades.copy()
    t["position"] = np.where(t["position"] != 0, 1.0, 0.0)
    t["gross_bps"] = t["position"] * t["ret_bps"].fillna(0)
    t["tc_bps"] = t["position"] * t["cost_bps"]
    t["net_bps"] = t["gross_bps"] - t["tc_bps"]
    t["strategy"] = name or "Always long, same windows"
    return t


def always_long_all(trades: pd.DataFrame, name: str = None) -> pd.DataFrame:
    """Long in EVERY eligible release window (no signal at all)."""
    t = trades.copy()
    t["position"] = np.where(t["eligible"], 1.0, 0.0)
    return always_long(t, name or "Always long, all release windows")


def daily_returns(trades: pd.DataFrame, trading_days: pd.DatetimeIndex,
                  col: str = "net_bps") -> pd.Series:
    """Daily return (decimal, one contract) = sum of trade returns on that
    ET date; 0 on days without a trade."""
    t = trades[trades["eligible"]]
    d = pd.to_datetime(pd.to_datetime(t["timestamp_utc"], utc=True)
                       .dt.tz_convert("America/New_York").dt.date)
    s = t.groupby(d.to_numpy())[col].sum() / 1e4
    s.index = pd.DatetimeIndex(s.index)
    return s.reindex(trading_days, fill_value=0.0)


def buy_hold_daily(daily_close: pd.DataFrame, trading_days) -> pd.Series:
    """Daily simple return of holding one ES contract (roll-adjusted closes)."""
    r = np.expm1(daily_close["logp_close"].diff())
    return r.reindex(trading_days).fillna(0.0)


def peer_metrics(daily: pd.Series, market: pd.Series = None, trades_per_year=None,
                 minutes_in_market=None, compound: bool = False) -> dict:
    """Total % (sum of daily returns, one contract), annualised vol, Sharpe
    (daily, sqrt(252)), max drawdown of the cumulative-sum curve, beta and
    correlation to buy-and-hold ES.

    compound=True (used for buy-and-hold): total and drawdown follow the
    actual price path of a held contract, i.e. cumulative product."""
    if compound:
        cum = (1 + daily).cumprod() - 1
        dd = (1 + cum) / (1 + cum).cummax() - 1
    else:
        cum = daily.cumsum()
        dd = cum - cum.cummax()
    sd = daily.std(ddof=1)
    out = {
        "total_pct": cum.iloc[-1] * 100,
        "vol_pct": sd * np.sqrt(252) * 100,
        "sharpe": daily.mean() / sd * np.sqrt(252) if sd > 0 else np.nan,
        "max_dd_pct": dd.min() * 100,
    }
    if market is not None:
        m = market.reindex(daily.index).fillna(0.0)
        v = m.var(ddof=1)
        out["beta"] = daily.cov(m) / v if v > 0 else np.nan
        out["corr"] = daily.corr(m)
    if trades_per_year is not None:
        out["trades_per_year"] = trades_per_year
    if minutes_in_market is not None:
        out["time_in_market_pct"] = minutes_in_market
    return out
