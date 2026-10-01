"""Item 1: Aaryen's event pipeline re-run with an honest entry lag and a cost model.

Faithful replication of `aaryen/code/02_feature_engineering.ipynb` +
`03_model_backtest.ipynb` / `04_final_strategy_backtest.ipynb`, with two changes:

  * `ENTRY_LAG_MINUTES` is swept over {0, 1, 2, 3} instead of fixed at 0. At lag 0 the
    entry price is the open of the 1-minute bar at the release timestamp, i.e. the
    pre-release price, so the backtest transacts on information it could not have had.
  * Transaction costs are charged. There are none in the original notebooks.

Everything else is held fixed: the same games (`aaryen/data/games_master.parquet` read
straight from `origin/aaryen`), 2013-04-01 cutoff, emergency FOMC meetings dropped,
5-sigma winsorisation, 60-minute z-normalised lead-up, same-event-type DTW with k=5,
momentum composite over 30/60/120 minutes, four seasonality flags, per-event-type
Gaussian NB over 5 quintile buckets, expanding-window retrain before every game,
position = clip(E[r] / (1.5 * train sd), -1, 1).

One substitution: the price file. Aaryen's `es_ohlcv_1m.parquet` is not committed, so
this uses `data/processed/es_1min_clean.parquet`. Lag-0 results reproduce his per-event
numbers closely enough to confirm the substitution is not what drives any difference.

Run: uv run python jack/aaryen_entry_lag.py
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from dtaidistance import dtw
from sklearn.naive_bayes import GaussianNB

ROOT = Path(__file__).resolve().parents[1]
ES_FILE = ROOT / "data" / "processed" / "es_1min_clean.parquet"
OUT_FILE = ROOT / "jack" / "data" / "aaryen_entry_lag_results.parquet"

# --- Aaryen's constants, verbatim ---
CUTOFF_DATE = pd.Timestamp("2013-04-01", tz="UTC")
LEADUP_MINUTES = 60
K_NEIGHBORS = 5
MIN_HISTORY_FOR_DTW = 10
WINSORIZE_SIGMA = 5.0
MOMENTUM_LOOKBACKS_MIN = [30, 60, 120]
BAR_LOOKUP_TOLERANCE = pd.Timedelta(minutes=5)
MIN_TRAIN_GAMES = 30
N_CLASSES = 5
POSITION_SCALE = 1.5

EVENT_FEATURES = {
    "CPI":  ["surprise_headline_w", "surprise_core_w", "momentum_composite", "dtw_expected_return",
             "is_opex_week", "is_triple_witching_week", "is_month_end", "is_quarter_end"],
    "PCE":  ["surprise_headline_w", "surprise_core_w", "momentum_composite", "dtw_expected_return",
             "is_opex_week", "is_triple_witching_week", "is_month_end", "is_quarter_end"],
    "NFP":  ["surprise_headline_w", "momentum_composite", "dtw_expected_return",
             "is_opex_week", "is_triple_witching_week", "is_month_end", "is_quarter_end"],
    "GDP":  ["surprise_headline_w", "momentum_composite", "dtw_expected_return",
             "is_opex_week", "is_triple_witching_week", "is_month_end", "is_quarter_end"],
    "FOMC": ["surprise_headline_w", "momentum_composite", "dtw_expected_return",
             "is_opex_week", "is_triple_witching_week", "is_month_end", "is_quarter_end"],
}

# --- what this script adds ---
ENTRY_LAGS = [0, 1, 2, 3]
WINDOWS = [10, 30, 60, 120]
COST_BPS = {"gross": 0.0, "net_half_tick": 0.5, "net_full_tick": 1.0}  # round trip, bps of notional


def from_git(ref_path: str) -> Path:
    out = Path(tempfile.mkdtemp()) / Path(ref_path).name
    with out.open("wb") as fh:
        subprocess.run(["git", "show", ref_path], cwd=ROOT, stdout=fh, check=True)
    return out


# --------------------------------------------------------------------------- #
# price access
# --------------------------------------------------------------------------- #
es = pd.read_parquet(ES_FILE, columns=["open", "close"])
es = es[~es.index.duplicated(keep="first")].sort_index()
ES_IDX = es.index
ES_OPEN = es["open"].to_numpy()
ES_CLOSE = es["close"].to_numpy()


def lookup_open(ts: pd.Timestamp) -> float:
    pos = ES_IDX.searchsorted(ts)
    if pos >= len(ES_IDX) or abs(ES_IDX[pos] - ts) > BAR_LOOKUP_TOLERANCE:
        return np.nan
    return ES_OPEN[pos]


def _slice_closes(start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
    lo = ES_IDX.searchsorted(start, "left")
    hi = ES_IDX.searchsorted(end, "right")
    return ES_CLOSE[lo:hi]


def leadup_returns(entry_ts: pd.Timestamp, minutes: int = LEADUP_MINUTES):
    closes = _slice_closes(entry_ts - pd.Timedelta(minutes=minutes + 1), entry_ts)
    if len(closes) < minutes * 0.8:
        return None
    rets = np.diff(np.log(closes))
    if len(rets) < 5 or np.nanstd(rets) == 0:
        return None
    return (rets - np.nanmean(rets)) / np.nanstd(rets)


def realized_vol(entry_ts: pd.Timestamp, minutes: int) -> float:
    closes = _slice_closes(entry_ts - pd.Timedelta(minutes=minutes + 1), entry_ts)
    if len(closes) < 3:
        return np.nan
    return float(np.nanstd(np.diff(np.log(closes))))


def lookback_return(entry_ts: pd.Timestamp, minutes: int) -> float:
    a = lookup_open(entry_ts - pd.Timedelta(minutes=minutes))
    b = lookup_open(entry_ts)
    if np.isnan(a) or np.isnan(b) or a == 0:
        return np.nan
    return float(np.log(b / a))


def third_friday(year: int, month: int) -> pd.Timestamp:
    d = pd.Timestamp(year=year, month=month, day=1)
    return pd.date_range(d, d + pd.offsets.MonthEnd(0), freq="W-FRI")[2]


def seasonality_flags(date: pd.Timestamp) -> dict:
    tf = third_friday(date.year, date.month)
    is_opex = (date - pd.Timedelta(days=date.weekday())) <= tf <= (date + pd.Timedelta(days=6 - date.weekday()))
    month_end = date + pd.offsets.MonthEnd(0)
    is_month_end = (month_end - date).days <= 3
    return {
        "is_opex_week": bool(is_opex),
        "is_triple_witching_week": bool(is_opex and date.month in (3, 6, 9, 12)),
        "is_month_end": bool(is_month_end),
        "is_quarter_end": bool(is_month_end and date.month in (3, 6, 9, 12)),
    }


# --------------------------------------------------------------------------- #
# feature build
# --------------------------------------------------------------------------- #
def load_games() -> pd.DataFrame:
    games = pd.read_parquet(from_git("origin/aaryen:aaryen/data/games_master.parquet"))
    games = games.sort_values("timestamp_utc").reset_index(drop=True)
    games = games[games.timestamp_utc >= CUTOFF_DATE]
    games = games[~games.is_emergency_meeting].copy()
    for col in ("surprise_headline", "surprise_core"):
        games[col + "_w"] = games[col].clip(-WINSORIZE_SIGMA, WINSORIZE_SIGMA)
    return games


def build_static(games: pd.DataFrame, lag: int) -> pd.DataFrame:
    """Everything that depends on the entry lag but not on the trade window."""
    rows = []
    for g in games.itertuples(index=False):
        entry_ts = g.timestamp_utc + pd.Timedelta(minutes=lag)
        entry_px = lookup_open(entry_ts)
        if np.isnan(entry_px) or entry_px == 0:
            continue
        lead = leadup_returns(entry_ts)
        if lead is None:
            continue
        mom = []
        for m in MOMENTUM_LOOKBACKS_MIN:
            rv = realized_vol(entry_ts, m)
            mom.append(lookback_return(entry_ts, m) / rv if rv and np.isfinite(rv) and rv > 0 else np.nan)
        row = {
            "event_type": g.event_type, "timestamp_utc": g.timestamp_utc,
            "entry_ts": entry_ts, "entry_px": entry_px,
            "surprise_headline_w": g.surprise_headline_w, "surprise_core_w": g.surprise_core_w,
            "momentum_composite": np.nanmean(mom) if np.isfinite(mom).any() else np.nan,
            "_leadup": lead.astype(np.double),
        }
        row.update(seasonality_flags(pd.Timestamp(g.release_date)))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("timestamp_utc").reset_index(drop=True)


def dtw_distance_blocks(static: pd.DataFrame) -> dict[str, tuple[list[int], np.ndarray]]:
    """Pairwise DTW distances within each event type. Depends on the lag only, so it is
    computed once per lag and reused across all four trade windows."""
    blocks = {}
    for event_type, grp in static.groupby("event_type"):
        idxs = sorted(grp.index)
        paths = [static.at[i, "_leadup"] for i in idxs]
        n = len(idxs)
        d = np.zeros((n, n))
        for a in range(n):
            for b in range(a + 1, n):
                dist = dtw.distance_fast(paths[a], paths[b], use_pruning=True)
                d[a, b] = d[b, a] = dist
        blocks[event_type] = (idxs, d)
    return blocks


def build_window_table(static: pd.DataFrame, blocks: dict, lag: int, window: int) -> pd.DataFrame:
    df = static.copy()
    exit_px, trade_return = [], []
    for row in df.itertuples(index=False):
        px = lookup_open(row.entry_ts + pd.Timedelta(minutes=window))
        exit_px.append(px)
        trade_return.append(np.log(px / row.entry_px) if np.isfinite(px) and px > 0 else np.nan)
    df["exit_px"] = exit_px
    df["trade_return"] = trade_return

    df["dtw_expected_return"] = np.nan
    for event_type, (idxs, dmat) in blocks.items():
        rets = df.loc[idxs, "trade_return"].to_numpy()
        for pos in range(len(idxs)):
            if pos < MIN_HISTORY_FOR_DTW:
                continue
            prior_d = dmat[pos, :pos]
            prior_r = rets[:pos]
            ok = np.isfinite(prior_r)
            if ok.sum() < MIN_HISTORY_FOR_DTW:
                continue
            d_ok, r_ok = prior_d[ok], prior_r[ok]
            k = min(K_NEIGHBORS, len(d_ok))
            nn = np.argpartition(d_ok, k - 1)[:k]
            w = 1.0 / (d_ok[nn] + 1e-8)
            df.at[idxs[pos], "dtw_expected_return"] = float(np.dot(w / w.sum(), r_ok[nn]))
    return df.drop(columns=["_leadup"])


# --------------------------------------------------------------------------- #
# walk-forward (Aaryen's engine, unchanged)
# --------------------------------------------------------------------------- #
def make_labels(train_returns: pd.Series, all_returns: pd.Series):
    edges = [-np.inf] + list(train_returns.quantile([0.2, 0.4, 0.6, 0.8]).values) + [np.inf]
    return pd.cut(all_returns, bins=edges, labels=False, include_lowest=True)


def walk_forward(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    df = df.sort_values("timestamp_utc").dropna(subset=features + ["trade_return"]).reset_index(drop=True)
    records = []
    for i in range(MIN_TRAIN_GAMES, len(df)):
        train, test_row = df.iloc[:i], df.iloc[i]
        train_labels = make_labels(train["trade_return"], train["trade_return"])
        class_stats = {}
        for c in range(N_CLASSES):
            mask = (train_labels == c).to_numpy()
            rets = train.loc[mask, "trade_return"] if mask.sum() else train["trade_return"]
            class_stats[c] = rets.mean()
        y_train = train_labels.to_numpy().astype(int)
        if len(np.unique(y_train)) < 2:
            continue
        clf = GaussianNB().fit(train[features].to_numpy(), y_train)
        proba = np.zeros(N_CLASSES)
        for c, p in zip(clf.classes_, clf.predict_proba(test_row[features].to_numpy().reshape(1, -1))[0]):
            proba[int(c)] = p
        expected_return = float(sum(proba[c] * class_stats[c] for c in range(N_CLASSES)))
        train_std = train["trade_return"].std()
        position = float(np.clip(expected_return / (POSITION_SCALE * train_std), -1, 1)) if train_std > 0 else 0.0
        records.append({
            "event_type": test_row["event_type"], "timestamp_utc": test_row["timestamp_utc"],
            "trade_return": test_row["trade_return"], "expected_return": expected_return,
            "position": position,
        })
    return pd.DataFrame(records)


def perf(r: pd.DataFrame, cost_bps: float) -> dict:
    r = r.sort_values("timestamp_utc")
    gross = r["position"] * (np.exp(r["trade_return"]) - 1)
    net = gross - r["position"].abs() * cost_bps / 1e4
    n = len(net)
    years = (r["timestamp_utc"].max() - r["timestamp_utc"].min()).days / 365.25
    tpy = n / years
    equity = (1 + net).cumprod()
    total = equity.iloc[-1] - 1
    ann_vol = net.std() * np.sqrt(tpy)
    return {
        "n_trades": n, "trades_per_year": round(tpy, 1),
        "bps_per_trade": round(net.mean() * 1e4, 2),
        "ann_return_%": round(((1 + total) ** (1 / years) - 1) * 100, 2),
        "sharpe": round((net.mean() * tpy) / ann_vol, 3) if ann_vol > 0 else np.nan,
        "max_dd_%": round((equity / equity.cummax() - 1).min() * 100, 2),
        "hit_rate_%": round((net > 0).mean() * 100, 1),
        "t_stat": round(net.mean() / (net.std() / np.sqrt(n)), 2),
    }


def main() -> None:
    games = load_games()
    print(f"games after cutoff + emergency filter: {len(games)}")
    all_preds, summary = [], []

    for lag in ENTRY_LAGS:
        static = build_static(games, lag)
        blocks = dtw_distance_blocks(static)
        for window in WINDOWS:
            table = build_window_table(static, blocks, lag, window)
            per_type = []
            for event_type, features in EVENT_FEATURES.items():
                sub = table[table.event_type == event_type]
                if len(sub) <= MIN_TRAIN_GAMES:
                    continue
                res = walk_forward(sub, features)
                if len(res):
                    per_type.append(res)
            preds = pd.concat(per_type, ignore_index=True)
            preds["lag"], preds["window"] = lag, window
            all_preds.append(preds)
            for cost_label, cost in COST_BPS.items():
                summary.append({"lag": lag, "window": window, "costs": cost_label,
                                **perf(preds, cost)})
            print(f"  lag={lag} window={window:>3}m -> {len(preds)} trades")

    preds = pd.concat(all_preds, ignore_index=True)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(OUT_FILE, index=False)
    summary_df = pd.DataFrame(summary)

    print("\n" + "=" * 78)
    print("PORTFOLIO, all five event types pooled")
    print("=" * 78)
    for cost_label in COST_BPS:
        print(f"\n--- {cost_label} ---")
        piv = summary_df[summary_df.costs == cost_label].pivot(index="lag", columns="window", values="sharpe")
        print("Sharpe:"); print(piv.to_string())
        piv = summary_df[summary_df.costs == cost_label].pivot(index="lag", columns="window", values="bps_per_trade")
        print("bps/trade:"); print(piv.to_string())

    print("\n" + "=" * 78)
    print("PER EVENT TYPE, bps per trade net of a full tick (t-stat in brackets)")
    print("=" * 78)
    for event_type in EVENT_FEATURES:
        print(f"\n{event_type}")
        print(f"  {'lag':>4} " + " ".join(f"{w:>17}m" for w in WINDOWS))
        for lag in ENTRY_LAGS:
            cells = []
            for window in WINDOWS:
                sub = preds[(preds.lag == lag) & (preds.window == window) & (preds.event_type == event_type)]
                if len(sub) < 10:
                    cells.append(f"{'-':>18}")
                    continue
                s = perf(sub, COST_BPS["net_full_tick"])
                cells.append(f"{s['bps_per_trade']:>8.2f} ({s['t_stat']:>+5.2f})")
            print(f"  {lag:>4} " + " ".join(cells))

    summary_df.to_csv(ROOT / "jack" / "data" / "aaryen_entry_lag_summary.csv", index=False)
    print(f"\nsaved predictions -> {OUT_FILE}")


if __name__ == "__main__":
    main()
