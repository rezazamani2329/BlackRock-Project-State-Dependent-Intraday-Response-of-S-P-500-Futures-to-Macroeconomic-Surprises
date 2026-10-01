"""Item 2: re-score Hoshea's DTW grid with a selection period and a confirmation period.

`hoshea_zeng/dtw_event_game_simple_generalized.py` searches 18 hyperparameter
combinations and reports the winner's IC (0.1162) on the full sample. Because the
winner is chosen on the same rows it is scored on, that number is a best-of-18 and
cannot be read as an expected IC.

Nothing in the factor is fitted, so no refit is needed to fix this: the factor for a
2023 game already uses only pre-2023 neighbours. The honest version simply *selects*
on 2013-2021 and *reports* on 2022-2026. This script rebuilds the full grid and does
exactly that, and breaks every result out by event type rather than pooling, because
the pooled IC in the original is dominated by 3,401 Initial Jobless Claims games that
carry no directional signal at all.

Construction is Hoshea's, unchanged: anchors at T0 in {-60,-30,0,+30,+60} minutes
around each release, input path [T0, T0+60] as z-scored log returns at `path_freq_min`
sampling, label = log P(T0+90) - log P(T0+60), Sakoe-Chiba banded DTW, k=15 neighbours
drawn from strictly earlier games at the same offset and ET clock, weighted by inverse
distance and by recency on a two-year halflife.

Run: uv run python jack/hoshea_grid_holdout.py
"""

from __future__ import annotations

from itertools import product
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from dtaidistance import dtw

ROOT = Path(__file__).resolve().parents[1]
PRICE_FILE = ROOT / "data" / "processed" / "es_1min_bars_2010_2026.parquet"
EVENT_FILE = ROOT / "data" / "processed" / "macro_event_calendar_expanded_2010_2026.parquet"
OUT_DIR = ROOT / "jack" / "data"

# --- Hoshea's constants, verbatim ---
EVENT_TYPE_GROUPS = {
    "core_major": ["CPI", "NFP", "PCE", "FOMC"],
    "fomc_jolts_pce_claims": ["FOMC", "JOLTS", "PCE", "Initial Jobless Claims"],
}
EVENT_OFFSETS_MIN = [-60, -30, 0, 30, 60]
PATH_MINUTES = 60
ENTRY_DELAY_MINUTES = 60
HOLD_MINUTES = 30
PATH_FREQ_MIN_OPTIONS = [2, 3, 5]
LOOKBACK_YEARS_OPTIONS = [2, 3, 5]
K_NEIGHBORS = 15
MIN_HISTORY = 20
RECENCY_HALFLIFE_DAYS = 365 * 2
WARP_TOLERANCE_MINUTES = 15

# --- what this script adds ---
SPLIT = pd.Timestamp("2022-01-01", tz="UTC")  # select before, confirm after
TICK_BPS = 0.5
NY_TZ = ZoneInfo("America/New_York")


def zscore(x: np.ndarray) -> np.ndarray:
    sd = float(np.std(x))
    return np.zeros_like(x) if sd < 1e-12 else (x - float(np.mean(x))) / sd


def extract_path(close: pd.Series, t0: pd.Timestamp, freq_min: int):
    idx = pd.date_range(t0, t0 + pd.Timedelta(minutes=PATH_MINUTES), freq=f"{freq_min}min", tz="UTC")
    px = close.reindex(idx)
    if px.isna().any():
        return None
    logret = np.diff(np.log(px.to_numpy(dtype=float)))
    if len(logret) == 0 or not np.isfinite(logret).all():
        return None
    return zscore(logret)


def extract_label(close: pd.Series, t0: pd.Timestamp):
    entry = t0 + pd.Timedelta(minutes=ENTRY_DELAY_MINUTES)
    px = close.reindex([entry, entry + pd.Timedelta(minutes=HOLD_MINUTES)])
    if px.isna().any():
        return None
    return float(np.log(px.iloc[1]) - np.log(px.iloc[0]))


def build_games(close: pd.Series, events: pd.DataFrame, types: list[str], freq_min: int) -> pd.DataFrame:
    sel = events[events.event_type.isin(types)]
    rows = []
    for row in sel.itertuples(index=False):
        for offset in EVENT_OFFSETS_MIN:
            t0 = row.timestamp_utc + pd.Timedelta(minutes=offset)
            path = extract_path(close, t0, freq_min)
            if path is None:
                continue
            label = extract_label(close, t0)
            if label is None:
                continue
            rows.append({"event_type": row.event_type, "event_ts": row.timestamp_utc,
                         "offset_min": offset, "t0_utc": t0,
                         "clock_et": t0.tz_convert(NY_TZ).strftime("%H:%M"),
                         "future_logret": label, "path": path})
    return pd.DataFrame(rows).sort_values("t0_utc").reset_index(drop=True)


def compute_factor(games: pd.DataFrame, lookback_years: int, window_steps: int) -> pd.DataFrame:
    out = games.copy()
    out["dtw_factor"] = np.nan
    lookback = pd.Timedelta(days=365 * lookback_years)
    for _, grp in out.groupby(["offset_min", "clock_et"], sort=False):
        idx = grp.index.to_numpy()
        t0s = out.loc[idx, "t0_utc"].tolist()
        paths = out.loc[idx, "path"].tolist()
        rets = out.loc[idx, "future_logret"].to_numpy(dtype=float)
        for pos, row_idx in enumerate(idx):
            cutoff = t0s[pos] - lookback
            hist = [h for h in range(pos) if t0s[h] >= cutoff]
            if len(hist) < MIN_HISTORY:
                continue
            x = np.asarray(paths[pos], dtype=np.double)
            d = np.array([dtw.distance_fast(x, np.asarray(paths[h], dtype=np.double),
                                            window=int(max(window_steps, abs(len(x) - len(paths[h])))),
                                            use_pruning=True) for h in hist])
            k = min(K_NEIGHBORS, len(hist))
            nn = np.argpartition(d, k - 1)[:k]
            gaps = np.array([(t0s[pos] - t0s[hist[j]]).days for j in nn], dtype=float)
            w = (1.0 / (d[nn] + 1e-6)) * np.exp(-gaps / RECENCY_HALFLIFE_DAYS)
            out.at[row_idx, "dtw_factor"] = float(np.dot(w, rets[[hist[j] for j in nn]]) / w.sum())
    return out


def clustered_t(df: pd.DataFrame, col: str, cluster: str = "event_ts") -> float:
    mean, n = df[col].mean(), len(df)
    resid = df.groupby(cluster)[col].apply(lambda s: (s - mean).sum())
    return mean / (np.sqrt((resid ** 2).sum()) / n)


def score(df: pd.DataFrame) -> dict:
    v = df.dropna(subset=["dtw_factor"])
    if len(v) < 50:
        return {"n": len(v), "ic": np.nan, "hit": np.nan, "bps": np.nan, "t_clust": np.nan}
    pnl = np.sign(v.dtw_factor) * v.future_logret * 1e4
    v = v.assign(pnl_bps=pnl)
    return {"n": len(v),
            "ic": float(v.dtw_factor.corr(v.future_logret)),
            "hit": float((np.sign(v.dtw_factor) == np.sign(v.future_logret)).mean()),
            "bps": float(pnl.mean()),
            "t_clust": float(clustered_t(v, "pnl_bps"))}


def main() -> None:
    close = pd.read_parquet(PRICE_FILE, columns=["close"]).sort_index()
    close.index = (close.index.tz_localize("UTC") if close.index.tz is None
                   else close.index.tz_convert("UTC"))
    close = close["close"].astype(float)
    events = pd.read_parquet(EVENT_FILE)
    events["timestamp_utc"] = pd.to_datetime(events.timestamp_utc, utc=True)
    events = events.sort_values("timestamp_utc").reset_index(drop=True)
    print(f"price rows {len(close):,}  events {len(events):,}")

    cache = {}
    for group, types in EVENT_TYPE_GROUPS.items():
        for freq in PATH_FREQ_MIN_OPTIONS:
            cache[(group, freq)] = build_games(close, events, types, freq)
            print(f"  built games: {group:<24} freq={freq}m -> {len(cache[(group, freq)]):,}")

    rows, panels = [], {}
    for group, freq, lookback in product(EVENT_TYPE_GROUPS, PATH_FREQ_MIN_OPTIONS, LOOKBACK_YEARS_OPTIONS):
        steps = max(1, int(round(WARP_TOLERANCE_MINUTES / freq)))
        scored = compute_factor(cache[(group, freq)], lookback, steps)
        panels[(group, freq, lookback)] = scored
        sel = score(scored[scored.t0_utc < SPLIT])
        con = score(scored[scored.t0_utc >= SPLIT])
        full = score(scored)
        rows.append({"event_group": group, "path_freq_min": freq, "lookback_years": lookback,
                     "full_ic": full["ic"],
                     "select_n": sel["n"], "select_ic": sel["ic"],
                     "confirm_n": con["n"], "confirm_ic": con["ic"],
                     "confirm_bps": con["bps"], "confirm_t": con["t_clust"]})
        print(f"  {group:<24} freq={freq} lb={lookback}  full IC {full['ic']:+.4f}  "
              f"select {sel['ic']:+.4f}  confirm {con['ic']:+.4f}")

    grid = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grid.to_csv(OUT_DIR / "hoshea_grid_holdout.csv", index=False)

    print("\n" + "=" * 90)
    print("GRID, ranked by the SELECTION period (2012-2021). Confirmation is 2022-2026.")
    print("=" * 90)
    print(grid.sort_values("select_ic", ascending=False).round(4).to_string(index=False))

    best = grid.loc[grid.select_ic.idxmax()]
    best_full = grid.loc[grid.full_ic.idxmax()]
    print(f"\nchosen on 2012-2021 : {best.event_group}, freq={int(best.path_freq_min)}m, "
          f"lookback={int(best.lookback_years)}y  -> selection IC {best.select_ic:+.4f}")
    print(f"  its 2022-2026 IC  : {best.confirm_ic:+.4f}  "
          f"({best.confirm_bps:+.2f} bps/game, clustered t {best.confirm_t:+.2f}, n={int(best.confirm_n)})")
    print(f"Hoshea's full-sample pick: {best_full.event_group}, freq={int(best_full.path_freq_min)}m, "
          f"lookback={int(best_full.lookback_years)}y -> full IC {best_full.full_ic:+.4f}, "
          f"confirm IC {best_full.confirm_ic:+.4f}")
    print(f"\nrank correlation, selection IC vs confirmation IC across the 18 cells: "
          f"{grid.select_ic.corr(grid.confirm_ic, method='spearman'):+.3f}")

    print("\n" + "=" * 90)
    print("PER EVENT TYPE under the selection-period winner")
    print("=" * 90)
    panel = panels[(best.event_group, int(best.path_freq_min), int(best.lookback_years))]
    print(f"  {'event':<24} {'period':<12} {'n':>5} {'IC':>9} {'hit':>7} {'bps':>8} {'t_clust':>8} {'net':>7}")
    for event_type, g in panel.groupby("event_type"):
        for label, sub in (("select", g[g.t0_utc < SPLIT]), ("confirm", g[g.t0_utc >= SPLIT])):
            s = score(sub)
            if not np.isfinite(s["ic"]):
                print(f"  {event_type:<24} {label:<12} {s['n']:>5}  (too few scored games)")
                continue
            print(f"  {event_type:<24} {label:<12} {s['n']:>5} {s['ic']:>+9.4f} {s['hit']:>6.1%} "
                  f"{s['bps']:>+8.2f} {s['t_clust']:>+8.2f} {s['bps']-TICK_BPS:>+7.2f}")

    panel.drop(columns=["path"]).to_parquet(OUT_DIR / "hoshea_holdout_best_panel.parquet", index=False)
    print(f"\nsaved -> {OUT_DIR/'hoshea_grid_holdout.csv'}, {OUT_DIR/'hoshea_holdout_best_panel.parquet'}")


if __name__ == "__main__":
    main()
