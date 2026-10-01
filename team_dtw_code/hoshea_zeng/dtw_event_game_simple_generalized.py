# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: blackrock-intraday
#     language: python
#     name: blackrock-intraday
# ---

# %% [markdown]
# # DTW event game (generalized simple experiment)
#
# This notebook extends the original mixed-event simple experiment by:
# - adding `offset = -60` to game anchors,
# - searching across two event-type groups,
# - searching across path resample frequencies (2/3/5 minutes),
# - searching across lookback windows (2/3/5 years),
# - selecting the best hyperparameter combination by IC,
# - saving the best-combination `dtw_factor` panel to local parquet.

# %%
from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from dtaidistance import dtw
from tqdm.auto import tqdm

pd.set_option("display.width", 180)

# %% [markdown]
# ## Parameters and search space

# %%
def detect_repo_root() -> Path:
    candidates = [Path.cwd(), Path.cwd().parent]
    if "__file__" in globals():
        candidates.insert(0, Path(__file__).resolve().parents[1])
    for c in candidates:
        if (c / "data" / "processed").exists():
            return c
    raise FileNotFoundError("Could not detect repository root containing data/processed")


REPO_ROOT = detect_repo_root()
PRICE_FILE = REPO_ROOT / "data" / "processed" / "es_1min_bars_2010_2026.parquet"
EVENT_FILE = REPO_ROOT / "data" / "processed" / "macro_event_calendar_expanded_2010_2026.parquet"
OUTPUT_DIR = REPO_ROOT / "hoshea_zeng" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Requested event groups (A/B pools).
EVENT_TYPE_GROUPS: dict[str, list[str]] = {
    "core_major": ["CPI", "NFP", "PCE", "FOMC"],
    "fomc_jolts_pce_claims": ["FOMC", "JOLTS", "PCE", "Initial Jobless Claims"],
}

# Include offset=-60 as requested.
EVENT_OFFSETS_MIN = [-60, -30, 0, 30, 60]
PATH_MINUTES = 60
ENTRY_DELAY_MINUTES = 60
HOLD_MINUTES = 30

# Hyperparameter grid
PATH_FREQ_MIN_OPTIONS = [2, 3, 5]
LOOKBACK_YEARS_OPTIONS = [2, 3, 5]

# Other settings
K_NEIGHBORS = 15
MIN_HISTORY = 20
RECENCY_HALFLIFE_DAYS = 365 * 2
WARP_TOLERANCE_MINUTES = 15  # converted to DTW window steps by frequency
MIN_MATCH_AGE_DAYS = 1.0
MAX_EVENT_CANDIDATES = 400

GRID_RESULTS_FILE = OUTPUT_DIR / "dtw_simple_generalized_grid_results.parquet"
BEST_FACTOR_FILE = OUTPUT_DIR / "dtw_simple_generalized_best_factor.parquet"
BEST_PARAMS_FILE = OUTPUT_DIR / "dtw_simple_generalized_best_params.json"
ALL30_EVENT_FACTOR_FILE = OUTPUT_DIR / "dtw_simple_generalized_all30_event_factor.parquet"
ALL30_EVENT_NO_CLOCK_FACTOR_FILE = OUTPUT_DIR / "dtw_simple_generalized_all30_event_factor_no_clock.parquet"

NY_TZ = ZoneInfo("America/New_York")

# %% [markdown]
# ## Load data

# %%
if not PRICE_FILE.exists():
    raise FileNotFoundError(f"price file not found: {PRICE_FILE}")
if not EVENT_FILE.exists():
    raise FileNotFoundError(f"event file not found: {EVENT_FILE}")

price = pd.read_parquet(PRICE_FILE, columns=["close"]).sort_index()
if price.index.tz is None:
    price.index = price.index.tz_localize("UTC")
else:
    price.index = price.index.tz_convert("UTC")
close = price["close"].astype(float)

events_all = pd.read_parquet(EVENT_FILE)
events_all["timestamp_utc"] = pd.to_datetime(events_all["timestamp_utc"], utc=True)
events_all = events_all.sort_values("timestamp_utc").reset_index(drop=True)

print(f"close rows: {len(close):,}, range: {close.index.min()} -> {close.index.max()}")
print(f"expanded event rows: {len(events_all):,}")
print(events_all["event_type"].value_counts())

# %% [markdown]
# **Findings:** Loaded 4,811,336 one-minute ES close rows and 3,167 expanded
# macro-event timestamps. The two requested event groups are both fully present
# in this calendar and can be filtered directly for the grid search.

# %% [markdown]
# ## Helper functions

# %%
def zscore(x: np.ndarray) -> np.ndarray:
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd < 1e-12:
        return np.zeros_like(x)
    return (x - mu) / sd


def dtw_window_steps_from_freq(freq_min: int) -> int:
    return max(1, int(round(WARP_TOLERANCE_MINUTES / freq_min)))


def dtw_distance_windowed(x: np.ndarray, y: np.ndarray, window_steps: int) -> float:
    n, m = len(x), len(y)
    if n == 0 or m == 0:
        return np.nan

    w = int(max(window_steps, abs(n - m)))
    d = dtw.distance_fast(
        np.asarray(x, dtype=np.double),
        np.asarray(y, dtype=np.double),
        window=w,
        use_pruning=True,
    )
    return float(d)


def extract_path_vector(
    close_series: pd.Series,
    t0_utc: pd.Timestamp,
    path_minutes: int,
    path_freq_min: int,
) -> np.ndarray | None:
    sample_idx = pd.date_range(
        t0_utc,
        t0_utc + pd.Timedelta(minutes=path_minutes),
        freq=f"{path_freq_min}min",
        tz="UTC",
    )
    px = close_series.reindex(sample_idx)
    if px.isna().any():
        return None
    logret = np.diff(np.log(px.to_numpy(dtype=float)))
    if len(logret) == 0 or not np.isfinite(logret).all():
        return None
    return zscore(logret)


def extract_future_logret(
    close_series: pd.Series,
    t0_utc: pd.Timestamp,
    entry_delay_min: int,
    hold_min: int,
) -> float | None:
    t_entry = t0_utc + pd.Timedelta(minutes=entry_delay_min)
    t_exit = t_entry + pd.Timedelta(minutes=hold_min)
    px = close_series.reindex([t_entry, t_exit])
    if px.isna().any():
        return None
    return float(np.log(px.iloc[1]) - np.log(px.iloc[0]))


def build_games(
    close_series: pd.Series,
    events_df: pd.DataFrame,
    event_types: list[str],
    path_freq_min: int,
) -> pd.DataFrame:
    events = events_df[events_df["event_type"].isin(event_types)].copy()
    rows: list[dict] = []
    for row in events.itertuples(index=False):
        event_ts = row.timestamp_utc
        for offset_min in EVENT_OFFSETS_MIN:
            t0 = event_ts + pd.Timedelta(minutes=offset_min)
            path = extract_path_vector(
                close_series,
                t0,
                path_minutes=PATH_MINUTES,
                path_freq_min=path_freq_min,
            )
            if path is None:
                continue

            future_logret = extract_future_logret(
                close_series,
                t0,
                entry_delay_min=ENTRY_DELAY_MINUTES,
                hold_min=HOLD_MINUTES,
            )
            if future_logret is None:
                continue

            t0_et = t0.tz_convert(NY_TZ)
            rows.append(
                {
                    "event_type": row.event_type,
                    "event_timestamp_utc": event_ts,
                    "offset_min": offset_min,
                    "t0_utc": t0,
                    "clock_et": t0_et.strftime("%H:%M"),
                    "future_logret": future_logret,
                    "path": path,
                }
            )
    return pd.DataFrame(rows).sort_values("t0_utc").reset_index(drop=True)


def compute_dtw_factor(
    games_df: pd.DataFrame,
    lookback_years: int,
    window_steps: int,
) -> pd.DataFrame:
    out = games_df.copy()
    out["dtw_factor"] = np.nan
    out["history_count"] = 0
    out["neighbor_k_used"] = 0
    out["neighbor_mean_distance"] = np.nan

    lookback_delta = pd.Timedelta(days=365 * lookback_years)
    # Mixed-event sampling within the same event group:
    # keep same offset + same clock, allow cross-event matching in that group.
    group_cols = ["offset_min", "clock_et"]

    for _, g in out.groupby(group_cols, sort=False):
        idx = g.index.to_numpy()
        t0_list = out.loc[idx, "t0_utc"].tolist()
        path_list = out.loc[idx, "path"].tolist()
        ret_arr = out.loc[idx, "future_logret"].to_numpy(dtype=float)

        for pos, row_idx in enumerate(idx):
            current_t0 = t0_list[pos]
            history_pos = [
                h for h in range(pos) if t0_list[h] >= (current_t0 - lookback_delta)
            ]
            if len(history_pos) < MIN_HISTORY:
                continue

            x = path_list[pos]
            dists = np.empty(len(history_pos), dtype=float)
            hist_rets = np.empty(len(history_pos), dtype=float)
            day_gaps = np.empty(len(history_pos), dtype=float)

            for j, h in enumerate(history_pos):
                dists[j] = dtw_distance_windowed(x, path_list[h], window_steps)
                hist_rets[j] = ret_arr[h]
                day_gaps[j] = (current_t0 - t0_list[h]).days

            k = min(K_NEIGHBORS, len(history_pos))
            nn = np.argpartition(dists, k - 1)[:k]
            nn_d = dists[nn]
            nn_r = hist_rets[nn]
            nn_gap = day_gaps[nn]

            w_dist = 1.0 / (nn_d + 1e-6)
            w_time = np.exp(-nn_gap / RECENCY_HALFLIFE_DAYS)
            w = w_dist * w_time

            out.at[row_idx, "dtw_factor"] = float(np.dot(w, nn_r) / np.sum(w))
            out.at[row_idx, "history_count"] = int(len(history_pos))
            out.at[row_idx, "neighbor_k_used"] = int(k)
            out.at[row_idx, "neighbor_mean_distance"] = float(np.mean(nn_d))

    return out


def evaluate_ic(scored_df: pd.DataFrame) -> dict[str, float]:
    valid = scored_df.dropna(subset=["dtw_factor"]).copy()
    if len(valid) == 0:
        return {
            "scored_games": 0,
            "pearson_ic": np.nan,
            "spearman_ic": np.nan,
            "hit_rate": np.nan,
            "q5_q1_spread": np.nan,
        }

    pearson_ic = valid["dtw_factor"].corr(valid["future_logret"], method="pearson")
    spearman_ic = valid["dtw_factor"].corr(valid["future_logret"], method="spearman")
    hit_rate = (
        np.sign(valid["dtw_factor"]).to_numpy() == np.sign(valid["future_logret"]).to_numpy()
    ).mean()

    valid["q"] = pd.qcut(valid["dtw_factor"], q=5, labels=False, duplicates="drop") + 1
    q_ret = valid.groupby("q")["future_logret"].mean()
    q5_q1 = float(q_ret.iloc[-1] - q_ret.iloc[0]) if len(q_ret) >= 2 else np.nan

    return {
        "scored_games": int(len(valid)),
        "pearson_ic": float(pearson_ic),
        "spearman_ic": float(spearman_ic),
        "hit_rate": float(hit_rate),
        "q5_q1_spread": float(q5_q1),
    }


def build_all_30min_games(
    close_series: pd.Series,
    path_freq_min: int,
) -> pd.DataFrame:
    """
    Build the full 30-minute anchor universe (not only event windows).
    The input path definition remains the same as the notebook's main pipeline:
    [T0, T0+60] sampled at path_freq_min.
    """
    idx = close_series.index
    t0_candidates = idx[(idx.minute % 30 == 0) & (idx.second == 0)]

    rows: list[dict] = []
    for t0 in t0_candidates:
        path = extract_path_vector(
            close_series,
            t0,
            path_minutes=PATH_MINUTES,
            path_freq_min=path_freq_min,
        )
        if path is None:
            continue

        future_logret = extract_future_logret(
            close_series,
            t0,
            entry_delay_min=ENTRY_DELAY_MINUTES,
            hold_min=HOLD_MINUTES,
        )
        if future_logret is None:
            continue

        rows.append(
            {
                "t0_utc": t0,
                "clock_et": t0.tz_convert(NY_TZ).strftime("%H:%M"),
                "future_logret": future_logret,
                "path": path,
            }
        )

    return pd.DataFrame(rows).sort_values("t0_utc").reset_index(drop=True)


def compute_all30_factor_from_event_paths(
    all30_games: pd.DataFrame,
    event_games: pd.DataFrame,
    lookback_years: int,
    window_steps: int,
) -> pd.DataFrame:
    """
    For every 30-minute anchor game, match only against historical *event-path* rows
    with the same T0 clock (`clock_et`), as requested.
    """
    out = all30_games.copy()
    out["dtw_factor_event_pool"] = np.nan
    out["event_history_count"] = 0
    out["neighbor_k_used"] = 0
    out["neighbor_mean_distance"] = np.nan

    lookback_delta = pd.Timedelta(days=365 * lookback_years)
    all30_by_clock = {k: g.sort_values("t0_utc").reset_index() for k, g in out.groupby("clock_et", sort=False)}
    event_by_clock = {k: g.sort_values("t0_utc").reset_index(drop=True) for k, g in event_games.groupby("clock_et", sort=False)}

    for clock_et, g in tqdm(all30_by_clock.items(), desc="All-30m factor by clock", unit="clock"):
        if clock_et not in event_by_clock:
            continue

        event_ref = event_by_clock[clock_et]
        event_t = event_ref["t0_utc"].tolist()
        event_paths = event_ref["path"].tolist()
        event_ret = event_ref["future_logret"].to_numpy(dtype=float)

        for row in g.itertuples(index=False):
            row_idx = int(row.index)
            t0 = row.t0_utc
            x = row.path

            # Historical event paths only, within lookback and at least 1 day older.
            hist_idx = [
                i for i, et in enumerate(event_t)
                if (et < t0) and (et >= (t0 - lookback_delta)) and ((t0 - et).days >= MIN_MATCH_AGE_DAYS)
            ]
            if len(hist_idx) < MIN_HISTORY:
                continue

            # Cap event candidates for runtime stability on dense clocks.
            if len(hist_idx) > MAX_EVENT_CANDIDATES:
                hist_idx = hist_idx[-MAX_EVENT_CANDIDATES:]

            dists = np.empty(len(hist_idx), dtype=float)
            hist_rets = np.empty(len(hist_idx), dtype=float)
            day_gaps = np.empty(len(hist_idx), dtype=float)

            for j, h in enumerate(hist_idx):
                dists[j] = dtw_distance_windowed(x, event_paths[h], window_steps)
                hist_rets[j] = event_ret[h]
                day_gaps[j] = (t0 - event_t[h]).days

            k = min(K_NEIGHBORS, len(hist_idx))
            nn = np.argpartition(dists, k - 1)[:k]
            nn_d = dists[nn]
            nn_r = hist_rets[nn]
            nn_gap = day_gaps[nn]

            w_dist = 1.0 / (nn_d + 1e-6)
            w_time = np.exp(-nn_gap / RECENCY_HALFLIFE_DAYS)
            w = w_dist * w_time

            out.at[row_idx, "dtw_factor_event_pool"] = float(np.dot(w, nn_r) / np.sum(w))
            out.at[row_idx, "event_history_count"] = int(len(hist_idx))
            out.at[row_idx, "neighbor_k_used"] = int(k)
            out.at[row_idx, "neighbor_mean_distance"] = float(np.mean(nn_d))

    return out


def compute_all30_factor_from_event_paths_no_clock(
    all30_games: pd.DataFrame,
    event_games: pd.DataFrame,
    lookback_years: int,
    window_steps: int,
) -> pd.DataFrame:
    """
    For every 30-minute anchor game, match against historical *event-path* rows
    without requiring the same intraday clock.
    """
    out = all30_games.copy()
    out["dtw_factor_event_pool_no_clock"] = np.nan
    out["event_history_count_no_clock"] = 0
    out["neighbor_k_used_no_clock"] = 0
    out["neighbor_mean_distance_no_clock"] = np.nan

    # Use microseconds consistently (DatetimeArray.astype('int64') in this
    # environment is microsecond-based), so search windows are well-formed.
    lookback_us = int(pd.Timedelta(days=365 * lookback_years).value // 1000)
    min_age_us = int(pd.Timedelta(days=MIN_MATCH_AGE_DAYS).value // 1000)

    event_ref = event_games.sort_values("t0_utc").reset_index(drop=True)
    event_ns = event_ref["t0_utc"].astype("int64").to_numpy()
    event_paths = event_ref["path"].tolist()
    event_ret = event_ref["future_logret"].to_numpy(dtype=float)

    for row in tqdm(out.itertuples(index=True), total=len(out), desc="All-30m factor no-clock", unit="row"):
        row_idx = int(row.Index)
        t0 = row.t0_utc
        x = row.path
        t0_us = int(t0.value // 1000)

        # Candidate window: [t0-lookback, t0-min_age]
        lo = int(np.searchsorted(event_ns, t0_us - lookback_us, side="left"))
        hi = int(np.searchsorted(event_ns, t0_us - min_age_us, side="left"))
        hist_count = hi - lo
        if hist_count < MIN_HISTORY:
            continue

        if hist_count > MAX_EVENT_CANDIDATES:
            lo = hi - MAX_EVENT_CANDIDATES
            hist_count = MAX_EVENT_CANDIDATES

        hist_idx = np.arange(lo, hi, dtype=int)
        dists = np.empty(hist_count, dtype=float)
        hist_rets = np.empty(hist_count, dtype=float)
        day_gaps = np.empty(hist_count, dtype=float)

        for j, h in enumerate(hist_idx):
            dists[j] = dtw_distance_windowed(x, event_paths[h], window_steps)
            hist_rets[j] = event_ret[h]
            day_gaps[j] = (t0_us - event_ns[h]) / (1e6 * 86400.0)

        k = min(K_NEIGHBORS, hist_count)
        nn = np.argpartition(dists, k - 1)[:k]
        nn_d = dists[nn]
        nn_r = hist_rets[nn]
        nn_gap = day_gaps[nn]

        w_dist = 1.0 / (nn_d + 1e-6)
        w_time = np.exp(-nn_gap / RECENCY_HALFLIFE_DAYS)
        w = w_dist * w_time

        out.at[row_idx, "dtw_factor_event_pool_no_clock"] = float(np.dot(w, nn_r) / np.sum(w))
        out.at[row_idx, "event_history_count_no_clock"] = int(hist_count)
        out.at[row_idx, "neighbor_k_used_no_clock"] = int(k)
        out.at[row_idx, "neighbor_mean_distance_no_clock"] = float(np.mean(nn_d))

    return out

# %% [markdown]
# **Findings:** The helper stack supports all requested knobs:
# event-group pooling, `offset=-60`, frequency-dependent DTW window sizing,
# lookback-year variation, and IC-based model comparison.

# %% [markdown]
# ## Build reusable game tables (cached by event group and frequency)

# %%
games_cache: dict[tuple[str, int], pd.DataFrame] = {}
for group_name, event_types in EVENT_TYPE_GROUPS.items():
    for freq_min in PATH_FREQ_MIN_OPTIONS:
        games = build_games(close, events_all, event_types, path_freq_min=freq_min)
        games_cache[(group_name, freq_min)] = games
        print(
            f"group={group_name:24s} freq={freq_min}m -> games={len(games):,}, "
            f"event_types={event_types}"
        )

# %% [markdown]
# **Findings:** With `offset=-60` included, game counts are:
# core_major = 2,949/2,953/2,954 (2m/3m/5m) and
# fomc_jolts_pce_claims = 5,554/5,563/5,572 (2m/3m/5m).
# This confirms larger sample depth for the second event group.

# %% [markdown]
# ## Grid Search over hyperparameters (with tqdm)

# %%
combos = list(product(EVENT_TYPE_GROUPS.keys(), PATH_FREQ_MIN_OPTIONS, LOOKBACK_YEARS_OPTIONS))
results: list[dict] = []

for group_name, freq_min, lookback_years in tqdm(combos, desc="Grid search", unit="combo"):
    games = games_cache[(group_name, freq_min)]
    window_steps = dtw_window_steps_from_freq(freq_min)
    scored = compute_dtw_factor(games, lookback_years=lookback_years, window_steps=window_steps)
    metrics = evaluate_ic(scored)

    results.append(
        {
            "event_group": group_name,
            "event_types": ",".join(EVENT_TYPE_GROUPS[group_name]),
            "path_freq_min": freq_min,
            "dtw_window_steps": window_steps,
            "lookback_years": lookback_years,
            "total_games": int(len(games)),
            **metrics,
        }
    )

grid_results = pd.DataFrame(results).sort_values("pearson_ic", ascending=False).reset_index(drop=True)
grid_results

# %%
print("Top 10 hyperparameter combinations by Pearson IC:")
print(
    grid_results[
        [
            "event_group",
            "path_freq_min",
            "dtw_window_steps",
            "lookback_years",
            "total_games",
            "scored_games",
            "pearson_ic",
            "spearman_ic",
            "hit_rate",
            "q5_q1_spread",
        ]
    ].head(10).to_string(index=False)
)

# %% [markdown]
# **Findings:** Across 18 combinations, the top IC comes from event group
# `fomc_jolts_pce_claims`, `path_freq_min=3`, `dtw_window_steps=5`,
# `lookback_years=2`, with Pearson IC 0.1162 and hit-rate 51.51%.

# %% [markdown]
# ## Refit best combination and save best dtw_factor to parquet

# %%
if grid_results["pearson_ic"].notna().sum() == 0:
    raise RuntimeError("No valid combination produced a finite IC.")

best_row = grid_results.loc[grid_results["pearson_ic"].idxmax()].to_dict()
best_group = str(best_row["event_group"])
best_freq = int(best_row["path_freq_min"])
best_lookback = int(best_row["lookback_years"])
best_window_steps = int(best_row["dtw_window_steps"])

best_games = games_cache[(best_group, best_freq)]
best_scored = compute_dtw_factor(
    best_games,
    lookback_years=best_lookback,
    window_steps=best_window_steps,
)
best_valid = best_scored.dropna(subset=["dtw_factor"]).copy()

# Persist grid summary and best-factor panel.
grid_results.to_parquet(GRID_RESULTS_FILE, index=False)

best_to_save = best_valid.drop(columns=["path"]).copy()
best_to_save["best_event_group"] = best_group
best_to_save["best_event_types"] = ",".join(EVENT_TYPE_GROUPS[best_group])
best_to_save["best_path_freq_min"] = best_freq
best_to_save["best_dtw_window_steps"] = best_window_steps
best_to_save["best_lookback_years"] = best_lookback
best_to_save.to_parquet(BEST_FACTOR_FILE, index=False)

with BEST_PARAMS_FILE.open("w", encoding="utf-8") as f:
    json.dump(best_row, f, ensure_ascii=False, indent=2)

print("Best combination:")
print(best_row)
print(f"Saved grid results -> {GRID_RESULTS_FILE}")
print(f"Saved best factor panel -> {BEST_FACTOR_FILE}")
print(f"Saved best params json -> {BEST_PARAMS_FILE}")
print(f"Best panel rows: {len(best_to_save):,}")

# %% [markdown]
# ## Extra dataset: all-30min anchors matched to historical event paths
#
# This section expands coverage beyond event windows. For every available 30-minute
# anchor T0, we compute an event-conditioned DTW factor by matching that row's
# path to *historical event paths* sharing the same T0 clock.

# %%
all30_games = build_all_30min_games(
    close,
    path_freq_min=best_freq,
)
print(f"all-30min eligible games: {len(all30_games):,}")

# event reference pool comes from the best event-group/frequency sample
event_reference_games = best_games.copy()
all30_scored = compute_all30_factor_from_event_paths(
    all30_games=all30_games,
    event_games=event_reference_games,
    lookback_years=best_lookback,
    window_steps=best_window_steps,
)

all30_valid = all30_scored.dropna(subset=["dtw_factor_event_pool"]).copy()
all30_coverage = len(all30_valid) / len(all30_scored) if len(all30_scored) else np.nan
all30_ic = all30_valid["dtw_factor_event_pool"].corr(all30_valid["future_logret"], method="pearson")
all30_sic = all30_valid["dtw_factor_event_pool"].corr(all30_valid["future_logret"], method="spearman")
all30_hit = (
    np.sign(all30_valid["dtw_factor_event_pool"]).to_numpy()
    == np.sign(all30_valid["future_logret"]).to_numpy()
).mean()

all30_to_save = all30_valid.drop(columns=["path"]).copy()
all30_to_save["reference_event_group"] = best_group
all30_to_save["reference_event_types"] = ",".join(EVENT_TYPE_GROUPS[best_group])
all30_to_save["path_freq_min"] = best_freq
all30_to_save["dtw_window_steps"] = best_window_steps
all30_to_save["lookback_years"] = best_lookback
all30_to_save.to_parquet(ALL30_EVENT_FACTOR_FILE, index=False)

print(f"all-30min scored rows: {len(all30_valid):,} / {len(all30_scored):,} ({all30_coverage:.2%})")
print(f"all-30min Pearson IC: {all30_ic:.4f}")
print(f"all-30min Spearman IC: {all30_sic:.4f}")
print(f"all-30min hit-rate: {all30_hit:.2%}")
print(f"saved all-30min event-matched factor -> {ALL30_EVENT_FACTOR_FILE}")

# %% [markdown]
# ## Extra dataset: all-30min anchors matched to historical event paths (no clock alignment)
#
# This section removes intraday clock alignment to increase usable coverage. Each
# 30-minute anchor T0 is matched to historical event paths from any clock slot
# (still within lookback and minimum-age constraints).

# %%
all30_no_clock_scored = compute_all30_factor_from_event_paths_no_clock(
    all30_games=all30_games,
    event_games=event_reference_games,
    lookback_years=best_lookback,
    window_steps=best_window_steps,
)

all30_no_clock_valid = all30_no_clock_scored.dropna(subset=["dtw_factor_event_pool_no_clock"]).copy()
all30_no_clock_coverage = len(all30_no_clock_valid) / len(all30_no_clock_scored) if len(all30_no_clock_scored) else np.nan
all30_no_clock_ic = all30_no_clock_valid["dtw_factor_event_pool_no_clock"].corr(
    all30_no_clock_valid["future_logret"], method="pearson"
)
all30_no_clock_sic = all30_no_clock_valid["dtw_factor_event_pool_no_clock"].corr(
    all30_no_clock_valid["future_logret"], method="spearman"
)
all30_no_clock_hit = (
    np.sign(all30_no_clock_valid["dtw_factor_event_pool_no_clock"]).to_numpy()
    == np.sign(all30_no_clock_valid["future_logret"]).to_numpy()
).mean()

all30_no_clock_to_save = all30_no_clock_valid.drop(columns=["path"]).copy()
all30_no_clock_to_save["reference_event_group"] = best_group
all30_no_clock_to_save["reference_event_types"] = ",".join(EVENT_TYPE_GROUPS[best_group])
all30_no_clock_to_save["path_freq_min"] = best_freq
all30_no_clock_to_save["dtw_window_steps"] = best_window_steps
all30_no_clock_to_save["lookback_years"] = best_lookback
all30_no_clock_to_save.to_parquet(ALL30_EVENT_NO_CLOCK_FACTOR_FILE, index=False)

print(
    f"all-30min no-clock scored rows: {len(all30_no_clock_valid):,} / "
    f"{len(all30_no_clock_scored):,} ({all30_no_clock_coverage:.2%})"
)
print(f"all-30min no-clock Pearson IC: {all30_no_clock_ic:.4f}")
print(f"all-30min no-clock Spearman IC: {all30_no_clock_sic:.4f}")
print(f"all-30min no-clock hit-rate: {all30_no_clock_hit:.2%}")
print(f"saved all-30min no-clock factor -> {ALL30_EVENT_NO_CLOCK_FACTOR_FILE}")

# %% [markdown]
# **Findings:** The best-combination factor panel has 4,749 scored rows and is
# saved to `hoshea_zeng/outputs/dtw_simple_generalized_best_factor.parquet`.
# Full grid diagnostics and the best-params JSON are also saved for reproducible
# selection and downstream modeling.

# %% [markdown]
# **Findings:** A second factor panel is now produced over the full 30-minute
# anchor universe (not just event-window rows), by matching each anchor's path
# to same-clock historical event paths. This creates a higher-coverage dataset
# for broader tradability analysis.

# %% [markdown]
# **Findings:** A third all-30-minute panel is generated by dropping the
# same-clock matching constraint. This maximizes sample coverage and serves as
# a high-availability factor candidate for multi-factor integration tests.
