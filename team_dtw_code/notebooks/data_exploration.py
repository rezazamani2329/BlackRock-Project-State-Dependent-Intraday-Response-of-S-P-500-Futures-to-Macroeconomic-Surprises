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

# %%
"""
Data exploration

Pure exploration of the already-cleaned ES 1-min bars in data/processed/
(see data_cleaning.py, which owns raw -> processed) — summary statistics,
anomaly scans (gaps, bad OHLC, price/volume spikes), and DTW prototyping.
This notebook only reads data/processed/; it never writes to it.
"""

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dtaidistance import dtw

DATA_DIR = Path("../data")
PROCESSED_FILE = DATA_DIR / "processed" / "es_1min_bars_2010_2026.parquet"

pd.set_option("display.width", 120)

# %% [markdown]
# ## Load processed data
#
# Assumes `data_cleaning.py` has already been run to produce this file.

# %%
df = pd.read_parquet(PROCESSED_FILE)
df.head()

# %% [markdown]
# ## Summary statistics

# %%
print("shape:", df.shape)
print("date range:", df.index.min(), "->", df.index.max())
print("duplicated timestamps:", df.index.duplicated().sum())
print("nulls per column:\n", df.isna().sum())

# %%
df.describe()

# %%
df.groupby(df.index.year).agg(
    rows=("close", "size"),
    avg_close=("close", "mean"),
    avg_volume=("volume", "mean"),
)

# %% [markdown]
# **Findings:** 2010-07-01 through 2026-06-30, no nulls or duplicate timestamps.
# Volume is heavily skewed (mean ~1,249/min, std ~21,400, max 6.2M) and, oddly,
# *declines* by year — avg volume/min drops from ~4,300 (2010-2012) to
# ~950-1,200 (2023-2026) even as price levels rise 6-7x. That's backwards for a
# market that's grown, and worth checking against the roll/continuous-contract
# methodology or a vendor aggregation change.

# %% [markdown]
# ## Anomaly scan: time gaps
#
# ES trades nearly 24h/day (CME Globex) with a ~1hr daily maintenance break
# and weekend closures, so gaps aren't inherently bad — but unusually large
# gaps outside that pattern are worth a look.

# %%
deltas = df.index.to_series().diff().dropna()
gap_counts = deltas.value_counts().sort_index(ascending=False)
gap_counts.head(15)

# %%
LARGE_GAP = pd.Timedelta(hours=2)
large_gaps = deltas[deltas > LARGE_GAP].sort_values(ascending=False)
print(f"{len(large_gaps)} gaps > {LARGE_GAP}")
large_gaps.head(20)

# %% [markdown]
# **Findings:** 1,508 gaps over 2 hours, topping out around 3d22h. All are
# consistent with weekends/holidays (e.g. Christmas, New Year's long weekends)
# — nothing unexplained beyond normal market closures.

# %% [markdown]
# ## Anomaly scan: OHLC consistency

# %%
bad_ohlc = df[
    (df["high"] < df["low"])
    | (df["high"] < df["open"])
    | (df["high"] < df["close"])
    | (df["low"] > df["open"])
    | (df["low"] > df["close"])
]
print(f"{len(bad_ohlc)} rows with inconsistent OHLC")
bad_ohlc.head(20)

# %%
non_positive = df[(df[["open", "high", "low", "close"]] <= 0).any(axis=1)]
print(f"{len(non_positive)} rows with non-positive prices")
non_positive.head(20)

# %% [markdown]
# **Findings:** Zero rows with inconsistent OHLC and zero with non-positive
# prices — the data is structurally clean on both counts.

# %% [markdown]
# ## Anomaly scan: price and volume spikes

# %%
returns = df["close"].pct_change()
z = (returns - returns.mean()) / returns.std()
price_spikes = df.loc[z.abs().sort_values(ascending=False).head(20).index]
price_spikes = price_spikes.assign(ret=returns.loc[price_spikes.index], z=z.loc[price_spikes.index])
price_spikes.sort_values("z", key=lambda s: s.abs(), ascending=False)

# %% [markdown]
# **Findings:** The largest 1-min moves line up with real events, not bad
# data — e.g. -10.4% at 2020-03-22 22:00 UTC (COVID Sunday-reopen crash),
# +6.7% at 2011-08-09 22:00 UTC (US credit downgrade weekend), -5.5% at
# 2020-03-16 (COVID circuit-breaker day). Most cluster right at the Sunday
# 22:00/23:00 UTC weekly reopen, where weekend news gets priced in on the
# first bar back.

# %%
zero_volume = df[df["volume"] == 0]
print(f"{len(zero_volume)} bars with zero volume ({len(zero_volume) / len(df):.3%})")

vol_z = (df["volume"] - df["volume"].mean()) / df["volume"].std()
volume_spikes = df.loc[vol_z.sort_values(ascending=False).head(20).index]
volume_spikes.assign(vol_z=vol_z.loc[volume_spikes.index])

# %% [markdown]
# **Findings:** No zero-volume bars. But most of the top-20 volume spikes
# fall exactly at `23:59:00` across unrelated dates (2011-08-09, 2011-08-05,
# 2014-10-15, ...) — a repeating minute-of-day pattern like that suggests a
# data artifact (session-boundary/settlement volume dumped into the last bar
# of the UTC day) rather than organic trading, and is worth confirming with
# databento/teammate before treating end-of-day volume as reliable.

# %% [markdown]
# ## Quick visual: full close series with largest gaps/spikes marked

# %%
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df.index, df["close"], linewidth=0.5)
ax.scatter(price_spikes.index, price_spikes["close"], color="red", s=15, label="price spike (top 20 |z|)")
ax.set_title("ES continuous front-month close, 2010-2026")
ax.legend()
fig.tight_layout()

# %% [markdown]
# ## DTW exploration: picking a calm month
#
# `dtaidistance`'s core distance function is C-compiled and fast per pair, but
# a full pairwise similarity matrix is O(N^2) in the number of sequences — so
# we start on a small, uneventful slice to get the API and cost right before
# scaling up. We pick the calmest full month by realized 1-min volatility.

# %%
monthly_ret = df["close"].pct_change()
monthly_vol = monthly_ret.groupby(df.index.tz_convert(None).to_period("M")).std().dropna()
monthly_rows = df.groupby(df.index.tz_convert(None).to_period("M")).size()
full_months = monthly_rows[monthly_rows > monthly_rows.median() * 0.9].index
calmest = monthly_vol.loc[monthly_vol.index.intersection(full_months)].sort_values().head(10)
calmest

# %%
CALM_MONTH = "2017-10"
month = df.loc[CALM_MONTH]
gaps_in_month = month.index.to_series().diff().dropna().sort_values(ascending=False)
print("rows:", len(month))
print("largest gaps:\n", gaps_in_month.head(5))
print("max abs 1-min return:", monthly_ret.loc[month.index].abs().max())

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(month.index, month["close"], linewidth=0.8)
ax.set_title(f"ES close, {CALM_MONTH}")
fig.tight_layout()

# %% [markdown]
# **Findings:** October 2017 is the calmest full month in the dataset by
# realized volatility (std of 1-min returns ~0.00011, vs. ~0.0002-0.0003 in a
# typical month) — consistent with 2017's well-known record-low-volatility
# regime. Its largest gaps are ordinary weekend closures (~2d1h) and its max
# 1-min move is 0.17%, so it's a clean, uneventful slice to prototype DTW on.

# %% [markdown]
# ## First DTW comparison: two days within the calm month
#
# Split the month into per-*session* close-price sequences (not raw UTC
# calendar date — ES trades ~22:00 UTC to ~21:00 UTC the next day, so naive
# `.date` grouping cuts a session in half) and time a single pairwise DTW
# distance, first unconstrained and then with a Sakoe-Chiba warping window.
#
# Note: the compiled C backend (`distance_fast`) isn't available in this env
# yet — it needs `libomp` via Homebrew, which is blocked by an untrusted tap
# already configured on this machine (`mongodb/brew`). Using the pure-Python
# `dtw.distance` for now; fine at this scale, revisit before scaling up.

# %%
session_date = (month.index.tz_convert("America/New_York") + pd.Timedelta(hours=6)).date
sessions = {str(day): g["close"].to_numpy() for day, g in month.groupby(session_date)}
# drop partial sessions at the edges of the month slice
sessions = {k: v for k, v in sessions.items() if len(v) > 1000}
session_keys = sorted(sessions.keys())
print(f"{len(session_keys)} full sessions in {CALM_MONTH}, lengths range:", min(len(v) for v in sessions.values()), "-", max(len(v) for v in sessions.values()))

s1, s2 = sessions[session_keys[0]], sessions[session_keys[1]]

t0 = time.perf_counter()
d_unconstrained = dtw.distance(s1, s2, use_c=False)
t1 = time.perf_counter()
d_windowed = dtw.distance(s1, s2, window=60, use_c=False)
t2 = time.perf_counter()

print(f"sessions compared: {session_keys[0]} (n={len(s1)}) vs {session_keys[1]} (n={len(s2)})")
print(f"unconstrained: distance={d_unconstrained:.4f}, time={t1 - t0:.4f}s")
print(f"window=60:     distance={d_windowed:.4f}, time={t2 - t1:.4f}s")

# %% [markdown]
# **Findings:** October 2017 has 22 full sessions, each ~1,310-1,360 minutes
# (consistent with a ~23h session minus the daily maintenance halt). One
# unconstrained pure-Python DTW pair (n≈1,350 each) took ~1.15s;
# `window=60` cut that to ~0.09s (~12x) at the cost of a higher, more
# constrained distance (282.9 vs 209.4 — expected, since a window can only
# raise the alignment cost relative to the unconstrained optimum). At
# ~1.15s/pair pure-Python, a full pairwise matrix over just this month's 22
# sessions is ~230 pairs / ~4-5 min; the full 2010-2026 dataset has ~4,000
# sessions (~8M pairs), which is not feasible without both the compiled C
# backend (blocked on the `libomp`/Homebrew tap issue above) and a warping
# window. Getting the C backend working is the next real blocker before any
# dataset-scale DTW work.
