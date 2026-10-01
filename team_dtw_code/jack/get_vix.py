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
# # VIX pull
#
# Fetches the CBOE volatility indices from FRED and writes them to `data/raw/`. Implied
# volatility is the best-documented predictor of realised volatility and consistently
# beats backward-looking estimators, and nothing else in the feature set is
# forward-looking, so it is the one input here that carries information the price history
# cannot.
#
# Two series:
#
# - `VIXCLS`, 30-day implied volatility on the S&P 500
# - `VXVCLS`, the 3-month equivalent, whose ratio to VIX gives a term-structure slope
#
# **Publication timing matters.** These are daily closes stamped at 16:15 ET. A bar can
# only use a close that has already been published, so the merge in
# `feature_engineering.py` is done against a publication timestamp rather than a date.
#
# Run this once, or whenever the sample is extended. It writes raw data and nothing else.

# %%
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

RAW = Path("../data/raw")
SERIES = {"vix": "VIXCLS", "vix3m": "VXVCLS"}
PUBLISH_ET = pd.Timedelta(hours=16, minutes=15)     # CBOE close stamp

load_dotenv(Path("../.env"))
fred = Fred(api_key=os.environ["FRED_API_KEY"])

# %%
frames = {}
for name, code in SERIES.items():
    s = fred.get_series(code).rename(name)
    s.index = pd.to_datetime(s.index)
    frames[name] = s
    print(f"{code:<8} {len(s):>6} observations  {s.index.min().date()} -> {s.index.max().date()}")

vix = pd.concat(frames.values(), axis=1).sort_index()
vix.index.name = "date"

# %% [markdown]
# Holidays and the odd missing print leave gaps. Forward-filling would invent a close
# that was never published, so rows without a VIX level are dropped instead and the
# as-of merge downstream carries the last real value.

# %%
before = len(vix)
vix = vix.dropna(subset=["vix"])
print(f"dropped {before - len(vix)} rows with no VIX close")
print(f"rows with no 3-month value: {vix.vix3m.isna().sum()} "
      f"(series starts {frames['vix3m'].index.min().date()})")

# publication instant, in UTC, so downstream merges cannot peek
vix["published_utc"] = (vix.index + PUBLISH_ET).tz_localize("America/New_York").tz_convert("UTC")

# %%
out = RAW / "vix_daily.parquet"
vix.to_parquet(out)
print(f"wrote {out}  {len(vix):,} rows x {len(vix.columns)} columns")
print(vix.tail(3).to_string())
print()
print(vix[["vix", "vix3m"]].describe().round(2).to_string())

# %% [markdown]
# **Findings:** 9,576 daily VIX closes back to 1990 and 4,901 three-month closes back to
# December 2007, so both cover the 2016 onward modelling sample in full. VIX averages
# 19.5 over the whole history with a maximum of 82.7 in March 2020. The file is raw data
# and gitignored, so anyone rebuilding the pipeline runs this notebook once.
