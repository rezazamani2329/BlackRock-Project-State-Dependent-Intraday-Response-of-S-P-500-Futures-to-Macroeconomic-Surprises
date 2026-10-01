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
Data cleaning

Turns the raw databento ES continuous-futures export into a clean OHLCV
parquet in data/processed/. This is the only notebook that writes to
data/processed/ — everything else (data_exploration, etc.) only reads from it.
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path("../data")
RAW_FILE = DATA_DIR / "raw" / "ES_1m_2010_2026_full.parquet"
PROCESSED_FILE = DATA_DIR / "processed" / "es_1min_bars_2010_2026.parquet"

pd.set_option("display.width", 120)

# %% [markdown]
# ## Load raw + clean

# %%
raw = pd.read_parquet(RAW_FILE)
raw.head()

# %%
# Drop databento request/session metadata that isn't part of the price series
# (rtype/publisher_id are constant; instrument_id changes as the continuous
# contract rolls and isn't needed for price analysis).
df = raw[["open", "high", "low", "close", "volume"]].copy()
df.index.name = "timestamp"
df = df.sort_index()
df.head()

# %%
PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(PROCESSED_FILE)
print(f"wrote {len(df):,} rows -> {PROCESSED_FILE}")

# %% [markdown]
# **Findings:** Dropped databento's `rtype`/`publisher_id`/`instrument_id` columns
# (session metadata, not price data) and wrote 4,811,336 clean OHLCV rows to
# `data/processed/es_1min_bars_2010_2026.parquet`.
