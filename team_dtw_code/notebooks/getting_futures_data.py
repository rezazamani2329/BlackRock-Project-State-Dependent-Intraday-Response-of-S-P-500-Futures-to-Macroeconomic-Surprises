# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: industry_venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 1. Getting Data
#
# Here, we use only the front-month continuous contract, ES.c.0. Since ES.c.0 is a continuous contract rather than an individual futures contract, it does not have a fixed expiration date. Instead, it dynamically maps to a specific underlying futures contract at each point in time and rolls from the current front-month contract to the next contract according to a predefined calendar-based rule.
#
# Specifically, ES.c.0 always refers to the contract with the nearest expiration among the available contracts. When the current front-month contract reaches its expiration, the series automatically rolls to the next available contract.
#
# One important caveat is that the continuous series is constructed by directly concatenating the original prices of the underlying contracts, without back-adjustment. As a result, the series may exhibit an artificial price jump at the rollover point, reflecting the difference in prices between the two futures contracts rather than an actual market movement.
#
# Moreover, trading volume typically migrates from the expiring contract to the next contract before the calendar-based rollover occurs. Consequently, the volume associated with ES.c.0 may become less representative of the most actively traded contract around the rollover period.
#
# Despite these limitations, ES.c.0 remains a commonly used continuous futures series for empirical research and historical analysis.

# %%
import os
import databento as db
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# %%
load_dotenv()  # reads DATABENTO_API_KEY from a .env file at the repo root

gotten_data = True

API_KEY = os.getenv("DATABENTO_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "DATABENTO_API_KEY not set. Copy .env.example to .env at the repo root "
        "and fill in your Databento API key."
    )
client = db.Historical(API_KEY)

output_file = "../data/raw/ES_1m_2010_2026_full.parquet"

if not gotten_data:
    # Monthly date breakpoints
    date_ranges = pd.date_range(start="2010-06-01", end="2026-07-01", freq="MS")

    df_list = []

    print("Starting batch download of ES 1-minute bars with all fields...")

    for i in tqdm(range(len(date_ranges) - 1), desc="Download progress", unit="month"):
        start_str = date_ranges[i].strftime("%Y-%m-%d")
        end_str = date_ranges[i + 1].strftime("%Y-%m-%d")

        try:
            data = client.timeseries.get_range(
                dataset="GLBX.MDP3",
                symbols=["ES.c.0"],
                stype_in="continuous",
                schema="ohlcv-1m",
                start=start_str,
                end=end_str,
            )

            monthly_df = data.to_df()

            if not monthly_df.empty:
                # Keep all original columns (symbol, rtype, publisher_id, and other metadata)
                df_list.append(monthly_df)

        except Exception as e:
            print(f"\nFailed to extract {start_str} to {end_str}: {e}")

    # Concatenate all monthly chunks
    if df_list:
        print("\nMerging data and saving to Parquet...")
        full_df = pd.concat(df_list)

        # Drop duplicate timestamps and sort
        full_df = full_df[~full_df.index.duplicated(keep='first')].sort_index()

        full_df.to_parquet(output_file)
        print(f"Download complete: {len(full_df):,} rows with all fields saved to `{output_file}`.")

    else:
        full_df = pd.read_parquet(output_file)

else:
    # Data already downloaded — load the existing raw file instead of re-pulling.
    full_df = pd.read_parquet(output_file)

full_df.index = full_df.index.tz_convert('US/Eastern')

# %%
full_df.tail()

# %% [markdown]
# ## 1.2 Plot a daily candlestick chart
#
# `plot_daily_candles` filters `full_df` to one calendar day, resamples the 1-minute bars, and draws candles plus volume. Call it once per date you want to inspect.

# %%
import mplfinance as mpf

def plot_daily_candles(df, plot_date, freq="5min", tz=None, title=None):
    """Plot one calendar day's candlesticks from 1-minute OHLCV data."""
    ohlcv = df[["open", "high", "low", "close", "volume"]].copy()
    if ohlcv.index.tz is None:
        ohlcv.index = ohlcv.index.tz_localize("UTC")
    if tz is None:
        tz = str(ohlcv.index.tz)
    ohlcv.index = ohlcv.index.tz_convert(tz).tz_localize(None)

    try:
        day = ohlcv.loc[plot_date]
    except KeyError:
        print(f"No bars on {plot_date} ({tz}). CME may have been closed that day.")
        return None

    bars = (
        day.resample(freq)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna(subset=["open"])
    )
    mpf.plot(
        bars,
        type="candle",
        volume=True,
        style="yahoo",
        title=title or f"ES.c.0  {plot_date}  {freq}  {tz}",
        ylabel="Price",
        ylabel_lower="Volume",
        figsize=(14, 7),
        tight_layout=True,
    )
    print(f"{plot_date}: {len(day):,} one-minute bars -> {len(bars):,} {freq} candles")
    return bars


# %%
plot_daily_candles(full_df, "2026-06-12")

# %%
plot_daily_candles(full_df, "2026-03-12")
