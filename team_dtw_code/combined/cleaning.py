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
# # Cleaning — sessions, holidays, rolls, for ES, NQ and ZN (and candidate YM)
#
# Turns the raw 1-minute exports from `pull.py` into one cleaned file per asset,
# `futures_1min_clean_v3_{ES,NQ,ZN}.parquet`, all with the same columns. One file per asset
# keeps each under GitHub's 100 MB limit, so the cleaned data can be committed.
#
# Every asset goes through the same steps:
#
# 1. **Sessions.** A session runs 18:00 ET to the next 17:00 ET and is dated by the day it
#    closes on.
# 2. **Session filter.** Drop exchange holidays, short sessions and, for ES, NQ and YM, quarterly
#    expiry Fridays. "Short" is relative to the asset: fewer than 85% of that asset's median
#    bars per weekday session that year. ZN and early NQ genuinely trade fewer minutes than
#    ES, so a fixed ES-sized cut would throw away most of their early years.
# 3. **Rolls.** Checked, not assumed: no contract change may fall in the daytime hours where
#    release windows live. The contract ID is kept so buy-and-hold can skip roll jumps.
# 4. **Missing minutes.** A 1-minute bar exists only if something traded. A minute with no
#    bar is filled with the last traded price, within the session only, and flagged
#    `is_filled`. This uses only past data. A fresh ZT pull matched tick-level trades
#    exactly, which is how we know the gaps are no-trade minutes rather than missing data.
#
# ZT (2-year note) was dropped: before 2019 under 7% of its release windows had a trade in
# every minute. `es_1min_clean_v2.parquet`, the old ES-only file, is no longer written.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar, GoodFriday, Holiday, USLaborDay, USMemorialDay,
    USThanksgivingDay, nearest_workday, sunday_to_monday,
)

sys.path.insert(0, str(Path(__file__).resolve().parent if "__file__" in globals() else "."))
from config import (ALL_ASSETS, ASSETS, CLEAN_BARS, CLEAN_BARS_FMT, MIN_SESSION_FRAC, MODEL_START,
                    OUT, SESSION_SHIFT)

pd.set_option("display.width", 150)

# %% [markdown]
# ## Exchange holidays
#
# The bar-count cut is the filter; this calendar is a cross-check. One observance
# subtlety: ES traded a full session on Friday 2021-12-31 even though 2022-01-01 was a
# Saturday, so New Year's uses `sunday_to_monday` while Independence Day and Christmas use
# `nearest_workday`. The cross-check below is what catches this if it is ever wrong.
#
# CBOT Treasury futures close on the same days. The one difference is Good Friday when it
# coincides with payrolls (2012, 2015, 2021, 2023, 2026), when rates open for an abbreviated
# session — too short to pass the session filter, so it is dropped either way.

# %%
class CMEEquityIndexCalendar(AbstractHolidayCalendar):
    """Days CME equity index futures (ES) are fully closed."""

    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-06-19", observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


# %% [markdown]
# ## One asset, start to finish
#
# **Rolls.** Databento's continuous series changes contract at a fixed instant: the Sunday
# 18:00 open for the ES/NQ calendar roll, and 00:00 UTC (19:00 or 20:00 ET) for the ZN
# volume roll. The ZN switch therefore lands an hour or two *inside* the evening session.
# That is harmless for the strategy as long as it never falls in the daytime, where every
# release window lives, so that is what is asserted.
#
# **Filling.** Each kept session is put on a complete minute grid from its first bar to its
# last. Filled minutes carry the previous close as open, high, low and close, zero volume,
# and `is_filled = True`. The session filter counts real bars only, before filling.

# %%
def clean_asset(sym: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_parquet(ALL_ASSETS[sym]["raw"]).sort_index()
    raw = raw[~raw.index.duplicated(keep="first")]
    assert str(raw.index.tz) == "UTC", f"{sym}: expected UTC index, got {raw.index.tz}"
    bars = raw[["open", "high", "low", "close", "volume"]].copy()
    bars["contract"] = raw.instrument_id
    bars.index.name = "timestamp"

    et = bars.index.tz_convert("America/New_York")
    session_date = pd.Series((et + SESSION_SHIFT).date, index=bars.index)

    # rolls: none in the daytime
    switch = (bars.contract.ne(bars.contract.shift()) & session_date.eq(session_date.shift())).to_numpy()
    minute = et.hour * 60 + et.minute
    daytime = (minute >= 6 * 60) & (minute < 17 * 60)
    assert not (switch & daytime).any(), f"{sym}: contract change during the day"
    n_evening_rolls = int(switch.sum())

    # session filter, on real bars
    sessions = pd.DataFrame({
        "bars": session_date.groupby(session_date).size(),
        "volume": bars.volume.groupby(session_date.values).sum(),
    })
    sessions.index = pd.DatetimeIndex(sessions.index)
    sessions = sessions.sort_index()
    weekday = sessions[sessions.index.dayofweek < 5]
    median_bars = weekday.bars.groupby(weekday.index.year).median()
    sessions["min_bars"] = MIN_SESSION_FRAC * sessions.index.year.map(median_bars).to_numpy(float)

    holidays = CMEEquityIndexCalendar().holidays(
        start=bars.index.min().date(), end=bars.index.max().date())
    sessions["is_complete"] = sessions.bars >= sessions.min_bars
    sessions["is_holiday"] = sessions.index.isin(holidays)
    # The equity index contracts expire at the 09:30 ET open on the third Friday of
    # Mar/Jun/Sep/Dec: the session is truncated on the expiring contract.
    sessions["is_roll"] = (
        (sym in ("ES", "NQ", "YM"))
        & (sessions.index.dayofweek == 4)
        & sessions.index.day.isin(range(15, 22))
        & sessions.index.month.isin([3, 6, 9, 12])
    )
    sessions["keep"] = sessions.is_complete & ~sessions.is_holiday & ~sessions.is_roll
    sessions.insert(0, "symbol", sym)
    sessions.attrs["evening_rolls"] = n_evening_rolls

    keep_dates = set(sessions.index[sessions.keep].date)
    mask = session_date.isin(keep_dates).to_numpy()
    real = bars[mask].assign(session_date=session_date[mask].to_numpy())

    # fill each kept session onto a complete minute grid
    span = real.groupby("session_date").apply(lambda g: (g.index[0], g.index[-1]),
                                              include_groups=False)
    grid = pd.DatetimeIndex(np.concatenate(
        [pd.date_range(a, b, freq="min").to_numpy() for a, b in span]), tz="UTC")
    clean = real.reindex(grid)
    clean.index.name = "timestamp"
    clean["is_filled"] = clean.close.isna()
    clean[["close", "contract", "session_date"]] = clean[["close", "contract", "session_date"]].ffill()
    for c in ("open", "high", "low"):
        clean[c] = clean[c].fillna(clean.close)
    clean["volume"] = clean.volume.fillna(0).astype("uint64")
    clean["contract"] = clean.contract.astype("uint32")
    # The 23:59 bar is a settlement aggregation: prices are fine, volume is a day-boundary
    # total. Keep it in any price path, exclude it from volume features.
    clean["is_settlement_bar"] = clean.index.time == pd.Timestamp("23:59").time()
    clean = clean[clean.index >= MODEL_START]
    return clean, sessions


def report(sym: str, clean: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    modelled = sessions[sessions.index >= MODEL_START.tz_localize(None)]
    drop = modelled[~modelled.keep]
    print(f"{sym}: sessions from {MODEL_START.date()}: {len(modelled):,}  "
          f"keep {modelled.keep.sum():,}  drop {len(drop):,}")
    print(f"  exchange holidays : {drop.is_holiday.sum()}")
    print(f"  quarterly rolls   : {(drop.is_roll & ~drop.is_holiday).sum()}")
    print(f"  short / incomplete: {(~drop.is_holiday & ~drop.is_roll).sum()}")
    print(f"  holiday sessions clearing the bar cut (early closes; dropped by the holiday "
          f"flag): {(sessions.is_holiday & sessions.is_complete).sum()}")
    print(f"  contract changes in the evening session (none in the day): "
          f"{sessions.attrs['evening_rolls']}")
    yr = modelled.groupby(modelled.index.year).agg(
        sessions=("bars", "size"), kept=("keep", "sum"), median_bars=("bars", "median"))
    yr["kept %"] = (100 * yr.kept / yr.sessions).round(1)
    yr["filled %"] = (100 * clean.is_filled.groupby(clean.index.year).mean()).round(1)
    return yr


# %% [markdown]
# ## Run every asset

# %%
# Every asset whose raw file is on this machine: the three book assets plus any candidate in
# EXTRA_ASSETS (e.g. YM) that has been pulled. An asset without its raw file is skipped and
# its existing cleaned file is left as it is.
TO_CLEAN = [s for s in ALL_ASSETS if ALL_ASSETS[s]["raw"].exists()]
SKIPPED = [s for s in ALL_ASSETS if s not in TO_CLEAN]
print("cleaning:", TO_CLEAN, "| no raw file, left as is:", SKIPPED)
if not TO_CLEAN:
    raise SystemExit("No raw files in data/raw/. Pull one first, e.g. "
                     "`uv run python combined/pull.py --pull YM`.")
cleaned, calendars = {}, {}
for sym in TO_CLEAN:
    cleaned[sym], calendars[sym] = clean_asset(sym)
    print(report(sym, cleaned[sym], calendars[sym]).to_string(), "\n")

if "ES" in cleaned and CLEAN_BARS.exists():
    old_es = pd.read_parquet(CLEAN_BARS, columns=["session_date"]).session_date.nunique()
    print(f"ES sessions: {cleaned['ES'].session_date.nunique():,} now vs {old_es:,} in the old "
          f"ES-only file (fixed 1,200-bar cut, no filling)")

# %% [markdown]
# **Findings:** All three assets clear the checks. Sessions kept: ES 3,894, NQ 3,797, ZN
# 3,916, with 92-97% kept every year except NQ 2010 (84%). ZN's 53 contract changes all fall
# at 19:00-20:00 ET, none in the day. Filled minutes: ES 1.4% overall (0-3% per year), NQ 4.0%
# (10-13% before 2016, near 0 from 2018), ZN 8.2% (4-14% every year). The relative cut keeps
# 17 more ES sessions than the old fixed 1,200 (3,894 vs 3,877). It lets early-close holidays
# clear the bar count for NQ (Labor Day 2015) and ZN (Memorial Day 2026), but the holiday flag
# still drops them.

# %% [markdown]
# ## Write

# %%
cal_path = OUT / "session_calendar.parquet"
cal = pd.concat(calendars.values())
if cal_path.exists():   # keep the rows of assets that were not re-cleaned this run
    prev = pd.read_parquet(cal_path)
    cal = pd.concat([prev[~prev.symbol.isin(list(calendars))], cal])
cal.to_parquet(cal_path)
for s, f in cleaned.items():
    dest = Path(CLEAN_BARS_FMT.format(symbol=s))
    f.to_parquet(dest, compression="zstd")
    print(f"wrote {dest.name}: {len(f):,} bars ({f.is_filled.mean():.1%} filled), "
          f"{f.session_date.nunique():,} sessions, {dest.stat().st_size / 1e6:.1f} MB")
