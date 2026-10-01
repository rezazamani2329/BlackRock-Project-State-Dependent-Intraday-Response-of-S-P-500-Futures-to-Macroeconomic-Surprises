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
# # Cleaning — sessions, holidays, and 30-minute bars
#
# Turns the shared 1-minute ES bars into the 30-minute modelling grid used by
# `feature_engineering.py`. Three jobs:
#
# 1. Define trading **sessions** and measure how complete each one is.
# 2. Drop **holidays, half-days and incomplete sessions** — we trade 24/7 but
#    skip days the exchange is closed or the data doesn't cover.
# 3. Resample to **30-minute bars** and attach the forward 30-minute return target.
#
# Reads the raw vendor export from `../data/raw/` and writes the cleaned 1-minute
# bars back to the shared `../data/processed/es_1min_clean.parquet`, so the rest of
# the team works from the same cleaned series. The session calendar and the
# 30-minute modelling grid stay in `jack/data/`, since both bake in choices specific to this
# strategy (see `jack/README.md`).

# %%
import numpy as np
import pandas as pd
from pathlib import Path
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMemorialDay,
    USThanksgivingDay,
    nearest_workday,
    sunday_to_monday,
)

SHARED_RAW = Path("../data/raw")
SHARED_PROCESSED = Path("../data/processed")
JACK_DATA = Path("data")
JACK_DATA.mkdir(exist_ok=True)

RAW_FILE = SHARED_RAW / "ES_1m_2010_2026_full.parquet"
CLEAN_FILE = SHARED_PROCESSED / "es_1min_clean.parquet"
# zstd over the default snappy: this file is committed, so the ~30% saving matters.
PARQUET_COMPRESSION = "zstd"

# A session runs from the 18:00 ET open to the next day's 17:00 ET close. Shifting
# ET forward 6h rolls the evening open onto the following calendar date, so a
# session gets the date of the day it *closes* on. Same convention as
# notebooks/data_exploration.py — keep the two in sync.
SESSION_SHIFT = pd.Timedelta(hours=6)

# A complete session is ~1380 minutes (24h minus the 1h daily maintenance halt).
# Anything materially short is a holiday, a half-day, or missing data.
MIN_SESSION_BARS = 1200

# The trading grid. We decide every 30 minutes, so that is the bar size.
BAR_FREQ = "30min"
BAR_MINUTES = 30
# Minimum 1-min bars required inside a 30-minute bar for it to be usable.
MIN_MINUTES_PER_BAR = 23

# History before this date has large gaps — see the coverage section below.
MODEL_START = pd.Timestamp("2016-01-01", tz="UTC")

pd.set_option("display.width", 140)

# %% [markdown]
# ## Load the raw 1-minute bars
#
# This reads **`data/raw/`** directly rather than the intermediate
# `es_1min_bars_2010_2026.parquet`, so the whole path from vendor export to cleaned
# output lives in one notebook. The raw Databento file carries four columns we do not
# need — `rtype`, `publisher_id`, `instrument_id` and `symbol`, the last of which is
# the constant `ES.c.0` — so the first step is to keep the OHLCV columns, name the
# index, and sort.

# %%
raw = pd.read_parquet(RAW_FILE)
print(f"raw: {len(raw):,} rows, columns {list(raw.columns)}")

bars = raw[["open", "high", "low", "close", "volume"]].copy()
bars.index.name = "timestamp"
bars = bars.sort_index()

assert not bars.index.has_duplicates, "duplicate timestamps in the raw file"
assert str(bars.index.tz) == "UTC", f"expected a UTC index, got {bars.index.tz}"
print(f"{len(bars):,} bars   {bars.index.min()} -> {bars.index.max()}")
bars.head()

# %% [markdown]
# ## Data coverage — how much of each year is actually there
#
# Before filtering holidays we need to know whether the data covers the period at
# all. Counting bars per (year, UTC hour) shows this immediately: a fully covered
# hour-of-day should have roughly `trading_days x 60` bars per year (~15,100).

# %%
coverage = pd.crosstab(bars.index.year, bars.index.hour)
print("bars per (year, UTC hour), thousands:")
print((coverage / 1000).round(1).to_string())

# %% [markdown]
# Now the same question per session, counting how many weekdays in each year have
# a complete session behind them.

# %%
session_date = pd.Series((bars.index.tz_convert("America/New_York") + SESSION_SHIFT).date, index=bars.index)

sessions = bars.groupby(session_date).agg(
    bars=("close", "size"),
    volume=("volume", "sum"),
    first_ts=("close", lambda s: s.index.min()),
    last_ts=("close", lambda s: s.index.max()),
)
sessions.index = pd.to_datetime(sessions.index)
sessions.index.name = "session_date"

weekdays = pd.date_range(bars.index.min().date(), bars.index.max().date(), freq="B")
expected = pd.Series(1, index=weekdays).groupby(weekdays.year).size()

complete = sessions[sessions.bars >= MIN_SESSION_BARS]
partial = sessions[sessions.bars < MIN_SESSION_BARS]
by_year = pd.DataFrame(
    {
        "expected_weekdays": expected,
        "complete_sessions": complete.groupby(complete.index.year).size(),
        "partial_sessions": partial.groupby(partial.index.year).size(),
    }
).fillna(0).astype(int)
by_year["pct_complete"] = (100 * by_year.complete_sessions / by_year.expected_weekdays).round(0)
print(by_year.to_string())

# %% [markdown]
# **Findings:** The data is only trustworthy from 2016 onward, and this is the
# single most consequential fact in the pipeline. From 2016 on, 93-95% of weekdays
# have a complete session and every UTC hour carries ~15,400 bars/year (a full
# ~15,100 expected). Before that, coverage collapses: 80% of weekdays complete in
# 2015, 69-74% in 2013-14, and just 20-37% in 2010-2012. It is not thin overnight
# trading — whole days are absent, including the US cash session (2010-09-15 has
# *zero* bars between 14:00 and 15:00 UTC). So `MODEL_START` is 2016-01-01, giving
# ~10.5 usable years. That is enough for a 5-year train plus five annual
# walk-forward folds, but it is half the history the raw file appears to offer.
# Recovering 2010-2015 would need a re-pull, not a cleaning fix.

# %% [markdown]
# ## Session bar-count distribution
#
# Where to put the completeness cut. A full session is ~1380 bars; half-days
# (13:00 ET close) come in around ~1080.

# %%
print(
    pd.cut(sessions.bars, [0, 2, 200, 700, 1000, 1200, 1300, 1381])
    .value_counts()
    .sort_index()
    .to_string()
)

# %% [markdown]
# ## Exchange holidays
#
# The completeness cut above is the *filter*; the holiday calendar is a
# **cross-check** on it. If the cut is doing its job, the sessions it removes
# should be the days CME equity index futures are fully closed, plus the
# half-days around them — not random weekdays with missing data.
#
# CME closes ES fully on New Year's Day, Good Friday, Memorial Day, Juneteenth
# (2022+), Independence Day, Labor Day, Thanksgiving and Christmas. It stays open
# with a shortened session on MLK Day and Presidents' Day, so those are *not*
# closures. Built from `pandas.tseries.holiday` so it needs no new dependency.
#
# One observance subtlety, and the cross-check below is what caught it: a holiday
# landing on a Saturday is observed on the preceding Friday — *except* New Year's
# Day, which is not. ES traded a full session on Friday 2021-12-31 even though
# 2022-01-01 was a Saturday, so New Year's uses `sunday_to_monday` while
# Independence Day and Christmas use `nearest_workday`.


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


holidays = CMEEquityIndexCalendar().holidays(
    start=bars.index.min().date(), end=bars.index.max().date()
)
print(f"{len(holidays)} full-closure holidays over the data span")

# %% [markdown]
# ## Classify and filter sessions

# %%
sessions["is_holiday"] = sessions.index.isin(holidays)
sessions["is_complete"] = sessions.bars >= MIN_SESSION_BARS
# Short but not a full closure: half-days (Thanksgiving Friday, Christmas Eve,
# July 3) and, in the early years, sessions the data simply doesn't cover.
sessions["is_short"] = ~sessions.is_complete & ~sessions.is_holiday
sessions["keep"] = sessions.is_complete & ~sessions.is_holiday

modelled = sessions[sessions.index >= MODEL_START.tz_localize(None)]
dropped = modelled[~modelled.keep]
print(f"sessions from {MODEL_START.date()}: {len(modelled)}  keep {modelled.keep.sum()}  drop {len(dropped)}")
print(f"  dropped that are known holidays: {dropped.is_holiday.sum()}")
print(f"  dropped that are short/incomplete: {(~dropped.is_holiday).sum()}")
print("\ndropped sessions per year:")
print(dropped.groupby(dropped.index.year).size().to_string())
# The non-holiday drops are not a random scatter of bad days. Most are quarterly
# roll Fridays: ES.c.0 is a front-month continuous series, so on the third Friday
# of Mar/Jun/Sep/Dec the expiring contract stops trading at 09:30 ET and the
# series switches contracts. The session is genuinely truncated and carries an
# artificial roll jump, so dropping it is the right call either way.
short_drops = dropped[~dropped.is_holiday].copy()
is_third_friday = (short_drops.index.dayofweek == 4) & (short_drops.index.day.isin(range(15, 22)))
short_drops["kind"] = np.where(
    is_third_friday & short_drops.index.month.isin([3, 6, 9, 12]),
    "quarterly_roll",
    np.where(short_drops.bars >= 400, "half_day", "stub_or_boundary"),
)
print("\nnon-holiday drops by kind:")
print(short_drops.kind.value_counts().to_string())
print("\nshortest dropped non-holiday sessions:")
print(short_drops.nsmallest(12, "bars")[["bars", "volume", "kind"]].to_string())

# %% [markdown]
# ### Cross-check: does a complete session ever land on a holiday?

# %%
holiday_but_complete = sessions[sessions.is_holiday & sessions.is_complete]
print(f"holidays with a full session (should be 0 or near it): {len(holiday_but_complete)}")
if len(holiday_but_complete):
    print(holiday_but_complete[["bars", "volume"]].to_string())

# %% [markdown]
# **Findings:** From 2016 the filter keeps 2,567 of 2,712 sessions and drops 145
# (~14/year), and the dropped set is fully explained rather than residual: 49 are
# exchange holidays, 53 are half-days, 41 are quarterly roll Fridays, and 2 are
# stubs at the data boundary. The holiday cross-check passes exactly — zero
# complete sessions land on a closure date. It also earned its keep: the first run
# flagged 2021-12-31 as a holiday with a full 1,379-bar session, which exposed a
# real calendar bug. A Saturday holiday is observed on the preceding Friday for
# Independence Day and Christmas but *not* for New Year's, so ES traded normally
# on 2021-12-31. The quarterly rolls are worth noting for later: on those days
# `ES.c.0` switches contracts, so they carry an artificial price jump on top of
# being truncated.

# %% [markdown]
# ## The 23:59 UTC volume artifact
#
# `data_exploration.py` flagged that most of the largest volume prints land exactly
# at 23:59 UTC — a day-boundary settlement aggregation rather than a real minute of
# trading. The price on those bars is fine; the volume is not. We keep the bar (it
# would otherwise punch a hole in the price path that DTW reads) and flag it so any
# volume-based feature can exclude it.

# %%
at_2359 = bars.index.time == pd.Timestamp("23:59").time()
print(f"23:59 bars: {at_2359.sum():,}")
print(f"  median volume at 23:59: {bars.volume[at_2359].median():,.0f}")
print(f"  median volume elsewhere: {bars.volume[~at_2359].median():,.0f}")

# %% [markdown]
# **Findings:** The artifact is real but confined to a tail, not a level shift. The
# *median* 23:59 bar trades 141 contracts, actually below the 231 of a typical
# minute — but the 90th percentile is 146,419 against 2,996 elsewhere, a ~50x jump,
# and 17 of the 20 largest volume bars in the entire dataset sit at 23:59. So
# roughly a tenth of these bars carry a settlement aggregation worth thousands of
# normal minutes. Prices are unaffected, so the bars stay in the path DTW reads and
# are flagged via `is_settlement_bar` for any volume feature to exclude.

# %% [markdown]
# ## Write the cleaned 1-minute bars

# %%
keep_dates = set(sessions.index[sessions.keep].date)
mask = pd.Series(session_date.values, index=bars.index).isin(keep_dates).values

clean = bars[mask].copy()
clean["session_date"] = pd.Series(session_date.values, index=bars.index)[mask].values
clean["is_settlement_bar"] = clean.index.time == pd.Timestamp("23:59").time()

# The cleaned 1-minute bars go to the shared `data/processed/` so the rest of the
# team can use them; the session calendar stays local since it is a by-product of the
# filtering choices above.
clean.to_parquet(CLEAN_FILE, compression=PARQUET_COMPRESSION)
sessions.to_parquet(JACK_DATA / "session_calendar.parquet")
print(f"wrote {CLEAN_FILE}   {len(clean):,} bars, {clean.session_date.nunique():,} sessions"
      f"   ({CLEAN_FILE.stat().st_size / 1e6:.1f} MB, {PARQUET_COMPRESSION})")
print(f"wrote session_calendar.parquet {len(sessions):,} sessions")

# %% [markdown]
# ## 30-minute bars and the forward-return target
#
# Resample to 30-minute boundaries. Pandas labels each bin by its **left edge**, so
# the bar labelled 10:00 spans `[10:00, 10:30)` and its close is the 10:29 print.
# That is the decision point: features at label *t* may use data through that bar's
# close, and the target is the move into the next bar — the next 30 minutes.

# %%
bars30 = clean.resample(BAR_FREQ).agg(
    open=("open", "first"),
    high=("high", "max"),
    low=("low", "min"),
    close=("close", "last"),
    volume=("volume", "sum"),
    n_minutes=("close", "size"),
)
bars30 = bars30[bars30.n_minutes >= MIN_MINUTES_PER_BAR]
print(f"30-minute bars after the >={MIN_MINUTES_PER_BAR}-minute filter: {len(bars30):,}")
print("\nbars per UTC hour (the thin hour is the daily maintenance halt, which moves with DST):")
print(bars30.groupby(bars30.index.hour).size().to_string())

# %% [markdown]
# The target is only defined when the next bar is genuinely the next 30 minutes.
# Across the maintenance halt, a weekend, or a dropped holiday the gap is larger,
# and a naive `shift(-1)` would silently produce a multi-day return labelled as a
# 30-minute one. Those are set to NaN.

# %%
next_close = bars30.close.shift(-1)
gap = bars30.index.to_series().shift(-1) - bars30.index.to_series()
contiguous = gap == pd.Timedelta(minutes=BAR_MINUTES)

bars30["fwd_ret_30m"] = np.where(contiguous, np.log(next_close / bars30.close), np.nan)
bars30["y"] = np.sign(bars30.fwd_ret_30m)
bars30["session_date"] = (bars30.index.tz_convert("America/New_York") + SESSION_SHIFT).date

bars30 = bars30[bars30.index >= MODEL_START]
bars_per_year = 2 * 23 * 252
print(f"30-minute bars from {MODEL_START.date()}: {len(bars30):,}")
print(f"  with a defined target: {bars30.fwd_ret_30m.notna().sum():,} "
      f"({100 * bars30.fwd_ret_30m.notna().mean():.1f}%)")
print(f"\ntarget balance: up {(bars30.y > 0).sum():,}  down {(bars30.y < 0).sum():,}  "
      f"flat {(bars30.y == 0).sum():,}")
print(f"fwd_ret_30m std: {bars30.fwd_ret_30m.std():.5f}   "
      f"annualised: {bars30.fwd_ret_30m.std() * np.sqrt(bars_per_year):.1%}")

# %% [markdown]
# ### Sanity check: 30-minute closes match the underlying 1-minute bars

# %%
probe = bars30.sample(5, random_state=0).index
for ts in sorted(probe):
    minute_close = clean.close.loc[ts:ts + pd.Timedelta(minutes=BAR_MINUTES - 1)].iloc[-1]
    assert np.isclose(bars30.close.loc[ts], minute_close), ts
print("30-minute closes reconcile against the 1-minute source for all probes")

# %%
bars30.to_parquet(JACK_DATA / "es_30min_bars.parquet")
print(f"wrote es_30min_bars.parquet  {len(bars30):,} rows  {bars30.index.min()} -> {bars30.index.max()}")
bars30.head()

# %% [markdown]
# **Findings:** 116,605 thirty-minute bars from 2016-01-03 to 2026-06-30, of which
# 96.6% have a defined forward return — the missing 3.4% are the bars before the
# daily halt, the weekend, and each dropped session, exactly where a `shift(-1)`
# would otherwise have manufactured a fake 30-minute return out of a multi-day gap.
#
# Halving the bar size roughly doubles the sample, from 58,974 hourly bars to
# 116,605. Realised volatility annualises to 17.3% against 17.4% on the hourly grid,
# which is the check that matters: both resamplings describe the same price process,
# so the finer grid adds observations rather than distorting the series.
#
# Two things carry into the modelling. The target is **not balanced** — 55,704 up
# against 51,667 down, so 51.9% of bars rise, and always-long remains the baseline to
# beat. And 5,240 bars (4.5%) close exactly unchanged, up from 3.3% on the hourly
# grid, since a shorter horizon gives price less room to move. For a
# continuous-target regression that is harmless; it only mattered when the target was
# a binary label.
