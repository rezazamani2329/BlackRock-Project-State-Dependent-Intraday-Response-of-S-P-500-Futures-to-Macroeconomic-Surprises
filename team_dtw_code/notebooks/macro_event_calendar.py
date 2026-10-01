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
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Macro event calendar (2010-2026)
#
# Builds a minute-precision, UTC-timestamped calendar of scheduled macro
# announcements for use as an event-proximity feature in the intraday ES
# DTW pipeline (e.g. "minutes since/until the next CPI print"). Only events
# that are *scheduled in advance* are included, since the feature is meant
# to capture anticipation/positioning ahead of a known release, not surprise
# news.
#
# ## Event types
#
# - **CPI** (Consumer Price Index) — the BLS's monthly inflation print.
#   Released 8:30am ET.
# - **NFP** (Nonfarm Payrolls / Employment Situation) — the BLS's monthly
#   jobs report. Released 8:30am ET.
# - **PCE** (Personal Income and Outlays) — the BEA's monthly report
#   containing the Fed's preferred inflation gauge (core PCE). Released
#   8:30am ET.
# - **FOMC** — the Federal Reserve's scheduled interest-rate decision,
#   8x/year. Release time has changed across eras (see below). Only
#   scheduled meetings are included — unscheduled/emergency Fed actions are
#   deliberately excluded since the goal is events known in advance.

# %% [markdown]
# ## Sources and release-time conventions
#
# - **CPI / NFP / PCE**: pulled live from FRED's release-calendar endpoint
#   (`fred/release/dates`), keyed by each series' `release_id`
#   (CPI=10, NFP=50, PCE=54). This returns *every* scheduled/actual release
#   date FRED has on record for that release, which is what lets us catch
#   the seasonal-adjustment and benchmark-revision duplicate dates flagged
#   below — a single "next release date" field would silently hide those.
#   Always released 8:30am ET.
# - **FOMC**: FRED has no revisable "series" for FOMC decision dates (it's a
#   calendar of meetings, not economic data), so there's no API to pull it
#   from. The scheduled-meeting dates below are hardcoded from the Fed's
#   published historical meeting calendar. FOMC statement release time has
#   moved across three eras — implemented as a function of meeting date,
#   not a single constant:
#   - 2010-01-01 to 2011-03-31: 14:15 ET (pre-press-conference era)
#   - 2011-04-01 to 2013-01-31: 12:30 ET (per the Fed's own historical note)
#   - 2013-02-01 onward: 14:00 ET (per the Fed's official 2013-03-13
#     announcement)
#
# ### Why `zoneinfo` and not a fixed UTC offset
#
# Every one of these releases is scheduled in **wall-clock Eastern Time**,
# and US Eastern Time itself alternates between EST (UTC-5) and EDT (UTC-4)
# across the year. An 8:30am ET release is 13:30 UTC when Eastern is on
# standard time (winter) but 12:30 UTC when Eastern is on daylight time
# (summer) — a full hour of drift depending on the date. A fixed offset
# (e.g. always "UTC-5") would get roughly half the year's events wrong by
# exactly one hour, which is fatal for a minute-precision, event-proximity
# feature. `zoneinfo`'s `America/New_York` calendar knows the actual
# US DST transition rules (including the fact that they've themselves
# changed historically), so localizing a wall-clock ET timestamp and
# converting to UTC handles this correctly for every date in 2010-2026
# without hardcoding transition dates ourselves.

# %%
import os
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()  # reads FRED_API_KEY from a .env file at the repo root

pd.set_option("display.width", 120)

START_DATE = date(2010, 1, 1)
END_DATE = date(2026, 12, 31)

EASTERN = ZoneInfo("America/New_York")
OUTPUT_FILE = Path("../data/processed/macro_event_calendar_2010_2026.csv")

# %% [markdown]
# ## Pull CPI / NFP / PCE release dates from FRED
#
# Note: the installed `fredapi` package (v0.5.2, current PyPI release) does
# not actually expose a `get_release_dates` wrapper method despite that
# being the documented way to do this. So this calls the same underlying
# FRED REST endpoint (`fred/release/dates`) directly with `requests`,
# using `FRED_API_KEY` exactly as `fredapi` itself would.

# %%
FRED_API_KEY = os.getenv("FRED_API_KEY")

if not FRED_API_KEY:
    print(
        "FRED_API_KEY not set.\n"
        "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html, "
        "then add FRED_API_KEY=<your key> to the .env file at the repo root."
    )
    raise RuntimeError("FRED_API_KEY not set")

RELEASE_IDS = {"CPI": 10, "NFP": 50, "PCE": 54}


def fetch_release_dates(release_id: int, start: date, end: date) -> list[date]:
    """
    Fetch every scheduled/actual release date FRED has on record for a given
    "release" (a recurring report identified by `release_id`, e.g. CPI=10),
    restricted to [start, end].

    Uses `include_release_dates_with_no_data=true` because it also returns
    scheduled *future* dates whose data hasn't been published yet (e.g. late
    2026 CPI prints) — omitting it silently drops legitimate future calendar
    dates, which we need since the calendar runs through end of 2026.
    """
    resp = requests.get(
        "https://api.stlouisfed.org/fred/release/dates",
        params={
            "release_id": release_id,
            "api_key": FRED_API_KEY,
            "file_type": "json",
            "sort_order": "asc",
            "limit": 10000,
            "include_release_dates_with_no_data": "true",
        },
        timeout=30,
    )
    resp.raise_for_status()
    all_dates = [
        datetime.strptime(d["date"], "%Y-%m-%d").date()
        for d in resp.json()["release_dates"]
    ]
    return sorted(d for d in all_dates if start <= d <= end)


release_dates_by_type = {}
for event_type, release_id in RELEASE_IDS.items():
    dates = fetch_release_dates(release_id, START_DATE, END_DATE)
    release_dates_by_type[event_type] = dates
    print(f"{event_type} (release_id={release_id}): {len(dates)} release dates fetched")

# %% [markdown]
# ## Hardcoded FOMC scheduled-meeting decision dates
#
# Not available via FRED (no API, since it's not a revisable data series).
# Sourced from the Fed's published historical meeting calendar. Dates from
# 2026-09 onward are tentative/unconfirmed — flagged later, not excluded.

# %%
FOMC_MEETING_DATES = [
    "2010-01-27", "2010-03-16", "2010-04-28", "2010-06-24", "2010-08-10",
    "2010-09-21", "2010-11-03", "2010-12-14",
    "2011-01-26", "2011-03-15", "2011-04-27", "2011-06-22", "2011-08-09",
    "2011-09-21", "2011-11-02", "2011-12-13",
    "2012-01-25", "2012-03-13", "2012-04-25", "2012-06-20", "2012-08-01",
    "2012-09-13", "2012-10-24", "2012-12-12",
    "2013-01-30", "2013-03-20", "2013-05-01", "2013-06-19", "2013-07-31",
    "2013-09-19", "2013-10-30", "2013-12-18",
    "2014-01-29", "2014-03-19", "2014-04-30", "2014-06-18", "2014-07-30",
    "2014-09-17", "2014-10-29", "2014-12-17",
    "2015-01-28", "2015-03-18", "2015-04-29", "2015-06-17", "2015-07-29",
    "2015-09-17", "2015-10-28", "2015-12-16",
    "2016-01-27", "2016-03-16", "2016-04-27", "2016-06-15", "2016-07-27",
    "2016-09-21", "2016-11-02", "2016-12-14",
    "2017-02-01", "2017-03-15", "2017-05-03", "2017-06-14", "2017-07-26",
    "2017-09-20", "2017-11-01", "2017-12-13",
    "2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13", "2018-08-01",
    "2018-09-26", "2018-11-08", "2018-12-19",
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19", "2019-07-31",
    "2019-09-18", "2019-10-30", "2019-12-11",
    "2020-01-29", "2020-04-29", "2020-06-10", "2020-07-29", "2020-09-16",
    "2020-11-05", "2020-12-16",
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16", "2021-07-28",
    "2021-09-22", "2021-11-03", "2021-12-15",
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27",
    "2022-09-21", "2022-11-02", "2022-12-14",
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26",
    "2023-09-20", "2023-11-01", "2023-12-13",
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31",
    "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30",
    "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29",
    "2026-09-16", "2026-10-28", "2026-12-09",  # tentative/unconfirmed
]

fomc_dates = sorted(datetime.strptime(d, "%Y-%m-%d").date() for d in FOMC_MEETING_DATES)
print(f"FOMC: {len(fomc_dates)} scheduled meeting dates hardcoded")

# %% [markdown]
# ## Release-time assignment and ET -> UTC conversion
#
# Two small, separately-testable pieces: (1) what wall-clock ET time does
# this event release at, and (2) turning a (date, ET time) pair into a
# tz-aware UTC timestamp via `zoneinfo` (see the DST explanation above for
# why this can't be a fixed offset).

# %%
def release_time_et(event_type: str, event_date: date) -> time:
    """
    Return the wall-clock America/New_York release time for a macro event.

    CPI, NFP, and PCE are always released 8:30am ET by BLS/BEA convention.
    FOMC statement release time has changed across three eras and is keyed
    on the meeting date:
      - 2010-01-01 to 2011-03-31: 14:15 ET (pre-press-conference era)
      - 2011-04-01 to 2013-01-31: 12:30 ET (per the Fed's own historical note)
      - 2013-02-01 onward:        14:00 ET (per the Fed's official
        2013-03-13 announcement)
    """
    if event_type in ("CPI", "NFP", "PCE"):
        return time(8, 30)
    if event_type == "FOMC":
        if event_date <= date(2011, 3, 31):
            return time(14, 15)
        if event_date <= date(2013, 1, 31):
            return time(12, 30)
        return time(14, 0)
    raise ValueError(f"unknown event_type: {event_type!r}")


def et_to_utc_timestamp(event_date: date, event_time: time) -> pd.Timestamp:
    """
    Combine a date and a wall-clock America/New_York time into a tz-aware
    UTC pandas Timestamp.

    Localizes via `zoneinfo` (not a fixed offset) so DST transitions are
    handled correctly: the same 8:30am ET wall-clock time becomes 13:30 UTC
    in EST (winter) and 12:30 UTC in EDT (summer), and zoneinfo resolves
    which applies for the given date automatically.
    """
    local_dt = datetime.combine(event_date, event_time, tzinfo=EASTERN)
    return pd.Timestamp(local_dt).tz_convert("UTC")

# %% [markdown]
# ## Duplicate/flag detection
#
# None of these get dropped or silently collapsed to one date per month —
# each gets an explicit `flag` describing *why* it's not a single clean
# monthly print, so a future reader (or the DTW feature code) can decide
# how to treat it rather than rediscovering the quirk from scratch.

# %%
FLAG_CPI_MULTIPLE = "multiple_releases_this_month__BLS_seasonal_adjustment_revision"
FLAG_PCE_MULTIPLE = "multiple_releases_this_month__BEA_NIPA_revision"
FLAG_NFP_MULTIPLE = "multiple_releases_this_month__BLS_benchmark_revision"
FLAG_FOMC_TENTATIVE = "TENTATIVE_unconfirmed_schedule"

FOMC_TENTATIVE_CUTOFF = date(2026, 9, 1)

MULTI_RELEASE_FLAG_BY_TYPE = {
    "CPI": FLAG_CPI_MULTIPLE,
    "NFP": FLAG_NFP_MULTIPLE,
    "PCE": FLAG_PCE_MULTIPLE,
}


def flag_multi_release_months(dates: list[date], flag: str) -> dict[date, str]:
    """
    Given a sorted list of release dates for one event type, return a
    {date: flag} map covering every date that shares a (year, month) with
    another release of the same type — e.g. CPI's annual February
    seasonal-adjustment revision, or NFP/PCE's occasional benchmark/NIPA
    revision announcements. Months with exactly one release aren't flagged.
    """
    by_month: dict[tuple[int, int], list[date]] = {}
    for d in dates:
        by_month.setdefault((d.year, d.month), []).append(d)
    return {d: flag for group in by_month.values() if len(group) > 1 for d in group}


flags_by_type: dict[str, dict[date, str]] = {}
for event_type, flag in MULTI_RELEASE_FLAG_BY_TYPE.items():
    month_flags = flag_multi_release_months(release_dates_by_type[event_type], flag)
    flags_by_type[event_type] = month_flags
    print(f"{event_type}: {len(month_flags)} dates flagged ({flag})")

fomc_flags = {d: FLAG_FOMC_TENTATIVE for d in fomc_dates if d >= FOMC_TENTATIVE_CUTOFF}
print(f"FOMC: {len(fomc_flags)} dates flagged ({FLAG_FOMC_TENTATIVE})")

# %% [markdown]
# ## Assemble, sort, export

# %%
rows = []

for event_type in ("CPI", "NFP", "PCE"):
    for d in release_dates_by_type[event_type]:
        t = release_time_et(event_type, d)
        rows.append(
            {
                "event_type": event_type,
                "date_et": d,
                "time_et": t,
                "timestamp_utc": et_to_utc_timestamp(d, t),
                "flag": flags_by_type[event_type].get(d, ""),
            }
        )

for d in fomc_dates:
    t = release_time_et("FOMC", d)
    rows.append(
        {
            "event_type": "FOMC",
            "date_et": d,
            "time_et": t,
            "timestamp_utc": et_to_utc_timestamp(d, t),
            "flag": fomc_flags.get(d, ""),
        }
    )

calendar = pd.DataFrame(rows).sort_values("timestamp_utc").reset_index(drop=True)
calendar.head()

# %%
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
calendar.to_csv(OUTPUT_FILE, index=False)

print(f"wrote {len(calendar):,} rows -> {OUTPUT_FILE}")
print("\ncounts per event_type:")
print(calendar["event_type"].value_counts())
print(f"\nflagged rows: {(calendar['flag'] != '').sum():,} / {len(calendar):,}")

# %% [markdown]
# **Findings:** Assembled 764 events total — CPI 216, NFP 208, PCE 205,
# FOMC 135 — saved to `data/processed/macro_event_calendar_2010_2026.csv`.
# 97 rows carry a flag: 26 CPI seasonal-adjustment-revision duplicates
# (nearly one per February, as expected), 58 PCE NIPA-revision-bundle
# duplicates, 10 NFP benchmark-revision duplicates, and 3 tentative
# 2026 FOMC dates.

# %% [markdown]
# ## Known caveats (read before using this calendar as a feature)
#
# - **CPI** has an extra release nearly every February — the BLS's annual
#   seasonal-adjustment-factor revision, not a second surprise print.
#   Flagged as `multiple_releases_this_month__BLS_seasonal_adjustment_revision`,
#   not dropped.
# - **PCE** occasionally has 2-3 dates in one month when the BEA bundles a
#   comprehensive/annual NIPA revision into the same release window.
#   Flagged as `multiple_releases_this_month__BEA_NIPA_revision`.
# - **NFP** occasionally has an extra date in a month (e.g. 2024-08-21) for
#   BLS's annual benchmark revision announcement, not a new monthly print.
#   Flagged as `multiple_releases_this_month__BLS_benchmark_revision`.
# - **2025-10 NFP and PCE are missing** from this calendar — both were
#   delayed by the government shutdown and folded into later releases. This
#   is real and expected, not a bug in the fetch logic.
# - **FOMC dates from 2026-09 onward** (2026-09-16, 2026-10-28, 2026-12-09)
#   are tentative/unconfirmed per the Fed's published calendar convention,
#   and are flagged `TENTATIVE_unconfirmed_schedule` rather than excluded.
# - **Unscheduled/emergency Fed actions are deliberately excluded** — this
#   calendar only covers events that were knowable in advance, since that's
#   what an event-proximity feature needs.

# %% [markdown]
# ## Considering other macro events from FRED

# %%
calendar = pd.read_csv(OUTPUT_FILE)

# %%
calendar['event_type'].unique()

# %% [markdown]
# ## Expand the calendar with additional macro releases
#
# Starting from the local base calendar (CPI/NFP/PCE/FOMC), pull additional
# release schedules from FRED using candidate release IDs, apply event-specific
# intraday release times (ET), and save an expanded calendar as parquet.

# %%
EXPANDED_OUTPUT_FILE = Path("../data/processed/macro_event_calendar_expanded_2010_2026.parquet")

ADDITIONAL_RELEASE_CONFIG = {
    # event_type: (release_id, release_time_et)
    "PPI": (46, time(8, 30)),                        # Producer Price Index
    "Retail Sales": (9, time(8, 30)),                # Advance Monthly Sales
    "Initial Jobless Claims": (180, time(8, 30)),    # Weekly claims
    "GDP": (53, time(8, 30)),                        # GDP release schedule
    "JOLTS": (192, time(10, 0)),                     # Job Openings and Labor Turnover Survey
    "Employment Cost Index": (11, time(8, 30)),      # Quarterly
    "Industrial Production": (13, time(9, 15)),      # G.17 release convention
    "Import/Export Prices": (188, time(8, 30)),      # U.S. Import and Export Price Indexes
    "Wholesale Trade": (290, time(10, 0)),           # Monthly Wholesale Trade
}


def build_additional_release_calendar(
    start: date,
    end: date,
    config: dict[str, tuple[int, time]],
) -> pd.DataFrame:
    rows = []
    for event_type, (release_id, release_time) in config.items():
        release_dates = fetch_release_dates(release_id, start, end)
        print(f"{event_type} (release_id={release_id}): {len(release_dates)} release dates fetched")
        for d in release_dates:
            rows.append(
                {
                    "event_type": event_type,
                    "date_et": d,
                    "time_et": release_time,
                    "timestamp_utc": et_to_utc_timestamp(d, release_time),
                    "flag": "",
                    "source": "fred_release_dates",
                    "release_id": release_id,
                }
            )

    additional = pd.DataFrame(rows)
    if additional.empty:
        return additional

    # Flag months with multiple releases for the same event type (often revisions/special updates).
    for event_type in additional["event_type"].unique():
        mask = additional["event_type"].eq(event_type)
        dates = sorted(additional.loc[mask, "date_et"].tolist())
        duplicated_month_flags = flag_multi_release_months(
            dates,
            "multiple_releases_this_month__possible_revision_or_special_release",
        )
        if duplicated_month_flags:
            additional.loc[mask, "flag"] = additional.loc[mask, "date_et"].map(duplicated_month_flags).fillna("")

    return additional.sort_values("timestamp_utc").reset_index(drop=True)


# %%
base_calendar = pd.read_csv(OUTPUT_FILE)
base_calendar["timestamp_utc"] = pd.to_datetime(base_calendar["timestamp_utc"], utc=True)
base_calendar["date_et"] = pd.to_datetime(base_calendar["date_et"]).dt.date
base_calendar["time_et"] = pd.to_datetime(base_calendar["time_et"], format="%H:%M:%S", errors="coerce").dt.time
base_calendar["source"] = "base_calendar"
base_calendar["release_id"] = pd.NA

additional_calendar = build_additional_release_calendar(
    START_DATE,
    END_DATE,
    ADDITIONAL_RELEASE_CONFIG,
)

expanded_calendar = pd.concat([base_calendar, additional_calendar], ignore_index=True)
expanded_calendar = expanded_calendar.drop_duplicates(
    subset=["event_type", "timestamp_utc"],
    keep="first",
).sort_values("timestamp_utc").reset_index(drop=True)

expanded_calendar.head(20)

# %%
expanded_calendar['event_type'].unique()

# %%
expanded_calendar['flag'].unique()

# %% [markdown]
# ### Save expanded calendar

# %%
EXPANDED_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
expanded_calendar.to_parquet(EXPANDED_OUTPUT_FILE, index=False)

print(f"wrote {len(expanded_calendar):,} rows -> {EXPANDED_OUTPUT_FILE}")
print("\ncounts per event_type:")
print(expanded_calendar["event_type"].value_counts())
print("\ncounts by source:")
print(expanded_calendar["source"].value_counts())

# %% [markdown]
# **Findings:** The expanded calendar appends nine additional macro event
# categories to the original CPI/NFP/PCE/FOMC base set, normalizes all event
# timestamps to UTC, and saves a merged parquet calendar for downstream
# event-window and DTW analyses.
