# data/

Market data for the intraday S&P futures / DTW project.

- `raw/` — untouched vendor exports (tick or bar-level S&P futures data). Gitignored; never edit these files in place.
- `processed/` — cleaned, resampled, or feature-engineered data derived from `raw/`. Also gitignored — regenerate from `raw/` via notebook code rather than hand-editing.

Most of this folder is not committed to git (see root `.gitignore`) since intraday futures data is large and often has redistribution restrictions. A few derived files in `processed/` are the exception, listed explicitly in `.gitignore` so the team works from the same inputs:

- `macro_event_calendar_2010_2026.csv` and `macro_event_calendar_expanded_2010_2026.parquet` — small, built by `notebooks/macro_event_calendar.py`.
- `es_1min_clean.parquet` — the cleaned 1-minute ES bars, built by `jack/cleaning.py` from `raw/ES_1m_2010_2026_full.parquet`. **This one is ~60 MB**, so treat it as a deliberate exception rather than a precedent: anything larger belongs in Git LFS or outside the repo. See below for exactly what changed.
- `futures_1min_clean_v3_{ES,NQ,ZN}.parquet` — the cleaned 1-minute bars for the multi-asset book, one file per asset (62-78 MB each, under GitHub's 100 MB limit), built by `combined/cleaning.py`. Missing minutes inside a session are filled with the last traded price and flagged `is_filled`; see `combined/README.md`.

Everything else, including the raw vendor exports, stays untracked.

## Conventions
- Prefer Parquet over CSV for anything beyond a quick look — it's faster and preserves dtypes.
- Name processed files after what they contain and how they were derived, e.g. `es_1min_bars_2024.parquet`, not `data2_final.parquet`.
- If a processed dataset takes more than a few seconds to build, save it here rather than recomputing it in every notebook.

## `es_1min_clean.parquet` — what changed from the raw file

Built by `jack/cleaning.py` from `raw/ES_1m_2010_2026_full.parquet` (ES continuous
front-month, `ES.c.0`, Databento `GLBX.MDP3`). **4,811,336 raw rows in, 4,499,182
out — 312,154 rows removed, 6.5%.**

### Schema

| | raw | clean |
|---|---|---|
| index | `ts_event`, UTC | `timestamp`, UTC (unchanged values) |
| kept | `open`, `high`, `low`, `close` (float64), `volume` (uint64) | same |
| dropped | `rtype`, `publisher_id`, `instrument_id`, `symbol` | — |
| added | — | `session_date` (date), `is_settlement_bar` (bool) |

`symbol` was dropped because it is the constant `ES.c.0` on every row. The two added
columns are conveniences, not corrections.

### Rows removed

Filtering is by **session**, not by bar: a session runs 18:00 ET to 17:00 ET and is
dated by the day it closes on. Of 4,130 sessions, 3,337 are kept and 793 dropped:

| reason | sessions |
|---|---|
| exchange holidays (CME full closures) | 75 |
| quarterly roll Fridays (Mar/Jun/Sep/Dec) | 63 |
| short or incomplete — before 2016 | 600 |
| short or incomplete — 2016 onward (half-days) | 55 |

A session is cut when it holds fewer than 1,200 bars against a full session's ~1,380.
Holidays are detected empirically by that bar count; a hardcoded CME calendar is used
only as a cross-check, and it passes exactly (no complete session lands on a closure
date). Quarterly rolls go because `ES.c.0` switches contracts on the third Friday, so
those sessions are both truncated and carry an artificial price jump.

### What was *not* changed

- **Prices are untouched** — no adjustment, no back-adjustment, no interpolation.
  Every surviving bar's OHLCV matches the raw file exactly.
- **Gaps are not filled.** The index is 1-minute but not contiguous; the daily
  maintenance halt, weekends and dropped sessions all appear as jumps. Do not assume
  a regular frequency.
- **The 23:59 UTC settlement bars are kept**, all 3,322 of them, and flagged via
  `is_settlement_bar`. Their prices are fine but their volume is a day-boundary
  aggregation — the 90th percentile is ~146,000 against ~3,000 for a normal minute,
  and 17 of the 20 largest volume bars in the dataset sit on that timestamp. Exclude
  them from any volume-based calculation; leave them in any price path.

### Coverage caveat — read this before using the early years

**The data is only trustworthy from 2016 onward.** Coverage collapses before then:
93-95% of weekdays have a complete session from 2016, against 80% in 2015, 69-74% in
2013-14 and just 20-37% in 2010-2012. This is not thin overnight trading — whole days
are absent including the US cash session (2010-09-15 has *zero* bars between 14:00
and 15:00 UTC). That is why 600 of the 793 dropped sessions predate 2016. Recovering
those years needs a re-pull from Databento, not a cleaning change.
