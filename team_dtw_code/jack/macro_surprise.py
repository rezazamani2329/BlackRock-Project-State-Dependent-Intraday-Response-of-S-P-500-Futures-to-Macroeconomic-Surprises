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
# # Macro surprises and post-announcement drift
#
# Every directional measurement in this project so far has come from price history, and
# none of it has worked. This notebook asks a different question: when a scheduled
# release prints a number the market did not expect, does ES move in a predictable
# direction *after* the initial jump?
#
# The input that makes this possible is Bloomberg consensus — `SURVEY_MEDIAN` against
# `ACTUAL` for every US release from 2010 to 2026. `is_event_window` in
# `feature_engineering.py` is a binary flag: it knows CPI is printing but not whether the
# print was hot or cold. A signed surprise is the difference between a magnitude feature
# and a directional one.
#
# **The distinction that decides everything is entry timing.** A regression of the
# release-window return on the surprise measures the *reaction*, which is not tradeable:
# by the time the number is public the jump has happened. The tradeable quantity is the
# drift that follows, so every return here starts at `T+1` at the earliest, and the
# decomposition between jump and drift is reported explicitly.
#
# | section | question |
# |---|---|
# | surprises | build causal z-scored surprises from Bloomberg consensus |
# | entry lag | how much of the CPI response is the untradeable jump? |
# | the sweep | run the drift test on all 99 releases, split 2020 |
# | the null | how many of those survive by chance? |
# | jobless claims | why the strongest raw t-statistic is an artefact |
# | the signal | the one release that survives, sized with costs |

# %%
from pathlib import Path

import numpy as np
import pandas as pd

SHARED_PROCESSED = Path("../data/processed")
SHARED_RAW = Path("../data/raw")
JACK_DATA = Path("data")

BLOOMBERG_FILE = SHARED_RAW / "Bloomberg Economic Releases.xlsx"

# A release is public the instant it prints, but a fill is not. Every return below starts
# at T+1 minute at the earliest; T+0 is reported only to size the leak it represents.
ENTRY_LAG_MIN = 1
DRIFT_END_MIN = 30
BAR_TOLERANCE_NS = 5 * 60 * 10**9        # nearest-bar tolerance for small data gaps
WINSOR_SIGMA = 5.0                       # COVID claims reach |z| = 108 without this
MIN_EVENTS = 40
SPLIT_YEAR = 2020                        # discovery <= 2020, confirmation 2021+
TICK_BPS = 0.52                          # one full ES tick at a ~4800 index level

# Chosen before looking at any result: the releases that move equity index futures, from
# the event study in `feature_engineering.py` and the standard macro-announcement
# literature. The remaining ~83 releases are still run, as the null the shortlist is
# judged against.
SHORTLIST = [
    "CPI MoM", "Core CPI MoM", "PPI Final Demand MoM", "PPI Ex Food and Energy MoM",
    "Change in Nonfarm Payrolls", "Unemployment Rate", "Average Hourly Earnings MoM",
    "Retail Sales Advance MoM", "Retail Sales Control Group", "PCE Price Index MoM",
    "Core PCE Price Index MoM", "ISM Manufacturing", "ISM Services Index",
    "ISM Prices Paid", "Initial Jobless Claims", "GDP Annualized QoQ",
]

print(f"Bloomberg workbook present: {BLOOMBERG_FILE.exists()}")

# %% [markdown]
# ## Surprises
#
# The workbook holds one sheet per year, 2010 to 2026, with a Bloomberg column header
# that has to be stripped back to the field name. A row is usable only when both the
# consensus and the actual are present.
#
# The surprise is `ACTUAL - SURVEY_MEDIAN`, which is in the units of each release and so
# cannot be compared across them. Standardising uses an **expanding** standard deviation
# of prior surprises only, shifted one release, so the scale at any date depends only on
# releases that had already happened. Winsorising at five sigma is not cosmetic: the
# March 2020 jobless claims surprise is 108 sigma and would otherwise decide any
# regression it appears in on its own.

# %%
sheets = pd.ExcelFile(BLOOMBERG_FILE)
frames = []
for sheet in sheets.sheet_names:
    part = pd.read_excel(BLOOMBERG_FILE, sheet_name=sheet)
    part.columns = [c.split(".")[-1] if "DROPNA" in str(c) else c for c in part.columns]
    frames.append(part)

bloom = pd.concat(frames, ignore_index=True)
bloom["RELEASE_DATE"] = pd.to_datetime(bloom.RELEASE_DATE, errors="coerce")
bloom["RELEASE_TIME"] = bloom.RELEASE_TIME.astype(str)
bloom = bloom.dropna(subset=["RELEASE_DATE"])

has_both = bloom.SURVEY_MEDIAN.notna() & bloom.ACTUAL.notna()
print(f"rows in workbook: {len(bloom):,}")
print(f"  with both consensus and actual: {has_both.sum():,} ({has_both.mean():.0%})")
bloom = bloom[has_both]
print(f"  date range: {bloom.RELEASE_DATE.min().date()} -> {bloom.RELEASE_DATE.max().date()}")


def surprise_series(event_name, release_time):
    """Causally z-scored, winsorised surprise for one (release, clock time) pair."""
    g = bloom[(bloom.EVENT_NAME == event_name)
              & (bloom.RELEASE_TIME == release_time)].sort_values("RELEASE_DATE").copy()
    g["surprise"] = g.ACTUAL - g.SURVEY_MEDIAN
    # expanding().std().shift(1): the scale uses prior releases only, never this one
    scale = g.surprise.expanding().std().shift(1)
    g["z"] = (g.surprise / scale).clip(-WINSOR_SIGMA, WINSOR_SIGMA)
    g = g[np.isfinite(g.z) & (scale > 0)]
    stamp = pd.to_datetime(g.RELEASE_DATE.dt.strftime("%Y-%m-%d") + " " + release_time)
    g["ts"] = (stamp.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
               .dt.tz_convert("UTC"))
    return g.dropna(subset=["ts"])


candidates = (bloom[bloom.RELEASE_DATE >= "2015-01-01"]
              .groupby(["EVENT_NAME", "RELEASE_TIME"]).size()
              .reset_index(name="n").query("n >= 60"))
print(f"\nreleases with at least 60 observations since 2015: {len(candidates)}")
print(f"  of which on the pre-specified shortlist: "
      f"{candidates.EVENT_NAME.isin(SHORTLIST).sum()}")

# %% [markdown]
# **Findings:** The workbook carries 28,159 rows, of which 20,941 (74%) have both a
# consensus and an actual, spanning 2010-01-04 to 2026-07-17. Ninety-nine distinct
# (release, clock time) pairs have at least 60 observations since 2015, so the shortlist
# of 16 is being drawn from a much larger pool — which is exactly why the remaining 83
# are kept as a null rather than discarded.

# %% [markdown]
# ## Event returns
#
# Prices come from the shared cleaned 1-minute series. The lookup takes the nearest bar
# at or after the requested minute, within a five-minute tolerance, so a release that
# lands in a small data gap is dropped rather than silently matched to a distant print.

# %%
minute = pd.read_parquet(SHARED_PROCESSED / "es_1min_clean.parquet",
                         columns=["open"]).sort_index()
minute = minute[~minute.index.duplicated(keep="first")]
px_values = minute.open.to_numpy()
px_stamps = minute.index.asi8
print(f"1-minute bars: {len(minute):,}   "
      f"{minute.index.min().date()} -> {minute.index.max().date()}")


def price_at(stamps_ns):
    """Nearest open at or after each timestamp, NaN beyond BAR_TOLERANCE_NS."""
    pos = np.searchsorted(px_stamps, stamps_ns)
    clipped = np.clip(pos, 0, len(px_stamps) - 1)
    out = np.full(len(stamps_ns), np.nan)
    ok = (pos < len(px_stamps)) & (np.abs(px_stamps[clipped] - stamps_ns) <= BAR_TOLERANCE_NS)
    out[ok] = px_values[clipped[ok]]
    return out


def event_prices(events, offsets_min):
    """Prices at T + each offset for one release's event table."""
    base = events.ts.values.astype("datetime64[ns]").astype("int64")
    return {k: price_at(base + k * 60 * 10**9) for k in offsets_min}


def ols_t(x, y):
    """Slope, t-statistic and R-squared of a univariate OLS, on finite pairs only."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(y) < 3 or np.std(x) == 0:
        return np.nan, np.nan, np.nan
    design = np.column_stack([np.ones(len(y)), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    se = np.sqrt((resid @ resid) / (len(y) - 2) * np.linalg.inv(design.T @ design)[1, 1])
    return beta[1], beta[1] / se, 1 - resid.var() / y.var()


# %% [markdown]
# ## How much of the response can actually be traded?
#
# The headline event-study result — regress the return spanning the release on the
# surprise — is not a forecast. It measures how far the market moves once the number is
# known, and every basis point of it is earned in the instant the number prints.
#
# Sweeping the entry minute while holding the exit at `T+30` separates the two. The
# regression below is run on CPI, the release with the largest published response.

# %%
cpi = surprise_series("CPI MoM", "08:30:00")
cpi_px = event_prices(cpi, [0, 1, 2, 3, 5, 10, 30])
usable = np.all([np.isfinite(v) for v in cpi_px.values()], axis=0)
cpi_z = cpi.z.to_numpy()[usable]
print(f"CPI releases with clean ES prices: {usable.sum()}\n")

print("ES move regressed on the causal CPI surprise, exit fixed at T+30.")
print("beta is basis points per one-sigma hot print; negative means hot CPI -> ES lower.\n")
print(f"{'entry':>7}{'beta bps':>11}{'t':>8}{'R2':>8}")
for lag in [0, 1, 2, 3, 5, 10]:
    ret = 1e4 * np.log(cpi_px[30][usable] / cpi_px[lag][usable])
    beta, tstat, r2 = ols_t(cpi_z, ret)
    flag = "   <- not tradeable" if lag == 0 else ""
    print(f"{'T+' + str(lag):>7}{beta:>11.2f}{tstat:>8.2f}{r2:>8.3f}{flag}")

print("\ndecomposition of the one-sigma response:")
for label, lo, hi in [("jump   T+0 -> T+1", 0, 1),
                      ("drift  T+1 -> T+30", 1, 30),
                      ("total  T+0 -> T+30", 0, 30)]:
    ret = 1e4 * np.log(cpi_px[hi][usable] / cpi_px[lo][usable])
    beta, tstat, _ = ols_t(cpi_z, ret)
    print(f"  {label:<20} beta {beta:+7.2f} bps   t {tstat:+6.2f}")

# %% [markdown]
# **Findings:** Entering at `T+0` gives a beta of −28.99 bps per sigma at t = −6.50 and
# an R-squared of 0.226. Delaying entry by a single minute drops that to −5.64 bps at
# t = −2.68. **Eighty-one percent of the CPI response is the jump between T+0 and T+1**,
# which is −23.35 bps at t = −7.34 and cannot be captured, because entering at T+0 means
# transacting at the pre-release price with post-release information.
#
# The remainder is not zero. The drift from T+1 to T+30 is −5.64 bps at t = −2.68, and it
# decays slowly rather than vanishing: −4.10 at T+2, −3.42 at T+5 and −3.31 even entering
# at T+10. That residual is the only part of this worth testing further, and everything
# below uses `ENTRY_LAG_MIN = 1`.

# %% [markdown]
# ## The sweep
#
# The drift test is run on every release with at least 60 observations, over two windows:
#
# - **`T+1` to `T+30`** — the drift itself, tradeable as an event-time overlay.
# - **`T+30` to `T+60`** — the next clean 30-minute bar, which is the only window that
#   could enter `MODEL_FEATURES` on the existing grid.
#
# Each is fitted on the full sample and then split at 2020, so a release has to keep both
# its sign and something like its magnitude across a regime break to be taken seriously.

# %%
records = []
for _, row in candidates.iterrows():
    events = surprise_series(row.EVENT_NAME, row.RELEASE_TIME)
    if len(events) < MIN_EVENTS:
        continue
    prices = event_prices(events, [1, 30, 60])
    ok = np.all([np.isfinite(v) for v in prices.values()], axis=0)
    if ok.sum() < MIN_EVENTS:
        continue
    z = events.z.to_numpy()[ok]
    year = events.ts.dt.year.to_numpy()[ok]
    rec = {"event": row.EVENT_NAME, "time": row.RELEASE_TIME, "n": int(ok.sum()),
           "shortlist": row.EVENT_NAME in SHORTLIST}
    for label, lo, hi in [("drift", 1, 30), ("nextbar", 30, 60)]:
        ret = 1e4 * np.log(prices[hi][ok] / prices[lo][ok])
        rec[f"beta_{label}"], rec[f"t_{label}"], _ = ols_t(z, ret)
        train, test = year <= SPLIT_YEAR, year > SPLIT_YEAR
        if train.sum() >= 25 and test.sum() >= 25:
            b_tr, _, _ = ols_t(z[train], ret[train])
            b_te, t_te, _ = ols_t(z[test], ret[test])
            rec[f"train_{label}"], rec[f"test_{label}"] = b_tr, b_te
            rec[f"ttest_{label}"] = t_te
            rec[f"agree_{label}"] = int(np.sign(b_tr) == np.sign(b_te))
    records.append(rec)

sweep = pd.DataFrame(records)
print(f"releases tested: {len(sweep)}   on the shortlist: {sweep.shortlist.sum()}\n")

short = sweep[sweep.shortlist].sort_values("t_drift", key=abs, ascending=False)
print("WINDOW A — the drift, T+1 -> T+30")
print(short[["event", "n", "beta_drift", "t_drift", "train_drift", "test_drift",
             "ttest_drift", "agree_drift"]]
      .rename(columns={"beta_drift": "beta", "t_drift": "t", "train_drift": "b<=2020",
                       "test_drift": "b2021+", "ttest_drift": "t2021+",
                       "agree_drift": "sign ok"}).round(2).to_string(index=False))

# %%
short_nb = sweep[sweep.shortlist].sort_values("t_nextbar", key=abs, ascending=False)
print("WINDOW B — T+30 -> T+60, the only window that fits the 30-minute grid")
print(short_nb[["event", "n", "beta_nextbar", "t_nextbar", "train_nextbar",
                "test_nextbar", "agree_nextbar"]]
      .rename(columns={"beta_nextbar": "beta", "t_nextbar": "t",
                       "train_nextbar": "b<=2020", "test_nextbar": "b2021+",
                       "agree_nextbar": "sign ok"}).round(2).to_string(index=False))

# %% [markdown]
# **Findings:** On the drift window only **CPI MoM and Core CPI MoM** keep their sign and
# their magnitude across the 2020 split — −4.01 then −6.99 for headline, −3.70 then −5.25
# for core, both strengthening in the confirmation half. Everything else with a large
# full-sample t-statistic fails: Initial Jobless Claims flips from +3.16 to −2.90, Retail
# Sales Advance from +3.60 to −0.02, and ISM Manufacturing keeps its sign but collapses
# from +6.14 to +0.03.
#
# Window B is the important negative result. **CPI has no drift left by T+30**, at
# +0.99 bps and t = +0.72 against −5.64 and t = −2.68 in the preceding half hour, and its
# sign flips across the split (+3.14 to −0.51). Core CPI is the same at t = +0.40. The
# signal is entirely consumed inside the first thirty minutes, which is what rules it out
# as a feature on the current grid.

# %% [markdown]
# ## The null
#
# Ninety-nine releases were tested and the shortlist is 16 of them, so the question is
# how many strong-looking results this many tests produces on their own. The relevant
# comparison is not the count of significant t-statistics but how often a sign survives
# the split, since that is the criterion actually being used.

# %%
for label, name in [("drift", "T+1 -> T+30"), ("nextbar", "T+30 -> T+60")]:
    t = sweep[f"t_{label}"].dropna()
    agree = sweep[f"agree_{label}"].dropna()
    strong = sweep[(sweep[f"t_{label}"].abs() > 2) & sweep[f"agree_{label}"].notna()]
    print(f"{name}: |t| over {len(t)} releases — median {t.abs().median():.2f}, "
          f"90th {t.abs().quantile(.9):.2f}, max {t.abs().max():.2f}")
    print(f"   |t| > 2.0: {(t.abs() > 2).sum():>2}  (expected at 5%: {0.05 * len(t):.0f})")
    print(f"   |t| > 2.5: {(t.abs() > 2.5).sum():>2}  (expected: {0.012 * len(t):.1f})")
    print(f"   of those, sign agrees across the split: "
          f"{int(strong[f'agree_{label}'].sum())}/{len(strong)}")
    print(f"   sign-agreement rate across all releases: {agree.mean():.0%}\n")

# %% [markdown]
# **Findings:** There is more than chance in aggregate — 15 releases clear |t| > 2 on the
# drift window against five expected, and 6 clear |t| > 2.5 against one. But the
# agreement rate tells the real story: across all 99 releases a sign survives the 2020
# split **54% of the time**, which is a coin flip. Passing |t| > 2 on the full sample is
# not evidence of anything by itself, and the filter that matters is the split.
#
# Note that 13 of the 14 splittable releases above |t| > 2 do keep their sign. That sounds
# impressive until it is read against what the sign is worth: most of those keep a sign
# while losing almost all of their magnitude, which is why the sweep above reports the
# confirmation-half beta rather than the agreement flag alone.

# %% [markdown]
# ## Why the largest t-statistic is an artefact
#
# Initial Jobless Claims returns t = +9.15 on the next-bar window, far above anything
# else in the sweep, on 663 observations. It is worth showing why it is discarded,
# because the failure mode is general: a weekly series through a macro shock has a
# surprise distribution that no trailing scale can keep up with.

# %%
claims = surprise_series("Initial Jobless Claims", "08:30:00")
claims_px = event_prices(claims, [30, 60])
ok = np.all([np.isfinite(v) for v in claims_px.values()], axis=0)
claims = claims[ok]
ret = 1e4 * np.log(claims_px[60][ok] / claims_px[30][ok])
raw_z = (claims.surprise / claims.surprise.expanding().std().shift(1)).to_numpy()
year = claims.ts.dt.year.to_numpy()

print("Initial Jobless Claims, T+30 -> T+60")
for label, z_use, mask in [
    ("unwinsorised", raw_z, np.ones(len(ret), bool)),
    ("winsorised ±5", claims.z.to_numpy(), np.ones(len(ret), bool)),
    ("excluding 2020", claims.z.to_numpy(), year != 2020),
]:
    beta, tstat, _ = ols_t(z_use[mask], ret[mask])
    print(f"  {label:<16} beta {beta:+6.2f}   t {tstat:+6.2f}   n {mask.sum()}")
worst = np.nanargmax(np.abs(raw_z))
print(f"\n  largest raw |z| = {abs(raw_z[worst]):.1f} on {claims.ts.dt.date.iloc[worst]}")
print(f"  observations beyond 5 sigma: {(np.abs(raw_z) > 5).sum()}")

# %% [markdown]
# **Findings:** The t-statistic of +9.68 is three observations. The largest raw surprise
# is **115 sigma**, on 2020-03-26, the week initial claims printed above three million
# against a consensus built on a 200,000 base. Winsorising at five sigma takes t to
# +3.68; dropping 2020 entirely takes it to **+0.23**. Nothing survives, in either half
# of the split.
#
# This is the reason `WINSOR_SIGMA` exists rather than being a tidiness measure, and it
# is worth remembering for any feature built on a ratio to a trailing scale.

# %% [markdown]
# ## The signal
#
# One release survives: the monthly CPI print at 08:30 ET. Headline and core are
# published together and measure the same thing, so they are averaged into a single
# inflation surprise rather than entered separately — two correlated regressors on 147
# observations would only split the same coefficient.
#
# The trade is an event-time overlay, not a bar on the existing grid: enter at 08:31,
# exit at 09:00, short ES on a hot print and long on a cold one. Costs are a full tick,
# which is more conservative than the quarter tick used in `backtest.py` because this
# window is thinner than the cash session.

# %%
def by_release_date(event_name):
    """Surprise z-score indexed by release date, so headline and core can be joined."""
    g = surprise_series(event_name, "08:30:00")
    return g.set_index(g.RELEASE_DATE.dt.normalize()).z


z_head, z_core = by_release_date("CPI MoM"), by_release_date("Core CPI MoM")
infl = pd.concat([z_head.rename("headline"), z_core.rename("core")], axis=1).dropna()
infl["surprise"] = infl[["headline", "core"]].mean(axis=1)

stamp = pd.to_datetime(infl.index.strftime("%Y-%m-%d") + " 08:30")
infl["ts"] = (stamp.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
              .tz_convert("UTC"))
infl = infl.dropna(subset=["ts"])
prices = event_prices(infl, [ENTRY_LAG_MIN, DRIFT_END_MIN])
ok = np.isfinite(prices[ENTRY_LAG_MIN]) & np.isfinite(prices[DRIFT_END_MIN])
infl = infl[ok]
drift = 1e4 * np.log(prices[DRIFT_END_MIN][ok] / prices[ENTRY_LAG_MIN][ok])
year = infl.index.year.to_numpy()

print(f"CPI releases: {len(infl)}   "
      f"{infl.index.min().date()} -> {infl.index.max().date()}\n")
for label, series in [("headline only", infl.headline), ("core only", infl.core),
                      ("combined", infl.surprise)]:
    beta, tstat, _ = ols_t(series, drift)
    print(f"  {label:<14} beta {beta:+6.2f} bps/sigma   t {tstat:+6.2f}")
print()
for label, mask in [("<=2020", year <= SPLIT_YEAR), ("2021+", year > SPLIT_YEAR)]:
    beta, tstat, _ = ols_t(infl.surprise[mask], drift[mask])
    print(f"  combined {label:<7} beta {beta:+6.2f}   t {tstat:+6.2f}   n {mask.sum()}")

# %%
print("sign trade, 08:31 -> 09:00, net of one full tick:\n")
span_years = (infl.index.max() - infl.index.min()).days / 365.25
print(f"{'filter':<14}{'n':>5}{'hit %':>8}{'gross':>9}{'net':>8}{'t':>7}{'per yr':>8}{'ann %':>9}")
for label, threshold in [("all", 0.0), ("|z| > 0.5", 0.5), ("|z| > 1.0", 1.0)]:
    mask = infl.surprise.abs() > threshold
    pnl = -np.sign(infl.surprise[mask]) * drift[mask]
    tstat = pnl.mean() / (pnl.std(ddof=1) / np.sqrt(mask.sum()))
    per_year = mask.sum() / span_years
    net = pnl.mean() - TICK_BPS
    print(f"{label:<14}{mask.sum():>5}{100 * (pnl > 0).mean():>8.1f}{pnl.mean():>9.2f}"
          f"{net:>8.2f}{tstat:>7.2f}{per_year:>8.0f}{per_year * net / 100:>9.2f}")

mask = infl.surprise.abs() > 0.5
pnl = -np.sign(infl.surprise[mask]) * drift[mask]
print("\n|z| > 0.5, split across 2020:")
for label, sub in [("<=2020", year[mask] <= SPLIT_YEAR), ("2021+", year[mask] > SPLIT_YEAR)]:
    part = pnl[sub]
    print(f"  {label:<8} n {sub.sum():>3}   hit {100 * (part > 0).mean():>5.1f}%   "
          f"gross {part.mean():+6.2f} bps")

# %%
out = infl[["headline", "core", "surprise", "ts"]].copy()
out["drift_bps"] = drift
out.to_parquet(JACK_DATA / "cpi_surprise.parquet")
print(f"wrote cpi_surprise.parquet   {len(out)} releases")

# %% [markdown]
# **Findings:** The combined surprise beats either component, at −5.82 bps per sigma and
# t = −2.78 against −5.64 for headline alone and −4.45 for core. It keeps its sign and
# gains magnitude across the split: −4.99 (t = −2.64) to 2020, −6.54 (t = −1.70) after,
# the weaker t-statistic reflecting 64 observations rather than a weaker effect.
#
# As a sign trade the hit rate rises monotonically with the size of the surprise — 60.2%
# unfiltered, 67.4% above half a sigma, 73.5% above one — which is the response a real
# effect gives and not the response a fitted threshold gives. At the half-sigma filter it
# is **+9.15 bps per trade net of a full tick, t = +3.38, on six trades a year**, worth
# about 0.5% of notional annually. Split across 2020 the hit rate is near-identical at
# 67.3% and 67.6%, and the confirmation half is the larger of the two in size, +15.53 bps
# against +5.84.
#
# Against a project where nothing directional has cleared a hit rate of 51%, a 67% hit
# rate at t = +3.38 is the first real directional result. It is also six trades a year on
# one contract, so it is an overlay rather than a strategy.

# %% [markdown]
# ## What this can and cannot do
#
# **It cannot enter `MODEL_FEATURES`.** CPI prints at 08:30:00 ET, exactly on a bar
# boundary. The bar labelled 08:00 has its decision point at 08:30, simultaneous with the
# release, so using the surprise to predict the 08:30-09:00 return would capture the jump
# — the same non-executable entry the lag sweep rules out. The next genuinely clean bar
# is 09:00-09:30, and the drift is gone by then at t = +0.66.
#
# **It can run as an event-time overlay.** Position at 08:31 on the twelve CPI days a
# year, sized by the surprise, flat at 09:00. It is uncorrelated with the main book,
# which runs at a beta of 1.32, and it does not compete for the same capital.
#
# Three caveats belong on the record. The sample is 147 releases and 86 above the
# half-sigma filter. The 0.5 threshold was chosen after seeing the gradient, though the
# gradient being monotone is what makes it defensible rather than fitted. And 2021 onward
# alone is t = −1.70, so the confirmation half supports the sign without establishing it
# independently.
#
# The obvious extension is FOMC at 14:00 ET, which is the one major release that does not
# land on a 30-minute boundary and so may fit the grid where CPI does not.
