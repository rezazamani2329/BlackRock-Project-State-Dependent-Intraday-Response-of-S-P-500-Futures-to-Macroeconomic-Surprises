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
# # Feature engineering
#
# Six continuous features for an OLS regression on the forward 30-minute ES return,
# fitted walk-forward. The backtest turns the prediction into a share count.
#
# | feature | what it says |
# |---|---|
# | `range_pos_30m`, `range_pos_2h` | where price closed inside its recent range |
# | `vol_z` | how volatile the regime is |
# | `above_ma100` | which side of the long-run trend |
# | `is_event_window` | whether a big macro release lands in the window being predicted |
# | `seas_exp` | how large a move this time of day usually brings |
#
# **Timing.** Bars are labelled by their left edge: the bar labelled 10:00 spans
# `[10:00, 10:30)` and closes on the 10:29 print. Features at *t* use data through bar
# *t*; the target is the move into *t+1*.

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from dtaidistance import dtw
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import RidgeCV
from statsmodels.tools.tools import add_constant

SHARED_PROCESSED = Path("../data/processed")
SHARED_RAW = Path("../data/raw")
JACK_DATA = Path("data")

BAR_MINUTES = 30
BARS_PER_HOUR = 60 // BAR_MINUTES
MA_SESSIONS = 100
RIDGE_ALPHAS = np.logspace(-3, 6, 60)
FIRST_TEST_YEAR = 2021

# Volatility: EWMA on a 24-hour (one trading day) halflife, chosen by QLIKE on the
# training period. One trading day is a full global cycle, so the estimate is not
# biased by which session a bar falls in. The current bar is included — it is known at
# the decision point and carries only 1.4% of the estimate at this halflife.
VOL_HALFLIFE_BARS = 24 * BARS_PER_HOUR
VOL_MIN_PERIODS = 24

# Walk-forward normalisation: ~1 year of bars, median/MAD for the fat tails.
Z_WINDOW = 250 * 23 * BARS_PER_HOUR
Z_MIN_PERIODS = 2000
Z_CLIP = 10.0

# Releases whose move is large once time of day is controlled for. The ratio of scaled
# |return| to the same-slot non-event baseline is CPI 4.7x, NFP 3.6x, FOMC 3.1x, then
# PPI and Retail Sales at 1.9x. Everything else in the calendar sits at or below 1.6x,
# and JOLTS and Wholesale Trade collapse from 1.5x to 1.1x once the 10:00 slot is
# controlled for, so their apparent size was seasonality rather than the release.
BIG_EVENTS = ["CPI", "NFP", "FOMC", "PPI", "Retail Sales"]

pd.set_option("display.width", 140)


# %%
def robust_z(values: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """Median/MAD z-score against the previous `window` observations.

    Shifted one step so a value never enters its own reference window. MAD is scaled by
    1.4826 to estimate a standard deviation on normal data.
    """
    series = pd.Series(values)
    roll = series.rolling(window, min_periods=min_periods)
    centre = roll.median().shift(1)
    spread = 1.4826 * roll.apply(
        lambda v: np.nanmedian(np.abs(v - np.nanmedian(v))), raw=True
    ).shift(1)
    return ((series - centre) / spread.where(spread > 0)).clip(-Z_CLIP, Z_CLIP).to_numpy()


# %%
bars = pd.read_parquet(JACK_DATA / "es_30min_bars.parquet")
calendar = pd.read_parquet(SHARED_PROCESSED / "macro_event_calendar_expanded_2010_2026.parquet")

features = pd.DataFrame(index=bars.index)
features["session_date"] = bars.session_date
features["fwd_ret_30m"] = bars.fwd_ret_30m

print(f"{len(bars):,} bars   {bars.index.min()} -> {bars.index.max()}")

# %% [markdown]
# ## Range position
#
# Where the close sits inside the recent high-low range, centred on zero:
# `(close - low) / (high - low) - 0.5`.
#
# A close-to-close return cannot distinguish a smooth rise from a spike that faded; the
# high and low can. A close at the top of the range means buyers pushed into the close,
# and that reverts — short-horizon reversal from liquidity provision. The measure is
# bounded and self-normalising, so it needs no volatility scaling.
#
# Two horizons only: the effect decays fast, with a partial t of -2.53 at 30 minutes,
# -1.63 at 1 hour and -0.94 at 2 hours once `vol_z` and `above_ma100` are controlled.

# %%
RANGE_BARS = {"30m": 1, "2h": 4}
RANGE_FEATURES = [f"range_pos_{label}" for label in RANGE_BARS]

for label, n in RANGE_BARS.items():
    high_n = bars.high.rolling(n).max()
    low_n = bars.low.rolling(n).min()
    span = (high_n - low_n).replace(0, np.nan)     # a flat bar has no position in range
    features[f"range_pos_{label}"] = (bars.close - low_n) / span - 0.5

print(features[RANGE_FEATURES].agg(["mean", "std"]).round(3).to_string())
print(f"correlation between the two: "
      f"{features[RANGE_FEATURES[0]].corr(features[RANGE_FEATURES[1]]):+.3f}")
for c in RANGE_FEATURES:
    d = features[[c, "fwd_ret_30m"]].dropna()
    print(f"{c:<16} univariate IC {d[c].corr(d.fwd_ret_30m):+.4f}")

# %% [markdown]
# **Findings:** Both centre near zero and span the full range by construction. The two
# correlate at 0.635 — the 30-minute bar is nested inside the 2-hour window. Univariate
# ICs are negative (-0.0108 and -0.0029), consistent with the reversal story.

# %% [markdown]
# ## Realised volatility
#
# EWMA volatility in its own right, not just as a scaler. It is strongly right-skewed
# and its level drifts, so the feature is `log(vol)` through a walk-forward z-score:
# how unusual is current volatility against the past year.

# %%
log_close = np.log(bars.close)
ret_bar = log_close.diff()
realised_vol = ret_bar.ewm(halflife=VOL_HALFLIFE_BARS, min_periods=VOL_MIN_PERIODS).std()

features["realised_vol"] = realised_vol
features["vol_z"] = robust_z(np.log(realised_vol.to_numpy()), Z_WINDOW, Z_MIN_PERIODS)

print(f"correlation with |forward return|:      {features.vol_z.corr(features.fwd_ret_30m.abs()):+.3f}")
print(f"correlation with signed forward return: {features.vol_z.corr(features.fwd_ret_30m):+.4f}")

# %% [markdown]
# **Findings:** `vol_z` correlates **+0.340 with the absolute forward return** and only
# +0.014 with the signed one — the strongest relationship in the feature set, and
# entirely about magnitude rather than direction.

# %% [markdown]
# ## Moving average
#
# Which side of the 100-session average price sits on. Built on session closes and
# shifted one session, so the current session's own close never enters the average it
# is compared against. Enters as a 0/1 dummy.

# %%
session_close = bars.groupby("session_date").close.last()
ma100 = session_close.rolling(MA_SESSIONS, min_periods=MA_SESSIONS).mean().shift(1)
mapped_ma = bars.session_date.map(ma100)

features["above_ma100"] = (bars.close > mapped_ma).astype(float)
features.loc[mapped_ma.isna(), "above_ma100"] = np.nan
print(f"bars above the 100-session MA: {features.above_ma100.mean():.1%}")

# %% [markdown]
# **Findings:** Price is above the average 78.2% of the time, so this is closer to a
# regime label than a signal, and it correlates -0.50 with `vol_z` — drawdowns are the
# volatile periods.

# %% [markdown]
# ## Events
#
# Whether one of the big macro releases lands inside the window being predicted, rather
# than merely somewhere on the same day. Release schedules are published weeks ahead, so
# this is a calendar fact known at the decision point.
#
# The bar labelled *t* spans `[t, t+30)` and its target is the move into `[t+30, t+60)`,
# so the flag fires when a release timestamp falls in that second interval. A day-level
# flag spreads 46 bars of quiet trading across the one bar that carries the release; this
# keeps the bar that actually contains it.

# %%
events = calendar[calendar.event_type.isin(BIG_EVENTS)].copy()
events["timestamp_utc"] = pd.to_datetime(events.timestamp_utc, utc=True)
release_ts = np.sort(events.timestamp_utc.to_numpy())

bar_ts = features.index.to_numpy()
pos = np.searchsorted(release_ts, bar_ts)
next_release = np.where(pos < len(release_ts),
                        release_ts[np.minimum(pos, len(release_ts) - 1)],
                        np.datetime64("2100-01-01"))
minutes_to_release = (next_release - bar_ts) / np.timedelta64(1, "m")

features["is_event_window"] = (
    (minutes_to_release >= BAR_MINUTES) & (minutes_to_release < 2 * BAR_MINUTES)
).astype(float)

scaled_abs = features.fwd_ret_30m.abs() / features.realised_vol
target_slot = (features.index + pd.Timedelta(minutes=BAR_MINUTES)).tz_convert("America/New_York")
target_slot = pd.Series(target_slot.strftime("%H:%M"), index=features.index)
quiet = features.is_event_window == 0
slot_baseline = scaled_abs[quiet].groupby(target_slot[quiet]).mean()

fired = features.is_event_window == 1
lift = (scaled_abs[fired] / target_slot[fired].map(slot_baseline)).mean()
print(f"bars flagged: {int(fired.sum()):,} ({fired.mean():.2%})")
print(f"scaled |return| on flagged bars: {scaled_abs[fired].mean():.2f} "
      f"vs {scaled_abs[quiet].mean():.2f} elsewhere")
print(f"lift against the same time-of-day baseline: {lift:.2f}x")

# %% [markdown]
# **Findings:** 557 bars, 0.48% of the sample, carrying 2.16x the scaled move of a
# typical bar and 2.77x the same time-of-day baseline. The day-level flag it replaces
# covered 44% of bars for a 1.13x lift, since it spread each release across the 46 other
# bars of its day. Controlling for time of day matters: JOLTS and Wholesale Trade look
# like 1.5x events on raw numbers but fall to 1.1x against their own 10:00 slot, so their
# size was seasonality and they are excluded. Like the flag it replaces this is a
# magnitude feature, and it adds nothing to a signed model.

# %% [markdown]
# ## Intraday volatility seasonality
#
# How large a move this half-hour usually brings, measured in units of the prevailing
# 24-hour volatility: the expanding mean of `|return| / vol` at this slot over all
# earlier sessions.
#
# `vol_z` is a 24-hour EWMA and carries no intraday structure by construction, so it
# measures the *level* of volatility while this measures the *time-of-day multiplier* on
# it. The two are close to orthogonal and multiply rather than overlap.
#
# Expanding rather than rolling. A 250-session rolling version correlates +0.975 with
# this and scores the same, and the expanding version has no window to justify. The
# profile comes from session opens, closes and scheduled release times, which are
# structural, so there is little reason to forget old observations.

# %%
slot = pd.Series(bars.index.tz_convert("America/New_York").strftime("%H:%M"), index=bars.index)
scaled_abs = bars.fwd_ret_30m.abs() / realised_vol

features["seas_exp"] = (
    scaled_abs.groupby(slot, group_keys=False)
    .apply(lambda s: s.shift(1).expanding(min_periods=40).mean())
)

print(f"seas_exp defined on {features.seas_exp.notna().sum():,} bars")
print(f"  range {features.seas_exp.min():.3f} to {features.seas_exp.max():.3f}")
print(f"  correlation with vol_z      {features.seas_exp.corr(features.vol_z):+.3f}")
print(f"  correlation with |fwd_ret|  "
      f"{features.seas_exp.corr(bars.fwd_ret_30m.abs() / realised_vol):+.3f}")
print(f"  correlation with fwd_ret    {features.seas_exp.corr(bars.fwd_ret_30m):+.4f}")

# %% [markdown]
# **Findings:** Correlation with the *absolute* forward return is **+0.41**, the largest
# in the feature set by a wide margin, and with the signed return it is near zero. It is
# orthogonal to `vol_z` (-0.02), which is the point: one is the volatility level, the
# other the time-of-day multiplier. This feature predicts how big the next move is, not
# which way it goes.

# %% [markdown]
# ## Shared timing
#
# Bar *t* spans `[t, t+30)` and closes on the print before `t+30`, so `t+30` is the
# moment a decision is made and everything below is measured against it. The 1-minute
# series is loaded once here and reused by the realised-variance and DTW sections.

# %%
decision_point = features.index + pd.Timedelta(minutes=BAR_MINUTES)
bar_slot = pd.Series(features.index.tz_convert("America/New_York").strftime("%H:%M"),
                     index=features.index)

minute = pd.read_parquet(SHARED_PROCESSED / "es_1min_clean.parquet",
                         columns=["close", "volume"]).sort_index()
minute = minute[~minute.index.duplicated(keep="first")]
print(f"1-minute bars: {len(minute):,}   "
      f"{minute.index.min().date()} -> {minute.index.max().date()}")

# %% [markdown]
# ## Realised variance and volume, from the 1-minute data
#
# `vol_z` is an EWMA of 30-minute bar returns. Estimating the same quantity from
# 1-minute returns instead is far more precise, which is the point of the realised
# volatility literature: variance is estimated better by sampling more finely, not by
# smoothing longer. Three measures over a trailing four-hour window, all ending at the
# decision point:
#
# - `log_rv4h`, log realised variance from 240 one-minute squared returns
# - `signed_jump`, realised semivariance skew, `(RS+ - RS-) / RV`, from
#   Barndorff-Nielsen, Kinnebrock and Shephard. Downside and upside variance carry
#   different information about what follows
# - `vol_surprise`, log traded volume against the trailing median for this clock slot.
#   Volume and volatility are driven jointly by information arrival, so a volume
#   surprise is a volatility surprise
#
# `vol_surprise` is a *surprise* rather than a level, so it does not restate the
# volume profile that `seas_exp` already carries.

# %%
RV_WINDOW_MIN = 240

minute_ret = np.log(minute.close).diff()
sq = minute_ret ** 2
up_sq = np.where(minute_ret > 0, sq, 0.0)
down_sq = np.where(minute_ret < 0, sq, 0.0)

roll = lambda x: pd.Series(x, index=minute.index).rolling(
    RV_WINDOW_MIN, min_periods=RV_WINDOW_MIN // 2).sum()
rv_4h, rs_up, rs_down = roll(sq), roll(up_sq), roll(down_sq)
vol_4h = minute.volume.rolling(RV_WINDOW_MIN, min_periods=RV_WINDOW_MIN // 2).sum()

# the last 1-minute bar fully known at the decision point
minute_pos = np.clip(minute.index.searchsorted(decision_point, side="left") - 1,
                     0, len(minute) - 1)
fresh = np.abs((minute.index[minute_pos] - decision_point).total_seconds()) <= 120

rv_at = rv_4h.to_numpy()[minute_pos]
features["log_rv4h"] = np.where(fresh, np.log(np.maximum(rv_at, 1e-14)), np.nan)
features["signed_jump"] = np.where(
    fresh, (rs_up.to_numpy()[minute_pos] - rs_down.to_numpy()[minute_pos])
    / np.maximum(rv_at, 1e-12), np.nan)

log_volume = np.where(fresh, np.log(np.maximum(vol_4h.to_numpy()[minute_pos], 1.0)), np.nan)
features["vol_surprise"] = (
    pd.Series(log_volume, index=features.index)
    .groupby(bar_slot, group_keys=False)
    .apply(lambda s: s - s.shift(1).expanding(min_periods=40).median())
)

for c in ["log_rv4h", "signed_jump", "vol_surprise"]:
    d = features[[c, "fwd_ret_30m"]].dropna()
    print(f"{c:<14} defined on {features[c].notna().sum():>7,}   "
          f"corr with |fwd_ret| {d[c].corr(d.fwd_ret_30m.abs()):+.3f}   "
          f"signed {d[c].corr(d.fwd_ret_30m):+.4f}")

# %% [markdown]
# **Findings:** `log_rv4h` correlates +0.451 with the absolute forward return, the
# strongest of the three and ahead of `vol_z`, which measures the same thing from coarser
# data. `vol_surprise` reaches +0.214 and `signed_jump` -0.046. Scored against the
# full model including the HAR terms, they are worth 5.9% and 8.9% of QLIKE on the two
# windows for `log_rv4h`, 1.1% and 3.3% for `vol_surprise`, and under 1% for
# `signed_jump`, which is kept for completeness rather than for its contribution.

# %% [markdown]
# ## The simple ones
#
# Two cheap features that survive on both windows, and a note on one that does not.
#
# `bar_range` is the bar's own high-low range in volatility units, the shortest-horizon
# realised measure available and the freshest. `tod_sin` and `tod_cos` encode the clock
# as a cycle, which lets the model use time of day directly rather than only through the
# single scalar `seas_exp` carries.
#
# **Raw price level is deliberately absent.** Log price correlates -0.18 on 2019-2020 and
# -0.11 on 2021-2026 with the absolute forward return, which looks usable and is not. Over
# the full sample the same correlation is -0.02, because the relationship is not stable;
# what is stable is its 0.97 correlation with the calendar year. It is a proxy for when
# rather than for what. In a walk-forward the
# test year always sits outside the training range, so the model extrapolates a trend it
# has no reason to believe. Added to the model it makes QLIKE *worse* on both windows,
# -1.33% and -0.38%. The correct way to use price is relative to something, which is what
# `range_pos` and `above_ma100` already do.
#
# The immediately preceding 30-minute return is also absent. It scores +0.16% on one
# window and -0.15% on the other, and `range_pos_30m` already carries the same
# information in stationary form.

# %%
features["bar_range"] = np.log(bars.high / bars.low) / realised_vol
minutes_et = (features.index.tz_convert("America/New_York").hour * 60
              + features.index.tz_convert("America/New_York").minute)
features["tod_sin"] = np.sin(2 * np.pi * minutes_et / 1440)
features["tod_cos"] = np.cos(2 * np.pi * minutes_et / 1440)

for c in ["bar_range", "tod_sin", "tod_cos"]:
    d = features[[c, "fwd_ret_30m"]].dropna()
    print(f"{c:<10} corr with |fwd_ret| {d[c].corr(d.fwd_ret_30m.abs()):+.3f}   "
          f"signed {d[c].corr(d.fwd_ret_30m):+.4f}")
lp = np.log(bars.close)
print(f"\nlog price corr with |fwd_ret| "
      f"{lp.corr(features.fwd_ret_30m.abs()):+.3f}, with calendar year "
      f"{np.corrcoef(lp, np.asarray(features.index.year))[0, 1]:+.3f}  (excluded)")

# %% [markdown]
# **Findings:** `bar_range` is the best of the simple candidates, worth 3.5% and 5.5% of
# QLIKE on the two windows on top of everything else, which is more than most of the
# engineered features contribute. The cyclical clock encoding adds 1.1% and 1.4%, small
# but consistent in sign. Neither is expensive to compute and both were sitting in the
# bar table already.

# %% [markdown]
# ## Implied volatility
#
# VIX is the one forward-looking input available. Everything else here is computed from
# past prices, and implied volatility consistently beats backward-looking estimators at
# forecasting realised volatility, so it carries information the rest of the feature set
# structurally cannot.
#
# Pulled by `get_vix.py` into `data/raw/`. Two features: the log level, and the slope of
# the term structure as `log(VIX3M / VIX)`, which is positive in calm regimes and inverts
# under stress.
#
# **The merge is on publication time, not date.** These are daily closes stamped 16:15
# ET. Joining on the calendar date would hand a morning bar a number published that
# afternoon, so the join is an as-of merge against the publication instant and a bar
# always carries the last close actually released before its decision point.

# %%
vix = pd.read_parquet(SHARED_RAW / "vix_daily.parquet").sort_values("published_utc")
vix["published_utc"] = vix.published_utc.astype("datetime64[ns, UTC]")
vix_asof = pd.merge_asof(
    pd.DataFrame({"decision": decision_point}).sort_values("decision"),
    vix[["published_utc", "vix", "vix3m"]],
    left_on="decision", right_on="published_utc", direction="backward",
)
vix_asof.index = features.index

features["vix_level"] = np.log(vix_asof.vix.to_numpy())
features["vix_slope"] = np.log(vix_asof.vix3m.to_numpy() / vix_asof.vix.to_numpy())

stale_days = (decision_point - vix_asof.published_utc).dt.total_seconds() / 86400
print(f"vix_level defined on {features.vix_level.notna().sum():,} bars")
print(f"  age of the VIX close used: median {stale_days.median():.2f} days, "
      f"max {stale_days.max():.2f}")
d = features[["vix_level", "fwd_ret_30m"]].dropna()
print(f"  corr with |fwd_ret| {d.vix_level.corr(d.fwd_ret_30m.abs()):+.3f}   "
      f"signed {d.vix_level.corr(d.fwd_ret_30m):+.4f}")
print(f"  corr with vol_z {features.vix_level.corr(features.vol_z):+.3f}")

# %% [markdown]
# **Findings:** The VIX close in use is a median 0.64 days old and never more than 4.0,
# the gap being weekends and holidays. Correlation with the absolute forward return is
# +0.380, and with `vol_z` +0.582, so it overlaps the realised measures substantially
# without being redundant with them. Its directional correlation is +0.016.

# %% [markdown]
# ## Parkinson range, the magnitude target
#
# If entry and exit can fall anywhere inside the half-hour rather than at its edges, the
# magnitude that matters is the window's high-low range, not its close-to-close move.
# The Parkinson estimator turns that range into a variance, and it is a far less noisy
# measure of the same thing: a ratio of standard deviation to mean of 2.55 against 4.00
# for the squared close-to-close return.
#
# This is a target, not a feature. It is written to the feature table so `vol_model.py`
# can score against it.

# %%
next_bar = bars.shift(-1)
contiguous_next = (bars.index.to_series().shift(-1) - bars.index.to_series()) == pd.Timedelta(
    minutes=BAR_MINUTES)
features["rv_park"] = np.where(
    np.asarray(contiguous_next),
    np.log(next_bar.high / next_bar.low) ** 2 / (4 * np.log(2)),
    np.nan,
)
ratio = features.rv_park / features.realised_vol ** 2
print(f"rv_park defined on {features.rv_park.notna().sum():,} bars")
print(f"  scaled: mean {ratio.mean():.3f}  sd {ratio.std():.3f}  "
      f"sd/mean {ratio.std() / ratio.mean():.2f}")

# %% [markdown]
# ## DTW on the volatility trajectory
#
# The idea that a half-hour resembles earlier half-hours, finally in a form that works.
# Four earlier DTW designs failed, and all of them z-normalised the path before matching,
# which throws away amplitude. That is exactly what magnitude prediction needs, so this
# version matches the *volatility trajectory* instead: the last two hours of five-minute
# absolute returns divided by prevailing volatility. The result is unitless and describes
# whether volatility is building, decaying or spiking, which a 24-hour EWMA cannot tell
# apart.
#
# Each neighbour contributes its own forward scaled Parkinson range, demeaned by time of
# day using training rows only, so intraday seasonality cannot come through the wrapper
# the way it did in the earlier magnitude attempt. Neighbours are weighted by inverse
# distance and by recency. Matches must be at least a day old, and the pool is drawn only
# from strictly earlier calendar years.

# %%
PATH_STEP, PATH_LEN = 5, 24          # 24 x 5min = a 2-hour lead-up
DTW_K, DTW_POOL = 25, 4000
SAKOE_WINDOW = 4
RECENCY_HALFLIFE_DAYS = 250.0
MIN_MATCH_AGE_DAYS = 1.0
DTW_FIRST_YEAR = 2018                # earlier years have no pool to draw on

close5 = minute.close.resample(f"{PATH_STEP}min").last().dropna()
ret5 = np.log(close5).diff()
vol5 = ret5.ewm(halflife=24 * 60 // PATH_STEP, min_periods=100).std()
traj5 = (ret5.abs() / vol5).to_numpy()
grid5 = close5.index

# A 5-minute bar labelled T holds data through T+5, so the last bar usable at the
# decision point is the one labelled t+25. Taking the bar labelled t+30 instead would
# put the first five minutes of the window being predicted inside the query path.
last_usable = decision_point - pd.Timedelta(minutes=PATH_STEP)
end = grid5.searchsorted(last_usable, side="right") - 1
paths = np.full((len(features), PATH_LEN), np.nan)
usable_end = (end >= PATH_LEN) & (end < len(grid5))
for i in np.flatnonzero(usable_end):
    paths[i] = traj5[end[i] - PATH_LEN + 1 : end[i] + 1]
has_path = usable_end & np.isfinite(paths).all(axis=1)
print(f"bars with a usable 2-hour volatility trajectory: {has_path.sum():,} "
      f"({has_path.mean():.1%})")

# %%
outcome_raw = np.log(features.rv_park / features.realised_vol ** 2)
eligible = (has_path & np.isfinite(outcome_raw.to_numpy())
            & np.isfinite(features.realised_vol.to_numpy())
            & (features.realised_vol.to_numpy() > 0))
slot_of = features.index.tz_convert("America/New_York").strftime("%H:%M").to_numpy()
days = (features.index.tz_convert("UTC").tz_localize(None).to_numpy()
        .astype("datetime64[ns]").astype("float64") / (1e9 * 86400))

paths_c = np.ascontiguousarray(np.nan_to_num(paths), dtype=np.double)
bar_year = np.asarray(features.index.year)
dtw_mag = np.full(len(features), np.nan)
rng = np.random.default_rng(0)

for test_year in range(DTW_FIRST_YEAR, bar_year.max() + 1):
    test_rows = np.flatnonzero((bar_year == test_year) & eligible)
    pool_all = np.flatnonzero((bar_year < test_year) & eligible)
    if len(test_rows) == 0 or len(pool_all) < 2000:
        continue
    pool = np.sort(rng.choice(pool_all, size=min(DTW_POOL, len(pool_all)), replace=False))

    stacked = np.vstack([paths_c[test_rows], paths_c[pool]])
    n_test = len(test_rows)
    dist = np.asarray(dtw.distance_matrix_fast(
        stacked, block=((0, n_test), (n_test, len(stacked))),
        window=SAKOE_WINDOW, compact=False, parallel=True))[:n_test, n_test:]

    slot_mean = pd.Series(outcome_raw.to_numpy()[pool_all]).groupby(slot_of[pool_all]).mean()
    pool_outcome = (outcome_raw.to_numpy() - pd.Series(slot_of).map(slot_mean).to_numpy())[pool]
    pool_days = days[pool]

    for i in range(n_test):
        age = days[test_rows[i]] - pool_days
        row = np.where(age >= MIN_MATCH_AGE_DAYS, dist[i], np.inf)
        ok = np.flatnonzero(np.isfinite(row))
        if len(ok) < DTW_K:
            continue
        near = ok[np.argpartition(row[ok], DTW_K)[:DTW_K]]
        w = (1.0 / (row[near] + 1e-8)) * np.exp(-age[near] / RECENCY_HALFLIFE_DAYS)
        dtw_mag[test_rows[i]] = float(np.dot(w / w.sum(), pool_outcome[near]))

features["dtw_mag"] = dtw_mag
scored = features.dtw_mag.notna()
# rv_park is heavily right-skewed, so rank correlation is the honest measure against it
print(f"dtw_mag defined on {scored.sum():,} bars from {DTW_FIRST_YEAR}")
print(f"  rank corr with rv_park      "
      f"{features.dtw_mag.corr(features.rv_park, method='spearman'):+.3f}")
print(f"  rank corr with |fwd_ret|    "
      f"{features.dtw_mag.corr(features.fwd_ret_30m.abs(), method='spearman'):+.3f}")
print(f"  rank corr with fwd_ret      "
      f"{features.dtw_mag.corr(features.fwd_ret_30m, method='spearman'):+.4f}")
print(f"  correlation with seas_exp   {features.dtw_mag.corr(features.seas_exp):+.3f}")

# %% [markdown]
# **Findings:** A magnitude feature and nothing else, so it is deliberately kept out of
# `MODEL_FEATURES`, which feeds the directional regression.
#
# The path alignment here is the whole ballgame. A first version of this feature ended the
# query at the bar labelled `t+30`, which on a left-labelled 5-minute grid holds data
# through `t+35` and therefore leaked the first five minutes of the window being
# predicted. That version scored a 13.45% QLIKE improvement; ending the path at `t+25`
# instead, as it now does, gives 7.39%. Roughly 45% of the apparent gain was the leak. The
# assertion in the lookahead section exists so this cannot come back.
#
# The controls are in `vol_model.py`. Against 20 shuffled-outcome nulls and 20
# random-neighbour controls the corrected feature still improves QLIKE by 7.39% on
# 2019-2020 and 5.30% on 2021-2026, with no control above 0.30%.

# %% [markdown]
# ## Multicollinearity
#
# Under OLS a correlated predictor inflates its own standard error rather than being
# shrunk, so this is a caveat on the coefficients. Inflation is `sqrt(VIF)`.

# %%
# Every engineered feature goes into the model. The assertion at the save step exists so
# a feature cannot be added to the table and silently fail to reach it.
MODEL_FEATURES = RANGE_FEATURES + [
    "vol_z", "seas_exp", "above_ma100", "is_event_window",
    "dtw_mag", "log_rv4h", "signed_jump", "vol_surprise",
    "vix_level", "vix_slope", "bar_range", "tod_sin", "tod_cos",
]
model_data = features.dropna(subset=MODEL_FEATURES + ["fwd_ret_30m"]).copy()

design = add_constant(model_data[MODEL_FEATURES].astype(float))
vif = pd.DataFrame({
    "feature": design.columns,
    "vif": [variance_inflation_factor(design.to_numpy(), i) for i in range(design.shape[1])],
}).set_index("feature").drop("const")
vif["se_inflation"] = np.sqrt(vif.vif)
print(vif.round(3).to_string())

# %% [markdown]
# **Findings:** Nothing above 1.73, so standard errors are inflated by at most 31%. The
# two range features are the tightest pair at 0.635; everything else is close to
# independent.

# %% [markdown]
# ## Lookahead check

# %%
rng = np.random.default_rng(7)

# The moving average uses only the 100 previous session closes.
for pos in rng.choice(np.arange(MA_SESSIONS, len(session_close)), size=200, replace=False):
    assert np.isclose(session_close.iloc[pos - MA_SESSIONS:pos].mean(), ma100.iloc[pos])

# range_pos_2h reconstructs from the four bars ending at the decision point.
rows = rng.choice(np.flatnonzero(features.range_pos_2h.notna().to_numpy()), size=500, replace=False)
for r in rows:
    hi = bars.high.iloc[r - 3:r + 1].max()
    lo = bars.low.iloc[r - 3:r + 1].min()
    assert np.isclose((bars.close.iloc[r] - lo) / (hi - lo) - 0.5, features.range_pos_2h.iloc[r])

# realised_vol is a causal EWMA over returns through bar t.
r = int(rows[0])
w_ewm = 0.5 ** (np.arange(r, -1, -1) / VOL_HALFLIFE_BARS)
hist = ret_bar.iloc[: r + 1].to_numpy()
ok_h = np.isfinite(hist)
mu_w = np.sum(w_ewm[ok_h] * hist[ok_h]) / np.sum(w_ewm[ok_h])
var_w = (np.sum(w_ewm[ok_h] * (hist[ok_h] - mu_w) ** 2)
         / (np.sum(w_ewm[ok_h]) - np.sum(w_ewm[ok_h] ** 2) / np.sum(w_ewm[ok_h])))
assert np.isclose(np.sqrt(var_w), realised_vol.iloc[r], rtol=1e-6)

# vol_z uses only strictly prior observations.
logv = np.log(realised_vol.to_numpy())
for i in rng.choice(np.flatnonzero(np.isfinite(features.vol_z.to_numpy())), size=40, replace=False):
    prior = logv[max(0, i - Z_WINDOW):i]
    prior = prior[np.isfinite(prior)]
    centre = np.median(prior)
    spread = 1.4826 * np.median(np.abs(prior - centre))
    assert np.isclose(np.clip((logv[i] - centre) / spread, -Z_CLIP, Z_CLIP),
                      features.vol_z.iloc[i], atol=1e-8)

# Leak canary: nothing should correlate implausibly with the target.
canary = {}
for c in MODEL_FEATURES:
    pair = features[[c, "fwd_ret_30m"]].dropna()
    canary[c] = abs(pair[c].corr(pair.fwd_ret_30m))
worst = max(canary, key=canary.get)
assert canary[worst] < 0.05, f"{worst} correlates {canary[worst]:.3f} with the target"

# the DTW path must end strictly before the window being predicted
probe = np.flatnonzero(features.dtw_mag.notna().to_numpy())[:200]
for r in probe:
    assert grid5[end[r]] + pd.Timedelta(minutes=PATH_STEP) <= features.index[r] + pd.Timedelta(
        minutes=BAR_MINUTES), "DTW path overlaps the target window"

# every 1-minute bar feeding log_rv4h must close at or before the decision point
last_minute_end = minute.index[minute_pos] + pd.Timedelta(minutes=1)
assert (last_minute_end[fresh] <= decision_point[fresh]).all(), "log_rv4h window overruns"

# the VIX close in use must already have been published
assert (vix_asof.published_utc.dropna()
        <= pd.Series(decision_point, index=features.index)[
            vix_asof.published_utc.notna()]).all(), "VIX close not yet published"

# bar_range uses bar t only, which is complete at the decision point
rows = rng.choice(np.flatnonzero(features.bar_range.notna().to_numpy()), size=200,
                  replace=False)
for r in rows:
    assert np.isclose(np.log(bars.high.iloc[r] / bars.low.iloc[r]) / realised_vol.iloc[r],
                      features.bar_range.iloc[r], rtol=1e-9)

print("all lookahead checks passed")
print(f"largest |corr(feature, target)|: {worst} = {canary[worst]:.4f}")

# %% [markdown]
# ## Feature table

# %%
TARGETS = ["session_date", "fwd_ret_30m", "realised_vol", "rv_park"]
ordered = TARGETS + MODEL_FEATURES
engineered = [c for c in features.columns if c not in TARGETS]
unaccounted = sorted(set(engineered) - set(MODEL_FEATURES))
assert not unaccounted, f"features reach neither list: {unaccounted}"
features[ordered].to_parquet(JACK_DATA / "features.parquet")

print(f"wrote features.parquet   {len(features):,} rows x {len(ordered)} columns")
print(f"complete rows for modelling: {len(model_data):,}")
print("\nfeature correlations:")
print(model_data[MODEL_FEATURES].corr().round(3).to_string())

# %% [markdown]
# ## Model — walk-forward OLS
#
# An expanding window trains on everything before the test year, predicts that year,
# and steps forward. Features are standardised on training-fold statistics only.
#
# Out-of-sample R-squared is measured against **zero**, not the training mean: the
# honest benchmark for a return forecast is "would you have done better assuming no
# move?", so a negative value means worse than predicting flat.

# %%
predictions, fold_stats = [], []
year = model_data.index.year

for test_year in range(FIRST_TEST_YEAR, int(year.max()) + 1):
    train = model_data[year < test_year]
    test = model_data[year == test_year]
    if len(test) == 0:
        continue

    mu, sigma = train[MODEL_FEATURES].mean(), train[MODEL_FEATURES].std()
    z_train = ((train[MODEL_FEATURES] - mu) / sigma).to_numpy()
    z_test = ((test[MODEL_FEATURES] - mu) / sigma).to_numpy()

    fit = RidgeCV(alphas=RIDGE_ALPHAS).fit(z_train, train.fwd_ret_30m.to_numpy())
    pred = pd.Series(fit.predict(z_test), index=test.index)
    in_sample_r2 = fit.score(z_train, train.fwd_ret_30m.to_numpy())

    predictions.append(pd.DataFrame({
        "pred": pred, "actual": test.fwd_ret_30m,
        "realised_vol": test.realised_vol, "fold": test_year,
    }))
    fold_stats.append({
        "fold": test_year,
        "n_train": len(train),
        "n_test": len(test),
        "r2_in": in_sample_r2,
        "alpha": fit.alpha_,
        "r2_oos": 1 - np.sum((test.fwd_ret_30m - pred) ** 2) / np.sum(test.fwd_ret_30m ** 2),
        "ic_oos": np.corrcoef(pred, test.fwd_ret_30m)[0, 1],
        "hit_rate": float(np.mean(np.sign(pred) == np.sign(test.fwd_ret_30m))),
        "long_rate": float(np.mean(pred > 0)),
    })

oos = pd.concat(predictions)
stats = pd.DataFrame(fold_stats).set_index("fold")
print(stats.round(5).to_string())

# %%
pooled_r2 = 1 - np.sum((oos.actual - oos.pred) ** 2) / np.sum(oos.actual ** 2)
print(f"rows {len(oos):,}   OOS R2 {pooled_r2:+.5f}   IC {oos.pred.corr(oos.actual):+.4f}")
print(f"hit rate {np.mean(np.sign(oos.pred) == np.sign(oos.actual)):.4f}   "
      f"always-long {np.mean(oos.actual > 0):.4f}   long share {np.mean(oos.pred > 0):.4f}")
print(f"prediction std {oos.pred.std() * 1e4:.2f} bps   actual std {oos.actual.std() * 1e4:.2f} bps")

# %% [markdown]
# ### Does the prediction rank the outcome?

# %%
decile = pd.qcut(oos.pred, 10, labels=False, duplicates="drop")
print(oos.groupby(decile).agg(size=("actual", "size"),
                              pred_bps=("pred", lambda s: s.mean() * 1e4),
                              actual_bps=("actual", lambda s: s.mean() * 1e4)).round(3).to_string())

# %% [markdown]
# ### Coefficients on the most recent fold
#
# Standardised: predicted move in basis points per one-standard-deviation change.

# %%
last_year = int(stats.index.max())
train = model_data[model_data.index.year < last_year]
mu, sigma = train[MODEL_FEATURES].mean(), train[MODEL_FEATURES].std()
final = sm.OLS(train.fwd_ret_30m, sm.add_constant((train[MODEL_FEATURES] - mu) / sigma)).fit()

print(pd.DataFrame({"coef_bps": final.params * 1e4, "t_stat": final.tvalues,
                    "p_value": final.pvalues}).round(3).to_string())
print(f"\nin-sample R2 {final.rsquared:.6f}   F p-value {final.f_pvalue:.4f}")

# %%
oos.to_parquet(JACK_DATA / "predictions.parquet")
print(f"wrote predictions.parquet  {len(oos):,} rows")

# %% [markdown]
# **Findings:** Two features carry the model: `range_pos_30m` at t = -2.99 (negative, as
# the reversal story predicts) and `vol_z` at t = 2.95. `range_pos_2h` is borderline at
# t = 1.76 and flips sign once the 30-minute version is controlled. Everything else sits
# below |t| = 0.5. In-sample R-squared is 0.000226 and the model is jointly significant
# at F p = 0.002 — real significance off ~99,000 rows, but a fiftieth of a percent of
# variance.
#
# Out of sample it does not work: R-squared -0.00011, IC +0.0056, hit rate 0.4947
# against 0.5030 for always-long, and the decile table has no ranking. Predictions span
# 0.29 bps against realised returns of 15.65.
#
# Swapping momentum for range position did improve every in-sample measure — R-squared
# from 0.000140, IC from +0.0009, and the F test from insignificant to significant — so
# the feature is doing something momentum was not. It is still an order of magnitude too
# small to trade.
