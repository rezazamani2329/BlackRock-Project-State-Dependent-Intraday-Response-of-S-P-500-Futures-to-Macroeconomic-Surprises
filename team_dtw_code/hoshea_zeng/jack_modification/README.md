# Intraday ES direction study

Can the direction of the next 30 minutes in ES futures be predicted well enough to
trade? The answer here is no. Fifteen features and four model families produce a forecast
whose out-of-sample R-squared is -0.00042, an information ratio against buy-and-hold of
-0.31, and no measurable alpha.

The work is still useful, because the same measurements that fail on direction succeed
consistently on magnitude. Realised variance, implied volatility, the intraday
seasonality profile, scheduled releases, and a DTW match on the recent volatility path
all say how large the next move will be. None of them say which way it goes.

The directional answer is stronger than "weak." Measured out of sample, the best of the
sixteen features tested reaches |t| = 2.31, against a median of 2.03 for the best of
sixteen *pure-noise* features scored on the same returns. A best-of-sixteen that strong
appears 28.8 percent of the time by chance. The feature set is not distinguishable from
noise.

**One directional result does exist, and it does not come from price history.** Scheduled
inflation releases move ES in the direction of the surprise, and part of that move arrives
*after* the number is public. A combined CPI surprise predicts the 08:31-09:00 return at
−5.82 basis points per sigma, t = −2.78, and as a sign trade above half a sigma it hits
67.4 percent net of a full tick on six trades a year. It is an event-time overlay rather
than a feature, for reasons set out in `macro_surprise.py` and below.

## Notebooks

Run in order. The `.py` files are the source of truth; sync with
`uv run jupytext --sync jack/<name>.py`.

| notebook | contents |
|---|---|
| `cleaning.py` | raw 1-minute bars to sessions, filtering, 30-minute grid |
| `feature_engineering.py` | fifteen features, walk-forward ridge, out-of-sample predictions |
| `backtest.py` | positions, costs, P&L against buy-and-hold |
| `intraday_analysis.py` | time-of-day attribution and win rates |
| `dtw_buckets.py` | six directional DTW designs, all rejected against nulls; the 16:00 ET lead closed as a leak |
| `get_vix.py` | pulls VIX and VIX3M from FRED into `data/raw/` |
| `vol_model.py` | the magnitude forecast scored against GARCH, HAR and a seasonal naive; DTW controls |
| `macro_surprise.py` | Bloomberg consensus surprises; post-announcement drift; the CPI overlay |

A DTW feature is now in the magnitude model. Four designs failed before one worked, and
the failures are kept in `dtw_buckets.py` and summarised below so they are not repeated.

Inputs live in `data/processed/es_1min_clean.parquet`, which is committed so the team
shares one cleaned series. Derived files stay in `jack/data/` and are gitignored.

## Data

The vendor file nominally covers 2010 to 2026, but whole days are missing before 2016,
including US cash sessions. Only 20 to 37 percent of weekdays in 2010 to 2012 have a
complete session, against 93 to 95 percent from 2016. `cleaning.py` measures this
rather than assuming it and sets the modelling start at 2016. Recovering the earlier
years needs a re-pull from Databento.

Sessions run 18:00 to 17:00 ET, dated by the day they close on. Of 4,130 sessions,
3,337 survive. The 793 dropped are 75 exchange holidays, 63 quarterly roll Fridays, and
655 short or incomplete sessions, 600 of which predate 2016. Roll Fridays go because
`ES.c.0` switches contracts on the third Friday of each quarter, so those sessions are
truncated and carry an artificial price jump.

Bars are 30 minutes, matching the trading interval, giving 116,605 bars from 2016 with
96.6 percent carrying a defined forward return. The gaps sit before the daily halt, the
weekend, and each dropped session, where a naive `shift(-1)` would have turned a
multi-day return into a 30-minute one. Realised volatility annualises to 17.3 percent
against 17.4 on an hourly grid, so the finer resampling adds observations without
distorting the series.

The target is mildly imbalanced at 51.9 percent up, which makes always-long the
benchmark. A further 4.5 percent of bars close exactly unchanged and count as a loss
for either side.

## Features

Information coefficients are univariate correlations with the forward 30-minute return.
The first pair is measured on the training period, the second is the same correlation
measured **out of sample** on the 2021-2026 walk-forward rows, and the third is the rank
correlation against the Parkinson range of the next bar, which is the magnitude target.

The out-of-sample column was added after the 16:00 ET DTW lead was found to be a leak,
because the training-period numbers were doing more work in this document than they could
support. Every directional IC shrinks by roughly two-thirds once it is measured on data
the model did not see.

| feature | IC (train) | t | **IC (OOS)** | **t** | rank IC vs range | t |
|---|---|---|---|---|---|---|
| `vix_level` | +0.0225 | 5.14 | +0.0072 | 1.70 | **+0.613** | 177 |
| `log_rv4h` | +0.0191 | 4.37 | +0.0079 | 1.89 | **+0.726** | 242 |
| `vol_z` | +0.0190 | 4.26 | +0.0055 | 1.30 | **+0.500** | 129 |
| `vix_slope` | −0.0155 | −3.56 | −0.0097 | −2.31 | −0.519 | −139 |
| `range_pos_30m` | −0.0136 | −3.11 | −0.0079 | −1.87 | −0.024 | −5.5 |
| `bar_range` | −0.0135 | −3.08 | +0.0027 | 0.64 | +0.443 | 113 |
| `is_event_window` | +0.0114 | 2.60 | −0.0047 | −1.13 | +0.059 | 13.4 |
| `signed_jump` | −0.0096 | −2.20 | +0.0001 | 0.02 | −0.074 | −17.0 |
| `vol_surprise` | +0.0058 | 1.31 | +0.0094 | 2.23 | +0.364 | 87.9 |
| `range_pos_2h` | −0.0056 | −1.28 | +0.0028 | 0.66 | −0.065 | −15.0 |
| `tod_sin` | +0.0036 | 0.82 | +0.0003 | 0.06 | +0.094 | 21.6 |
| `seas_exp` | +0.0033 | 0.75 | −0.0010 | −0.23 | **+0.494** | 128 |
| `tod_cos` | −0.0019 | −0.43 | −0.0007 | −0.17 | −0.396 | −98.5 |
| `dtw_mag` | −0.0008 | −0.15 | +0.0002 | 0.04 | +0.328 | 61.4 |
| `above_ma100` | −0.0164 | −3.60 | −0.0020 | −0.46 | −0.381 | −90.2 |
| `dtw_anchor` (rejected) | | | −0.0007 | −0.16 | | |

The pattern in that table is the main finding of the project. No directional IC reaches
0.023 in training and none reaches 0.010 out of sample. Seven magnitude ICs exceed 0.36,
and the largest is 0.73.

**The out-of-sample column is not a ranking of signal strength.** On 56,522 common rows
the largest |t| across the sixteen features is 2.31. Simulating sixteen pure-noise
features against the same returns gives a best-of-sixteen |t| with median 2.03 and 95th
percentile 2.96, so P(best-of-sixteen ≥ 2.31) = 0.288. Nothing in the set separates from
noise, which is the same conclusion the ridge reaches mechanically when it selects a
penalty above 42,000 and goes long on 82 percent of bars.

### Range position

Where price closed inside its recent high-low range, `(close - low) / (high - low) - 0.5`,
over one bar and four bars. A close-to-close return cannot distinguish a smooth rise
from a spike that faded, but the high and low can. Closing at the top of the range
means buyers pushed into the close, and that tends to revert over the following half
hour. The measure is bounded and self-normalising, so it needs no volatility scaling.

Two horizons only. Controlling for `vol_z` and `above_ma100`, the partial t-statistic
falls from −2.53 at 30 minutes to −1.63 at one hour and −0.94 at two hours.

**These replaced return-based momentum**, which was tried first at 1h/4h/24h and then
at 4h/24h/1 week. Momentum looked promising univariately, with the IC strengthening
monotonically as the lookback grew (t = −2.61 at one week). It did not survive
controls. Once `vol_z` and `above_ma100` are in the model, the partial |t| of a
close-to-close return never exceeds 0.96 at any horizon from 30 minutes to a week. The
long-horizon signal was the volatility regime counted a second time: `mom_1w`
correlates −0.37 with `vol_z` and +0.27 with `above_ma100`.

### Volatility

An EWMA of bar returns on a 24-hour halflife, then `log(vol)` through a walk-forward
median/MAD z-score against the trailing year. The halflife was chosen by QLIKE on the
training period, which prefers 24 hours over 12 and 48 and over a flat 48-bar window.
One trading day is a complete global cycle, so the estimate does not depend on which
session a bar falls in. The current bar is included because it is known at the decision
point and carries only 1.4 percent of the estimate at this halflife.

`vol_z` is the only feature that survives every model and every reformulation, and its
relationship is to magnitude rather than direction.

### Moving average

Close against the 100-session average, built on session closes and shifted one session
so the current close never enters the average it is compared against. It enters as a
0/1 dummy. Price sits above the average 78.2 percent of the time over this sample,
which makes it closer to a regime label than a signal. It correlates −0.50 with
`vol_z`, since drawdowns are the volatile periods, and it loses its significance in the
multivariate fit for that reason.

### Event windows

A flag for whether one of five market-moving releases (CPI, NFP, FOMC, PPI, Retail
Sales) lands inside the half-hour being predicted. Release schedules are published weeks
ahead, so this is a calendar fact rather than a forecast. It covers 0.48 percent of bars.

This replaced a day-level flag over ten releases that covered 44 percent of bars. The
day-level version spread each release across the 46 other bars of its day, which diluted
a 2.8x effect into a 1.13x one. Measured in units of prevailing volatility, a flagged bar
carries 2.16 times the move of a typical bar and 2.77 times the average for its own time
of day.

Choosing the five required controlling for time of day. On raw numbers JOLTS and
Wholesale Trade look like 1.5x events, but both release at 10:00 and fall to 1.1x against
the 10:00 baseline, so their apparent size was the slot rather than the news. Against
their own slots the ranking is CPI 4.7x, NFP 3.6x, FOMC 3.1x, then PPI and Retail Sales
at 1.9x, and nothing else above 1.6x. FOMC is worth noting: it has no tradeable surprise,
since rate decisions are well telegraphed, but it is the third largest volatility event
in the calendar.

The magnitude IC of +0.043 understates the effect, because a binary firing on 0.5 percent
of bars has a low correlation ceiling regardless of how large its effect is. In a
regression of log scaled `|return|` on `seas_exp` and `vol_z` it enters at t = +8.5,
against t = +5.1 for the day-level flag it replaces.

### Intraday volatility seasonality

How large a move this half-hour usually brings, in units of the prevailing 24-hour
volatility. Session opens, closes and scheduled release times give the day a stable
shape, and a 24-hour volatility estimate cannot see it.

The expanding mean of `|return| / vol` at this slot across all earlier sessions.
Expanding rather than rolling: a 250-session rolling version correlates +0.975 with it
and scores the same, and the expanding version has no window to justify.

**IC +0.0033 (t 0.75). Against |ret|: +0.2424 (t 56.2).** It correlates −0.016 with
`vol_z` and carries a VIF of 1.004, so it is close to orthogonal to everything else.
That is the point of it: `vol_z` measures the level of volatility and this measures the
time-of-day multiplier on it, so the two multiply rather than overlap.

### Dynamic time warping

The idea was that price paths repeat, and that finding past windows whose shape
resembled the current one would say something about what follows. DTW was chosen over
correlation because its elastic alignment tolerates timing shifts.

Eight designs were built and one works. It predicts magnitude; the seven that target
direction all failed. The first two pooled candidates by volatility regime and by clock
slot, and both produced ICs indistinguishable from zero. The third, in
`dtw_buckets.py`, pooled by intraday bucket on the idea that today's opening half-hour
should resemble other days' opens, with the query running from the bucket start to the
decision point and the forward return read from wherever the match ended. Thirty-four
configurations were swept across bucket definition, alignment tolerance, lookback, k,
resolution and path normalisation, each scored against a null built by shuffling
outcomes. The best reached |IC| 0.0167 against a null whose median best was 0.0144,
putting it at the 65th percentile of its own null.

The fourth pointed DTW at magnitude instead of direction, which looked promising at
IC +0.26 until the control was run. Removing the DTW selection entirely scored **higher**
at +0.38, and demeaning by slot collapsed both to +0.004. The apparent result was
intraday volatility seasonality arriving through a DTW wrapper that was making it worse.

The matcher itself works. A query's nearest neighbour has 0.58 shape correlation against
0.00 for a random pool member, so DTW is finding genuinely similar paths; what has not
worked is turning those neighbours into a forecast.

**The design that works.** What the four failures had in common is that every one of
them z-normalised the path before matching, which discards amplitude. That is precisely
the information magnitude prediction needs, so the working version matches the
*volatility trajectory* instead: the last two hours of five-minute absolute returns
divided by prevailing volatility, a unitless series describing whether volatility is
building, decaying or spiking. A 24-hour EWMA cannot tell those apart, which is what
leaves room for the feature.

Each neighbour contributes its own forward Parkinson range, scaled by its volatility and
demeaned by time of day using training rows only, so seasonality cannot arrive through
the wrapper the way it did in the fourth design. Neighbours are weighted by inverse
distance and by recency with a 250-session halflife, must be at least a day old, and are
drawn only from strictly earlier calendar years. Pool 4,000, k 25, Sakoe-Chiba band 4.

Rank correlation with the Parkinson range is **+0.347**. With the signed return it is
+0.015, so it is kept out of `MODEL_FEATURES` and used only in the magnitude model.

**One correction is worth recording.** The first version of this feature ended its query
path at the bar labelled `t+30`. On a left-labelled five-minute grid that bar holds data
through `t+35`, so the path contained the first five minutes of the window being
predicted. It scored a 13.54% QLIKE improvement and a rank correlation of +0.424. Ending
the path at `t+25` instead gives the figures below. Roughly 45% of the original gain was
the leak. `feature_engineering.py` now asserts that the path ends before the target
window opens.

It passed the controls that killed the fourth design, both repeated 20 times. The
shuffled-outcome null keeps the DTW selection and permutes what the neighbours
contribute; the random-neighbour control keeps the pool and picks neighbours at random.

| window | baseline QLIKE | with `dtw_mag` | improvement | best of 40 controls |
|---|---|---|---|---|
| 2019-2020, untouched | 0.5116 | 0.4735 | **+7.44%** | +0.30% |
| 2021-2026 | 0.5058 | 0.4780 | **+5.51%** | +0.07% |

No control on either window exceeds +0.30%, against a real improvement above 5%. Partial
t-stat is +24.8 with the volatility level and the seasonal term already in the model, and
it correlates +0.34 with `seas_exp` without being explained by it.

The matcher was never the problem. A query's nearest neighbour has 0.58 shape correlation
against 0.00 for a random pool member, so DTW was always finding similar paths; what took
five attempts was asking it the right question.

**Direction was tried three more times and failed each time.** All three are written up
in `dtw_buckets.py`.

*Signed paths, three targets.* Scaled by prevailing volatility rather than z-scored per
window, against the forward return, the gap between upside and downside excursion inside
the window, and whether the high precedes the low. The last two exist because flexible
entry and exit can capture an excursion that never reaches the close. Between 10 and 32 of
40 controls beat the real feature in every cell, with z-scores from -1.0 to +0.8.

*Anchored paths.* The cumulative return from the most recent cash open or cash close,
pooled so that opens match opens and closes match closes. Anchoring beat a free-floating
trailing path in every cell, so the reference-point intuition points the right way, but
best-of-four against best-of-four controls still left 32 of 40 controls ahead on the
confirmation window. The scalar version of the same idea, drift from the anchor alone,
scores -0.0156 and +0.0001.

*Six normalisations.* Because dropping the z-normalisation is what made the magnitude
feature work, the same question was put to direction: `volscale`, `zscore`, `minmax`,
`rank`, `l2` and `detrend_z` on the anchored path. Best-of-six against best-of-six
controls leaves 24 of 40 controls ahead on the confirmation window. The sign flips settle
it, with `l2` at +0.0053 then -0.0074 across adjacent windows and `detrend_z` at -0.0026
then +0.0071. For magnitude, normalisation was the whole difference, because amplitude was
the discarded signal. For direction there is no equivalent, and no transform recovers what
is not there.

That is six directional DTW designs and no signal, alongside fifteen features and four
model families. The directional question in this project is answered.

### The 16:00 ET lead, and why it was a leak

An earlier version of this document carried one open lead. Splitting the anchored result
by session phase, bars labelled 16:00 ET predicting the 16:30-17:00 ET window scored an
IC of **+0.1557**, with none of 40 controls beating it, a split-half of +0.117 and +0.160,
and a sign-based +1.11 basis points per trade net of a quarter-tick at t = +4.29. It was
held at arm's length on four grounds — best of ten post-hoc subgroup tests, no
confirmation window available, 1,099 bars in one clock slot, unmodelled spreads — and
kept as something to test prospectively.

**It was none of those four things. It was the five-minute timing leak, again.**

The section had no code behind it. The anchored design, the phase split and the
normalisation sweep were all written up as prose and never committed, so the number could
not be re-run by anyone including its author. Rebuilding the feature from the description
settles it. The reconstruction reproduces the stated sample exactly at 1,099 bars, which
is the check that it is the same object, and then:

| | reconstructed | as reported |
|---|---|---|
| IC, 16:00 ET slot | **+0.0135** | +0.1557 |
| split-half, 2022-23 / 2024-26 | +0.0018 / +0.0010 | +0.117 / +0.160 |
| sign-based P&L | +0.087 bps, **t = +0.33** | +1.24 bps, t = +4.29 |

A 48-configuration sweep over normalisation, k, recency halflife and outcome z-scoring
reaches a best |IC| of 0.0845 with correct timing, and that is itself a best-of-48. No
setting gets near 0.1557.

Re-running the identical code with the query path ending at the bar labelled `t+30`
instead of `t+25` — the same leak documented above, which puts the first five minutes of
the target window inside the query — does:

```
correct (t+25):  IC +0.0360   halves -0.0340 / +0.0635   P&L +0.50 bps  t=+1.93
LEAKED  (t+30):  IC +0.2260   halves +0.2022 / +0.2372   P&L +1.82 bps  t=+7.12
as reported   :  IC +0.1557   halves +0.1170 / +0.1600   P&L +1.24 bps  t=+4.29
```

Every reported statistic is bracketed by those two rows and sits nearer the leaked one.
The levels matter less than the **structure**: a leak contaminates every bar uniformly, so
it produces two strongly positive, similar halves, which is what was reported. Correct
timing produces halves that flip sign, which is what noise looks like. The reported
"z = +7.0" for the controls is almost certainly the leaked P&L t-statistic of +7.12
transcribed into the wrong sentence.

The original code no longer exists, so this is inference rather than proof. What can be
said is that no correctly-timed configuration reproduces the claim and the leak reproduces
both its magnitude and its signature.

**The four caveats were the wrong ones.** Post-hoc selection and the small sample do not
explain an IC of 0.1557 — against a proper null a pure-noise predictor on these returns
reaches 0.089 at the 99.9th percentile. The caveat that mattered was the one treated as a
footnote: the 16:00 ET bar does not exist before 2021, so the confirmation window that
would have caught this was unavailable. The leak survived because the only out-of-sample
test capable of killing it could not be run.

Two further results from the reconstruction. Scored as a feature across all slots,
`dtw_anchor` has an out-of-sample IC of −0.0007 (t = −0.16), ranking 13th of 16; added to
the ridge it moves out-of-sample R-squared from −0.000904 to −0.000773 while IC moves the
wrong way, and its partial t-statistic is −0.167 at p = 0.87. And across 43 clock slots, a
slot's directional IC in 2022-2023 correlates **+0.06 (p = 0.69)** with its IC in
2024-2026, so slot-level directional results do not persist at all — the same answer
`intraday_analysis.py` already reached for slot-level P&L at −0.05.

**There is no open directional lead.** Six DTW designs, fifteen features and four model
families have been pointed at direction and none of them found anything.

## Model

Walk-forward ridge on an expanding window. Each fold trains on everything before its test
year and predicts that year, giving six folds from 2021 to 2026. Features are standardised
using training-fold statistics only, and the penalty is chosen by cross-validation inside
each training fold.

All fifteen features go in. Ridge rather than OLS because at fifteen features OLS
overfits: it scores an out-of-sample R-squared of −0.00164 against ridge's −0.00042, and
its predictions have twice the spread for no extra accuracy. Lasso and elastic net shrink
to a single non-zero coefficient and do worse still at −0.00086.

| estimator | features | OOS R² | IC | pred sd, bps |
|---|---|---|---|---|
| OLS | 6 | −0.00077 | −0.0041 | 0.368 |
| OLS | 15 | −0.00164 | −0.0044 | 0.562 |
| **ridge** | **15** | **−0.00042** | −0.0049 | 0.265 |
| lasso / elastic net | 15 | −0.00086 | −0.0182 | 0.232 |

A categorical Naive Bayes came first, with sign-aware binning of each continuous feature.
Its independence assumption turned out to matter: three correlated momentum horizons made
it double-count trend evidence, predicting a 7.4 point spread in up-rate across deciles
where 2.4 was delivered.

The expanding window was chosen over a rolling one to test whether accumulating history
helps. It does not: training data more than doubles across the folds with no improvement
in any metric.

**Results.** Out-of-sample R-squared is −0.00042 and IC is −0.0049 on 59,103 bars. The hit
rate is 0.4999 against 0.5033 for always-long. The cross-validated penalty lands between
42,000 and 245,000, which is severe shrinkage, and the consequence shows up in the
positions: the model is long on 82% of bars and the prediction standard deviation is 0.27
basis points against a realised 15.56. Ridge is choosing to say almost nothing, which is
the correct response to a feature set with no directional content.

## Backtest

### From prediction to share count

The ridge output is `pred`, a forecast of the next 30-minute **log return**. Position size
is proportional to it rather than binary:

```
sd       = expanding std of earlier predictions      # causal, no forward look
position = clip(pred / sd, -3, +3)                   # contracts, signed and fractional
```

One contract is a one-sigma prediction, so a weak signal produces a small position rather
than a full one. Mean absolute position is 0.71 contracts and 0.9% of bars reach the
three-contract cap, so the cap is not shaping the result.

Costs are **$3.125 per contract**, a quarter tick with fees excluded. That is optimistic
given ES quotes one tick wide, and is set low deliberately so the signal is visible before
execution dominates. There is no turnover gate. Returns are percentages of one contract's
notional (`close × $50`), so they compare directly to holding it.

### Results

| | total % | ann % | vol % | Sharpe | IR vs hold | max DD % |
|---|---|---|---|---|---|---|
| Sized (net) | 47.62 | 8.75 | 26.85 | 0.326 | **−0.31** | 59.45 |
| Sized (gross) | 66.76 | 12.27 | 26.85 | 0.457 | −0.10 | 56.28 |
| Binary (net) | 51.30 | 9.43 | 16.16 | 0.583 | **−0.54** | 27.83 |
| Binary (gross) | 68.83 | 12.65 | 16.16 | 0.783 | −0.16 | 27.71 |
| **Hold 1 ES** | **76.35** | **14.03** | 16.16 | **0.868** | n/a | **24.01** |

**Every book loses to buy-and-hold, and the information ratio says so directly.** IR is
the annualised mean of the excess return over holding one contract, divided by the
volatility of that excess. All four are negative. Even gross of all costs, the best book
is at −0.10: the strategy is not paying for its own turnover, it is failing before costs
are charged.

Sizing buys return with risk rather than skill. Against the binary rule it lifts net
return from 51.3% to 47.6%, which is worse, while raising volatility from 16.2% to 26.8%
and drawdown from 28% to 59%. Its only effect is leverage.

### Win rates

Two splits, because they answer different questions. Conditioning on the realised move
asks how often we were positioned correctly; conditioning on the position asks how often
a call paid. The baseline for a long is the up-rate and for a short the **down-rate**, not
one minus the up-rate, because 3.0% of bars close exactly flat and lose for either side.

| | value | baseline |
|---|---|---|
| market up, we were long | 82.79% | |
| market down, we were short | 17.81% | |
| long calls, n 48,186 | 50.67% | 50.33% |
| short calls, n 10,417 | 46.76% | 46.66% |

Both books sit within a third of a percentage point of their own baselines. The first
split reflects the 82% long bias the ridge shrinkage produces, not skill.

### Does a bigger prediction mean a better call?

If the sizing rule is to be justified, the model has to be right more often when it
predicts a larger move. Deciles of `|pred|`, with the hit rate measured against the
correct per-decile baseline, the up-rate for longs and the down-rate for shorts:

| decile | mean \|pred\|, bps | hit % | baseline % | edge, pp | mean \|actual\|, bps |
|---|---|---|---|---|---|
| 1 | 0.011 | 48.41 | 47.74 | +0.66 | 5.26 |
| 2 | 0.035 | 47.92 | 48.26 | −0.34 | 5.39 |
| 3 | 0.060 | 49.13 | 49.65 | −0.52 | 5.96 |
| 4 | 0.087 | 49.78 | 48.99 | +0.79 | 6.54 |
| 5 | 0.120 | 51.12 | 49.88 | +1.24 | 7.31 |
| 6 | 0.160 | 51.49 | 50.80 | +0.69 | 8.18 |
| 7 | 0.212 | 50.38 | 50.15 | +0.22 | 9.59 |
| 8 | 0.296 | 50.75 | 50.77 | −0.02 | 11.36 |
| 9 | 0.461 | 50.03 | 50.17 | −0.14 | 13.43 |
| 10 | 0.789 | 50.74 | 50.85 | −0.11 | 18.65 |

**No.** Regressing edge on decile gives a slope of **−0.032 percentage points per decile,
t = −0.49, p = 0.64**. If anything it drifts the wrong way. The raw hit rate does climb
from 48.4% to 50.7%, but the baseline climbs with it, because a large prediction mostly
identifies a bar where the market was more likely to rise anyway.

What the model *does* know is size: mean absolute realised move rises monotonically from
5.3 to 18.7 basis points across the deciles. That is the same magnitude skill visible
everywhere else in this project, arriving through a model that was asked for direction.

P&L-positive rate by decile, all four books:

| book | all | d1 | d5 | d10 | d10 − d1 |
|---|---|---|---|---|---|
| Sized (net) | 49.30 | 42.13 | 51.12 | 50.74 | +8.62 |
| Sized (gross) | 49.97 | 48.40 | 51.12 | 50.74 | +2.34 |
| Binary (net) | 49.97 | 48.40 | 51.12 | 50.74 | +2.34 |
| Binary (gross) | 49.97 | 48.40 | 51.12 | 50.74 | +2.34 |

The +8.62 for the sized net book is not skill. In the bottom decile the position is a
fraction of a contract, so gross P&L is small enough that a quarter tick flips its sign;
42.13% against 48.40% gross is the cost of trading a signal too weak to pay for itself.
The other three books are identical because they share the position's sign.

### Beta

| strategy | beta | alpha, bps | alpha t | R² |
|---|---|---|---|---|
| Sized (net) | **1.323** | −0.091 | −1.41 | **0.634** |
| Sized (gross) | 1.323 | −0.058 | −0.90 | 0.634 |
| Binary (net) | 0.863 | −0.025 | −0.77 | 0.745 |

**It is largely a market play.** Beta is 1.32 and 63% of the strategy's variance is the
index. Alpha is indistinguishable from zero, −6.3% annualised gross and −9.8% net,
neither significant.

The sizing rule's clearest effect is raising market exposure from beta 0.86 to 1.32, and
the headline return is that beta. It beat the benchmark in 2 years of 6, spread from
−21.6 points in 2023 to +12.9 in 2025, which at beta 1.3 is what an index position with
noise on top looks like.

### Year by year

| year | net % | hold % | Sharpe | vs hold |
|---|---|---|---|---|
| 2021 | 24.03 | 24.82 | 1.42 | −0.79 |
| 2022 | −28.86 | −12.93 | −0.56 | −15.93 |
| 2023 | 2.70 | 24.30 | 0.45 | −21.60 |
| 2024 | 7.11 | 12.63 | 0.87 | −5.52 |
| 2025 | 25.80 | 12.91 | 0.93 | +12.88 |
| 2026 | 16.84 | 14.62 | 2.19 | +2.23 |

Five of six years are positive and two beat the benchmark. 2022 is the year the sizing
rule does its damage: mean absolute position reaches 1.61 contracts into a falling market
and the drawdown is 59%.

## Intraday analysis

Volume peaks around the cash open and close, at 158,000 contracts in the 09:30 half-hour
and 166,000 into the 15:30 one, against 2,500 to 10,000 overnight. Volatility
tracks it closely, with a rank correlation of +0.85 and a five-fold range in return
standard deviation, from 6.2 basis points at 23:00 ET to 30.0 at 09:00.

Transaction costs do not follow. The switch rate sits between 25.9 and 33.4 percent in
every half-hour slot, and the six most expensive slots hold 14.8 percent of total cost
against 13.6 percent for a perfectly uniform split. The reason is mechanical: sign
changes are driven by `range_pos_30m`, which is bounded and self-normalising, so it
flips at the same rate in a dead overnight hour as at the open.

What does vary is how much the fixed hurdle bites. The 0.57 basis point reversal cost is
2.6 percent of a typical 09:00 move and 14.4 percent of a typical 23:00 one.

An intraday seasonality feature was considered and rejected on the evidence. Slot-level
P&L looks dispersed, but the spread of slot Sharpes is 1.08 times what pure noise
produces, and a slot's mean return in 2021 to 2023 correlates −0.05 with its mean return
in 2024 to 2026. Grouping slots into volatility terciles gives a cleaner-looking result
that reverses out of sample: the high-volatility tercile runs a Sharpe of −0.46 in the
first half and +0.59 in the second.

The intraday volatility profile itself is persistent, at +0.87 between the training and
test periods, which makes it usable as an input. Scaling positions by it does not help,
in either direction.

### Win rates

A win is a bar where position and realised move agree. The benchmark is the always-long
win rate in the same bucket rather than 50 percent, because the market drifts up and a
long position inherits that for free.

| section | n | model win % | hold win % | long win % | short win % |
|---|---|---|---|---|---|
| Asia 18-03 | 23,755 | 49.16 | 49.19 | 49.71 | 46.88 |
| Europe 03-08 | 13,200 | 50.60 | 50.62 | 51.20 | 47.42 |
| US pre 08-09:30 | 3,960 | 50.20 | 51.26 | 51.14 | 45.67 |
| US morning | 9,240 | 50.43 | 51.81 | 51.61 | 45.01 |
| US afternoon | 8,948 | 50.76 | 50.97 | 51.24 | 48.25 |

The model never beats always-long in any section. Long calls win 50.68 percent overall
against a 50.33 percent up-rate; short calls win 46.80 percent against a 46.66 percent
down-rate. Both are inside a third of a percentage point of their own baseline.

**Nothing here persists.** Correlating win rate across the first and second halves of the
sample gives +0.13 across the 23 hours and **−0.05** across the five sections. Whichever
hour looked good in 2021-23 carries no information about 2024-26, which is what a ranking
of noise looks like.

## Macro surprises

Everything above measures direction from price history and none of it works. This section
measures it from a different input: Bloomberg consensus against the actual print, for every
US release from 2010 to 2026. The workbook carries 28,159 rows, 20,941 of which have both a
consensus and an actual, giving 99 distinct releases with at least 60 observations since
2015. Work is in `macro_surprise.py`.

`is_event_window` knows that CPI is printing but not whether the print was hot or cold. A
signed surprise is the difference between a magnitude feature and a directional one.

### Entry timing decides everything

A regression of the release-window return on the surprise measures the **reaction**, not a
forecast. By the time the number is public the jump has happened. Sweeping the entry minute
with the exit fixed at `T+30` separates the tradeable part from the rest:

| entry | beta, bps per sigma | t | R² |
|---|---|---|---|
| **T+0** | **−28.99** | **−6.50** | **0.226** |
| T+1 | −5.64 | −2.68 | 0.047 |
| T+2 | −4.10 | −2.11 | 0.030 |
| T+5 | −3.42 | −1.98 | 0.026 |
| T+10 | −3.31 | −2.43 | 0.039 |

**Eighty-one percent of the CPI response is the jump between T+0 and T+1**, −23.35 bps at
t = −7.34. Entering at T+0 means transacting at the pre-release price with post-release
information, so that portion is not available. Any event study reporting an R² above 0.2 on
this data is reporting the jump.

What remains is real. The drift from T+1 to T+30 is −5.64 bps at t = −2.68 and decays
slowly, still −3.31 bps entering a full ten minutes late.

### The sweep, and what survives

All 99 releases were tested on two windows: `T+1` to `T+30`, the drift; and `T+30` to
`T+60`, the next clean bar and the only window that could enter `MODEL_FEATURES`. Each was
fitted on the full sample and split at 2020, so a release has to keep its sign and its
magnitude across a regime break.

| release | n | beta | t | ≤2020 | 2021+ | sign holds |
|---|---|---|---|---|---|---|
| **CPI MoM** | 147 | **−5.64** | **−2.68** | −4.01 | **−6.99** | yes |
| **Core CPI MoM** | 147 | **−4.45** | **−2.46** | −3.70 | **−5.25** | yes |
| Initial Jobless Claims | 663 | +3.05 | +3.08 | +3.16 | −2.90 | no |
| ISM Manufacturing | 167 | +4.52 | +2.11 | +6.14 | +0.03 | collapses |
| Retail Sales Advance | 156 | +2.61 | +1.95 | +3.60 | −0.02 | no |

Only the two CPI prints keep both sign and magnitude, and both are *stronger* in the
confirmation half. PPI, PCE, NFP, Unemployment, Average Hourly Earnings, GDP and the ISM
family give nothing.

**The largest raw t-statistic in the whole sweep is an artefact.** Initial Jobless Claims
scores t = +9.68 on the next-bar window across 663 observations. It is three data points:
the largest raw surprise is **115 sigma**, on 2020-03-26, the week claims printed above
three million against a consensus built on a 200,000 base. Winsorising at five sigma takes
it to +3.68 and dropping 2020 takes it to **+0.23**. Any feature built as a ratio to a
trailing scale needs this check.

### The null

Ninety-nine releases were tested to find two, so the sweep is judged against itself. Fifteen
releases clear |t| > 2 on the drift window against five expected, and six clear |t| > 2.5
against one, so there is more than chance in aggregate. But across all 99 a sign survives
the 2020 split **54 percent of the time**, which is a coin flip. Clearing |t| > 2 on the
full sample is not evidence by itself; the split is the filter that matters.

### The signal

Headline and core CPI publish together and measure the same thing, so they are averaged
into one inflation surprise rather than entered as two correlated regressors on 147
observations. The combined version beats either alone.

| | beta, bps per sigma | t | n |
|---|---|---|---|
| headline only | −5.64 | −2.68 | 147 |
| core only | −4.45 | −2.46 | 147 |
| **combined** | **−5.82** | **−2.78** | 147 |
| — ≤2020 | −4.99 | −2.64 | 83 |
| — 2021+ | −6.54 | −1.70 | 64 |

As a sign trade, short ES on a hot print and long on a cold one, entering 08:31 and flat at
09:00, net of a full tick:

| filter | n | hit % | gross bps | net bps | t | per year | ann % |
|---|---|---|---|---|---|---|---|
| all | 128 | 60.2 | +5.61 | +5.09 | +2.42 | 8 | +0.43 |
| **\|z\| > 0.5** | **86** | **67.4** | **+9.67** | **+9.15** | **+3.38** | **6** | **+0.52** |
| \|z\| > 1.0 | 49 | 73.5 | +12.86 | +12.34 | +2.94 | 3 | +0.40 |

The hit rate rises monotonically with the size of the surprise, which is the response a real
effect gives rather than a fitted threshold. At the half-sigma filter the split across 2020
is near-identical at 67.3 percent and 67.6 percent, and the confirmation half is the larger
in size, +15.53 bps against +5.84.

Against a project where no directional measurement has cleared a 51 percent hit rate, 67.4
percent at t = +3.38 is the first real one.

### Why it cannot go in `MODEL_FEATURES`

CPI prints at 08:30:00 ET, exactly on a bar boundary. The bar labelled 08:00 has its
decision point at 08:30, simultaneous with the release, so using the surprise to predict the
08:30-09:00 return would capture the jump — the same non-executable entry the lag sweep
rules out. The next genuinely clean bar is 09:00-09:30, and **the drift is gone by then**,
at t = +0.72 with the sign flipping across the split.

So it runs as an event-time overlay: position at 08:31 on the twelve CPI days a year, sized
by the surprise, flat at 09:00. It is uncorrelated with the main book, which runs at a beta
of 1.32, and does not compete for the same capital.

Three caveats on the record. The sample is 147 releases and 86 above the filter. The 0.5
threshold was chosen after seeing the gradient, though the gradient being monotone is what
makes it defensible rather than fitted. And 2021 onward alone is t = −1.70, so the
confirmation half supports the sign without establishing it independently.

## What was tried and did not work

| approach | outcome |
|---|---|
| Categorical Naive Bayes with binned features | probabilities overconfident by a factor of three; abandoned for a continuous model |
| Lasso and elastic net on 15 features | shrink to a single non-zero coefficient; OOS R² −0.00086 against ridge's −0.00042 |
| Return momentum, 1h/4h/24h and 4h/24h/1wk | partial \|t\| never above 0.96 after controlling for regime |
| EWMA-smoothed momentum | cross-horizon correlation rises from 0.52 to 0.82 with no measurable gain |
| Holding-period-aware gate | net −25.4 percent at Sharpe −0.29; the signal's sign does not persist over the holding period it assumes |
| Volatility-targeted sizing | strategy falls from 67.3 to 55.4 percent, and buy-and-hold from 64.3 to 46.8 |
| Conviction sizing by \|pred\| | gross nearly doubles to 97.5 percent, but turnover rises and net stays at −34.0 |
| Time-of-day position scaling | −92.5 percent scaling up in busy hours, −63.7 scaling down |
| Restricting to busy or quiet hours | net improves in proportion to trading removed; gross falls just as fast |
| Intraday seasonality feature | slot-level edge does not persist (corr −0.05 across halves) |
| Bucket-conditioned DTW, 34 configurations | best \|IC\| 0.0167 against a null whose median best is 0.0144 |
| DTW pointed at magnitude, first attempt | IC +0.26, but removing DTW scored +0.38 and demeaning by slot collapsed both to +0.004 |
| Anchored and normalised directional DTW | six designs; best-of-N inside its own null on both windows |
| The 16:00 ET anchored lead | reconstructed at IC +0.0135, not +0.1557; the reported figure was the `t+30` timing leak |
| `dtw_anchor` as a model feature | OOS IC −0.0007 (t −0.16), 13th of 16; partial t −0.167 in the ridge, p 0.87 |
| Slot-level directional results | a slot's IC in 2022-23 correlates +0.06 with its IC in 2024-26 across 43 slots |
| Macro surprises on 95 of 99 releases | PPI, PCE, NFP, Unemployment, AHE, GDP, ISM: sign flips or magnitude collapses across the 2020 split |
| Initial Jobless Claims drift, t +9.68 | three COVID observations; 115-sigma surprise on 2020-03-26, t +0.23 excluding 2020 |
| CPI surprise as a `MODEL_FEATURES` column | drift is consumed inside 30 minutes; nothing left in the next clean bar (t +0.72) |
| Raw price level as a feature | 0.97 correlated with calendar year; QLIKE worse on both windows |
| Prediction-proportional sizing | raises beta from 0.86 to 1.32; IR against hold −0.31 net, −0.10 gross |
| Cost-aware turnover gate | cut trading to 12 position changes in 5.5 years, converging on buy-and-hold |

## Where this points

No directional measurement in this project exceeds an IC of 0.023 in training, or 0.010
out of sample. Several magnitude measurements are two orders of magnitude larger:
`log_rv4h` at +0.73 rank correlation against the Parkinson range, `vix_level` at +0.61,
`vol_z` at +0.50, `seas_exp` at +0.49, and `|pred|` sorting realised absolute moves from
5.3 to 18.7 basis points across its deciles even though the model producing it was asked
for direction.

The directional side is not merely small, it is absent. Out of sample the best of sixteen
features reaches |t| = 2.31 where the best of sixteen pure-noise features has a median of
2.03, giving P = 0.288. Ranking these features by IC ranks noise.

**The exception points away from price history entirely.** The one directional result in
this project comes from a scheduled inflation release and a consensus forecast, not from
the shape of the tape. That is consistent with everything else here: price history sets the
size of the next move, and new information sets its sign. Six DTW designs and fifteen
features searched the first for the second.

The directional model is now fifteen features under a cross-validated ridge, which is the
best of four estimators tried, and it still returns an out-of-sample R-squared of
−0.00042 and an information ratio against buy-and-hold of −0.31. The penalty it selects is
large enough that it predicts long on 82 percent of bars. The model is telling us there is
nothing to say.

## The magnitude forecast

Target is the realised variance of the next 30-minute bar. A second target, the
Parkinson range variance, is used where the DTW feature is scored: if entry and exit can
fall anywhere inside the window rather than at its edges, the range is what an execution
can actually capture, and it is far less noisy, with a ratio of standard deviation to
mean of 2.55 against 4.00. Loss is QLIKE, standard for
variance forecasts because it tolerates the noise in a single squared return and
punishes under-forecasting asymmetrically. R-squared on log variance is reported next to
it as a symmetric measure of information. Walk-forward by year over 2021 to 2026, 53,458
bars.

| model | QLIKE | R² log |
|---|---|---|
| EWMA level only | 2.036 | 0.113 |
| HAR | 1.976 | 0.140 |
| GARCH(1,1) raw | 1.841 | 0.167 |
| seasonal naive | 1.648 | 0.260 |
| GARCH(1,1) seasonally adjusted | 1.514 | 0.256 |
| features, OLS on log variance | 1.625 | 0.266 |
| features, Gamma GLM | **1.470** | 0.264 |
| GARCH-seasonal + features | **1.421** | **0.274** |

**The fitting method mattered more than any feature.** Fitted by OLS on log variance the
model loses to seasonally adjusted GARCH, 1.625 against 1.514, while having the higher
R-squared. It carried more information and was worse calibrated. Gamma deviance with a
log link is exactly QLIKE, so refitting the same features as a Gamma GLM optimises the
loss being scored: QLIKE drops to 1.470 and R-squared barely moves. Diebold-Mariano on
the loss differential, Newey-West at one session of lags, gives t = −21.66 for that
change alone.

Against the benchmarks the model beats raw GARCH at t = −5.95 and seasonally adjusted
GARCH at t = −2.48. Combining the two is better than either, t = −6.55 against GARCH
alone, so the features carry information GARCH cannot reach. `is_event_window` is the
clearest case, since a scheduled release is a calendar fact rather than a function of
past returns.

**Most of it is the clock.** Starting from the volatility level times the time-of-day
curve, the HAR terms add 0.28 percent, the event window 3.07 percent, and the rest of
the features together 3.69 percent. The intraday volatility profile and the release
calendar are both public and both sit in the options surface already, so beating GARCH
is not the same as having an edge. The comparison that decides that is against implied
volatility, and it has not been run.

Ridge, Lasso and ElasticNet were tested on the same specification and none of them help.
Ridge matches OLS to four decimals because cross-validation selects an alpha near zero,
and the penalised fits are marginally worse. With 11 features and 53,458 rows the
coefficients are already precisely estimated. Lasso does drop `vol_z` outright, which is
informative rather than surprising: the seasonal term is the log of the level times the
squared time-of-day multiplier, so it already contains the level.

## Standard features added later

Six additions from the volatility literature, scored on the Parkinson target by Gamma GLM
and built up in order. Percentages are the marginal gain of each block on the previous.

| block | 2019-2020 | 2021-2026 |
|---|---|---|
| six original features | QLIKE 1.1067 | QLIKE 0.6351 |
| + volatility level and seasonal term | +53.8% | +20.4% |
| + HAR terms | -1.5% | +2.5% |
| + `dtw_mag` | +7.8% | +4.6% |
| + realised variance and volume | +6.9% | +10.9% |
| + VIX | +3.9% | +3.1% |
| + bar range and clock | +5.1% | +6.5% |
| **final** | **QLIKE 0.4063** | **QLIKE 0.3799** |

`log_rv4h` is realised variance over four hours computed from 1-minute squared returns,
and it is the largest single addition. The reason is unglamorous: `vol_z` estimates the
same quantity from 30-minute bars, and variance is estimated better by sampling more
finely. `signed_jump` is the realised semivariance skew of Barndorff-Nielsen, Kinnebrock
and Shephard; it is kept for completeness and contributes under 1%. `vol_surprise` is log
four-hour volume against the trailing median for that clock slot, on the standard result
that volume and volatility are driven jointly by information arrival.

`vix_level` and `vix_slope` come from `get_vix.py`, which pulls VIXCLS and VXVCLS from
FRED into `data/raw/`. They are merged as-of the 16:15 ET publication instant rather than
by date, so a morning bar cannot use a close published that afternoon; the VIX value in
use is a median 0.64 days old. Worth 3.9% and 3.1%, modest, but the only forward-looking
input in the set.

`bar_range`, the bar's own high-low range in volatility units, is nearly as valuable as
the realised-variance group and was already in the bar table. A cyclical clock encoding
adds a little on top of `seas_exp`.

**Raw price level is excluded on purpose.** Log price correlates -0.18 and -0.11 with the
absolute forward return on the two windows, which looks usable; over the full sample the
same correlation is -0.02, because the relationship is not stable. What is stable is its
0.97 correlation with the calendar year. In a walk-forward the test year always sits
outside the training range, so a model given price level extrapolates a trend it has no
reason to believe, and QLIKE gets worse on both windows. Price belongs in the model
relative to something, which is what `range_pos` and `above_ma100` already do. The
immediately preceding 30-minute return is excluded for a duller reason: it scores +0.16%
and -0.15%, and `range_pos_30m` already carries it in stationary form.

## Where to go next

Three directions, in the order they should be taken.

**Compare against implied volatility properly.** `vix_level` is in the model, but VIX is a
daily 30-day number and the real benchmark is the 30-minute variance the options market
prices. Databento carries CME options on ES. Pull the at-the-money 0DTE straddle at each
decision point, convert it, and regress realised variance on both that and this forecast.
If the forecast's coefficient is zero once implied is in the model, there is nothing to
trade and no execution work needs building.

**A longer forecast horizon.** Two or four hours instead of thirty minutes, so the
holding period comes from the forecast rather than being wrestled out of a turnover rule.

**FOMC at 14:00 ET.** The CPI overlay works but is six trades a year, and the obvious way to
add events is the one major release that does not land on a 30-minute boundary. A 14:00
statement puts the drift window inside a bar whose decision point comes after it, which is
the arrangement CPI fails. `feature_engineering.py` already ranks FOMC the third largest
volatility event in the calendar.

**Surprise-sized event windows.** `is_event_window` is binary and covers 0.48 percent of
bars. Replacing it with the signed surprise magnitude is a strict improvement for the
*magnitude* model even where the direction is not tradeable, and the data is now local.

**A directional signal, still missing.** Sizing without direction is not a strategy.
Six DTW designs, fifteen features and four model families have now been pointed at
direction without finding anything, so the next attempt should probably change the
instrument or the horizon rather than the estimator.

**A process note, from the 16:00 ET leak.** That result survived for one reason: it was
written up as prose with no committed code, so nobody — including whoever produced it —
could re-run it. Every number in this document should be reproducible from a notebook in
this folder. Where a result is a lead rather than a finding, the out-of-sample test that
would kill it should be named explicitly and, if it cannot be run, that should be the
headline caveat rather than the fourth one.
