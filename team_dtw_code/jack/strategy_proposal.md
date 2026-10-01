# Strategy proposal — the impact-screened event book

A systematic ES mini strategy at 30-minute frequency, traded around scheduled US data
releases, assembled from all four workstreams and fitted on **2016-2021** with
**2022-2026** held back and read once.

This proposal was commissioned as a DTW strategy: dynamic time warping supplying
direction, other features supplying magnitude, sized proportionally. Six new DTW designs
were built and tested to that specification. **None of them survives the split**, and the
detail matters enough that it is Section 2 rather than an appendix. What is proposed here
is the same architecture — impact-screened events, executable entry, proportional
volatility-scaled sizing — with the directional input coming from the consensus surprise,
and a stated, falsifiable condition under which DTW earns its place back.

Supporting evidence and the older record are in [`strategy_analysis.md`](./strategy_analysis.md).
Every number below is reproducible from the five scripts in the appendix.

Cost convention: **one ES tick ≈ 0.5 bp of notional**; a round trip crossing a tick each
way is charged at 1 bp on every trade. Any per-trade edge below that is not a trade.

---

## 1. The one-paragraph version

Scheduled releases are the only place in this project where direction has ever been
measurable, and the impact screen below is a clean, non-peeking way to decide which
releases those are. Entry must be at T+1 — at T the price on the tape predates the
information, which is the single correction that invalidated the strongest-looking
backtest anyone here produced. At T+1 the surprise is worth about **+0.4 bps a trade net
of a tick, identically on training and test data**, which is real, consistent, and too
small to deploy on its own. The one sizing result that replicates across both periods is
that the edge scales *with* volatility rather than against it: sizing proportional to
signal × volatility turns that +0.4 into **+4.30 bps in training and +2.44 out of
sample**, where inverse-volatility sizing is negative in both. And DTW, asked six
different ways, does not predict direction around these events — the best configuration
is negative in **all six training years** and positive in four of five test years, which
is a regime break or a coin, not an edge.

---

## 2. What happened to DTW

This section is first because it is the part of the brief that did not survive, and
because the six designs are cheap for the next person to re-run rather than repeat.

### 2.1 The six designs

All are built on Hoshea's event-game structure — anchors around a release, a fixed
forward window, neighbours restricted to the same clock slot — and all are scored on
2016-2021 / 2022-2026.

| # | design | what changed | confirmation result |
|---|---|---|---|
| 1 | Hoshea's winning panel, re-split | his `fomc_jolts_pce_claims` grid on the requested years | pooled IC **+0.1225**, but **+0.69 bps/game at t = 0.39** |
| 2 | impact universe, same-family pool | neighbours restricted to the same release | best-by-training cell **+0.07 bps net, t = 0.27** |
| 3 | impact universe, clock-slot pool | neighbours = any admitted release at the same slot | mean confirmation **t = −0.90**; worse than same-family |
| 4 | dense pool | trade the admitted ten, match against **all 65 families** | confirmation IC negative in 7 of 8 cells; select→confirm rank corr **−0.45** |
| 5 | reaction path, z-scored | path `[T−30, T+1]`, entry T+1 — the path contains the print | best cell **+0.56 bps net, t = 0.64** |
| 6 | reaction path, volatility-scaled | same, amplitude preserved instead of z-scored | mean confirmation **+1.21 bps gross, t = 0.24** |

### 2.2 Why the published PCE result does not generalise

Hoshea's grid found `fomc_jolts_pce_claims` beat `core_major` on every one of nine cells,
and Jack's holdout confirmed it at IC +0.12 to +0.15. Split by event on the requested
years, the whole thing is one release:

| event | share of panel | train IC | test IC | test bps | test t |
|---|---|---|---|---|---|
| **PCE** | 16% | +0.3457 | **+0.3917** | +3.76 | +1.89 |
| JOLTS | 16% | +0.0249 | −0.0364 | +1.14 | +0.97 |
| Initial Jobless Claims | 68% | +0.0470 | +0.0369 | −0.13 | −0.14 |

PCE is genuinely strong and Claims — two-thirds of the sample — is exactly nothing. The
mechanism is not "PCE paths repeat." PCE prints at 08:30 alongside 1,573 other releases,
and its neighbours are drawn from that dense library; Claims *is* the library and has no
signal of its own. So the natural hypothesis is that **pool density is the active
ingredient**, and that hypothesis is directly testable: admit high-impact releases for
trading but let the matcher draw neighbours from every release at the slot.

Design 4 is that test and it fails outright — negative confirmation IC in 7 of 8 cells,
and a select-to-confirm rank correlation of **−0.45**, meaning the configurations that
look best in training are the ones that do worst afterwards. Density is not the mechanism
either. The PCE result is specific to one pooling configuration and does not survive
being stated as a rule.

### 2.3 The structural reason the original design cannot use news

Every DTW game in this project ends its path at `T0+60` and opens the position there. For
the anchors that carry the sample, that is 30 to 120 minutes after the release — and
Jack's entry sweep shows the news drift is fully spent by T+30 (the next clean bar scores
t = +0.72). **The existing games enter after the tradeable window has closed.** That is
why blending a consensus surprise into them is flat at every weight tested, from
w = 1.0 to w = 0.0, on both halves of the split.

Designs 5 and 6 fix this: the path runs `[T−30, T+1]`, so it contains the announcement
reaction itself, and entry is at T+1 where the drift still has content. This is the best
DTW idea in the project — **all 16 configurations are positive on the test period**, mean
+1.1 bps gross — and it is still not enough:

| DTW-only book | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| net bps/trade | −2.89 | −1.92 | −0.82 | −2.03 | −0.48 | −0.62 | **+4.38** | +0.06 | +1.70 | +2.66 | −2.50 |

Negative in **every one of the six training years**, positive in four of the five test
years. Pooled that is +1.67 bps at t = 1.38, falling to +0.85 at t = 0.73 once 2022 is
removed, against a random-sign null that beats it 11.6% of the time. A signal that loses
money in all six years it was developed on cannot be proposed on the strength of the five
years it was checked against — that is the same selection error, run backwards.

### 2.4 DTW does not carry magnitude here either

The one DTW result that replicates in this repository is Jack's `dtw_mag`, at rank IC
+0.347 against the next bar's Parkinson range over 116,605 bars. **On event-window bars
specifically it is +0.016.** Whatever it knows about the size of the next move, it does
not know it around scheduled releases, which is where this strategy trades. The magnitude
inputs that do work on event bars are `vix_level` (+0.468) and `log_rv4h` (+0.457).

---

## 3. The model

Four rules. Nothing below was chosen after seeing a test-period P&L curve.

### 3.1 Universe: the impact screen

Once a year, for every US Bloomberg release carrying a published date and time:

1. Collapse release **lines** into **families** — lines sharing ≥80% of their release
   timestamps are one print, not several trades. 157 lines collapse to 65 families.
2. For each family with ≥30 prior releases, measure **marginal market impact** on history
   only:

   ```
   impact = mean |30-min move| over the family's release windows
          / mean |30-min move| at the same clock slot, on days when some OTHER
            family releases at that slot but this one does not
   ```

3. Admit the top 10.

**The control is the whole trick.** Ranking by the naive version — release window over
the same slot on non-release days — scores CPI 1.88, PPI 1.83, Retail Sales 1.76 and
**Trade Balance 1.75**, all within noise of each other, because it credits every 08:30
release with the fact that 08:30 on a release day is a volatile minute. Minor releases
ride the majors. Asking instead "given something is printing at 08:30, does it matter
that it is *this*?" separates them properly. This uses no return data from the test period
and no P&L at all.

Today's admitted list, on 2016-2021:

| # | family | clock | impact | t | economic content |
|---|---|---|---|---|---|
| 1 | FOMC Rate Decision | 14:00 | 1.728 | 2.05 | policy |
| 2 | Average Hourly Earnings | 08:30 | 1.564 | **3.78** | wages — prints with NFP |
| 3 | S&P CoreLogic CS 20-City | 09:00 | 1.483 | 0.26 | house prices |
| 4 | CPI Index NSA | 08:30 | 1.298 | **3.22** | the policy-relevant inflation print |
| 5 | ISM Services | 10:00 | 1.271 | 1.36 | survey, services |
| 6 | PPI Ex Food and Energy | 08:30 | 1.261 | 0.66 | producer inflation |
| 7 | Core PCE Price Index QoQ | 08:30 | 1.235 | 2.25 | inflation — prints with GDP |
| 8 | JOLTS Job Openings | 10:00 | 1.232 | 0.83 | labour demand |
| 9 | Business Inventories | 10:00 | 1.231 | 0.66 | inventories |
| 10 | Retail Sales Advance | 08:30 | 1.209 | 1.64 | consumption |

Two honest notes. Only four of the ten clear |t| = 2 on their own impact statistic, so the
bottom of the list is weakly separated from the eleventh — the screen is a *ranking* rule,
and K = 10 was your specification rather than a discovered optimum. And ranks 3, 8 and 9
are releases nobody would name as market-moving; they survive because the measure is
mechanical, which is the price of not hand-picking.

### 3.2 Direction

The **causally standardised Bloomberg surprise**: actual minus consensus, scaled by the
expanding standard deviation of prior surprises for that line, shifted one release so the
scale never sees the current print, winsorised at ±5σ, and averaged across the lines in a
family.

Its sign comes from the **training-period jump regression**, never from drift P&L. If a hot
print has historically moved ES down, a hot print is a short.

### 3.3 Timing

- **Entry T+1.** Not T. At T the tape price predates the release, and transacting there
  means using the print to trade at a price that came before it. Rebuilt with a one-minute
  lag, Aaryen's pipeline falls from Sharpe 2.17 to 0.29 and from +8.22 to +0.33 bps.
- **Exit T+31.** A flat 30-minute hold, matching the 30-minute decision frequency.
  Inherited rather than optimised, deliberately.
- **One position per timestamp.** Simultaneous admitted families average into one position.

### 3.4 Sizing — the one place the evidence is strong

**Position = clip(signal / σ_signal, −3, +3) × (prevailing volatility / its median).**

Scale *up* in high volatility, not down. This is the only component that replicates
cleanly across the split:

| sizing | train 16-21 | TEST 22-26 |
|---|---|---|
| fixed sign | −0.84 bps, Sharpe −0.61 | +0.51 bps, Sharpe 0.33 |
| proportional to signal | +0.35, Sharpe 0.22 | +0.46, Sharpe 0.31 |
| inverse-volatility | **−0.54, Sharpe −0.61** | **−0.61, Sharpe −0.49** |
| proportional ÷ volatility | +0.19 | −0.31 |
| **proportional × volatility** | **+4.30, Sharpe 0.49** | **+2.44, Sharpe 0.85** |

Inverse-volatility sizing — the textbook default — is negative in **both** periods. The
per-trade edge is roughly proportional to volatility, so scaling down in volatile regimes
discards precisely the trades that pay. Jack reached the same conclusion independently on
a different book (Sharpe 1.18 ungated → 1.08 vol-sized), which makes this two
confirmations on non-overlapping evidence.

### 3.5 Expected shape

About **100 trades a year**, 30 minutes of exposure each, so capital is deployed under
0.6% of the time. Beta to the index is near zero. On the test period the sized book
returns of the order of **240 bps a year on one contract's notional** at Sharpe 0.85.

---

## 4. The features

Five inputs. Every one is known before the entry instant.

| feature | definition | role |
|---|---|---|
| `family_id` | release lines sharing ≥80% of timestamps | collapses duplicate views of one print into one trade |
| `impact` | release-window move ÷ same-slot-other-release move, on history only | universe admission |
| `surprise_z` | causally standardised, winsorised consensus surprise | direction |
| `jump_sign` | sign of the training-period jump regression, per family | trade direction |
| `vol_4h` | standard deviation of 1-minute log returns over the 4 hours before the print | sizing multiplier |

**Deliberately absent:** the DTW path factor (Section 2), momentum, seasonality flags, and
volatility-*state* interactions — the cleanest of which, NFP drift being stronger in low
volatility, is clean through 2020 and then vanishes (t −4.62 → +0.17).

---

## 5. Where each component came from

| component | source | what was taken |
|---|---|---|
| **The event-anchor unit** | **Hoshea** | The structural contribution. His game grid established that the scheduled release, not the calendar bar, is the right unit — every strategy here that trades all bars loses to buy-and-hold, and every result that survives is anchored to a print. His grid also showed hyperparameter rankings can generalise (+0.61 rank correlation) where event rankings do not. |
| **The DTW designs and their falsification** | **Hoshea**, extended | Designs 1-6 are his construction re-pooled, re-anchored and re-split. The negative result is his method used properly, and it is worth more than the original positive because it is stated as a rule and tested as one. |
| **The surprise construction** | **Reza** and **Aaryen**, independently | Both built causally standardised surprises from the same 28,159-row workbook. Reza's expanding-window standardisation and Aaryen's 5σ winsorisation are both in the definition. The winsorisation is not cosmetic — the March 2020 claims surprise is over 100σ. |
| **The direction sign** | **Reza** | His jump regressions are exactly the right measurement for which way a surprise trades, and using the *sign* rather than the R² keeps it free of drift P&L. |
| **The walk-forward protocol** | **Aaryen** | Expanding-window retrain, training-only statistics, minimum history before a model is trusted, and an explicit selection-honesty statement. |
| **The entry-lag correction** | **Jack** | The sweep separating the non-executable jump from the tradeable drift, and the finding that 81% of the CPI response is spent between T+0 and T+1. |
| **Cost discipline and null testing** | **Jack** | One-tick costs, cluster-robust errors by release date, and best-of-N nulls — which closed the 16:00 ET lead and the NFP drift, and which give this book P = 0.116. |
| **Volatility-proportional sizing** | **Jack**, confirmed here | His finding that vol-scaling *hurt* the news book is reproduced on a different universe and a different signal. |

---

## 6. What would put DTW back in

Stated in advance so it is a test rather than a search:

1. **The reaction path needs a positive training period.** Designs 5 and 6 are positive on
   2022-2026 and negative on 2016-2021. If re-running them on 2013-2015 — which needs a
   Databento re-pull, since only 20-37% of 2010-2012 weekdays have a complete session —
   makes the pre-2022 record positive, the sign flip is a data-coverage artefact and the
   factor is live. If it stays negative, it is closed.
2. **A best-of-N null must be cleared.** Six designs × sixteen configurations is 96 cells.
   Any future cell has to beat a best-of-96 null on the same returns, not a t-statistic.
3. **DTW is already earning its keep elsewhere.** `dtw_mag` at rank IC +0.347 on ordinary
   bars is the largest DTW effect anyone here has produced. It belongs to the magnitude
   product, not to this book, and it should be developed there rather than forced into a
   directional role it has now failed in nine separate tests.

---

## 7. Risks, and what would kill it

1. **The directional edge is not significant.** +0.4 bps a trade unsized, P = 0.116
   against a random-sign null on the sized book. This is a pilot, not a deployment.
2. **Most of the test-period return is 2022.** The book is +4.71 bps in 2022 and negative
   in 2023 and 2025. High consensus dispersion is the regime this works in.
3. **Execution is unmodelled.** T+1 after a scheduled print is the widest spread of the
   day. A half-tick of extra slippage removes a large fraction of a 2.4 bp edge, and it is
   the single most likely reason a live book underperforms.
4. **Volatility-proportional sizing is procyclical by construction.** It puts on the most
   risk exactly when the market is most volatile. The ×3 cap is doing real work and the
   drawdown behaviour of this sizing rule has not been stress-tested.
5. **Three of the ten admitted families have no plausible economic story.** If the screen
   is mostly ranking noise below rank 4, the universe is thinner than it looks.

**Pre-registered kill criteria:** stop if the trailing 100 trades are net negative after
costs; if realised slippage exceeds one tick per side; or if the admitted basket's
trailing impact score falls below 1.15 for the releases carrying most of the risk.

---

## 8. What to do next, in order

1. **Paper-trade for one quarter and record fills, not signals.** Slippage is the one
   input no amount of history answers, and at 2.4 bps it is decisive.
2. **Re-pull 2010-2015 from Databento.** It is the single highest-value data task: it
   nearly doubles the training window, and it is what would settle whether the DTW sign
   flip in Section 2.3 is real or a coverage artefact.
3. **Run the implied-volatility regression on ES options.** The magnitude forecast
   (QLIKE 1.421 against 1.841 for raw GARCH, Diebold-Mariano t = −6.55) is the largest and
   most stable effect anyone here has produced, and the only question that matters about it
   — does it beat implied, or is it already priced — is one afternoon's work once the data
   exists. If it beats implied it is worth more than this proposal.
4. **Build the evaluation harness.** Ten directional architectures have now been tested out
   of sample across three rounds and nine are dead. The binding constraint is not ideas, it
   is that each one takes a week to kill.

---

## Appendix — reproduction

Inputs are `data/processed/es_1min_clean.parquet`,
`data/processed/macro_event_calendar_expanded_2010_2026.parquet`,
`data/raw/Bloomberg Economic Releases.xlsx` and `jack/data/features.parquet`.

| script | what it produces | runtime |
|---|---|---|
| `jack/dtw_impact_book.py` | the family grouping, the impact screen, and the DTW grid over the admitted universe under two pooling rules | ~12 min |
| `jack/dtw_dense_pool.py` | designs 4 — trade the admitted ten, match against all 65 families | ~10 min |
| `jack/dtw_reaction_path.py` | designs 5 and 6 — path `[T−PRE, T+1]`, entry T+1, two normalisations | ~15 min |
| `jack/reaction_book.py` | the blend sweep, the sizing table, per-year P&L and the random-sign null | seconds |
| `jack/dtw_clockpool_research.py` | clock-slot pooling on the 13-event calendar (superseded by `dtw_impact_book.py`) | ~40 min |

Derived outputs land in `jack/data/` (gitignored): `impact_screen.csv`,
`dtw_impact_panel.parquet`, `dtw_impact_grid.csv`, `dtw_dense_panel.parquet`,
`dtw_dense_grid.csv`, `dtw_reaction_panel.parquet`, `dtw_reaction_grid.csv`.

### Scoreboard — directional architectures tested out of sample

| architecture | verdict |
|---|---|
| 15 features × 4 estimators on 30-min bars | dead — OOS R² −0.00042, IR vs hold −0.31 |
| six directional DTW designs (Jack, round one) | dead — all inside their own nulls |
| the 16:00 ET anchored lead | dead — a five-minute timing leak |
| Aaryen's event pipeline | dead at any honest entry lag |
| NFP ten-minute drift | dead — P(best-of-32 null ≥ real) = 0.69 |
| macro state interactions | dead — none survives outlier or period splits |
| DTW factor, event-agnostic over 13 events | dead — cross-event persistence −0.31 |
| post-release breakout continuation | dead — negative at every threshold |
| **DTW on the impact universe (designs 2-4)** | **dead — +0.07 bps net; dense-pool select→confirm rank corr −0.45** |
| **DTW reaction path (designs 5-6)** | **not established — negative in all six training years, P = 0.116** |
| reaction-screened news drift | marginal — +4.25 bps, Sharpe 0.71, t 1.49 OOS |
| **impact-screened news drift, vol-proportional** | **marginal — +2.44 bps, Sharpe 0.85, P = 0.116** |
