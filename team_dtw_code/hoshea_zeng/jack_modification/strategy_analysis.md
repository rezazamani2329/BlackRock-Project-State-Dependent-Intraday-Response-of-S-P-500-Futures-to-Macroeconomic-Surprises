# Cross-team strategy analysis

Four workstreams, four folders, one question: is there a systematic trade in ES mini?

This note summarises what each person has found, separates the results that survive an
independent check from the ones that do not, reports the three follow-up actions that were
run, and ends with **three strategy propositions to choose between**. Every number attributed
to a teammate is their own; every number I add is reproducible from the scripts listed in the
appendix.

**Round three re-ran the DTW question on 2016-2021 / 2022-2026 with an impact-screened
universe — see [Round three](#round-three-the-dtw-brief-on-2016-2021--2022-2026). Four more
DTW designs died; volatility-proportional sizing replicated.**

**Jump to [the three propositions](#three-propositions)** if you only want the decision.
The chosen one is written up in full in
[`strategy_proposal.md`](./strategy_proposal.md), including the event-family ranking and
the basket-size analysis.

Cost convention throughout: **one ES tick ≈ 0.5 bp of notional**, matching `jack/backtest.py`.
A round trip crossing a tick each way is ~1 bp. Any per-trade edge below that is not a trade.

---

## The one-paragraph version

Three independent lines of work converge on the same structure, and none of the four of us
said it out loud. **Price history predicts how big the next move is, not which way it goes.
Direction comes from scheduled information, and it is only available for a few minutes after
the print.** Jack's fifteen features and six DTW designs establish the first half on 59,103
bars. Hoshea's all-30-minute-anchor panel independently confirms it (IC −0.0027 on 24,406
rows). Reza and Aaryen both measure the second half, and both measure it at an entry time
that cannot be filled.

**Round two tested whether any of it generalises, and the answer reshaped the conclusion.**
The path signal fails when applied event-agnostically across all 13 releases, and per-event
performance *anti*-persists across time (rank correlation −0.31). What survives is a rule
rather than a signal: choosing which releases to trade by how strongly the market reacts to
them works out of sample, while choosing them by past profitability decays. That rule is
Proposition 1.

---

## Hoshea — DTW event games (`hoshea_zeng/`, merged to `main`)

**What was built.** Anchors at `T0 ∈ {T−60, T−30, T, T+30, T+60}` around each macro release.
The input is the 60-minute path `[T0, T0+60]` as z-scored log returns; the target is the
forward 30-minute return `[T0+60, T0+90]`. DTW with a Sakoe-Chiba band finds the k nearest
historical paths at the same clock time; the factor is their distance- and recency-weighted
mean forward return. Three experiments: mixed major-event pool (IC 0.070), single-event pool
(IC 0.011), and an 18-point grid search (IC **0.116**, hit 51.5%, 4,749 games).

**The construction is clean.** I checked specifically for the two failure modes that have
already bitten this project. There is no timing leak — the path ends exactly where the label
begins, at `T0+60`. There is no lookahead in neighbour selection — history is `h < pos` and
inside the lookback. The grid is also stable rather than knife-edge: all nine
`fomc_jolts_pce_claims` combinations beat all nine `core_major` ones, spanning IC 0.093–0.116,
so the headline is not one lucky cell.

**But the aggregate IC hides where the signal is.** Splitting the winning panel by event type:

| event | n | Pearson IC | hit | bps/game | t (clustered by release) |
|---|---|---|---|---|---|
| **PCE** | 769 | **+0.3162** | **57.1%** | **+3.12** | **+3.20** |
| JOLTS | 579 | +0.0270 | 51.8% | +1.32 | +1.48 |
| Initial Jobless Claims | 3,401 | +0.0691 | 50.2% | +0.20 | +0.47 |
| FOMC | 0 | — | — | — | — |

Three things follow. **Jobless Claims is 72% of the sample and contributes nothing** — 50.2%
hit rate, t = +0.47 — so the pooled IC of 0.116 describes a panel that is mostly noise with a
strong signal buried in it. **FOMC does not appear in the winning panel at all**: at a 2-year
lookback, the 14:00 ET clock slots never accumulate the 20 prior neighbours `MIN_HISTORY`
requires, so the README's FOMC narrative from Experiments A/B does not carry into Experiment C.
And the reason the `fomc_jolts_pce_claims` group beats `core_major` is not economic — PCE is in
*both* groups. What changes is that PCE's 08:30 neighbours become weekly Claims paths instead
of monthly CPI/NFP ones, so the matching pool is four times denser. **The grid selected for
pool density, not event economics.**

**The PCE result itself holds up.** Clustered by release date, across every split I tried:

| split | n | dates | gross bps | t | net of a tick |
|---|---|---|---|---|---|
| all | 769 | 157 | +3.12 | +3.20 | +2.62 |
| excl 2020 | 709 | 145 | +2.37 | +2.61 | +1.87 |
| excl 2020 & 2026 | 674 | 138 | +1.95 | +2.18 | +1.45 |
| 2013–2019 | 388 | 80 | +2.28 | +2.94 | +1.78 |
| 2020–2026 | 381 | 77 | +3.98 | +2.22 | +3.48 |

Positive in both halves, survives dropping the two outlier years, and clears costs by 3–5x.
That is the strongest directional evidence in the repository after Jack's CPI overlay, and
unlike CPI it comes from price shape rather than a consensus forecast — so the two are
independent inputs.

**Two things to fix.** The 18-point grid was scored on the full sample with no holdout, so
0.116 is a best-of-18 in-sample number; re-score on 2013–2021 and confirm on 2022–2026.
And the all-30-minute-anchor extension should be dropped: over 24,406 rows it returns
**IC −0.0027, hit rate 49.19%, and a sign trade of exactly −0.000 bps (t = −0.00)**. The factor
carries nothing away from event windows — which is a useful negative result, and an exact
independent replication of Jack's six failed directional DTW designs.

**One bonus finding.** Quintiles of `|dtw_factor|` sort the realised absolute move from 14.30
to 19.72 bps. Like everything else in this project, it knows size better than sign.

---

## Aaryen — surprise → Gaussian NB event backtest (`origin/aaryen`)

**What was built.** Bloomberg consensus surprises for CPI/NFP/PCE/GDP/FOMC, winsorised at
5σ, plus a same-event-type DTW feature, a momentum composite and four seasonality flags.
A separate Gaussian NB per event type over 5 ordinal return buckets, retrained before every
game on an expanding window, posterior converted to an expected return, position =
`clip(E[r] / (1.5 · train σ), −1, 1)`. Four window lengths compared out of sample; 10 minutes
selected. Headline: **507 trades, Sharpe 2.14, hit rate 63.3%, max drawdown −1.03%**, and a
held-out last 1.5 years at Sharpe 2.58.

**The walk-forward machinery is the best-engineered in the repo.** Quintile bin edges from
training data only, expanding retrain before every single game, per-class statistics from
training rows only, minimum 30 training games, emergency FOMC meetings excluded. I looked
hard for a lookahead in the labelling and there isn't one. The selection-honesty markdown at
the top of notebook 04 is exactly the right instinct.

**The problem is the fill, not the model.** `ENTRY_LAG_MINUTES = 0`, and entry price is the
open of the 1-minute bar at the release timestamp. CPI prints at 08:30:00; the open of the
08:30 bar is the *pre-release* price. So the backtest buys at the pre-print price using the
print. There are also no transaction costs anywhere in notebooks 03 or 04.

The tell is already in the results: Sharpe falls monotonically with holding window —
**2.14 / 1.98 / 1.51 / 0.94** for 10 / 30 / 60 / 120 minutes. An edge that is largest at the
shortest horizon and decays smoothly to nothing is an instantaneous jump, not a drift.

Replacing the NB with a plain sign rule on the same releases, the same entry/exit convention,
and the same 2013-04-01 cutoff, and sweeping only the entry lag:

| release | window | lag 0 | lag 1 | lag 2 |
|---|---|---|---|---|
| CPI MoM | 10 min | **+30.75 bps, 70.8%, t 4.89** | +4.42, 53.1%, t 1.94 | +2.17, 55.2%, t 1.09 |
| CPI MoM | 30 min | **+34.53 bps, 71.9%, t 5.04** | +7.92, 59.4%, t 2.81 | +4.94, 56.2%, t 1.94 |
| Core PCE MoM | 10 min | **+5.61 bps, 64.8%, t 2.90** | +1.44, 52.1%, t 1.02 | +1.90, 56.3%, t 1.68 |
| GDP Annualized QoQ | 10 min | **+2.93 bps, 52.8%, t 2.68** | +0.78, 51.2%, t 0.75 | −0.22, 48.0%, t −0.23 |
| Nonfarm Payrolls | 10 min | +4.38 bps, 59.6%, t 1.45 | **+3.04, 61.0%, t 2.12** | +3.06, 58.2%, t 2.05 |

A sign rule at lag 0 reproduces the scale of Aaryen's per-event numbers (CPI 70.8% here vs
78.1% reported with sizing). One minute of lag removes 86% of the CPI edge and essentially
all of PCE's and GDP's. This is the same measurement Jack's `macro_surprise.py` reached
independently: 81% of the CPI response is the jump between T+0 and T+1.

**Two smaller notes.** The held-out period isn't untouched — the full walk-forward ran over
all data and was split retrospectively, and window selection saw 2025–26. And the reported
Sharpe treats each trade as deploying full notional for 10 minutes; the capital is idle 99.5%
of the time, so it is a per-trade Sharpe annualised by frequency, not a portfolio Sharpe.

**And one genuinely new lead.** NFP is the single release that *gains* from the lag: +3.04 bps
at 61.0% hit, t = +2.12 at lag 1, still +3.06 at lag 2, and it fades by lag 5. Jack's sweep
tested T+1→T+30 and concluded NFP gives nothing — which the 30-minute row here confirms
(+2.75, t = 1.24). The drift is real but it is spent inside ten minutes. Nobody has tested
that window. It is the cheapest open question in the repo.

---

## Reza — macro surprise event study (`Reza/`)

**What was built.** The cleanest data work of the four: 28,159 Bloomberg rows across 17 yearly
sheets, causally standardised surprises, VIF diagnostics, univariate robustness checks
alongside every multivariate fit, and explicit interpretation at each step. Headline results:

| event | horizon | beta (bps/σ) | t | R² |
|---|---|---|---|---|
| CPI headline | 1 min | −18.11 | −3.30 | 0.299 |
| CPI headline | 30 min | −23.77 | −2.92 | 0.252 |
| CPI headline | 60 min | −23.57 | −2.90 | 0.231 |
| NFP payrolls | 15 min | +3.02 | +1.87 | 0.050 |
| PCE headline | 1 min | −3.69 | −1.95 | 0.138 |

**What it establishes.** CPI dominates on both market impact (38.5 bps mean absolute move at
30 min vs 29.6 for NFP, 14.7 for PCE) and directional explanatory power. NFP is economically
ambiguous — growth and policy pull opposite ways — and the numbers say so, with R² between
0.03 and 0.07 and only average hourly earnings clearing |t| = 2 at any horizon. Core CPI adds
nothing once headline is in the regression. All of that is correct and worth keeping.

**The caveat is the same one as Aaryen's.** Returns are `log P(T+h−1) / P(T−1)` — the window
opens one minute *before* the release. Every R² in the table therefore includes the
announcement jump, which is why the 1-minute R² of 0.299 is the *highest* in the series and
decays with horizon. Jack's entry sweep puts the same regression at R² 0.226 entering at T+0
and 0.047 entering at T+1. As a **reaction study** the work is valid and well executed. It is
not a forecast, and the write-up should say so — the sentence "hotter inflation → ES lower" is
true and untradeable at the same time.

**The missing half is the title.** "State-Dependent Intraday Response" promises a regime
interaction and none is implemented — no VIX split, no volatility tercile, no
pre-release-positioning conditioner. That is the most valuable untested question in the whole
repo and Reza is the closest to being able to answer it.

---

## Jack — full-grid directional and magnitude models (`jack/`)

Summarised from `jack/README.md`; included here so the comparison is complete.

**Direction is dead, and it was measured properly.** Fifteen features, four estimators,
walk-forward ridge over 2021–2026: out-of-sample R² **−0.00042**, IC −0.0049, information
ratio against buy-and-hold **−0.31** net and −0.10 gross. The best of sixteen features reaches
|t| = 2.31 out of sample where the best of sixteen *pure-noise* features has a median of 2.03
(P = 0.288). Six directional DTW designs all failed against their own nulls, and the one
apparent success — the 16:00 ET anchored lead at IC 0.1557 — was reconstructed and shown to be
a five-minute timing leak (correct timing: IC +0.0135).

**Magnitude works, consistently.** Rank correlations against the next bar's Parkinson range:
`log_rv4h` +0.73, `vix_level` +0.61, `vol_z` +0.50, `seas_exp` +0.49, `dtw_mag` +0.33. The
combined forecast reaches QLIKE **1.421** against 1.841 for raw GARCH(1,1) and 1.514 for
seasonally adjusted GARCH, with Diebold-Mariano t = −6.55 against GARCH alone. The DTW feature
that works matches the *volatility trajectory* rather than the z-normalised price path —
dropping the normalisation is precisely what made it work, because amplitude was the signal.

**One directional result survives, from outside price history.** A combined CPI surprise
predicts the 08:31–09:00 return at −5.82 bps/σ, t = −2.78. As a sign trade above half a sigma:
86 trades, **67.4% hit, +9.15 bps net of a full tick, t = +3.38**, six trades a year. It runs as
an event-time overlay, not a feature, because CPI prints exactly on a 30-minute bar boundary.

---

## Where everyone agrees, and where the evidence actually is

| claim | evidence for | evidence against | verdict |
|---|---|---|---|
| Price history predicts **magnitude** | Jack: 5 features > 0.49 rank IC, QLIKE 1.421 vs 1.841; Hoshea: \|factor\| quintiles 14.3 → 19.7 bps | none | **Established** |
| Price history predicts **direction** on ordinary bars | — | Jack: 15 features, 6 DTW designs, OOS R² −0.0004; Hoshea all-30 panel: IC −0.003, 49.2% hit on 24,406 rows | **Refuted twice, independently** |
| CPI surprise → direction, after the print | Jack: 67.4% hit, +9.15 bps net, t +3.38; my sweep: +7.92 bps, t 2.81 at lag 1 | small n (86–96), 12 trades/yr | **Real, small** |
| PCE **path shape** → direction | Hoshea: +3.12 bps, t +3.20 clustered, both halves positive | in-sample grid selection | **Promising, needs a clean holdout** |
| NFP surprise → 10-minute drift | my sweep: +3.04 bps, 61.0%, t +2.12 at lag 1 | untested window, one of ~8 cells looked at | **Open lead, cheap to settle** |
| Event backtests at Sharpe > 2 | Aaryen: 2.14 full, 2.58 holdout | entry at T+0, no costs; 86% of CPI edge gone at lag 1 | **Not yet executable** |
| Macro event study R² ≈ 0.30 | Reza: CPI, 1 min | window opens at T−1, spans the jump | **Reaction, not forecast** |

---

## Assessment: the best idea, and one model

The best idea anyone has had is **Hoshea's event-anchored game structure**, and it is worth
more than the DTW factor sitting on top of it. Every backtest in this repo that trades all
bars loses to buy-and-hold; every result that survives is anchored to a scheduled release. The
game grid — anchors at fixed offsets around an event, a short fixed forward window, matching
restricted to the same clock slot — is the right unit of analysis, and it is the only piece of
infrastructure all four workstreams can share. Jack's 59,103-bar panel and Hoshea's own
24,406-row all-anchor panel are the control experiment proving it: outside event time, the
directional IC is −0.003 and the sign trade pays exactly zero.

Within that structure, three signals have independent evidence and should be combined rather
than competed. They are uncorrelated by construction, which is the whole argument for stacking
them: the CPI surprise is *news*, the PCE factor is *tape shape*, and the volatility forecast
is *size*.

**Layer 1 — the gate.** Trade only event games. Anchors at T−60…T+60 around each release in the
expanded calendar, 30-minute forward windows, entry no earlier than T+1. This throws away
99.5% of bars, which is the point: every cost-aware turnover gate Jack tried converged on
buy-and-hold precisely because it was gating noise instead of selecting information.

**Layer 2 — direction.** One pooled regression on event games with three inputs: the signed
standardised surprise (entered at lag ≥ 1 minute, so it is executable), Hoshea's DTW factor,
and event-type fixed effects. CPI carries the surprise term, PCE carries the DTW term, and the
two do not overlap. Score it by clustered t on bps-per-trade net of a tick, never by pooled
Pearson IC — the pooled IC is what let 3,401 worthless Claims games hide a real PCE signal.

**Layer 3 — size.** Jack's magnitude model turns the edge into contracts. Position =
(predicted bps) / (forecast σ), using the GARCH-seasonal + features forecast at QLIKE 1.421
rather than a trailing standard deviation. This is the only place the magnitude work becomes
P&L, and `is_event_window` is already one of its strongest terms because releases are calendar
facts. It is a free Sharpe improvement on a signal we already have.

> **This layer was tested and is wrong — see the results section below.** Volatility-scaled
> sizing lowers the book's Sharpe from 1.18 to 1.08 and volatility gating does not help either.
> The directional edge is roughly *proportional* to volatility, so a fixed contract count
> already captures it; scaling down in high volatility discards the best trades. The magnitude
> model is a separate product, not a sizing layer for this one.

**Be honest about the scale.** CPI is ~12 trades a year at ~9 bps net. PCE is ~55 game-trades
a year at ~2 bps net. NFP, if it confirms, is another ~12 at ~3 bps. Hoshea's extreme-quintile
overlay adds ~138 trades a year at ~1 bp. That is single-digit basis points annualised on one
contract's notional — a real edge, but one that only becomes interesting levered, and only if
execution inside the first minutes after a print is solved. Reporting it as "Sharpe 2.1" on
idle capital will not survive contact with anyone who trades.

**Three things to do first, in order.** (1) Re-run Aaryen's pipeline with
`ENTRY_LAG_MINUTES ∈ {1, 2, 3}` and a cost model — a one-line change that converts the
strongest-looking result in the repo into an honest one, and settles the NFP 10-minute lead at
the same time. (2) Re-score Hoshea's grid on 2013–2021 and confirm on 2022–2026, reporting per
event type rather than pooled. (3) Have Reza add the state interaction the title already
promises — whether the post-print drift is larger in high-VIX regimes is genuinely unknown, and
if it is, it multiplies every number above.

---

## Results of the three actions

All three were run. Scripts: `jack/aaryen_entry_lag.py`, `jack/nfp_drift_test.py`,
`jack/hoshea_grid_holdout.py`, `jack/state_dependence.py`, `jack/candidate_books.py`.

### 1. Aaryen's pipeline with an honest entry lag and costs

His pipeline was rebuilt end to end — same `games_master.parquet`, same 2013-04 cutoff, same
winsorisation, same DTW/momentum/seasonality features, same per-event Gaussian NB with
expanding retrain — with only the entry lag swept and costs charged.

**At lag 0 it replicates almost exactly**, which confirms the rebuild is faithful: Sharpe
2.17 / 2.00 / 1.68 / 0.95 at 10 / 30 / 60 / 120 minutes against his 2.14 / 1.98 / 1.51 / 0.94.

Portfolio Sharpe, net of a full tick:

| entry lag | 10 min | 30 min | 60 min | 120 min |
|---|---|---|---|---|
| **0** | **2.17** | **2.00** | **1.68** | **0.95** |
| 1 | 0.29 | 0.23 | −0.27 | −0.18 |
| 2 | −0.30 | 0.70 | −0.45 | 0.11 |
| 3 | 0.15 | 0.12 | −0.57 | −0.15 |

In bps per trade, the pooled book goes from **+8.22 to +0.33**. Per event at the 10-minute
window: CPI +20.65 → +1.67, NFP +11.77 → +0.62, PCE +3.68 → −0.16, GDP +1.26 → −0.10. Not
one of the sixteen lag ≥ 1 cells is materially positive. **The entire result was the
announcement jump.** The modelling was never the problem — the fill was.

### 1b. The NFP ten-minute lead is closed

The one cell that gained from lagging deserved its own test, and it fails it. Entry T+1,
exit +10 min: +3.04 bps, 61.0% hit, **t = +2.11**. But:

- the 2021+ half is **+0.83 bps at t = +0.38** — the effect is entirely pre-2020;
- the |z| gradient is not monotone in significance (t = 2.11 → 1.93 → 0.98 → 1.21);
- and it was the best of 32 cells. Against a best-of-32 null on the same returns, the median
  is **t = 2.33** and **P(best-of-32 ≥ 2.11) = 0.693**. The null beats it more often than not.

I raised this lead in the first version of this document. It is now closed. NFP has no
tradeable post-print drift.

### 2. Hoshea's grid re-scored on a real holdout — it survives

The full 18-combination grid was rebuilt and split: hyperparameters chosen on 2012-2021,
reported on 2022-2026. Because nothing in the factor is fitted, this is the clean test.

**The selection did not inflate the headline.** The cell chosen on 2012-2021 (freq 2 min,
2-year lookback) delivers a **confirmation IC of +0.1225**, and Hoshea's own full-sample pick
delivers **+0.1339** on the confirmation period. Across the 18 cells, selection IC and
confirmation IC have a **rank correlation of +0.61** — the hyperparameter ranking carries
genuine information rather than fitting noise. Every `fomc_jolts_pce_claims` cell confirms
between +0.099 and +0.155; every `core_major` cell confirms between −0.015 and +0.097.

**But the per-event split is the real story**, and it is decisive:

| event | period | n | IC | hit | bps/game | clustered t | net of a tick |
|---|---|---|---|---|---|---|---|
| **PCE** | select | 508 | +0.3029 | 58.9% | +3.54 | **+3.60** | +3.04 |
| **PCE** | confirm | 261 | **+0.3917** | 57.5% | +3.76 | +1.92 | +3.26 |
| JOLTS | select | 309 | +0.0178 | 53.7% | +3.38 | +3.09 | +2.88 |
| JOLTS | confirm | 270 | −0.0364 | 52.2% | +1.14 | +0.98 | +0.64 |
| Jobless Claims | select | 2,262 | +0.0657 | 49.6% | +0.12 | +0.25 | −0.38 |
| Jobless Claims | confirm | 1,130 | +0.0369 | 49.9% | −0.13 | −0.14 | −0.63 |
| FOMC | both | 0 | — | — | — | — | — |

PCE gets *stronger* out of sample. Claims is below a 50% hit rate in both halves despite being
two-thirds of the panel. JOLTS decays to nothing. **Hoshea has found a PCE signal, not a macro
event signal** — and that is a better result than the pooled number suggested, because it is
concentrated rather than diffuse.

> **Partly retracted.** This holds only inside the mixed-pool configuration. Round two runs the
> same factor with same-event matching across all 13 event types and PCE reverses sign out of
> sample. What survives from this item is the narrower and still useful finding that Hoshea's
> *hyperparameter* ranking generalises (+0.61 rank correlation) even though his *event* ranking
> does not (−0.50).

### 3. State dependence — measured, and mostly absent

Reza's regressions were rebuilt with the jump and the drift separated and the bar convention
fixed (opens, not closes — bars are left-labelled, so a close at index T already covers
[T, T+1) and silently shifts every window by a minute).

| family | n | jump β | t | R² | drift β | t | R² |
|---|---|---|---|---|---|---|---|
| CPI | 128 | −23.89 | −4.45 | 0.292 | **−6.46** | **−2.21** | 0.072 |
| NFP | 151 | +6.41 | +3.46 | 0.044 | +2.47 | +0.90 | 0.009 |
| PCE | 77 | −1.71 | −1.85 | 0.053 | −0.29 | −0.18 | 0.001 |

**CPI is the only release whose surprise predicts anything after the print.** NFP and PCE
drift betas are statistically zero. That retrospectively explains Aaryen's and Reza's per-event
numbers for those two: they were jump, in full.

Four states were tested — VIX, pre-release 4-hour realised vol, the VIX3M/VIX slope, and the
signed over/under-reaction residual — as interactions on the drift. Several clear |t| > 3, and
**none survives robustness**:

| interaction | all | drop 3 largest \|drift\| | ≤2020 | 2021+ |
|---|---|---|---|---|
| NFP × rv_4h | t −4.62 | t −5.16 | t −4.84 | **t +0.17** |
| NFP × VIX | t −3.47 | t −3.74 | t −3.44 | **t +0.18** |
| PCE × rv_4h | t +9.96 | **t +2.83** | t +2.18 | t +8.76 |
| CPI × reaction | t −5.09 | **t −2.71** | **t −1.50** | t −4.68 |

The NFP pattern (drift stronger when pre-release vol is low) is clean through 2020 and then
vanishes outright. PCE's t = 9.96 with R² = 0.60 on n = 77 is three observations. **There is no
usable state interaction.** The valuable output of this item is the jump/drift table above,
not the interactions.

### The surprise finding: magnitude does not size direction

I proposed using the magnitude forecast to size the directional book. Tested on the combined
book, split by pre-trade 4-hour realised vol:

| vol tercile | n | bps/trade | hit | t |
|---|---|---|---|---|
| low | 284 | +2.83 | 58.1% | +3.15 |
| mid | 284 | +0.98 | 58.1% | +0.87 |
| high | 284 | **+7.50** | 62.0% | +3.79 |

The edge is roughly proportional to volatility, so inverse-vol sizing actively hurts: Sharpe
falls from **1.18 ungated to 1.08 sized**. Gating to high-vol trades does not help either
(dropping the low tercile gives 0.95). Fixed contract count is correct. The magnitude model is
a separate product, not a layer.

---

## What is actually buildable today

Two books can be assembled right now from components that have each been confirmed out of
sample. One ES contract per trade, net of a full tick, no leverage:

| book | n | trades/yr | bps/trade | bps/yr | hit | Sharpe | max DD | clustered t |
|---|---|---|---|---|---|---|---|---|
| CPI leg only | 86 | 5.7 | +9.17 | +53 | 67.4% | 0.83 | −1.52% | +3.22 |
| PCE leg only | 769 | 56.0 | +3.11 | +174 | 58.3% | 0.99 | −2.79% | +3.35 |
| **A: CPI + PCE** | **855** | **56.3** | **+3.72** | **+209** | **59.2%** | **1.17** | **−2.55%** | **+4.20** |
| B: pooled Q1/Q5 | 1,896 | 137.9 | +1.15 | +158 | 53.4% | 0.54 | −7.98% | +1.84 |

The two legs of Book A correlate **+0.11** month to month, so they genuinely diversify. Across
the 2022 boundary, Book A holds (Sharpe 1.16 → 1.29, clustered t 3.45 → 2.53) while Book B
decays (0.64 → 0.40, t 1.88 → 0.75). **Narrow beats broad here**: adding the other 1,000 games
adds trades that do not pay for their own tick.

Losing years exist and should be stated: Book A is negative in 2012 (−42 bps), 2025 (−225 bps),
and roughly flat in 2015 and 2017. 2022 alone contributes +890 bps.

> **Read this table with Round two in hand.** Book A's PCE leg is the mixed-pool configuration
> that does not survive event-agnostic re-testing, and Book A's construction — take the two
> events that worked — is precisely the selection rule that Round two shows anti-persists.
> The table is kept because it is the honest measurement of what the cherry-picked version
> delivers; it is not the recommendation. Proposition 1 replaces it with a rule.

---

## Round two: does any of this generalise?

The first set of propositions named the events that worked — CPI and PCE — which is
selection after the fact. The question that decides whether there is a strategy here is
whether the *method* works on events it was not chosen on. Four tests, all with the
universe fixed in advance and one parameter set applied to everything.

### The path signal does not generalise across events

Hoshea's factor was run over **all 13 event types** in the expanded calendar with strictly
same-event matching — his Experiment B design, the event-agnostic one — at one fixed
parameter set, no per-event tuning. 12,403 games.

Nothing works. The largest full-sample IC is +0.057 (CPI); hit rates sit between 47.3% and
55.7%. Worse, **performance does not persist across events**:

| | rank correlation, 2012-2021 vs 2022-2026 |
|---|---|
| per-event bps per trade | **−0.31** |
| per-event IC | **−0.50** |

A rule that admits every event profitable in 2012-2021 returns **−0.41 bps** in 2022-2026,
slightly *worse* than the basket it rejected (−0.25). And none of the four properties that
might have predicted which events work — the volatility multiple, pool density, number of
releases, whether a consensus exists — correlates with out-of-sample performance
(+0.11, −0.24, −0.16, −0.51 against confirmation IC).

**This forces a correction to what I reported last round.** I said PCE was confirmed out of
sample, on the strength of Hoshea's mixed-event pool (selection IC +0.303 → confirmation
+0.392). Under same-event matching, PCE's confirmation IC is **−0.093** and its confirmation
P&L is **−1.69 bps**. The signal is not "PCE paths repeat" — it depends on matching PCE
paths against the dense 08:30 Jobless Claims library, and it reverses when that pool is
removed. That is a configuration-specific result, not a mechanism, and I over-read it.

### Combining path and news is additive but still zero

The two independent sources — DTW path shape and the signed Bloomberg surprise — were fitted
together on identical games, walk-forward by year, features standardised within event type,
trading the extreme quintiles.

| model | OOS IC | bps/trade net | hit | Sharpe |
|---|---|---|---|---|
| path only | −0.0006 | −0.72 | 49.4% | −0.47 |
| news only | +0.0021 | −0.50 | 47.5% | −0.37 |
| **both** | **+0.0048** | **−0.03** | 49.4% | −0.02 |

**Combining genuinely helps** — "both" beats either alone in every configuration tested, so
the two sources do carry independent information. But the sum is zero. Screening the universe
down to the highest-reaction event types does not rescue it either: top-2, top-3 and top-4 by
jump response all return between −0.80 and −1.42 bps. The path leg is a drag once the
universe is screened.

### Post-release breakout continuation does not exist

If direction is unpredictable but size is not, the natural ES-only expression is a symmetric
bracket: let the market pick the direction, then ride it. Tested over 1,903 releases — after
an initial move of k sigma in the 5 or 10 minutes after a print, the next 30 minutes in that
direction returns between **−0.14 and −0.69 bps gross** at every threshold from 0 to 2 sigma,
in both halves of the sample. There is no post-event momentum to capture.

### What *does* generalise: screen on reaction, not on P&L

One rule survives, and it is the most useful thing to come out of this exercise. Instead of
choosing which releases to trade by their past profitability, choose them by whether the
market demonstrably reacts to the number at all:

```
jump_R2 = R² of ( log open(T+1) / open(T) )  on  the standardised surprise
```

That is a property of the information, estimable from the selection period, and it never
touches drift P&L. Ninety-three Bloomberg releases were ranked on 2012-2021 and the resulting
basket traded on 2022-2026, one position per timestamp so the six CPI variants that print
together count as one trade:

| basket, chosen on 2012-2021 | trades/yr | bps/trade net | hit | Sharpe | t |
|---|---|---|---|---|---|
| **admitted: top-8 by jump response** | 36.0 | **+4.25** | 54.4% | **0.71** | 1.49 |
| rejected: the other 85 | 354.5 | +0.03 | 50.1% | −0.04 | −0.08 |
| control: top-8 by past drift P&L | 61.2 | +1.41 | 51.8% | 0.38 | 0.79 |
| all 93 releases | 363.9 | +0.41 | 50.4% | 0.33 | 0.69 |

The two selection rules move in opposite directions across the split. Selecting on past P&L
goes from **+3.97 bps in sample to +1.41 out** (t 4.42 → 0.79) — it decays, exactly as the
cross-event persistence numbers predict. Selecting on reaction strength goes from **+1.10 in
sample to +4.25 out** (t 0.93 → 1.49) — it does not.

It is also insensitive to how many releases you admit, which is what a real rule looks like:

| K | 5 | 8 | 12 | 20 |
|---|---|---|---|---|
| bps/trade net | +4.36 | +4.25 | +3.08 | +2.87 |
| Sharpe | 0.73 | 0.71 | 0.76 | 1.00 |

And it is not a CPI wrapper. Removing every CPI variant from the candidate pool entirely and
re-running the screen gives **+4.05 bps, 57.2% hit, Sharpe 0.79, t = +1.66** on 159 trades.
Two related criteria (the absolute jump beta) behave the same way; raw move size does not,
which is informative — it is the *explained* part of the reaction that matters, not how big
the move is.

---

---

## Round three: the DTW brief, on 2016-2021 / 2022-2026

Rounds one and two used a 2012-2021 selection window, which includes years the data
README flags as untrustworthy — only 20-37% of 2010-2012 weekdays carry a complete
session. Round three re-runs the DTW question on **2016-2021 train / 2022-2026 test**,
entirely inside the trustworthy range, with the universe chosen by market impact rather
than by past P&L or by hand. Scripts: `jack/dtw_impact_book.py`, `jack/dtw_dense_pool.py`,
`jack/dtw_reaction_path.py`, `jack/reaction_book.py`.

### The impact screen, and why the obvious version is wrong

Ranking releases by release-window move over the same clock slot on non-release days
scores CPI 1.88, PPI 1.83, Core PCE QoQ 1.79, Retail Sales 1.76 and **Trade Balance 1.75**
— all within noise of one another. The measure credits every 08:30 release with the fact
that 08:30 on a release day is volatile, so minor releases ride the majors.

Changing the control to *the same slot on days when some other family releases there but
this one does not* separates them: FOMC 1.73 (t 2.05), Average Hourly Earnings 1.56
(t 3.78), CPI 1.30 (t 3.22), Core PCE QoQ 1.24 (t 2.25), and everything else below
|t| = 2. This is a universe rule with no P&L in it and no test-period data in it.

### Four new DTW results, all negative

| design | confirmation |
|---|---|
| impact universe, same-family pool | best-by-training cell **+0.07 bps net, t = 0.27** |
| impact universe, clock-slot pool | mean confirmation **t = −0.90** — worse than same-family |
| dense pool (trade 10, match against all 65 families) | IC negative in 7 of 8 cells; select→confirm rank corr **−0.45** |
| reaction path `[T−30, T+1]`, entry T+1 | best cell +0.56 bps net, t = 0.64; **negative in all six training years** |

**The PCE retraction is now complete.** Round two showed PCE reverses under same-event
matching. Round three tested the remaining explanation — that pool density was the active
ingredient, since PCE's 08:30 neighbours come from the dense Claims library — by trading
the impact-screened ten while matching against every release at the slot. It fails
outright, and anti-persistently. There is no pooling rule under which the PCE result
generalises.

### The structural finding

Every DTW game in this project ends its path at `T0+60` and enters there, which for the
anchors carrying the sample is 30 to 120 minutes after the release. Jack's entry sweep
already showed the drift is spent by T+30. **The games enter after the tradeable window
has closed**, which is why blending a consensus surprise into them is flat at every weight
from w = 1.0 to w = 0.0 on both halves of the split. Moving the path to `[T−30, T+1]` and
entering at T+1 is the fix, and it produces the best DTW numbers in the project — all 16
configurations positive on test, mean +1.1 bps gross — but negative in all six training
years, P = 0.116 against a random-sign null.

### What replicated instead

**Volatility-proportional sizing.** Scaling position *up* with prevailing volatility:
+4.30 bps in training and +2.44 out of sample, against −0.54 and −0.61 for
inverse-volatility sizing. Negative in both periods for the textbook default, positive in
both for its inverse. Jack reached the same conclusion on the news book (Sharpe 1.18
ungated → 1.08 vol-sized), so this is two confirmations on non-overlapping evidence and
the most transferable result of round three.

**And one more negative worth recording:** `dtw_mag`, the one DTW feature that replicates
anywhere in this repository at rank IC +0.347 against the Parkinson range, scores
**+0.016 on event-window bars**. DTW does not carry magnitude around releases either. The
magnitude inputs that do are `vix_level` (+0.468) and `log_rv4h` (+0.457).

The proposal built on all of this is in [`strategy_proposal.md`](./strategy_proposal.md).

## Three propositions

These are three bets on where the edge is, not three lists of events. Each is a rule that
re-derives itself, so none depends on a choice anyone made after seeing a P&L curve.

### Proposition 1 — The reaction-screened macro drift book

**What it is.** Every January, for each US release with at least 30 prior observations,
regress its announcement jump on its standardised surprise and record the R². Admit the top
K (anything from 5 to 20 works). Take the trade direction from the sign of that jump
regression — never from past returns. Then trade every admitted release the same way:
position at T+1 in the direction of the surprise, flat at T+30, one position per timestamp,
equal size, no volatility scaling. The universe, the signs and the sizing are all mechanical.

**How it uses everyone's work.** Reza's event-response study is the screen — his jump
regressions are exactly the right measurement, and the reframing is that the R² he reports as
a *result* is better used as an *admission criterion*. Aaryen's walk-forward discipline and
his surprise pipeline supply the machinery. Jack's entry-lag sweep fixes the one thing that
made the earlier event backtests unusable. Hoshea's contribution is structural: the event
anchor, not the calendar bar, is the right unit, and his grid is what demonstrated that
hyperparameter rankings can generalise even when individual events cannot.

**Benefits.** The selection criterion is *proven not to decay* on this data, while the
obvious alternative is proven to decay — that is a mechanism-level result, not a backtest
artefact, and it is the one finding here that would transfer to a different asset or a
different desk. It is insensitive to K, which means it has essentially no tuned parameters.
It survives deleting CPI, so it is not one release in disguise. Out of sample it returns
+4.25 bps a trade at Sharpe 0.71 on a period that was never used to choose anything. Capital
is deployed under 1% of the time, so beta is near zero and it composes with anything else.
And the whole thing fits on one page, which matters when it stops working and someone has to
diagnose it.

**Caveats.** **t = 1.49 is not significance.** On its own evidence this is a promising rule,
not an established edge, and the honest position is that we would need another three to four
years of live data to distinguish it from luck. The confirmation window is four years and
2022 alone contributes +484 of the +680 total bps — the high-inflation era was unusually kind
to macro-surprise trading, and a return to a low-dispersion consensus regime plausibly kills
it. CPI still contributes disproportionately even though the rule does not need it. A release
needs 30 prior observations before it can be judged, so new releases take three years to
enter. And execution at T+1 after a print — the widest spread of the day — is entirely
unmodelled; at 36 trades a year, an extra half-tick of slippage removes 12% of the edge.

### Proposition 2 — Make the magnitude forecast the product

**What it is.** Stop trading direction and sell the thing that actually generalises. Across
every test in this repository, magnitude prediction replicates and direction does not: rank
correlations of +0.73, +0.61, +0.50 and +0.49 against realised range from four independent
features, QLIKE 1.421 against 1.841 for raw GARCH and 1.514 for seasonally adjusted GARCH,
Diebold-Mariano t = −6.55, and an independent confirmation from Hoshea's `|dtw_factor|`
sorting realised absolute moves from 14.30 to 19.72 bps without having been built for it. The
same screening logic as Proposition 1 applies unchanged: the release calendar says where
variance is elevated, and `is_event_window` is already the strongest non-clock term in the
model. Expressed as a trade this is ES options — the at-the-money 0DTE straddle at each
decision point, realised against implied, concentrated in release windows.

**How it uses everyone's work.** Jack's magnitude model is the core. Hoshea's factor becomes
a magnitude feature rather than a directional one, which is what it measurably is. Reza's
market-impact ranking (CPI 38.5 bps mean absolute move at 30 minutes, NFP 29.6, PCE 14.7) is
the event screen. Aaryen's walk-forward supplies the evaluation.

**Benefits.** The effect is two orders of magnitude larger than anything directional here and
stable year to year rather than concentrated in one regime. It has been independently
replicated by two people using different methods. There are vastly more opportunities — every
30-minute window has a variance, against 36 usable prints a year — so a Sharpe estimate would
be trustworthy inside a year instead of inside a decade. And it converts two months of
existing magnitude work from a by-product into the main line.

**Caveats.** It is a different instrument and arguably a different project: the brief is ES
mini and this trades ES options. It needs data we do not have and a pricing layer nobody here
has built. Most seriously, **the decisive benchmark has never been run** — beating GARCH is
not beating implied volatility, and both the intraday seasonal profile and the release
calendar are public and already in the options surface. The honest possibility is that the
entire edge is priced, leaving a good forecast and no trade. Short-variance books also carry
fat left tails that a 30-minute holding period does not remove.

### Proposition 3 — Build the killing machine, then pick

**What it is.** Not a signal — a commitment to stop adding them for one cycle. Nine
directional architectures have now been tested out of sample in this project and eight are
dead: fifteen features under four estimators, six directional DTW designs, the 16:00 ET
anchored lead, Aaryen's event pipeline at any honest lag, the NFP drift, the DTW factor
applied event-agnostically, the pooled path-plus-news model screened and unscreened, and
post-release breakout continuation. The ninth is at t = 1.49. The binding constraint is not
ideas, it is that each one takes a week to kill. So build the harness — an event-game object
with walk-forward, one-trade-per-timestamp collapsing, cluster-robust standard errors,
best-of-N nulls and cross-event persistence checks baked in, so any new idea is scored the
same way in an afternoon — and in parallel pull the ES options data so the one genuinely open
question gets settled.

**How it uses everyone's work.** It generalises the *methods* rather than the signals, which
is where the transferable value has actually been: Jack's best-of-N nulls (which killed the
16:00 lead and the NFP drift), Aaryen's expanding-window retrain and his selection-honesty
protocol, Hoshea's game grid as the canonical data structure, Reza's causal surprise
standardisation. Every script written this week is a first draft of a piece of it.

**Benefits.** It is the only proposition whose payoff does not depend on a t-statistic
between 1 and 2. The cross-event persistence result (−0.31, −0.50) means that any future idea
selected the way the first set of propositions was selected will disappoint, and the harness
is what prevents that from happening again. It makes the two decisive experiments cheap: does
the magnitude forecast beat implied, and does the screened drift hold on fresh data. And it
compounds — the fifth idea after the harness exists costs a day, not a week.

**Caveats.** It produces no P&L this quarter and is hard to show a stakeholder. It risks
gold-plating infrastructure for a research programme whose honest conclusion may be that there
is no tradeable intraday direction in ES at all, in which case the right answer was
Proposition 2 from the start. And it defers the one rule that is currently working, which
loses four years of live track record that we cannot get back.

---

## My recommendation

**Proposition 1 as the thing to trade, Proposition 3 as the thing to build, and
Proposition 2 gated on one afternoon's work.**

Proposition 1 is the only architecture that is both general and positive out of sample, and
its selection rule is the single most transferable result any of us has produced. It is not
significant and should be sized as a pilot rather than a strategy — but a rule that improves
from in-sample to out-of-sample, is flat in K, and survives deleting its biggest contributor
is worth putting on the tape at small size while the evidence accumulates.

Proposition 2 should not be committed to until the implied-volatility regression is run. That
is one afternoon once the options data exists, and it converts the biggest effect in the
project from "probably already priced" to a number.

Proposition 3 is what makes the next round faster, and the cross-event persistence result is
the argument for it: without a harness, the next promising cell will be cherry-picked the same
way the last one was.

---

## Appendix — reproduction

Every number in this document is reproducible. Scripts read teammates' committed artefacts
straight from git via `git show`, so they need a fetched remote but never modify the working
tree. Inputs are `data/processed/es_1min_clean.parquet`,
`data/processed/es_1min_bars_2010_2026.parquet`,
`data/processed/macro_event_calendar_expanded_2010_2026.parquet`,
`data/raw/Bloomberg Economic Releases.xlsx` and `data/raw/vix_daily.parquet`.

| script | what it produces | runtime |
|---|---|---|
| `jack/verify_teammate_claims.py` | per-event decomposition of Hoshea's factor, clustered t-stats, all-30-minute panel check, entry-lag sweep on a sign rule | ~1 min |
| `jack/aaryen_entry_lag.py` | Aaryen's pipeline rebuilt over 4 entry lags x 4 windows with costs | ~3 min |
| `jack/nfp_drift_test.py` | exit-window profile, 2020 split, \|z\| gradient and best-of-32 null for the NFP lead | ~1 min |
| `jack/hoshea_grid_holdout.py` | the 18-cell grid re-scored with 2012-2021 selection / 2022-2026 confirmation, per event type | ~25 min |
| `jack/state_dependence.py` | jump/drift decomposition, four state interactions, tercile splits, robustness | ~1 min |
| `jack/candidate_books.py` | Books A and B with clustered t-stats, yearly P&L and the 2022 split | seconds |
| `jack/event_tradeability_rule.py` | the factor over all 13 event types at one fixed parameter set; per-event persistence; whether any observable property predicts tradeability | ~20 min |
| `jack/combined_event_model.py` | path-only / news-only / both, walk-forward by year, pooled and per event | ~2 min |
| `jack/news_admission_rule.py` | the jump-response screen over 93 Bloomberg releases, one trade per timestamp, against a past-P&L control | ~3 min |
| `jack/event_family_screen.py` | release lines grouped into event families from calendar overlap, families ranked by jump response, basket-size sweep K=1..21 | ~3 min |

Derived outputs land in `jack/data/` (gitignored): `aaryen_entry_lag_results.parquet`,
`aaryen_entry_lag_summary.csv`, `hoshea_grid_holdout.csv`, `hoshea_holdout_best_panel.parquet`,
`state_dependence_panel.parquet`, `event_tradeability_panel.parquet`,
`event_tradeability_rule.csv`, `combined_event_predictions.parquet`,
`news_admission_rule.csv`, `news_admission_trades.parquet`, `event_family_screen.csv`,
`event_family_sweep.csv`, `event_family_top10_trades.parquet`.

### Scoreboard

Nine directional architectures tested out of sample across both rounds:

| architecture | verdict |
|---|---|
| 15 features x 4 estimators on 30-min bars | dead — OOS R² −0.00042, IR vs hold −0.31 |
| six directional DTW designs | dead — all inside their own nulls |
| the 16:00 ET anchored lead | dead — a five-minute timing leak |
| Aaryen's event pipeline | dead at any honest entry lag |
| NFP ten-minute drift | dead — P(best-of-32 null ≥ real) = 0.69 |
| macro state interactions | dead — none survives outlier or period splits |
| DTW factor, event-agnostic over 13 events | dead — cross-event persistence −0.31 |
| pooled path + news, screened and unscreened | dead — Sharpe −0.02 pooled, negative screened |
| post-release breakout continuation | dead — negative at every threshold |
| **reaction-screened news drift** | **marginal — +4.25 bps, Sharpe 0.71, t 1.49 OOS** |
| DTW on the impact universe, 3 pooling rules | dead — +0.07 bps net; dense-pool select→confirm rank corr −0.45 |
| DTW reaction path, entry T+1 | not established — negative in all six training years, P = 0.116 |
| **impact-screened news drift, vol-proportional** | **marginal — +2.44 bps, Sharpe 0.85 OOS, P = 0.116** |
