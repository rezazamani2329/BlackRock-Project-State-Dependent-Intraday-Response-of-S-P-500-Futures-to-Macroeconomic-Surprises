# Design notes

Why the ES event reaction book is shaped the way it is, what was ruled out,
and what is still open. The evidence behind every claim is in `jack/strategy_analysis.md`.

---

## 0. The strategy in brief (from the former README)

**Universe.** Release lines sharing ≥80% of their timestamps collapse into one *family*
(163 lines → 67 families), so CPI MoM and Core CPI MoM are one trade, not two. A family is
admitted on **marginal impact**: its mean `|30-min move|` over the same statistic at the
same clock slot on days when some *other* family prints there but this one does not. That
control is the whole trick — measured against quiet days, every 08:30 release scores ~1.8
simply because 08:30 is volatile, and minor releases ride the majors.

**Timing — the anchor ladder.** Five back-to-back 30-minute windows after each print:
enter at `T+1`, `T+31`, `T+61`, `T+91`, `T+121`, hold 30 minutes each. Entry at `T` is not
available — the tape price at `T` predates the number, and that correction alone takes the
strongest backtest in this project from Sharpe 2.17 to 0.29. The ladder is Hoshea's
multi-anchor design; it raises the book from ~58 to ~289 candidate games a year, of which
the tail gate trades ~185.

**Direction — the DTW factor, used as a ranking.** Today's z-scored **price path** over
the 31 minutes ending at entry is matched by DTW against every prior release at the same
clock slot and anchor; the k = 15 nearest contribute a distance- and recency-weighted mean
forward return (`dtw_dir`). It needs no consensus, so it works on releases Bloomberg
carries no survey for.

Hoshea's conclusion was that this factor is *"better treated as a ranking/probability
feature than a standalone directional rule"*, and that is how it is used: **trade only
games whose factor falls outside the middle 40% of the training distribution** (`TAIL_Q =
0.30`, thresholds recomputed each fold). Hit rate rises monotonically as the cut tightens —
0.502, 0.505, 0.506, 0.516, 0.528 — which is the response a real effect gives rather than a
fitted threshold, and the sweep is smooth rather than spiked.

**Sizing.** Inversely by `dtw_disp`, the dispersion of the neighbours' forward returns: a
wide spread means the analogues disagreed, so take less. Capped at ±2 contracts. This is
worth little — 0.652 Sharpe against 0.637 flat — and is kept mainly because it halves the
drawdown. The tail gate is where the gain comes from.

**What is deliberately not in the book.** No direction regression, no magnitude GLM, no
`news_z`, no `jump_z`, no conviction interactions. Every one was tested and every one
subtracts value: the fitted model is **negative** on the ladder (−0.482 Sharpe), `E[r]/E|r|`
sizing costs ~5x the per-trade edge, and `sign(news_z)` alone reaches only 0.189. They all
remain in `backtest.py` as comparison arms, because knowing they fail is what justifies a
rule with no estimation in it.

**Evaluation.** 5-year initial train, then expanding walk-forward, so every year from 2018
is out of sample. Costs **a quarter tick per side** (0.25 bp round trip). `2025-07` onward
is computed but not looked at while iterating.

---

## 0b. Status as of the ES-only book (from the former README)

Whole pipeline built and run, at a quarter tick per side. 1,669 trades (~185/yr) over
2018-2026, all out of sample.

| | gross | net |
|---|---|---|
| Sharpe | **0.737** | **0.613** |
| total return | 22.4% | 18.6% |
| bps/trade | 1.341 | 1.114 |

Max drawdown **−4.9%**, IR 0.629, alpha t 1.99, clustered t **+2.27**, and **P = 0.016**
against a random-sign null. The exposure-matched control — always long in the same windows
— returns **−8.4%**.

### What to distrust

- **Construction fragility.** Across nine interchangeable constructions (path length x
  pooling key) the ladder's Sharpe spans **−0.49 to +0.34** at half a tick, only a third
  positive. This configuration is the best of those nine, so P = 0.016 does not survive a
  best-of-N adjustment. That sweep has not been re-run at a quarter tick.
- **`dtw_dir` is not stable.** Standardised, its coefficient runs −1.99, −0.99, −0.35, then
  +1.92 … +1.41 — a sign flip in 2021 that never reverts. The traded rule takes only the
  sign of the raw factor, so it sidesteps this, but the underlying factor is not steady.
- **Only `jump_z` is a stable predictor**, consistently negative across all nine folds, and
  that is short-term reversal rather than DTW. It is not in the book.
- **Three panel fixes are still queued** — family naming, the sparse-slot control that
  wrongly excludes FOMC Rate Decision, and `jump_z`/`news_z` hygiene — see
  section 7 below. The **leak test** is wired (`LEAK_TEST_MINUTES = 5`) but has
  not been run.

---

## 1. The one constraint that determines the architecture

News drift is spent by `T+30` — the next clean bar scores t = +0.72. A DTW path has to end
where the position opens. So there is exactly **one instant** at which a path signal and a
news signal are both available and executable: **`T+1`**.

Every earlier DTW game in this project ended its path at `T0+60` and entered there, which
for the anchors carrying the sample is 30 to 120 minutes after the print. That is *why*
blending news into them was flat at every weight from 1.0 to 0.0 — a timing problem, not a
weighting problem. The whole book is built around fixing it.

## 2. Where the features fit

```
                        ┌─ jump regression ──────► sign of the news signal
  per family, per year ─┤
                        └─ impact screen ────────► is this family in the universe

  DIRECTION   E[r]  = b1·dtw_dir + b2·news_z + b3·(dtw_dir × dtw_agree)
                                             + b4·(dtw_dir × news_z)
                      OLS on the T+1→T+31 return, in bps

  MAGNITUDE   E|r|  = Gamma GLM on vix_level, log_rv4h, impact, seas_exp, dtw_disp

  POSITION    pos   = clip( E[r] / E|r| , ±2 )  contracts
```

**Superseded — the book contains no regression.** Both models were tested and both
subtract value: the direction model is **−0.482 Sharpe** on the ladder, and `E[r]/E|r|`
sizing costs roughly 5x the per-trade edge. What replaced them came from Hoshea rather than
Aaryen, and the reason is that his method was never a regression:

* **direction** = the sign of `dtw_dir`, traded only when the factor is in the outer
  `TAIL_Q` of the training distribution. Hoshea's own conclusion was that the factor is a
  ranking, not a standalone rule; taking him literally lifts Sharpe 0.541 → 0.613, halves
  drawdown, and moves clustered t from 1.61 to 2.27.
* **size** = inversely with `dtw_disp`, capped at ±2.

One caveat on the sizing, stated because an earlier version of this note got it wrong:
`dtw_disp` does **not** beat the magnitude model at forecasting the move. Rank IC against
the realised absolute move is +0.15 for `dtw_disp` against **+0.43 for the GLM's own
prediction** and +0.37 for `log_rv4h` alone. Inverse-`dtw_disp` is used because it gives
the best Sharpe and the smallest drawdown on this book (0.652 / −4.9% against 0.637 / −5.6%
flat, and 0.516 when sizing by the better forecast), which is unexplained. The margin over
flat sizing is small; treat the sizing layer as unproven and the tail gate as the result.

Both models stay in `backtest.py` as arms, because knowing they fail is what justifies a
rule with no estimation in it.

Four coefficients for direction. The overfitting in this project came from 15 features × 4
estimator families × walk-forward × per-slot models, not from small pooled fits — with
1,200+ training observations, four coefficients are comfortably identified. The two
interaction terms replace hard gates: no threshold has to be chosen.

## 2b. Standardisation

Direction and magnitude features are **standardised on training-fold statistics only**,
recomputed every fold, and the interactions are built from the standardised bases rather
than standardised after multiplying.

This was missing in the first implementation and it mattered for readability more than for
results. Raw, the direction block spans **16,525x in standard deviation** — `dtw_dir` sits
at 8.8e-4 while `jump_z` is 3.2 — giving a design-matrix condition number of **~26,000**
and coefficients four orders of magnitude apart (`dtw_dir` at −5352 against `news_z` at
−1.5). OLS is scale-equivariant, so the fit, R², t-stats and the traded book were all
unaffected; what was broken was that the coefficient table and the stability diagnostic
could not be read. After standardising, the condition number is **8** and every coefficient
is bps per 1sd of its feature, so the columns are directly comparable.

`dtw_agree` is left unscaled and merely centred at 0.5 — it is a bounded fraction already.

**What the readable table then shows is worse than what the unreadable one hid.**
`dtw_dir_z` runs −1.99, −0.99, −0.35, then +1.92, +1.78, +2.37, +2.23, +2.00, +1.41: a
clean sign flip in 2021 that never reverts. `dtw_x_agree` swings from **+6.4 to −13.9 bps
per sd**, which is larger than half the target's standard deviation, and flips at the same
moment — the two terms are trading off against each other rather than measuring anything.
Only `jump_z_z` is stable, consistently negative at −0.21 to −0.53, and that is short-term
reversal rather than DTW. In-sample adjusted R² on 864 training games is **−0.0005**.

## 3. What was excluded, and why

| | measured reason |
|---|---|
| `above_ma100` | OOS IC −0.0020 (t −0.46), 13th of 16; −0.50 collinear with `vol_z`. Above the MA 78.2% of the time — a regime label, not a signal |
| momentum, any horizon | partial \|t\| never exceeds 0.96 once `vol_z` and `above_ma100` are controlled. The long-horizon version was the volatility regime counted twice |
| `dtw_mag` | +0.016 rank IC **on event bars**, against +0.347 on ordinary bars. It does not work where we trade |
| slot seasonality flags | a slot's directional IC in 2022-23 correlates +0.06 (p = 0.69) with 2024-26 |
| volatility *state* interactions | four tested; none survives outlier or period splits. The cleanest goes t −4.62 → +0.17 across 2020 |

**Momentum has one untested version.** As a standalone direction signal it is dead. As a
*positioning conditioner* — does the post-print drift depend on which way the tape leaned
into the print — it is economically distinct and never tested. Second pass, clearly
labelled, so it cannot inflate the first result.

## 4. Sizing — a correction worth reading

Volatility-proportional sizing was previously reported here as the most robust result
(+4.30 train, +2.44 test, against −0.54/−0.61 for inverse-vol). **That was stated more
strongly than the evidence supports.**

The underlying tercile pattern is **+2.83 / +0.98 / +7.50** bps — *not monotone*. It is not
"edge scales with volatility", it is "the top tercile is good", and the middle bucket is
the worst of the three.

There is also a theory problem: if μ = k·σ, mean-variance sizing gives position ∝ μ/σ² =
k/σ — **inverse** vol, the opposite of what the data said. The two reconcile only if IC
itself rises with volatility (hit rate does go 58.1% → 62.0%), which is a stronger claim
than a scale effect.

Hence `E[r]/E|r|`: simple, intuitive, not procyclical. Volatility sizing gets revisited
only if the relationship proves monotone across quintiles on the full training window.

## 5. Evaluation

Five-year initial train, then expanding walk-forward — refit the universe, signs,
standardisation and both regressions on everything strictly before each year, then trade
that year. Every year from 2018 is out of sample.

**Why walk-forward rather than one split.** The fixed 2022 boundary was inherited, never
designed. Train length was never the problem — the *test* was: 4.5 years, ~450 trades, and
2022 dominated everything. Removing 2022 took the last book from +3.90 to +0.98 bps and the
DTW factor from +1.67 to +0.85. Walk-forward roughly doubles the out-of-sample count.

The cost is that walk-forward averages over regime breaks, so **the per-year table is the
primary diagnostic**, printed before any Sharpe.

Reported every run: cost curve 0 → 1.5 bp with 0.25 bp round trip as base case; per-year
P&L; clustered errors by release date; best-of-N null with N stated; results with the best
year deleted.

**Leak tests, mechanical:** path ends at or before entry; neighbours strictly earlier and
inside the lookback; and deliberately extend the path 5 minutes past entry to confirm the
result *changes a lot*. If it does not, the timing plumbing is broken. That last test is
what would have caught the 16:00 ET leak, which sat at IC 0.1557 and reconstructed at 0.0135.

## 6. Honest prior

Nine directional architectures have died in this project. The realistic good outcome is
that `dtw_agree` and the news interaction concentrate ~0.6 bps across all trades into 3-5
bps on a third of them — a real result, and a small one.

**The thing most likely to change the answer is not in this document.** At these edge sizes
execution *is* the strategy: a quarter tick per side assumes fills at or inside mid one minute
after a print, which is the widest spread of the day. That needs fill data, not research.

---

## 7. Open items

Three fixes are queued. They batch into one re-run because the DTW block takes ~1 hour.

1. **Family naming.** Families are named after their longest-history member, which picks
   obscure labels — the NFP complex comes out as "Change in Manufact. Payrolls @ 08:30",
   CPI as "CPI Index NSA @ 08:30". The grouping is correct; only the label is wrong. Names
   are keys, so fixing it needs the re-run.

2. **The sparse-slot control.** At a clock slot where the only other releases are
   comparably large, the control cannibalises the event. **FOMC Rate Decision fails
   admission** because its only 14:00 slot-mates are FOMC Minutes and Federal Budget
   Balance. Proposed: require ≥3 distinct other families in the control, else fall back to
   the quiet-day baseline, reporting both.

3. **`jump_z` and `news_z` hygiene.** `jump_z` runs to ±25σ and needs winsorising at ±10 on
   the same argument as the surprise. `news_z` covers 59% of games; the rest should enter
   as `news_z = 0` with a `has_news` indicator rather than being dropped.

**Open question.** `IMPACT_MIN_T = 2.0` admits 8 families at ~50 trades/year, and the panel
starts in 2013 because a family needs 30 prior releases before it can be screened. With a
5-year initial train the first out-of-sample year is **2018**, giving ~500 OOS trades
rather than the ~1,200 originally projected. A looser threshold widens the universe at the
cost of admitting families with no economic story.

## 8. The holdout

`HOLDOUT_START = 2025-07-01` — the last 12 months, computed but not inspected while
iterating.

**It is underpowered and should be treated as a sanity check, not a test.** At ~50 trades a
year it holds ~50 trades, which cannot distinguish anything. The real out-of-sample
evidence is the walk-forward record from 2018. Options, in order of my preference:

1. Keep it as a final sanity check, labelled as such, and rely on the walk-forward.
2. Move it back to `2024-07-01` for ~100 trades — still thin, and costs a year of
   walk-forward evidence.
3. Drop it, and rely on the walk-forward alone plus the per-year table.
