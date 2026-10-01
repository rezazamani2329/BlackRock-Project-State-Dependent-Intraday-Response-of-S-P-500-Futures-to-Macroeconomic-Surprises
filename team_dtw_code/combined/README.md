# combined — DTW event reaction book

An intraday strategy traded in the 30-minute windows after scheduled US data releases, on
ES (S&P 500), NQ (Nasdaq-100) and ZN (10-year note). Direction comes from `dtw_dir`, and
position size from `dtw_disp`. Two multi-asset books are reported: **equities** (ES + NQ) and
**combined** (ES + NQ + ZN). ZT (2-year note) was dropped at the data stage.

## Structure

```
combined/
  README.md      this file
  config.py      every parameter in one place: paths, assets, DTW settings, split, costs
  pull.py        downloads 1-minute bars from Databento for the assets in config.py
  cleaning.py    filters sessions, holidays and rolls, fills no-trade minutes, one file per asset
  features.py    builds dtw_dir, dtw_disp and the universe screen, and scores the DTW setting
  backtest.py    trades ES, NQ, ZN, the equities and combined books, and compares them with buy-and-hold
  surprise_dtw_book.py  Reza's surprise and surprise + DTW books, on the same data, years, costs and metrics as backtest.py
  team_book.py   all four workstreams (Reza, Jack, Hoshea, Aaryen) on the same terms, with team portfolios and a team report by strategy
  experiments/   side studies: tune_dtw.py, tune_dtw2.py and tune_local.py are the DTW tuning rounds (none adopted); surprise_dtw.py tests surprise direction with DTW sizing; weighting.py compares dynamic book-weighting schemes (inverse vol + cost gate adopted)
  output/        derived files written by the scripts (gitignored)
```

Run `pull.py`, `cleaning.py`, `features.py` and `backtest.py` in that order.
`surprise_dtw_book.py` needs only `cleaning.py` (not `features.py`). It imports Reza's pipeline
from `Reza/src` and reads the Bloomberg file from `data/raw/`, or from Reza's project folder next
to this repo if it isn't there. It takes a few minutes, mostly for the DTW step on three markets.
`cleaning.py`, `features.py` and `backtest.py` are jupytext notebooks: the `.py` is the
source of truth, and after editing one, sync its `.ipynb` with
`uv run jupytext --sync combined/<name>.py`.

## Data

### Sources (`data/raw/`)

| file | source | contents |
|---|---|---|
| `ES_1m_2010_2026_full_v2.parquet` | Databento `GLBX.MDP3`, `ohlcv-1m`, `ES.c.0` (calendar roll) | E-mini S&P 500, 1-minute bars, Jul 2010 – Jun 2026 |
| `NQ_1m_2010_2026.parquet` | Databento `GLBX.MDP3`, `ohlcv-1m`, `NQ.c.0` (calendar roll) | E-mini Nasdaq-100, 1-minute bars, Jul 2010 – Jun 2026 |
| `ZN_1m_2010_2026.parquet` | Databento `GLBX.MDP3`, `ohlcv-1m`, `ZN.v.0` (volume roll) | 10-year T-note, 1-minute bars, Jul 2010 – Jun 2026 |
| `YM_1m_2010_2026.parquet` | Databento `GLBX.MDP3`, `ohlcv-1m`, `YM.c.0` (calendar roll) | E-mini Dow, 1-minute bars, Jul 2010 – Jun 2026 (candidate asset, see the YM section) |
| `Bloomberg Economic Releases.xlsx` | Bloomberg | US release calendar (event names and release times) |

The three bar files come from `pull.py`. ZT (2-year note) was pulled but dropped: it trades
too thinly minute to minute before 2019.

### Outputs

| file | written by | contents |
|---|---|---|
| `data/processed/futures_1min_clean_v3_{ES,NQ,ZN}.parquet` | `cleaning.py` | cleaned 1-minute bars, one file per asset (committed to the repo) |
| `data/processed/futures_1min_clean_v3_YM.parquet` | `cleaning.py` | cleaned YM bars, same columns (committed to the repo) |
| `combined/output/session_calendar.parquet` | `cleaning.py` | every session per asset, with its bar count and why it was kept or dropped |
| `combined/output/features.parquet` | `features.py` | one row per (asset, release, anchor): forward return, `admitted`, `dtw_dir`, `dtw_disp` |
| `combined/output/dtw_tuning.csv` | `features.py` | training-period scores for every DTW setting, per asset |
| `combined/output/backtest_summary.csv` | `backtest.py` | the metric tables for each asset and the equities and combined books |
| `combined/output/surprise_dtw_book_summary.csv` | `surprise_dtw_book.py` | metric tables for the five surprise / DTW books, per market and combined, validation and test |
| `combined/output/surprise_vs_dtw_test.csv` | `surprise_dtw_book.py` | Sharpe of the DTW book next to the five books, 2021-2026, at every cost level |
| `combined/output/surprise_dtw_book_cumulative.png` | `surprise_dtw_book.py` | cumulative P&L of the five books, test years, a quarter tick |

## Features

One row per (asset, release, anchor), for five back-to-back 30-minute windows entering at
T+1, T+31, T+61, T+91 and T+121. The path is the z-scored price over the 31 minutes ending
exactly at entry. It is matched by DTW against every earlier game on the same asset, clock
slot and anchor, and the k nearest neighbours give:

- **`dtw_dir`**: the mean forward 30-minute return of the neighbours. Its sign is the trade
  direction.
- **`dtw_disp`**: the standard deviation of the neighbours' forward returns. Position size is
  `median(dtw_disp) / dtw_disp`, capped at 2 contracts.

**The setting** (`config.py`): the original ES book's (commit `9201a51`): Sakoe-Chiba band 5
minutes, 3-year lookback, pool capped at the 600 most recent games, 2-year recency half-life,
k = 15, 1/distance weights. Rank IC with the next return, train / validation: ES −0.060 /
+0.069, NQ −0.042 / +0.091, ZN −0.003 / −0.005. `dtw_disp` ranks the size of the move at +0.10
to +0.24 on all three assets in both periods.

`experiments/tune_dtw.py` found a long-history setting (band 10, 8-year lookback, no cap, no
decay, k = 40, uniform) that was positive on 2013-2020 for both assets. On 2021-2026 it scored
ES 0.22 against this setting's 0.61, so it was reverted.

`experiments/distribution_dtw.py` tried taking the neighbours as the nearest 5% of the pool
(30 games when the pool is full) instead of the k = 15 nearest. It beat k = 15 on ES and NQ
in 2014-2020 (ES + NQ Sharpe −0.44 / +1.19 against −0.95 / +0.92, train / valid) and ranked
the size of the move better (`dtw_disp` IC ES +0.19 against +0.14 in training). On 2021-2026,
at a quarter tick, it scored ES 0.15 against 0.64, NQ 0.37 against 0.11 and equities −0.08
against 0.34, so it was reverted. Blending the neighbours with the pool's average return
(90/10) changed nothing.

## Sample split
| period | years | used for |
|---|---|---|
| burn-in | 2010-2012 | neighbour library and screen history only |
| train | 2013-2017 | choosing hyperparameters |
| validation | 2018-2020 | checking the choice holds |
| test | 2021-2026 | results, run once |

## Results

`backtest.py`, test years 2021-2026 only. The gate is fixed at the original book's rule: trade
the outer 60% of `dtw_dir` (q = 0.3), sized by `median(dtw_disp) / dtw_disp` capped at 2
contracts. The cut-offs come from 2013-2020.

**Book weights: inverse vol with a cost gate, walk-forward.** Each day an asset's weight in a
multi-asset book is `1 / trailing vol of its underlying` (daily returns, previous 60 sessions),
or zero if it fails the cost gate, normalised to sum to one. The **cost gate** (checked monthly)
trades an asset only while a round trip costs under 10% of its median 30-minute move on admitted
releases over the past year: it asks whether costs are small next to the moves being traded,
not whether past trades made money. Each cost column below is the book a trader facing that cost
would have run, so at zero cost every asset passes. The buy-and-hold benchmark is weighted
1 / trailing vol. Weights only scale the event trades: the book holds positions only in the
30-minute release windows, about 2% of trading minutes.

The scheme was chosen on 2016-2020 among eleven dynamic schemes (`experiments/weighting.py`).
Gates built on trailing P&L (3-year or expanding Sharpe, mean / variance) zeroed every weight on
59-85% of those days, because the direction lost on every asset in 2013-2017, so they skip most
event trades. Before 2021 the cost ratio was 1-7% on ES and NQ and 13-21% on ZN in every year,
so any cut-off from about 8% to 13% gives the same choice.

Sharpe:

| | trades (¼ tick) | gross | quarter tick | half tick | buy-and-hold | max DD, quarter tick |
|---|---|---|---|---|---|---|
| ES | 1,202 | 0.78 | 0.64 | 0.49 | 0.69 | −5.0% |
| NQ | 1,141 | 0.14 | 0.11 | 0.08 | 0.58 | −8.2% |
| ZN | 1,945 | 0.34 | −0.97 | −2.24 | −0.69 | −8.0% |
| **Equities** (ES + NQ) | 2,343 | 0.66 | **0.54** | 0.42 | 0.65 | −5.3% |
| **Combined** (ES + NQ + ZN) | 2,476 | 0.76 | **0.40** | 0.41 | 0.24 | −5.3% |

**The equities and combined books at each cost level:**

| | Equities, gross | Equities, ¼ tick | Equities, ½ tick | Combined, gross | Combined, ¼ tick | Combined, ½ tick |
|---|---|---|---|---|---|---|
| trades | 2,343 | 2,343 | 2,343 | 4,288 | 2,476 | 2,343 |
| return (total, 2021-2026) | 9.1% | 7.4% | 5.7% | 5.4% | 5.4% | 5.7% |
| ann. vol | 2.6% | 2.6% | 2.6% | 1.3% | 2.5% | 2.6% |
| Sharpe | 0.66 | 0.54 | 0.42 | 0.76 | 0.40 | 0.41 |
| max drawdown | −5.0% | −5.3% | −5.6% | −2.2% | −5.3% | −5.6% |
| beta | 0.003 | 0.003 | 0.003 | −0.003 | 0.003 | 0.003 |
| win rate | 49.8% | 49.8% | 49.3% | 48.5% | 49.3% | 49.3% |
| mean weight ES / NQ / ZN | 58 / 42 / – | 58 / 42 / – | 58 / 42 / – | 23 / 17 / 60 | 56 / 41 / 4 | 58 / 42 / 0 |

- **Equities is the book to run.** The weights stay near 58% ES / 42% NQ, Sharpe is 0.54 at a
  quarter tick with a −5.3% drawdown and no beta, and it breaks even at 1.35 ticks per side.
- **ZN only helps with no costs.** With free trading ZN passes the gate, takes about 60% of the
  weight, and the combined book reaches 0.76 with a −2.2% drawdown. At a quarter tick a round trip
  is 10-17% of ZN's typical move, so it fails the gate except in part of 2023, when bigger rate
  moves let it in and it lost (combined −0.23 that year, against 0.70 for equities). At a half
  tick it never passes, so the combined book is the equities book and scores slightly higher
  than at a quarter tick.
- **Other weightings, same test years, quarter tick:** fixed equal-risk (the original, set on
  2013-2020) gave equities 0.52 and combined −0.23 (ZN 60% of the weight); trailing 3-year net
  Sharpe gave 0.34 and 0.34.

**By year** (`backtest.py` section 5f, a quarter tick; 2026 is a partial year):

| | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|
| ES | 0.67 | −0.48 | 0.89 | 1.33 | 0.76 | 1.27 |
| NQ | 0.12 | −0.84 | −0.06 | −0.28 | 0.94 | 1.36 |
| ZN | −1.24 | −1.01 | −1.73 | −0.52 | −0.25 | −0.89 |
| **Equities** | 0.71 | −0.87 | 0.70 | 0.78 | 1.04 | 1.58 |
| **Combined** | 0.71 | −0.87 | −0.23 | 0.77 | 1.03 | 1.57 |
| Equities, return | +1.3% | −2.5% | +1.6% | +1.7% | +3.1% | +2.3% |
| Equities, win rate (long / short) | 50% (54 / 45) | 50% (49 / 51) | 47% (51 / 43) | 50% (52 / 48) | 49% (49 / 49) | 54% (58 / 49) |
| Equities, % long | 55% | 55% | 52% | 54% | 47% | 53% |

Win rates are 45-56% in every book-year, so the equity edge comes from winners being larger,
not more frequent. Equities wins more often long than short in every year except 2022 and 2025.
Section 5f has return, vol, max drawdown, beta, long/short shares and long/short win rates for
every book and year.

**Checks** (`backtest.py` section 5, 2021-2026, a quarter tick unless noted):

| | ES | NQ | ZN | Equities | Combined |
|---|---|---|---|---|---|
| Leak test: Sharpe with the path running 5 min past entry | 2.86 (real 0.64) | 2.85 (real 0.11) | 2.73 (real −0.97) | 3.62 (real 0.54) | 3.58 (real 0.40) |
| Random-sign test: p-value, 5,000 draws | **0.041** | 0.37 | 0.21 | 0.064 | 0.11 |
| Always long in the same windows: total return | −2.4% (real +10.9%) | −1.0% (real +2.1%) | −7.4% (real −7.4%) | −1.8% (real +7.4%) | −2.0% (real +5.4%) |
| Break-even cost, ticks per side | 1.36 | 1.10 | 0.07 | 1.35 | 0.95 |
| Rules with Sharpe > 0, of 12 (gate × cut-offs × sizing) | 12 | 7 | 0 | 12 | 9 |
| Appraisal ratio (alpha ÷ residual vol vs buy-and-hold) | **0.61** (alpha t 1.41) | 0.11 | −0.97 (alpha t −2.25) | 0.53 (alpha t 1.22) | 0.39 (alpha t 0.92) |
| Fundamental law: direction IC × √(bets / year) | 0.027 × √225 = 0.41 | −0.002 × √214 ≈ 0 | 0.016 × √364 = 0.31 | 0.24 | 0.11 |

The leak test changes the result as it should, so the timing is sound. ES's direction beats
random signs, and the P&L comes from direction rather than from being in the market at
release times. Equities is at the edge of significance (p = 0.064); NQ, ZN and the combined book
are not distinguishable from random. The edge needs execution near a quarter tick: ES is 0.21
and equities 0.17 at a full tick.

**Framing.** The right IR is the appraisal ratio. The book has a beta of about zero and is in
the market about 2% of the time, so its benchmark is cash, and the IR against buy-and-hold
(−0.57) only measures being out of a rising market. ES's edge is a very small per-trade IC
(0.027, a 50% hit rate) spread over about 225 bets a year. The fundamental law puts that at
an IR of about 0.4; the realised 0.6 is above it, partly from sizing and partly luck, so
0.4-0.6 is the fair range. At 3.2% annual volatility the capital is mostly idle: scaled to
10% volatility the ES book would return about 6% a year, if execution stays near a quarter
tick.

**Why the first `combined/` pass scored ES 0.21.** Two changes from the ES book: `features.py`
picked k = 10 (it tuned on training-year IC, which was negative in every setting), and
`backtest.py` widened the gate to the outer 80%. ES on 2021-2026 at a quarter tick, by
construction:

| | k = 10 | k = 15 | k = 25 |
|---|---|---|---|
| outer 60%, cut-offs fixed from 2013-2020 | 0.36 | **0.61** | 0.42 |
| outer 60%, cut-offs recomputed each year | 0.50 | 0.50 | 0.51 |
| outer 80%, cut-offs fixed from 2013-2020 | 0.17 | 0.64 | 0.44 |
| outer 80%, cut-offs recomputed each year | 0.34 | 0.61 | 0.36 |

ES ranges from 0.17 to 0.64 over these equally reasonable constructions, so the result is
fragile to them.

## Surprise and surprise + DTW books (`surprise_dtw_book.py`)

Reza's macro-surprise strategies (`Reza/`, notebooks 04-07), re-run on this folder's terms so they
can be compared row for row with the DTW book.

**Shared with `backtest.py`:** the cleaned `v3` bars, contract specs and cost levels from
`config.py`, the session grid, buy-and-hold with roll jumps skipped, the metric definitions, the
equal-risk combined book, and the 2021-2026 test years. One extra cost level is reported:
2 ticks round trip + $4.50 commission.

**Different by design:** one 30-minute window per release (entry at T+1), not the five-window
ladder. Walk-forward estimation: history from 2013, with the screen, line signs, CPI slope and DTW
gate refit each year on earlier years only. Every year from 2018 is out of sample, and validation
(2018-2020) and test (2021-2026) are reported separately.

| book | direction |
|---|---|
| CPI · surprise model | CPI only; trade the sign of the predicted move when it exceeds the round-trip cost |
| All · naive | top-10 families by marginal impact, plus CPI; sign of the consensus surprise |
| All · DTW only | the same releases; sign of Reza's single-window DTW score, gated outside the middle 40% |
| All · surprise + DTW agree | the naive book, kept only when DTW points the same way |
| CPI · surprise + DTW agree | the CPI model, kept only when DTW agrees |

**Results, Sharpe at a quarter tick, 2021-2026:**

| | DTW book | All · naive | CPI model | surprise + DTW agree | CPI + DTW agree | DTW only |
|---|---|---|---|---|---|---|
| ES | 0.21 | **1.09** | 0.51 | 0.37 | 0.74 | −0.26 |
| NQ | 0.41 | **0.75** | 0.46 | 0.48 | 0.58 | −0.28 |
| ZN | −0.85 | −0.16 | 0.17 | 0.12 | 0.48 | 0.19 |
| combined | −0.19 | **0.84** | 0.47 | 0.45 | 0.72 | −0.23 |
| ES + NQ | – | **1.00** | 0.49 | 0.48 | 0.68 | −0.32 |

- **The surprise direction is the stronger signal in equities.** It beats the DTW book on ES, NQ
  and the combined book with about a third of the trades. ES + NQ scores 1.00 at a quarter tick
  and 0.76 at 2 ticks + $4.50, with a −2.9% max drawdown.
- **ZN does not survive costs in either workstream.** It gets 44% of the equal-risk weight, which
  pulls the three-market book from 0.84 at a quarter tick down to 0.16 at 2 ticks + $4.50.
- **DTW flips between periods.** DTW only is +0.57 (ES) and +0.78 (NQ) in 2018-2020 but negative
  in 2021-2026, the same break as `dtw_dir`'s sign flip. So surprise + DTW agree beats the naive
  book in validation and loses to it in test. As a directional filter, DTW does not carry over.
- **Caveats.** About 12 of the ES naive book's +16% came in 2022. The constants (`K_FAMILIES`,
  `MIN_TRAIN`, `THRESHOLD`) were fixed in `Reza/` while looking at 2019-2026, so 2021-2026 is not
  an untouched test for these books either. The CPI model's trade count changes with the cost
  level, because the cost is also its trade threshold.

*Update (2026-09-30).* The **DTW book** column above is the first `combined/` pass. The current
`backtest.py` (commit ac384c3) scores ES 0.64, NQ 0.11, ZN −0.97, equities (ES + NQ) 0.54 and
combined 0.40 at a quarter tick; `surprise_dtw_book.py` section 9 now reads those numbers
(from `output/backtest_summary.csv` when it exists). Against them the naive surprise book still
leads on ES (1.09), NQ (0.75) and ES + NQ (1.00). The two books also lose in different years:
2022 is the DTW equities book's only losing year and the surprise book's best.

## YM — E-mini Dow (candidate asset)

**Why.** YM is the CME E-mini Dow ($5 × DJIA, tick 1 index point = $5, about 1 bp in 2010 and
0.25 bp now): the only other US equity-index future on the same Databento feed (`GLBX.MDP3`) with
1-minute history back to 2010. (The E-mini Russell 2000 returned to CME only in July 2017.) Same
18:00-17:00 ET session, quarterly calendar roll and third-Friday expiry as ES and NQ.

**How it is wired in.** `config.py` lists YM in `EXTRA_ASSETS`, not in `ASSETS`, so adding it
changes no existing result: `features.py` and `backtest.py` (the DTW book) do not trade it.
`pull.py` and `cleaning.py` handle every asset in `ALL_ASSETS = ASSETS + EXTRA_ASSETS`, and
`surprise_dtw_book.py` and `team_book.py` add YM automatically when its cleaned file exists.

```
uv run python combined/pull.py YM              # cost only
uv run python combined/pull.py --pull YM       # download (pull.py now takes symbols)
uv run python combined/cleaning.py             # cleans every asset whose raw file is on disk
```

**Data.** 5,238,612 raw bars (Jul 2010 – Jun 2026). Cleaning keeps 3,754 sessions (84-95% a
year; dropped: 68 holidays, 62 quarterly expiries, 220 short sessions), 5,195,343 bars, 4.8%
filled minutes overall: 8-15% before 2016, like NQ, and 0.6-1.8% from 2018. No contract change
falls in the daytime. File size 73 MB.

**Results so far (Reza's project notebooks 02b-08b, 2019-2026, 2 ticks + $4.50 per round trip):**

| | YM | ES | NQ |
|---|---|---|---|
| CPI: 5-min reaction per 1-SD surprise | −19.2 bps (t −2.83) | −20.8 | −27.7 |
| CPI: share of the 5-min move explained (R²) | 0.27 | 0.27 | 0.30 |
| CPI sign rule, net Sharpe | 0.99 | 1.03 | 1.00 |
| All releases · naive, net Sharpe | 0.68 | 0.65 | 0.71 |
| All releases · surprise + DTW agree | 0.94 | 0.31 | 0.62 |

- YM behaves like a slightly smaller ES: same releases, same signs, about 90% of ES's CPI response.
- Adding YM to the equity surprise book raises the ES + NQ portfolio from 0.72 to **0.78**
  (ES + NQ + YM, about 290 trades a year, 90% CI 0.10-1.50), because the YM book's monthly
  returns correlate only 0.66 with ES's and 0.43 with NQ's.
- DTW agreement looks better on YM than on ES and NQ, but the gain is not significant
  (P(not better) = 0.23; blend 0.06) and does not appear on the other markets.
- Caveat: YM was added after the ES and NQ results were known.

## Open items

- **The test period has been looked at several times**: first pass (ES 0.21), the
  long-history setting (ES 0.19), and the reconciliation table above. 2021-2026 is no longer a
  clean test, and the restored setting was chosen knowing its 2021-2026 result.
- **Local tune with the gate fixed** (`experiments/tune_local.py`, 17,280 settings around the
  book's): the best on 2013-2020 scored ES 0.22 / NQ 0.19 on 2021-2026 against the book's 0.61 /
  0.11, so the config was kept. The 2013-2020 ranking does carry over on ES (Spearman 0.46
  across settings; top-decile settings average 0.47 on test) but not on NQ (0.03). Picking
  one best setting is noise; averaging over the top settings may not be.
- **The direction signal changes regime around 2018.** In `experiments/tune_dtw.py`, 93% of
  138k rule settings lose on train (2014-2017) and 73% win on validation (2018-2020). Train and
  validation Sharpe correlate at only 0.15 across settings, so which setting wins mostly depends
  on the period. Flipping the sign on `dtw_dir`'s trailing record fixes train and breaks
  validation. The long-history setting now in `config.py` is positive on ES in both periods but
  still did not carry to the test years on NQ.
- **`dtw_disp` is the robust result:** it ranks move size at +0.17 to +0.26 on both assets and
  both periods, and its ranking across settings carries from train to validation (0.67). Large
  k, one neighbour per entry minute and a 1-year half-life lift it to about +0.30 on train. But
  sizing by it moves Sharpe by only about ±0.2.
- **Gate thresholds are fixed, not walk-forward.** `backtest.py` computes the gate cut-offs once
  from 2013-2020; `tune_dtw.py` recomputes them each year from earlier years. On ES the old book
  swung from 0.36 to 0.78 on this choice alone, so it is worth settling.
- **Surprise direction with `dtw_disp` sizing** (`experiments/surprise_dtw.py`, T+1 only, games
  with a Bloomberg survey): ES Sharpe +0.63 / +0.72 and NQ +0.64 / +0.97 (train / validation) at a
  quarter tick. It trades about 20-30 releases a year per asset, far fewer than the ladder.
- **`surprise_dtw_book.py` compares against `output/backtest_summary.csv`**, which has a ZN row
  again. Its own combined books still use fixed equal-risk weights, so its "combined" row is not
  weighted the same way as `backtest.py`'s; it has not been rerun since the weighting changed.
- **ZN and the cost gate.** The gate let ZN in during 2023, when its moves were larger, and it
  still lost: bigger moves do not mean the direction works on ZN. ZN's direction IC is positive
  (+0.016) but below its cost at any realistic execution.
