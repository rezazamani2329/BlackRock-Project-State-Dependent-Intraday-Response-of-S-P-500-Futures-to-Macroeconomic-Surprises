# result_analysis_by_month

Visual breakdown of the **test period (2021–2026)** in `combined/backtest.py`. This folder does not change backtest or feature code under `combined/`. The trading rule is the book's current one:

- direction: `sign(dtw_dir)`, middle 40% sits out (`q = 0.3`)
- gate and `disp_ref`: one per-asset set from **2013–2020** (not re-fit per ladder rung)
- size: `median(dtw_disp) / dtw_disp`, cap 2
- cost: **quarter tick** per side
- always-long: same windows and sizes, always long

Numbers come from the existing `combined/output/features.parquet`. If that panel is an earlier pass, totals will not match the ES 0.64 in Jack's latest combined write-up.

The `.py` file is the source of truth. After edits:

```bash
uv run jupytext --sync combined/result_analysis_by_month/<name>.py
.venv\Scripts\python.exe combined/result_analysis_by_month/<name>.py
```

Run `monthly_match_similarity.py` before `ladder_by_anchor.py` (the latter reads the trade-level similarity file).

---

## Scripts

| Script | What it does |
|---|---|
| `plot_test_pnl.py` | Full book (all five rungs together): cumulative P&L vs always-long; monthly return bars and heatmap |
| `monthly_match_similarity.py` | Rebuilds the k-neighbour match for each executed trade, applies the same weights used for `dtw_dir`, then averages by month |
| `ladder_by_anchor.py` | Splits the same book by ladder rung `T+{1,31,61,91,121}`: P&L, monthly / full-test similarity, direction hit rate |

---

## Outputs

### Figures `figures/`

| File | Content |
|---|---|
| `pnl_curves_test.png` | ES / NQ / ZN cumulative net P&L vs always-long |
| `monthly_returns_bars.png` | Monthly returns, strategy next to always-long |
| `monthly_returns_heatmap.png` | Year × calendar-month return heatmap |
| `monthly_weighted_similarity.png` | Full-book monthly mean weighted similarity (grey bars = trade count) |
| `monthly_weighted_similarity_overlay.png` | Three assets on the same similarity axis |
| `pnl_curves_by_anchor.png` | 3 × 5: cumulative P&L vs always-long by rung |
| `monthly_similarity_by_anchor.png` | 3 × 5: monthly similarity by rung (dotted = full-test mean) |
| `direction_hit_by_anchor.png` | Hit rate of `sign(dtw_dir)` vs the 30-minute return, by rung |

### Tables `output/`

| File | Content |
|---|---|
| `test_summary.csv` | Full-book test total / Sharpe / drawdown |
| `monthly_returns.csv` | Month-by-month returns (strategy and always-long) |
| `monthly_stats.csv` | Monthly hit rate, best / worst month |
| `yearly_from_months.csv` | Calendar-year sums of monthly returns |
| `trade_weighted_similarity.csv` | Per-trade weighted similarity |
| `monthly_weighted_similarity.csv` | Full-book monthly mean similarity |
| `pnl_by_anchor.csv` | Strategy vs always-long by rung |
| `monthly_similarity_by_anchor.csv` | Monthly similarity by rung |
| `test_similarity_by_anchor.csv` | Full-test mean similarity by rung |
| `direction_by_anchor.csv` | Hit rate, long / short hit, `corr(dtw_dir, fwd_ret)` |

Weighted similarity (same weights as `features.py`):

\[
\sum_i w_i \cdot \frac{1}{d_i+\varepsilon},\quad
w_i \propto \frac{1}{d_i}\exp(-\mathrm{age}_i/730),\ \sum_i w_i=1
\]

---

## Findings

### Full book (all five rungs, quarter tick)

| | Trades | Strategy | Always-long |
|---|---|---|---|
| ES | 1,187 | +8.0%, Sharpe 0.43 | −5.5%, −0.31 |
| NQ | 1,082 | +11.7%, 0.53 | −12.8%, −0.57 |
| ZN | 1,896 | −6.7%, −0.86 | −7.5%, −0.90 |

On the equity books the strategy and always-long diverge, so the edge is direction, not “being in the release window.” On ZN the two lines sit on top of each other. Monthly hit rate is about 53% on ES/NQ and 38% on ZN. Equity P&L is concentrated in 2021 and 2025; 2022 is the loss year.

**2022 H1.** The release windows themselves were down (about −10 bp on ES, −12 bp on NQ). ES was long-heavy (~63%): longs lost, shorts made a little, direction IC ≈ 0. The worst single month (May) was short into a bounce, not long into a dump. NQ was closer to 50/50 and lost less than always-long; May was also short-heavy into a bounce. DTW does **not** use surprise. It takes the historical forward return of similar paths as the direction. In 2022 the paths still matched; what those paths used to do next had changed.

### Match similarity

Monthly mean weighted similarity is about **0.51 on ES/NQ and 0.46 on ZN**. 2022 H1 is not a poor-match window (ES 0.51, NQ 0.50, ZN 0.48). That year’s drawdown is **not** “no close neighbours.” Matches were as close as usual; the path-to-return map broke.

### By ladder rung

The edge is concentrated on **T+1**:

| | T+1 strategy / always-long | Hit rate | Full-test weighted sim |
|---|---|---|---|
| ES | **+9.0% (Sharpe 0.93)** / +0.9% | **55%** | 0.52 |
| NQ | **+9.7% (0.86)** / +2.3% | **54%** | 0.52 |
| ZN | −1.1% / −1.5% | 50% | 0.48 |

Later rungs get worse: ES at **T+121 is −6.9%, Sharpe −1.06, hit rate 43%**. Similarity is almost flat across rungs (~0.44–0.54). Later windows are not worse matches — they are worse **direction**. That matches the idea that surprise / reaction content is still usable near T+1; the last four rungs are leftover path. ZN is negative or noise on every rung.

---

## Relation to Jack’s 2018-start ES slide

That slide is an earlier **ES-only, 2018–2026 walk-forward** (~185 trades/year, net Sharpe 0.61, +18.6%). This folder only scores **2021–2026**, with a fixed 2013–2020 gate, on the local panel. The numbers are not meant to match that slide.
