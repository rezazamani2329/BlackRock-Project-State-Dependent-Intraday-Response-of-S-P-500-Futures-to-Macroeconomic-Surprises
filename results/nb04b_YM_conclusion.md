# Purpose, Research Questions, Answers, and Conclusion (YM)

## Purpose of Notebook 04b (YM)

Notebook 03b showed that YM reacts to CPI surprises like ES, with the same volatility dependence (CPI × 20-day volatility, t = 3.7, survives FDR). Notebook 04b asks the same **practical** question as Notebook 04, for the E-mini Dow:

> **Could a trader have made money from CPI surprises in YM in real time, after transaction costs, using only information available at each point in time?**

**Design (identical to Notebook 04):** entry at the close of the first post-release bar, exit 30 minutes after the release, walk-forward re-estimation with the first 36 releases of each family used only for training, evaluation 2019–2026, costs of 2 ticks + $4.50 per round trip on the YM contract (tick 1 index point, $5 per point).

### Are costs a problem for YM?

**Answer:** No. The average round trip is about **1.0 bps** (ES about 1.7, NQ about 0.7), falling from 1.6 bps in 2016 to 0.6 bps in 2026 as the Dow rose. That is small next to the mean absolute 30-minute drift after CPI (18.3 bps).

### Does the surprise predict YM's drift out of sample?

**Answer:** Yes, for CPI. Over 2019–2026:

| CPI strategy | Trades | Hit rate | Net bps/trade | Net Sharpe | 90% CI | Max DD |
|---|---|---|---|---|---|---|
| **Naive sign** | 77 | 57.1% | 8.9 | **0.99** | 0.27 to 1.72 | −1.8% |
| Surprise only | 48 | 58.3% | 10.4 | 0.84 | 0.07 to 1.61 | −1.9% |
| Surprise + State (`rv_20d`) | 51 | 56.9% | 9.6 | 0.83 | 0.13 to 1.51 | −1.9% |
| Surprise + State filter | 28 | 67.9% | 18.4 | 0.98 | 0.25 to 1.65 | −1.6% |
| Surprise + Adaptive state | 53 | 52.8% | 8.3 | 0.74 | 0.02 to 1.44 | −1.9% |

Every CPI strategy has a 90% bootstrap interval above zero. The naive sign rule is the best (P(Sharpe ≤ 0) = 1.3%), the same conclusion as for NQ and very close to ES (CPI sign rule 1.03 in ES and 1.00 in NQ over the same years).

### Is the profit from the direction, or from being in the market?

**Answer:** From the direction. Trading the CPI surprise only, the net return is **+5.0%**; being always long in exactly the same 30-minute windows loses **−5.5%**. The direction of the surprise is worth **10.5 percentage points**. The beta to YM buy-and-hold is −0.001, and the strategy is in the market about 0.06% of the time.

### Does the pre-release state help in real time?

**Answer:** Not in a statistically reliable way. The walk-forward interaction slope on 20-day volatility grows over time (t = 0.6 in 2019, 2.9 in 2021, 4.2–4.9 in 2022–2024, 2.4–2.5 in 2025–2026), and the adaptive model picks `rv_20d` for every CPI release from 2022. The P&L shows why it is tempting: in the highest volatility tercile the surprise-only CPI trades earn **+22.6 bps each** (67% hit), against **−0.9 bps** in the lowest tercile. But compared with surprise only, the full state model changes the Sharpe by −0.01 (P = 0.52), and the state filter, which skips the low-volatility tercile, by +0.14 (P = 0.28). Neither difference is significant.

### Does it work beyond CPI?

**Answer:** Partly. Adding PCE keeps the result (surprise only 0.89, state filter 1.06, naive 0.87). Adding NFP lowers it: on all three families the naive sign rule falls to 0.57, while surprise only stays at 0.78, because the fitted model trades only the NFP releases with a large predicted move.

### Is the result robust?

**Answer:** Yes, to costs; less so across years.
- **Costs:** every strategy keeps a positive Sharpe up to 8 ticks per round trip (naive sign 0.81 at 8 ticks).
- **Estimation choices:** surprise only stays at 0.69–0.99 on CPI across expanding or rolling windows and trade thresholds.
- **By year:** the CPI book makes most of its money in **2022 (+4.1%)** and 2020 (+1.4%); 2021 and 2025 lose about 1%. As in ES, the 2022 inflation regime carries much of the result.

## Final Comparison: Surprise Only vs Surprise + State (YM, with ES and NQ for reference)

| Model | YM CPI net Sharpe | Verdict for YM |
|---|---|---|
| Naive sign | **0.99** | Best; same ranking as NQ |
| Surprise only | 0.84 | Positive, CI above zero |
| Surprise + State | 0.83 | No improvement over surprise only |
| Surprise + State filter | 0.98 | Higher per-trade edge (18 bps), not significantly better |

## Conclusion

1. **The ES result carries over to the Dow.** Trading YM in the direction of the CPI surprise, one minute after the release, for 30 minutes, earns a **net Sharpe of about 1.0** over 2019–2026, with a hit rate of 57%, a maximum drawdown under 2% and no market beta.
2. **The simplest rule is the best**, as in NQ: with costs of about 1 bps, trading every CPI release by its sign beats the fitted model.
3. **The volatility state is real in the data but adds nothing reliable in real time**, the same verdict as for ES and NQ.
4. **Caveats:** 48–77 CPI trades over 7.5 years; returns per unit of notional before leverage; first-minute execution may be worse than modelled; most of the profit comes from 2022.

> **Overall:** YM confirms the ES result on a third equity index. For YM the recommended CPI model is the naive sign rule (Sharpe 0.99), and Notebook 05b tests whether the edge extends to other macro releases.
