## 8. Interpretation and Conclusion (with YM)

### Purpose of Notebook 08b

Notebook 08 put the four workstreams (Reza's macro-surprise books, Jack's DTW ladder, Hoshea's DTW event games and Aaryen's classifier) on the same trades, years (2019–2026), metrics and three cost levels. Notebook 08b adds two books from Notebook 07b: the **YM** multi-release surprise book and the **three-equity (ES + NQ + YM)** book. Everything else is unchanged: Jack's ladder is our rebuild of his slide strategy, Hoshea's games and Aaryen's classifier come from their own output files (our returns correlate 0.998–0.999 with Aaryen's own), and costs are gross, Jack's quarter tick per side, and our base case of 2 ticks + $4.50 per round trip.

### 1. At equal costs, which workstream has the highest Sharpe?

| Book (2019–2026) | Trades / yr | Gross | Jack's cost (¼ tick) | Our cost (2 ticks + $4.50) |
|---|---|---|---|---|
| Aaryen · classifier, 30m, lag 0 (as reported) | 55 | 2.04 | 2.03 | 1.97 |
| **Reza · ES + NQ + YM · all releases** | **288** | **1.18** | **1.10** | **0.78** |
| Reza · ES · all releases | 98 | 1.17 | 1.06 | 0.65 |
| Aaryen · classifier, 30m, lag 1 (executable) | 55 | 1.08 | 1.03 | 0.84 |
| Reza · ES + NQ · all releases | 195 | 1.09 | 1.01 | 0.72 |
| Reza · YM · all releases | 93 | 1.01 | 0.96 | 0.68 |
| Reza · ES · CPI surprise model | 9 | 0.88 | 0.85 | 0.76 |
| Reza · NQ · all releases | 97 | 0.87 | 0.84 | 0.71 |
| Hoshea · DTW event games | 372 | 0.44 | 0.24 | −0.50 |
| Jack · DTW 5-window ladder (replication) | 315 | 0.31 | 0.10 | −0.68 |

**Answer:** Among the **executable** strategies, the three-equity surprise book (ES + NQ + YM) has the highest Sharpe at Jack's cost (**1.10**, against 1.03 for Aaryen's classifier entered one minute after the release, 0.24 for Hoshea and 0.10 for the Jack replication). At our higher cost Aaryen's executable classifier is slightly ahead (0.84 against 0.78), because it trades about 55 times a year while the surprise book trades about 290. Aaryen's reported Sharpe of about 2 uses entry at the pre-release price and is not executable (Notebook 08).

### 2. What does YM add?

**Answer:** YM on its own is close to ES and NQ (0.96 at Jack's cost, 0.68 at ours), and adding it to ES + NQ raises the equity book's Sharpe from 1.01 to **1.10** at Jack's cost and from 0.72 to **0.78** at ours. Total net P&L at Jack's cost rises from 35.9% to **53.2%** over 2019–2026, with 2.6% of the time in the market and a beta of about zero. The Sharpe without 2022 is 0.79 at Jack's cost (ES + NQ: 0.73). The price is a larger book and drawdown (−8.3% against −7.0%).

### 3. Is the profit from direction?

**Answer:** Yes for every surprise book. Being always long in the same windows loses in all of them (YM −5.5%, ES + NQ + YM −9.8% at Jack's cost), while the surprise books gain, and the direction beats random signs (p ≤ 0.002 for YM and the three-equity book).

### 4. How do the books respond to costs?

**Answer:** The surprise books lose about 0.3–0.4 of Sharpe between Jack's cost and ours, because they trade 90–290 times a year. The DTW books of Jack and Hoshea, which trade 315–372 times a year with about 0.1–0.4 bps per trade, turn clearly negative at our cost. Aaryen's executable classifier loses least, because it trades least.

## Conclusion

1. **The macro-surprise book is the strongest executable strategy in the team at realistic costs, and YM strengthens it.** The ES + NQ + YM book earns a Sharpe of 1.10 at a quarter tick and 0.78 at 2 ticks + $4.50, with about 290 trades a year and no market beta.
2. **YM behaves like ES and NQ in every comparison**, which supports the ES result as a general property of equity-index futures rather than a feature of one contract.
3. **Aaryen's classifier entered one minute late is the closest competitor** (1.03 and 0.84), with far fewer trades; the two approaches use similar surprise information and are a natural candidate for combination.
4. **The DTW strategies remain cost-sensitive**: positive gross, close to zero at a quarter tick, negative at our cost.
5. **Caveats:** YM was added after the ES and NQ results were known; the improvement from adding YM is small relative to the confidence intervals; Jack's ladder is a replication that does not match his slide (Sharpe 0.16 against 0.61 on 2018–2026), and the newer team book (`combined/team_book.py`) should be used for his and Hoshea's current versions.

> **Recommendation:** report the **ES + NQ + YM macro-surprise book** as Reza's main strategy in the team comparison, with the single-market books (ES, NQ, YM) shown next to it.
