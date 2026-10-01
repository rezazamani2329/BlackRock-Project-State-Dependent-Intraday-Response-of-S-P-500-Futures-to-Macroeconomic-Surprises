## 8. Interpretation and Conclusion (with YM)

### Purpose of Notebook 07b

Notebook 07 applied the surprise and DTW strategies to ES, NQ and ZN and combined them. Notebooks 02b–06b (YM) showed that the E-mini Dow behaves like ES. Notebook 07b runs the same pipeline, unchanged, on **four markets (ES, NQ, ZN, YM)** and asks:

> **Does the surprise strategy work on YM inside the common multi-market pipeline, and does adding YM to ES and NQ improve the equity portfolio?**

**Data check.** All four markets have 6,097 release timestamps since 2016; 94–97% have a complete 30-minute window (YM 5,708, ES 5,920, NQ 5,852, ZN 5,878). ES reproduces Notebook 05 exactly (max difference 1e-14 bps), and the YM rows reproduce Notebooks 05b–06b (YM).

### 1. Does YM react to the same news, in the same direction?

**Answer:** Yes. The first-minute jumps of YM and ES after CPI move almost one for one (correlation **0.99**; YM–NQ 0.96, YM–ZN 0.86), and the correlation of YM's jump with the CPI MoM surprise is −0.56 (ES −0.58, NQ −0.60). The line signs estimated on 2016–2018 are the same for YM as for ES and NQ: hot inflation pushes all equity indices down, and strong payrolls, retail sales and ISM push them up (the opposite for ZN).

### 2. How does YM compare with the other markets? (2019–2026, net of costs)

| Market | Median cost (bps) | All · naive Sharpe | Trades / yr | Ex-2022 | Max DD | CPI model Sharpe | All · surprise + DTW agree |
|---|---|---|---|---|---|---|---|
| ES | 1.57 | 0.65 | 98 | 0.28 | −4.8% | 0.76 | 0.31 |
| NQ | 0.61 | 0.71 | 97 | 0.44 | −5.7% | 0.55 | 0.62 |
| **YM** | **0.95** | **0.68** | **93** | **0.40** | **−4.5%** | 0.45 | **0.94** |
| ZN | 2.93 | −2.42 | 82 | −2.53 | −20.6% | −0.37 | −1.57 |

**Answer:** The multi-release book on YM (0.68) sits between ES (0.65) and NQ (0.71), with the smallest drawdown of the three, and the direction beats random signs (p = 0.006). YM's costs (about 0.95 bps) are between NQ's and ES's. At 4 ticks per round trip YM keeps 0.45 (ES 0.20, NQ 0.60). ZN loses at every cost level above zero, as in Notebook 07.

YM is also the one market where requiring DTW to agree raises the Sharpe (0.94, against 0.31 in ES and 0.62 in NQ); see Notebook 06b (YM) for why this is not yet treated as a confirmed result.

### 3. Which releases earn money on YM?

**Answer:** CPI (+3.8% total net), ISM Manufacturing (+3.3%), JOLTS (+2.3%), Retail Sales (+1.7%) and Trade Balance (+1.2%), close to the ES list (CPI, JOLTS, Retail Sales). NFP earns nothing on YM (−0.05%) and loses on ES and NQ. Wholesale Inventories (−2.7%) and Wholesale Trade Sales (−1.6%) lose on YM only.

### 4. Does YM diversify the equity book?

**Answer:** More than expected. The first-minute reactions are almost identical, but the monthly net returns of the multi-release books are less correlated: **YM–ES 0.66 and YM–NQ 0.43**, against ES–NQ 0.79. For the CPI model, YM is closer (0.86 with ES, 0.76 with NQ). The different families each market admits and trades explain most of this.

### 5. Portfolios (equal risk, weights fixed on 2016–2018: ES 1.00, NQ 0.71, YM 1.02, ZN 2.67)

| Portfolio | Trades / yr | Net Sharpe | 90% CI | Ex-2022 | Total net | Max DD | Random-sign p |
|---|---|---|---|---|---|---|---|
| 2-market (ES+NQ) · All · naive | 195 | 0.72 | −0.01 to 1.44 | 0.38 | 25.2% | −7.4% | 0.000 |
| **3-equity (ES+NQ+YM) · All · naive** | **288** | **0.78** | **0.10 to 1.50** | **0.43** | **37.5%** | −9.2% | **0.000** |
| 2-market (ES+NQ) · CPI · surprise model | 20 | 0.69 | −0.01 to 1.39 | 0.02 | 9.9% | −3.3% | 0.023 |
| 3-equity (ES+NQ+YM) · CPI · surprise model | 29 | 0.63 | −0.10 to 1.37 | −0.04 | 13.0% | −5.3% | 0.020 |
| 4-market · All · naive | 370 | −0.34 | −1.08 to 0.40 | −0.75 | −17.4% | −23.4% | 0.000 |
| 4-market · CPI · surprise model | 29 | 0.61 | −0.13 to 1.35 | −0.08 | 12.6% | −5.8% | 0.018 |

**Answer:** Adding YM to ES and NQ improves the multi-release equity book: the Sharpe rises from **0.72 to 0.78**, total net P&L from 25% to 38%, trades from 195 to 288 a year, and the 90% confidence interval moves fully above zero (0.10 to 1.50). The Sharpe without 2022 also rises (0.38 to 0.43). The cost is a larger drawdown (−9.2% against −7.4%), because the book is bigger. For the CPI-only model YM adds little (0.69 to 0.63), because the CPI books of the three indices are highly correlated.

**ZN still breaks every portfolio it enters.** Equal-risk weighting gives ZN the largest weight (2.67, because its moves are small), and its multi-release book loses (−2.42), so the four-market naive book is negative (−0.34). As in Notebook 07, ZN should be excluded from the traded book.

## Conclusion

1. **YM is a third equity market for the same strategy.** Inside the common pipeline it earns a multi-release Sharpe of 0.68 (ES 0.65, NQ 0.71), with the smallest drawdown of the three and a direction that beats random signs.
2. **The best book is the three-equity portfolio (ES + NQ + YM), All · naive:** about 290 trades a year, net Sharpe **0.78** with a 90% interval above zero, 0.43 without 2022, 37.5% total net P&L over 2019–2026 and no market beta. It improves on ES + NQ (0.72) because the YM book is less correlated with the ES and NQ books than its price reaction suggests.
3. **ZN should stay out of the traded book**: its edge does not survive its tick size, and equal-risk weighting would give it the largest weight.
4. **Caveats:** the portfolio weights and all strategy settings were fixed before YM was added, but YM was added after the ES and NQ results were known; the improvement from 0.72 to 0.78 is small relative to the confidence intervals; and the 2022 inflation year still contributes a large share of the P&L.

> **Recommendation:** use the **ES + NQ + YM multi-release surprise book** as the main equity book in the team comparison (Notebook 08b), with ZN excluded.
