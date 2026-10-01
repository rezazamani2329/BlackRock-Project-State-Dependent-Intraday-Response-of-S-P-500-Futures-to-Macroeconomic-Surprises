# 12. Purpose, Research Questions, Answers, and Conclusion (YM)

## Purpose of Notebook 05b (YM)

Notebook 04b showed that trading YM in the direction of the **CPI surprise** is profitable after costs (naive sign Sharpe 0.99), but with only about 10 trades a year. Notebook 05b repeats Notebook 05 on the E-mini Dow and asks:

> **If we trade the surprises of many US macro releases on YM, not only CPI, how many trades do we get, and do we keep a positive, credible edge after costs?**

**Design (identical to Notebook 05):** entry at the close of the first post-release bar (T+1), exit 30 minutes after the release, walk-forward over **2019–2026**; each year, using only earlier releases, an impact screen picks the 10 families that move YM most, each line's sign comes from its historical first-minute reaction on YM, and costs are 2 ticks + $4.50 per round trip (about 1.0 bps on YM).

**Data check.** 179 Bloomberg lines group into 75 families with at least 30 releases. 5,708 of 6,097 release timestamps since 2016 have a complete 30-minute YM window. The 30-minute trade return is identical to Notebook 04b on all 77 matched CPI releases.

### Which releases does YM react to?

**Answer:** The same core as ES and NQ. **NFP** ranks first or second every year, **CPI** first from 2023, followed by the **FOMC decision** (and the Fed's interest-on-reserves line, which prints at the same minute), **ISM Manufacturing**, **ISM Services**, **Retail Sales**, JOLTS and PPI. U. of Michigan Sentiment enters the top 10 from 2023. Trade Balance ranked high in 2019–2022 and then dropped out.

### Does the signed signal relate to YM's reaction?

**Answer:** Yes, for the main families. The correlation between the family signal and YM's first-minute jump in 2019–2026 is **+0.54 for CPI**, +0.61 for U. of Michigan, +0.32 for PPI, +0.28 for Retail Sales and +0.25 for NFP. Case-Shiller (−0.63) and FHFA (−0.13) have the wrong sign out of sample: their historical signs did not carry over.

### Does trading many releases work?

**Answer:** Yes, with the simplest rule. Over 2019–2026, net of costs:

| Strategy | Trades / yr | Net bps / trade | Net Sharpe | 90% CI | Sharpe ex-2022 | Max DD | P(random sign) |
|---|---|---|---|---|---|---|---|
| CPI · naive | 11 | 4.5 | 0.54 | −0.18 to 1.28 | −0.05 | −2.3% | 0.030 |
| CPI · signal × vol | 11 | 8.8 | 0.61 | −0.11 to 1.33 | 0.03 | −2.8% | 0.027 |
| **All · naive** | **93** | 1.7 | **0.68** | **0.08 to 1.37** | **0.40** | −4.5% | **0.006** |
| All · surprise model | 57 | 1.3 | 0.34 | −0.35 to 0.91 | 0.13 | −5.6% | 0.061 |
| All · signal × vol | 96 | 1.1 | 0.29 | −0.36 to 1.13 | −0.05 | −12.8% | 0.116 |
| All · naive · 2nd window | 93 | −0.4 | −0.15 | −0.66 to 0.51 | −0.40 | −7.4% | 0.237 |

The **multi-release naive book** is the best YM result: about **93 trades a year on 90 days**, a net Sharpe of **0.68** with a 90% interval above zero, 12.0% total net P&L, and a direction that beats random signs (p = 0.006). It is the only book that stays clearly positive without 2022 (0.40). This is almost the same as ES (0.65) and NQ (0.71).

### Which releases earn money?

**Answer:** CPI (+3.8% naive, +6.3% with signal × vol), ISM Manufacturing (+3.3%), JOLTS (+2.3%), Retail Sales (+1.7%) and Trade Balance (+1.2%) earn the most. **NFP earns nothing** in YM (−0.05 bps per trade, 43% hit), and Wholesale Inventories (−2.7%) and Wholesale Trade Sales (−1.6%) lose. As in ES and NQ, the broad book is a diversified set of small edges, not one release.

### Does the edge last beyond 30 minutes?

**Answer:** No. Trading the second 30-minute window (T+30 to T+60) loses (−0.4 bps per trade, Sharpe −0.15), and the two-window book loses too (−0.12). The surprise's edge on YM is spent within the first 30 minutes, as in ES and NQ.

### Is it robust to costs and market-neutral?

**Answer:** Yes. The naive multi-release book has a Sharpe of 0.91 before costs, 0.68 at 2 ticks and 0.45 at 4 ticks. Beta to YM buy-and-hold is between −0.015 and +0.009 for every strategy, and time in market is below 1% (1.8% for the two-window book).

---

## The CPI Surprise Definition (as in ES and NQ)

The YM **CPI naive** book here (Sharpe 0.54) is weaker than in Notebook 04b (0.99), **with identical trade returns**. The reason is the same as in ES and NQ: Notebook 04 uses the headline CPI MoM surprise, while Notebook 05 averages all CPI lines, and the two disagree in direction on releases where the headline surprise is close to zero. The CPI-only books here also trade a few more releases (85 against 77), because the history requirement is shorter.

---

## Final Comparison: CPI Only vs CPI + Other Surprises (YM)

| | **CPI only** (best: signal × vol) | **CPI + other surprises** (All · naive) | Peer: Jack DTW (ES) |
|---|---|---|---|
| Trades per year | 11 | **93** | 185 |
| Trading days per year | 11 | **90** | — |
| Net bps per trade | **8.8** | 1.7 | 1.2 |
| Total net P&L (2019–26) | 7.5% | **12.0%** | 18.6% (2018–26) |
| Net Sharpe | 0.61 | **0.68** | 0.61 |
| Max drawdown | −2.8% | −4.5% | −4.9% |
| Sharpe ex-2022 | 0.03 | **0.40** | — |
| Sharpe at 4-tick costs | — | **0.45** | — |

---

## Conclusion

**1. YM supports a broad macro-surprise book.** Trading the surprises of about 10 screened releases gives **~93 trades a year on ~90 days**, against about 11 for CPI only.

**2. The edge matches ES and NQ.** The multi-release naive book earns a **net Sharpe of 0.68** (ES 0.65, NQ 0.71), with a −4.5% drawdown and a direction that beats random signs (p = 0.006). It is the only YM book that remains positive without 2022.

**3. Simple beats fitted.** The surprise model and the volatility-scaled versions do worse than trading the sign; sizing by volatility mainly adds drawdown (−12.8%).

**4. The edge lasts about 30 minutes.** The second window loses, as in ES and NQ.

**5. Caveats:** the screen admits some families whose historical signs failed out of sample (Case-Shiller, FHFA, Wholesale Inventories); NFP, the top-ranked release by impact, earns nothing in YM; and returns are per unit of notional before leverage.

> **Overall:** on YM, trading the macro surprise across about 10 major US releases gives ~93 trades a year with a net Sharpe of 0.68, close to ES and NQ. YM is a third equity market for the same strategy; Notebook 06b tests whether DTW adds anything on YM, and Notebook 07b whether adding YM to ES and NQ improves the portfolio.
