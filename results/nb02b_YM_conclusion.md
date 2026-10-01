# 8. Purpose, Research Questions, Answers, and Conclusion (YM)

## Purpose of Notebook 02b (YM)

Notebooks 01–02 showed for ES that CPI is the release whose surprise moves the market, and Notebook 02b did the same for NQ and ZN. This notebook repeats the Notebook 02 event study for the **E-mini Dow (YM)**, with ES and NQ as references, on the same 363 CPI, NFP and PCE releases (2016–2026). It asks whether YM reacts like ES, and whether part of its reaction remains after the first minute, where it can be traded.

**Data check.** 348 of 363 releases have a complete 60-minute YM window (ES 363, NQ 354), so coverage is close to the other equity markets.

### Section 4: How much does each release move YM?

**Answer:** YM moves a little less than ES and much less than NQ, with the same ranking of releases. Mean absolute 30-minute move after CPI: **YM 33.2 bps**, ES 38.5, NQ 53.6. After NFP: 27.9, 29.6 and 36.0. After PCE: 11.9, 14.9 and 18.5. Relative to its own average release, YM's profile is almost identical to ES: CPI 1.37 (ES 1.39), NFP 1.15 (ES 1.07), PCE 0.49 (ES 0.54).

### Section 5: Does the surprise explain YM's reaction?

**Answer:** Yes for CPI, as in ES. The CPI surprise explains **28% of YM's first-minute move and 27% of the 5-minute move** (ES 30% and 27%, NQ 32% and 30%). A one-standard-deviation hot headline CPI print lowers YM by **19.2 bps over 5 minutes** (t = −2.83) and 20.9 bps over 30 minutes (t = −2.54), about 92% of the ES response (−20.8 and −23.8) and 70% of NQ's (−27.7). As in ES, core CPI adds nothing once headline CPI is in the regression.

NFP is the one difference. Its surprise explains more of YM's move (R² 0.07–0.14 at 5–60 minutes) than of ES's (0.03–0.07) or NQ's (about 0.01), consistent with the Dow's tilt to industrials and banks, which gain from strong growth. No single NFP component is significant, however (|t| ≤ 1.06). PCE behaves as in ES: a small, mostly first-minute effect.

### Section 6: Is there a tradeable drift after the first minute?

**Answer:** The same weak drift as in ES. From T+1 to T+30, the headline CPI coefficient is **−5.3 bps per SD (t = −1.37)**, almost identical to ES (−5.7, t = −1.37) and NQ (−4.6, t = −0.90). Its sign is the same in 2016–2020 (−5.95, t = −1.88) and 2021–2026 (−5.54, t = −0.79). No release passes the strict rule (drift |t| > 2 with a stable sign) in YM, and none does in ES or NQ either. PCE's drift is significant in 2016–2020 (t = −2.25) but reverses sign in 2021–2026, so it is not reliable.

## Conclusion

1. **YM behaves like a slightly smaller ES.** Same releases, same signs, about 90% of ES's CPI response, and the same share of the move explained by the surprise.
2. **The CPI drift is present but, on CPI alone, not significant**, exactly as in ES and NQ. In ES and NQ the tradeable edge was established in Notebooks 04–05, which aggregate releases with walk-forward signs and test the trading P&L directly, not by this single-release regression. The same tests are needed for YM.
3. **NFP may matter more for YM than for ES and NQ**, which could change which families the screen admits in Notebook 05b.
4. **Expectation for Notebooks 03b–07b (YM):** results close to ES, and a YM book highly correlated with the ES book. YM is therefore more useful as an out-of-sample check of the ES result on a third equity index than as a source of diversification.

**Next:** Notebook 03b (YM).
