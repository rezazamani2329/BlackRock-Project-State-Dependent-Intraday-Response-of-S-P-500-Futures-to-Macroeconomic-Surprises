# Purpose, Research Questions, Answers, and Conclusion (YM)

## Purpose of Notebook 06b (YM)

Notebook 06 showed for ES that a DTW pattern signal adds no reliable directional information to the macro surprise, and Notebook 06b found the same for NQ. Notebook 05b showed that the multi-release surprise book works on YM (Sharpe 0.68, ~93 trades a year). Notebook 06b repeats Notebook 06 on the E-mini Dow:

> **Does the shape of YM's price path around a release add information beyond the surprise itself?**

**Design (identical to Notebook 06):** the peer DTW specification (31-minute z-scored path ending at entry, same clock slot, 3-year lookback, 600 most recent candidates, k = 15, Sakoe–Chiba band 5, two-year recency weighting); the DTW score is gated outside the middle 40% of its trailing three-year distribution; walk-forward 2019–2026, costs of 2 ticks + $4.50 per round trip. Nothing is re-tuned for YM.

**Data check.** 5,711 of 6,097 release timestamps since 2016 have a complete path; the DTW signal is computed for 5,232 releases in 59 clock slots. The 30-minute trade return matches Notebook 05b exactly.

### Does DTW forecast the direction of YM's next 30 minutes?

**Answer:** Weakly. The rank IC of the DTW forecast with the next 30-minute return is **+0.030** over all 3,949 releases in 2019–2026 (hit rate 51.0%) and **+0.069** on the 746 releases the surprise book trades (52.7%). By year it ranges from −0.040 (2024) to +0.101 (2025–2026) and is negative in 2 of 8 years. This is about the same weak level as in ES and NQ.

### Is DTW the same information as the surprise?

**Answer:** No. The correlation between the DTW score and the signed surprise is 0.13, and their signs agree on only 51.5% of releases, so DTW is close to independent of the surprise.

### Does combining them improve the book?

**Answer:** For the multi-release book on YM, **yes in the point estimates, but not significantly**, which is different from ES and NQ.

| Book (~10 releases) | Trades / yr | Net bps / trade | Net Sharpe | 2019–21 | 2022–26 | Ex-2022 | Max DD | vs surprise only: P(not better) |
|---|---|---|---|---|---|---|---|---|
| Surprise only (Notebook 05b) | 93 | 1.7 | 0.68 | 0.61 | 0.72 | 0.40 | −4.5% | — |
| DTW only | 61 | 1.3 | 0.47 | 0.60 | 0.40 | 0.58 | −4.0% | 0.67 |
| Surprise + DTW agree | 49 | 2.9 | 0.94 | 1.11 | 0.84 | 0.94 | −1.7% | 0.23 |
| **Surprise + DTW blend** | 70 | 2.0 | **1.06** | 1.08 | 1.04 | 0.87 | **−1.5%** | **0.06** |

When DTW agrees with the surprise, the trade earns **2.9 bps** (hit 51.5%); when it disagrees, **0.3 bps** (hit 48.0%). The blend raises the Sharpe from 0.68 to 1.06 and cuts the drawdown from −4.5% to −1.5%, and the improvement holds in both 2019–2021 and 2022–2026. But the bootstrap probability that the blend is not better than surprise only is **6%**, and that the agreement filter is not better is 23%, so neither passes a 5% test.

**For CPI alone, DTW does not help:** surprise only 0.45, DTW agree 0.32, blend 0.51 (P(not better) 0.39–0.76).

### Does DTW extend the trading window? (The ladder)

**Answer:** No. Beyond the first 30 minutes both signals lose: the surprise's second to fifth windows have Sharpe ratios of −0.15 to −0.80, and DTW's −1.03 to +0.15. The five-window DTW ladder loses (Sharpe −0.23, 298 trades a year, −12.4% drawdown), the surprise ladder loses more (−0.34), and surprise in the first window plus DTW in windows 2–5 earns nothing (0.00). On YM, as on ES and NQ, the information is spent within 30 minutes of the release.

## Final Comparison: Surprise Only vs Surprise + State vs Surprise + Pattern (YM)

| Model | Trades / yr | Net Sharpe | Max DD | Verdict |
|---|---|---|---|---|
| Surprise only — CPI (Notebook 04b) | 6 | 0.84 | −1.9% | Positive, few trades |
| Surprise + State — CPI (Notebook 04b) | 7 | 0.83 | −1.9% | No improvement |
| Surprise only — ~10 releases (Notebook 05b) | 93 | 0.68 | −4.5% | The benchmark |
| **Surprise + Pattern (blend) — ~10 releases** | 70 | **1.06** | **−1.5%** | Best point estimate; P(not better) = 0.06 |
| Surprise + Pattern (agree) — ~10 releases | 49 | 0.94 | −1.7% | Better, not significant |
| Pattern only (DTW) — ~10 releases | 61 | 0.47 | −4.0% | Weaker than surprise |
| DTW 5-window ladder | 298 | −0.23 | −12.4% | Loses |

## Conclusion (YM)

1. **DTW has weak directional information on YM** (rank IC +0.03, hit rate 51%), the same level as in ES and NQ.
2. **YM is the one market where combining DTW with the surprise looks better**: the blend raises the multi-release Sharpe from 0.68 to 1.06 and cuts the drawdown by two-thirds, in both halves of the sample. In ES the agreement filter helped in 2018–2020 and hurt in 2021–2026, and in NQ no combination beat the surprise.
3. **This should not be read as a new result yet.** The improvement is not significant at 5% (P = 0.06 for the blend), it is one of several combinations tested on several markets, and it does not appear in ES or NQ with the same settings. The honest reading is that DTW may add a little on YM, and that this needs an out-of-sample check before it is used.
4. **The surprise's edge lasts about 30 minutes**, and the ladder adds only losing trades.
5. **Recommendation for YM:** keep **surprise only, first 30 minutes, ~10 releases** (Sharpe 0.68) as the main model, the same as for ES and NQ, and report the surprise + DTW blend (1.06) as a promising but unconfirmed variant.

> **Overall:** YM confirms that the macro surprise in the first 30 minutes is the robust directional signal. Unlike in ES and NQ, DTW agreement appears to improve it on YM, but the evidence is not strong enough to change the model. Notebook 07b adds YM to the multi-market portfolio.
