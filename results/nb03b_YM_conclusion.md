# 14. Purpose, Research Questions, and Answers (YM)

## Purpose of Notebook 03b (YM)

Notebook 03 showed for ES that the reaction to a CPI surprise depends on the market's state before the release, mainly 20-day volatility, and Notebook 03b showed a weaker version of the same pattern in NQ. This notebook repeats the full Notebook 03 analysis on the **E-mini Dow (YM)**, with the same code, states, tests and selection rule. It asks:

> **Does YM's reaction to the same surprise depend on the pre-release state, and is it the same state as in ES?**

**Data.** 348 of the 363 CPI, NFP and PCE releases (2016–2026) have complete YM prices. Six primary states (20-day realised volatility, previous-day intraday volatility, 20-day momentum, 60-day drawdown, overnight return, previous surprise) are measured before each release, converted to trailing percentiles, and checked for look-ahead (all checks passed). 64 quarterly rolls are removed from the multi-day features.

### Section 2: Is the surprise signal defined correctly for YM?

**Answer:** Yes. With the same signs as ES (a hot CPI or PCE print is bad news, strong payrolls good news), the correlation between the signed surprise $x$ and YM's 5-minute return is **0.51 for CPI**, 0.30 for PCE and **0.22 for NFP**. CPI and PCE are close to ES and NQ; NFP is clearly stronger than in NQ (0.03), in line with Notebook 02b, where payrolls explained more of YM's move than of ES's or NQ's.

### Section 4: What did YM's state look like?

**Answer:** 20-day realised volatility has a median of about **11.8%** (ES about 13%, NQ about 16%) and peaks at about 93% in March 2020; the 60-day drawdown reaches −39%. The six states are only moderately correlated (largest |ρ| = 0.70, between previous-day volatility and drawdown), so they measure different things.

### Section 5: How much of YM's reaction remains after the first minute?

**Answer:** A meaningful part. The mean absolute drift from T+1 to T+30 is **18.3 bps after CPI**, 17.5 bps after NFP and 10.2 bps after PCE, against a total 30-minute reaction of 33.2, 27.9 and 11.9 bps (Notebook 02b, YM).

### Section 7–9: Does the state change the surprise beta? (48 primary tests)

**Answer:** Yes for CPI, and the effect is the same as in ES.

- **CPI × 20-day volatility** is the strongest result. The 5-minute CPI beta rises by **+58.7 bps** from the calmest to the most volatile state (t = 3.74, permutation p = 0.004, BH q = 0.014), almost the same size as in ES (about +61 bps) and NQ (+58 bps). By bucket, the 5-minute beta is 7.6 bps per SD in low-volatility states, 42.8 in mid and 51.8 in high. The sign is the same in 2016–2020 (t = 1.21) and 2021–2026 (t = 2.52), and the effect is stronger in the later period, as in ES.
- **On the tradeable drift** (T+1 to T+30) the same interaction is +24.6 bps (t = 2.52, permutation p = 0.03), but it does not survive the false-discovery correction (q = 0.11).
- **NFP × volatility** (20-day and previous-day) is significant with a negative sign (d = −17 and −21 bps, t ≈ −3.1, q = 0.04): strong payrolls help YM less when volatility is high. But this effect comes entirely from 2016–2020 (t = −3.7 and −3.2); in 2021–2026 it is not distinguishable from zero (t = −0.10 and −0.95). This is the same pattern Jack found for NFP in ES, clean through 2020 and then gone.

### Section 10: How many results survive multiple testing?

**Answer:** About **2.4** false positives are expected by chance; **10** tests are significant before correction (8 by permutation), and **3** survive the Benjamini–Hochberg FDR at q < 0.1: CPI × 20-day volatility and the two NFP × volatility effects. This is more than in ES (2) and NQ (0).

### Section 11: Do states predict the size of the move?

**Answer:** Yes, strongly, as in ES and NQ. **156 of 360** magnitude tests survive the FDR correction (ES 198, NQ 154). High pre-release volatility and deep drawdowns predict larger YM moves after every release, independent of the direction of the surprise (for example, previous-day volatility on the 30-minute drift: t = 5.2; 60-day drawdown: t = −5.0).

## Conclusion

YM repeats the ES picture almost exactly:

1. **Direction:** CPI surprises move YM much more when 20-day volatility is high (+59 bps from calm to stressed, t = 3.7, survives FDR), the same state, sign and size as in ES and NQ. The effect is concentrated in the first minutes and is weaker on the tradeable drift.
2. **NFP × volatility** survives FDR but only because of 2016–2020; it should not be traded.
3. **Size:** volatility and drawdown strongly predict how large the YM move will be (156 of 360 tests survive FDR).

## Implications for Notebook 04b (YM)

| Model | Use |
|---|---|
| **Surprise only** | The benchmark: trade the sign of the CPI surprise, entry at T+1, exit at T+30 |
| **Surprise + State** | Uses `rv_20d`, the state selected in ES and confirmed here for YM; Notebook 04b tests whether it improves the trading result out of sample |
| **NFP** | Not a candidate for state-based trading: the NFP × volatility effect does not hold after 2020 |

**Caveats:** about 113–119 events per family (about 40 per tercile); the selection rule picks three state–family pairs, but only the CPI × volatility one is stable across periods; and, as in ES, the CPI effect is strongest after 2021, so it may reflect the 2022 inflation regime rather than a permanent feature.
