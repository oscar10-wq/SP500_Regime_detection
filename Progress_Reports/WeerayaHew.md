# [2026-03-12] Timing Analysis
## Event Study & Lead-Lag Cross-Correlation Analysis
**Script:** `analysis/Event Study/event_study.py`
**Outputs:** `analysis/Event Study/outputs/`

- Enhance the existing Feature Importance and Granger Causality analyses by telling *when* in addition to *whether* Fed rate variables predict regimes (i.e., does the market fall before or after a hike?) This analysis directly tests the **causal timing** of the relationship to address the core research question.

## 1. Event Study

- Identified all **hike cycle starts** from 2000–2025 — defined as the first rate hike after at least 3 consecutive months with no hike. This avoids counting every individual hike within a cycle and focuses on the structural start of tightening.
- Found **23 hike cycle starts** across the dataset.
- For each cycle start (t = 0), extracted a window of **[-12 months, +24 months]** of S&P 500 data.
- Computed **cumulative log-returns** anchored to 0 at t = 0 for each episode.
- Computed the **bear market frequency** (% of episodes in a bear regime) at each window point.

### Results

#### Bear Market Frequency: Pre vs. Post Hike

| Window | Bear Market Frequency |
|---|---|
| Pre-hike average (12 months before) | 23.6% |
| At hike start (t = 0) | 21.7% |
| Post-hike 1–12 months | 18.5% |
| Post-hike 13–24 months | 11.2% |

#### Key Findings

1. **Bear market frequency falls after hikes start, not rises.** The post-hike period (18.5% and 11.2%) is consistently less bearish than the pre-hike period (23.6%). This is the opposite of what the "hikes trigger bear markets" hypothesis predicts.

2. **On average, the S&P 500 continues rising after the first hike.** The average cumulative return (blue line in the spaghetti plot) is positive and gently upward across the entire window. Hike cycle starts do not mark market peaks.

3. **There is large heterogeneity across cycles.** The wide standard deviation band reflects the fact that some cycles ended in bear markets (e.g. 2007, 2022) while others continued in bull conditions (e.g. 2015–2018). This heterogeneity itself is evidence against a reliable trigger mechanism.

4. **Markets are already rising before hikes begin.** The strong pre-hike bull trend (left of t = 0) confirms that hikes happen *during* good economic conditions, not as the onset of decline.

---

## 2. Lead-Lag Cross-Correlation

For each Fed variable and lag L ∈ [−24, +24] months, computed:

$$\text{CCF}(L) = \text{corr}(\text{Fed}[t],\ \text{Bear}[t + L])$$

- **L > 0**: Fed at time t is correlated with bear regime L months later → **Fed leads Bear** (consistent with trigger hypothesis)
- **L < 0**: Fed at time t is correlated with bear regime that occurred |L| months earlier → **Bear leads Fed** (consistent with reaction hypothesis)

Three Fed variables tested:
- `Fed_Funds_Rate` — the absolute level of the policy rate
- `Fed_Change` — month-over-month change in the rate
- `Fed_Cycle` — persistent hike/cut direction (+1 or −1), carrying forward the last move

95% significance threshold: ±0.11 (n ≈ 300 observations)

### Results

| Variable | Peak Lag | Peak r | Interpretation |
|---|---|---|---|
| Fed_Funds_Rate (level) | +22 months | +0.40 | Fed leads Bear by ~2 years |
| Fed_Change (monthly change) | +13 months | −0.28 | Rate increases associated with fewer bear markets 13m later |
| Fed_Cycle (direction) | +7 months | −0.42 | Hiking cycles strongly co-occur with bull markets |

#### Key Findings

1. **Fed rate level peaks at lag +22m (r = 0.40).** High rate levels are associated with bear markets arriving roughly 2 years later. This supports a *cumulative tightening* effect — not a sharp trigger. Importantly, the contemporaneous correlation at lag 0 is near zero (~0.05), meaning high rates and bear markets do not coincide right now.

2. **Fed_Change peaks at +13m with a negative correlation (r = −0.28).** Rate *increases* are associated with fewer bear markets 13 months later, not more. This is because the Fed only raises rates when the economy is strong (bull market). The negative sign directly contradicts the trigger hypothesis.

3. **Fed_Cycle is the most striking panel — all bars are negative (red).** The peak correlation is −0.42 at +7 months, meaning active hiking cycles are strongly associated with *bull* market conditions. The Fed hikes precisely during the good times.

4. **At negative lags (L < 0) for Fed_Change, correlations are positive and significant.** This means bear markets at time t are correlated with Fed rate *increases* happening in the future — i.e., bear regimes precede rate changes, not the other way around. The Fed *reacts* to deteriorating conditions, not the cause of them.

---

## Overall Conclusion

Both analyses converge on the same finding:

> **Fed rate hikes are not a reliable trigger for bear market regimes. Historically, hike cycles begin during bull markets (strong economy, rising prices), and bear markets are more likely to arrive before or long after hikes — not immediately after. The 2022 bear market (rapid 425bp hikes in 12 months) is the notable exception, which explains why the level variable shows a positive correlation at long lags.**

This directly supports and strengthens the findings from Feature Importance (Fed rate ranked #7.7/12 average) and Granger Causality (Fed variables do not independently Granger-cause regimes after controlling for market signals).

### Limitations

- **23 hike cycles is a small sample** — some results (especially the event study averages) are sensitive to individual episodes like 2007–2008 and 2022.
- **Cross-correlation is not causation** — it measures linear co-movement at different lags, but does not control for confounders (same limitation as basic Granger, partially addressed by the conditional Granger analysis).
- **The rule-based 20% regime definition** means bear markets are labelled retrospectively, which may slightly distort the timing in the lead-lag analysis.