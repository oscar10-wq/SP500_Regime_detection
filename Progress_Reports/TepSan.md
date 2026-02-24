# Week 1: Feature Importance Analysis

### Methodology
- Trained 3 classification models on 12 features (excluding SPX_Close to prevent data leakage) over 312 monthly observations (Jan 2000 – Dec 2025)
- Used a chronological 80/20 train/test split to respect the time-series nature of the data
- Extracted feature importance rankings from each model:
  - **Random Forest** — Mean Decrease in Impurity (MDI)
  - **Gradient Boosting** — Mean Decrease in Impurity (MDI)
  - **Logistic Regression** — Absolute coefficient magnitude (on standardised features)

### Results

#### Model Accuracy
| Model               | Accuracy |
|---------------------|----------|
| Logistic Regression | 90.48%   |
| Random Forest       | 80.95%   |
| Gradient Boosting   | 80.95%   |

#### Fed Funds Rate Ranking Across Models
| Model               | Fed Rate Rank | Top Feature |
|---------------------|---------------|-------------|
| Logistic Regression | #2 / 12       | SPX_Volume  |
| Random Forest       | #11 / 12      | SPX_RSI     |
| Gradient Boosting   | #10 / 12      | SPX_RSI     |

**Cross-model average rank: 7.7 / 12**

#### Top 5 Features (Random Forest)
1. SPX_RSI (0.2339)
2. SPX_ROC (0.1848)
3. SPX_MACDH (0.1321)
4. SPX_MACD (0.0936)
5. VIX_Close (0.0774)

### Key Findings

1. **Fed Funds Rate is NOT the dominant driver of regime switches.** Tree-based models (Random Forest, Gradient Boosting) consistently rank it near the bottom (#11 and #10 out of 12).

2. **Technical momentum indicators are the strongest predictors.** SPX_RSI and SPX_ROC dominate across both tree-based models, suggesting that price momentum patterns are more informative than macroeconomic variables for detecting regime changes.

3. **Fed Funds Rate does have a strong linear relationship with regimes.** Logistic Regression ranks it #2, indicating that when considered in isolation (linear model), higher rates are associated with bull regimes. This makes economic sense — the Fed tends to raise rates during strong economies.

4. **The VIX (volatility index) outranks all macro indicators in Random Forest.** Market-implied volatility captures regime-switching dynamics better than GDP, unemployment, inflation, or interest rates.

5. **Class imbalance is a limitation.** Bull months outnumber bear months roughly 5:1 (260 vs 52), which biases the tree-based models toward predicting Bull. Logistic Regression handles this better, achieving 90.48% accuracy with 50% bear recall.


### Interpretation — What This Means

1. **RSI and ROC are #1 and #2** → The market's own **momentum** is the best predictor of regime switches. When momentum slows down (RSI drops, rate of change weakens), a crash is approaching. Think of it like a car — you can tell it's about to stop by watching it decelerate, not by looking at the fuel price.

2. **VIX outranks all macro variables** → **Fear in the market** (VIX = the "fear index") is a better signal than any economic data the Fed uses. When traders panic and start buying protection, that's a stronger regime signal than interest rates.

3. **Fed rate ranks near the bottom in tree models** → Interest rates change **slowly** (the Fed adjusts every few months). But regime switches happen based on **fast-moving market dynamics**. By the time the Fed reacts, the market has already moved.

4. **But Logistic Regression ranks Fed rate #2** → There IS a correlation — higher rates tend to coincide with bull markets (strong economy → Fed raises rates). But **correlation ≠ causation**. The Fed raises rates *because* the economy is strong, not the other way around.

> **One-liner for the presentation:** *"Markets predict their own regime switches through momentum and volatility signals. Fed rate hikes are a symptom of the economic cycle, not the trigger for bear markets."*

### Conclusion
The answer to the research question is **nuanced**: Fed Funds Rate is a contributing factor to regime identification, but it is not the primary trigger. The market's own momentum signals (RSI, ROC) and volatility (VIX) are stronger predictors of when the S&P 500 transitions between bull and bear regimes.
