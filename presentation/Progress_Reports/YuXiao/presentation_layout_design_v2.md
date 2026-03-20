# Presentation Layout Design V2

## Slide 1 — Question: Do Fed Hikes Trigger Bear Markets?

**Main message**  
The presentation asks one central question: are Fed hikes the trigger for S&P 500 bull-to-bear regime switches?

**Key points**

* This is a common market narrative, but it is rarely tested directly.
* The project compares Fed variables with macro indicators and market technical signals.
* The goal is not just to find correlation, but to assess timing, explanatory value, and predictive usefulness.
* The presentation is structured to answer the trigger question as clearly as possible.

**Method (optional, very brief)**  
Monthly S&P 500 regime data from January 2000 to January 2026, with regimes defined by the 20% rule.

**Figure suggestion**

* No figure.
* Layout: keep the slide clean with title and subtitle only.

**Speaker guidance (IMPORTANT)**  
This presentation focuses on one simple but important question: when the S&P 500 switches from a bull market to a bear market, is a Fed hiking cycle usually the trigger? That is the story we often hear in financial commentary, but it needs to be tested carefully. To do that, I compare Fed variables with broader macro indicators and with market-based technical signals. The structure of the talk is designed around evidence, not around methods. So each section moves us closer to a direct answer: is the Fed the trigger, or is something else more closely linked to regime change?

## Slide 2 — Data: One Monthly Regime Dataset

**Main message**  
The whole analysis is built on one monthly dataset that makes policy, macro, and market signals directly comparable.

**Key points**

* The dataset combines S&P 500, VIX, Fed Funds Rate, GDP, inflation, unemployment, and the 10Y2Y spread.
* All variables are aligned to monthly frequency to create a common observation unit.
* The target is a bull/bear regime label constructed from the standard 20% market rule.
* `SPX_Close` is excluded from prediction features to avoid leakage into the target.

**Method (optional, very brief)**  
Lower-frequency macro series are forward-filled to monthly dates, then technical and lagged features are added where needed.

**Figure suggestion**

* Use `/mnt/data/project_workflow_diagram.png`.
* Layout: left text, right figure.

**Speaker guidance (IMPORTANT)**  
This slide gives the full data setup in one page. The main point is that all variables are put onto the same monthly timeline, so we can compare policy, macro, and market conditions fairly. The dataset includes the S&P 500, VIX, several macro variables, and the Fed Funds Rate. The target is a bull or bear regime label defined using the 20% rule. That turns the market state into something we can analyse consistently across methods. We also remove the raw S&P 500 price from predictive features so the later modelling does not accidentally learn the target definition itself.

## Slide 3 — Analysis I: Correlation and Granger

**Main message**  
The first relationship tests suggest that Fed variables are not the strongest direct explanation of regime changes.

**Key points**

* Correlation patterns are stronger for market momentum and volatility variables such as `RSI`, `ROC`, and `VIX`.
* The raw Fed Funds Rate does not appear as a dominant variable in the descriptive structure.
* Conditional Granger tests ask whether past Fed variables add information after market controls are included.
* That extra explanatory gain is limited, peaking at about `2.5%`.

**Method (optional, very brief)**  
This slide combines correlation screening with conditional Granger causality, controlling for `SPX_Return`, `VIX_Close`, and `SPX_RSI`.

**Figure suggestion**

* Use `/mnt/data/data_preparation_pipeline.png`.
* Layout: left text, right figure.

**Speaker guidance (IMPORTANT)**  
The first evidence layer asks whether Fed variables look strong in the data before we move to more direct tests. The answer is mostly no. In the correlation structure, the regime label sits closer to momentum and volatility measures than to the Fed Funds Rate. Then conditional Granger testing asks a stricter question: once we already know past market information, do past Fed variables add much more? The answer is only a little. The reported gain is small, around two and a half percent at best. So this stage does not support the idea that the Fed is the strongest direct driver of regime switches.

## Slide 4 — Analysis I: Event Study

**Main message**  
The event study gives the clearest answer: Fed hikes do not usually trigger bear markets.

**Key points**

* The analysis tracks `23` hiking-cycle starts from `-12` to `+24` months around the first hike.
* Bear frequency falls rather than rises after hikes start.
* Average cumulative returns stay positive around the typical first hike.
* This is the strongest direct evidence against the immediate-trigger story.

**Method (optional, very brief)**  
Event dates are defined as the first hike after at least three months without a hike.

**Figure suggestion**

* Use `/mnt/data/correlation_heatmap.png`.
* Layout: text on the left, main figure on the right.

**Speaker guidance (IMPORTANT)**  
This is the core slide because it most directly answers the research question. Instead of asking whether variables move together, we ask what actually happens when a hiking cycle begins. Across 23 events, bear-market frequency does not increase after the first hike. In fact, it falls on average, and the cumulative return path stays positive around the event window. That means the typical first hike does not coincide with an immediate transition into a bear market. Some individual episodes do end badly, but the average pattern is not consistent with the claim that Fed hikes are the main trigger of bear regimes.

## Slide 5 — Analysis I: Network and Lead-Lag

**Main message**  
Supporting evidence shows that Fed variables are not central in the regime structure and do not show a clean immediate lead over bear markets.

**Key points**

* In the regime-centred network, the closest neighbours are `ROC`, `RSI`, `MACDH`, and `VIX`, not the Fed Funds Rate.
* `Fed_Funds_Rate` is relatively isolated, with very weak direct correlation to the regime node.
* Lead-lag results point to delayed or mixed relationships rather than an immediate trigger effect.
* This slide strengthens the event-study conclusion from a structural perspective.

**Method (optional, very brief)**  
Network analysis uses correlation-based links; lead-lag analysis checks whether Fed variables systematically lead future bear states.

**Figure suggestion**

* Use `/mnt/data/lead_lag_ccf.png`.
* Layout: text on the left, main figure on the right.

**Speaker guidance (IMPORTANT)**  
This slide is a reinforcement slide. The event study already tells us that hikes do not usually trigger bear markets, and here we ask whether the Fed at least looks central in the wider regime structure. The answer is again no. In the network view, the regime node sits much closer to momentum and volatility variables than to the Fed Funds Rate, which appears relatively isolated. Lead-lag patterns also fail to show a clean short-horizon lead from Fed tightening into bear markets. Together, these results strengthen the argument that the Fed is not the central immediate driver of regime change.

## Slide 6 — Models Overview

**Main message**  
The predictive section uses a small set of complementary models to test whether the same conclusion holds in classification.

**Key points**

* The models are Logistic Regression, Random Forest, and Gradient Boosting.
* They are chosen to compare linear and non-linear views of the same monthly regime problem.
* All models use the same chronological train-test split.
* The focus is on what variables matter, not on model mechanics.

**Method (optional, very brief)**  
`Regime` is treated as a binary target under a chronological `80/20` split.

**Figure suggestion**

* Use `/mnt/data/conditional_granger.png`.
* Layout: text on the left, main figure on the right.

**Speaker guidance (IMPORTANT)**  
I will move quickly through the predictive modelling setup because the important part is the result, not the algorithms themselves. I use three standard classifiers: logistic regression for interpretability, random forest for non-linear ranking, and gradient boosting as a second non-linear benchmark. All of them are trained on the same monthly regime task and evaluated with the same chronological split. That means we can compare them fairly. The role of this section is to check whether the conclusion from the earlier relationship analysis also appears when we ask a stricter question: which variables actually help predict regime states?

## Slide 7 — Predictive Results

**Main message**  
Predictive results again favour technical indicators, while Fed effects become more useful only after lag and delta engineering.

**Key points**

* In baseline models, `RSI`, `ROC`, `MACDH`, and `VIX` are more consistently important than the raw Fed Funds Rate.
* The raw Fed Funds Rate ranks `#2/12` in Logistic Regression, but only `#11/12` in Random Forest and `#10/12` in Gradient Boosting.
* This makes the raw Fed signal unstable rather than dominant.
* After feature engineering, `Fed_Funds_Rate_Lag3M` and `Fed_Funds_Rate_Delta6M` become more informative, suggesting delayed policy effects.

**Method (optional, very brief)**  
This slide combines baseline feature importance with the later lag/delta extension into one concise result page.

**Figure suggestion**

* Use `/mnt/data/confusion_matrices.png`.
* Layout: text on the left, main figure on the right.

**Speaker guidance (IMPORTANT)**  
The predictive results tell the same story as the earlier explanatory analysis. In the baseline feature set, the most useful predictors are technical and volatility-related variables, not the raw Fed Funds Rate. The Fed rate looks reasonably important in logistic regression, but it falls near the bottom in both tree-based models, so it is not a robust top signal. The more interesting result comes after feature engineering. Once we include lagged and delta versions of the Fed rate, some policy variables become more useful. That suggests the Fed effect is better understood as delayed and cumulative, rather than as an immediate trigger.

## Slide 8 — Conclusion

**Main message**  
Fed hikes are not the main immediate trigger of S&P 500 bear regimes, although delayed policy effects may still matter.

**Key points**

* Correlation, Granger, network, and predictive results all show that Fed variables are weaker than market-state signals in the short run.
* The event study provides the strongest direct evidence against the trigger hypothesis.
* Delayed and cumulative Fed features are more informative than the raw current rate.
* The main limits are class imbalance, only `23` hike events, and a rule-based regime definition.

**Method (optional, very brief)**  
No new method here; this slide closes the argument and acknowledges the study limits.

**Figure suggestion**

* Use `/mnt/data/gb_feature_importance.png`.
* Layout: text on the left, main figure on the right.

**Speaker guidance (IMPORTANT)**  
To conclude, the project does not support the common idea that Fed hikes are the main immediate trigger of S&P 500 bear markets. The strongest short-run signals come from the market itself, especially momentum and volatility indicators. The event study is the most direct piece of evidence, because it shows that bear-market frequency does not typically rise after hiking cycles begin. At the same time, the later modelling suggests that policy effects are not absent, just slower and more cumulative than the popular trigger story implies. So the final message is clear: not immediate trigger, but possible delayed influence.
