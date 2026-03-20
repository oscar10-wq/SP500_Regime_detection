# English PPT Outline and Layout Plan

## Slide 1: Project Background

**Main message**  
This project asks whether Fed rate hikes are the main trigger of S&P 500 bull-to-bear regime switches.

**Point 1**  
Research motivation: Fed policy is often assumed to drive stock market turning points.

**Point 1 explanation**  
Rate hikes are frequently blamed for bear markets, but that claim is not always supported by timing or empirical evidence. We therefore test whether Fed hikes are a direct trigger or simply part of the broader economic cycle.

**Point 2**  
Research goal: compare Fed variables, macro indicators, and market technical indicators in explaining and predicting bull/bear regimes.

**Point 2 explanation**  
We want to know not only whether a relationship exists, but also who leads whom, how strong the relationship is, and whether it has predictive value.

**Point 3**  
Overall research path: `Data Preparation -> Relationship/Causality Checks -> Predictive Modelling -> Conclusion and Limitations`

**Point 3 explanation**  
We first construct a monthly regime dataset, then use explanatory and predictive analyses to answer the research question.

**Layout suggestion**  
Place the research question and motivation on the left.  
Place the workflow diagram on the right.  
Use: [project_workflow_diagram.png](/Users/xiaoyu/Documents/UCL/YuYu-UCL/COMP0047_Data%20Science/project/SP500_Regime_detection/presentation/assets/project_workflow_diagram.png)

---

## Slide 2: Data Preparation I - Data Sources and Feature Construction

**Main message**  
We combine raw data from multiple sources into one unified monthly feature set for regime analysis.

**Point 1**  
Data sources: Yahoo Finance, FRED, and futures data.

**Point 1 explanation**  
Yahoo Finance provides S&P 500 and VIX market data. FRED provides Fed Funds Rate, GDP, inflation, unemployment, and the 10Y2Y spread. Futures data adds market expectations about rates and bonds.

**Point 2**  
Dataset size: the final monthly dataset contains approximately `312 observations` from `January 2000 to December 2025 / January 2026`, depending on the analysis cut.

**Point 2 explanation**  
This monthly sample is the common basis used across the main modelling and explanatory analyses, making results comparable across methods.

**Point 3**  
Features are grouped into technical, macroeconomic, and Fed-related variables.

**Point 3 explanation**  
Technical indicators include `RSI`, `ROC`, `MACD`, and `MACDH`. Macro variables include `Real GDP`, `Inflation`, `Unemployment`, and `10Y2Y Spread`. Fed-related inputs include the `Fed Funds Rate` and selected futures-based variables.

**Point 4**  
Why this feature design?

**Point 4 explanation**  
The project is fundamentally a comparison between market-based signals and policy/macroeconomic signals in explaining bull/bear switches.

**Point 5**  
Outcome: a multi-source monthly feature framework covering market behaviour, volatility, macro conditions, and policy signals.

**Layout suggestion**  
Use a clean three-column layout:
- Left: `Data source`
- Middle: `Example variables`
- Right: `Why included`

No image is necessary on this slide. A structured table layout is clearer.

---

## Slide 3: Data Preparation II - Cleaning, Alignment and Regime Label

**Main message**  
To make the data comparable and model-ready, we aligned frequencies, cleaned missing values, removed redundant variables, and constructed the regime label.

**Point 1**  
Method: align all sources to monthly frequency.

**Point 1 explanation**  
Technical indicators update faster than Fed and macro variables. Without frequency alignment, models would be biased toward high-frequency inputs.

**Point 2**  
Method: missing value handling and removal of redundant variables.

**Point 2 explanation**  
Low-frequency macro series are forward-filled, and sparse or overlapping variables are removed to reduce noise and avoid leakage.

**Point 3**  
Method: construct the `Regime` label using the 20% peak-to-trough / trough-to-peak rule.

**Point 3 explanation**  
A 20% fall from a peak defines a bear market, while a 20% recovery from a trough defines a bull market. This transforms an abstract market state into a classification target.

**Point 4**  
Additional feature engineering: lag and delta terms.

**Point 4 explanation**  
Variables such as `Fed_Funds_Rate_Lag3M` and `Fed_Funds_Rate_Delta6M` help test delayed and cumulative policy effects.

**Point 5**  
Outcome: a clean monthly regime dataset ready for both explanatory analysis and predictive modelling.

**Layout suggestion**  
Place the pipeline graphic on the left or centre.  
Highlight three keywords on the right or below: `alignment`, `leakage prevention`, `label construction`.  
Use: [data_preparation_pipeline.png](/Users/xiaoyu/Documents/UCL/YuYu-UCL/COMP0047_Data%20Science/project/SP500_Regime_detection/presentation/assets/data_preparation_pipeline.png)

---

## Slide 4: Analysis I - Correlation and Lead-Lag Checks

**Main message**  
The first analysis stage does not aim to predict directly. Instead, it tests whether Fed variables are related to market regimes and whether they lead or lag market changes.

**Point 1**  
Which method?  
`EDA correlation heatmap + lagged correlation analysis`

**Point 1 explanation**  
The heatmap shows the overall linear relationship structure between variables. Lagged correlation tests whether some variables provide earlier signals for future market states.

**Point 2**  
Why this method?

**Point 2 explanation**  
This is the most important screening step before formal modelling. It reveals strong correlations, multicollinearity, and whether Fed variables show any stable descriptive relationship with regime changes.

**Point 3**  
Methodology

**Point 3 explanation**  
- Measure pairwise correlations between features and between features and the target  
- Compare relationships at different lags such as 1, 6, and 12 months  
- Apply differencing or percentage change where needed, supported by ADF stationarity checks

**Point 4**  
Validation / rigor

**Point 4 explanation**  
This is not predictive validation. Rigor comes from stationarity checks, multicollinearity discussion, and explicit lag structure inspection.

**Point 5**  
Outcome

**Point 5 explanation**  
Technical indicators show stronger relationships with market regimes than the raw Fed rate, and the contemporaneous Fed-regime relationship is weak.

**Layout suggestion**  
Place the correlation heatmap on the right and the explanation on the left.  
Suggested split: text 40%, figure 60%.  
Use your existing heatmap figure.

---

## Slide 5: Analysis I - Conditional Granger Causality

**Main message**  
To test whether the Fed genuinely leads market regimes and adds incremental explanatory power, we use conditional Granger causality.

**Point 1**  
Which model?  
`Conditional Granger Causality`

**Point 1 explanation**  
This method tests whether past Fed variables significantly improve regime prediction once other market information is already included.

**Point 2**  
Why this model?

**Point 2 explanation**  
Simple correlation is not causation. We need to distinguish whether the Fed truly leads the market or merely moves alongside it.

**Point 3**  
Methodology

**Point 3 explanation**  
- Tested variables: `Fed_Cycle`, `Fed_Change`, `Fed_Funds_Rate`, `10Y2Y_Spread`  
- Control variables: `SPX_Return`, `VIX_Close`, `SPX_RSI`  
- Compared restricted and unrestricted models across multiple lags

**Point 4**  
Validation / rigor

**Point 4 explanation**  
- Use p-values and F-statistics  
- Apply `BH correction` for multiple testing  
- Compare `R² gain` to assess not only significance but also practical explanatory value

**Point 5**  
Outcome

**Point 5 explanation**  
After controlling for market variables, most Fed variables do not significantly Granger-cause regimes. Even when weak signals appear, the extra explanatory gain is very small, peaking at around `2.5%`.

**Layout suggestion**  
Place text on the left and the figure on the right.  
Figure width can take around 55% of the slide.  
Use either:
- [3_model_comparison.png](/Users/xiaoyu/Documents/UCL/YuYu-UCL/COMP0047_Data%20Science/project/SP500_Regime_detection/analysis/Feature%20Importance/causality_outputs/3_model_comparison.png)
- [2_conditional_granger.png](/Users/xiaoyu/Documents/UCL/YuYu-UCL/COMP0047_Data%20Science/project/SP500_Regime_detection/analysis/Feature%20Importance/causality_outputs/2_conditional_granger.png)

---

## Slide 6: Analysis I - Event Study and Network Analysis

**Main message**  
We then test the question from both an event perspective and a structural perspective: do hikes make bear markets more likely, and does the Fed sit close to the regime node in the feature network?

**Point 1**  
Which methods?  
`Event Study + Lead-Lag Cross-Correlation + Network Analysis`

**Point 2**  
Why these methods?

**Point 2 explanation**  
Event study answers what happens around the start of hiking cycles, while network analysis shows which variables sit structurally closest to the regime label.

**Point 3**  
Methodology

**Point 3 explanation**  
- Event Study: identify 23 hike-cycle starts and extract a `[-12, +24] months` window  
- Lead-Lag: compare `Fed[t]` with `Bear[t+L]`  
- Network: build a correlation-based feature network and inspect the neighbours of the `Regime` node

**Point 4**  
Validation / rigor

**Point 4 explanation**  
- Event study uses a fixed event definition and fixed windows  
- Lead-lag uses explicit lag interpretation and significance thresholds  
- Network analysis is used as structural confirmation rather than predictive validation

**Point 5**  
Outcome

**Point 5 explanation**  
- Bear market frequency does not rise immediately after hikes begin  
- Markets do not typically turn bearish right after the start of a hike cycle  
- In the network, `RSI`, `ROC`, `MACDH`, and `VIX` sit closer to `Regime`, while `Fed_Funds_Rate` is relatively isolated

**Layout suggestion**  
Use one main figure to avoid overcrowding.  
Prefer [1_event_study.png](/Users/xiaoyu/Documents/UCL/YuYu-UCL/COMP0047_Data%20Science/project/SP500_Regime_detection/analysis/Event%20Study/outputs/1_event_study.png) as the main visual.  
If space allows, add a small supporting figure in the lower right corner: [2_regime_ego_network.png](/Users/xiaoyu/Documents/UCL/YuYu-UCL/COMP0047_Data%20Science/project/SP500_Regime_detection/analysis/Network%20Analysis/outputs/2_regime_ego_network.png)

Recommended layout: text on the left, main figure on the upper right, optional small figure on the lower right.

---

## Slide 7: Analysis II - Predictive Models Overview

**Main message**  
The second analysis stage treats `Regime` as a classification target and directly compares how well different variables predict bull and bear states.

**Point 1**  
Which models?  
`Logistic Regression + Random Forest + Gradient Boosting`

**Point 1 explanation**  
We use both linear and non-linear models so that conclusions do not depend on a single algorithm.

**Point 2**  
Why these models?

**Point 2 explanation**  
- Logistic Regression: strong interpretability and coefficient direction  
- Random Forest: handles non-linearity and provides feature importance  
- Gradient Boosting: captures more complex non-linear relationships as a complementary benchmark

**Point 3**  
Methodology

**Point 3 explanation**  
- Treat `Regime` as the classification target  
- Use a chronological `80/20` train-test split  
- Compare test performance and feature importance across models

**Point 4**  
Validation

**Point 4 explanation**  
- `chronological hold-out validation`  
- accuracy, classification report, and confusion matrices  
- explicit discussion of class imbalance

**Point 5**  
Outcome

**Point 5 explanation**  
All three models suggest that technical indicators are more predictive of regime switches than the raw Fed rate.

**Layout suggestion**  
No large figure is required.  
Use a structured comparison table with three columns:
- `Model`
- `Why used`
- `What it tells us`

---

## Slide 8: Analysis II - Baseline Feature Importance Results

**Main message**  
Baseline classification results show that the Fed Funds Rate is not the strongest regime predictor; market momentum and volatility are more informative.

**Point 1**  
Which model / result?  
Baseline classification on 12 features.

**Point 1 explanation**  
Before applying more complex lag/delta engineering, we first evaluate the original feature set.

**Point 2**  
Methodology

**Point 2 explanation**  
- `312` monthly observations  
- chronological `80/20` split  
- three models trained and compared using feature rankings

**Point 3**  
Validation

**Point 3 explanation**  
- test accuracy  
- confusion matrices  
- class imbalance interpretation

**Point 4**  
Outcome

**Point 4 explanation**  
- Logistic Regression: `90.48%`  
- Random Forest: `80.95%`  
- Gradient Boosting: `80.95%`  
- Fed Funds Rate average rank: about `7.7 / 12`  
- Stronger features: `SPX_RSI`, `SPX_ROC`, `VIX`

**Point 5**  
Interpretation

**Point 5 explanation**  
This suggests that market momentum and volatility reflect bull/bear switching more clearly than the current policy rate alone.

**Layout suggestion**  
Use [rf_feature_importance.png](/Users/xiaoyu/Documents/UCL/YuYu-UCL/COMP0047_Data%20Science/project/SP500_Regime_detection/analysis/Feature%20Importance/rf_feature_importance.png) as the main figure.  
If space allows, add [lr_coefficients.png](/Users/xiaoyu/Documents/UCL/YuYu-UCL/COMP0047_Data%20Science/project/SP500_Regime_detection/analysis/Feature%20Importance/lr_coefficients.png) below or beside it.

Recommended layout: key numbers on the left, chart(s) on the right.

---

## Slide 9: Analysis II - Monthly Resampling, Lag/Delta and L1 Pruning

**Main message**  
To test whether the Fed effect is underestimated by using only the current rate, we extend the model with monthly resampling, lag/delta features, and L1-based feature pruning.

**Point 1**  
Which model?  
`Monthly resampling + engineered features + L1-regularised Logistic Regression + re-modelling`

**Point 2**  
Why this model?

**Point 2 explanation**  
Fed effects may not be immediate. They may operate through delayed and cumulative tightening. Once many lag/delta terms are added, we also need a way to control multicollinearity.

**Point 3**  
Methodology

**Point 3 explanation**  
- First run a monthly-only resampling experiment  
- Then add `Lag3M / Lag6M / Delta3M / Delta6M` features  
- Use `L1 Logistic Regression` for pruning  
- Retrain Random Forest and Gradient Boosting on the pruned dataset

**Point 4**  
Validation

**Point 4 explanation**  
- Keep the chronological test split  
- Compare model behaviour before and after pruning  
- Emphasise that `L1 regularisation is feature selection, not validation itself`

**Point 5**  
Outcome

**Point 5 explanation**  
- Monthly resampling alone does not change the conclusion much  
- After lag/delta engineering, `Fed_Funds_Rate_Lag3M` and `Fed_Funds_Rate_Delta6M` become more important  
- This suggests that Fed effects are more `lagged / cumulative` than immediate

**Layout suggestion**  
Use [cell_50_fig_0.png](/Users/xiaoyu/Documents/UCL/YuYu-UCL/COMP0047_Data%20Science/project/SP500_Regime_detection/presentation/figures/cell_50_fig_0.png) or [cell_49_fig_1.png](/Users/xiaoyu/Documents/UCL/YuYu-UCL/COMP0047_Data%20Science/project/SP500_Regime_detection/presentation/figures/cell_49_fig_1.png) as the main visual.

Recommended layout: text on the left, figure on the right.  
Emphasise one takeaway sentence: `Raw Fed rate is weak, engineered Fed signals are stronger.`

---

## Slide 10: Conclusion and Limitations

**Main message**  
Overall, Fed hikes are not the main immediate trigger of S&P 500 bear regimes, but their cumulative influence becomes more visible after lag and momentum-based feature engineering.

**Point 1**  
Conclusion 1

**Point 1 explanation**  
Technical indicators and market volatility are stronger regime signals. `RSI`, `ROC`, and `VIX` are consistently stronger across correlation analysis, network analysis, and classification models.

**Point 2**  
Conclusion 2

**Point 2 explanation**  
The raw Fed Funds Rate has limited direct explanatory and predictive power. Feature importance, Granger causality, and event study results do not support the idea that hikes immediately trigger bear markets.

**Point 3**  
Conclusion 3

**Point 3 explanation**  
Fed effects appear more delayed and cumulative than immediate. Lag/delta features are more informative than the raw current rate.

**Point 4**  
Limitations

**Point 4 explanation**  
- relatively few bear observations, creating class imbalance  
- only 23 hike cycles in the event study  
- Granger and correlation analyses rely mainly on linear relationships  
- the regime label is rule-based and therefore retrospective

**Point 5**  
Closing line

**Point 5 explanation**  
Our conclusion is not that the Fed has no effect at all, but rather: **Fed hikes are not the main immediate trigger of bear regimes.**

**Layout suggestion**  
Place the three conclusions on the left and the limitations on the right.  
Add a takeaway sentence at the bottom.  
No figure is strictly necessary, but if you want a visual closing element, use: [2_regime_ego_network.png](/Users/xiaoyu/Documents/UCL/YuYu-UCL/COMP0047_Data%20Science/project/SP500_Regime_detection/analysis/Network%20Analysis/outputs/2_regime_ego_network.png)
