# Presentation Layout Design V3

## Slide 1: Project Background

**Main message**  
This presentation tests whether Fed rate hikes are the main trigger of S&P 500 bull-to-bear regime switches.

**Point 1**  
Research motivation: this is a common market narrative, but it needs empirical testing.

**Point 1 explanation**  
Fed tightening is often blamed for market downturns, yet the timing is not always clear. The project therefore asks whether hikes directly trigger regime change or whether other signals are more informative.

**Point 2**  
Research goal: compare Fed, macro, and technical indicators in explaining and predicting regime shifts.

**Point 2 explanation**  
The aim is not only to check correlation, but also to compare short-run market signals with slower policy and macro signals.

**Point 3**  
Overall path: `Background -> Data Preparation -> Relationship Screening -> Predictive Modelling -> Conclusion`

**Point 3 explanation**  
The talk moves from dataset construction to descriptive evidence, then to classification experiments, and finally to the main conclusion.

**Layout suggestion**  
Place the research question and motivation on the left.  
Place the workflow diagram on the right.  
Use: [project_workflow_diagram.png](/Users/xiaoyu/Documents/UCL/YuYu-UCL/COMP0047_Data%20Science/project/SP500_Regime_detection/presentation/assets/project_workflow_diagram.png)

---

## Slide 2: Data Preparation I - Data Sources and Monthly Alignment

**Main message**  
The project is built on one unified monthly dataset that makes market, macro, and policy variables directly comparable.

**Point 1**  
Data sources: Yahoo Finance, FRED, and futures data.

**Point 1 explanation**  
Yahoo Finance provides S&P 500 and VIX data. FRED provides GDP, inflation, unemployment, the Fed Funds Rate, and the 10Y2Y spread. Futures data adds policy-expectation variables before cleaning.

**Point 2**  
Monthly alignment is essential.

**Point 2 explanation**  
Technical indicators, macro series, and policy variables update at different frequencies. Aligning everything to monthly dates prevents high-frequency market data from dominating slower macro signals.

**Point 3**  
Outcome: one common monthly observation unit.

**Point 3 explanation**  
This gives a consistent basis for both descriptive analysis and predictive modelling across the full sample period.

**Layout suggestion**  
Place text on the left and the pipeline figure on the right.  
Use: [data_preparation_pipeline.png](/Users/xiaoyu/Documents/UCL/YuYu-UCL/COMP0047_Data%20Science/project/SP500_Regime_detection/presentation/assets/data_preparation_pipeline.png)

---

## Slide 3: Data Preparation II - Cleaning, Transformations and Final Dataset

**Main message**  
Before modelling, the data is cleaned to reduce leakage, redundancy, and non-stationarity.

**Point 1**  
Key removals improve validity.

**Point 1 explanation**  
`SPX_Close` is removed from predictors because the regime label is derived from it. `SPX_MACDS` is redundant, `Fed_Funds_Future` is almost perfectly collinear with the Fed Funds Rate, and `10Y_Treasury_Future` has too many missing values.

**Point 2**  
Stationarity transformations are applied where needed.

**Point 2 explanation**  
Variables such as `SPX_Volume`, `Real_GDP`, `SPX_MACD`, and `Inflation` are transformed so the modelling stage uses stable series.

**Point 3**  
Regime and engineered features complete the dataset.

**Point 3 explanation**  
The target is a bull/bear regime built with the 20% rule, and lag/delta features are added to test delayed policy effects.

**Point 4**  
Final dataset summary.

**Point 4 explanation**  
The validated base dataset contains `310` monthly observations and `11` core predictors, with an imbalanced bear class.

**Layout suggestion**  
Use a two-column layout.  
Left: cleaning and transformation bullets.  
Right: a compact summary box with `310 observations`, `11 predictors`, `Regime target`, and `class imbalance`.

---

## Slide 4: Analysis I - Correlation and Lead-Lag Screening

**Main message**  
The descriptive evidence suggests that technical indicators and volatility are more closely linked to regime changes than the raw Fed Funds Rate.

**Point 1**  
Method: `correlation heatmap + lagged correlation analysis`

**Point 1 explanation**  
These methods screen broad relationship patterns before predictive modelling and show which variables appear most related to the regime label.

**Point 2**  
Why this method?

**Point 2 explanation**  
It is a fast but informative way to detect multicollinearity, compare variables at different lags, and see whether Fed signals look strong even before formal models are trained.

**Point 3**  
Key finding.

**Point 3 explanation**  
`SPX_ROC`, `SPX_RSI`, `SPX_MACDH`, and `VIX_Close` are much more strongly associated with `Regime` than `Fed_Funds_Rate`, whose contemporaneous correlation is close to zero.

**Layout suggestion**  
Place explanation on the left and one main figure on the right.  
Use: [correlation_heatmap.png](/Users/xiaoyu/Documents/UCL/YuYu-UCL/COMP0047_Data%20Science/project/SP500_Regime_detection/presentation/figures/correlation_heatmap.png)

---

## Slide 5: Predictive Modelling I - Baseline Holdout Results

**Main message**  
In the baseline classification experiment, technical and volatility features dominate the prediction task, while the raw Fed rate ranks much lower.

**Point 1**  
Experiment 1 setup.

**Point 1 explanation**  
The base 11-feature monthly dataset is split chronologically into `248` training months and `62` test months.

**Point 2**  
Models used.

**Point 2 explanation**  
Logistic Regression, Random Forest, and Gradient Boosting provide a comparison between linear and non-linear classifiers.

**Point 3**  
Key result.

**Point 3 explanation**  
Across models, the Fed Funds Rate ranks in the lower half, while top features are `SPX_RSI`, `SPX_ROC`, `SPX_MACDH`, and `VIX_Close`.

**Point 4**  
Metric interpretation.

**Point 4 explanation**  
Because bear months are rare, bear recall and bear F1 are more informative than accuracy alone.

**Layout suggestion**  
Use the Random Forest feature importance chart as the main figure.  
Text on the left, figure on the right.  
Use: [baseline_rf_feature_importance.png](/Users/xiaoyu/Documents/UCL/YuYu-UCL/COMP0047_Data%20Science/project/SP500_Regime_detection/presentation/figures/baseline_rf_feature_importance.png)

---

## Slide 6: Predictive Modelling II - Lag/Delta Features and L1 Pruning

**Main message**  
Fed-related variables become more useful after lag and delta engineering, which suggests delayed policy effects rather than immediate ones.

**Point 1**  
Experiment 2 setup.

**Point 1 explanation**  
The feature set is expanded with 3-, 6-, and 12-month lags and deltas, then pruned with L1 regularisation.

**Point 2**  
Why this matters.

**Point 2 explanation**  
If policy affects markets with delay, the current rate level may look weak while lagged or cumulative changes carry more signal.

**Point 3**  
Key result.

**Point 3 explanation**  
L1 pruning keeps `Fed_Funds_Rate_Lag6M` and `Fed_Funds_Rate_Lag12M`, and the best Fed feature improves to an average rank of about `4.0`.

**Point 4**  
Performance trade-off.

**Point 4 explanation**  
Using a stricter threshold improves bear detection, even though it may reduce overall accuracy.

**Layout suggestion**  
Use one figure only: the L1 survivor chart.  
Text on the left, figure on the right.  
Use: [monthly_resampling_l1_survivors.png](/Users/xiaoyu/Documents/UCL/YuYu-UCL/COMP0047_Data%20Science/project/SP500_Regime_detection/presentation/figures/monthly_resampling_l1_survivors.png)

---

## Slide 7: Predictive Modelling III - Pure Macro Walk-Forward

**Main message**  
When technical indicators are removed, the clearest macro signal becomes the 12-month change in the Fed Funds Rate.

**Point 1**  
Experiment 3 setup.

**Point 1 explanation**  
This experiment uses only macro and policy variables under walk-forward validation, which is stricter than a single holdout split.

**Point 2**  
Why this matters.

**Point 2 explanation**  
Removing technical indicators helps reveal whether Fed-related variables still carry predictive information on their own.

**Point 3**  
Key result.

**Point 3 explanation**  
`Fed_Funds_Rate_Delta12M` becomes the top predictor in both tree-based models, with an average rank of about `2.0`.

**Point 4**  
Limitation inside the result.

**Point 4 explanation**  
Macro-only models detect fragile conditions, but their bear precision stays low, so exact timing remains difficult.

**Layout suggestion**  
Place the walk-forward setup figure on the right and concise result bullets on the left.  
Use: [walk_forward_setup.png](/Users/xiaoyu/Documents/UCL/YuYu-UCL/COMP0047_Data%20Science/project/SP500_Regime_detection/presentation/figures/walk_forward_setup.png)

---

## Slide 8: Predictive Modelling IV - Full Features Walk-Forward Validation

**Main message**  
When technical features are added back, market signals dominate again, but lagged Fed information still survives and keeps independent value.

**Point 1**  
Experiment 4 setup.

**Point 1 explanation**  
The full feature set is tested under the same walk-forward framework as Experiment 3.

**Point 2**  
Key result.

**Point 2 explanation**  
Overall accuracy improves to about `89%`, but that improvement is driven largely by technical indicators tied closely to market-state behaviour.

**Point 3**  
Fed implication.

**Point 3 explanation**  
Even in the full-feature setting, lagged Fed features still survive pruning and rank reasonably highly across models.

**Point 4**  
Interpretation.

**Point 4 explanation**  
This supports a balanced conclusion: technical signals dominate short-run prediction, but delayed Fed effects are not irrelevant.

**Layout suggestion**  
Use the full-feature walk-forward Random Forest importance chart.  
Text on the left, figure on the right.  
Use: [full_feature_walkforward_rf_importance.png](/Users/xiaoyu/Documents/UCL/YuYu-UCL/COMP0047_Data%20Science/project/SP500_Regime_detection/presentation/figures/full_feature_walkforward_rf_importance.png)

---

## Slide 9: Conclusion and Limitations

**Main message**  
The project does not support the claim that Fed hikes are the main immediate trigger of S&P 500 bear regimes.

**Point 1**  
Main conclusion.

**Point 1 explanation**  
Across descriptive and predictive analyses, technical momentum and volatility are the strongest short-horizon signals.

**Point 2**  
More nuanced conclusion.

**Point 2 explanation**  
Fed-related effects matter more through lagged or cumulative tightening measures than through the raw current rate.

**Point 3**  
Key limitations.

**Point 3 explanation**  
Bear months are rare, macro-only timing is difficult, and technical variables can dominate because they are closely tied to market-state behaviour.

**Layout suggestion**  
Use a clean closing slide with no heavy figure.  
Left: 3 conclusion bullets.  
Right: 3 limitation bullets.  
Optional small footer line: `Main takeaway: not immediate trigger, but possible delayed influence`.
