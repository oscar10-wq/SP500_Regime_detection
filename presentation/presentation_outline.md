# PPT Outline
## Presentation Notes
Total presentation time: 10 minutes  
Estimated speaking time per slide: 1–2 minutes  
Expected number of content slides (excluding the table of contents and section cover slides): 8–12 slides

## 1. Project Background (1 slide)
This section answers:  
**Why are we studying this problem, and what is the overall project about?**

## 2. Data Preparation (1-2 slides)
This section answers:  
**How do we transform raw data into an analysable dataset?**

structure:  
**Methodology + Why this method + outcome**

Include:
- Data sources: Yahoo Finance, FRED, futures data
- Monthly frequency alignment
- Missing value handling
- Removal of redundant variables
- Stationarity testing and transformations
- Construction of technical indicators: RSI, ROC, MACD
- Regime label construction
- Lag and delta feature engineering
- Final dataset summary: 310 monthly observations, 11 predictors, class imbalance

## 3. Analysis I: Relationship and Lead-Lag Screening (1-2 slides)
This section answers:  
**Is there an observable relationship between Fed variables and market regimes? If so, what does that relationship look like before predictive modelling?** 

structure:  
**Which method? + Why this method? + Methodology + outcome**  

Include:
- EDA / correlation heatmap
- Lagged correlation analysis
- Key descriptive findings:
  - Technical indicators and volatility are more strongly associated with regime than the raw Fed Funds Rate
  - `Fed_Funds_Rate` has near-zero contemporaneous correlation with `Regime`

These methods mainly focus on:
- identifying correlations
- examining lead-lag relationships
- screening variables before predictive modelling

## 4. Analysis II: Predictive Modelling (3 slides)
This section answers:  
**If regime is treated as a classification target, which variables are most predictive of it? How strong is the predictive power of Fed-related variables?**

structure:  
**Experiment setup + Why this setup? + Validation method + key outcome**

Include:
- Experiment 1: Baseline holdout classification
- Logistic Regression
- Random Forest
- Gradient Boosting
- Feature importance comparison
- Experiment 2: Monthly resampling with lag/delta feature engineering
- L1 pruning and re-modelling on pruned features
- Experiment 3: Pure macro + walk-forward testing
- Experiment 4: Full features + walk-forward validation check
- Validation emphasis:
  - holdout for Experiments 1-2
  - walk-forward for Experiments 3-4
  - report bear recall / bear F1, not accuracy alone

## 5. Conclusion and Limitations (1 slides)
This section answers:  
**What are the conclusions, and what are the limitations of our study?**

conclusions:
- Fed rate hikes are not the main immediate trigger of S&P 500 bear regimes
- The strongest short-horizon signals come from market momentum and volatility features
- Fed-related effects appear more meaningful through lagged and delta-based policy features than through the raw current rate
- `Fed_Funds_Rate_Delta12M` becomes especially important in the macro-only walk-forward setting

limitations:
- Bear-market months are a minority class, so performance metrics are sensitive to class imbalance
- Macro-only models can identify fragile conditions better than they can time the exact month of regime switching
- Technical indicators can dominate predictive rankings because they are closely tied to market-state behaviour
