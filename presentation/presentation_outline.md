# PPT Outline
## 1. Project Background
This section answers:
**Why are we studying this problem, and what is the overall project about?**

## 2. Data Preparation
This section answers:
**How do we transform raw data into an analysable dataset?**
structure:
**Methodology + Why this method + outcome**
Include:
- Data sources: Yahoo Finance, FRED, futures data
- Monthly frequency alignment
- Missing value handling
- Removal of redundant variables
- Construction of technical indicators: RSI, ROC, MACD
- Regime label construction
- Lag and delta feature engineering

## 3. Analysis I: Relationship and Causality Checks
This section answers:
**Is there a relationship between Fed variables and market regimes? If so, what does that relationship look like, and who leads whom?**
structure:
**Whitch model? + Why this model? + Methodology + outcome**
Include:
- EDA / correlation heatmap
- Lagged correlation analysis
- Conditional Granger causality
- Event Study
- Network Analysis

These methods mainly focus on:
- identifying correlations
- examining lead-lag relationships
- testing causal or quasi-causal effects
- interpreting structural relationships
- comparing market behaviour before and after events

## 4. Analysis II: Predictive Modelling
This section answers:
**If regime is treated as a classification target, which variables are most predictive of it? How strong is the predictive power of Fed-related variables?**
structure:
**Whitch model? + Why this model? + Methodology + outcome**
Include:
- Feature importance classification
- Monthly resampling with lag/delta feature engineering
- Logistic Regression
- Random Forest
- Gradient Boosting
- Re-modelling after L1 pruning

## 5. Conclusion and Limitations
This section answers:
**What are the conclusions, and what are the limitations of our study?**

conclusions:

limitations:
