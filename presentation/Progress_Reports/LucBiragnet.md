## [2026-02-14] Data Loading Setup for FRED API

### New Dependencies Installed
- `fredapi`: official Python wrapper for the St. Louis Fed API.
  - **Why?** - Much easier to use an existing wrapper than to build everything from scratch.
- `python-dotenv`: for securely managing environment variable (API keys).

### Configuration Changes
- Configured .gitignore to prevent leaking API keys.
- Added config directory within `data/scripts` to hold .env API key files.

### Data Pipeline Changes
- `BaseDataLoader` interface for fetching data from multiple sources, `FredDataLoader` implementation to fetch data from the Federal Reserve API.
  - **Why?** - Easily extendable for future data sources (e.g., `YahooDataLoader`).


## [2026-03-08] Monthly Resampling + Lagging Experiments

### Analysis Additions
- **Monthly resampling experiment**
  - The justification for this experiment is the fact that the FED rate changes roughly every six weeks, whereas the technical indicators update daily, which could introduce bias by pushing tree-based models to put heavy importance on the noisy fluctuations from technical indicators.  Hence, we down-sample the dataset to a monthly basis to level the playing field between technical indicators and the FED rate changes \(as well as other macroeconomic data\).
  - Results - the feature importances / ranking remained largely unchanged from the daily data baseline.
  
- **Monthly resampling with lagged/momentum feature engineering**
  - Hypothesis 1: even with a monthly resample, the Fed rate does not necessarily have high predicting power on regimes in our previous experiments because the market anticipates rate changes before they are announced, effectively pricing in their effect before they happen. 
  - Hypothesis 2: in combination with hypothesis 1, the Fed rate announcement is not as defining for the market as is the general direction of Fed rate changes over the past few months. The momentum of Fed rates is important.
  - We attempt a second experiment to put these hypotheses to the test: first by introducing 3 and 6-month lagged versions of macro-related features, and by adding their momentum as features. To avoid keeping too many strongly correlated features, we apply an L1 penalty to prune features with logistic regression before training the tree-based models.
  - Results: Compared to the previously set benchmarks, the 6-month delta of the Fed rate is given high importance by RF and ranks higher on average for all models. The 3-month lagged Fed rate feature is given high importance by GB, and ranks higher on average for all models.

