import pandas as pd
import ssl
import certifi
import yfinance as yf
import pandas_ta_classic as ta
import pandas as pd
from fredapi import Fred
import os
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def identify_hike_cycles(fed_funds_df: pd.DataFrame, window: int = 3):
    """
    Identifies periods where interest rates are increasing.
    Returns a boolean series where True = "Tightening/Hike Cycle".
    """
    # Calculate the change in rates
    diff = fed_funds_df
    
    # A 'Hike' is defined here as a positive change in the Fed Funds Rate
    is_hike = diff > 0
    
    # Optional: Smooth this out so one flat month doesn't break a cycle
    hike_regime = is_hike.rolling(window=window).max().fillna(0).astype(bool)
    
    return hike_regime


def calculate_feature_lagged_correlation(target_df, features_df, lags=[1, 6, 12]):
    results = {}
    
    for lag in lags:
        correlationlist = [] 
        # Shift the target once per lag
        future_target = target_df.shift(-lag)
        
        for col in features_df.columns:
            # Calculate correlation between feature and shifted target
            corr_value = features_df[col].corr(future_target)
            correlationlist.append(corr_value)
            
        # Now both have the exact same length (number of columns)
        results[lag] = pd.DataFrame({
            "Feature": features_df.columns,
            "Correlation": correlationlist
        }).sort_values(by="Correlation", key=abs, ascending=False).reset_index(drop=True)
        
    return results

def plot_correlation_ranking(results):

    fig, ax = plt.subplots(len(results), 1, figsize=(12, 6 * len(results)))
    for i, (lag, df) in enumerate(results.items()):
        sns.barplot(x="Correlation", y="Feature", data=df.head(20), palette="viridis", ax=ax[i])
        ax[i].set_title(f"Top 20 Feature Correlations with SPX_Close at Lag {lag} Months")
        ax[i].set_xlabel("Correlation Coefficient")
        ax[i].set_ylabel("Feature")
    plt.tight_layout()
    plt.show()

def classify_regimes(feature_set: pd.DataFrame, column: str = "SPX_Close") -> pd.Series:
    """Labels regimes as 1 (Bull) or 0 (Bear) based on 20% thresholds.
    Args:
        feature_set (pd.DataFrame): set of features
        column (str, optional): column to use. Defaults to "SPX_Close".

    Returns:
        pd.Series: regimes
    """
    prices = feature_set[column]
    regime = pd.Series(index=feature_set.index, dtype=int)

    # Initialize with the first state (assuming Bull for start of 2000)
    current_regime = 1
    last_peak = prices.iloc[0]
    last_trough = prices.iloc[0]

    for i in range(len(prices)):
        price = prices.iloc[i]

        if current_regime == 1:  # In a Bull market
            if price > last_peak:
                last_peak = price
            # If price drops 20% from the peak, it's now a Bear market
            if price <= last_peak * 0.80:
                current_regime = 0
                last_trough = price

        else:  # In a Bear market
            if price < last_trough:
                last_trough = price
            # If price rises 20% from the trough, it's now a Bull market
            if price >= last_trough * 1.20:
                current_regime = 1
                last_peak = price

        regime.iloc[i] = current_regime

    return regime