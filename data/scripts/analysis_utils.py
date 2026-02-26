import pandas as pd
import ssl
import certifi
import yfinance as yf
import pandas_ta_classic as ta
import pandas as pd
from fredapi import Fred
import os
from dotenv import load_dotenv


def identify_hike_cycles(fed_funds_df: pd.DataFrame, window: int = 3):
    """
    Identifies periods where interest rates are increasing.
    Returns a boolean series where True = "Tightening/Hike Cycle".
    """
    # Calculate the change in rates
    diff = fed_funds_df.diff()
    
    # A 'Hike' is defined here as a positive change in the Fed Funds Rate
    is_hike = diff > 0
    
    # Optional: Smooth this out so one flat month doesn't break a cycle
    hike_regime = is_hike.rolling(window=window).max().fillna(0).astype(bool)
    
    return hike_regime


def calculate_regime_lag(price_df: pd.DataFrame, rates_df: pd.DataFrame, lags=[30, 90, 180]):
    """
    Calculates the correlation between rate changes and future market returns.
    Helps determine the 'Lead Time' of a Fed trigger.
    """
    results = {}
    returns = price_df.pct_change()
    rate_changes = rates_df.diff()

    for lag in lags:
        # Shift returns backwards to see how today's rate change affects future returns
        future_returns = returns.shift(-lag)
        correlation = rate_changes.corr(future_returns) 
        results[f"{lag}_day_lag_corr"] = correlation
        
    return results


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