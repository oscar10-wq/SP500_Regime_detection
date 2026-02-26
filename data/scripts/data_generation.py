import ssl
import certifi
import yfinance as yf
import pandas_ta_classic as ta
import pandas as pd
from fredapi import Fred
import os
from dotenv import load_dotenv

custom_context = ssl.create_default_context(cafile=certifi.where())
load_dotenv()
api_key = os.getenv("0d472975ba8d0e5ee549648673b1e3de")


def get_yahoo_finance_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Gets yahoo finance input data (spx and vix close prices
    plus some momentum indicators)

    Args:
        start_date (str): start date (YYYY-MM-DD)
        end_date (str): end date (YYYY-MM-DD)

    Returns:
        pd.DataFrame: clean input yahoo finance data
    """
    # We define a fetch date that is before our start date for the RSI
    # MACD and ROC calculations, so our main set is completely populated
    fetch_date = (pd.Timestamp(start_date) - pd.DateOffset(months=36)).date()
    # Pulling and cleaning the raw data
    raw_spx_data = yf.download(
        "^GSPC", start=str(fetch_date), interval="1mo", progress=False
    )
    yahoo_finance_inputs = pd.DataFrame(
        {
            "SPX_Close": raw_spx_data.loc[:, ("Close", "^GSPC")],
            "SPX_Volume": raw_spx_data.loc[:, ("Volume", "^GSPC")],
        }
    )
    # ROC Calcultations
    yahoo_finance_inputs["SPX_ROC"] = ta.roc(
        yahoo_finance_inputs["SPX_Close"], length=12
    )
    # RSI Calculations
    yahoo_finance_inputs["SPX_RSI"] = ta.rsi(
        yahoo_finance_inputs.loc[:, "SPX_Close"], length=14
    )
    # MACD Calcultations
    macd = ta.macd(yahoo_finance_inputs.loc[:, "SPX_Close"], fast=12, slow=26, signal=9)
    yahoo_finance_inputs = pd.concat([yahoo_finance_inputs, macd], axis=1)
    yahoo_finance_inputs.rename(
        {
            "MACD_12_26_9": "SPX_MACD",
            "MACDh_12_26_9": "SPX_MACDH",
            "MACDs_12_26_9": "SPX_MACDS",
        },
        inplace=True,
        axis=1,
    )
    # Adding VIX (Volatility Index)
    raw_vix_data = yf.download("^VIX", start=start_date, interval="1mo", progress=False)
    yahoo_finance_inputs["VIX_Close"] = raw_vix_data.loc[:, ("Close", "^VIX")]
    yahoo_finance_inputs = yahoo_finance_inputs[
        (yahoo_finance_inputs.index >= pd.Timestamp(start_date))
        & (yahoo_finance_inputs.index < pd.Timestamp(end_date))
    ].copy()
    return yahoo_finance_inputs


def get_fred_input_data(
    start_date: str, end_date: str, api_key: str = api_key
) -> pd.DataFrame:
    """Pulls macroeconomic data from FRED and aligns it to a monthly frequency.

    Args:
        start_date (str): The research start date (YYYY-MM-DD).
        end_date (str): end date (YYYY-MM-DD)
        api_key (str, optional): FRED api key. Defaults to api_key.

    Returns:
        pd.DataFrame: clean input FRED macro data.
    """
    # We again use a fetch date here, similar logic as above
    fetch_date = (pd.Timestamp(start_date) - pd.DateOffset(months=8)).date()
    fred = Fred(api_key=api_key)

    # This manually handles the SSL verification for each series
    def get_series_safe(series_id):
        return fred.get_series(series_id, observation_start=fetch_date)

    import urllib.request

    opener = urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=custom_context)
    )
    urllib.request.install_opener(opener)

    raw_macro_data = {
        "Real_GDP": get_series_safe("GDPC1"),
        "Unemployment": get_series_safe("UNRATE"),
        "Inflation": get_series_safe("PCEPILFE"),
        "Fed_Funds_Rate": get_series_safe("FEDFUNDS"),
        "10Y2Y_Spread": get_series_safe("T10Y2Y"),
    }

    fred_inputs = pd.DataFrame(raw_macro_data)

    # Alignment & Forward-filling (so there are no date mismatches between quarterly and monthly values)
    full_range = pd.date_range(start=fetch_date, end=fred_inputs.index.max(), freq="MS")
    fred_inputs = fred_inputs.reindex(full_range).ffill()
    fred_inputs = fred_inputs[
        (fred_inputs.index >= pd.Timestamp(start_date))
        & (fred_inputs.index < pd.Timestamp(end_date))
    ].copy()

    return fred_inputs



def get_futures_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Gets interest rate futures data (Fed Funds and 10Y Treasury).

    Args:
        start_date (str): start date (YYYY-MM-DD)
        end_date (str): end date (YYYY-MM-DD)

    Returns:
        pd.DataFrame: futures data (Close prices)
    """
    # Define tickers: ZQ=F (Fed Funds), ZN=F (10Y Note)
    tickers = {"Fed_Funds_Future": "ZQ=F", "10Y_Treasury_Future": "ZN=F"}
    
    # Fetch data
    raw_futures = yf.download(
        list(tickers.values()), 
        start=start_date, 
        end=end_date, 
        interval="1mo", 
        progress=False
    )

    # Clean multi-index columns if necessary
    futures_data = pd.DataFrame(index=raw_futures.index)
    for name, ticker in tickers.items():
        if ticker in raw_futures['Close'].columns:
            futures_data[name] = raw_futures['Close'][ticker]
            
    # Ensure monthly alignment (MS = Month Start) to match FRED data
    futures_data.index = futures_data.index.to_period('M').to_timestamp()
    
    return futures_data

