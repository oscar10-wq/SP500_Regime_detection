import os
import pandas as pd
from abc import ABC, abstractmethod
from fredapi import Fred
from dotenv import load_dotenv
from pathlib import Path


class BaseDataLoader(ABC):
    """
    Abstract interface for loading data from different sources.
    """

    @abstractmethod
    def connect(self):
        """
        Establish connection to the data source.
        Needed for FRED API and possibly other sources (not for yfinance though)."""
        pass

    @abstractmethod
    def get_data(self, symbol: str, start_date: str) -> pd.DataFrame:
        """
        Fetch data for a given symbol (Ticker or Series ID).
        Return a clean dataframe with a Date index.
        """
        pass


class FredDataLoader:
    """
    Handle data loading from FRED API.
    To get a FRED API key and for documentation, follow:
    https://fred.stlouisfed.org/docs/api/fred/
    (The website is sometimes insanely slow for some reason)
    """

    def __init__(self, env_path: str = None):
        """
        Initialize the FRED client.
        "env_path" argument is the custom path to the .env file.
        It defaults to finding a .env file in project root
        """
        self._load_api_key(env_path)
        self.client = Fred(api_key=self.api_key)

    def _load_api_key(self, env_path: str):
        """Safely load and validate the API key."""
        if env_path:
            load_dotenv(Path(env_path))
        else:
            load_dotenv(override=True)
        self.api_key = os.getenv("FRED_API_KEY")
        if not self.api_key:
            raise ValueError(
                "FRED_API_KEY not found. Please ensure you have a .env file "
                "with the key set, or set it as an environment variable."
            )

    def get_data(self, symbol: str, start_date: str = "2000-01-01") -> pd.DataFrame:
        """
        Standardised method to fetch a series.
        """
        try:
            print(f"Fetching {symbol}...")
            # FRED API returns pd.Series by default
            series = self.client.get_series(symbol, observation_start=start_date)
            df = pd.DataFrame(series, columns=[symbol])
            df.index.name = "Date"
            return df
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()

    def get_data_with_info(self, symbol: str, start_date: str = "2000-01-01"):
        """
        Fetches data and the metadata (title, units, frequency).
        """
        try:
            df = self.get_data(symbol, start_date)
            info = self.client.get_series_info(symbol)  # Metadata
            return {
                "data": df,
                "title": info.get("title"),
                "units": info.get("units"),
                "freq": info.get("frequency"),
            }
        except Exception as e:
            print(f"Error fetching info for {symbol}: {e}")
            return None

    def search(self, text: str, limit: int = 5):
        """
        Search for series IDs.
        """
        print(f"Searching FRED for '{text}'...")
        results = self.client.search(text)
        return results[["id", "title", "frequency_short"]].head(limit)

    def load_basic_regime_dataset(self, start_date="2000-01-01"):
        """
        Loads and aligns basic macro indicators (example / test).
        """
        # FEDFUNDS: Interest Rate (Monthly)
        # T10Y2Y: 10-Year minus 2-Year Treasury Yield (Daily)
        # UNRATE: Unemployment Rate (Monthly)
        # Will have to change this to allow for any series to be selected.
        series_ids = ["FEDFUNDS", "T10Y2Y", "UNRATE"]

        data_frames = []
        for s_id in series_ids:
            df = self.get_data(s_id, start_date)
            data_frames.append(df)

        macro_data = pd.concat(data_frames, axis=1, join="outer", sort=True)
        macro_data = macro_data.ffill()  # Handle frequency mismatch

        return macro_data.dropna()


# TEST STUFF
if __name__ == "__main__":
    # Might need to wrap this whole section in a function for analysis team
    script_dir = Path(__file__).parent
    env_path = script_dir / "config" / "api_keys.env"
    print(f"Loading keys from: {env_path}")
    loader = FredDataLoader(env_path)

    # Useful if you don't know the code (ID) of the series you are looking for
    print(loader.search("Consumer Price Index"))

    inflation = loader.get_data_with_info("CPIAUCSL")  # standard CPI
    print(f"\nLoaded: {inflation['title']} ({inflation['units']})")

    df_macro = loader.load_basic_regime_dataset()
    print("\n--------- Aligned Macro Data ---------")
    print(df_macro.tail())
