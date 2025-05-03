import os
import datetime
import glob
import pandas as pd
import backtrader as bt
from abc import ABC, abstractmethod


class DataFetcher(ABC):
    """Interface for data fetchers that load financial data"""

    @abstractmethod
    def get_data(self, symbol, start_date=None, end_date=None, years=None):
        """
        Get data for a symbol within a date range

        Args:
            symbol (str): The ticker symbol
            start_date (str or datetime, optional): Start date
            end_date (str or datetime, optional): End date
            years (int, optional): Number of years of data to fetch

        Returns:
            DataFrame: Pandas DataFrame with the data
        """
        pass


class CSVDataFetcher(DataFetcher):
    """Fetches data from CSV files in a directory"""

    def __init__(self, data_dir='data'):
        self.data_dir = data_dir

    def _find_csv_files(self, symbol, years=None):
        """Find CSV files for a symbol with optional year filtering"""
        # Get all files for the symbol
        all_files = glob.glob(os.path.join(self.data_dir, f'{symbol}*.csv'))

        if years is None:
            return all_files

        # Filter for files covering specific years
        filtered_files = []

        # Look for the specific year period in filename
        if years == 3:
            # Look for 2020-2023 (3 years)
            for file_path in all_files:
                filename = os.path.basename(file_path)
                if '2020-01-01_2023-12-31' in filename:
                    filtered_files.append(file_path)

        # If no specific file is found, get the largest one as a fallback
        if not filtered_files and all_files:
            filtered_files = [sorted(all_files, key=os.path.getsize, reverse=True)[0]]

        return filtered_files

    def get_data(self, symbol, start_date=None, end_date=None, years=None):
        """
        Get stock data from CSV files

        Args:
            symbol (str): Stock symbol (e.g., 'AAPL')
            start_date (str or datetime, optional): Start date
            end_date (str or datetime, optional): End date
            years (int, optional): Number of years of data

        Returns:
            DataFrame: Pandas DataFrame with OHLCV data
        """
        # Find the appropriate CSV files
        csv_files = self._find_csv_files(symbol, years)

        if not csv_files:
            print(f"No CSV files found for {symbol}")
            return None

        # Load data from the first matching file
        file_path = csv_files[0]
        print(f"Loading data from: {file_path}")

        # Read the CSV file
        df = pd.read_csv(file_path)

        # Ensure the date column is parsed as datetime
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)

        # Apply date filters if provided
        if start_date:
            start_date = pd.to_datetime(start_date)
            df = df[df.index >= start_date]

        if end_date:
            end_date = pd.to_datetime(end_date)
            df = df[df.index <= end_date]

        return df


class BacktraderFeeder:
    """Feeds data into Backtrader for analysis"""

    def __init__(self, data_fetcher=None):
        self.data_fetcher = data_fetcher or CSVDataFetcher()
        self.cerebro = bt.Cerebro()

    def add_data(self, symbol, start_date=None, end_date=None, years=None):
        """Add data for a symbol to Backtrader"""
        # Get the data as DataFrame
        df = self.data_fetcher.get_data(symbol, start_date, end_date, years)

        if df is None or df.empty:
            print(f"No data available for {symbol}")
            return False

        # Ensure the DataFrame has the expected format
        # The index should be a datetime and columns should include OHLCV
        if not isinstance(df.index, pd.DatetimeIndex):
            print("DataFrame index must be a DatetimeIndex")
            return False

        # Create a PandasData feed directly from the DataFrame
        data = bt.feeds.PandasData(
            dataname=df,
            # Map DataFrame columns to Backtrader expected names
            # If column exists, use it, otherwise use the default column index
            open='Open',
            high='High',
            low='Low',
            close='Close',
            volume='Volume',
            openinterest=-1  # -1 means not present
        )

        self.cerebro.adddata(data, name=symbol)
        return True

    def run(self):
        """Run the Backtrader engine"""
        # Create a simple strategy to load the data
        class DataLoadStrategy(bt.Strategy):
            def __init__(self):
                pass

            def next(self):
                pass

        # Add the strategy
        self.cerebro.addstrategy(DataLoadStrategy)

        # Run cerebro to load the data
        return self.cerebro.run()


# Helper functions for displaying data
def print_dataframe_info(df, symbol):
    """Print information about a pandas DataFrame"""
    print(f"\nDataFrame Info for {symbol}:")
    print(f"  - Shape: {df.shape}")
    print(f"  - Date Range: {df.index.min()} to {df.index.max()}")
    print(f"  - Columns: {', '.join(df.columns)}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nLast 5 rows:")
    print(df.tail())


def print_backtrader_info(cerebro):
    """Print information about loaded Backtrader data feeds"""
    if not cerebro.datas:
        print("No data feeds have been loaded.")
        return

    for i, d in enumerate(cerebro.datas):
        print(f"\nData Feed {i}:")
        print(f"  - Name: {d._name}")

        # Check if the data has been loaded
        if len(d) > 0:
            try:
                print(f"  - Start Date: {d.datetime.datetime(0)}")
                print(f"  - End Date: {d.datetime.datetime(-1)}")
                print(f"  - Number of bars: {len(d)}")
                print(f"  - Current Open: {d.open[0]}")
                print(f"  - Current High: {d.high[0]}")
                print(f"  - Current Low: {d.low[0]}")
                print(f"  - Current Close: {d.close[0]}")
                print(f"  - Current Volume: {d.volume[0]}")
            except (IndexError, ValueError) as e:
                print(f"  - Error accessing data: {e}")
                print(f"  - Data may not be properly loaded or format is incorrect")
        else:
            print(f"  - No data bars loaded")

        print("------------------------------")


if __name__ == "__main__":
    # Test the CSVDataFetcher
    symbol = "AAPL"
    years = 3

    print(f"Testing CSVDataFetcher with {symbol} for {years} years")

    # Create data fetcher and load data into pandas DataFrame
    csv_fetcher = CSVDataFetcher()
    df = csv_fetcher.get_data(symbol, years=years)

    if df is not None and not df.empty:
        # Print information about the DataFrame
        print_dataframe_info(df, symbol)

        # Test loading into Backtrader
        print(f"\nLoading {symbol} data into Backtrader")
        bt_feeder = BacktraderFeeder(csv_fetcher)

        # Add data and run
        if bt_feeder.add_data(symbol, years=years):
            results = bt_feeder.run()

            # Print Backtrader data information
            print_backtrader_info(bt_feeder.cerebro)

            print("\nBacktrader data loaded successfully!")
        else:
            print(f"Failed to add {symbol} data to Backtrader")
    else:
        print(f"No data found for {symbol}")