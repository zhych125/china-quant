import os
import datetime
import glob
import pandas as pd
import backtrader as bt
from abc import ABC, abstractmethod
import yfinance as yf


class DataFetcher(ABC):
    """Interface for data fetchers that load financial data"""

    @abstractmethod
    def get_data(self, symbol, start_date=None, end_date=None):
        """
        Get data for a symbol within a date range

        Args:
            symbol (str): The ticker symbol
            start_date (str or datetime, optional): Start date
            end_date (str or datetime, optional): End date

        Returns:
            DataFrame: Pandas DataFrame with the data
        """
        pass


class YahooFinanceDataFetcher(DataFetcher):
    """Fetches data directly from Yahoo Finance"""

    def __init__(self, data_dir='data', auto_save=True):
        """
        Initialize the Yahoo Finance data fetcher

        Args:
            data_dir (str): Directory to save the data files
            auto_save (bool): Whether to automatically save data to CSV file
        """
        self.data_dir = data_dir
        self.auto_save = auto_save

        # Create data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

    def get_data(self, symbol, start_date=None, end_date=None):
        """
        Get stock data directly from Yahoo Finance

        Args:
            symbol (str): Stock symbol (e.g., 'AAPL')
            start_date (str or datetime, optional): Start date
            end_date (str or datetime, optional): End date

        Returns:
            DataFrame: Pandas DataFrame with OHLCV data
        """
        print(f"Fetching {symbol} data from Yahoo Finance...")

        if start_date is None:
            start_date = "2019-01-01"
        if end_date is None:
            end_date = datetime.datetime.now().strftime("%Y-%m-%d")

        # Format dates if they are datetime objects
        if isinstance(start_date, datetime.datetime):
            start_date = start_date.strftime("%Y-%m-%d")
        if isinstance(end_date, datetime.datetime):
            end_date = end_date.strftime("%Y-%m-%d")

        print(f"Date range: {start_date} to {end_date}")

        try:
            # Download data from Yahoo Finance
            df = yf.download(symbol, start=start_date, end=end_date)

            if df.empty:
                print(f"No data available for {symbol}")
                return None

            # Make sure index is a DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                print("Warning: DataFrame index is not a DatetimeIndex. Converting...")
                df.index = pd.to_datetime(df.index)

            print(f"Successfully retrieved {len(df)} rows of {symbol} data")
            print(f"Data range: {df.index.min()} to {df.index.max()}")

            # Apply explicit date filters if needed (yfinance should already do this,
            # but we do it again for consistency)
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)

            df = df[(df.index >= start_dt) & (df.index <= end_dt)]

            # Auto-save if enabled
            if self.auto_save:
                self.save_data(df, symbol, start_date, end_date)

            return df

        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None

    def save_data(self, df, symbol, start_date, end_date):
        """
        Save data to a CSV file

        Args:
            df (DataFrame): Data to save
            symbol (str): Stock symbol
            start_date (str): Start date
            end_date (str): End date

        Returns:
            str: Path to the saved file
        """
        if df is None or df.empty:
            print("No data to save")
            return None

        # Format filename
        filename = f"{symbol}_{start_date}_{end_date}.csv"
        file_path = os.path.join(self.data_dir, filename)

        # Save to CSV
        df.to_csv(file_path)
        print(f"Data saved to: {file_path}")

        return file_path


class CSVDataFetcher(DataFetcher):
    """Fetches data from CSV files in a directory"""

    def __init__(self, data_dir='data'):
        self.data_dir = data_dir

    def _find_csv_files(self, symbol, start_date=None, end_date=None):
        """Find CSV files for a symbol with optional date filtering"""
        # Get all files for the symbol
        all_files = glob.glob(os.path.join(self.data_dir, f'{symbol}*.csv'))

        if not all_files:
            return []

        # If date filtering is required, try to find a file matching the date range
        if start_date or end_date:
            # Parse dates if they're strings
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)

            filtered_files = []
            for file_path in all_files:
                filename = os.path.basename(file_path)
                # Look for date pattern in filename (assuming format like SYMBOL_YYYY-MM-DD_YYYY-MM-DD.csv)
                file_parts = filename.split('_')
                if len(file_parts) >= 3:
                    try:
                        file_start = pd.to_datetime(file_parts[1])
                        file_end = pd.to_datetime(file_parts[2].split('.')[0])

                        # Check if file's date range overlaps with requested range
                        if ((start_date is None or file_end >= start_date) and
                            (end_date is None or file_start <= end_date)):
                            filtered_files.append(file_path)
                    except:
                        # If we can't parse dates from filename, keep the file
                        filtered_files.append(file_path)
                else:
                    # No date information in filename, keep the file
                    filtered_files.append(file_path)

            if filtered_files:
                return filtered_files

        # If no specific file is found, get the largest one as a fallback
        if all_files:
            return [sorted(all_files, key=os.path.getsize, reverse=True)[0]]

        return []

    def get_data(self, symbol, start_date=None, end_date=None):
        """
        Get stock data from CSV files

        Args:
            symbol (str): Stock symbol (e.g., 'AAPL')
            start_date (str or datetime, optional): Start date
            end_date (str or datetime, optional): End date

        Returns:
            DataFrame: Pandas DataFrame with OHLCV data
        """
        # Find the appropriate CSV files
        csv_files = self._find_csv_files(symbol, start_date, end_date)

        if not csv_files:
            print(f"No CSV files found for {symbol}")
            return None

        # Load data from the first matching file
        file_path = csv_files[0]
        print(f"Loading data from: {file_path}")

        try:
            # Read the CSV file
            df = pd.read_csv(file_path)

            # Check if 'Date' is already a column
            if 'Date' in df.columns:
                # Convert to datetime
                df['Date'] = pd.to_datetime(df['Date'])
                # Set as index if not already
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.set_index('Date', inplace=True)
            else:
                # If index is numeric and we don't have a Date column,
                # the file format might be incompatible
                print("Warning: No 'Date' column found in the CSV file")
                return None

            # Apply date filters if provided
            if start_date:
                start_date = pd.to_datetime(start_date)
                df = df[df.index >= start_date]

            if end_date:
                end_date = pd.to_datetime(end_date)
                df = df[df.index <= end_date]

            print(f"Successfully loaded {len(df)} rows of data")
            print(f"Date range: {df.index.min()} to {df.index.max()}")

            return df

        except Exception as e:
            print(f"Error loading data from {file_path}: {e}")
            print("Try running download_yahoo.py to fetch fresh data.")
            return None

    def save_data(self, df, symbol, start_date, end_date):
        """Save data to a CSV file"""
        if df is None or df.empty:
            print("No data to save")
            return None

        # Format filename
        start_str = start_date
        end_str = end_date

        if isinstance(start_date, datetime.datetime):
            start_str = start_date.strftime("%Y-%m-%d")
        if isinstance(end_date, datetime.datetime):
            end_str = end_date.strftime("%Y-%m-%d")

        filename = f"{symbol}_{start_str}_{end_str}.csv"
        file_path = os.path.join(self.data_dir, filename)

        # Save to CSV
        df.to_csv(file_path)
        print(f"Data saved to: {file_path}")

        return file_path


class BacktraderFeeder:
    """Feeds data into Backtrader for analysis"""

    def __init__(self, data_fetcher=None):
        self.data_fetcher = data_fetcher or CSVDataFetcher()
        self.cerebro = bt.Cerebro()

    def add_data(self, symbol, start_date=None, end_date=None):
        """Add data for a symbol to Backtrader"""
        # Get the data as DataFrame
        df = self.data_fetcher.get_data(symbol, start_date, end_date)

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
    # Test both the CSVDataFetcher and YahooFinanceDataFetcher
    symbol = "AAPL"
    start_date = "2019-01-01"
    end_date = "2022-12-31"

    print(f"Testing data fetchers with {symbol} from {start_date} to {end_date}")

    # First try YahooFinanceDataFetcher to get fresh data
    print("\n1. Testing YahooFinanceDataFetcher:")
    yahoo_fetcher = YahooFinanceDataFetcher()
    df_yahoo = yahoo_fetcher.get_data(symbol, start_date=start_date, end_date=end_date)

    if df_yahoo is not None and not df_yahoo.empty:
        print_dataframe_info(df_yahoo, f"{symbol} (Yahoo Finance)")
    else:
        print(f"Could not fetch {symbol} data from Yahoo Finance")

    # Then test CSVDataFetcher to load from saved data
    print("\n2. Testing CSVDataFetcher:")
    csv_fetcher = CSVDataFetcher()
    df_csv = csv_fetcher.get_data(symbol, start_date=start_date, end_date=end_date)

    if df_csv is not None and not df_csv.empty:
        # Print information about the DataFrame
        print_dataframe_info(df_csv, f"{symbol} (CSV)")

        # Test loading into Backtrader
        print(f"\nLoading {symbol} data into Backtrader")
        bt_feeder = BacktraderFeeder(csv_fetcher)

        # Add data and run
        if bt_feeder.add_data(symbol, start_date=start_date, end_date=end_date):
            results = bt_feeder.run()

            # Print Backtrader data information
            print_backtrader_info(bt_feeder.cerebro)

            print("\nBacktrader data loaded successfully!")
        else:
            print(f"Failed to add {symbol} data to Backtrader")
    else:
        print(f"No CSV data found for {symbol}")