import pandas as pd
import numpy as np


class TechnicalIndicators:
    """Class for calculating various technical indicators"""

    def __init__(self):
        pass

    def add_all_indicators(self, df):
        """
        Add all technical indicators to the DataFrame

        Args:
            df (DataFrame): OHLCV DataFrame

        Returns:
            DataFrame: DataFrame with added indicators
        """
        if df is None or df.empty:
            return df

        # Create a copy to avoid modifying the original DataFrame
        result = df.copy()

        # Make sure we have expected columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in result.columns for col in required_columns):
            print("Warning: DataFrame is missing required columns")
            print(f"Expected: {required_columns}")
            print(f"Got: {result.columns}")
            return result

        # Add each group of indicators
        result = self.add_returns(result)
        result = self.add_moving_averages(result)
        result = self.add_volatility(result)
        result = self.add_volume(result)
        result = self.add_momentum(result)
        result = self.add_rsi(result)
        result = self.add_macd(result)
        result = self.add_bollinger_bands(result)

        # Fill NaN values instead of dropping rows
        result = result.fillna(method='bfill').fillna(method='ffill')

        return result

    def add_returns(self, df):
        """Add return indicators"""
        # 1, 3, 5 day percentage returns
        df['return_1'] = df['Close'].pct_change(1)
        df['return_3'] = df['Close'].pct_change(3)
        df['return_5'] = df['Close'].pct_change(5)

        return df

    def add_moving_averages(self, df):
        """Add moving average indicators"""
        # Simple moving averages
        df['ma_5'] = df['Close'].rolling(window=5).mean()
        df['ma_10'] = df['Close'].rolling(window=10).mean()
        df['ma_20'] = df['Close'].rolling(window=20).mean()

        # Price to moving average ratios
        df['price_ma5_ratio'] = df['Close'] / df['ma_5']
        df['price_ma10_ratio'] = df['Close'] / df['ma_10']

        return df

    def add_volatility(self, df):
        """Add volatility indicators"""
        # Volatility (standard deviation of returns)
        df['volatility_5'] = df['Close'].rolling(window=5).std()
        df['volatility_10'] = df['Close'].rolling(window=10).std()

        # High-Low range as a percentage of close
        df['high_low_range'] = (df['High'] - df['Low'])

        df['atr_14'] = (df['High'] - df['Low']).rolling(window=14).mean()

        return df

    def add_volume(self, df):
        """Add volume indicators"""
        # Volume ratio (current volume / 5-day average volume)
        df['volume_ratio_5'] = df['Volume'] / df['Volume'].rolling(window=5).mean()

        return df

    def add_momentum(self, df):
        """Add momentum indicators"""
        # Number of up days in last 5 days
        df['momentum'] = df['Close'] - df['Open']
        df['up_days'] = (df['Close'] > df['Open']).rolling(window=5).sum()

        return df

    def add_rsi(self, df):
        """Add Relative Strength Index"""
        # RSI - 14 period is standard
        delta = df['Close'].diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        avg_gain = up.rolling(14).mean()
        avg_loss = down.rolling(14).mean()
        rs = avg_gain / avg_loss
        df['rsi_14'] = 100 - (100 / (1 + rs))

        return df

    def add_macd(self, df):
        """Add Moving Average Convergence Divergence"""
        # MACD with standard parameters (12, 26, 9)
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']

        return df

    def add_bollinger_bands(self, df):
        """Add Bollinger Bands"""
        # Bollinger Bands - 20 period is standard
        ma_20 = df['Close'].rolling(20).mean()
        std_20 = df['Close'].rolling(20).std()
        df['boll_upper'] = ma_20 + 2 * std_20
        df['boll_lower'] = ma_20 - 2 * std_20
        df['boll_pct'] = (df['Close'] - df['boll_lower']) / (df['boll_upper'] - df['boll_lower'])

        return df


if __name__ == "__main__":
    # Simple test with sample data
    from data_fetcher import CSVDataFetcher

    # Create a data fetcher and get AAPL data
    data_fetcher = CSVDataFetcher()
    df = data_fetcher.get_data('AAPL', years=3)

    if df is not None and not df.empty:
        # Create a technical indicator generator and add indicators
        indicator_generator = TechnicalIndicators()
        df_with_indicators = indicator_generator.add_all_indicators(df)

        # Print information about the processed DataFrame
        print("\nDataFrame with Indicators:")
        print(f"Shape: {df_with_indicators.shape}")
        print(f"Columns: {', '.join(df_with_indicators.columns)}")

        # Show sample of the data with indicators
        print("\nSample of data with indicators:")
        print(df_with_indicators.tail(5))
    else:
        print("No data available for testing")
