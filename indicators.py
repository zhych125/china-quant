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

        # Convert price and volume columns to numeric
        # This is crucial for calculations to work properly
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            if col in result.columns:
                result[col] = pd.to_numeric(result[col], errors='coerce')

        # Check if conversion worked and we have valid numeric data
        if result['Close'].isna().all():
            print("Error: Could not convert 'Close' column to numeric values")
            return df

        print(f"Data types after conversion: {result.dtypes[numeric_columns]}")

        # Add normalized versions of price and volume data
        result = self.add_normalized_data(result)

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

        # Normalize unbounded technical indicators
        result = self.normalize_technical_indicators(result)

        return result

    def add_normalized_data(self, df):
        """
        Add normalized versions of price and volume data

        Args:
            df (DataFrame): OHLCV DataFrame

        Returns:
            DataFrame: DataFrame with added normalized columns
        """
        # Create period-based normalization (normalize to previous N-day window)
        # Using a 20-day window for normalization (roughly one trading month)
        window_size = 20

        # Z-score normalization for price data (Open, High, Low, Close)
        for col in ['Open', 'High', 'Low', 'Close']:
            # Rolling mean and standard deviation
            rolling_mean = df[col].rolling(window=window_size).mean()
            rolling_std = df[col].rolling(window=window_size).std()

            # Z-score normalization: (x - mean) / std
            # Handle zero std case to avoid division by zero
            df[f'norm_{col}'] = (df[col] - rolling_mean) / rolling_std.replace(0, 1)

        # Log transformation + Z-score for Volume (volume is often highly skewed)
        df['log_volume'] = np.log1p(df['Volume'])  # log(1+x) to handle zeros
        rolling_mean_vol = df['log_volume'].rolling(window=window_size).mean()
        rolling_std_vol = df['log_volume'].rolling(window=window_size).std()
        df['norm_Volume'] = (df['log_volume'] - rolling_mean_vol) / rolling_std_vol.replace(0, 1)

        # Also add min-max normalized price relative to recent window
        for col in ['Open', 'High', 'Low', 'Close']:
            rolling_min = df[col].rolling(window=window_size).min()
            rolling_max = df[col].rolling(window=window_size).max()

            # Min-max: (x - min) / (max - min)
            # Handle case where min == max to avoid division by zero
            denominator = rolling_max - rolling_min
            denominator = denominator.replace(0, 1)  # Replace zeros with ones
            df[f'minmax_{col}'] = (df[col] - rolling_min) / denominator

        # Drop intermediate log_volume column
        df = df.drop('log_volume', axis=1)

        return df

    def normalize_technical_indicators(self, df):
        """
        Normalize derived technical indicators that don't have inherent bounds

        Args:
            df (DataFrame): DataFrame with technical indicators

        Returns:
            DataFrame: DataFrame with normalized technical indicators
        """
        # List of already normalized/bounded indicators that don't need normalization
        # RSI is 0-100, Bollinger band % is 0-1, etc.
        bounded_indicators = [
            'rsi_14',             # RSI is already bounded 0-100
            'boll_pct',           # Already normalized between 0-1
            'up_days',            # Count of up days, already bounded 0-5
            'price_ma5_ratio',    # Already a ratio
            'price_ma10_ratio'    # Already a ratio
        ]

        # List of indicators known to be potentially unbounded
        unbounded_indicators = [
            # Volatility metrics
            'volatility_5', 'volatility_10', 'high_low_range', 'atr_14',

            # Momentum metrics
            'momentum',

            # MACD metrics
            'macd', 'macd_signal', 'macd_diff',

            # Return metrics
            'return_1', 'return_3', 'return_5',

            # Volume metrics
            'volume_ratio_5'
        ]

        # Window for normalization
        window_size = 20

        # Apply Z-score normalization to unbounded indicators
        for col in unbounded_indicators:
            if col in df.columns:
                # Calculate rolling mean and std
                rolling_mean = df[col].rolling(window=window_size).mean()
                rolling_std = df[col].rolling(window=window_size).std()

                # Create normalized version
                norm_col = f'norm_{col}'
                df[norm_col] = (df[col] - rolling_mean) / rolling_std.replace(0, 1)

                # Drop original column and keep normalized version
                df = df.drop(col, axis=1)

        # Bollinger bands don't need independent normalization as we already have boll_pct
        bands_to_drop = ['boll_upper', 'boll_lower']
        for col in bands_to_drop:
            if col in df.columns:
                df = df.drop(col, axis=1)

        # Drop moving averages too as their relationship to price is captured in ratios
        mas_to_drop = ['ma_5', 'ma_10', 'ma_20']
        for col in mas_to_drop:
            if col in df.columns:
                df = df.drop(col, axis=1)

        return df

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
    df = data_fetcher.get_data('AAPL', start_date="2020-01-01", end_date="2023-01-01")

    if df is not None and not df.empty:
        # Create a technical indicator generator and add indicators
        indicator_generator = TechnicalIndicators()
        df_with_indicators = indicator_generator.add_all_indicators(df)

        # Print information about the processed DataFrame
        print("\nDataFrame with Indicators:")
        print(f"Shape: {df_with_indicators.shape}")
        print(f"Columns: {', '.join(df_with_indicators.columns)}")

        # Show the difference in number of features
        print(f"Original features: {len(df.columns)}")
        print(f"Normalized features: {len(df_with_indicators.columns)}")

        # Show sample of the data with indicators
        print("\nSample of data with indicators:")
        print(df_with_indicators.tail(5))
    else:
        print("No data available for testing")
