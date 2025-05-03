import pandas as pd
import numpy as np
from indicators import TechnicalIndicators


class Strategy:
    """Base class for trading strategies"""

    def __init__(self, name="Base Strategy"):
        self.name = name
        self.indicators = TechnicalIndicators()

    def generate_signals(self, df):
        """
        Generate trading signals

        Args:
            df (DataFrame): OHLCV DataFrame

        Returns:
            DataFrame: DataFrame with added signals
        """
        # This is a base method to be overridden by specific strategies
        return df


class MACDStrategy(Strategy):
    """MACD-based trading strategy"""

    def __init__(self):
        super().__init__(name="MACD Strategy")

    def generate_signals(self, df):
        """
        Generate trading signals based on MACD

        Args:
            df (DataFrame): OHLCV DataFrame

        Returns:
            DataFrame: DataFrame with added signals
        """
        # Make a copy to avoid modifying the original
        result = df.copy()

        # Ensure DataFrame has the necessary indicators
        if not all(col in result.columns for col in ['macd', 'macd_signal']):
            result = self.indicators.add_all_indicators(result)

        # Add a signal column (1 for buy, -1 for sell, 0 for hold/neutral)
        result['signal'] = 0

        # Calculate previous values for crossover detection
        result['prev_macd'] = result['macd'].shift(1)
        result['prev_macd_signal'] = result['macd_signal'].shift(1)

        # Buy signal: MACD crosses above signal line
        buy_signal = (result['macd'] > result['macd_signal']) & \
                     (result['prev_macd'] <= result['prev_macd_signal'])

        # Sell signal: MACD crosses below signal line
        sell_signal = (result['macd'] < result['macd_signal']) & \
                      (result['prev_macd'] >= result['prev_macd_signal'])

        # Apply signals to DataFrame
        result.loc[buy_signal, 'signal'] = 1
        result.loc[sell_signal, 'signal'] = -1

        # Clean up temporary columns
        result = result.drop(['prev_macd', 'prev_macd_signal'], axis=1)

        return result


class StrategyTester:
    """Class for testing and evaluating trading strategies"""

    def __init__(self):
        pass

    def test_strategy(self, df, strategy):
        """
        Test a trading strategy on historical data

        Args:
            df (DataFrame): OHLCV DataFrame
            strategy (Strategy): Strategy to test

        Returns:
            DataFrame: DataFrame with signals and performance metrics
        """
        # Generate signals
        result = strategy.generate_signals(df)

        # Calculate strategy performance
        result = self._calculate_performance(result)

        return result

    def _calculate_performance(self, df):
        """
        Calculate performance metrics for the strategy

        Args:
            df (DataFrame): DataFrame with signals

        Returns:
            DataFrame: DataFrame with performance metrics
        """
        # Copy the DataFrame
        result = df.copy()

        # Position: 1 for long, 0 for cash, -1 for short
        # Initialize position to 0
        result['position'] = 0

        # Create positions based on signals (assuming signals are 1, -1, 0)
        # Position starts from the next day after the signal
        result['position'] = result['signal'].shift(1)

        # Fill NaN values with 0
        result['position'] = result['position'].fillna(0)

        # Calculate returns
        # Market return (buy and hold)
        result['market_return'] = result['Close'].pct_change()

        # Strategy return
        result['strategy_return'] = result['position'] * result['market_return']

        # Cumulative returns
        result['cum_market_return'] = (1 + result['market_return']).cumprod() - 1
        result['cum_strategy_return'] = (1 + result['strategy_return']).cumprod() - 1

        return result


if __name__ == "__main__":
    # Test the strategy with sample data
    from data_fetcher import CSVDataFetcher

    # Create a data fetcher and get AAPL data
    data_fetcher = CSVDataFetcher()
    df = data_fetcher.get_data('AAPL', years=3)

    if df is not None and not df.empty:
        # Create and apply indicators
        indicator_generator = TechnicalIndicators()
        df_with_indicators = indicator_generator.add_all_indicators(df)

        # Create strategy and generate signals
        macd_strategy = MACDStrategy()
        strategy_tester = StrategyTester()

        # Test the strategy
        result_df = strategy_tester.test_strategy(df_with_indicators, macd_strategy)

        # Print summary statistics
        print(f"\nStrategy: {macd_strategy.name}")
        print(f"Time period: {result_df.index.min()} to {result_df.index.max()}")
        print(f"Number of trading days: {len(result_df)}")

        # Count buy and sell signals
        buy_signals = result_df[result_df['signal'] == 1].shape[0]
        sell_signals = result_df[result_df['signal'] == -1].shape[0]
        print(f"Buy signals: {buy_signals}")
        print(f"Sell signals: {sell_signals}")

        # Calculate performance metrics
        total_market_return = result_df['cum_market_return'].iloc[-1] * 100
        total_strategy_return = result_df['cum_strategy_return'].iloc[-1] * 100

        print(f"\nPerformance Summary:")
        print(f"Market return: {total_market_return:.2f}%")
        print(f"Strategy return: {total_strategy_return:.2f}%")
        print(f"Outperformance: {total_strategy_return - total_market_return:.2f}%")

        # Print recent signals
        print("\nMost recent signals:")
        recent_signals = result_df[result_df['signal'] != 0].tail(5)
        if not recent_signals.empty:
            for date, row in recent_signals.iterrows():
                signal_type = "BUY" if row['signal'] == 1 else "SELL"
                print(f"{date.date()}: {signal_type} - Close: {row['Close']:.2f}, MACD: {row['macd']:.4f}, Signal: {row['macd_signal']:.4f}")
        else:
            print("No recent signals found in the data")
    else:
        print("No data available for testing")