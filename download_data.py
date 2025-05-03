#!/usr/bin/env python3
"""
Script to download stock data from Yahoo Finance and save it to the data directory
"""
import os
import sys
import argparse
from datetime import datetime
from data_fetcher import YahooFinanceDataFetcher

def download_stock_data(symbols, start_date, end_date, output_dir='data'):
    """
    Download stock data for the specified symbols and date range

    Args:
        symbols (list): List of stock symbols to download
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        output_dir (str): Directory to save data files
    """
    # Create the data fetcher
    fetcher = YahooFinanceDataFetcher(data_dir=output_dir)

    for symbol in symbols:
        print(f"\nDownloading {symbol} data...")
        df = fetcher.get_data(symbol, start_date=start_date, end_date=end_date)

        if df is not None and not df.empty:
            print(f"Successfully downloaded {len(df)} rows for {symbol}")
            print(f"Date range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
        else:
            print(f"Failed to download data for {symbol}")

def main():
    """Main function to parse arguments and download data"""
    parser = argparse.ArgumentParser(description='Download stock data from Yahoo Finance')
    parser.add_argument('symbols', nargs='+', help='Stock symbols to download (e.g., AAPL MSFT GOOG)')
    parser.add_argument('--start_date', type=str, default='2019-01-01',
                        help='Start date in YYYY-MM-DD format (default: 2019-01-01)')
    parser.add_argument('--end_date', type=str, default=datetime.now().strftime('%Y-%m-%d'),
                        help='End date in YYYY-MM-DD format (default: today)')
    parser.add_argument('--output_dir', type=str, default='data',
                        help='Directory to save data files (default: data)')

    args = parser.parse_args()

    print(f"Downloading data for: {', '.join(args.symbols)}")
    print(f"Date range: {args.start_date} to {args.end_date}")
    print(f"Output directory: {args.output_dir}")

    download_stock_data(args.symbols, args.start_date, args.end_date, args.output_dir)

    print("\nDownload complete!")

if __name__ == "__main__":
    main()