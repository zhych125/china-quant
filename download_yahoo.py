#!/usr/bin/env python3
"""
Simple script to download AAPL data directly using yfinance
"""
import os
import pandas as pd
import yfinance as yf
from datetime import datetime

def main():
    symbol = "AAPL"
    start_date = "2019-01-01"
    end_date = "2024-12-31"
    output_dir = "data"

    print(f"Downloading {symbol} data from {start_date} to {end_date}...")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Download data
        data = yf.download(symbol, start=start_date, end=end_date)

        if data.empty:
            print(f"No data available for {symbol}")
            return

        # Format the output filename
        filename = f"{symbol}_{start_date}_{end_date}.csv"
        file_path = os.path.join(output_dir, filename)

        # Save to CSV and make sure Date is a column, not the index
        data.reset_index(inplace=True)  # Move Date from index to column
        data.to_csv(file_path, index=False)

        print(f"Successfully downloaded {len(data)} rows of {symbol} data")
        print(f"Date range: {data['Date'].min().strftime('%Y-%m-%d')} to {data['Date'].max().strftime('%Y-%m-%d')}")
        print(f"Saved to: {file_path}")

        # Display sample data
        print("\nSample data:")
        print(data.head())

    except Exception as e:
        print(f"Error downloading {symbol} data: {e}")

if __name__ == "__main__":
    main()