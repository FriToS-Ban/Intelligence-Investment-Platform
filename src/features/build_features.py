import os
import sys
import pandas as pd
import numpy as np

# Project root = two levels up from this file (src/features/build_features.py)
_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # Wilder's smoothing
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def process_stock_data(filepath, metadata=None):
    df = pd.read_csv(filepath)

    # Guard: skip empty or near-empty files (e.g. INFRATEL stub)
    if df.empty or len(df) < 20:
        raise ValueError(f"File {filepath} has too few rows ({len(df)}) — skipping.")

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Forward fill then backward fill missing prices/volumes
    # (bfill handles the rare case of NaN at the very beginning of the series)
    cols_to_fill = ['Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']
    for col in cols_to_fill:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()
    
    # Moving Averages
    df['MA20']  = df['Close'].rolling(window=20).mean()
    df['MA50']  = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # EMAs
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    df['MACD']        = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # RSI
    df['RSI'] = compute_rsi(df['Close'], 14)
    
    # Bollinger Bands
    df['BB_Mid']  = df['Close'].rolling(window=20).mean()
    df['BB_Std']  = df['Close'].rolling(window=20).std()
    df['BB_High'] = df['BB_Mid'] + 2 * df['BB_Std']
    df['BB_Low']  = df['BB_Mid'] - 2 * df['BB_Std']
    
    # Returns and volatility
    df['Daily_Return']   = df['Close'].pct_change()
    df['Volatility_30d'] = df['Daily_Return'].rolling(window=30).std() * np.sqrt(252)
    
    # Momentum
    df['Momentum_1m']  = df['Close'].pct_change(periods=21)
    df['Momentum_3m']  = df['Close'].pct_change(periods=63)
    df['Momentum_6m']  = df['Close'].pct_change(periods=126)
    df['Momentum_12m'] = df['Close'].pct_change(periods=252)
    
    # Targets for ML (next 5-day forward return and price direction)
    df['Target_5d_Return']    = df['Close'].shift(-5) / df['Close'] - 1
    df['Target_5d_Direction'] = (df['Target_5d_Return'] > 0).astype(int)
    df['Target_5d_Close']     = df['Close'].shift(-5)
    
    # Merge metadata if provided
    if metadata is not None and 'Symbol' in df.columns:
        symbol = str(df['Symbol'].iloc[0]).strip()
        # Strip whitespace from metadata Symbol column to avoid mismatches
        meta_row = metadata[metadata['Symbol'].str.strip() == symbol]
        if not meta_row.empty:
            df['Company Name'] = meta_row['Company Name'].values[0]
            df['Industry']     = meta_row['Industry'].values[0]
        else:
            df['Company Name'] = symbol
            df['Industry']     = 'Unknown'
    else:
        df['Company Name'] = df['Symbol'].iloc[0] if 'Symbol' in df.columns else 'Unknown'
        df['Industry']     = 'Unknown'
    
    return df

def run_pipeline(raw_dir, processed_dir):
    os.makedirs(processed_dir, exist_ok=True)
    metadata_path = os.path.join(raw_dir, 'stock_metadata.csv')
    if os.path.exists(metadata_path):
        metadata = pd.read_csv(metadata_path)
    else:
        metadata = None
        
    all_stocks = []
    
    for file in os.listdir(raw_dir):
        if file.endswith('.csv') and file != 'stock_metadata.csv' and file != 'NIFTY50_all.csv':
            filepath = os.path.join(raw_dir, file)
            print(f"Processing {file}...")
            try:
                processed_df = process_stock_data(filepath, metadata)
                # Save individual processed files
                processed_df.to_csv(os.path.join(processed_dir, f"{os.path.splitext(file)[0]}_processed.csv"), index=False)
                all_stocks.append(processed_df)
            except Exception as e:
                print(f"Error processing {file}: {e}")
                
    if all_stocks:
        consolidated = pd.concat(all_stocks, ignore_index=True)
        consolidated.to_csv(os.path.join(processed_dir, 'consolidated_stocks.csv'), index=False)
        print("Pipeline run completed successfully.")
        
if __name__ == "__main__":
    run_pipeline(
        raw_dir=os.path.join(PROJECT_ROOT, "data", "raw"),
        processed_dir=os.path.join(PROJECT_ROOT, "data", "processed")
    )
