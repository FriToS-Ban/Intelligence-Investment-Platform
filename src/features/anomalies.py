import numpy as np
import pandas as pd

def detect_anomalies(df, z_threshold=3.0, volume_factor=3.0):
    """Add anomaly flag columns to a copy of df."""
    df = df.copy().sort_values('Date').reset_index(drop=True)

    # Fill NaN Daily_Return with 0 for Z-score (NaN at row 0 from pct_change)
    ret = df['Daily_Return'].fillna(0)
    ret_mean = ret.mean()
    ret_std  = ret.std()
    df['Return_ZScore']   = (ret - ret_mean) / (ret_std + 1e-10)
    df['Anomaly_Return']  = df['Return_ZScore'].abs() > z_threshold

    # Volatility spike — guard against all-NaN column
    if df['Volatility_30d'].notna().any():
        vol_threshold = df['Volatility_30d'].quantile(0.95)
        df['Anomaly_Volatility'] = df['Volatility_30d'] > vol_threshold
    else:
        df['Anomaly_Volatility'] = False

    # Volume spike
    df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
    valid_volume   = (df['Volume'] > 0) & (df['Vol_MA20'] > 0)
    df['Anomaly_Volume'] = valid_volume & (df['Volume'] > volume_factor * df['Vol_MA20'])

    df['Is_Anomaly'] = df['Anomaly_Return'] | df['Anomaly_Volatility'] | df['Anomaly_Volume']
    return df

def get_anomaly_summary(df):
    anomalies_df = detect_anomalies(df)
    anomalies    = anomalies_df[anomalies_df['Is_Anomaly']].copy()

    symbol_col = df['Symbol'].iloc[0] if 'Symbol' in df.columns else 'N/A'

    summary = []
    for _, row in anomalies.iterrows():
        reasons = []
        if row.get('Anomaly_Return', False):
            reasons.append(f"Return spike (Z={row['Return_ZScore']:.2f})")
        if row.get('Anomaly_Volatility', False):
            v = row['Volatility_30d']
            reasons.append(f"Volatility spike ({v*100:.1f}% annualised)")
        if row.get('Anomaly_Volume', False):
            ratio = row['Volume'] / (row['Vol_MA20'] + 1e-10)
            reasons.append(f"Volume surge ({ratio:.1f}× 20d MA)")

        summary.append({
            'Date':         row['Date'],
            'Symbol':       row.get('Symbol', symbol_col),
            'Close':        round(row['Close'], 2),
            'Daily_Return': f"{row['Daily_Return']*100:.2f}%" if pd.notna(row['Daily_Return']) else 'N/A',
            'Reason':       ", ".join(reasons)
        })

    return pd.DataFrame(summary)
