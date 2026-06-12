import numpy as np
import pandas as pd

def calculate_volatility(returns, annual_factor=252):
    return returns.std() * np.sqrt(annual_factor)

def calculate_sharpe_ratio(returns, rf_rate=0.06, annual_factor=252):
    vol = calculate_volatility(returns, annual_factor)
    if vol == 0:
        return 0
    # Annualized return
    avg_daily_return = returns.mean()
    ann_return = avg_daily_return * annual_factor
    return (ann_return - rf_rate) / vol

def calculate_sortino_ratio(returns, rf_rate=0.06, annual_factor=252):
    downside_returns = returns[returns < 0]
    if len(downside_returns) == 0:
        # No negative returns at all — extremely strong risk-adjusted return
        ann_return = returns.mean() * annual_factor
        return (ann_return - rf_rate) / 1e-6  # effectively infinity; cap in caller
    downside_vol = downside_returns.std() * np.sqrt(annual_factor)
    if downside_vol == 0 or np.isnan(downside_vol):
        return 0.0
    ann_return = returns.mean() * annual_factor
    return (ann_return - rf_rate) / downside_vol

def calculate_max_drawdown(prices):
    if len(prices) == 0:
        return 0
    cum_returns = prices / prices.iloc[0]
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    return drawdown.min()

def calculate_var_95(returns):
    # Historical Value at Risk at 95% confidence level (5th percentile)
    if len(returns) == 0:
        return 0.0
    return np.percentile(returns.dropna(), 5)

def calculate_beta(stock_returns, index_returns):
    # Align both series to have the same dates
    aligned = pd.concat([stock_returns, index_returns], axis=1).dropna()
    if aligned.empty or len(aligned) < 2:
        return 0.0
    
    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
    market_var = np.var(aligned.iloc[:, 1])
    if market_var == 0:
        return 0.0
    # Beta = Cov(stock, index) / Var(index)
    return cov[0, 1] / market_var

def get_risk_profile_metrics(df, index_df=None, rf_rate=0.06):
    """Compute risk metrics for a single stock DataFrame.
    
    Args:
        df: DataFrame with 'Close', 'Daily_Return', and 'Date' columns.
        index_df: Optional NIFTY-50 index DataFrame for beta calculation.
        rf_rate: Annual risk-free rate (default 6% for India).
    Returns:
        dict of labelled risk metrics.
    """
    returns = df['Daily_Return'].dropna()
    prices  = df.sort_values('Date')['Close'].reset_index(drop=True)

    if len(returns) == 0:
        return {
            'Annualized Volatility': 0.0, 'Sharpe Ratio': 0.0,
            'Sortino Ratio': 0.0, 'Max Drawdown': 0.0,
            'VaR (95%)': 0.0, 'Beta': 1.0
        }

    vol     = calculate_volatility(returns)
    sharpe  = calculate_sharpe_ratio(returns, rf_rate)
    sortino = calculate_sortino_ratio(returns, rf_rate)
    # Cap Sortino at ±999 for display sanity (no-downside-return edge case)
    sortino = float(np.clip(sortino, -999.0, 999.0))
    max_dd  = calculate_max_drawdown(prices)
    var_95  = calculate_var_95(returns)

    beta = 1.0  # default if no index provided
    if index_df is not None and not index_df.empty:
        # Align on Date column for correct beta calculation
        aligned = pd.merge(
            df[['Date', 'Daily_Return']],
            index_df[['Date', 'Daily_Return']],
            on='Date', suffixes=('_stock', '_index')
        ).dropna()
        if len(aligned) >= 2:
            beta = calculate_beta(
                aligned['Daily_Return_stock'],
                aligned['Daily_Return_index']
            )

    return {
        'Annualized Volatility': float(vol),
        'Sharpe Ratio':          float(sharpe),
        'Sortino Ratio':         float(sortino),
        'Max Drawdown':          float(max_dd),
        'VaR (95%)':             float(var_95),
        'Beta':                  float(beta),
    }
