import numpy as np
import pandas as pd
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")


def compute_portfolio_metrics(weights, expected_returns, cov_matrix, rf_rate=0.06):
    port_return = float(np.sum(expected_returns * weights) * 252)
    port_vol    = float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights))))
    sharpe      = (port_return - rf_rate) / (port_vol + 1e-10)
    return port_return, port_vol, sharpe


def optimize_portfolio_scipy(expected_returns, cov_matrix,
                              objective='sharpe', target_return=None, rf_rate=0.06):
    num_assets = len(expected_returns)
    if num_assets == 0:
        raise ValueError("No assets available — portfolio matrix is empty.")

    # Dynamic per-stock cap so constraints remain feasible even with few assets
    max_weight = min(1.0, max(0.4, 2.0 / num_assets))
    init_guess = np.array([1.0 / num_assets] * num_assets)
    bounds     = tuple((0.0, max_weight) for _ in range(num_assets))

    constraints = [{'type': 'eq', 'fun': lambda x: float(np.sum(x)) - 1.0}]
    if target_return is not None:
        constraints.append({
            'type': 'eq',
            'fun': lambda x: float(np.sum(expected_returns * x)) * 252 - target_return
        })

    if objective == 'sharpe':
        def obj_fn(w):
            return -compute_portfolio_metrics(w, expected_returns, cov_matrix, rf_rate)[2]
    elif objective == 'volatility':
        def obj_fn(w):
            return compute_portfolio_metrics(w, expected_returns, cov_matrix, rf_rate)[1]
    else:   # maximize return
        def obj_fn(w):
            return -compute_portfolio_metrics(w, expected_returns, cov_matrix, rf_rate)[0]

    res = minimize(obj_fn, init_guess, method='SLSQP',
                   bounds=bounds, constraints=constraints,
                   options={'maxiter': 1000, 'ftol': 1e-9})

    weights = res.x
    # Safety: clip negatives and renormalise
    weights = np.clip(weights, 0, None)
    total   = weights.sum()
    if total < 1e-10:
        weights = init_guess                    # fall back to equal weight
    else:
        weights /= total
    return weights


def _prepare_returns(stock_returns_df, min_coverage=0.3):
    """
    Build a (date × symbol) daily-return matrix safe for MVO.
    Key insight: NEVER use dropna() across all columns — stocks have different
    listing dates so requiring ALL columns non-NaN yields 0 rows.
    Instead keep rows where ≥70% of stocks have data, then fill remaining gaps.
    """
    pivot = stock_returns_df.pivot_table(
        index='Date', columns='Symbol', values='Daily_Return', aggfunc='mean'
    )

    # 1. Drop columns that are entirely NaN (empty tickers like INFRATEL)
    pivot = pivot.dropna(axis=1, how='all')
    if pivot.empty or pivot.shape[1] == 0:
        return pd.DataFrame()

    # 2. Forward-fill up to 10 days (handles holidays/weekends in sparse data)
    pivot = pivot.ffill(limit=10)

    # 3. Keep rows where ≥70% of stocks have data  ← THIS is the key fix
    #    (avoids the "all stocks must be non-NaN" trap that wipes all rows)
    row_coverage = pivot.notna().mean(axis=1)
    pivot = pivot[row_coverage >= 0.70]
    if pivot.empty:
        return pd.DataFrame()

    # 4. Keep only stocks with enough rows of data in this filtered set
    n_rows = len(pivot)
    valid  = pivot.count() >= int(n_rows * min_coverage)
    pivot  = pivot.loc[:, valid]
    if pivot.shape[1] == 0:
        return pd.DataFrame()

    # 5. Fill any remaining NaN cells with the column mean (last resort)
    col_means = pivot.mean()
    pivot = pivot.fillna(col_means)

    # 6. Drop any rows that are still fully NaN (edge case safety)
    pivot = pivot.dropna(how='all')

    return pivot


def get_portfolio_allocations(stock_returns_df, profile='balanced', rf_rate=0.06):
    daily_returns = _prepare_returns(stock_returns_df)

    if daily_returns.empty or daily_returns.shape[1] < 2:
        raise ValueError(
            "Not enough valid overlapping stock data for optimisation. "
            f"Got {daily_returns.shape[1] if not daily_returns.empty else 0} assets "
            f"and {len(daily_returns)} common trading days."
        )

    symbols  = daily_returns.columns.tolist()
    exp_ret  = daily_returns.mean().values
    cov_mat  = daily_returns.cov().values

    # Replace any NaN in covariance (constant / short-history columns) with 0
    cov_mat  = np.nan_to_num(cov_mat, nan=0.0)
    exp_ret  = np.nan_to_num(exp_ret, nan=0.0)

    if profile == 'conservative':
        weights = optimize_portfolio_scipy(exp_ret, cov_mat, objective='volatility', rf_rate=rf_rate)
    elif profile == 'aggressive':
        ann_returns = exp_ret * 252
        high_target = float(np.percentile(ann_returns, 75))
        try:
            weights = optimize_portfolio_scipy(exp_ret, cov_mat, objective='sharpe',
                                               target_return=high_target, rf_rate=rf_rate)
        except Exception:
            weights = optimize_portfolio_scipy(exp_ret, cov_mat, objective='return', rf_rate=rf_rate)
    else:   # balanced
        weights = optimize_portfolio_scipy(exp_ret, cov_mat, objective='sharpe', rf_rate=rf_rate)

    port_ret, port_vol, port_sharpe = compute_portfolio_metrics(weights, exp_ret, cov_mat, rf_rate)

    allocation = {symbols[i]: float(weights[i]) for i in range(len(symbols)) if weights[i] > 0.005}
    allocation = dict(sorted(allocation.items(), key=lambda x: x[1], reverse=True))

    return {
        'allocation':      allocation,
        'expected_return': port_ret,
        'volatility':      port_vol,
        'sharpe_ratio':    port_sharpe,
        'num_assets':      len(daily_returns.columns),
        'num_days':        len(daily_returns),
    }


def get_efficient_frontier_points(stock_returns_df, num_points=15, rf_rate=0.06):
    daily_returns = _prepare_returns(stock_returns_df)
    if daily_returns.empty or daily_returns.shape[1] < 2:
        return [], []

    exp_ret = np.nan_to_num(daily_returns.mean().values,  nan=0.0)
    cov_mat = np.nan_to_num(daily_returns.cov().values,   nan=0.0)
    ann_ret = exp_ret * 252
    targets = np.linspace(ann_ret.min(), ann_ret.max(), num_points)

    vols, rets = [], []
    for target in targets:
        try:
            w = optimize_portfolio_scipy(exp_ret, cov_mat, objective='volatility',
                                         target_return=float(target), rf_rate=rf_rate)
            r, v, _ = compute_portfolio_metrics(w, exp_ret, cov_mat, rf_rate)
            vols.append(v)
            rets.append(r)
        except Exception:
            pass

    return vols, rets


def backtest_portfolio(stock_df, allocation, index_df, start_date='2020-01-01'):
    """
    Backtest a static portfolio allocation starting from start_date.
    Compares cumulative return of the portfolio vs equal-weighted market index.
    """
    symbols = list(allocation.keys())
    
    sub_df = stock_df[(stock_df['Symbol'].isin(symbols)) & (stock_df['Date'] >= start_date)]
    if sub_df.empty:
        return pd.DataFrame()
        
    pivot = sub_df.pivot_table(index='Date', columns='Symbol', values='Daily_Return', aggfunc='mean')
    pivot = pivot.ffill().bfill().fillna(0.0)
    
    # Keep only symbols that are present in the pivot columns
    valid_symbols = [s for s in symbols if s in pivot.columns]
    if not valid_symbols:
        return pd.DataFrame()
        
    # Re-normalize weights for valid symbols
    weights = np.array([allocation[s] for s in valid_symbols])
    total_weight = weights.sum()
    if total_weight > 0:
        weights = weights / total_weight
    else:
        weights = np.array([1.0 / len(valid_symbols)] * len(valid_symbols))
        
    pivot = pivot[valid_symbols]
    
    # Calculate daily portfolio returns
    port_daily_ret = pivot.values @ weights
    
    backtest_df = pd.DataFrame(index=pivot.index)
    backtest_df['Portfolio_Return'] = port_daily_ret
    
    # Get index returns
    idx_sub = index_df[index_df['Date'] >= start_date].set_index('Date')
    backtest_df = backtest_df.join(idx_sub['Daily_Return'].rename('Market_Return'), how='left').fillna(0.0)
    
    # Compute cumulative returns
    backtest_df['Portfolio_Cumulative'] = (1 + backtest_df['Portfolio_Return']).cumprod() - 1
    backtest_df['Market_Cumulative'] = (1 + backtest_df['Market_Return']).cumprod() - 1
    
    return backtest_df.reset_index()

