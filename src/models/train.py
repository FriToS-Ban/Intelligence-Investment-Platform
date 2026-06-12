import os
import sys
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                              r2_score, accuracy_score, f1_score)
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor, MLPClassifier
from xgboost import XGBRegressor, XGBClassifier
from statsmodels.tsa.arima.model import ARIMA

# ── Project root (two levels up from src/models/train.py) ─────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))

np.random.seed(42)

# ──────────────────────────────────────────────────────────────────────────────
class CustomProphetRegressor:
    """
    Piecewise linear trend + yearly & weekly seasonality via sinusoidal features.
    Acts as a lightweight Prophet substitute requiring no C++ toolchain.
    """
    def __init__(self):
        self.params     = None
        self.start_date = None

    def _build_X(self, t):
        return np.column_stack([
            np.ones(len(t)),
            t,
            np.sin(2 * np.pi * t / 365.25), np.cos(2 * np.pi * t / 365.25),
            np.sin(2 * np.pi * t / 7.0),    np.cos(2 * np.pi * t / 7.0),
        ])

    def fit(self, dates, y):
        self.start_date = dates.iloc[0]
        t               = (dates - self.start_date).dt.days.values.astype(float)
        X               = self._build_X(t)
        self.params     = np.linalg.lstsq(X, np.asarray(y), rcond=None)[0]

    def predict(self, dates):
        t = (dates - self.start_date).dt.days.values.astype(float)
        return self._build_X(t) @ self.params   # returns numpy array, no index issues


# ──────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    'MA20', 'MA50', 'MA200',
    'EMA12', 'EMA26',
    'MACD', 'MACD_Signal',
    'RSI',
    'BB_High', 'BB_Low',
    'Daily_Return', 'Volatility_30d',
    'Momentum_1m', 'Momentum_3m', 'Momentum_6m', 'Momentum_12m',
]

def get_stationary_features(df):
    """
    Transform price-level features to stationary percentage differences/ratios
    relative to the 'Close' price, so that models generalize across price scales.
    """
    X_df = pd.DataFrame(index=df.index)
    close = df['Close']
    
    # Deviations from Close
    X_df['MA20_pct'] = (df['MA20'] / close) - 1.0
    X_df['MA50_pct'] = (df['MA50'] / close) - 1.0
    X_df['MA200_pct'] = (df['MA200'] / close) - 1.0
    X_df['EMA12_pct'] = (df['EMA12'] / close) - 1.0
    X_df['EMA26_pct'] = (df['EMA26'] / close) - 1.0
    X_df['BB_High_pct'] = (df['BB_High'] / close) - 1.0
    X_df['BB_Low_pct'] = (df['BB_Low'] / close) - 1.0
    X_df['MACD_pct'] = df['MACD'] / close
    X_df['MACD_Signal_pct'] = df['MACD_Signal'] / close
    
    # Naturally stationary indicators
    X_df['RSI'] = df['RSI']
    X_df['Daily_Return'] = df['Daily_Return']
    X_df['Volatility_30d'] = df['Volatility_30d']
    X_df['Momentum_1m'] = df['Momentum_1m']
    X_df['Momentum_3m'] = df['Momentum_3m']
    X_df['Momentum_6m'] = df['Momentum_6m']
    X_df['Momentum_12m'] = df['Momentum_12m']
    
    return X_df.values

def train_and_evaluate_stock(stock_df, symbol):
    stock_df = stock_df.sort_values('Date').reset_index(drop=True)

    target_cols = ['Target_5d_Return', 'Target_5d_Direction', 'Target_5d_Close']
    df_clean    = stock_df.dropna(subset=FEATURE_COLS + target_cols).reset_index(drop=True)

    if len(df_clean) < 500:
        print(f"  Skipping {symbol}: insufficient data ({len(df_clean)} rows).")
        return None

    # ── Chronological split (no leakage) ──────────────────────────────────────
    train_df = df_clean[df_clean['Date'] <= '2018-12-31'].reset_index(drop=True)
    val_df   = df_clean[(df_clean['Date'] >= '2019-01-01') &
                        (df_clean['Date'] <= '2019-12-31')].reset_index(drop=True)
    test_df  = df_clean[df_clean['Date'] >= '2020-01-01'].reset_index(drop=True)

    if len(train_df) < 200 or len(test_df) < 20:
        print(f"  Skipping {symbol}: train={len(train_df)} or test={len(test_df)} too small.")
        return None

    # Get stationary features for robust cross-regime forecasting
    X_train = get_stationary_features(train_df)
    X_test  = get_stationary_features(test_df)

    y_train_ret = train_df['Target_5d_Return'].values
    y_test_ret  = test_df['Target_5d_Return'].values
    y_train_dir = train_df['Target_5d_Direction'].values
    y_test_dir  = test_df['Target_5d_Direction'].values

    # ── 1. XGBoost with high regularization to combat noise ──────────────────
    xgb_reg = XGBRegressor(
        random_state=42, n_estimators=50, max_depth=2,
        learning_rate=0.01, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=10, reg_alpha=10.0, reg_lambda=10.0,
    )
    xgb_reg.fit(X_train, y_train_ret)
    pred_xgb_ret = xgb_reg.predict(X_test)

    xgb_cls = XGBClassifier(
        random_state=42, n_estimators=50, max_depth=2,
        learning_rate=0.01, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=10, reg_alpha=10.0, reg_lambda=10.0,
        eval_metric='logloss',
    )
    xgb_cls.fit(X_train, y_train_dir)
    pred_xgb_dir = xgb_cls.predict(X_test)

    # ── 2. MLP with scaled stationary features and weight decay ─────────────
    scaler      = StandardScaler()
    X_train_sc  = scaler.fit_transform(X_train)   # fit only on train!
    X_test_sc   = scaler.transform(X_test)         # transform test with same params

    mlp_reg = MLPRegressor(
        random_state=42, hidden_layer_sizes=(16, 8),
        max_iter=300, early_stopping=True, validation_fraction=0.1,
        learning_rate_init=0.005, alpha=0.5,
    )
    mlp_reg.fit(X_train_sc, y_train_ret)
    pred_mlp_ret = mlp_reg.predict(X_test_sc)

    mlp_cls = MLPClassifier(
        random_state=42, hidden_layer_sizes=(16, 8),
        max_iter=300, early_stopping=True, validation_fraction=0.1,
        learning_rate_init=0.005, alpha=0.5,
    )
    mlp_cls.fit(X_train_sc, y_train_dir)
    pred_mlp_dir = mlp_cls.predict(X_test_sc)

    # ── 3. Custom Prophet (trend + seasonality on Close prices) ───────────────
    prophet = CustomProphetRegressor()
    prophet.fit(train_df['Date'], train_df['Close'].values)
    pred_prophet_close = prophet.predict(test_df['Date'])   # pure numpy array

    # Use .values everywhere to avoid pandas index misalignment!
    test_close = test_df['Close'].values
    pred_prophet_ret = (pred_prophet_close / (test_close + 1e-10)) - 1

    # ── 4. ARIMA baseline ──────────────────────────────────────────────────────
    try:
        arima_model = ARIMA(train_df['Close'].values, order=(5, 1, 0))
        arima_fit   = arima_model.fit()
        forecast_close = arima_fit.forecast(steps=len(test_df))   # numpy array
        pred_arima_ret = (np.asarray(forecast_close) / (test_close + 1e-10)) - 1
    except Exception as e:
        print(f"  ARIMA failed for {symbol}: {e} — using zero forecast.")
        pred_arima_ret = np.zeros(len(test_df))

    # ── Build results DataFrame ────────────────────────────────────────────────
    results_df = pd.DataFrame({
        'Date':              test_df['Date'].values,
        'Symbol':            symbol,
        'Close':             test_close,
        'Actual_Return':     y_test_ret,
        'Actual_Direction':  y_test_dir,
        'XGB_Ret_Pred':      pred_xgb_ret,
        'XGB_Dir_Pred':      pred_xgb_dir,
        'MLP_Ret_Pred':      pred_mlp_ret,
        'MLP_Dir_Pred':      pred_mlp_dir,
        'Prophet_Ret_Pred':  pred_prophet_ret,
        'ARIMA_Ret_Pred':    pred_arima_ret,
    })

    # ── Metrics ────────────────────────────────────────────────────────────────
    metrics = {}
    for name, pred in [('XGB',    pred_xgb_ret),
                        ('MLP',    pred_mlp_ret),
                        ('Prophet',pred_prophet_ret),
                        ('ARIMA',  pred_arima_ret)]:
        metrics[f'{name}_MAE']  = mean_absolute_error(y_test_ret, pred)
        metrics[f'{name}_RMSE'] = np.sqrt(mean_squared_error(y_test_ret, pred))
        metrics[f'{name}_R2']   = r2_score(y_test_ret, pred)

    for name, pred in [('XGB', pred_xgb_dir), ('MLP', pred_mlp_dir)]:
        metrics[f'{name}_Accuracy'] = accuracy_score(y_test_dir, pred)
        metrics[f'{name}_F1']       = f1_score(y_test_dir, pred, zero_division=0)

    return results_df, metrics, (xgb_reg, xgb_cls, mlp_reg, mlp_cls, prophet, scaler)


# ──────────────────────────────────────────────────────────────────────────────
def run_training_pipeline(processed_dir):
    consolidated_path = os.path.join(processed_dir, 'consolidated_stocks.csv')
    if not os.path.exists(consolidated_path):
        print(f"ERROR: {consolidated_path} not found. Run build_features.py first.")
        return

    df = pd.read_csv(consolidated_path)
    df['Date'] = pd.to_datetime(df['Date'])

    symbols     = df['Symbol'].unique()
    print(f"Training models for {len(symbols)} stocks …")

    models_dir = os.path.join(processed_dir, '..', 'models')
    os.makedirs(models_dir, exist_ok=True)

    all_results   = []
    summary_mets  = []
    SAVE_SYMBOLS  = {'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ASIANPAINT',
                     'ICICIBANK', 'HINDUNILVR', 'AXISBANK'}

    for symbol in symbols:
        stock_df = df[df['Symbol'] == symbol]
        print(f"Training {symbol} …")
        res = train_and_evaluate_stock(stock_df, symbol)
        if res is None:
            continue
        res_df, mets, models = res
        all_results.append(res_df)
        mets['Symbol'] = symbol
        summary_mets.append(mets)

        if symbol in SAVE_SYMBOLS:
            with open(os.path.join(models_dir, f"{symbol}_models.pkl"), 'wb') as f:
                pickle.dump(models, f)

    if all_results:
        pd.concat(all_results, ignore_index=True).to_csv(
            os.path.join(processed_dir, 'predictions.csv'), index=False)
        pd.DataFrame(summary_mets).to_csv(
            os.path.join(processed_dir, 'metrics.csv'), index=False)
        print("Training pipeline completed — predictions.csv and metrics.csv saved.")


if __name__ == "__main__":
    run_training_pipeline(os.path.join(PROJECT_ROOT, "data", "processed"))
