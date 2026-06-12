import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import sys

# Inject custom path to import modules — resolve to project root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT_DIR)

from src.risk.risk_metrics import get_risk_profile_metrics
from src.portfolio.optimizer import get_portfolio_allocations, get_efficient_frontier_points, backtest_portfolio
from src.features.anomalies import detect_anomalies, get_anomaly_summary
import pickle


st.set_page_config(
    page_title="NIFTY-50 AI Investment Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling — fix: use unsafe_allow_html (not unsafe_allow_name_allowed)
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    h1, h2, h3 { font-family: 'Segoe UI', sans-serif; }
    .metric-card {
        background-color: #1E293B;
        border-radius: 8px;
        padding: 20px;
        border: 1px solid #334155;
        margin-bottom: 15px;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0px 0px;
        padding: 8px 16px;
    }
</style>
""", unsafe_allow_html=True)

# ── Data paths resolved relative to the project root ──────────────────────────
DATA_PROCESSED = os.path.join(ROOT_DIR, "data", "processed")
DATA_RAW       = os.path.join(ROOT_DIR, "data", "raw")

@st.cache_data
def load_processed_data():
    consolidated_path = os.path.join(DATA_PROCESSED, "consolidated_stocks.csv")
    predictions_path  = os.path.join(DATA_PROCESSED, "predictions.csv")
    metrics_path      = os.path.join(DATA_PROCESSED, "metrics.csv")

    df = pd.read_csv(consolidated_path)
    df['Date'] = pd.to_datetime(df['Date'])

    preds = None
    if os.path.exists(predictions_path):
        preds = pd.read_csv(predictions_path)
        preds['Date'] = pd.to_datetime(preds['Date'])

    metrics = pd.read_csv(metrics_path) if os.path.exists(metrics_path) else None

    # Build a synthetic equal-weighted NIFTY-50 market index from the consolidated data.
    # NIFTY50_all.csv contains individual stock histories, NOT a single index price series.
    # We compute the daily median return across all stocks as a proxy for the market.
    pivot_ret = df.pivot_table(index='Date', columns='Symbol', values='Daily_Return', aggfunc='mean')
    # Equal-weight: mean of available returns each day
    market_ret = pivot_ret.mean(axis=1).rename('Daily_Return').reset_index()
    market_ret['Date'] = pd.to_datetime(market_ret['Date'])
    index_df = market_ret  # columns: ['Date', 'Daily_Return']

    return df, preds, metrics, index_df

try:
    df, preds, metrics, index_df = load_processed_data()
except Exception as e:
    st.error(f"❌ Error loading datasets. Have you run the pipeline? Details: {e}")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
st.title("📊 NIFTY-50 AI Investment Intelligence Platform")
st.markdown("*Historical data: Jan 2000 – Apr 2021 | Seed: 42 | No live data*")
st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Stock Explorer",
    "🔮 Predictor Engine",
    "💼 Portfolio Builder",
    "⚠️ Risk Dashboard",
    "🚨 Anomaly Timeline"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: Stock Explorer
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.header("Stock Explorer")
    symbols = sorted(df['Symbol'].unique())
    selected_stock = st.selectbox("Select Stock", symbols, key="explorer_stock")

    stock_data = df[df['Symbol'] == selected_stock].sort_values('Date')

    col1, col2, col3 = st.columns(3)
    latest = stock_data.iloc[-1]

    daily_ret = latest['Daily_Return'] if pd.notna(latest['Daily_Return']) else 0.0
    rsi_val   = latest['RSI'] if pd.notna(latest['RSI']) else float('nan')
    vol_val   = latest['Volatility_30d'] if pd.notna(latest['Volatility_30d']) else float('nan')

    col1.metric("Latest Close",      f"₹{latest['Close']:,.2f}", f"{daily_ret*100:.2f}%")
    col2.metric("RSI (14)",          f"{rsi_val:.2f}" if pd.notna(rsi_val) else "N/A")
    col3.metric("Volatility (30d)",  f"{vol_val*100:.2f}%" if pd.notna(vol_val) else "N/A")

    # Indicator overlay
    indicator = st.multiselect(
        "Overlay Indicators",
        ["MA20", "MA50", "MA200", "Bollinger Bands", "EMA12", "EMA26"],
        default=["MA20", "MA50"]
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=stock_data['Date'], y=stock_data['Close'],
        name='Close Price', line=dict(color='#38BDF8', width=2)
    ))
    for ind in indicator:
        if ind == "Bollinger Bands":
            fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['BB_High'],
                                     name='BB High', line=dict(color='#F43F5E', width=1, dash='dash')))
            fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['BB_Low'],
                                     name='BB Low',  line=dict(color='#10B981', width=1, dash='dash')))
        elif ind in stock_data.columns:
            fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data[ind],
                                     name=ind, line=dict(width=1.5)))
    fig.update_layout(
        title=f"{selected_stock} Price History & Technical Indicators",
        template="plotly_dark", height=500
    )
    st.plotly_chart(fig, use_container_width=True)

    # MACD sub-chart
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['MACD'],
                                  name='MACD', line=dict(color='#FB7185')))
    fig_macd.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['MACD_Signal'],
                                  name='Signal', line=dict(color='#818CF8')))
    macd_hist = stock_data['MACD'] - stock_data['MACD_Signal']
    fig_macd.add_trace(go.Bar(x=stock_data['Date'], y=macd_hist, name='Histogram',
                               marker_color=['#10B981' if v >= 0 else '#F43F5E' for v in macd_hist]))
    fig_macd.update_layout(title="MACD Oscillator", template="plotly_dark", height=250)
    st.plotly_chart(fig_macd, use_container_width=True)

    # RSI chart
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['RSI'],
                                 name='RSI(14)', line=dict(color='#FB923C')))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="#F43F5E", annotation_text="Overbought (70)")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="#10B981", annotation_text="Oversold (30)")
    fig_rsi.update_layout(title="RSI (14)", template="plotly_dark", height=220, yaxis=dict(range=[0, 100]))
    st.plotly_chart(fig_rsi, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: Predictor Engine
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.header("Predictor Engine")
    if preds is None:
        st.warning("Prediction results not found. Run: `python src/models/train.py --seed 42`")
    else:
        pred_symbols = sorted(preds['Symbol'].unique())
        pred_stock   = st.selectbox("Select Stock to Forecast", pred_symbols, key="pred_stock")
        p_df         = preds[preds['Symbol'] == pred_stock].sort_values('Date')

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=p_df['Date'], y=p_df['Actual_Return']*100,
                                      name='Actual Return',          line=dict(color='#10B981')))
        fig_pred.add_trace(go.Scatter(x=p_df['Date'], y=p_df['XGB_Ret_Pred']*100,
                                      name='XGBoost Predictor',      line=dict(color='#F59E0B')))
        fig_pred.add_trace(go.Scatter(x=p_df['Date'], y=p_df['Prophet_Ret_Pred']*100,
                                      name='Prophet Trend Predictor', line=dict(color='#EC4899')))
        fig_pred.add_trace(go.Scatter(x=p_df['Date'], y=p_df['ARIMA_Ret_Pred']*100,
                                      name='ARIMA Baseline',          line=dict(color='#8B5CF6')))
        fig_pred.add_trace(go.Scatter(x=p_df['Date'], y=p_df['MLP_Ret_Pred']*100,
                                      name='Neural Network (MLP)',    line=dict(color='#3B82F6')))
        fig_pred.update_layout(
            title=f"5-Day Forward Return Forecast — {pred_stock} (%)",
            template="plotly_dark", height=450
        )
        st.plotly_chart(fig_pred, use_container_width=True)

        # Metrics table
        st.subheader("Model Performance Evaluation (Test Set: 2020–2021)")

        # Context callout — critical for interpreting stock prediction metrics
        st.info(
            """📊 **How to read these metrics:**  
- **R² (R-squared)** near 0 or negative is *normal and expected* for 5-day forward return prediction.
  Stock returns are close to a random walk — even top quant funds achieve R² of 0.01–0.05.
  A low R² does **not** mean the model is broken; it means the signal-to-noise ratio in markets is inherently low.
- **Directional Accuracy** (>50%) and **low MAE** are the primary metrics that matter in practice.
  A model hitting 52–55% direction correctly is genuinely profitable.
- **The test period (2020–2021) includes the COVID crash** — an extreme regime that degrades all model performance."""
        )

        s_metrics = metrics[metrics['Symbol'] == pred_stock] if metrics is not None else None
        if s_metrics is not None and not s_metrics.empty:
            met_row = s_metrics.iloc[0]

            def fmt(col, pct=False, fallback="N/A"):
                try:
                    v = met_row[col]
                    if pd.isna(v):
                        return fallback
                    return f"{v*100:.2f}%" if pct else f"{v:.4f}"
                except Exception:
                    return fallback

            metrics_df = pd.DataFrame({
                'Metric':          ['MAE ↓', 'RMSE ↓', 'R² (↑, near 0 is OK)', 'Directional Accuracy ↑', 'F1-Score ↑'],
                'XGBoost':         [fmt('XGB_MAE'), fmt('XGB_RMSE'), fmt('XGB_R2'), fmt('XGB_Accuracy', pct=True), fmt('XGB_F1')],
                'Neural Net (MLP)':[fmt('MLP_MAE'), fmt('MLP_RMSE'), fmt('MLP_R2'), fmt('MLP_Accuracy', pct=True), fmt('MLP_F1')],
                'Prophet':         [fmt('Prophet_MAE'), fmt('Prophet_RMSE'), fmt('Prophet_R2'), 'N/A', 'N/A'],
                'ARIMA':           [fmt('ARIMA_MAE'),   fmt('ARIMA_RMSE'),   fmt('ARIMA_R2'),   'N/A', 'N/A'],
            })
            st.table(metrics_df)

            # Visual summary bar chart for key metrics
            accs = {}
            for m in [('XGB', 'XGBoost'), ('MLP', 'Neural Net (MLP)')]:
                try:
                    accs[m[1]] = met_row[f'{m[0]}_Accuracy'] * 100
                except Exception:
                    pass
            if accs:
                fig_acc = go.Figure(go.Bar(
                    x=list(accs.keys()), y=list(accs.values()),
                    marker_color=['#10B981' if v > 50 else '#F43F5E' for v in accs.values()],
                    text=[f"{v:.1f}%" for v in accs.values()], textposition='outside'
                ))
                fig_acc.add_hline(y=50, line_dash='dash', line_color='white',
                                  annotation_text='Baseline (random = 50%)')
                fig_acc.update_layout(
                    title='Directional Accuracy vs Random Baseline',
                    template='plotly_dark', height=280,
                    yaxis=dict(range=[40, 70], title='Accuracy (%)')
                )
                st.plotly_chart(fig_acc, use_container_width=True)
        else:
            st.info("No metrics available for this stock.")

        # Explainability
        st.subheader("Explainability & Recommendation")
        last_pred = p_df.iloc[-1]

        # Calculate cross-sectional percentile rank of this stock relative to the market
        latest_date = preds['Date'].max()
        latest_preds = preds[preds['Date'] == latest_date]

        selected_pred = last_pred['XGB_Ret_Pred']
        all_preds = latest_preds['XGB_Ret_Pred'].dropna().values

        if len(all_preds) > 0:
            rank_val = (all_preds < selected_pred).mean()  # Percentile rank
        else:
            rank_val = 0.5  # default if no other stocks

        # Classify recommendation based on percentile ranking (Alpha)
        if rank_val >= 0.70:
            action = "BUY (Top 30% Market Outperformer)"
            clr = "#10B981"
            explanation = "This stock is expected to significantly outperform the NIFTY-50 index over the next 5 days based on technical momentum and cross-sectional alpha."
        elif rank_val >= 0.30:
            action = "HOLD (Average Market Performer)"
            clr = "#FB923C"
            explanation = "This stock is expected to perform in line with the broader market. Consider holding existing positions."
        else:
            action = "SELL / AVOID (Bottom 30% Market Underperformer)"
            clr = "#F43F5E"
            explanation = "This stock is expected to underperform the NIFTY-50 index over the next 5 days. Consider locking in profits or avoiding new entries."

        direction = "UPWARD" if last_pred.get('XGB_Dir_Pred', 0) == 1 else "DOWNWARD"

        latest_rsi = df[df['Symbol'] == pred_stock]['RSI'].dropna()
        rsi_display = f"{latest_rsi.iloc[-1]:.1f}" if not latest_rsi.empty else "N/A"

        st.markdown(
            f"**AI Recommendation:** <span style='color:{clr}; font-size:20px; font-weight:bold;'>{action}</span>",
            unsafe_allow_html=True
        )
        st.write(explanation)
        st.markdown(f"""
- **XGBoost 5-day return forecast:** `{selected_pred*100:.2f}%` (Market Percentile Rank: **{rank_val*100:.1f}th percentile**)
- **Directional signal:** {direction} price momentum predicted
- **RSI (14):** {rsi_display} — {"oversold signal ✅" if rsi_display != "N/A" and float(rsi_display) < 30 else ("overbought signal ⚠️" if rsi_display != "N/A" and float(rsi_display) > 70 else "neutral range")}
""")

        # Model feature importance display
        models_dir = os.path.join(ROOT_DIR, "data", "models")
        model_symbol = pred_stock if pred_stock in ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ASIANPAINT', 'ICICIBANK', 'HINDUNILVR', 'AXISBANK'] else 'RELIANCE'
        model_path = os.path.join(models_dir, f"{model_symbol}_models.pkl")
        
        feature_importance_df = None
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    xgb_reg, xgb_cls, mlp_reg, mlp_cls, prophet, scaler = pickle.load(f)
                importances = xgb_reg.feature_importances_
                feature_names = [
                    'MA20 Deviation', 'MA50 Deviation', 'MA200 Deviation',
                    'EMA12 Deviation', 'EMA26 Deviation',
                    'Bollinger High Deviation', 'Bollinger Low Deviation',
                    'MACD Ratio', 'MACD Signal Ratio',
                    'RSI (14)', 'Daily Return', 'Volatility (30d)',
                    'Momentum (1m)', 'Momentum (3m)', 'Momentum (6m)', 'Momentum (12m)'
                ]
                feat_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=True)
                feature_importance_df = feat_df
            except Exception as e:
                pass
                
        if feature_importance_df is not None:
            st.markdown("### 🔍 Model Feature Importances (XGBoost)")
            if model_symbol != pred_stock:
                st.caption(f"Note: Specific model for **{pred_stock}** is not saved. Displaying **{model_symbol}** feature importances as a representative market proxy.")
            else:
                st.caption(f"Displaying feature importances for **{pred_stock}**.")
            
            fig_imp = px.bar(
                feature_importance_df,
                y='Feature',
                x='Importance',
                orientation='h',
                template='plotly_dark',
                color='Importance',
                color_continuous_scale='Viridis'
            )
            fig_imp.update_layout(
                height=400,
                margin=dict(l=10, r=10, t=30, b=10),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_imp, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: Portfolio Builder
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.header("Portfolio Builder & Optimization")
    profile = st.selectbox("Select Investor Risk Profile",
                           ["Conservative", "Balanced", "Aggressive"],
                           key="port_profile")

    with st.spinner("Running Markowitz Mean-Variance Optimization..."):
        try:
            port_results = get_portfolio_allocations(df, profile=profile.lower())

            col1, col2, col3 = st.columns(3)
            col1.metric("Expected Annual Return", f"{port_results['expected_return']*100:.2f}%")
            col2.metric("Annual Volatility",      f"{port_results['volatility']*100:.2f}%")
            col3.metric("Sharpe Ratio",            f"{port_results['sharpe_ratio']:.2f}")

            alloc_df = pd.DataFrame(
                list(port_results['allocation'].items()), columns=['Stock', 'Allocation']
            )
            alloc_df['Weight (%)'] = alloc_df['Allocation'].apply(lambda x: f"{x*100:.2f}%")
            alloc_df = alloc_df[alloc_df['Allocation'] > 0.005]

            col_a, col_b = st.columns([1, 1])
            with col_a:
                fig_pie = px.pie(alloc_df, values='Allocation', names='Stock',
                                 title='Portfolio Allocation', template="plotly_dark",
                                 color_discrete_sequence=px.colors.qualitative.Set3)
                st.plotly_chart(fig_pie, use_container_width=True)
            with col_b:
                st.subheader("Stock Allocation Table")
                st.dataframe(alloc_df[['Stock', 'Weight (%)']].reset_index(drop=True), use_container_width=True)

            # Sector Breakdown
            st.subheader("Sector (Industry) Allocation")
            stock_industries = df[['Symbol', 'Industry']].drop_duplicates()
            sector_alloc_df = alloc_df.merge(stock_industries, left_on='Stock', right_on='Symbol', how='left')
            sector_df = sector_alloc_df.groupby('Industry')['Allocation'].sum().reset_index()
            sector_df['Weight (%)'] = sector_df['Allocation'].apply(lambda x: f"{x*100:.2f}%")
            sector_df = sector_df.sort_values('Allocation', ascending=False)
            
            col_sec_a, col_sec_b = st.columns([1, 1])
            with col_sec_a:
                fig_sector = px.pie(sector_df, values='Allocation', names='Industry',
                                    title='Sector Allocation', template="plotly_dark",
                                    color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig_sector, use_container_width=True)
            with col_sec_b:
                st.dataframe(sector_df[['Industry', 'Weight (%)']].reset_index(drop=True), use_container_width=True)

            # Portfolio Backtesting
            st.subheader("Historical Backtesting (Test Period: 2020–2021)")
            with st.spinner("Running portfolio backtest..."):
                backtest_results = backtest_portfolio(df, port_results['allocation'], index_df)
                if not backtest_results.empty:
                    fig_bt = go.Figure()
                    fig_bt.add_trace(go.Scatter(
                        x=backtest_results['Date'], y=backtest_results['Portfolio_Cumulative']*100,
                        name=f'Selected Portfolio ({profile})', line=dict(color='#10B981', width=2)
                    ))
                    fig_bt.add_trace(go.Scatter(
                        x=backtest_results['Date'], y=backtest_results['Market_Cumulative']*100,
                        name='Market Benchmark (Equal-Weighted NIFTY-50)', line=dict(color='#94A3B8', width=1.5, dash='dash')
                    ))
                    fig_bt.update_layout(
                        title=f"Cumulative Returns: Portfolio vs Market (Benchmark)",
                        xaxis_title="Date",
                        yaxis_title="Cumulative Return (%)",
                        template="plotly_dark",
                        height=400
                    )
                    st.plotly_chart(fig_bt, use_container_width=True)
                    
                    # Compute backtest summary metrics
                    port_final_ret = backtest_results['Portfolio_Cumulative'].iloc[-1] * 100
                    mkt_final_ret = backtest_results['Market_Cumulative'].iloc[-1] * 100
                    st.write(f"📈 **Final Cumulative Return (2020–2021):** Portfolio: **{port_final_ret:.2f}%** | Market Benchmark: **{mkt_final_ret:.2f}%**")
                else:
                    st.info("No data available for backtesting.")

        except Exception as e:
            st.error(f"Portfolio optimization failed: {e}")


    # Efficient Frontier
    st.subheader("Efficient Frontier Curve")
    if st.checkbox("Generate Efficient Frontier (may take ~15s)"):
        with st.spinner("Computing efficient frontier..."):
            try:
                vols, rets = get_efficient_frontier_points(df)
                fig_ef = go.Figure()
                fig_ef.add_trace(go.Scatter(
                    x=vols, y=rets, mode='lines+markers',
                    name='Efficient Frontier', line=dict(color='#38BDF8')
                ))
                # Show selected portfolio as a gold star if optimisation succeeded
                try:
                    fig_ef.add_trace(go.Scatter(
                        x=[port_results['volatility']], y=[port_results['expected_return']],
                        mode='markers', name='Selected Portfolio',
                        marker=dict(size=14, color='#F59E0B', symbol='star')
                    ))
                except NameError:
                    pass  # port_results not in scope (optimisation failed above)
                fig_ef.update_layout(
                    title="Markowitz Efficient Frontier",
                    xaxis_title="Annualized Volatility",
                    yaxis_title="Expected Annual Return",
                    template="plotly_dark"
                )
                st.plotly_chart(fig_ef, use_container_width=True)
            except Exception as e:
                st.error(f"Could not compute efficient frontier: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: Risk Dashboard
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.header("Risk Analytics Dashboard")
    risk_stock = st.selectbox("Select Stock for Risk Profile", symbols, key="risk_stock")

    r_stock_df = df[df['Symbol'] == risk_stock]
    try:
        risk_mets = get_risk_profile_metrics(r_stock_df, index_df)

        col1, col2, col3 = st.columns(3)
        col1.metric("Annual Volatility", f"{risk_mets['Annualized Volatility']*100:.2f}%")
        col2.metric("Sharpe Ratio",       f"{risk_mets['Sharpe Ratio']:.2f}")
        col3.metric("Sortino Ratio",      f"{risk_mets['Sortino Ratio']:.2f}")

        col4, col5, col6 = st.columns(3)
        col4.metric("Maximum Drawdown",  f"{risk_mets['Max Drawdown']*100:.2f}%")
        col5.metric("VaR (95% Daily)",   f"{risk_mets['VaR (95%)']*100:.2f}%")
        col6.metric("Beta vs NIFTY-50",  f"{risk_mets['Beta']:.2f}")

        # Drawdown chart
        st.subheader("Drawdown Chart")
        prices = r_stock_df.sort_values('Date')['Close']
        cum_ret = prices / prices.iloc[0]
        running_max = cum_ret.cummax()
        drawdown = (cum_ret - running_max) / running_max

        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=r_stock_df.sort_values('Date')['Date'], y=drawdown*100,
            fill='tozeroy', name='Drawdown',
            line=dict(color='#F43F5E'), fillcolor='rgba(244,63,94,0.2)'
        ))
        fig_dd.update_layout(
            title=f"{risk_stock} Historical Drawdown (%)",
            template="plotly_dark", height=300,
            yaxis_title="Drawdown (%)"
        )
        st.plotly_chart(fig_dd, use_container_width=True)

    except Exception as e:
        st.error(f"Risk calculation failed: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5: Anomaly Timeline
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.header("Market Anomaly Timeline")
    anomaly_stock = st.selectbox("Select Stock to View Anomalies", symbols, key="anomaly_stock")

    a_df = df[df['Symbol'] == anomaly_stock].copy()
    try:
        anomaly_summary = get_anomaly_summary(a_df)

        if anomaly_summary.empty:
            st.info("No anomalies detected for this stock.")
        else:
            st.metric("Total Anomaly Events Detected", len(anomaly_summary))
            st.subheader("Historical Anomalies Log")
            st.dataframe(anomaly_summary, use_container_width=True)

            a_plot_df   = detect_anomalies(a_df)
            anom_points = a_plot_df[a_plot_df['Is_Anomaly'] == True]

            fig_anom = go.Figure()
            fig_anom.add_trace(go.Scatter(
                x=a_plot_df['Date'], y=a_plot_df['Close'],
                name='Close Price', line=dict(color='#475569')
            ))
            fig_anom.add_trace(go.Scatter(
                x=anom_points['Date'], y=anom_points['Close'],
                mode='markers', name='Anomaly Flag',
                marker=dict(color='#F43F5E', size=7, symbol='x')
            ))
            fig_anom.update_layout(
                title=f"Anomaly Occurrences — {anomaly_stock}",
                template="plotly_dark", height=450
            )
            st.plotly_chart(fig_anom, use_container_width=True)

            # Return Z-score chart (detect_anomalies always adds Return_ZScore)
            if 'Return_ZScore' not in a_plot_df.columns:
                a_plot_df['Return_ZScore'] = 0.0
            fig_z = go.Figure()
            fig_z.add_trace(go.Scatter(
                x=a_plot_df['Date'], y=a_plot_df['Return_ZScore'],
                name='Return Z-Score', line=dict(color='#818CF8')
            ))
            fig_z.add_hline(y=3,  line_dash="dash", line_color="#F43F5E", annotation_text="+3σ")
            fig_z.add_hline(y=-3, line_dash="dash", line_color="#F43F5E", annotation_text="-3σ")
            fig_z.update_layout(title="Daily Return Z-Score", template="plotly_dark", height=250)
            st.plotly_chart(fig_z, use_container_width=True)

    except Exception as e:
        st.error(f"Anomaly detection failed: {e}")
