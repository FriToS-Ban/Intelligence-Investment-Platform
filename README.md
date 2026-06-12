# NIFTY-50 AI Investment Intelligence Platform

An AI-powered decision support platform utilizing historical NIFTY-50 stock market data (Jan 2000 – Apr 2021).

## Project Structure
```
nifty50-platform/
├── data/
│   ├── raw/                # Raw stock datasets & metadata
│   └── processed/          # Preprocessed data, predictions, and metrics
├── src/
│   ├── features/           # Feature indicators & anomaly detection
│   ├── models/             # Predictor engines (XGBoost, ARIMA, MLP, custom Prophet)
│   ├── portfolio/          # MVO allocations
│   ├── risk/               # Risk assessment calculations
│   └── app/                # Streamlit Web App / Dashboard
├── reports/                # Technical report (PDF)
├── README.md
└── requirements.txt
```

## Setup & Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Organize datasets:**
   Ensure the Kaggle stock dataset files (e.g. `ADANIPORTS.csv`, `ASIANPAINT.csv`, `stock_metadata.csv`, `NIFTY50_all.csv`) are placed in the `data/raw/` directory.

## Running the Pipelines & Application

1. **Run Feature Pipeline:**
   Computes all moving averages, MACD, RSI, Bollinger Bands, returns, volatility, and momentum indicators.
   ```bash
   python src/features/build_features.py
   ```

2. **Train & Evaluate Models:**
   Trains all 4 models (XGBoost, ARIMA, MLP, custom ProphetRegressor) using a chronological split (Train <= 2018, Val 2019, Test >= 2020).
   ```bash
   python src/models/train.py --seed 42
   ```

3. **Generate Technical Report PDF:**
   ```bash
   python reports/generate_report.py
   ```

4. **Launch Streamlit Dashboard:**
   ```bash
   streamlit run src/app/main.py
   ```
