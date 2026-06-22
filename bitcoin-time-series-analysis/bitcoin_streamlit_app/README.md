# Live Bitcoin Price Forecasting AI — Streamlit App

A professional Streamlit dashboard that fetches fresh Bitcoin data from Yahoo Finance and predicts future BTC price using multiple forecasting models.

## Live data source

The app uses `yfinance` to fetch Yahoo Finance data. Default ticker is:

```text
BTC-USD
```

You can change this from the sidebar or inside `.env`.

## Models included

- ARIMA(1,1,1)
- Holt-Winters damped trend
- Random Forest Regressor
- Gradient Boosting Regressor

## Features

- Fetch live BTC-USD data from Yahoo Finance
- Optional CSV upload/fallback
- Forecast BTC price for 1 to 90 days
- Auto-select best model by test MAPE
- Interactive Plotly charts
- Model leaderboard with MAPE, MAE, and RMSE
- Forecast CSV download
- 15-minute cache for Yahoo Finance data
- `.env` configuration

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## .env setup

Create or edit `.env`:

```text
YAHOO_TICKER=BTC-USD
YAHOO_PERIOD=5y
YAHOO_INTERVAL=1d
ALLOW_CSV_FALLBACK=true
```

Supported common Yahoo periods:

```text
6mo, 1y, 2y, 5y, 10y, max
```

Supported app intervals:

```text
1d, 1wk, 1mo
```

For best model quality, use `1d` interval with `2y`, `5y`, or `10y` period.

## Project structure

```text
bitcoin_streamlit_app/
├── app.py
├── requirements.txt
├── README.md
├── .env
├── .env.example
├── data/
│   └── bitcoin_price.csv
└── .streamlit/
    └── config.toml
```

## Important disclaimer

This app is for educational and analytical forecasting only. Cryptocurrency prices are highly volatile. Do not use this output as financial advice.
