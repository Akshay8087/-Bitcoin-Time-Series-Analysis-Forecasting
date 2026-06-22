import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv
from plotly.subplots import make_subplots
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

APP_DIR = Path(__file__).parent
DEFAULT_DATA = APP_DIR / "data" / "bitcoin_price.csv"
ENV_PATH = APP_DIR / ".env"
load_dotenv(ENV_PATH)

DEFAULT_TICKER = os.getenv("YAHOO_TICKER", "BTC-USD")
DEFAULT_PERIOD = os.getenv("YAHOO_PERIOD", "5y")
DEFAULT_INTERVAL = os.getenv("YAHOO_INTERVAL", "1d")
ALLOW_FALLBACK = os.getenv("ALLOW_CSV_FALLBACK", "true").lower() == "true"

st.set_page_config(
    page_title="Live Bitcoin Price Forecasting AI",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main { background: radial-gradient(circle at top left, rgba(247,147,26,.14), transparent 31%), #080d1a; }
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
    .hero {
        padding: 30px 32px; border-radius: 26px;
        background: linear-gradient(135deg, rgba(247,147,26,.20), rgba(37,99,235,.13), rgba(16,185,129,.08));
        border: 1px solid rgba(255,255,255,.11);
        box-shadow: 0 22px 70px rgba(0,0,0,.30);
    }
    .hero h1 { font-size: 2.55rem; margin-bottom: 0.25rem; }
    .hero p { color: #cbd5e1; font-size: 1.05rem; margin: 0; }
    .metric-card {
        padding: 18px 20px; border-radius: 18px; background: rgba(17,24,39,.80);
        border: 1px solid rgba(255,255,255,.08); height: 100%;
    }
    .metric-label { color: #94a3b8; font-size: .82rem; text-transform: uppercase; letter-spacing: .08em; }
    .metric-value { color: #f8fafc; font-size: 1.62rem; font-weight: 800; margin-top: .25rem; }
    .metric-help { color: #cbd5e1; font-size: .9rem; margin-top: .3rem; }
    .status-box {
        padding: 13px 16px; border-radius: 16px; background: rgba(16,185,129,.10);
        border: 1px solid rgba(16,185,129,.30); color: #bbf7d0;
    }
    .warning-box {
        padding: 14px 16px; border-radius: 16px; background: rgba(247,147,26,.11);
        border: 1px solid rgba(247,147,26,.30); color: #fed7aa;
    }
    div[data-testid="stDataFrame"] { border-radius: 16px; overflow: hidden; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@dataclass
class ModelOutput:
    name: str
    prediction: pd.Series
    mae: float
    mape: float
    rmse: float


def money(x: float) -> str:
    return f"${x:,.2f}"


def clean_yahoo_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance and CSV data into Date-indexed OHLCV format."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date")
    else:
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()]

    rename_map = {"Adj Close": "Adj Close", "Close": "Close", "Open": "Open", "High": "High", "Low": "Low", "Volume": "Volume"}
    available = [c for c in rename_map if c in df.columns]
    if not available:
        raise ValueError("Data must contain Yahoo Finance columns such as Close, Open, High, Low, Volume.")

    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    price_col = "Adj Close" if "Adj Close" in df.columns and df["Adj Close"].notna().any() else "Close"
    if price_col not in df.columns:
        raise ValueError("Data must contain Close or Adj Close column.")

    df = df.dropna(subset=[price_col]).sort_index()
    df = df.drop_duplicates(keep="last")
    df["Price"] = df[price_col].astype(float)
    return df


@st.cache_data(ttl=900, show_spinner=False)
def fetch_live_yahoo_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    raw = yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    if raw is None or raw.empty:
        raise ValueError(f"No data returned from Yahoo Finance for ticker {ticker}.")
    return clean_yahoo_columns(raw)


@st.cache_data(show_spinner=False)
def load_csv_data(uploaded_file=None) -> pd.DataFrame:
    if uploaded_file is not None:
        raw = pd.read_csv(uploaded_file)
    else:
        raw = pd.read_csv(DEFAULT_DATA)
    return clean_yahoo_columns(raw)


def load_price_data(source: str, ticker: str, period: str, interval: str, uploaded_file=None) -> Tuple[pd.DataFrame, str]:
    if source == "Yahoo Finance Live":
        try:
            df = fetch_live_yahoo_data(ticker, period, interval)
            return df, f"Live Yahoo Finance: {ticker}, {period}, {interval}"
        except Exception as exc:
            if not ALLOW_FALLBACK:
                raise
            fallback = load_csv_data(uploaded_file)
            st.warning(f"Yahoo Finance fetch failed, so app used CSV fallback. Reason: {exc}")
            return fallback, "CSV fallback after Yahoo Finance error"

    return load_csv_data(uploaded_file), "Uploaded CSV" if uploaded_file is not None else "Local CSV dataset"


def make_features(series: pd.Series) -> pd.DataFrame:
    df_ml = pd.DataFrame({"price": series.astype(float)})
    for lag in [1, 2, 3, 5, 7, 14, 21, 30]:
        df_ml[f"lag_{lag}"] = series.shift(lag)
    for win in [7, 14, 30, 60]:
        df_ml[f"ma_{win}"] = series.rolling(win).mean()
        df_ml[f"std_{win}"] = series.rolling(win).std()
    df_ml["momentum_7"] = series / series.shift(7) - 1
    df_ml["momentum_30"] = series / series.shift(30) - 1
    df_ml["day_of_week"] = pd.to_datetime(series.index).dayofweek
    df_ml["month"] = pd.to_datetime(series.index).month
    return df_ml.dropna()


def metrics(actual: np.ndarray, pred: np.ndarray) -> Tuple[float, float, float]:
    mae = mean_absolute_error(actual, pred)
    mape = mean_absolute_percentage_error(actual, pred) * 100
    rmse = np.sqrt(mean_squared_error(actual, pred))
    return mae, mape, rmse


@st.cache_resource(show_spinner=True)
def train_models(df: pd.DataFrame, train_ratio: float = 0.85) -> Tuple[Dict[str, ModelOutput], dict]:
    price = df["Price"].asfreq("D").interpolate("time").dropna()
    if len(price) < 150:
        raise ValueError("Need at least 150 daily rows for reliable training. Use a longer Yahoo period like 2y or 5y.")

    log_price = np.log(price)
    split = int(len(price) * train_ratio)
    train, test = price.iloc[:split], price.iloc[split:]
    log_train = log_price.iloc[:split]
    results: Dict[str, ModelOutput] = {}

    arima_model = ARIMA(log_train, order=(1, 1, 1)).fit()
    arima_pred = np.exp(arima_model.forecast(steps=len(test)))
    arima_pred.index = test.index
    mae, mape, rmse = metrics(test.values, arima_pred.values)
    results["ARIMA(1,1,1)"] = ModelOutput("ARIMA(1,1,1)", arima_pred, mae, mape, rmse)

    hw_model = ExponentialSmoothing(train, trend="add", seasonal=None, damped_trend=True).fit(optimized=True)
    hw_pred = hw_model.forecast(steps=len(test))
    hw_pred.index = test.index
    mae, mape, rmse = metrics(test.values, hw_pred.values)
    results["Holt-Winters"] = ModelOutput("Holt-Winters", hw_pred, mae, mape, rmse)

    full_feat = make_features(log_price)
    full_feat["target"] = full_feat["price"].shift(-1)
    full_feat = full_feat.dropna()
    X = full_feat.drop(["price", "target"], axis=1)
    y = full_feat["target"]
    X_train = X[X.index < test.index[0]]
    y_train = y[y.index < test.index[0]]
    X_test = X[X.index >= test.index[0]]
    y_test = y[y.index >= test.index[0]]
    actual_test = np.exp(y_test.values)

    rf = RandomForestRegressor(n_estimators=260, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = np.exp(rf.predict(X_test))
    mae, mape, rmse = metrics(actual_test, rf_pred)
    results["Random Forest"] = ModelOutput("Random Forest", pd.Series(rf_pred, index=X_test.index), mae, mape, rmse)

    gb = GradientBoostingRegressor(n_estimators=360, max_depth=5, learning_rate=0.045, subsample=0.85, random_state=42)
    gb.fit(X_train, y_train)
    gb_pred = np.exp(gb.predict(X_test))
    mae, mape, rmse = metrics(actual_test, gb_pred)
    results["Gradient Boosting"] = ModelOutput("Gradient Boosting", pd.Series(gb_pred, index=X_test.index), mae, mape, rmse)

    artifacts = {
        "price": price,
        "log_price": log_price,
        "train": train,
        "test": test,
        "rf_model": rf,
        "gb_model": gb,
        "feature_columns": list(X.columns),
        "train_ratio": train_ratio,
    }
    return results, artifacts


def recursive_ml_forecast(model, log_history: pd.Series, days: int, feature_columns: list) -> pd.Series:
    history = log_history.copy()
    preds = []
    future_dates = pd.date_range(history.index[-1] + pd.Timedelta(days=1), periods=days, freq="D")
    for next_date in future_dates:
        feat_df = make_features(history)
        row = feat_df.iloc[[-1]].drop(columns=["price"])
        row["day_of_week"] = next_date.dayofweek
        row["month"] = next_date.month
        row = row.reindex(columns=feature_columns)
        pred_log = float(model.predict(row)[0])
        history.loc[next_date] = pred_log
        preds.append(np.exp(pred_log))
    return pd.Series(preds, index=future_dates, name="Forecast")


def forecast_future(model_name: str, artifacts: dict, days: int) -> pd.Series:
    price = artifacts["price"]
    log_price = artifacts["log_price"]
    future_dates = pd.date_range(price.index[-1] + pd.Timedelta(days=1), periods=days, freq="D")

    if model_name == "ARIMA(1,1,1)":
        model = ARIMA(log_price, order=(1, 1, 1)).fit()
        pred = np.exp(model.forecast(steps=days))
        return pd.Series(pred.values, index=future_dates, name="Forecast")

    if model_name == "Holt-Winters":
        model = ExponentialSmoothing(price, trend="add", seasonal=None, damped_trend=True).fit(optimized=True)
        pred = model.forecast(steps=days)
        return pd.Series(pred.values, index=future_dates, name="Forecast")

    if model_name == "Random Forest":
        return recursive_ml_forecast(artifacts["rf_model"], log_price, days, artifacts["feature_columns"])

    return recursive_ml_forecast(artifacts["gb_model"], log_price, days, artifacts["feature_columns"])


def price_chart(price: pd.Series, forecast: pd.Series, title: str):
    fig = go.Figure()
    recent = price.tail(900)
    fig.add_trace(go.Scatter(x=recent.index, y=recent.values, mode="lines", name="Live / Historical BTC Price", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, mode="lines+markers", name="Forecast", line=dict(width=3, dash="dash")))
    fig.update_layout(title=title, height=535, template="plotly_dark", hovermode="x unified", margin=dict(l=20, r=20, t=60, b=20), yaxis_title="USD Price", xaxis_title="Date")
    return fig


def evaluation_chart(results: Dict[str, ModelOutput], test: pd.Series):
    fig = make_subplots(rows=2, cols=2, subplot_titles=list(results.keys()))
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    for (name, out), (r, c) in zip(results.items(), positions):
        n = min(180, len(out.prediction))
        pred = out.prediction.tail(n)
        actual = test.loc[pred.index.min():pred.index.max()].reindex(pred.index).interpolate()
        fig.add_trace(go.Scatter(x=pred.index, y=actual.values, mode="lines", name=f"Actual {name}", showlegend=False), row=r, col=c)
        fig.add_trace(go.Scatter(x=pred.index, y=pred.values, mode="lines", name=name, showlegend=False, line=dict(dash="dash")), row=r, col=c)
    fig.update_layout(height=720, template="plotly_dark", margin=dict(l=20, r=20, t=60, b=20))
    return fig


def leaderboard_df(results: Dict[str, ModelOutput]) -> pd.DataFrame:
    rows = []
    for name, out in results.items():
        rows.append({"Model": name, "MAPE %": out.mape, "MAE USD": out.mae, "RMSE USD": out.rmse})
    return pd.DataFrame(rows).sort_values("MAPE %").reset_index(drop=True)


st.markdown(
    """
    <div class="hero">
        <h1>₿ Live Bitcoin Price Forecasting AI</h1>
        <p>Fetches fresh BTC-USD data from Yahoo Finance, trains forecasting models, and predicts future Bitcoin price with an interactive dashboard.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("⚙️ Live Forecast Controls")
    data_source = st.radio("Data source", ["Yahoo Finance Live", "CSV / Upload"], index=0)
    ticker = st.text_input("Yahoo ticker", value=DEFAULT_TICKER, help="BTC-USD is Bitcoin priced in USD on Yahoo Finance.")
    period = st.selectbox("Yahoo history period", ["6mo", "1y", "2y", "5y", "10y", "max"], index=["6mo", "1y", "2y", "5y", "10y", "max"].index(DEFAULT_PERIOD) if DEFAULT_PERIOD in ["6mo", "1y", "2y", "5y", "10y", "max"] else 3)
    interval = st.selectbox("Yahoo interval", ["1d", "1wk", "1mo"], index=["1d", "1wk", "1mo"].index(DEFAULT_INTERVAL) if DEFAULT_INTERVAL in ["1d", "1wk", "1mo"] else 0)
    uploaded = st.file_uploader("Optional CSV fallback / upload", type=["csv"])
    if st.button("🔄 Refresh Yahoo Data"):
        fetch_live_yahoo_data.clear()
        train_models.clear()
        st.rerun()

    st.markdown("---")
    horizon = st.slider("Forecast horizon", min_value=1, max_value=90, value=30, step=1)
    train_ratio = st.slider("Training data ratio", min_value=0.70, max_value=0.95, value=0.85, step=0.01)
    selected_model = st.selectbox("Prediction model", ["Best by MAPE", "ARIMA(1,1,1)", "Holt-Winters", "Random Forest", "Gradient Boosting"])
    st.caption("For live predictions, use 1d interval and 2y/5y history. Longer horizons become less reliable.")

try:
    df, source_label = load_price_data(data_source, ticker.strip(), period, interval, uploaded)
    with st.spinner("Fetching data and training forecasting models..."):
        results, artifacts = train_models(df, train_ratio=train_ratio)

    board = leaderboard_df(results)
    best_model = board.iloc[0]["Model"]
    model_to_use = best_model if selected_model == "Best by MAPE" else selected_model
    forecast = forecast_future(model_to_use, artifacts, horizon)
    price = artifacts["price"]
    last_price = float(price.iloc[-1])
    final_pred = float(forecast.iloc[-1])
    change_pct = (final_pred / last_price - 1) * 100

    st.markdown(f"<div class='status-box'>Data source: <b>{source_label}</b> · Latest data date: <b>{price.index[-1].date()}</b> · Cached live data refreshes every 15 minutes.</div>", unsafe_allow_html=True)
    st.write("")

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("Latest BTC Price", money(last_price), str(price.index[-1].date())),
        (f"{horizon}-Day Forecast", money(final_pred), f"Model: {model_to_use}"),
        ("Expected Change", f"{change_pct:+.2f}%", "Forecast vs latest price"),
        ("Best Test MAPE", f"{board.iloc[0]['MAPE %']:.2f}%", best_model),
    ]
    for col, (label, value, help_text) in zip([c1, c2, c3, c4], cards):
        with col:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-help">{help_text}</div>
                </div>
                """, unsafe_allow_html=True)

    st.plotly_chart(price_chart(price, forecast, f"BTC Forecast — {model_to_use}"), use_container_width=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Forecast Table", "🏆 Model Leaderboard", "🧪 Test Evaluation", "📊 Live Data Explorer"])

    with tab1:
        forecast_table = pd.DataFrame({"Date": forecast.index.date, "Predicted BTC Price": forecast.values, "Change vs Latest %": (forecast.values / last_price - 1) * 100})
        st.dataframe(forecast_table.style.format({"Predicted BTC Price": "${:,.2f}", "Change vs Latest %": "{:+.2f}%"}), use_container_width=True, hide_index=True)
        csv = forecast_table.to_csv(index=False).encode("utf-8")
        st.download_button("Download Forecast CSV", csv, "bitcoin_live_forecast.csv", "text/csv")

    with tab2:
        st.dataframe(board.style.format({"MAPE %": "{:.2f}", "MAE USD": "${:,.0f}", "RMSE USD": "${:,.0f}"}), use_container_width=True, hide_index=True)
        st.markdown("<div class='warning-box'>Lower MAPE is better. Crypto prices are extremely volatile, so use this as an analytical estimate only — not financial advice.</div>", unsafe_allow_html=True)

    with tab3:
        st.plotly_chart(evaluation_chart(results, artifacts["test"]), use_container_width=True)

    with tab4:
        left, right = st.columns([1, 1])
        with left:
            st.subheader("Dataset Summary")
            st.write(f"Rows: **{len(df):,}**")
            st.write(f"Date range: **{df.index.min().date()} → {df.index.max().date()}**")
            st.write(f"Price range: **{money(df['Price'].min())} – {money(df['Price'].max())}**")
            st.write(f"Average volume: **{df['Volume'].mean():,.0f}**" if "Volume" in df.columns else "Volume column not available")
        with right:
            st.subheader("Recent Live / Historical Data")
            display_cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Price"] if c in df.columns]
            st.dataframe(df[display_cols].tail(12), use_container_width=True)

except Exception as exc:
    st.error("App could not fetch/process the data.")
    st.exception(exc)
