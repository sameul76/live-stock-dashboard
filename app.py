import streamlit as st
import pandas as pd
import numpy as np
import datetime
from sklearn.neighbors import KNeighborsRegressor
from ta.trend import sma_indicator, ema_indicator, macd
from ta.momentum import rsi
from alpaca_trade_api.rest import REST

# Load Alpaca API keys from Streamlit secrets
ALPACA_API_KEY = st.secrets["ALPACA_API_KEY"]
ALPACA_SECRET_KEY = st.secrets["ALPACA_SECRET_KEY"]
BASE_URL = "https://paper-api.alpaca.markets"
api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=BASE_URL)

# Page config
st.set_page_config(page_title="📈 Live Stock Dashboard", layout="wide")
st.title("📈 Live Stock Dashboard with MLMI & Indicators")

# Sidebar controls
symbol = st.sidebar.selectbox("Choose symbol", ["LCID", "MTC", "ADIL", "JAGX", "ADD", "TPET"])
lookback_minutes = st.sidebar.slider("Lookback (minutes)", 15, 240, 60, step=15)
refresh_rate = st.sidebar.slider("Auto-refresh (seconds)", 15, 300, 60)
indicators_to_show = st.sidebar.multiselect(
    "Select indicators to plot",
    options=["sma_20", "ema_20", "rsi_14", "macd", "mlmi"],
    default=["sma_20", "ema_20", "rsi_14", "macd", "mlmi"]
)

# Define time window
end_time = datetime.datetime.now(datetime.timezone.utc)
start_time = end_time - datetime.timedelta(minutes=lookback_minutes)

# MLMI helper
def compute_mlmi(series, window=14):
    mlmi = [None] * window
    for i in range(window, len(series)):
        X = np.arange(window).reshape(-1, 1)
        y = series[i - window:i]
        model = KNeighborsRegressor(n_neighbors=3)
        model.fit(X, y)
        pred = model.predict(np.array([[window]]))
        mlmi.append(pred[0])
    return pd.Series(mlmi, index=series.index)

# Add indicators to DataFrame
def add_indicators(df):
    if len(df) < 14:
        st.warning(f"⚠️ Only {len(df)} rows — not enough for indicators.")
        for col in ["sma_20", "ema_20", "rsi_14", "macd", "mlmi"]:
            df[col] = None
        return df
    df["sma_20"] = sma_indicator(df["close"], window=20)
    df["ema_20"] = ema_indicator(df["close"], window=20)
    df["rsi_14"] = rsi(df["close"], window=14)
    df["macd"] = macd(df["close"])
    df["mlmi"] = compute_mlmi(df["close"], window=14)
    return df

# Fetch and process data
@st.cache_data(ttl=refresh_rate)
def fetch_data():
    bars = api.get_bars(symbol, "5Min", start=start_time.isoformat(), end=end_time.isoformat(), feed="sip").df
    bars.reset_index(inplace=True)
    bars = bars.rename(columns={"timestamp": "datetime"})
    bars["datetime"] = pd.to_datetime(bars["datetime"], utc=True)
    bars = add_indicators(bars)
    return bars

# Load data
df = fetch_data()

if df.empty:
    st.error("❌ No data returned.")
else:
    available_indicators = [col for col in indicators_to_show if col in df.columns]
    st.subheader(f"{symbol} — Last {lookback_minutes} min")
    st.line_chart(df.set_index("datetime")[["close"] + available_indicators])

    for ind in available_indicators:
        if df[ind].notna().any():
            st.subheader(f"{ind.upper()} Indicator")
            st.line_chart(df.set_index("datetime")[[ind]])

    st.subheader("🧾 Latest Data")
    st.dataframe(df.tail(50))
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, f"{symbol}_live.csv", "text/csv")
