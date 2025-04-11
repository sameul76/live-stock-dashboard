import streamlit as st
import pandas as pd
import numpy as np
import datetime
from sklearn.neighbors import KNeighborsRegressor
from ta.trend import sma_indicator, ema_indicator, macd
from ta.momentum import rsi
from alpaca_trade_api.rest import REST

# Load Alpaca API keys from secrets
ALPACA_API_KEY = st.secrets["ALPACA_API_KEY"]
ALPACA_SECRET_KEY = st.secrets["ALPACA_SECRET_KEY"]
BASE_URL = "https://paper-api.alpaca.markets"
api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=BASE_URL)

st.set_page_config(page_title="📈 Live Stock Dashboard", layout="wide")
st.title("📈 Live Stock Dashboard with MLMI & Indicators")

symbol = st.sidebar.selectbox("Choose symbol", ["LCID", "MTC", "ADIL", "JAGX", "ADD", "TPET"])
lookback_minutes = st.sidebar.slider("Lookback window (minutes)", 15, 240, 60, step=15)
refresh_rate = st.sidebar.slider("Auto-refresh (seconds)", 15, 300, 60)

end_time = datetime.datetime.now(datetime.timezone.utc)
start_time = end_time - datetime.timedelta(minutes=lookback_minutes)

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

def add_indicators(df):
    if len(df) < 14:
        st.warning(f"Only {len(df)} rows — not enough for full indicators.")
        df["sma_20"] = df["ema_20"] = df["rsi_14"] = df["macd"] = df["mlmi"] = None
        return df
    df["sma_20"] = sma_indicator(df["close"], window=20)
    df["ema_20"] = ema_indicator(df["close"], window=20)
    df["rsi_14"] = rsi(df["close"], window=14)
    df["macd"] = macd(df["close"])
    df["mlmi"] = compute_mlmi(df["close"], window=14)
    return df

@st.cache_data(ttl=refresh_rate)
def fetch_data():
    bars = api.get_bars(symbol, "1Min", start=start_time.isoformat(), end=end_time.isoformat(), feed="sip").df
    bars.reset_index(inplace=True)
    bars = bars.rename(columns={"timestamp": "datetime"})
    bars["datetime"] = pd.to_datetime(bars["datetime"], utc=True)
    bars = add_indicators(bars)
    return bars

df = fetch_data()

if df.empty:
    st.error("No data returned.")
else:
    st.subheader(f"{symbol} — {lookback_minutes} min view")
    st.line_chart(df.set_index("datetime")[["close", "sma_20", "ema_20"]])
    st.subheader("📊 Indicators")
    st.line_chart(df.set_index("datetime")[["rsi_14", "macd", "mlmi"]])
    st.dataframe(df.tail(50))
    st.download_button("⬇ Download CSV", df.to_csv(index=False), f"{symbol}_live.csv", mime="text/csv")
