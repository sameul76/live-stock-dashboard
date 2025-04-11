import streamlit as st
import pandas as pd
import numpy as np
import datetime
from sklearn.neighbors import KNeighborsRegressor
from ta.trend import sma_indicator, ema_indicator, macd
from ta.momentum import rsi
from alpaca_trade_api.rest import REST
import plotly.express as px

# Load API keys securely from Streamlit secrets
ALPACA_API_KEY = st.secrets["ALPACA_API_KEY"]
ALPACA_SECRET_KEY = st.secrets["ALPACA_SECRET_KEY"]
BASE_URL = "https://paper-api.alpaca.markets"
api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=BASE_URL)

# Page layout
st.set_page_config(page_title="📈 Live Stock Dashboard", layout="wide")
st.title("📈 Live Stock Dashboard with MLMI & Technical Indicators")

# Sidebar controls
symbol = st.sidebar.selectbox("Choose symbol", ["LCID", "MTC", "ADIL", "JAGX", "ADD", "TPET"])
lookback_minutes = st.sidebar.slider("Lookback (minutes)", 15, 240, 60, step=15)
refresh_rate = st.sidebar.slider("Auto-refresh (seconds)", 15, 300, 60)
indicators_to_show = st.sidebar.multiselect(
    "Select indicators to plot",
    options=["sma_20", "ema_20", "rsi_14", "macd", "mlmi"],
    default=["sma_20", "ema_20", "rsi_14", "macd", "mlmi"]
)

# Time window
end_time = datetime.datetime.now(datetime.timezone.utc)
start_time = end_time - datetime.timedelta(minutes=lookback_minutes)

# MLMI calculation
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

# Add indicators
def add_indicators(df):
    if len(df) < 20:
        st.warning(f"⚠️ Only {len(df)} rows — not enough for indicators.")
        for col in ["sma_20", "ema_20", "rsi_14", "macd", "mlmi"]:
            df[col] = None
        return df
    df["sma_20"] = sma_indicator(df["close"], window=20)
    df["ema_20"] = ema_indicator(df["close"], window=20)
    df["rsi_14"] = rsi(df["close"], window=14)
    macd_df = macd(df["close"])
    df["macd"] = macd_df["MACD"]
    df["mlmi"] = compute_mlmi(df["close"], window=14)
    return df

# Fetch stock data
@st.cache_data(ttl=60)
def fetch_data():
    try:
        bars = api.get_bars(symbol, "5Min", start=start_time.isoformat(), end=end_time.isoformat(), feed="sip").df
        if bars.empty:
            return pd.DataFrame()
        bars.reset_index(inplace=True)
        bars = bars.rename(columns={"timestamp": "datetime"})
        bars["datetime"] = bars["datetime"].dt.tz_convert("America/Los_Angeles")  # Convert to PST
        bars = add_indicators(bars)
        return bars
    except Exception as e:
        st.error(f"⚠️ Failed to fetch data: {e}")
        return pd.DataFrame()

# Refresh mechanism
if "manual_refresh" not in st.session_state:
    st.session_state.manual_refresh = False

if st.button("🔄 Refresh Now"):
    st.session_state.manual_refresh = True

# Clear cache manually if the button was clicked
if st.session_state.manual_refresh:
    fetch_data.clear()
    st.session_state.manual_refresh = False
    st.info("🔄 Refresh requested — please click the button again or reload the app.")

# Load the data
df = fetch_data()

if df.empty:
    st.error("❌ No data returned.")
else:
    available_indicators = [col for col in indicators_to_show if col in df.columns]

    plot_columns = ["close"] + [ind for ind in available_indicators if ind in df.columns and df[ind].notna().any()]
    if not plot_columns or df[plot_columns].dropna().empty:
        st.warning("⚠️ No valid indicators with data to plot.")
    else:
        st.subheader(f"{symbol} — Last {lookback_minutes} Minutes")
        fig = px.line(df, x="datetime", y=plot_columns, title=f"{symbol} Price and Indicators")
        st.plotly_chart(fig, use_container_width=True)

        for ind in available_indicators:
            if ind in df.columns and df[ind].notna().any():
                st.subheader(f"{ind.upper()} Indicator")
                ind_fig = px.line(df, x="datetime", y=ind, title=f"{ind.upper()} Over Time")
                st.plotly_chart(ind_fig, use_container_width=True)

    st.subheader("🧾 Latest Data")
    st.dataframe(df.tail(50))

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, f"{symbol}_live.csv", "text/csv")
