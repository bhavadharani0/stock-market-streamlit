import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from streamlit_autorefresh import st_autorefresh

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Live Stock Prediction", layout="wide")

# -------------------- AUTO REFRESH --------------------
# Refresh every 60 seconds (60000 ms)
st_autorefresh(interval=60000, key="stock_refresh")

st.title("📈 Live Stock Market Prediction App")
st.caption("🔄 Auto refresh every 60 seconds")

# -------------------- USER INPUT --------------------
ticker = st.text_input("Enter Stock Symbol (Example: AAPL, TSLA, INFY.NS)", "AAPL")

period = st.selectbox(
    "Select Data Period",
    ["6mo", "1y", "2y", "5y"],
    index=1
)

# Manual refresh button (optional)
if st.button("🔄 Refresh Now"):
    st.experimental_rerun()

# -------------------- FETCH STOCK DATA --------------------
try:
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)

    if data.empty:
        st.error("❌ Invalid stock symbol or no data found!")
        st.stop()

except Exception:
    st.error("❌ Error fetching stock data")
    st.stop()

# -------------------- DISPLAY DATA --------------------
st.subheader("📄 Latest Stock Data")
st.dataframe(data.tail())

# -------------------- LIVE PRICE CHART --------------------
st.subheader("📊 Live Stock Price Chart")

price_fig = go.Figure()
price_fig.add_trace(go.Scatter(
    x=data.index,
    y=data["Close"],
    mode="lines",
    name="Close Price"
))

price_fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Price",
    template="plotly_dark"
)

st.plotly_chart(price_fig, use_container_width=True)

# -------------------- MACHINE LEARNING DATA --------------------
ml_data = data[['Close']].copy()
ml_data['Prediction'] = ml_data['Close'].shift(-1)
ml_data.dropna(inplace=True)

X = np.array(ml_data['Close']).reshape(-1, 1)
y = np.array(ml_data['Prediction'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# -------------------- LINEAR REGRESSION --------------------
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_prediction = lr.predict(X[-1].reshape(1, -1))[0]

# -------------------- RANDOM FOREST --------------------
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
rf.fit(X_train, y_train)
rf_prediction = rf.predict(X[-1].reshape(1, -1))[0]

# -------------------- SHOW PREDICTIONS --------------------
st.subheader("🤖 Next Day Price Prediction")

col1, col2 = st.columns(2)

col1.metric(
    label="Linear Regression",
    value=f"${lr_prediction:.2f}"
)

col2.metric(
    label="Random Forest",
    value=f"${rf_prediction:.2f}"
)

# -------------------- PREDICTION CHART --------------------
st.subheader("📉 Prediction Chart")

pred_fig = go.Figure()

pred_fig.add_trace(go.Scatter(
    x=data.index,
    y=data["Close"],
    mode="lines",
    name="Actual Price"
))

pred_fig.add_trace(go.Scatter(
    x=[data.index[-1]],
    y=[lr_prediction],
    mode="markers",
    marker=dict(size=12, color="green"),
    name="Linear Regression"
))

pred_fig.add_trace(go.Scatter(
    x=[data.index[-1]],
    y=[rf_prediction],
    mode="markers",
    marker=dict(size=12, color="red"),
    name="Random Forest"
))

pred_fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Price",
    template="plotly_dark"
)

st.plotly_chart(pred_fig, use_container_width=True)

# -------------------- FOOTER --------------------
st.warning("⚠ Educational purpose only. Not financial advice.")