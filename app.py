import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import random
import re

# ---------------------------------
# Load Model
# ---------------------------------
try:
    model = load_model("Stock Model.keras")
except Exception as e:
    st.warning(f"⚠️ Model could not be loaded: {e}")
    model = None

# ---------------------------------
# Streamlit Config
# ---------------------------------
st.set_page_config(page_title="📈 Stock Market Predictor", layout="wide")
st.title(":chart_with_upwards_trend: Stock Market Prediction & Analysis App")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

# ---------------------------------
# Sidebar – Combined Dropdown + Custom Input
# ---------------------------------
st.sidebar.header("🔍 Stock Selection")

# Predefined stock list
us_stocks = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NVDA"]
india_stocks = [ 'NFLX', 'INTC', 'AMD', 'CSCO', 'ADBE', 'PYPL',
    'PUM', 'BAC', 'ADS', 'V', 'JPM', 'WMT', 'MA', 'CRM', 'PG', 'KO', 'PEP', 'MCD', 'NKE', 'BA', 'XOM',
    'TM', 'HMC', 'NVS', 'TSM', 'RIO', 'BP', 'UL', 'IBM', 'ORCL', 'JD', 'C', 'WFC', 'VZ', 'T', 'GE', 'LMT',
    'INFY', 'TATASTEEL.NS', 'RELIANCE.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'HINDUNILVR.NS', 'AXISBANK.NS', 'SBIN.NS','TCS.NS', 'COALINDIA.NS', 'TITAN.NS']
top_stocks = us_stocks + india_stocks

# Dropdown with option to type custom
selected_stock = st.sidebar.selectbox(
    "Select or Type a Stock Symbol",
    options=top_stocks ,
    index=0
)

# Show input box only if "Type custom..." selected
if selected_stock == "🔹 Type custom...":
    custom_stock = st.sidebar.text_input("Enter custom stock symbol (e.g. ADANIENT.NS, NFLX):", "")
    stock = custom_stock.strip().upper() if custom_stock else None
else:
    stock = selected_stock

# Date selection (below stock input)
start = st.sidebar.date_input("📅 Start Date", pd.to_datetime("2013-01-01"))
end = st.sidebar.date_input("📅 End Date", pd.to_datetime("2025-11-13"))

# Info message if no stock selected yet
if not stock:
    st.sidebar.info("👆 Select or type a stock symbol to proceed.")

# -------------------------------
# 🟢 Live Stock Price Tracker
# -------------------------------
with st.expander("📡 Fetch Live Stock Price "):
    st.write("Get latest live prices of your favorite stocks.")

    # Predefined list of popular stocks
    top_stocks = [
        "AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NVDA",
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", 
        "ICICIBANK.NS", "SBIN.NS", "HINDUNILVR.NS"
    ]

    # User can select from dropdown or type their own
    tickers = st.multiselect(
        "Select or Type Stock Symbols (you can pick multiple)",
        options=top_stocks,
        default=[],
        help="Select from list or type any valid stock symbol (e.g. ADANIENT.NS, NFLX)."
    )

    if not tickers:
        st.info("👆 Select or type at least one stock symbol above to view live data.")
    else:
        if st.button("🔄 Refresh Live Data"):
            live_data = []
            for symbol in tickers:
                try:
                    info = yf.Ticker(symbol).info
                    current = info.get("currentPrice")
                    previous = info.get("previousClose")
                    if current and previous:
                        change = current - previous
                        percent = (change / previous) * 100
                        live_data.append([
                            symbol, round(current, 2), round(previous, 2),
                            f"{change:+.2f}", f"{percent:+.2f}%"
                        ])
                except Exception:
                    continue

            if live_data:
                df_live = pd.DataFrame(
                    live_data,
                    columns=["Stock", "Current Price", "Prev Close", "Change", "% Change"]
                )
                st.dataframe(df_live, use_container_width=True)
            else:
                st.warning("⚠️ Unable to fetch live prices. Try again later.")

# ---------------------------------
# Fetch Data
# ---------------------------------
data = yf.download(stock, start, end)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = [col[0] for col in data.columns]

if data.empty:
    st.error("❌ No data found for this symbol. Please try another one.")
    st.stop()

# ---------------------------------
# Display Stock Data
# ---------------------------------
st.subheader("📊 Historical Stock Data")
st.write(data.tail())

col1, col2, col3 = st.columns(3)
col1.metric("📈 Highest Price", f"{data['High'].max():.2f}")
col2.metric("📉 Lowest Price", f"{data['Low'].min():.2f}")
col3.metric("💹 Total Volume", f"{data['Volume'].sum():,.0f}")

# ---------------------------------
# Growth Calculation
# ---------------------------------
try:
    start_price = float(data['Close'].iloc[0])
    end_price = float(data['Close'].iloc[-1])
    growth = ((end_price - start_price) / start_price) * 100
    st.subheader("📆 Growth Over Selected Period")
    st.write(f"The company's stock price grew by **{growth:.2f}%** from **{start}** to **{end}**.")
except Exception:
    st.warning("Could not calculate growth (missing data).")

# ---------------------------------
# Price Line Chart
# ---------------------------------
price_data = data.reset_index()
fig = px.line(price_data, x='Date', y='Close', title=f'{stock} Stock Price Trend')
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------
# Moving Averages
# ---------------------------------
st.subheader("📉 Moving Averages Comparison")
ma_50 = data['Close'].rolling(50).mean()
ma_100 = data['Close'].rolling(100).mean()
ma_200 = data['Close'].rolling(200).mean()

fig_ma = go.Figure()
fig_ma.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Original Price'))
fig_ma.add_trace(go.Scatter(x=data.index, y=ma_50, mode='lines', name='MA 50 days'))
fig_ma.add_trace(go.Scatter(x=data.index, y=ma_100, mode='lines', name='MA 100 days'))
fig_ma.add_trace(go.Scatter(x=data.index, y=ma_200, mode='lines', name='MA 200 days'))
fig_ma.update_layout(title='Price vs Moving Averages', xaxis_title='Date', yaxis_title='Price')
st.plotly_chart(fig_ma, use_container_width=True)

# ---------------------------------
# Prediction Section
# ---------------------------------
if model is not None and len(data) > 120:
    st.subheader("🤖 LSTM Model Prediction")

    data_train = pd.DataFrame(data.Close[0:int(len(data)*0.80)])
    data_test = pd.DataFrame(data.Close[int(len(data)*0.80):len(data)])
    scaler = MinMaxScaler(feature_range=(0,1))
    past_100_days = data_train.tail(100)
    data_test = pd.concat([past_100_days, data_test], ignore_index=True)
    data_test_scale = scaler.fit_transform(data_test)

    x, y = [], []
    for i in range(100, data_test_scale.shape[0]):
        x.append(data_test_scale[i-100:i])
        y.append(data_test_scale[i, 0])
    x, y = np.array(x), np.array(y)

    predict = model.predict(x)
    scale = 1/scaler.scale_
    predict = predict * scale
    y = y * scale

    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(y=predict.flatten(), mode='lines', name='Predicted Price'))
    fig_pred.add_trace(go.Scatter(y=y.flatten(), mode='lines', name='Actual Price'))
    fig_pred.update_layout(title='Predicted vs Actual Prices', xaxis_title='Time', yaxis_title='Price')
    st.plotly_chart(fig_pred, use_container_width=True)

    # ---------------------------------
    # Stabilized Next 10 Days Forecast
    # ---------------------------------
    st.subheader("🔮 Next 10 Days Price Forecast")

    last_100 = data_test.tail(100)
    data_scaled = scaler.fit_transform(last_100)
    x_pred = np.array([data_scaled])

    predicted_prices = []
    for _ in range(10):
        pred = model.predict(x_pred)
        predicted_prices.append(pred[0])
        new_data = np.array([[[pred[0][0]]]])
        x_pred = np.concatenate((x_pred[:, 1:, :], new_data), axis=1)

    predicted_prices = np.array(predicted_prices).reshape(-1, 1)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    # Stabilization logic: make predictions more neutral
    last_price = data['Close'].iloc[-1]
    stabilized_prices = []
    for p in predicted_prices:
        noise_factor = random.uniform(-0.02, 0.02)
        adjusted = p[0] * (1 + noise_factor)
        adjusted = (adjusted + last_price) / 2
        stabilized_prices.append(adjusted)
        last_price = adjusted

    next_10_days = pd.date_range(start=data.index[-1] + pd.DateOffset(1), periods=10)
    forecast_df = pd.DataFrame({'Date': next_10_days, 'Predicted Price': stabilized_prices})
    st.write(forecast_df)

    fig_next = px.line(forecast_df, x='Date', y='Predicted Price', title='Next 10 Days Forecast (Stabilized)')
    st.plotly_chart(fig_next, use_container_width=True)

# -------------------------------
# Contact Section
# -------------------------------
st.markdown("---")
colA, colB = st.columns(2)

with colA:
    st.header("📬 Contact Me")
    name = st.text_input("Your Name")
    email = st.text_input("Your Email")
    message = st.text_area("Message")

    def save_contact_info(name, email, message):
        with open('contact_info.txt', 'a') as f:
            f.write(f"Name: {name}\nEmail: {email}\nMessage: {message}\n\n")

    def validate_email(email):
        pattern = r"[^@]+@[^@]+\.[^@]+"
        return re.match(pattern, email)

    if st.button("Send Message"):
        if not name or not validate_email(email) or not message:
            st.warning("⚠️ Please provide a valid email and fill all fields.")
        else:
            save_contact_info(name, email, message)
            st.success("✅ Message sent successfully!")

with colB:
    st.header("ℹ️ About")
    st.write("""
    This app lets you:
    - 📊 View historical stock data
    - 📈 Analyze growth & moving averages
    - 🤖 Predict future prices with LSTM model
    - 📬 Send feedback or contact the developer
    """)

st.markdown("---")
st.caption("© 2025 Stock Predictor by Anshu Yadav")
