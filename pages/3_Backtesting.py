import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date, timedelta
from ML_MODEL import predict_signals, transform_api_data

st.title("Backtesting Simulator")
st.markdown("Evaluate historical trade scenarios based on our ML model predictions.")

# User inputs
ticker = st.selectbox("Select Company for Backtesting", ['AAPL', 'AMZN', 'GOOGL', 'MSFT', 'FB', 'TSLA', 'BRK.B', 'V', 'JNJ', 'WMT'])
backtest_start_date = st.date_input("Backtest Start Date", date(2020, 1, 1))
initial_capital = st.number_input("Initial Capital ($)", value=10000)

# Load historical data from the cleaned CSV (assumed to be produced by your ETL process)
@st.cache_data
def load_historical_data(ticker, start_date):
    try:
        df = pd.read_csv("cleaned_stock_data_train.csv", parse_dates=["date"])
        df = df[df["ticker"] == ticker].copy()
        df.sort_values("date", inplace=True)
        df = df[df["date"] >= pd.to_datetime(start_date)]
        return df
    except Exception as e:
        st.error(f"Error loading historical data: {e}")
        return pd.DataFrame()

hist_data = load_historical_data(ticker, backtest_start_date)
if hist_data.empty:
    st.error("No historical data available for the selected ticker and start date.")
    st.stop()

st.write(f"Backtesting {ticker} from {backtest_start_date} with initial capital ${initial_capital}")

# Load saved model (we assume the model file exists from previous training)
try:
    model, feature_list, lbl_encoders = joblib.load("best_trading_model.pkl")
except Exception as e:
    st.error("ML model not found. Please train the model first.")
    st.stop()

# Simulation parameters
cash = initial_capital
shares = 0
portfolio_values = []  # to store portfolio value for each day
dates = []

# Backtesting loop: iterate over each trading day in the historical data
for idx, row in hist_data.iterrows():
    current_date = row["date"]
    close_price = row["close"]

    # Create a single-row DataFrame for the current day
    day_df = pd.DataFrame([row])
    # Transform API data to match model expectations
    day_df_transformed = transform_api_data(day_df)
    # Generate trading signal using the saved ML model
    signals_df = predict_signals(day_df_transformed)
    if signals_df.empty:
        action = "HOLD"
    else:
        signal = signals_df.iloc[0]
        action = signal.get("Action", "HOLD")
    
    # Simple trading strategy:
    # - If BUY: invest 10% of current cash to buy shares (if possible)
    # - If SELL: sell all shares
    # - If HOLD: do nothing
    if action == "BUY" and cash > 0:
        invest_amount = cash * 0.10  # invest 10% of cash
        shares_to_buy = invest_amount // close_price
        if shares_to_buy > 0:
            cost = shares_to_buy * close_price
            cash -= cost
            shares += shares_to_buy
    elif action == "SELL" and shares > 0:
        cash += shares * close_price
        shares = 0

    # Compute portfolio value for the day
    portfolio_value = cash + shares * close_price
    portfolio_values.append(portfolio_value)
    dates.append(current_date)

# Create a DataFrame for portfolio performance
portfolio_df = pd.DataFrame({"date": dates, "portfolio_value": portfolio_values})
portfolio_df.set_index("date", inplace=True)

st.subheader("Backtesting Results")
st.write(f"Final Portfolio Value: ${portfolio_values[-1]:.2f}")
st.line_chart(portfolio_df)
