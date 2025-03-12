import streamlit as st
import pandas as pd
from datetime import date, timedelta, datetime
from pysimfin import PySimFin
from ML_MODEL import predict_signals, transform_api_data

# Define the top 10 companies (as filtered in your ETL)
top_companies = ['AAPL', 'AMZN', 'GOOG', 'MSFT', 'META', 'TSLA', 'NUVB', 'V', 'JNJ', 'WMT']

st.title("Go Live: Stock Data, Financial Statements, and Trading Signals")

# ------------------------------
# Global Ticker Selection
# ------------------------------
st.sidebar.header("Global Settings")
# Optionally add a placeholder option so the user must select one:
common_ticker = st.sidebar.selectbox("Select a Company", ["Select a ticker"] + top_companies)
if common_ticker == "Select a ticker":
    st.sidebar.warning("Please select a company to proceed.")
    st.stop()

# Initialize the API wrapper with your API key.
API_KEY = "79731e86-a369-4e0c-9683-f1044bdfba88"
simfin = PySimFin(API_KEY)

##########################
# Section 1: Price Data
##########################
st.header("1. Price Data")
st.markdown("View the last month's share prices for the selected company. Adjust the date range as desired for your analysis.")

# Allow the user to modify the date range for price data
price_start_date = st.sidebar.date_input("Price Data Start Date", date.today() - timedelta(days=30), key="price_start")
price_end_date = st.sidebar.date_input("Price Data End Date", date.today(), key="price_end")

st.write(f"Fetching share prices for **{common_ticker}** from **{price_start_date}** to **{price_end_date}**.")

try:
    price_df = simfin.get_share_prices(ticker=common_ticker, 
                                       start=price_start_date.strftime("%Y-%m-%d"), 
                                       end=price_end_date.strftime("%Y-%m-%d"))
    if price_df.empty:
        st.error("No price data found for the selected ticker and date range.")
    else:
        st.subheader(f"Share Prices for {common_ticker}")
        st.dataframe(price_df.head())
        # Plot a line chart of closing prices (prefer "Last Closing Price")
        close_col = "Last Closing Price" if "Last Closing Price" in price_df.columns else "close"
        if "Date" in price_df.columns:
            price_df["Date"] = pd.to_datetime(price_df["Date"])
            price_df.sort_values("Date", inplace=True)
            st.line_chart(price_df.set_index("Date")[close_col])
except Exception as e:
    st.error(f"Error fetching price data: {e}")

################################
# Section 2: Financial Statements
################################
st.header("2. Financial Statements")
st.markdown("The most recent financial statements are displayed below. By default, data for 2024-Q4 is shown; you can modify the year and period.")

# Allow user to modify year and period for financial statements
fin_year = st.sidebar.number_input("Fiscal Year", value=2024, step=1, key="fin_year")
# Set default index to 3 (0-based indexing: 0 -> Q1, 1 -> Q2, 2 -> Q3, 3 -> Q4, 4 -> FY)
fin_period = st.sidebar.selectbox("Fiscal Period", ["Q1", "Q2", "Q3", "Q4", "FY"], index=3, key="fin_period")
fin_stmt_options = st.sidebar.multiselect(
    "Select Statement Types",
    options=["PL", "BS", "CF", "DERIVED"],
    default=["PL", "BS", "CF", "DERIVED"],
    key="fin_stmt"
)

st.write(f"Fetching financial statements for **{common_ticker}** for fiscal year **{fin_year}** ({fin_period}).")
try:
    statements_str = ",".join(fin_stmt_options)
    fin_data = simfin.get_financial_statement(
        ticker=common_ticker,
        fyear=fin_year,
        period=fin_period,
        statements=statements_str
    )
    if fin_data:
        for stmt_type, df in fin_data.items():
            st.subheader(f"Financial Statement: {stmt_type}")
            st.dataframe(df.head())
    else:
        st.error("No financial statement data found for the selected parameters.")
except Exception as e:
    st.error(f"Error fetching financial statements: {e}")

############################
# Section 3: Trading Signals
############################
st.header("3. Trading Signals")
st.markdown("Using yesterday's price data, our ML model predicts whether you should BUY, SELL, or HOLD.")

# For trading signals, use yesterday's date for a single day's data.
yesterday = date.today() - timedelta(days=1)
st.write(f"Fetching trading data for **{common_ticker}** on **{yesterday.strftime('%Y-%m-%d')}** for prediction.")

try:
    # Retrieve share price data via the API wrapper.
    trade_df = simfin.get_share_prices(
        ticker=common_ticker,
        start=yesterday.strftime("%Y-%m-%d"),
        end=yesterday.strftime("%Y-%m-%d")
    )
    if trade_df.empty:
        st.error("No trading data available for the selected ticker on the specified date.")
    else:
        # Transform API data to the format expected by the ML model.
        transformed_trade_df = transform_api_data(trade_df)
        st.subheader("Transformed Data for Prediction")
        st.dataframe(transformed_trade_df.head())
        # Generate trading signals using the saved ML model.
        signals = predict_signals(transformed_trade_df)
        st.subheader("Predicted Trading Signal")
        if not signals.empty:
            # Display the prediction as a large, visually appealing message.
            signal = signals.iloc[0]
            action = signal["Action"]
            quantity = signal["Quantity"]
            buy_prob = signal["Buy Probability"]
            if action == "BUY":
                st.markdown(
                    f"<h1 style='color: green;'>ACTION: BUY {quantity} Shares NOW!</h1>", 
                    unsafe_allow_html=True
                )
            elif action == "SELL":
                st.markdown(
                    f"<h1 style='color: red;'>ACTION: SELL {quantity} Shares NOW!</h1>", 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<h1 style='color: blue;'>ACTION: HOLD - No immediate action required</h1>", 
                    unsafe_allow_html=True
                )
            st.markdown(f"<h3>Buy Probability: {buy_prob:.2f}</h3>", unsafe_allow_html=True)
        else:
            st.error("No trading signals generated.")
except Exception as e:
    st.error(f"Error generating trading signals: {e}")