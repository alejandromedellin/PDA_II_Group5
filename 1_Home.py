import streamlit as st

st.title("Welcome to the Trading System")
st.write("""
This interactive application allows you to explore historical and real-time stock market data, analyze financial statements,
and view model-generated trading signals. Our system uses data only for the top 10 most traded companies to ensure reliability and focus.
""")

st.subheader("Core Functionalities")
st.markdown("""
- **Data Analytics:** Cleaned and feature-engineered financial data via our ETL pipeline.
- **Predictive Modeling:** A machine learning model that forecasts next-day market movements.
- **Trading Strategy:** Simple rules to generate actionable trading signals (BUY, SELL, or HOLD).
- **Interactive Dashboard:** Explore stock data in real time and view predictions.
""")


st.subheader("About the Development Team")
st.write("""
- **Samir Barakat** – Lead Developer  
- **Joy Zhong and Noureldin Sewlilam** – Data Scientists
- **Thomas Renwick and Pedro Alejandro Medellín** – DevOps Specialists
""")