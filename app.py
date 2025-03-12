import streamlit as st

st.set_page_config(
    page_title="Trading System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Trading System Web Application")
st.write("""
Welcome to our Trading System! This application integrates an ML model to predict next-day market movements,
an API wrapper to retrieve financial data from SimFin, and an interactive dashboard to display trading signals.
Use the sidebar to navigate between pages.
""")