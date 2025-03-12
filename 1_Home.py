import streamlit as st

# ---------------------------------------------------
# Optional: Set a custom page config for a nice layout
# ---------------------------------------------------
st.set_page_config(
    page_title="Home - Trading System",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# HERO SECTION
# ---------------------------------------------------
st.markdown(
    """
    <div style="
        text-align: center; 
        padding: 2rem 0 1rem 0;
        border-radius: 5px;">
      <h1 style="font-size:3em; margin-bottom:0.2em;">
        Welcome to the Trading System
      </h1>
      <p style="font-size:1.2em; max-width: 600px; margin: 0 auto;">
        An interactive application that harnesses ML-models, real-time data, and streamlined analytics
        to empower your trading decisions.
      </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------
# 1) OVERVIEW OF THE TRADING SYSTEM
# ---------------------------------------------------
st.markdown("---")
st.subheader("Core Functionalities")

col1, col2 = st.columns([1,1])

with col1:
    st.write("**Data Analytics**")
    st.markdown(
        """
        - Cleaned and feature-engineered market & fundamental data  
        - Customizable date ranges and real-time updates
        """
    )
    st.write("**Predictive Modeling**")
    st.markdown(
        """
        - Next-day price movement forecasts  
        - Automated feature engineering (rolling averages, volatility, etc.)
        """
    )

with col2:
    st.write("**Trading Strategy**")
    st.markdown(
        """
        - Simple or advanced rule-based signals (BUY, SELL, HOLD)  
        - Backtesting and performance evaluation
        """
    )
    st.write("**Interactive Dashboard**")
    st.markdown(
        """
        - Visualize stock data with dynamic charts  
        - Explore fundamental statements in real time
        """
    )

# ---------------------------------------------------
# 2) DEVELOPMENT TEAM
# ---------------------------------------------------
st.markdown("---")
st.subheader("Meet the Development Team")

team_html = """
<div style="display: flex; flex-direction: row; flex-wrap: wrap;">
  <div style="margin-right: 2rem; margin-bottom: 1rem;">
    <h4 style="margin:0;">
      <a href="https://www.linkedin.com/in/samir-barakat-245420127/" target="_blank"
         style="text-decoration: none; color: #2c3e50;">
         Samir Barakat
      </a>
    </h4>
    <p style="margin:0;">Lead Developer</p>
  </div>
  <div style="margin-right: 2rem; margin-bottom: 1rem;">
    <h4 style="margin:0;">
      <a href="https://www.linkedin.com/in/noursewilam/" target="_blank"
         style="text-decoration: none; color: #2c3e50;">
         Noureldin Sewlilam
      </a>
      &nbsp;and&nbsp;
      <a href="https://www.linkedin.com/in/joy-zhong/" target="_blank"
         style="text-decoration: none; color: #2c3e50;">
         Joy Zhong
      </a>
    </h4>
    <p style="margin:0;">Data Scientists</p>
  </div>
  <div style="margin-right: 2rem; margin-bottom: 1rem;">
    <h4 style="margin:0;">
      <a href="https://www.linkedin.com/in/thomasrenwickmorales/" target="_blank"
         style="text-decoration: none; color: #2c3e50;">
         Thomas Renwick
      </a>
      &nbsp;and&nbsp;
      <a href="https://www.linkedin.com/in/pedroalejandromedellin/" target="_blank"
         style="text-decoration: none; color: #2c3e50;">
         Pedro Alejandro Medellín
      </a>
    </h4>
    <p style="margin:0;">DevOps Specialists</p>
  </div>
</div>
"""
st.markdown(team_html, unsafe_allow_html=True)

# ---------------------------------------------------
# 3) SYSTEM PURPOSE & OBJECTIVES
# ---------------------------------------------------
st.markdown("---")
st.subheader("Our Purpose & Objectives")
st.markdown(
    """
    The Trading System is designed to **empower users** with:
    - **Actionable insights**: We combine top data sources and machine learning 
      models to produce highly relevant trading signals.
    - **Transparency**: Each signal is backed by well-documented 
      analytics and auditable data pipelines.
    - **Efficiency**: Our streamlined workflow integrates real-time data ingestion,
      model inference, and live dashboards in a single, user-friendly interface.
    
    ### Our Vision
    We believe **data-driven** decisions can revolutionize how traders and analysts 
    approach the market. By **lowering the barrier** to advanced analytics, 
    we aim to democratize access to powerful insights typically reserved for large institutions.
    
    ### Who Is It For?
    - **Individual Investors** seeking a user-friendly tool to analyze and trade top companies.
    - **Financial Analysts** looking for quick, robust data explorations.
    - **Developers & Data Scientists** who want to extend the platform with custom features.
    
    ---
    **Ready to Explore?**  
    Use the sidebar to navigate through:
    - **Go Live** to see real-time predictions.
    - **Backtesting** to validate your strategies historically.
    """
)

# ---------------------------------------------------
# 4) CLOSING REMARKS OR CALL-TO-ACTION
# ---------------------------------------------------
st.markdown(
    """
    <div style="
        background-color: #f9f9f9; 
        padding: 1.5rem; 
        margin-top: 2rem; 
        border-radius: 5px;">
      <h3 style="margin-top:0;">Get Started Now!</h3>
      <p style="margin-bottom:0.5em;">
        Leverage our AI-driven approach to make confident trading decisions.
      </p>
      <p style="font-size:0.9em;">
        Have feedback or questions? Reach out to our development team at
        <a href="mailto:group5@tradingsystem.io">group5@tradingsystem.io</a>.
      </p>
    </div>
    """,
    unsafe_allow_html=True
)
