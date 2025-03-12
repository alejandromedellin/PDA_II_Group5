# Trading System Web Application

Welcome to our Trading System Web Application! This project combines an ETL process, a machine learning model, a Python API wrapper for SimFin, and a Streamlit-based web interface to deliver real-time stock data, financial statements, trading signals, and a backtesting simulator.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Required CSV Files](#required-csv-files)
- [Usage](#usage)
- [Deployment](#deployment)
- [License](#license)
- [Contact](#contact)

---

## Overview

This application provides:
1. **Share Prices:** Fetch daily share price data for selected tickers.
2. **Financial Statements:** Retrieve quarterly or annual statements from SimFin.
3. **ML Model Predictions:** Predict next-day price movements (BUY, SELL, HOLD).
4. **Trading Strategy & Signals:** Suggest actions based on model output.
5. **Optional Backtesting:** Simulate how a strategy would have performed historically.

The system uses an ETL script to clean and prepare data, an XGBoost-based machine learning model to predict stock movements, and a Streamlit multipage interface for user interaction.

---

## Project Structure
