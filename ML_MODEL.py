import pandas as pd
import numpy as np
import logging
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# -------------------------------------------------------------
# SETUP LOGGING
# -------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def train_model():
    """
    Loads preprocessed train and test datasets, trains an XGBoost model for next-day price movement,
    evaluates the model, and saves the model (with feature list and label encoders) to 'best_trading_model.pkl'.
    """
    # -------------------------------------------------------------
    # LOAD DATA
    # -------------------------------------------------------------
    logging.info("📂 Loading preprocessed training and testing datasets...")
    train_df = pd.read_csv("cleaned_stock_data_train.csv", low_memory=False)
    test_df = pd.read_csv("cleaned_stock_data_test.csv", low_memory=False)

    # Debug: Print columns at the start
    print("🔍 [START] Train Columns:", train_df.columns.tolist())
    print("🔍 [START] Test Columns:", test_df.columns.tolist())

    # Ensure "close" is present
    if "close" not in train_df.columns or "close" not in test_df.columns:
        logging.error("❌ 'close' column is missing from train or test DataFrame!")
        raise KeyError("'close' column is missing!")

    # -------------------------------------------------------------
    # PRESERVE DATE, TICKER, CLOSE (FOR LATER MERGE)
    # -------------------------------------------------------------
    train_dates = train_df[["date", "ticker", "close"]].copy()
    test_dates  = test_df[["date", "ticker", "close"]].copy()

    # -------------------------------------------------------------
    # DEFINE TARGET & FEATURES
    # -------------------------------------------------------------
    target = "price_movement"
    remove_cols = ["ticker", "date", "company_name", "isin", "market", "main_currency"]
    if "close" in remove_cols:
        remove_cols.remove("close")
    features = [c for c in train_df.columns if c not in remove_cols + [target]]
    logging.info(f"🔎 Feature selection completed! Using {len(features)} features.")
    print("🔍 Feature List:", features)

    # -------------------------------------------------------------
    # CONVERT CATEGORICAL -> NUMERIC (LABEL ENCODING) - REVISED
    # -------------------------------------------------------------
    logging.info("🔄 Converting categorical columns to numeric format...")
    categorical_cols = train_df.select_dtypes(include=["object"]).columns
    label_encoders = {}
    for col in categorical_cols:
        if col != "close":  # Skip numeric "close"
            le = LabelEncoder()
            train_df[col] = train_df[col].astype(str)
            le.fit(train_df[col])
            # Transform training data
            train_df[col] = le.transform(train_df[col])
            # Create mapping for test set
            mapping = {cat: idx for idx, cat in enumerate(le.classes_)}
            test_df[col] = test_df[col].astype(str).map(mapping).fillna(-1).astype(int)
            label_encoders[col] = le
    logging.info("✅ Categorical encoding completed!")

    # -------------------------------------------------------------
    # SHIFT TARGET FOR NEXT-DAY PREDICTION
    # -------------------------------------------------------------
    logging.info("🔄 Adjusting target variable for next-day predictions...")
    train_df[target] = train_df[target].shift(-1)
    test_df[target]  = test_df[target].shift(-1)
    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)

    # -------------------------------------------------------------
    # FILTER OUT ROWS WITH CLOSE = 0
    # -------------------------------------------------------------
    train_df = train_df[train_df["close"] > 0]
    test_df  = test_df[test_df["close"] > 0]

    print("🔍 [BEFORE MERGE] Train Columns:", train_df.columns.tolist())
    print("🔍 [BEFORE MERGE] Test Columns:", test_df.columns.tolist())

    # -------------------------------------------------------------
    # REATTACH date, ticker, close (restore original info)
    # -------------------------------------------------------------
    train_df = train_df.merge(train_dates, left_index=True, right_index=True, how="left", suffixes=("", "_DROPPED"))
    test_df  = test_df.merge(test_dates, left_index=True, right_index=True, how="left", suffixes=("", "_DROPPED"))
    print("🔍 [AFTER MERGE] Train Columns:", train_df.columns.tolist())
    print("🔍 [AFTER MERGE] Test Columns:", test_df.columns.tolist())

    # -------------------------------------------------------------
    # RENAME DUPLICATE COLUMNS (if present)
    # -------------------------------------------------------------
    if "close_x" in train_df.columns:
        train_df.drop(columns=["close_y"], errors="ignore", inplace=True)
        train_df.rename(columns={"close_x": "close"}, inplace=True)
    if "ticker_x" in train_df.columns:
        train_df.drop(columns=["ticker_y"], errors="ignore", inplace=True)
        train_df.rename(columns={"ticker_x": "ticker"}, inplace=True)
    if "date_x" in train_df.columns:
        train_df.drop(columns=["date_y"], errors="ignore", inplace=True)
        train_df.rename(columns={"date_x": "date"}, inplace=True)
    if "close_x" in test_df.columns:
        test_df.drop(columns=["close_y"], errors="ignore", inplace=True)
        test_df.rename(columns={"close_x": "close"}, inplace=True)
    if "ticker_x" in test_df.columns:
        test_df.drop(columns=["ticker_y"], errors="ignore", inplace=True)
        test_df.rename(columns={"ticker_x": "ticker"}, inplace=True)
    if "date_x" in test_df.columns:
        test_df.drop(columns=["date_y"], errors="ignore", inplace=True)
        test_df.rename(columns={"date_x": "date"}, inplace=True)
    print("🔍 [FINAL] Train Columns:", train_df.columns.tolist())
    print("🔍 [FINAL] Test  Columns:", test_df.columns.tolist())

    # -------------------------------------------------------------
    # PREPARE FINAL X, Y
    # -------------------------------------------------------------
    X_train = train_df[features]
    y_train = train_df[target]
    X_test  = test_df[features]
    y_test  = test_df[target]
    logging.info("🚀 Training XGBoost model...")

    # -------------------------------------------------------------
    # TRAIN THE MODEL
    # -------------------------------------------------------------
    xgb_model = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.7,
        colsample_bytree=0.7,
        objective='binary:logistic',
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)

    # -------------------------------------------------------------
    # EVALUATE THE MODEL
    # -------------------------------------------------------------
    y_pred = xgb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"📉 XGBoost Accuracy: {accuracy:.4f}")

    # -------------------------------------------------------------
    # SAVE THE MODEL
    # -------------------------------------------------------------
    joblib.dump((xgb_model, features, label_encoders), "best_trading_model.pkl")
    logging.info("✅ Model saved successfully!")

def predict_signals(data: pd.DataFrame) -> pd.DataFrame:
    """
    Predict trading signals using the saved XGBoost model and apply a simple trading strategy.
    
    The strategy uses the predicted probability for an upward price movement (class 1):
      - If probability > 0.55: signal "BUY" for a fixed quantity.
      - If probability < 0.45: signal "SELL" for a fixed quantity.
      - Otherwise: signal "HOLD" (no trade).
    
    Returns a DataFrame with the original 'date', 'ticker', 'close', plus:
      - "Predicted Signal": raw model prediction (0 or 1)
      - "Buy Probability": probability of upward movement (class 1)
      - "Action": trading action ("BUY", "SELL", or "HOLD")
      - "Quantity": number of shares to trade (0 for HOLD)
    """
    import logging
    import joblib
    import numpy as np
    import pandas as pd

    if data.empty:
        logging.warning("⚠️ No data provided for prediction.")
        return pd.DataFrame({"Error": ["No data available for prediction."]})
    try:
        model, feature_list, lbl_encoders = joblib.load("best_trading_model.pkl")
    except FileNotFoundError:
        logging.error("❌ Model file not found! Train the model first.")
        return pd.DataFrame({"Error": ["Model file not found!"]})
    
    # First, transform the API data to include all expected features.
    data_transformed = transform_api_data(data)
    
    # Preserve essential columns for later merge.
    prediction_data = data_transformed[["date", "ticker", "close"]].copy()
    missing_features = [f for f in feature_list if f not in data_transformed.columns]
    if missing_features:
        logging.error(f"❌ Missing features in provided data: {missing_features}")
        return pd.DataFrame({"Error": [f"Missing features: {missing_features}"]})
    
    for col, le in lbl_encoders.items():
        if col in data_transformed.columns:
            data_transformed.loc[:, col] = data_transformed[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
    data_features = data_transformed[feature_list].apply(pd.to_numeric, errors='coerce').astype(np.float32)
    probabilities = model.predict_proba(data_features)
    p_buy = probabilities[:, 1]
    threshold_buy = 0.55
    threshold_sell = 0.45
    fixed_quantity = 100
    actions = []
    quantities = []
    for prob in p_buy:
        if prob > threshold_buy:
            actions.append("BUY")
            quantities.append(fixed_quantity)
        elif prob < threshold_sell:
            actions.append("SELL")
            quantities.append(fixed_quantity)
        else:
            actions.append("HOLD")
            quantities.append(0)
    predicted_signal = model.predict(data_features)
    prediction_data["Predicted Signal"] = predicted_signal
    prediction_data["Buy Probability"] = p_buy
    prediction_data["Action"] = actions
    prediction_data["Quantity"] = quantities
    return prediction_data

def transform_api_data(api_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw API data into the format expected by the ML model.
    
    Steps:
      - Rename columns to match training data.
      - Use "Last Closing Price" as 'close' if available; otherwise, use "Adjusted Closing Price".
      - Ensure numeric conversion for price-related columns.
      - Compute derived features: ma_5, ma_20, volatility_10 from 'adj._close'.
      - For any columns expected in training but missing in the API data, add them with default values.
    """
    df = api_df.copy()
    if df.empty:
        return df  # Return empty DataFrame if no data is available.
    
    # Rename columns to match training
    rename_dict = {
        "Date": "date",
        "Dividend Paid": "dividend",
        "Common Shares Outstanding": "shares_outstanding",
        "Adjusted Closing Price": "adj._close",
        "Highest Price": "high",
        "Lowest Price": "low",
        "Opening Price": "open",
        "Trading Volume": "volume"
    }
    df.rename(columns=rename_dict, inplace=True)
    
    # Ensure 'ticker' is present. If not, add it (here we assume the ticker from API wrapper was intended)
    if "ticker" not in df.columns:
        # As a fallback, set to an empty string (or you could pass ticker as a parameter)
        df["ticker"] = ""
    
    # Set 'close': prefer "Last Closing Price" if available.
    if "Last Closing Price" in api_df.columns:
        df["close"] = api_df["Last Closing Price"]
    else:
        df["close"] = df["adj._close"]
    
    # Convert numeric columns
    numeric_cols = ["open", "high", "low", "close", "adj._close", "volume", "dividend", "shares_outstanding"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Convert date and sort
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    
    # Compute derived features if 'adj._close' exists
    if "adj._close" in df.columns:
        df["ma_5"] = df["adj._close"].rolling(window=5, min_periods=1).mean()
        df["ma_20"] = df["adj._close"].rolling(window=20, min_periods=1).mean()
        df["volatility_10"] = df["adj._close"].rolling(window=10, min_periods=1).std()
    else:
        # If missing, add the columns with default NaN values
        df["ma_5"] = np.nan
        df["ma_20"] = np.nan
        df["volatility_10"] = np.nan
    
    # Fill any remaining NaN values
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)
    
    # Add default values for additional columns expected by training:
    for col, default in {
        "simfinid_x": 0,
        "simfinid_y": 0,
        "industryid": "Unknown Industry",
        "end_of_financial_year_(month)": 12
    }.items():
        if col not in df.columns:
            df[col] = default
    
    return df

# -------------------------------------------------------------
# MAIN: Train the model only if this script is executed directly.
# -------------------------------------------------------------
if __name__ == "__main__":
    train_model()