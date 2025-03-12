import pandas as pd
import numpy as np
import logging
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import random

# -------------------------------------------------------------
# SETUP LOGGING
# -------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def train_model():
    """
    Loads preprocessed train and test datasets, trains an XGBoost model for next-day price movement
    with a +1% threshold, does hyperparameter tuning, evaluates the model, and saves the best model.
    """
    # -------------------------------------------------------------
    # 1) LOAD DATA
    # -------------------------------------------------------------
    logging.info("📂 Loading preprocessed training and testing datasets...")
    train_df = pd.read_csv("cleaned_stock_data_train.csv", low_memory=False)
    test_df = pd.read_csv("cleaned_stock_data_test.csv", low_memory=False)

    # Basic checks
    required_cols = {"date", "ticker", "close"}
    if not required_cols.issubset(train_df.columns) or not required_cols.issubset(test_df.columns):
        raise KeyError(f"Missing {required_cols} in train/test data.")

    # -------------------------------------------------------------
    # 2) PRESERVE DATE, TICKER, CLOSE FOR LATER MERGE
    # -------------------------------------------------------------
    train_dates = train_df[["date", "ticker", "close"]].copy()
    test_dates  = test_df[["date", "ticker", "close"]].copy()

    # -------------------------------------------------------------
    # 3) DEFINE NEW TARGET (+1% PRICE MOVEMENT)
    #    We label 1 if next day's close is >= 1% above today's close, else 0.
    # -------------------------------------------------------------
    logging.info("⚙️ Defining new target with a +1% threshold for upward movement...")
    train_df["price_movement"] = ((train_df["close"].shift(-1) - train_df["close"]) / train_df["close"] >= 0.01).astype(int)
    test_df["price_movement"]  = ((test_df["close"].shift(-1) - test_df["close"]) / test_df["close"] >= 0.01).astype(int)

    # Remove rows with NaNs (due to shifting)
    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)

    # -------------------------------------------------------------
    # 4) SELECT FEATURES (EXCLUDE date, ticker, etc.)
    # -------------------------------------------------------------
    remove_cols = ["ticker", "date", "company_name", "isin", "market", "main_currency", "price_movement"]
    # If "close" is not used as a feature, you can remove it from the model inputs
    # but we keep it to let the model see today's close price. Adjust as you prefer.
    features = [c for c in train_df.columns if c not in remove_cols]

    logging.info(f"🔎 Using {len(features)} features: {features}")

    # -------------------------------------------------------------
    # 5) LABEL ENCODE ANY CATEGORICAL COLUMNS
    # -------------------------------------------------------------
    logging.info("🔄 Converting categorical columns to numeric format...")
    categorical_cols = train_df.select_dtypes(include=["object"]).columns
    label_encoders = {}
    for col in categorical_cols:
        if col not in remove_cols:  # Only encode columns we use as features
            le = LabelEncoder()
            train_df[col] = train_df[col].astype(str)
            test_df[col]  = test_df[col].astype(str)

            # Fit on train, transform train
            le.fit(train_df[col])
            train_df[col] = le.transform(train_df[col])

            # Map test set; unseen classes become -1
            mapping = {cat: idx for idx, cat in enumerate(le.classes_)}
            test_df[col] = test_df[col].apply(lambda x: mapping[x] if x in mapping else -1)
            label_encoders[col] = le

    # -------------------------------------------------------------
    # 6) SPLIT INTO X, Y
    # -------------------------------------------------------------
    X_train = train_df[features]
    y_train = train_df["price_movement"]
    X_test  = test_df[features]
    y_test  = test_df["price_movement"]

    logging.info(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

    # -------------------------------------------------------------
    # 7) HYPERPARAMETER TUNING WITH RANDOMIZED SEARCH
    # -------------------------------------------------------------
    logging.info("🔍 Starting RandomizedSearchCV for XGBoost hyperparameters...")

    xgb_clf = xgb.XGBClassifier(objective='binary:logistic', n_jobs=-1, eval_metric='logloss', seed=42)

    param_dist = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    # We'll do a quick random search with 10 iterations and 3-fold CV
    rand_search = RandomizedSearchCV(
        estimator=xgb_clf,
        param_distributions=param_dist,
        n_iter=10,
        scoring='accuracy',  # you could use 'f1' or 'roc_auc'
        cv=3,
        verbose=1,
        random_state=42
    )

    rand_search.fit(X_train, y_train)
    best_model = rand_search.best_estimator_
    logging.info(f"✅ Best parameters found: {rand_search.best_params_}")

    # -------------------------------------------------------------
    # 8) EVALUATE THE MODEL
    # -------------------------------------------------------------
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"📉 XGBoost Accuracy: {accuracy:.4f}")

    # -------------------------------------------------------------
    # 9) SAVE THE BEST MODEL
    # -------------------------------------------------------------
    joblib.dump((best_model, features, label_encoders), "best_trading_model.pkl")
    logging.info("✅ Model saved successfully as best_trading_model.pkl")

def predict_signals(data: pd.DataFrame) -> pd.DataFrame:
    """
    Predict trading signals using the saved XGBoost model and apply a simple trading strategy.

    Strategy with updated thresholds:
      - If probability > 0.52: signal "BUY"
      - If probability < 0.48: signal "SELL"
      - Otherwise: "HOLD"

    Returns a DataFrame with columns:
      - date, ticker, close (copied from input if present)
      - Predicted Signal (0 or 1)
      - Buy Probability (class=1)
      - Action (BUY, SELL, or HOLD)
      - Quantity (fixed at 100 shares for demonstration)
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

    # 1) Transform the data so that it has all columns the model expects
    data_transformed = transform_api_data(data)

    # 2) Preserve essential columns for later merge
    #    We'll keep them if they exist in data_transformed
    keep_cols = ["date", "ticker", "close"]
    prediction_data = pd.DataFrame()
    for col in keep_cols:
        if col in data_transformed.columns:
            prediction_data[col] = data_transformed[col]

    # 3) Check that all required features are present
    missing_features = [f for f in feature_list if f not in data_transformed.columns]
    if missing_features:
        msg = f"❌ Missing features in provided data: {missing_features}"
        logging.error(msg)
        return pd.DataFrame({"Error": [msg]})

    # 4) Apply label encoders if needed
    for col, le in lbl_encoders.items():
        if col in data_transformed.columns:
            data_transformed[col] = data_transformed[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )

    # 5) Prepare feature matrix
    X_live = data_transformed[feature_list].apply(pd.to_numeric, errors='coerce').astype(np.float32)

    # 6) Predict probabilities
    probabilities = model.predict_proba(X_live)
    p_buy = probabilities[:, 1]  # Probability of class=1
    predicted_signal = model.predict(X_live)

    # 7) Define thresholds for BUY/SELL
    threshold_buy = 0.52
    threshold_sell = 0.48
    fixed_quantity = 1

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

    # 8) Construct the output
    prediction_data["Predicted Signal"] = predicted_signal
    prediction_data["Buy Probability"] = p_buy
    prediction_data["Action"] = actions
    prediction_data["Quantity"] = quantities

    return prediction_data

def transform_api_data(api_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw API data (or local data) into the format expected by the ML model.

    Steps:
      - Rename columns to match training data (if needed).
      - Compute rolling features (MA, volatility, etc.) if not already present.
      - Fill missing data.
    """
    df = api_df.copy()
    if df.empty:
        return df  # Return empty DataFrame if no data is available.

    # Example rename dict if data has these columns:
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

    # If "Last Closing Price" exists, set it as 'close'
    if "Last Closing Price" in df.columns:
        df["close"] = df["Last Closing Price"]

    # If 'close' is still missing, fallback to 'adj._close'
    if "close" not in df.columns and "adj._close" in df.columns:
        df["close"] = df["adj._close"]

    # Ensure columns exist
    numeric_cols = ["open", "high", "low", "close", "adj._close", "volume", "dividend", "shares_outstanding"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date", inplace=True)

    # Compute rolling features if desired
    if "adj._close" in df.columns:
        df["ma_5"] = df["adj._close"].rolling(window=5, min_periods=1).mean()
        df["ma_20"] = df["adj._close"].rolling(window=20, min_periods=1).mean()
        df["volatility_10"] = df["adj._close"].rolling(window=10, min_periods=1).std()
    else:
        # Add placeholders if missing
        df["ma_5"] = np.nan
        df["ma_20"] = np.nan
        df["volatility_10"] = np.nan

    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)

    # Add default columns if your model expects them
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
# MAIN: Train the model only if this script is executed directly
# -------------------------------------------------------------
if __name__ == "__main__":
    train_model()