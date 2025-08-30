import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# -----------------------
# Load + Preprocess Data
# -----------------------
def load_and_preprocess_data(data):
    """Cleans stock dataset, converts dates, handles NaN, adds lag features"""
    
    # Ensure Date is datetime
    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
        data = data.dropna(subset=["Date"])
        data = data.sort_values("Date")

    # Ensure numeric columns
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna()

    # Add lag feature
    data["Close_lag1"] = data["Close"].shift(1)
    data = data.dropna()

    return data


# -----------------------
# Add Technical Indicators
# -----------------------
def add_technical_indicators(data):
    """Adds basic technical indicators like SMA and EMA"""

    data["SMA_10"] = data["Close"].rolling(window=10).mean()
    data["SMA_50"] = data["Close"].rolling(window=50).mean()
    data["EMA_10"] = data["Close"].ewm(span=10, adjust=False).mean()
    data["EMA_50"] = data["Close"].ewm(span=50, adjust=False).mean()

    # Drop rows with NaN due to rolling
    data = data.dropna()
    return data


# -----------------------
# Train Models
# -----------------------
def train_models(data, rf_model=None, lr_model=None):
    """Trains Random Forest and Linear Regression models"""

    # Features and target
    features = ["Open", "High", "Low", "Volume", "Close_lag1",
                "SMA_10", "SMA_50", "EMA_10", "EMA_50"]
    target = "Close"

    X = data[features]
    y = data[target]

    # Train/test split (time series -> no shuffle)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Train new models if not provided
    if rf_model is None:
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

    if lr_model is None:
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)

    # Predictions
    y_pred_rf = rf_model.predict(X_test)
    y_pred_lr = lr_model.predict(X_test)

    return rf_model, lr_model, X_test, y_test, y_pred_rf, y_pred_lr
