import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from file import load_and_preprocess_data, train_models, add_technical_indicators

# -----------------------
# Load models if available
# -----------------------
@st.cache_resource
def load_models():
    try:
        rf_model = joblib.load("rf_model.pkl")
        lr_model = joblib.load("lr_model.pkl")
        return rf_model, lr_model
    except:
        return None, None

# -----------------------
# Main Streamlit App
# -----------------------
def main():
    st.title("📈 Long-Term Stock Price Prediction App")

    st.sidebar.header("Upload Options")
    uploaded_file = st.sidebar.file_uploader("Upload your stock CSV", type=["csv"])

    # Default dataset path (keep in same folder as app.py)
    default_stock_path = "stock_data.csv"

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.success("✅ Custom stock dataset uploaded successfully!")
    else:
        data = pd.read_csv(default_stock_path)
        st.info("ℹ️ Using default dataset (stock_data.csv).")

    # -----------------------
    # Data Preprocessing
    # -----------------------
    st.subheader("🔍 Data Preview")
    st.write(data.head())

    # Preprocess and add indicators
    data = load_and_preprocess_data(data)
    data = add_technical_indicators(data)

    # -----------------------
    # Exploratory Data Analysis
    # -----------------------
    st.subheader("📊 Exploratory Data Analysis")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data["Date"], data["Close"], label="Closing Price", color="blue")
    ax.set_title("Stock Closing Price Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

    # Correlation heatmap
    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # -----------------------
    # Train Models
    # -----------------------
    st.subheader("⚙️ Model Training")

    rf_model, lr_model = load_models()

    if rf_model is None or lr_model is None:
        st.warning("No trained models found. Training new models...")
        rf_model, lr_model, X_test, y_test, y_pred_rf, y_pred_lr = train_models(data)
        joblib.dump(rf_model, "rf_model.pkl")
        joblib.dump(lr_model, "lr_model.pkl")
        st.success("✅ Models trained and saved successfully!")
    else:
        st.success("✅ Pre-trained models loaded!")
        _, _, X_test, y_test, y_pred_rf, y_pred_lr = train_models(data, rf_model, lr_model)

    # -----------------------
    # Model Evaluation
    # -----------------------
    st.subheader("📉 Model Evaluation Metrics")

    metrics = {
        "Random Forest": {
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_rf)),
            "MAPE": mean_absolute_percentage_error(y_test, y_pred_rf),
            "R²": r2_score(y_test, y_pred_rf),
        },
        "Linear Regression": {
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_lr)),
            "MAPE": mean_absolute_percentage_error(y_test, y_pred_lr),
            "R²": r2_score(y_test, y_pred_lr),
        },
    }

    st.write(pd.DataFrame(metrics).T)

    # -----------------------
    # Prediction Visualization
    # -----------------------
    st.subheader("📈 Predicted vs Actual Prices")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(y_test.index, y_test.values, label="Actual", color="black")
    ax.plot(y_test.index, y_pred_rf, label="Random Forest Predictions", color="green")
    ax.plot(y_test.index, y_pred_lr, label="Linear Regression Predictions", color="red")
    ax.set_title("Predicted vs Actual Stock Prices")
    ax.legend()
    st.pyplot(fig)

    # -----------------------
    # Forecast Future Prices
    # -----------------------
    st.subheader("🔮 Forecast Future Prices (Next 180 Days)")
    future = pd.DataFrame(X_test.tail(1).values, columns=X_test.columns)
    forecasts = []

    for i in range(180):
        next_pred = rf_model.predict(future)[0]
        forecasts.append(next_pred)

        # Shift future input (simulate time step)
        future = pd.DataFrame(future.values, columns=future.columns)
        future["Close_lag1"] = next_pred

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, 181), forecasts, label="Forecast (RF)", color="blue")
    ax.set_title("Future Stock Price Forecast (180 Days)")
    ax.set_xlabel("Days Ahead")
    ax.set_ylabel("Predicted Price")
    ax.legend()
    st.pyplot(fig)

if __name__ == "__main__":
    main()
