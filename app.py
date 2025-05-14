import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from utils.models import (
    create_features,
    train_random_forest,
    train_xgboost,
    tune_xgboost,
    prepare_lstm_data,
    train_lstm,
    predict_next_steps
)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- App Configuration ---
st.set_page_config(page_title="📈 Stock Predictor", layout="wide")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("merged_data.csv", parse_dates=['Datetime'])
    return df

merged_df = load_data()

# --- Sidebar Controls ---
page = st.sidebar.selectbox("Choose a view", ["Data Insight", "Investor"])

# --- Feature Engineering ---
X, y = create_features(merged_df)

# --- Model Training ---
rf_model, X_rf_test, y_rf_test, y_rf_pred = train_random_forest(X, y)
xgb_model, X_xgb_test, y_xgb_test, y_xgb_pred = train_xgboost(X, y)
best_xgb = tune_xgboost(X, y)

split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]
y_pred_tuned = best_xgb.predict(X_test)

X_lstm, y_lstm, scaler_y = prepare_lstm_data(merged_df)
lstm_model, y_lstm_test, y_lstm_pred = train_lstm(X_lstm, y_lstm)
y_lstm_test = scaler_y.inverse_transform(y_lstm_test)
y_lstm_pred = scaler_y.inverse_transform(y_lstm_pred)

# Align LSTM and XGBoost for ensemble
min_len = min(len(y_pred_tuned), len(y_lstm_pred))
y_pred_tuned = y_pred_tuned[-min_len:]
y_lstm_pred = y_lstm_pred[-min_len:]
y_test_ens = y_test[-min_len:]
ensemble_pred = (y_lstm_pred.flatten() + y_pred_tuned.flatten()) / 2

# --- Evaluation Function ---
def evaluate(y_true, y_pred):
    return (
        np.sqrt(mean_squared_error(y_true, y_pred)),
        mean_absolute_error(y_true, y_pred),
        r2_score(y_true, y_pred)
    )

rmse_rf, mae_rf, r2_rf = evaluate(y_rf_test, y_rf_pred)
rmse_xgb, mae_xgb, r2_xgb = evaluate(y_xgb_test, y_xgb_pred)
rmse_tuned, mae_tuned, r2_tuned = evaluate(y_test[-len(y_pred_tuned):], y_pred_tuned)
rmse_lstm, mae_lstm, r2_lstm = evaluate(y_lstm_test, y_lstm_pred)
rmse_ens, mae_ens, r2_ens = evaluate(y_test_ens, ensemble_pred)

# --- Tab Views ---
if page == "Data Insight":
    st.title("🔍 Data Insight")

    st.subheader("📊 Performance Comparison Table")
    metrics_table = pd.DataFrame({
        'Model': ['Random Forest', 'XGBoost', 'Tuned XGBoost', 'LSTM', 'Ensemble'],
        'RMSE': [rmse_rf, rmse_xgb, rmse_tuned, rmse_lstm, rmse_ens],
        'MAE': [mae_rf, mae_xgb, mae_tuned, mae_lstm, mae_ens],
        'R²': [r2_rf, r2_xgb, r2_tuned, r2_lstm, r2_ens]
    })
    st.dataframe(metrics_table.style.format({col: "{:.4f}" for col in ['RMSE', 'MAE', 'R²']}))

    st.subheader("📈 XGBoost Feature Importance")
    importance_df = pd.DataFrame({
        'Feature': best_xgb.get_booster().feature_names,
        'Importance': best_xgb.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    fig_imp = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title="Feature Importance")
    st.plotly_chart(fig_imp)

    st.subheader("🔗 Correlation Matrix")
    corr = merged_df[['price', 'sma_14', 'ema_14', 'rsi_14', 'macd_diff', 'sentiment_smoothed']].corr()
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Matrix")
    st.plotly_chart(fig_corr)

elif page == "Investor":
    st.title("💰 Investor Dashboard")

    st.subheader("📈 Price Chart")
    fig_price = go.Figure()

    # Candlestick chart
    fig_price.add_trace(go.Candlestick(
        x=merged_df['Datetime'],
        open=merged_df['Open'],
        high=merged_df['High'],
        low=merged_df['Low'],
        close=merged_df['price'],
        name='Actual Price'
    ))

    # Forecast overlay
    next_prices = predict_next_steps(merged_df, best_xgb, steps=4)
    forecast_times = pd.date_range(start=merged_df['Datetime'].iloc[-1], periods=5, freq='15min')[1:]

    fig_price.add_trace(go.Scatter(
        x=forecast_times,
        y=next_prices,
        mode='lines+markers+text',
        name='Predicted Price',
        line=dict(color='magenta', dash='dash'),
        marker=dict(size=8),
        text=[f"${p:.2f}" for p in next_prices],
        textposition="top center"
    ))

    fig_price.update_layout(
        title="📉 Actual vs Predicted Prices (Next 1 Hour)",
        xaxis_title="Datetime",
        yaxis_title="Price",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig_price, use_container_width=True)

    st.subheader("🔮 Next-Hour Forecast")
    forecast_df = pd.DataFrame({
        'Interval': ['+15 min', '+30 min', '+45 min', '+60 min'],
        'Forecasted Price': [f"${p:.2f}" for p in next_prices],
        'Confidence (R²)': [f"{r2_tuned:.3f}"] * 4
    })
    st.table(forecast_df)

    st.subheader("💵 ROI Calculator")
    buy = st.number_input("Buy Price", value=float(merged_df['price'].iloc[-1]))
    sell = st.number_input("Sell Price", value=float(merged_df['price'].iloc[-1]))
    qty = st.number_input("Quantity", value=1)
    if st.button("Calculate Profit/Loss"):
        profit = (sell - buy) * qty
        pct = (sell - buy) / buy * 100
        st.write(f"**Profit/Loss:** ${profit:.2f} ({pct:.2f}%)")
