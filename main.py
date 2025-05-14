# main.py using merged_data.csv directly

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from utils.models import (
    create_features,
    train_random_forest,
    train_xgboost,
    tune_xgboost,
    prepare_lstm_data,
    train_lstm,
    plot_predictions,
    predict_next_steps
)

# Load pre-merged data
print("📂 Loading merged dataset from CSV...")
merged_df = pd.read_csv("merged_data.csv", parse_dates=['Datetime'])

print("✅ Loaded merged_data.csv with shape:", merged_df.shape)

# -------- Feature Engineering --------
X, y = create_features(merged_df)

import seaborn as sns
import matplotlib.pyplot as plt

# ---- Correlation Matrix ----
print("\n📊 Correlation Matrix (Feature vs Price):")
corr_df = merged_df.copy()
X_corr, y_corr = create_features(corr_df)
corr_df = X_corr.copy()
corr_df['price'] = y_corr
correlation_matrix = corr_df.corr()
print(correlation_matrix['price'].sort_values(ascending=False))

# Optional: Visual heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix with Price")
plt.tight_layout()
plt.savefig("correlation_matrix.png")  # Save for presentation
plt.show()


# -------- Random Forest --------
rf_model, X_test_rf, y_test_rf, y_pred_rf = train_random_forest(X, y)
plot_predictions(y_test_rf, y_pred_rf, title="Random Forest: Actual vs Predicted")

# -------- XGBoost --------
xgb_model, X_test_xgb, y_test_xgb, y_pred_xgb = train_xgboost(X, y)
plot_predictions(y_test_xgb, y_pred_xgb, title="XGBoost: Actual vs Predicted")

# -------- Tuned XGBoost --------
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

best_xgb = tune_xgboost(X_train, y_train)
y_pred_best = best_xgb.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred_best))
mae = mean_absolute_error(y_test, y_pred_best)
r2 = r2_score(y_test, y_pred_best)

print("\n📊 Tuned XGBoost:")
print(f"🔹 RMSE: {rmse:.4f}")
print(f"🔹 MAE:  {mae:.4f}")
print(f"🔹 R²:   {r2:.4f}")

plot_predictions(y_test, y_pred_best, title="Tuned XGBoost: Actual vs Predicted")

from xgboost import plot_importance

# ---- XGBoost Feature Importance ----
plt.figure(figsize=(8, 6))
plot_importance(best_xgb, max_num_features=10, height=0.5)
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.savefig("xgboost_feature_importance.png")  # Save for presentation
plt.show()


# -------- LSTM --------
X_lstm, y_lstm, scaler_y = prepare_lstm_data(merged_df)
lstm_model, y_test_lstm, y_pred_lstm = train_lstm(X_lstm, y_lstm)

# Inverse transform
y_pred_lstm = scaler_y.inverse_transform(y_pred_lstm)
y_test_lstm = scaler_y.inverse_transform(y_test_lstm)

# -------------------------
# Evaluate LSTM
# -------------------------
rmse_lstm = np.sqrt(mean_squared_error(y_test_lstm, y_pred_lstm))
mae_lstm = mean_absolute_error(y_test_lstm, y_pred_lstm)
r2_lstm = r2_score(y_test_lstm, y_pred_lstm)

print("\n📊 Model Evaluation (LSTM):")
print(f"🔹 RMSE: {rmse_lstm:.4f}")
print(f"🔹 MAE:  {mae_lstm:.4f}")
print(f"🔹 R²:   {r2_lstm:.4f}")

def plot_lstm_predictions(y_true, y_pred, title='LSTM: Actual vs Predicted'):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual')
    plt.plot(y_pred, label='Predicted', linestyle='--')
    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("lstm_actual_vs_predicted.png")
    plt.show()

plot_lstm_predictions(y_test_lstm, y_pred_lstm)

# -------- Ensemble: XGBoost + LSTM --------
min_len = min(len(y_pred_lstm), len(y_pred_best))
ensemble_pred = (y_pred_lstm[-min_len:].flatten() + y_pred_best[-min_len:].flatten()) / 2
y_test_ensemble = y_test[-min_len:]

rmse = np.sqrt(mean_squared_error(y_test_ensemble, ensemble_pred))
mae = mean_absolute_error(y_test_ensemble, ensemble_pred)
r2 = r2_score(y_test_ensemble, ensemble_pred)

print("\n🤝 Ensemble (LSTM + XGBoost):")
print(f"🔹 RMSE: {rmse:.4f}")
print(f"🔹 MAE:  {mae:.4f}")
print(f"🔹 R²:   {r2:.4f}")

plot_predictions(y_test_ensemble, ensemble_pred, title="LSTM + XGBoost Ensemble")

# -------- Predict Next 4 Steps --------
next_preds = predict_next_steps(merged_df, best_xgb, steps=4)
print("\n🔮 Next Price Predictions:")
for i, p in enumerate(next_preds, 1):
    print(f"In {i*15} minutes: ${p:.2f}")
