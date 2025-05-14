import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

def create_features(df, lags=4, include_sentiment=True):
    df = df.copy()

    # Create lag features
    for i in range(1, lags + 1):
        df[f'lag_{i}'] = df['price'].shift(i)

    df.dropna(inplace=True)

    # Base features (price + technical indicators)
    base_features = [f'lag_{i}' for i in range(1, lags + 1)] + ['sma_14', 'ema_14', 'rsi_14', 'macd_diff']
    
    # Add sentiment only if requested
    if include_sentiment:
        feature_cols = base_features + ['sentiment_smoothed']
    else:
        feature_cols = base_features

    # Drop rows with missing values in chosen columns
    df = df.dropna(subset=feature_cols + ['price'])

    X = df[feature_cols]
    y = df['price']
    return X, y



def train_random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n📊 Model Evaluation (Random Forest):")
    print(f"🔹 RMSE: {rmse:.4f}")
    print(f"🔹 MAE:  {mae:.4f}")
    print(f"🔹 R²:   {r2:.4f}")

    return model, X_test, y_test, y_pred


def train_xgboost(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n📊 Model Evaluation (XGBoost):")
    print(f"🔹 RMSE: {rmse:.4f}")
    print(f"🔹 MAE:  {mae:.4f}")
    print(f"🔹 R²:   {r2:.4f}")

    return model, X_test, y_test, y_pred


def plot_predictions(y_test, y_pred, title='Model Predictions'):
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test.values, label='Actual', linewidth=2)
    plt.plot(y_test.index, y_pred, label='Predicted', linestyle='--')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def tune_xgboost(X, y):
    print("🔍 Starting GridSearchCV for XGBoost...")

    tscv = TimeSeriesSplit(n_splits=5)

    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    model = XGBRegressor(random_state=42, objective='reg:squarederror')

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='neg_root_mean_squared_error',
        cv=tscv,
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X, y)

    print("✅ Best Parameters Found:")
    print(grid_search.best_params_)

    return grid_search.best_estimator_

def prepare_lstm_data(df, lags=10, include_sentiment=True):
    df = df.copy()

    # Create lag features
    for i in range(1, lags + 1):
        df[f'lag_{i}'] = df['price'].shift(i)

    base_features = [f'lag_{i}' for i in range(1, lags + 1)] + [
        'sma_14', 'ema_14', 'rsi_14', 'macd_diff'
    ]
    if include_sentiment:
        feature_cols = base_features + ['sentiment_smoothed']
    else:
        feature_cols = base_features

    df.dropna(subset=feature_cols + ['price'], inplace=True)

    X = df[feature_cols].values
    y = df['price'].values

    # Scale features
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

    X_seq, y_seq = [], []
    for i in range(lags, len(X_scaled)):
        X_seq.append(X_scaled[i - lags:i])
        y_seq.append(y_scaled[i])

    return np.array(X_seq), np.array(y_seq), scaler_y

def train_lstm(X, y, epochs=50, batch_size=16):
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential()
    model.add(LSTM(64, activation='relu', return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )

    y_pred = model.predict(X_test)
    return model, y_test, y_pred

def predict_next_steps(merged_df, model, steps=4, lags=4, include_sentiment=True):
    df = merged_df.copy()
    feature_cols = [f'lag_{i}' for i in range(1, lags + 1)] + [
        'sma_14', 'ema_14', 'rsi_14', 'macd_diff'
    ]
    if include_sentiment:
        feature_cols.append('sentiment_smoothed')

    last_rows = df.tail(lags).copy()
    lag_values = last_rows['price'].values[-lags:].tolist()
    indicators = df.iloc[-1][['sma_14', 'ema_14', 'rsi_14', 'macd_diff']]
    sentiment = df.iloc[-1]['sentiment_smoothed'] if include_sentiment else None

    preds = []
    for _ in range(steps):
        input_values = lag_values + indicators.tolist()
        if include_sentiment:
            input_values.append(sentiment)

        input_df = pd.DataFrame([input_values], columns=feature_cols)
        pred = model.predict(input_df)[0]
        preds.append(pred)

        # Update lag values for next step
        lag_values = lag_values[1:] + [pred]

    return preds
