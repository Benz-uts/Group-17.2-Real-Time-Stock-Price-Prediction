# utils/data_loader.py
import os
import yfinance as yf
import finnhub
import pandas as pd
import datetime
import ta

def get_stock_data(ticker='AAPL', interval='1h', period='1y'):
    df = yf.download(ticker, interval=interval, period=period)

    if df.empty:
        raise ValueError("❌ Stock data download failed or API limit reached.")

    # Reset index to bring Datetime into column
    df = df.reset_index()

    # Rename the datetime column correctly (check first column name to confirm)
    if 'Date' in df.columns:
        df.rename(columns={'Date': 'Datetime'}, inplace=True)
    elif 'Datetime' not in df.columns:
        raise KeyError("❌ No datetime column found after reset_index")

    # Parse datetime
    df['Datetime'] = pd.to_datetime(df['Datetime']).dt.tz_localize(None)

    # Rename and compute indicators
    df.rename(columns={'Close': 'price'}, inplace=True)

    # Add technical indicators
    df['sma_14'] = ta.trend.sma_indicator(df['price'], window=14)
    df['ema_14'] = ta.trend.ema_indicator(df['price'], window=14)
    df['rsi_14'] = ta.momentum.rsi(df['price'], window=14)
    df['macd_diff'] = ta.trend.macd_diff(df['price'])

    # Drop missing rows caused by rolling windows
    df.dropna(subset=['sma_14', 'ema_14', 'rsi_14', 'macd_diff'], inplace=True)

    print("📊 Stock with indicators:", df.columns.tolist())
    return df


def get_news_data(finnhub_api_key, symbol='AAPL', days_back=365):
    import datetime, pandas as pd, finnhub

    client = finnhub.Client(api_key=finnhub_api_key)
    today = datetime.date.today()
    past  = today - datetime.timedelta(days=days_back)

    all_news = []
    delta    = datetime.timedelta(days=7)
    start_dt = past

    while start_dt < today:
        frm = start_dt.strftime("%Y-%m-%d")
        to  = (start_dt + delta).strftime("%Y-%m-%d")
        try:
            batch = client.company_news(symbol, _from=frm, to=to)
            all_news.extend(batch)
        except Exception as e:
            print(f"⚠️ {e} on {frm}→{to}")
        start_dt += delta

    # build DataFrame
    df = pd.DataFrame(all_news)

    # guard against empty / malformed results
    if df.empty or 'datetime' not in df.columns or 'headline' not in df.columns:
        print("⚠️ No valid news returned, returning empty DataFrame")
        return pd.DataFrame(columns=['datetime', 'headline'])

    # parse and clean
    df['datetime'] = pd.to_datetime(df['datetime'], unit='s', errors='coerce')
    df.dropna(subset=['datetime','headline'], inplace=True)

    # only keep company‐related items
    if 'related' in df.columns:
        df = df[df['related']==symbol]

    # final output
    return df[['datetime','headline']]