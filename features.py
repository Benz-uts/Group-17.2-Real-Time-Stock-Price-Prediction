import pandas as pd

def merge_price_with_sentiment(stock_df, sentiment_df):
    # work on copies
    stock = stock_df.copy()
    sent  = sentiment_df.copy()

    # make sure the stock-datetime is timezone-naive
    stock['Datetime'] = pd.to_datetime(stock['Datetime']).dt.tz_localize(None)

    # if we have no sentiment columns, add them with defaults
    for col, default in [
        ('datetime',      pd.NaT),
        ('headline',      ''),
        ('label',         'neutral'),
        ('confidence',    0.0),
        ('sentiment_score', 0.0)
    ]:
        if col not in sent.columns:
            sent[col] = default

    # drop any rows where datetime or headline is null
    sent.dropna(subset=['datetime','headline'], inplace=True)

    # parse and sort
    sent['datetime'] = pd.to_datetime(sent['datetime']).dt.tz_localize(None)
    stock = stock.sort_values('Datetime')
    sent  = sent.sort_values('datetime')

    # now asof‐merge, pulling the last headline before each price point
    merged = pd.merge_asof(
        stock,
        sent[['datetime','headline','label','confidence','sentiment_score']],
        left_on='Datetime',
        right_on='datetime',
        direction='backward'
    )

    # clean up
    merged.drop(columns=['datetime'], inplace=True)
    merged['sentiment_score'] = merged['sentiment_score'].fillna(0.0)
    merged['confidence']      = merged['confidence'].fillna(0.0)
    merged['headline']        = merged['headline'].fillna('No related news')
    merged['label']           = merged['label'].fillna('neutral')

    # add a little smoothing so single stray headlines don’t dominate
    merged['sentiment_smoothed'] = (
        merged['sentiment_score']
              .rolling(window=3, min_periods=1)
              .mean()
    )

    return merged