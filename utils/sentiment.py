# utils/sentiment.py

from transformers import BertTokenizer, BertForSequenceClassification, TextClassificationPipeline
import pandas as pd

tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")
finbert = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)

def get_sentiment_scores(headlines):
    results = []
    for text in headlines:
        prediction = finbert(text)[0]
        top = max(prediction, key=lambda x: x['score'])
        score = {
            'label': top['label'],
            'confidence': round(top['score'], 4),
            'sentiment_score': {
                'positive': 1,
                'neutral': 0,
                'negative': -1
            }[top['label']] * top['score']
        }
        results.append(score)
    return pd.DataFrame(results)
