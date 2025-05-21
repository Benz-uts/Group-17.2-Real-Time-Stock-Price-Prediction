# Real-Time Stock Price Prediction Using Machine Learning and Sentiment Analysis

This repository contains the source code and documentation for a course project titled **"Real-Time Stock Price Prediction"**. The project aims to predict short-term stock prices using a combination of machine learning models, deep learning, technical indicators, and sentiment analysis derived from financial news.

---

## Course Information

- **Subject**: 36127 Innovation Lab: Capstone Project - Autumn 2025
- **Team**: Group 17-2  
- **University**: University of Technology Sydney

---

##  How to Run
### Step 1: Install Dependencies

pip install -r requirements.txt

### Step 2: Launch Streamlit Application

streamlit run app.py

---

## Project Objectives

- Integrate stock market data with financial news sentiment
- Extract meaningful features using technical indicators and natural language processing
- Build and evaluate predictive models (Random Forest, XGBoost, LSTM)
- Compare model performance and provide interpretability
- Deliver a user-friendly Streamlit dashboard for interactive analysis

---

## 🗂️ Directory Structure

.
├── app.py # Streamlit web app
├── main.py # Script for model training & evaluation
├── merged_data.csv # Combined dataset with indicators and sentiment
├── requirements.txt # Python dependency list
├── utils/
│ ├── init.py
│ ├── data_loader.py # Downloads stock + news data
│ ├── features.py # Merges technical indicators with sentiment
│ ├── models.py # Model training and prediction functions
│ └── sentiment.py # FinBERT-based sentiment scoring

---

## Models Implemented

| Model         | 
|---------------|
| **Random Forest**  | 
| **XGBoost**        |
| **Tuned XGBoost**  |
| **LSTM**           | 
| **Ensemble**       | 

---

## Features Used

- **Lagged Prices**: Previous price values as predictors
- **Technical Indicators**: SMA, EMA, RSI, MACD
- **Sentiment Scores**: Derived from FinBERT on financial headlines
- **Smoothed Sentiment**: Rolling window average to reduce noise

---

## Evaluation Metrics

- **RMSE**
- **MAE**
- **R² Score**

---
