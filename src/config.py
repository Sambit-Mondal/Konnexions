"""
Configuration module for the Sentiment Surge project.
"""

import os

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Default target stocks
DEFAULT_TARGET_STOCKS = ['TSLA', 'NVDA', 'AAPL', 'MSFT', 'GOOG']

# Target stocks to analyze (can be overridden via command line)
TARGET_STOCKS = DEFAULT_TARGET_STOCKS.copy()

# API keys (replace with your own in production)
ALPHA_VANTAGE_API_KEY = "6TC7FSSBMCUOCWMY"  # Sample key, replace with your own

# Sentiment analysis parameters
SENTIMENT_THRESHOLDS = {
    'positive': 0.25,
    'negative': -0.25
}

# Model parameters
MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42
    }
}

# Evaluation parameters
TRAIN_TEST_SPLIT_RATIO = 0.8
