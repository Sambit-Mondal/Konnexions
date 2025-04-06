"""
Script to perform sentiment analysis on financial news data.
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import traceback

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import project configuration
from src.config import TARGET_STOCKS, DATA_DIR

def perform_sentiment_analysis():
    """Perform sentiment analysis on collected financial news data."""
    print(f"Performing sentiment analysis for: {', '.join(TARGET_STOCKS)}")
    
    # Create sentiment directory if it doesn't exist
    sentiment_dir = os.path.join(DATA_DIR, 'sentiment')
    os.makedirs(sentiment_dir, exist_ok=True)
    
    # Process each target stock
    for symbol in TARGET_STOCKS:
        print(f"\nAnalyzing sentiment for {symbol}...")
        
        try:
            # Load news data
            news_path = os.path.join(DATA_DIR, 'news', f"{symbol}_news.csv")
            
            # If news data doesn't exist, try sample data
            if not os.path.exists(news_path):
                sample_path = os.path.join(DATA_DIR, 'news', f"{symbol}_news_sample.csv")
                if os.path.exists(sample_path):
                    import shutil
                    shutil.copy(sample_path, news_path)
                    print(f"Using sample news data for {symbol}")
                else:
                    print(f"No news data found for {symbol}")
                    continue
            
            # Load news data
            news_df = pd.read_csv(news_path)
            
            if news_df.empty:
                print(f"News data is empty for {symbol}")
                continue
            
            print(f"Loaded {len(news_df)} news items for {symbol}")
            
            # Check if sentiment scores are already in the data
            if 'overall_sentiment_score' in news_df.columns and 'ticker_sentiment_score' in news_df.columns:
                print(f"Sentiment scores already exist in the data for {symbol}")
                
                # Map sentiment scores to categories
                news_df['sentiment_category'] = news_df['ticker_sentiment_score'].apply(categorize_sentiment)
                
                # Create sentiment summary
                create_sentiment_summary(news_df, symbol)
                
            else:
                print(f"Performing sentiment analysis for {symbol}...")
                
                # Analyze sentiment for each news item
                news_df['sentiment_score'] = news_df['summary'].apply(analyze_sentiment)
                news_df['sentiment_category'] = news_df['sentiment_score'].apply(categorize_sentiment)
                
                # Save processed data
                output_path = os.path.join(sentiment_dir, f"{symbol}_news_sentiment.csv")
                news_df.to_csv(output_path, index=False)
                print(f"Sentiment analysis saved to {output_path}")
                
                # Create sentiment summary
                create_sentiment_summary(news_df, symbol)
            
        except Exception as e:
            print(f"Error analyzing sentiment for {symbol}: {e}")
            print(traceback.format_exc())

def analyze_sentiment(text):
    """
    Analyze sentiment of a text using a simplified rule-based approach.
    In a real implementation, this would use a pre-trained LLM or transformer model.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        float: Sentiment score (-1 to 1)
    """
    if pd.isna(text) or text == '':
        return 0.0
    
    # Define simple positive and negative word lists
    positive_words = [
        'up', 'rise', 'rising', 'gain', 'gains', 'positive', 'bull', 'bullish',
        'growth', 'growing', 'increase', 'increasing', 'outperform', 'beat',
        'exceeded', 'strong', 'strength', 'opportunity', 'optimistic', 'good',
        'great', 'excellent', 'profit', 'profitable', 'success', 'successful'
    ]
    
    negative_words = [
        'down', 'fall', 'falling', 'drop', 'dropping', 'decline', 'declining',
        'loss', 'losses', 'negative', 'bear', 'bearish', 'weak', 'weakness',
        'underperform', 'miss', 'missed', 'below', 'concern', 'concerns',
        'risk', 'risks', 'risky', 'warning', 'bad', 'poor', 'trouble', 'fail'
    ]
    
    # Convert text to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    # Count occurrences of positive and negative words
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    # Calculate sentiment score
    total_count = positive_count + negative_count
    if total_count == 0:
        return 0.0  # Neutral
    
    return (positive_count - negative_count) / total_count

def categorize_sentiment(score):
    """
    Categorize sentiment score into positive, negative, or neutral.
    
    Args:
        score (float): Sentiment score
        
    Returns:
        str: Sentiment category
    """
    if pd.isna(score):
        return 'neutral'
    
    if score > 0.25:
        return 'positive'
    elif score < -0.25:
        return 'negative'
    else:
        return 'neutral'

def create_sentiment_summary(news_df, symbol):
    """
    Create a daily summary of sentiment scores.
    
    Args:
        news_df (pandas.DataFrame): DataFrame with sentiment analysis
        symbol (str): Stock symbol
    """
    try:
        # Ensure date column exists
        if 'published' in news_df.columns:
            # Convert to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(news_df['published']):
                news_df['date'] = pd.to_datetime(news_df['published'])
            else:
                news_df['date'] = news_df['published']
        else:
            # If no date column, use current date
            news_df['date'] = datetime.now()
        
        # Extract date only (no time)
        news_df['date'] = pd.to_datetime(news_df['date']).dt.date
        
        # Determine which sentiment score column to use
        if 'ticker_sentiment_score' in news_df.columns:
            score_col = 'ticker_sentiment_score'
        elif 'overall_sentiment_score' in news_df.columns:
            score_col = 'overall_sentiment_score'
        elif 'sentiment_score' in news_df.columns:
            score_col = 'sentiment_score'
        else:
            print(f"No sentiment score column found for {symbol}")
            return
        
        # Group by date and calculate average sentiment scores
        daily_sentiment = news_df.groupby('date').agg({
            score_col: 'mean',
            'sentiment_category': lambda x: x.value_counts().index[0]  # Most common category
        }).reset_index()
        
        # Rename columns for consistency
        daily_sentiment.rename(columns={
            score_col: 'sentiment_score',
            'sentiment_category': 'dominant_sentiment'
        }, inplace=True)
        
        # Add sentiment scores by category
        sentiment_counts = news_df.groupby(['date', 'sentiment_category']).size().unstack(fill_value=0)
        
        # If the expected columns don't exist, create them
        for category in ['positive', 'negative', 'neutral']:
            if category not in sentiment_counts.columns:
                sentiment_counts[category] = 0
        
        # Calculate percentage of each sentiment category
        sentiment_total = sentiment_counts.sum(axis=1)
        sentiment_pct = sentiment_counts.div(sentiment_total, axis=0)
        
        # Rename columns
        sentiment_pct.columns = [f'sentiment_{col}' for col in sentiment_pct.columns]
        
        # Reset index to merge with daily_sentiment
        sentiment_pct = sentiment_pct.reset_index()
        
        # Merge with daily_sentiment
        daily_sentiment = pd.merge(daily_sentiment, sentiment_pct, on='date', how='left')
        
        # Fill NaN values
        daily_sentiment.fillna(0, inplace=True)
        
        # Save daily sentiment summary
        output_path = os.path.join(DATA_DIR, 'sentiment', f"{symbol}_daily_sentiment.csv")
        daily_sentiment.to_csv(output_path, index=False)
        print(f"Daily sentiment summary saved to {output_path}")
        
        # Print summary statistics
        print(f"Sentiment summary for {symbol}:")
        print(f"Average sentiment score: {daily_sentiment['sentiment_score'].mean():.4f}")
        print(f"Dominant sentiment: {daily_sentiment['dominant_sentiment'].value_counts().to_dict()}")
        
        return daily_sentiment
        
    except Exception as e:
        print(f"Error creating sentiment summary for {symbol}: {e}")
        print(traceback.format_exc())
        return None

if __name__ == "__main__":
    # If command line arguments are provided, update TARGET_STOCKS
    if len(sys.argv) > 1:
        from src.config import DEFAULT_TARGET_STOCKS
        import src.config as config
        config.TARGET_STOCKS = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_TARGET_STOCKS
        print(f"Using command line symbols: {config.TARGET_STOCKS}")
    
    # Perform sentiment analysis
    perform_sentiment_analysis()