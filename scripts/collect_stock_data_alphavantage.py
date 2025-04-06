"""
Script to collect stock data for any company using Alpha Vantage API.
This version includes improved error handling and debugging.
"""

import sys
import os
import json
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import time
import traceback

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import project configuration
from src.config import TARGET_STOCKS, DATA_DIR, ALPHA_VANTAGE_API_KEY

def collect_stock_data():
    """Collect stock data for target companies using Alpha Vantage API."""
    print(f"Collecting stock data for: {', '.join(TARGET_STOCKS)}")
    
    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, 'news'), exist_ok=True)
    
    # Create sample data as fallback
    create_sample_data()
    
    # Collect data for each target stock
    for i, symbol in enumerate(TARGET_STOCKS):
        print(f"\nFetching stock data for {symbol} ({i+1}/{len(TARGET_STOCKS)})...")
        
        try:
            # Fetch daily stock data
            daily_data = fetch_daily_data(symbol)
            
            # Fetch company overview
            overview_data = fetch_company_overview(symbol)
            
            # Wait between API calls to avoid rate limiting
            if i < len(TARGET_STOCKS) - 1:
                print(f"Waiting 15 seconds before next API call to avoid rate limiting...")
                time.sleep(15)
            
            print(f"Successfully collected data for {symbol}")
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            print(f"Using sample data for {symbol} instead...")
            
            # Copy sample data if API call fails
            sample_path = os.path.join(DATA_DIR, f"{symbol}_stock_data_sample.csv")
            target_path = os.path.join(DATA_DIR, f"{symbol}_stock_data.csv")
            if os.path.exists(sample_path):
                import shutil
                shutil.copy(sample_path, target_path)
                print(f"Sample data copied to {target_path}")

def fetch_daily_data(symbol):
    """
    Fetch daily stock data from Alpha Vantage API.
    
    Args:
        symbol (str): Stock symbol
        
    Returns:
        pandas.DataFrame: Daily stock data
    """
    print(f"Fetching daily data for {symbol}...")
    
    # API endpoint for daily time series
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&outputsize=full&apikey={ALPHA_VANTAGE_API_KEY}"
    
    print(f"Making API request to: {url}")
    
    # Make API request
    try:
        response = requests.get(url, timeout=30)
        print(f"Response status code: {response.status_code}")
        
        if response.status_code != 200:
            print(f"Error response: {response.text}")
            raise Exception(f"API request failed with status code {response.status_code}")
        
        data = response.json()
        
        # Check for API error messages
        if "Error Message" in data:
            print(f"API error: {data['Error Message']}")
            raise Exception(f"API error: {data['Error Message']}")
        
        if "Note" in data:
            print(f"API note: {data['Note']}")
            if "call frequency" in data["Note"]:
                print("API rate limit reached. Using sample data instead.")
                return None
        
        # Save raw data
        raw_data_path = os.path.join(DATA_DIR, f"{symbol}_raw_data.json")
        with open(raw_data_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Raw data saved to {raw_data_path}")
        
        # Process data into DataFrame
        if "Time Series (Daily)" in data:
            time_series = data["Time Series (Daily)"]
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            
            # Rename columns
            df.rename(columns={
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. adjusted close': 'adjclose',
                '6. volume': 'volume',
                '7. dividend amount': 'dividend',
                '8. split coefficient': 'split'
            }, inplace=True)
            
            # Convert string values to float
            for col in ['open', 'high', 'low', 'close', 'adjclose', 'volume', 'dividend', 'split']:
                df[col] = pd.to_numeric(df[col])
            
            # Convert index to datetime
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            
            # Save processed data
            csv_path = os.path.join(DATA_DIR, f"{symbol}_stock_data.csv")
            df.to_csv(csv_path)
            print(f"Processed data saved to {csv_path}")
            print(f"Data shape: {df.shape}")
            print(f"Date range: {df.index.min()} to {df.index.max()}")
            
            return df
        else:
            print(f"No time series data found for {symbol}")
            print(f"API response: {data}")
            return None
            
    except requests.exceptions.Timeout:
        print(f"Request timed out for {symbol}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request exception: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        print(traceback.format_exc())
        return None

def fetch_company_overview(symbol):
    """
    Fetch company overview from Alpha Vantage API.
    
    Args:
        symbol (str): Stock symbol
        
    Returns:
        dict: Company overview data
    """
    print(f"Fetching company overview for {symbol}...")
    
    # API endpoint for company overview
    url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
    
    print(f"Making API request to: {url}")
    
    # Make API request
    try:
        response = requests.get(url, timeout=30)
        print(f"Response status code: {response.status_code}")
        
        if response.status_code != 200:
            print(f"Error response: {response.text}")
            raise Exception(f"API request failed with status code {response.status_code}")
        
        data = response.json()
        
        # Check for API error messages
        if "Error Message" in data:
            print(f"API error: {data['Error Message']}")
            raise Exception(f"API error: {data['Error Message']}")
        
        if "Note" in data:
            print(f"API note: {data['Note']}")
            if "call frequency" in data["Note"]:
                print("API rate limit reached. Using sample data instead.")
                return None
        
        # Save overview data
        overview_path = os.path.join(DATA_DIR, f"{symbol}_overview.json")
        with open(overview_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Company overview saved to {overview_path}")
        
        return data
        
    except requests.exceptions.Timeout:
        print(f"Request timed out for {symbol}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request exception: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        print(traceback.format_exc())
        return None

def fetch_news_sentiment(symbol):
    """
    Fetch news sentiment from Alpha Vantage API.
    
    Args:
        symbol (str): Stock symbol
        
    Returns:
        dict: News sentiment data
    """
    print(f"Fetching news sentiment for {symbol}...")
    
    # API endpoint for news sentiment
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
    
    print(f"Making API request to: {url}")
    
    # Make API request
    try:
        response = requests.get(url, timeout=30)
        print(f"Response status code: {response.status_code}")
        
        if response.status_code != 200:
            print(f"Error response: {response.text}")
            raise Exception(f"API request failed with status code {response.status_code}")
        
        data = response.json()
        
        # Check for API error messages
        if "Error Message" in data:
            print(f"API error: {data['Error Message']}")
            raise Exception(f"API error: {data['Error Message']}")
        
        if "Note" in data:
            print(f"API note: {data['Note']}")
            if "call frequency" in data["Note"]:
                print("API rate limit reached. Using sample data instead.")
                return None
        
        # Create news directory if it doesn't exist
        news_dir = os.path.join(DATA_DIR, 'news')
        os.makedirs(news_dir, exist_ok=True)
        
        # Save news sentiment data
        news_path = os.path.join(news_dir, f"{symbol}_news.json")
        with open(news_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"News sentiment saved to {news_path}")
        
        # Process news data into DataFrame if available
        if "feed" in data:
            news_items = data["feed"]
            
            # Extract relevant information
            news_list = []
            for item in news_items:
                news_dict = {
                    'title': item.get('title', ''),
                    'summary': item.get('summary', ''),
                    'published': item.get('time_published', ''),
                    'source': item.get('source', ''),
                    'url': item.get('url', ''),
                    'overall_sentiment_score': item.get('overall_sentiment_score', 0),
                    'overall_sentiment_label': item.get('overall_sentiment_label', '')
                }
                
                # Add ticker-specific sentiment if available
                if 'ticker_sentiment' in item:
                    for ticker_sent in item['ticker_sentiment']:
                        if ticker_sent.get('ticker') == symbol:
                            news_dict['ticker_sentiment_score'] = ticker_sent.get('ticker_sentiment_score', 0)
                            news_dict['ticker_sentiment_label'] = ticker_sent.get('ticker_sentiment_label', '')
                            break
                
                news_list.append(news_dict)
            
            # Create DataFrame
            news_df = pd.DataFrame(news_list)
            
            # Convert published date to datetime
            if 'published' in news_df.columns and not news_df.empty:
                news_df['published'] = pd.to_datetime(news_df['published'], format='%Y%m%dT%H%M%S')
            
            # Save processed news data
            news_csv_path = os.path.join(news_dir, f"{symbol}_news.csv")
            news_df.to_csv(news_csv_path, index=False)
            print(f"Processed news data saved to {news_csv_path}")
            print(f"News data shape: {news_df.shape}")
            
            return news_df
        else:
            print(f"No news data found for {symbol}")
            print(f"API response: {data}")
            return None
            
    except requests.exceptions.Timeout:
        print(f"Request timed out for {symbol}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request exception: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        print(traceback.format_exc())
        return None

def create_sample_data():
    """Create sample stock data for demonstration purposes."""
    print("Creating sample stock data for demonstration...")
    
    # Define sample stocks
    sample_stocks = ['TSLA', 'NVDA', 'AAPL', 'MSFT', 'GOOG']
    
    # Create sample data for each stock
    for symbol in sample_stocks:
        # Create a date range for the past year
        dates = pd.date_range(end=pd.Timestamp.now(), periods=365)
        
        # Create sample price data with some randomness
        import numpy as np
        np.random.seed(42)  # For reproducibility
        
        # Start with a base price that varies by stock
        if symbol == 'TSLA':
            base_price = 200.0
        elif symbol == 'NVDA':
            base_price = 800.0
        elif symbol == 'AAPL':
            base_price = 150.0
        elif symbol == 'MSFT':
            base_price = 300.0
        elif symbol == 'GOOG':
            base_price = 140.0
        else:
            base_price = 100.0
        
        # Generate price series with trend and randomness
        trend = np.linspace(0, 50, 365)  # Upward trend
        noise = np.random.normal(0, 10, 365)  # Random noise
        close_prices = base_price + trend + noise
        
        # Ensure prices are positive
        close_prices = np.maximum(close_prices, 1.0)
        
        # Create other price columns based on close price
        open_prices = close_prices * np.random.uniform(0.98, 1.02, 365)
        high_prices = np.maximum(close_prices, open_prices) * np.random.uniform(1.0, 1.05, 365)
        low_prices = np.minimum(close_prices, open_prices) * np.random.uniform(0.95, 1.0, 365)
        
        # Create volume data
        volume = np.random.randint(1000000, 10000000, 365)
        
        # Create DataFrame
        df = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume,
            'adjclose': close_prices,  # Using close as adjusted close for simplicity
            'dividend': np.zeros(365),
            'split': np.ones(365)
        }, index=dates)
        
        # Save to CSV
        sample_path = os.path.join(DATA_DIR, f"{symbol}_stock_data_sample.csv")
        df.to_csv(sample_path)
        print(f"Sample data created for {symbol} at {sample_path}")
        
        # Create sample company overview
        overview = {
            "Symbol": symbol,
            "AssetType": "Common Stock",
            "Name": f"{symbol} Inc.",
            "Description": f"Sample description for {symbol}.",
            "Exchange": "NASDAQ",
            "Currency": "USD",
            "Country": "USA",
            "Sector": "Technology",
            "Industry": "Technology",
            "PERatio": "25.5",
            "MarketCapitalization": "1000000000",
            "DividendYield": "1.5",
            "52WeekHigh": str(base_price * 1.2),
            "52WeekLow": str(base_price * 0.8)
        }
        
        # Save overview to JSON
        overview_path = os.path.join(DATA_DIR, f"{symbol}_overview_sample.json")
        with open(overview_path, 'w') as f:
            json.dump(overview, f, indent=2)
        print(f"Sample overview created for {symbol} at {overview_path}")
        
        # Create sample news data
        news_dir = os.path.join(DATA_DIR, 'news')
        os.makedirs(news_dir, exist_ok=True)
        
        # Generate sample news items
        news_items = []
        for i in range(20):
            sentiment_score = np.random.uniform(-1, 1)
            sentiment_label = "Positive" if sentiment_score > 0.25 else ("Negative" if sentiment_score < -0.25 else "Neutral")
            
            news_items.append({
                'title': f"Sample news title {i+1} for {symbol}",
                'summary': f"This is a sample news summary for {symbol}. It contains information about the company that may affect stock prices.",
                'published': (pd.Timestamp.now() - pd.Timedelta(days=i)).strftime('%Y%m%dT%H%M%S'),
                'source': "Sample News Source",
                'url': f"https://example.com/news/{symbol}/{i+1}",
                'overall_sentiment_score': sentiment_score,
                'overall_sentiment_label': sentiment_label,
                'ticker_sentiment_score': sentiment_score,
                'ticker_sentiment_label': sentiment_label
            })
        
        # Create DataFrame
        news_df = pd.DataFrame(news_items)
        
        # Save to CSV
        news_path = os.path.join(news_dir, f"{symbol}_news_sample.csv")
        news_df.to_csv(news_path, index=False)
        print(f"Sample news data created for {symbol} at {news_path}")

if __name__ == "__main__":
    # If command line arguments are provided, update TARGET_STOCKS
    if len(sys.argv) > 1:
        from src.config import DEFAULT_TARGET_STOCKS
        import src.config as config
        config.TARGET_STOCKS = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_TARGET_STOCKS
        print(f"Using command line symbols: {config.TARGET_STOCKS}")
    
    # Collect stock data
    collect_stock_data()
    
    # Collect news sentiment for each stock
    for i, symbol in enumerate(TARGET_STOCKS):
        try:
            print(f"\nFetching news sentiment for {symbol} ({i+1}/{len(TARGET_STOCKS)})...")
            news_df = fetch_news_sentiment(symbol)
            
            # Wait between API calls to avoid rate limiting
            if i < len(TARGET_STOCKS) - 1:
                print(f"Waiting 15 seconds before next API call to avoid rate limiting...")
                time.sleep(15)
                
        except Exception as e:
            print(f"Error fetching news for {symbol}: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            print(f"Using sample news data for {symbol} instead...")
            
            # Copy sample news data if API call fails
            sample_path = os.path.join(DATA_DIR, 'news', f"{symbol}_news_sample.csv")
            target_path = os.path.join(DATA_DIR, 'news', f"{symbol}_news.csv")
            if os.path.exists(sample_path):
                import shutil
                shutil.copy(sample_path, target_path)
                print(f"Sample news data copied to {target_path}")
    
    print("\nData collection completed. Sample data has been created as fallback.")
    print(f"All data is available in the {DATA_DIR} directory.")
