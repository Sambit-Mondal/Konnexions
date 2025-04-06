import os
import pandas as pd
import traceback
import matplotlib.pyplot as plt
import seaborn as sns

# Directories
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SENTIMENT_DATA_DIR = os.path.join(DATA_DIR, "sentiment")
STOCK_DATA_DIR = os.path.join(DATA_DIR, "stocks")
MERGED_DATA_DIR = os.path.join(DATA_DIR, "merged")
CORRELATION_RESULTS_DIR = os.path.join(DATA_DIR, "correlation_results")
PLOTS_DIR = os.path.join(DATA_DIR, "plots")

# Ensure output directories exist
os.makedirs(MERGED_DATA_DIR, exist_ok=True)
os.makedirs(CORRELATION_RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

tickers = ["TSLA", "NVDA"]

def correlate_sentiment_with_stock_movements(tickers):
    print("=== Correlating Sentiment with Stock Movements ===")
    print(f"Correlating sentiment with stock movements for: {', '.join(tickers)}")

    correlation_results = {}

    for ticker in tickers:
        print(f"\nAnalyzing correlation for {ticker}...")

        try:
            sentiment_path = os.path.join(SENTIMENT_DATA_DIR, f"{ticker}_daily_sentiment.csv")
            stock_path = os.path.join(STOCK_DATA_DIR, f"{ticker}_stock_data.csv")

            if not os.path.exists(sentiment_path) or not os.path.exists(stock_path):
                raise FileNotFoundError(f"Missing data files for {ticker}")

            sentiment_df = pd.read_csv(sentiment_path)
            stock_df = pd.read_csv(stock_path)

            if 'date' not in sentiment_df.columns or 'date' not in stock_df.columns:
                raise ValueError(f"'date' column missing in one of the CSVs for {ticker}")

            sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
            stock_df['date'] = pd.to_datetime(stock_df['date'])

            # Merge the dataframes on 'date'
            merged_df = pd.merge(sentiment_df, stock_df, on='date', how='inner')

            # Save the merged data
            merged_path = os.path.join(MERGED_DATA_DIR, f"{ticker}_merged.csv")
            merged_df.to_csv(merged_path, index=False)

            # Compute correlation
            correlation = merged_df['sentiment_score'].corr(merged_df['close'])
            correlation_results[ticker] = correlation

            # Save correlation results
            result_path = os.path.join(CORRELATION_RESULTS_DIR, f"{ticker}_correlation.csv")
            with open(result_path, "w") as f:
                f.write("ticker,correlation\n")
                f.write(f"{ticker},{correlation:.4f}\n")

            # Generate scatter plot
            plot_path = os.path.join(PLOTS_DIR, f"{ticker}_correlation.png")
            plt.figure(figsize=(8, 5))
            sns.scatterplot(data=merged_df, x='sentiment_score', y='close')
            plt.title(f"Sentiment vs Stock Price for {ticker}")
            plt.xlabel("Sentiment Score")
            plt.ylabel("Stock Close Price")
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()

        except Exception as e:
            print(f"Error analyzing correlation for {ticker}: {e}")
            traceback.print_exc()

    return correlation_results

# If this script is run directly
if __name__ == "__main__":
    tickers = ["TSLA", "NVDA"]  # Add more tickers as needed
    correlate_sentiment_with_stock_movements(tickers)