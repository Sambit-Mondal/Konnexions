"""
Main entry point for the Sentiment Surge project.
"""

import os
import sys
import argparse
from datetime import datetime

tickers = ["TSLA", "NVDA"]

def main():
    """Main function to run the Sentiment Surge pipeline."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Sentiment Surge: Predicting Stock Movements Through Market Sentiment')
    parser.add_argument('--stocks', nargs='+', default=['TSLA', 'NVDA'], 
                        help='Stock symbols to analyze (default: TSLA NVDA)')
    parser.add_argument('--all', action='store_true', 
                        help='Run the complete pipeline')
    parser.add_argument('--collect-data', action='store_true', 
                        help='Collect stock data')
    parser.add_argument('--analyze-sentiment', action='store_true', 
                        help='Analyze sentiment')
    parser.add_argument('--correlate', action='store_true', 
                        help='Correlate sentiment with stock movements')
    parser.add_argument('--build-model', action='store_true', 
                        help='Build and train prediction model')
    parser.add_argument('--evaluate', action='store_true', 
                        help='Evaluate model performance')
    parser.add_argument('--generate-insights', action='store_true', 
                        help='Generate actionable insights')
    
    args = parser.parse_args()
    
    # Update target stocks in config
    from src.config import DEFAULT_TARGET_STOCKS
    import src.config as config
    config.TARGET_STOCKS = args.stocks
    
    print(f"Sentiment Surge: Analyzing stocks {', '.join(config.TARGET_STOCKS)}")
    print("\n" + "="*80)
    print("SENTIMENT SURGE: PREDICTING STOCK MOVEMENTS THROUGH MARKET SENTIMENT")
    print("="*80)
    
    # Determine which steps to run
    run_all = args.all or not any([
        args.collect_data, args.analyze_sentiment, args.correlate,
        args.build_model, args.evaluate, args.generate_insights
    ])
    
    # Run selected steps
    if run_all or args.collect_data:
        print("\n=== Collecting Stock Data ===")
        from scripts.collect_stock_data_alphavantage import collect_stock_data
        collect_stock_data()
    
    if run_all or args.analyze_sentiment:
        print("\n=== Performing Sentiment Analysis ===")
        from scripts.perform_sentiment_analysis import perform_sentiment_analysis
        perform_sentiment_analysis()
    
    if run_all or args.correlate:
        print("\n=== Correlating Sentiment with Stock Movements ===")
        from scripts.correlate_sentiment import correlate_sentiment_with_stock_movements
        correlate_sentiment_with_stock_movements(tickers)
    
    if run_all or args.build_model:
        print("\n=== Building and Training Prediction Model ===")
        from scripts.build_prediction_model import build_prediction_model
        build_prediction_model()
    
    if run_all or args.evaluate:
        print("\n=== Evaluating Model Performance ===")
        from scripts.evaluate_model import evaluate_model_performance
        evaluate_model_performance()
    
    if run_all or args.generate_insights:
        print("\n=== Generating Actionable Insights ===")
        from scripts.generate_insights import generate_insights
        generate_insights()
    
    print("\n" + "="*80)
    print("SENTIMENT SURGE PIPELINE COMPLETED")
    print("="*80)
    print(f"Analyzed stocks: {', '.join(config.TARGET_STOCKS)}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Results are available in the data/results directory")
    print("="*80)

if __name__ == "__main__":
    main()