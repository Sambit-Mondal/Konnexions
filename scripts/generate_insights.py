"""
Script to generate actionable investment insights based on the Sentiment Surge model.
"""

import sys
import os
import json
from datetime import datetime

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import the insights generator
from src.insights_generator import InsightsGenerator, generate_investment_report

def generate_insights():
    """Generate actionable investment insights."""
    print("Generating actionable investment insights...")
    
    # Define directories
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    results_dir = os.path.join(data_dir, 'results')
    
    # Create directories if they don't exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Create insights generator
    generator = InsightsGenerator(data_dir, results_dir)
    
    # Define target stocks
    symbols = ['TSLA', 'NVDA', 'AAPL', 'MSFT', 'GOOG']
    
    # Generate insights
    insights = generator.generate_insights(symbols)
    
    # Generate investment report
    report_path = os.path.join(results_dir, "investment_report.md")
    generate_investment_report(insights, report_path)
    
    # Print summary
    print("\nInvestment Insights Summary:")
    print(f"Generated insights for {len(insights) - 1} stocks")  # Subtract 1 for portfolio
    
    # Print portfolio recommendation
    portfolio = insights['portfolio']
    print(f"\nMarket Outlook: {portfolio['market_outlook']}")
    print(f"Portfolio Recommendation: {portfolio['portfolio_recommendation']}")
    print(f"Recommended Action: {portfolio['portfolio_action']}")
    
    # Print individual stock recommendations
    print("\nStock Recommendations:")
    for symbol in symbols:
        if symbol in insights:
            stock = insights[symbol]
            print(f"{symbol}: {stock['recommendation']} (Confidence: {stock['confidence']:.1%})")
    
    print(f"\nDetailed investment report saved to {report_path}")
    print("Investment insights generation completed.")

if __name__ == "__main__":
    generate_insights()
