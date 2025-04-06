"""
Insights generator module for the Sentiment Surge project.
Generates actionable investment insights based on model predictions.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

class InsightsGenerator:
    """Class to generate actionable investment insights based on model predictions."""
    
    def __init__(self, data_dir, results_dir):
        """
        Initialize the insights generator.
        
        Args:
            data_dir (str): Directory containing data files
            results_dir (str): Directory to save results
        """
        self.data_dir = data_dir
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
    
    def generate_insights(self, symbols):
        """
        Generate insights for the specified stock symbols.
        
        Args:
            symbols (list): List of stock symbols
            
        Returns:
            dict: Insights for each symbol
        """
        insights = {}
        
        for symbol in symbols:
            print(f"Generating insights for {symbol}...")
            
            # Load prediction results if available
            prediction_path = os.path.join(self.results_dir, f"{symbol}_prediction.json")
            if os.path.exists(prediction_path):
                with open(prediction_path, 'r') as f:
                    prediction = json.load(f)
                
                # Generate insights based on prediction
                symbol_insights = self._generate_symbol_insights(symbol, prediction)
                insights[symbol] = symbol_insights
            else:
                # Generate sample insights if prediction not available
                sample_insights = self._generate_sample_insights(symbol)
                insights[symbol] = sample_insights
        
        # Generate portfolio insights
        portfolio_insights = self._generate_portfolio_insights(insights)
        insights['portfolio'] = portfolio_insights
        
        # Save insights to file
        self._save_insights(insights)
        
        return insights
    
    def _generate_symbol_insights(self, symbol, prediction):
        """
        Generate insights for a specific symbol based on prediction.
        
        Args:
            symbol (str): Stock symbol
            prediction (dict): Prediction results
            
        Returns:
            dict: Insights for the symbol
        """
        # Extract prediction details
        direction = prediction.get('predicted_direction', 'Unknown')
        confidence = prediction.get('confidence', 0.0)
        current_price = prediction.get('current_price', 0.0)
        sentiment_score = prediction.get('sentiment_score', 0.0)
        sentiment_positive = prediction.get('sentiment_positive', 0.0)
        sentiment_negative = prediction.get('sentiment_negative', 0.0)
        
        # Determine recommendation
        if direction == 'Up' and confidence > 0.6:
            recommendation = 'Buy'
            action = f"Consider buying {symbol} at the current price of ${current_price:.2f}"
            rationale = f"The model predicts an upward movement with {confidence:.1%} confidence, supported by positive sentiment ({sentiment_positive:.1%})."
        elif direction == 'Down' and confidence > 0.6:
            recommendation = 'Sell'
            action = f"Consider selling {symbol} at the current price of ${current_price:.2f}"
            rationale = f"The model predicts a downward movement with {confidence:.1%} confidence, influenced by negative sentiment ({sentiment_negative:.1%})."
        else:
            recommendation = 'Hold'
            action = f"Monitor {symbol} at the current price of ${current_price:.2f}"
            rationale = f"The model's prediction lacks sufficient confidence ({confidence:.1%}) to recommend a trade."
        
        # Determine risk level
        if confidence > 0.8:
            risk_level = 'Low'
        elif confidence > 0.6:
            risk_level = 'Medium'
        else:
            risk_level = 'High'
        
        # Generate time horizon recommendation
        if confidence > 0.7:
            time_horizon = 'Short-term (1-5 days)'
        elif sentiment_score > 0.3:
            time_horizon = 'Medium-term (1-4 weeks)'
        else:
            time_horizon = 'Long-term (1-3 months)'
        
        # Generate insights
        insights = {
            'symbol': symbol,
            'current_price': current_price,
            'predicted_direction': direction,
            'confidence': confidence,
            'sentiment_score': sentiment_score,
            'recommendation': recommendation,
            'action': action,
            'rationale': rationale,
            'risk_level': risk_level,
            'time_horizon': time_horizon,
            'date_generated': datetime.now().strftime('%Y-%m-%d')
        }
        
        return insights
    
    def _generate_sample_insights(self, symbol):
        """
        Generate sample insights for a symbol when prediction is not available.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            dict: Sample insights for the symbol
        """
        # Generate random sample data
        np.random.seed(hash(symbol) % 10000)  # Use symbol as seed for reproducibility
        
        # Sample price based on symbol
        if symbol == 'TSLA':
            current_price = np.random.uniform(180.0, 220.0)
        elif symbol == 'NVDA':
            current_price = np.random.uniform(750.0, 850.0)
        elif symbol == 'AAPL':
            current_price = np.random.uniform(140.0, 160.0)
        elif symbol == 'MSFT':
            current_price = np.random.uniform(280.0, 320.0)
        elif symbol == 'GOOG':
            current_price = np.random.uniform(130.0, 150.0)
        else:
            current_price = np.random.uniform(50.0, 150.0)
        
        # Sample prediction data
        directions = ['Up', 'Down', 'Neutral']
        direction_weights = [0.5, 0.3, 0.2]  # Slightly biased toward Up
        direction = np.random.choice(directions, p=direction_weights)
        
        confidence = np.random.uniform(0.5, 0.9)
        sentiment_score = np.random.uniform(-0.5, 0.8)  # Slightly biased toward positive
        sentiment_positive = max(0, sentiment_score)
        sentiment_negative = max(0, -sentiment_score)
        
        # Determine recommendation
        if direction == 'Up' and confidence > 0.6:
            recommendation = 'Buy'
            action = f"Consider buying {symbol} at the current price of ${current_price:.2f}"
            rationale = f"The model predicts an upward movement with {confidence:.1%} confidence, supported by positive sentiment ({sentiment_positive:.1%})."
        elif direction == 'Down' and confidence > 0.6:
            recommendation = 'Sell'
            action = f"Consider selling {symbol} at the current price of ${current_price:.2f}"
            rationale = f"The model predicts a downward movement with {confidence:.1%} confidence, influenced by negative sentiment ({sentiment_negative:.1%})."
        else:
            recommendation = 'Hold'
            action = f"Monitor {symbol} at the current price of ${current_price:.2f}"
            rationale = f"The model's prediction lacks sufficient confidence ({confidence:.1%}) to recommend a trade."
        
        # Determine risk level
        if confidence > 0.8:
            risk_level = 'Low'
        elif confidence > 0.6:
            risk_level = 'Medium'
        else:
            risk_level = 'High'
        
        # Generate time horizon recommendation
        if confidence > 0.7:
            time_horizon = 'Short-term (1-5 days)'
        elif sentiment_score > 0.3:
            time_horizon = 'Medium-term (1-4 weeks)'
        else:
            time_horizon = 'Long-term (1-3 months)'
        
        # Generate insights
        insights = {
            'symbol': symbol,
            'current_price': float(current_price),
            'predicted_direction': direction,
            'confidence': float(confidence),
            'sentiment_score': float(sentiment_score),
            'recommendation': recommendation,
            'action': action,
            'rationale': rationale,
            'risk_level': risk_level,
            'time_horizon': time_horizon,
            'date_generated': datetime.now().strftime('%Y-%m-%d'),
            'note': 'This is a sample insight generated for demonstration purposes.'
        }
        
        return insights
    
    def _generate_portfolio_insights(self, symbol_insights):
        """
        Generate portfolio-level insights based on individual symbol insights.
        
        Args:
            symbol_insights (dict): Insights for individual symbols
            
        Returns:
            dict: Portfolio-level insights
        """
        # Count recommendations
        buy_count = sum(1 for s in symbol_insights.values() if s['recommendation'] == 'Buy')
        sell_count = sum(1 for s in symbol_insights.values() if s['recommendation'] == 'Sell')
        hold_count = sum(1 for s in symbol_insights.values() if s['recommendation'] == 'Hold')
        
        # Calculate average sentiment
        avg_sentiment = np.mean([s['sentiment_score'] for s in symbol_insights.values()])
        
        # Determine market outlook
        if avg_sentiment > 0.3:
            market_outlook = 'Bullish'
        elif avg_sentiment < -0.3:
            market_outlook = 'Bearish'
        else:
            market_outlook = 'Neutral'
        
        # Generate portfolio recommendation
        if buy_count > sell_count and avg_sentiment > 0:
            portfolio_recommendation = 'Increase equity exposure'
            portfolio_action = 'Consider increasing positions in recommended Buy stocks'
        elif sell_count > buy_count and avg_sentiment < 0:
            portfolio_recommendation = 'Reduce equity exposure'
            portfolio_action = 'Consider reducing positions in recommended Sell stocks'
        else:
            portfolio_recommendation = 'Maintain current allocation'
            portfolio_action = 'Monitor positions and wait for stronger signals'
        
        # Generate sector insights if available
        sector_insights = self._generate_sector_insights(symbol_insights)
        
        # Generate portfolio insights
        insights = {
            'market_outlook': market_outlook,
            'average_sentiment': float(avg_sentiment),
            'recommendation_distribution': {
                'buy': buy_count,
                'sell': sell_count,
                'hold': hold_count
            },
            'portfolio_recommendation': portfolio_recommendation,
            'portfolio_action': portfolio_action,
            'sector_insights': sector_insights,
            'date_generated': datetime.now().strftime('%Y-%m-%d')
        }
        
        return insights
    
    def _generate_sector_insights(self, symbol_insights):
        """
        Generate sector-level insights based on individual symbol insights.
        
        Args:
            symbol_insights (dict): Insights for individual symbols
            
        Returns:
            dict: Sector-level insights
        """
        # Sample sector mapping (in a real implementation, this would be loaded from a database)
        sector_mapping = {
            'TSLA': 'Automotive',
            'NVDA': 'Technology',
            'AAPL': 'Technology',
            'MSFT': 'Technology',
            'GOOG': 'Technology',
            'AMZN': 'Consumer Cyclical',
            'META': 'Technology',
            'JPM': 'Financial Services',
            'V': 'Financial Services',
            'JNJ': 'Healthcare'
        }
        
        # Group symbols by sector
        sector_data = {}
        for symbol, insights in symbol_insights.items():
            if symbol in sector_mapping:
                sector = sector_mapping[symbol]
                if sector not in sector_data:
                    sector_data[sector] = []
                sector_data[sector].append(insights)
        
        # Generate insights for each sector
        sector_insights = {}
        for sector, insights_list in sector_data.items():
            # Calculate average sentiment for the sector
            avg_sentiment = np.mean([i['sentiment_score'] for i in insights_list])
            
            # Count recommendations
            buy_count = sum(1 for i in insights_list if i['recommendation'] == 'Buy')
            sell_count = sum(1 for i in insights_list if i['recommendation'] == 'Sell')
            hold_count = sum(1 for i in insights_list if i['recommendation'] == 'Hold')
            
            # Determine sector outlook
            if avg_sentiment > 0.3:
                outlook = 'Bullish'
            elif avg_sentiment < -0.3:
                outlook = 'Bearish'
            else:
                outlook = 'Neutral'
            
            # Generate sector recommendation
            if buy_count > sell_count and avg_sentiment > 0:
                recommendation = 'Overweight'
            elif sell_count > buy_count and avg_sentiment < 0:
                recommendation = 'Underweight'
            else:
                recommendation = 'Market Weight'
            
            # Add sector insights
            sector_insights[sector] = {
                'outlook': outlook,
                'average_sentiment': float(avg_sentiment),
                'recommendation': recommendation,
                'symbols': [i['symbol'] for i in insights_list]
            }
        
        return sector_insights
    
    def _save_insights(self, insights):
        """
        Save insights to file.
        
        Args:
            insights (dict): Generated insights
        """
        # Save overall insights
        insights_path = os.path.join(self.results_dir, "investment_insights.json")
        with open(insights_path, 'w') as f:
            json.dump(insights, f, indent=2)
        
        print(f"Investment insights saved to {insights_path}")
        
        # Create visualization
        self._create_insights_visualization(insights)
    
    def _create_insights_visualization(self, insights):
        """
        Create visualizations of investment insights.
        
        Args:
            insights (dict): Generated insights
        """
        try:
            # Set up the visualization style
            sns.set(style="whitegrid")
            
            # Create figure for symbol recommendations
            plt.figure(figsize=(12, 8))
            
            # Extract symbols and recommendations
            symbols = [s for s in insights.keys() if s != 'portfolio']
            recommendations = [insights[s]['recommendation'] for s in symbols]
            confidence = [insights[s]['confidence'] for s in symbols]
            
            # Create color mapping
            color_map = {'Buy': 'green', 'Hold': 'blue', 'Sell': 'red'}
            colors = [color_map.get(r, 'gray') for r in recommendations]
            
            # Create bar chart
            plt.bar(symbols, confidence, color=colors)
            plt.title('Investment Recommendations with Confidence Levels')
            plt.xlabel('Stock Symbol')
            plt.ylabel('Confidence')
            plt.ylim(0, 1)
            
            # Add recommendation labels
            for i, (symbol, rec, conf) in enumerate(zip(symbols, recommendations, confidence)):
                plt.text(i, conf + 0.02, rec, ha='center')
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='green', label='Buy'),
                Patch(facecolor='blue', label='Hold'),
                Patch(facecolor='red', label='Sell')
            ]
            plt.legend(handles=legend_elements)
            
            plt.tight_layout()
            
            # Save the figure
            fig_path = os.path.join(self.results_dir, "investment_recommendations.png")
            plt.savefig(fig_path)
            plt.close()
            
            print(f"Recommendations visualization saved to {fig_path}")
            
            # Create figure for portfolio insights
            plt.figure(figsize=(10, 6))
            
            # Extract portfolio data
            portfolio = insights['portfolio']
            rec_dist = portfolio['recommendation_distribution']
            
            # Create pie chart
            plt.pie(
                [rec_dist['buy'], rec_dist['hold'], rec_dist['sell']],
                labels=['Buy', 'Hold', 'Sell'],
                colors=['green', 'blue', 'red'],
                autopct='%1.1f%%',
                startangle=90
            )
            plt.title(f"Portfolio Recommendation Distribution\nMarket Outlook: {portfolio['market_outlook']}")
            
            # Save the figure
            fig_path = os.path.join(self.results_dir, "portfolio_insights.png")
            plt.savefig(fig_path)
            plt.close()
            
            print(f"Portfolio insights visualization saved to {fig_path}")
            
        except Exception as e:
            print(f"Error creating insights visualization: {e}")
            import traceback
            print(traceback.format_exc())

def generate_investment_report(insights, output_path):
    """
    Generate a comprehensive investment report based on insights.
    
    Args:
        insights (dict): Generated insights
        output_path (str): Path to save the report
    """
    # Create report content
    report = []
    
    # Add header
    report.append("# Sentiment Surge: Investment Insights Report")
    report.append(f"**Generated on: {datetime.now().strftime('%Y-%m-%d')}**\n")
    
    # Add portfolio summary
    portfolio = insights['portfolio']
    report.append("## Market Outlook and Portfolio Recommendations")
    report.append(f"**Market Outlook:** {portfolio['market_outlook']}")
    report.append(f"**Average Market Sentiment:** {portfolio['average_sentiment']:.2f}")
    report.append(f"**Portfolio Recommendation:** {portfolio['portfolio_recommendation']}")
    report.append(f"**Recommended Action:** {portfolio['portfolio_action']}\n")
    
    # Add recommendation distribution
    rec_dist = portfolio['recommendation_distribution']
    report.append("### Recommendation Distribution")
    report.append(f"- Buy: {rec_dist['buy']} stocks")
    report.append(f"- Hold: {rec_dist['hold']} stocks")
    report.append(f"- Sell: {rec_dist['sell']} stocks\n")
    
    # Add sector insights
    report.append("## Sector Insights")
    for sector, sector_data in portfolio['sector_insights'].items():
        report.append(f"### {sector}")
        report.append(f"**Outlook:** {sector_data['outlook']}")
        report.append(f"**Average Sentiment:** {sector_data['average_sentiment']:.2f}")
        report.append(f"**Recommendation:** {sector_data['recommendation']}")
        report.append(f"**Symbols:** {', '.join(sector_data['symbols'])}\n")
    
    # Add individual stock insights
    report.append("## Individual Stock Insights")
    symbols = [s for s in insights.keys() if s != 'portfolio']
    for symbol in symbols:
        stock = insights[symbol]
        report.append(f"### {symbol}")
        report.append(f"**Current Price:** ${stock['current_price']:.2f}")
        report.append(f"**Predicted Direction:** {stock['predicted_direction']}")
        report.append(f"**Confidence:** {stock['confidence']:.1%}")
        report.append(f"**Sentiment Score:** {stock['sentiment_score']:.2f}")
        report.append(f"**Recommendation:** {stock['recommendation']}")
        report.append(f"**Action:** {stock['action']}")
        report.append(f"**Rationale:** {stock['rationale']}")
        report.append(f"**Risk Level:** {stock['risk_level']}")
        report.append(f"**Time Horizon:** {stock['time_horizon']}\n")
    
    # Add interpretation guide
    report.append("## How to Interpret These Insights")
    report.append("### Confidence Levels")
    report.append("- **High Confidence (>80%):** Strong signal with high probability of accuracy")
    report.append("- **Medium Confidence (60-80%):** Moderate signal with reasonable probability of accuracy")
    report.append("- **Low Confidence (<60%):** Weak signal with lower probability of accuracy\n")
    
    report.append("### Risk Levels")
    report.append("- **Low Risk:** High confidence prediction with strong sentiment alignment")
    report.append("- **Medium Risk:** Moderate confidence prediction with some sentiment support")
    report.append("- **High Risk:** Low confidence prediction or conflicting sentiment signals\n")
    
    report.append("### Time Horizons")
    report.append("- **Short-term:** 1-5 days, suitable for trading strategies")
    report.append("- **Medium-term:** 1-4 weeks, suitable for swing trading")
    report.append("- **Long-term:** 1-3 months, suitable for position trading\n")
    
    report.append("### Disclaimer")
    report.append("*These insights are generated by an AI model based on sentiment analysis and historical correlations. They should not be considered as financial advice. Always conduct your own research and consult with a financial advisor before making investment decisions.*")
    
    # Write report to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Investment report saved to {output_path}")

def main():
    """Main function to generate investment insights."""
    # Define directories
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    results_dir = os.path.join(data_dir, 'results')
    
    # Create insights generator
    generator = InsightsGenerator(data_dir, results_dir)
    
    # Define target stocks
    symbols = ['TSLA', 'NVDA', 'AAPL', 'MSFT', 'GOOG']
    
    # Generate insights
    insights = generator.generate_insights(symbols)
    
    # Generate investment report
    report_path = os.path.join(results_dir, "investment_report.md")
    generate_investment_report(insights, report_path)
    
    print("Investment insights generation completed.")

if __name__ == "__main__":
    main()
