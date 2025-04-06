# Sentiment Surge: Predicting Stock Movements Through Market Sentiment

## Project Overview

Sentiment Surge is a machine learning model that predicts stock price movements by analyzing market sentiment from financial news sources. The model scrapes real-time financial news, performs sentiment analysis using NLP techniques, and correlates sentiment data with stock price movements to generate actionable investment insights.

## Features

- **Generalized Architecture**: Works with any company's stock data, not just specific stocks
- **Data Collection**: Collects stock data from financial APIs (Alpha Vantage)
- **Sentiment Analysis**: Classifies financial news into positive, negative, or neutral categories
- **Correlation Analysis**: Measures relationship between sentiment and stock movements using PCC
- **Prediction Model**: Uses sentiment data and technical indicators to predict stock directions
- **Actionable Insights**: Generates investment recommendations with confidence scores
- **Comprehensive Evaluation**: Includes accuracy metrics and MAPE as specified

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/sentiment-surge.git
cd sentiment-surge

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Command Line Interface

The project includes a command-line interface for easy use:

```bash
# Run the complete pipeline for specific stocks
python main.py --stocks AAPL MSFT GOOG --all

# Collect data only
python main.py --stocks TSLA NVDA --collect-data

# Analyze sentiment only
python main.py --stocks AAPL --analyze-sentiment

# Generate insights
python scripts/generate_insights.py
```

### Jupyter Notebook

For interactive exploration, use the provided Jupyter notebook:

```bash
jupyter notebook notebooks/sentiment_surge.ipynb
```

## Project Structure

```
sentiment_surge/
├── data/               # Data storage
│   ├── news/           # Financial news data
│   ├── sentiment/      # Sentiment analysis results
│   └── results/        # Correlation and prediction results
├── models/             # Trained models
├── notebooks/          # Jupyter notebooks
├── scripts/            # Utility scripts
│   ├── collect_stock_data.py
│   ├── perform_sentiment_analysis.py
│   ├── correlate_sentiment.py
│   ├── build_prediction_model.py
│   ├── evaluate_model.py
│   └── generate_insights.py
├── src/                # Source code
│   ├── config.py
│   ├── data_collector.py
│   ├── sentiment_analyzer.py
│   ├── correlation_analyzer.py
│   ├── prediction_model.py
│   └── insights_generator.py
├── main.py             # Main entry point
└── requirements.txt    # Dependencies
```

## Approach

### 1. Data Collection

The system collects historical stock data and financial news from multiple sources:
- Stock price data from Alpha Vantage API
- Technical indicators calculated from price data

### 2. Sentiment Analysis

Financial news is analyzed to determine sentiment:
- Text is classified as positive, negative, or neutral
- Sentiment scores are aggregated by day
- Daily sentiment summaries are created for correlation analysis

### 3. Correlation Analysis

The system measures the relationship between sentiment and stock movements:
- Calculates Pearson Correlation Coefficient (PCC)
- Measures correlation for both same-day and next-day returns
- Calculates Mean Absolute Percentage Error (MAPE)

### 4. Prediction Model

A machine learning model predicts stock movements:
- Features include sentiment scores and technical indicators
- Random Forest classifier predicts price direction (up, down, neutral)
- Model is trained on historical data and evaluated on test set

### 5. Actionable Insights

The system generates investment recommendations:
- Buy/Sell/Hold recommendations with confidence scores
- Portfolio-level insights and sector analysis
- Risk assessment and time horizon recommendations
- Detailed investment report with visualizations

## Evaluation Metrics

As specified in the problem statement, the model is evaluated using:
- **PCC (Pearson Correlation Coefficient)**: Measures linear correlation between sentiment and stock returns
- **MAPE (Mean Absolute Percentage Error)**: Measures prediction accuracy

Additional metrics include:
- Accuracy, Precision, Recall, and F1 Score
- Direction prediction accuracy
- Feature importance analysis

## Results

The model demonstrates significant correlation between sentiment and stock movements, with particularly strong results for technology stocks. The prediction model achieves good accuracy in predicting price direction, especially when combining sentiment data with technical indicators.

## Future Improvements

- Incorporate more news sources for broader sentiment analysis
- Implement more sophisticated NLP techniques for sentiment classification
- Explore deep learning models for improved prediction accuracy
- Add real-time monitoring and alerting for investment opportunities

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Alpha Vantage for financial data APIs
- The scikit-learn team for machine learning tools
