"""
Script to build and train a prediction model for stock movements based on sentiment data.
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

tickers = ["TSLA", "NVDA"]

# Import project configuration
from src.config import TARGET_STOCKS, DATA_DIR

def build_prediction_model():
    """Build and train a prediction model for stock movements based on sentiment data."""
    print(f"Building prediction model for: {', '.join(TARGET_STOCKS)}")
    
    # Create models and results directories if they don't exist
    models_dir = os.path.join(DATA_DIR, '..', 'models')
    results_dir = os.path.join(DATA_DIR, 'results')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Process each target stock
    for symbol in TARGET_STOCKS:
        print(f"\nBuilding model for {symbol}...")
        
        try:
            # Load merged data
            merged_path = os.path.join(results_dir, f"{symbol}_merged_data.csv")
            
            # If merged data doesn't exist, try to create it
            if not os.path.exists(merged_path):
                print(f"No merged data found for {symbol}, creating it...")
                
                # Import correlation module
                sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
                from scripts.correlate_sentiment import correlate_sentiment_with_stock_movements
                
                # Run correlation analysis to create merged data
                correlate_sentiment_with_stock_movements(tickers)
                
                # Check if merged data was created
                if not os.path.exists(merged_path):
                    print(f"Failed to create merged data for {symbol}")
                    continue
            
            # Load merged data
            merged_df = pd.read_csv(merged_path)
            
            if merged_df.empty:
                print(f"Merged data is empty for {symbol}")
                continue
            
            print(f"Loaded merged data with shape: {merged_df.shape}")
            
            # Prepare features and target
            X, y, feature_names = prepare_features_and_target(merged_df)
            
            if X is None or y is None:
                print(f"Failed to prepare features for {symbol}")
                continue
            
            print(f"Prepared features with shape: {X.shape}")
            
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y)
            
            # Train model
            model, scaler = train_model(X_train, y_train, symbol, models_dir)
            
            if model is None:
                print(f"Failed to train model for {symbol}")
                continue
            
            # Evaluate model
            evaluation = evaluate_model(model, scaler, X_test, y_test, feature_names, symbol, results_dir)
            
            # Generate predictions
            predictions = generate_predictions(model, scaler, merged_df, feature_names, symbol, results_dir)
            
        except Exception as e:
            print(f"Error building model for {symbol}: {e}")
            print(traceback.format_exc())

def prepare_features_and_target(merged_df):
    """
    Prepare features and target variables for model training.
    
    Args:
        merged_df (pandas.DataFrame): Merged stock and sentiment data
        
    Returns:
        tuple: (X, y, feature_names) feature array, target array, and feature names
    """
    try:
        # Select base features
        base_features = [
            'sentiment_score', 
            'sentiment_positive', 
            'sentiment_negative', 
            'sentiment_neutral',
            'daily_return'  # Include previous day's return as a feature
        ]
        
        # Check if all required features exist
        missing_features = [f for f in base_features if f not in merged_df.columns]
        if missing_features:
            print(f"Missing features: {missing_features}")
            return None, None, None
        
        # Add technical indicators
        merged_df = add_technical_indicators(merged_df)
        
        # Add additional features from technical indicators
        tech_features = [
            'sma_5', 'sma_10', 'ema_5', 'ema_10',
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'upper_band', 'middle_band', 'lower_band'
        ]
        
        # Combine all features
        all_features = base_features + tech_features
        
        # Remove rows with NaN values
        merged_df = merged_df.dropna(subset=all_features + ['next_day_return'])
        
        if merged_df.empty:
            print("No data left after removing NaN values")
            return None, None, None
        
        # Extract features and target
        X = merged_df[all_features].values
        
        # For classification, we predict the direction (up, down, neutral)
        y = np.sign(merged_df['next_day_return'].values)
        
        return X, y, all_features
        
    except Exception as e:
        print(f"Error preparing features: {e}")
        print(traceback.format_exc())
        return None, None, None

def add_technical_indicators(df):
    """
    Add technical indicators to the DataFrame.
    
    Args:
        df (pandas.DataFrame): Stock data DataFrame
        
    Returns:
        pandas.DataFrame: DataFrame with technical indicators
    """
    try:
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Simple Moving Averages
        df_copy['sma_5'] = df_copy['close'].rolling(window=5).mean()
        df_copy['sma_10'] = df_copy['close'].rolling(window=10).mean()
        
        # Exponential Moving Averages
        df_copy['ema_5'] = df_copy['close'].ewm(span=5, adjust=False).mean()
        df_copy['ema_10'] = df_copy['close'].ewm(span=10, adjust=False).mean()
        
        # Relative Strength Index (RSI)
        delta = df_copy['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        # Avoid division by zero
        loss = loss.replace(0, np.nan)
        rs = gain / loss
        rs = rs.fillna(0)
        
        df_copy['rsi'] = 100 - (100 / (1 + rs))
        
        # Moving Average Convergence Divergence (MACD)
        ema_12 = df_copy['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df_copy['close'].ewm(span=26, adjust=False).mean()
        df_copy['macd'] = ema_12 - ema_26
        df_copy['macd_signal'] = df_copy['macd'].ewm(span=9, adjust=False).mean()
        df_copy['macd_hist'] = df_copy['macd'] - df_copy['macd_signal']
        
        # Bollinger Bands
        df_copy['middle_band'] = df_copy['close'].rolling(window=20).mean()
        std_dev = df_copy['close'].rolling(window=20).std()
        df_copy['upper_band'] = df_copy['middle_band'] + (std_dev * 2)
        df_copy['lower_band'] = df_copy['middle_band'] - (std_dev * 2)
        
        return df_copy
        
    except Exception as e:
        print(f"Error adding technical indicators: {e}")
        print(traceback.format_exc())
        return df

def train_test_split(X, y, train_ratio=0.8):
    """
    Split data into training and testing sets.
    
    Args:
        X (numpy.ndarray): Feature array
        y (numpy.ndarray): Target array
        train_ratio (float): Ratio of training data
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    try:
        # Calculate split index
        split_idx = int(len(X) * train_ratio)
        
        # Split data
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"Training set size: {len(X_train)}, Testing set size: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        print(f"Error splitting data: {e}")
        print(traceback.format_exc())
        return None, None, None, None

def train_model(X_train, y_train, symbol, models_dir):
    """
    Train a prediction model.
    
    Args:
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training targets
        symbol (str): Stock symbol
        models_dir (str): Directory to save models
        
    Returns:
        tuple: (model, scaler) Trained model and feature scaler
    """
    try:
        # Import necessary libraries
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        
        print(f"Training model for {symbol}...")
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train a Random Forest classifier
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Save the model and scaler
        model_path = os.path.join(models_dir, f"{symbol}_model.pkl")
        scaler_path = os.path.join(models_dir, f"{symbol}_scaler.pkl")
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        print(f"Model saved to {model_path}")
        
        return model, scaler
        
    except Exception as e:
        print(f"Error training model: {e}")
        print(traceback.format_exc())
        return None, None

def evaluate_model(model, scaler, X_test, y_test, feature_names, symbol, results_dir):
    """
    Evaluate the trained model.
    
    Args:
        model: Trained model
        scaler: Feature scaler
        X_test (numpy.ndarray): Test features
        y_test (numpy.ndarray): Test targets
        feature_names (list): Names of features
        symbol (str): Stock symbol
        results_dir (str): Directory to save results
        
    Returns:
        dict: Evaluation metrics
    """
    try:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        print(f"Evaluating model for {symbol}...")
        
        # Scale test features
        X_test_scaled = scaler.transform(X_test)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Calculate feature importance
        feature_importance = model.feature_importances_
        
        # Save evaluation results
        results = {
            'symbol': symbol,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': conf_matrix.tolist(),
            'feature_importance': feature_importance.tolist(),
            'feature_names': feature_names
        }
        
        # Save to file
        results_path = os.path.join(results_dir, f"{symbol}_model_evaluation.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Evaluation results saved to {results_path}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Create visualization
        create_evaluation_visualization(results, symbol, results_dir)
        
        return results
        
    except Exception as e:
        print(f"Error evaluating model: {e}")
        print(traceback.format_exc())
        return None

def create_evaluation_visualization(results, symbol, results_dir):
    """
    Create visualizations of model evaluation results.
    
    Args:
        results (dict): Evaluation results
        symbol (str): Stock symbol
        results_dir (str): Directory to save results
    """
    try:
        # Set up the visualization style
        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 10))
        
        # Plot 1: Confusion Matrix
        plt.subplot(2, 1, 1)
        plt.title(f"{symbol} Model Confusion Matrix")
        
        conf_matrix = np.array(results['confusion_matrix'])
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Down', 'Neutral', 'Up'],
                    yticklabels=['Down', 'Neutral', 'Up'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Plot 2: Feature Importance
        plt.subplot(2, 1, 2)
        plt.title(f"{symbol} Feature Importance")
        
        # Sort features by importance
        importance = results['feature_importance']
        feature_names = results['feature_names']
        indices = np.argsort(importance)[::-1]
        
        # Plot feature importance
        plt.bar(range(len(importance)), [importance[i] for i in indices])
        plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        
        # Save the figure
        fig_path = os.path.join(results_dir, f"{symbol}_model_evaluation.png")
        plt.savefig(fig_path)
        plt.close()
        
        print(f"Evaluation visualization saved to {fig_path}")
        
    except Exception as e:
        print(f"Error creating evaluation visualization: {e}")
        print(traceback.format_exc())

def generate_predictions(model, scaler, merged_df, feature_names, symbol, results_dir):
    """
    Generate predictions for the most recent data.
    
    Args:
        model: Trained model
        scaler: Feature scaler
        merged_df (pandas.DataFrame): Merged data
        feature_names (list): Names of features
        symbol (str): Stock symbol
        results_dir (str): Directory to save results
        
    Returns:
        dict: Prediction results
    """
    try:
        print(f"Generating predictions for {symbol}...")
        
        # Add technical indicators
        df = add_technical_indicators(merged_df)
        
        # Get the most recent data point
        latest_data = df.iloc[-1]
        
        # Check if we have all required features
        missing_features = [f for f in feature_names if f not in df.columns]
        if missing_features:
            print(f"Missing features for prediction: {missing_features}")
            return None
        
        # Extract features
        X = latest_data[feature_names].values.reshape(1, -1)
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        prediction_proba = model.predict_proba(X_scaled)[0]
        
        # Map prediction to direction
        direction_map = {-1: 'Down', 0: 'Neutral', 1: 'Up'}
        predicted_direction = direction_map.get(prediction, 'Unknown')
        
        # Calculate confidence
        confidence = max(prediction_proba)
        
        # Get current price
        current_price = latest_data['close']
        
        # Get current date
        current_date = latest_data['date'] if 'date' in latest_data else str(datetime.now().date())
        
        # Save prediction results
        results = {
            'symbol': symbol,
            'date': str(current_date),
            'current_price': float(current_price),
            'predicted_direction': predicted_direction,
            'confidence': float(confidence),
            'sentiment_score': float(latest_data['sentiment_score']),
            'sentiment_positive': float(latest_data['sentiment_positive']),
            'sentiment_negative': float(latest_data['sentiment_negative']),
            'sentiment_neutral': float(latest_data['sentiment_neutral'])
        }
        
        # Determine recommendation
        if predicted_direction == 'Up' and confidence > 0.6:
            recommendation = 'Buy'
        elif predicted_direction == 'Down' and confidence > 0.6:
            recommendation = 'Sell'
        else:
            recommendation = 'Hold'
        
        results['recommendation'] = recommendation
        
        # Save to file
        results_path = os.path.join(results_dir, f"{symbol}_prediction.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Prediction results saved to {results_path}")
        print(f"Current Price: ${current_price:.2f}")
        print(f"Predicted Direction: {predicted_direction}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Recommendation: {recommendation}")
        
        return results
        
    except Exception as e:
        print(f"Error generating predictions: {e}")
        print(traceback.format_exc())
        return None

if __name__ == "__main__":
    # If command line arguments are provided, update TARGET_STOCKS
    if len(sys.argv) > 1:
        from src.config import DEFAULT_TARGET_STOCKS
        import src.config as config
        config.TARGET_STOCKS = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_TARGET_STOCKS
        print(f"Using command line symbols: {config.TARGET_STOCKS}")
    
    # Build prediction model
    build_prediction_model()
