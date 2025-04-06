"""
Script to evaluate model performance using the metrics specified in the problem statement.
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

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

tickers = ["TSLA", "NVDA"]

# Import project configuration
from src.config import TARGET_STOCKS, DATA_DIR

def evaluate_model_performance():
    """Evaluate model performance using PCC and MAPE metrics."""
    print(f"Evaluating model performance for: {', '.join(TARGET_STOCKS)}")
    print("\n" + "="*80)
    print("SENTIMENT SURGE MODEL EVALUATION")
    print("="*80)
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join(DATA_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Collect evaluation results for all stocks
    all_results = {}
    
    # Process each target stock
    for symbol in TARGET_STOCKS:
        print(f"\nEvaluating model for {symbol}...")
        
        try:
            # Load correlation results (for PCC)
            correlation_path = os.path.join(results_dir, f"{symbol}_correlation.json")
            
            # If correlation results don't exist, try to create them
            if not os.path.exists(correlation_path):
                print(f"No correlation results found for {symbol}, creating them...")
                
                # Import correlation module
                sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
                from scripts.correlate_sentiment import correlate_sentiment_with_stock_movements
                
                # Run correlation analysis
                correlate_sentiment_with_stock_movements(tickers)
                
                # Check if correlation results were created
                if not os.path.exists(correlation_path):
                    print(f"Failed to create correlation results for {symbol}")
                    continue
            
            # Load model evaluation results
            model_eval_path = os.path.join(results_dir, f"{symbol}_model_evaluation.json")
            
            # If model evaluation results don't exist, try to create them
            if not os.path.exists(model_eval_path):
                print(f"No model evaluation results found for {symbol}, creating them...")
                
                # Import model building module
                from scripts.build_prediction_model import build_prediction_model
                
                # Build and evaluate model
                build_prediction_model()
                
                # Check if model evaluation results were created
                if not os.path.exists(model_eval_path):
                    print(f"Failed to create model evaluation results for {symbol}")
                    continue
            
            # Load correlation results
            with open(correlation_path, 'r') as f:
                correlation_results = json.load(f)
            
            # Load model evaluation results
            with open(model_eval_path, 'r') as f:
                model_eval_results = json.load(f)
            
            # Extract PCC (Pearson Correlation Coefficient)
            pcc_same_day = correlation_results['same_day_correlation']['sentiment_score']
            pcc_next_day = correlation_results['next_day_correlation']['sentiment_score']
            
            # Extract MAPE (Mean Absolute Percentage Error)
            mape = correlation_results['mape']
            
            # Extract model accuracy metrics
            accuracy = model_eval_results['accuracy']
            precision = model_eval_results['precision']
            recall = model_eval_results['recall']
            f1_score = model_eval_results['f1_score']
            
            # Combine results
            results = {
                'symbol': symbol,
                'pcc_same_day': pcc_same_day,
                'pcc_next_day': pcc_next_day,
                'mape': mape,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score
            }
            
            # Save combined results
            combined_path = os.path.join(results_dir, f"{symbol}_combined_evaluation.json")
            with open(combined_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Display results with emphasis on PCC and MAPE
            print("\n" + "-"*50)
            print(f"EVALUATION METRICS FOR {symbol}")
            print("-"*50)
            print(f"PCC (Pearson Correlation Coefficient):")
            print(f"  - Same Day: {pcc_same_day:.4f}")
            print(f"  - Next Day: {pcc_next_day:.4f}")
            print(f"MAPE (Mean Absolute Percentage Error): {mape:.4f}")
            print("-"*50)
            print(f"Additional Metrics:")
            print(f"  - Accuracy: {accuracy:.4f}")
            print(f"  - Precision: {precision:.4f}")
            print(f"  - Recall: {recall:.4f}")
            print(f"  - F1 Score: {f1_score:.4f}")
            print("-"*50)
            
            # Add to all results
            all_results[symbol] = results
            
            # Create visualization
            create_evaluation_visualization(results, symbol, results_dir)
            
        except Exception as e:
            print(f"Error evaluating model for {symbol}: {e}")
            print(traceback.format_exc())
    
    # Calculate average metrics across all stocks
    if all_results:
        avg_results = calculate_average_metrics(all_results)
        
        # Save average results
        avg_path = os.path.join(results_dir, "average_evaluation.json")
        with open(avg_path, 'w') as f:
            json.dump(avg_results, f, indent=2)
        
        # Display average results with emphasis on PCC and MAPE
        print("\n" + "="*80)
        print("OVERALL MODEL PERFORMANCE SUMMARY")
        print("="*80)
        print(f"Average PCC (Same Day): {avg_results['avg_pcc_same_day']:.4f}")
        print(f"Average PCC (Next Day): {avg_results['avg_pcc_next_day']:.4f}")
        print(f"Average MAPE: {avg_results['avg_mape']:.4f}")
        print(f"Average Accuracy: {avg_results['avg_accuracy']:.4f}")
        print(f"Average F1 Score: {avg_results['avg_f1_score']:.4f}")
        print("="*80)
        
        # Create comparison visualization
        create_comparison_visualization(all_results, results_dir)

    return {
        "per_stock_results": all_results,
        "average_metrics": avg_results if all_results else {}
    }

def calculate_average_metrics(all_results):
    """
    Calculate average metrics across all stocks.
    
    Args:
        all_results (dict): Results for all stocks
        
    Returns:
        dict: Average metrics
    """
    # Extract metrics for each stock
    pcc_same_day_values = [r['pcc_same_day'] for r in all_results.values() if r['pcc_same_day'] is not None]
    pcc_next_day_values = [r['pcc_next_day'] for r in all_results.values() if r['pcc_next_day'] is not None]
    mape_values = [r['mape'] for r in all_results.values() if r['mape'] is not None]
    accuracy_values = [r['accuracy'] for r in all_results.values() if r['accuracy'] is not None]
    precision_values = [r['precision'] for r in all_results.values() if r['precision'] is not None]
    recall_values = [r['recall'] for r in all_results.values() if r['recall'] is not None]
    f1_score_values = [r['f1_score'] for r in all_results.values() if r['f1_score'] is not None]
    
    # Calculate averages
    avg_pcc_same_day = np.mean(pcc_same_day_values) if pcc_same_day_values else None
    avg_pcc_next_day = np.mean(pcc_next_day_values) if pcc_next_day_values else None
    avg_mape = np.mean(mape_values) if mape_values else None
    avg_accuracy = np.mean(accuracy_values) if accuracy_values else None
    avg_precision = np.mean(precision_values) if precision_values else None
    avg_recall = np.mean(recall_values) if recall_values else None
    avg_f1_score = np.mean(f1_score_values) if f1_score_values else None
    
    # Return average metrics
    return {
        'avg_pcc_same_day': float(avg_pcc_same_day) if avg_pcc_same_day is not None else None,
        'avg_pcc_next_day': float(avg_pcc_next_day) if avg_pcc_next_day is not None else None,
        'avg_mape': float(avg_mape) if avg_mape is not None else None,
        'avg_accuracy': float(avg_accuracy) if avg_accuracy is not None else None,
        'avg_precision': float(avg_precision) if avg_precision is not None else None,
        'avg_recall': float(avg_recall) if avg_recall is not None else None,
        'avg_f1_score': float(avg_f1_score) if avg_f1_score is not None else None
    }

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
        plt.figure(figsize=(12, 8))
        
        # Create a bar chart of metrics
        metrics = ['pcc_same_day', 'pcc_next_day', 'accuracy', 'precision', 'recall', 'f1_score']
        metric_values = [results.get(m, 0) for m in metrics]
        metric_labels = ['PCC (Same Day)', 'PCC (Next Day)', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        # Create bar chart
        plt.bar(metric_labels, metric_values, color='skyblue')
        plt.title(f"{symbol} Model Performance Metrics")
        plt.ylabel('Score')
        plt.ylim(0, 1)
        
        # Add MAPE as text annotation
        if results.get('mape') is not None:
            plt.text(0.5, 0.9, f"MAPE: {results['mape']:.4f}", 
                     horizontalalignment='center',
                     transform=plt.gca().transAxes,
                     bbox=dict(facecolor='white', alpha=0.8))
        
        # Add value labels on top of bars
        for i, v in enumerate(metric_values):
            plt.text(i, v + 0.02, f"{v:.3f}", ha='center')
        
        plt.tight_layout()
        
        # Save the figure
        fig_path = os.path.join(results_dir, f"{symbol}_performance_metrics.png")
        plt.savefig(fig_path)
        plt.close()
        
        print(f"Performance visualization saved to {fig_path}")
        
    except Exception as e:
        print(f"Error creating evaluation visualization: {e}")
        print(traceback.format_exc())

def create_comparison_visualization(all_results, results_dir):
    """
    Create visualizations comparing metrics across all stocks.
    
    Args:
        all_results (dict): Results for all stocks
        results_dir (str): Directory to save results
    """
    try:
        # Set up the visualization style
        sns.set(style="whitegrid")
        plt.figure(figsize=(14, 10))
        
        # Extract data for visualization
        symbols = list(all_results.keys())
        
        # Create DataFrame for easier plotting
        metrics_df = pd.DataFrame({
            'Symbol': symbols,
            'PCC (Same Day)': [all_results[s]['pcc_same_day'] for s in symbols],
            'PCC (Next Day)': [all_results[s]['pcc_next_day'] for s in symbols],
            'Accuracy': [all_results[s]['accuracy'] for s in symbols],
            'F1 Score': [all_results[s]['f1_score'] for s in symbols],
            'MAPE': [all_results[s]['mape'] for s in symbols]
        })
        
        # Melt DataFrame for easier plotting
        melted_df = pd.melt(metrics_df, id_vars=['Symbol'], 
                           value_vars=['PCC (Same Day)', 'PCC (Next Day)', 'Accuracy', 'F1 Score'],
                           var_name='Metric', value_name='Value')
        
        # Create grouped bar chart
        plt.subplot(2, 1, 1)
        sns.barplot(x='Symbol', y='Value', hue='Metric', data=melted_df)
        plt.title('Model Performance Metrics Comparison')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        plt.legend(title='Metric')
        
        # Create MAPE comparison
        plt.subplot(2, 1, 2)
        sns.barplot(x='Symbol', y='MAPE', data=metrics_df, color='coral')
        plt.title('MAPE Comparison (Lower is Better)')
        plt.ylabel('MAPE')
        
        # Add value labels
        for i, v in enumerate(metrics_df['MAPE']):
            if not pd.isna(v):
                plt.text(i, v + 0.02, f"{v:.3f}", ha='center')
        
        plt.tight_layout()
        
        # Save the figure
        fig_path = os.path.join(results_dir, "metrics_comparison.png")
        plt.savefig(fig_path)
        plt.close()
        
        print(f"Comparison visualization saved to {fig_path}")
        
    except Exception as e:
        print(f"Error creating comparison visualization: {e}")
        print(traceback.format_exc())

if __name__ == "_main_":
    # If command line arguments are provided, update TARGET_STOCKS
    if len(sys.argv) > 1:
        from src.config import DEFAULT_TARGET_STOCKS
        import src.config as config
        config.TARGET_STOCKS = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_TARGET_STOCKS
        print(f"Using command line symbols: {config.TARGET_STOCKS}")
    
    # Evaluate model performance and get the metrics
    results = evaluate_model_performance()

    # Optional: Print summary return if you want
    print("\nReturned PCC & MAPE Summary:")
    for symbol, data in results["per_stock_results"].items():
        print(f"{symbol} - PCC (Same Day): {data['pcc_same_day']:.4f}, MAPE: {data['mape']:.4f}")