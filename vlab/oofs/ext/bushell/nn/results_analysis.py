"""
Results analysis and visualization for hierarchical surrogate neural network research.
Generate plots and statistics for paper publication.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import torch

def analyze_training_convergence(csv_file):
    """Analyze training convergence and create publication-quality plots"""
    df = pd.read_csv(csv_file)
    
    # Training convergence plot
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Loss components over time
    plt.subplot(2, 2, 1)
    plt.plot(df['run #'], df['cost_loss'], label='Cost Loss', alpha=0.7)
    plt.plot(df['run #'], df['count_loss'], label='Count Loss', alpha=0.7)
    plt.plot(df['run #'], df['coord_reg'], label='Coord Regularization', alpha=0.7)
    plt.xlabel('Training Iteration')
    plt.ylabel('Loss Value')
    plt.title('Loss Component Evolution')
    plt.legend()
    plt.yscale('log')
    
    # Plot 2: Prediction accuracy
    plt.subplot(2, 2, 2)
    relative_error = np.abs(df['pred_cost'] - df['true_cost']) / df['true_cost']
    moving_avg = pd.Series(relative_error).rolling(window=100).mean()
    plt.plot(df['run #'], relative_error, alpha=0.3, color='blue', label='Relative Error')
    plt.plot(df['run #'], moving_avg, color='red', linewidth=2, label='100-sample Moving Average')
    plt.axhline(y=0.05, color='green', linestyle='--', label='5% Accuracy Threshold')
    plt.xlabel('Training Iteration')
    plt.ylabel('Relative Error')
    plt.title('Prediction Accuracy Over Training')
    plt.legend()
    plt.yscale('log')
    
    # Plot 3: Cost prediction vs true cost scatter
    plt.subplot(2, 2, 3)
    plt.scatter(df['true_cost'], df['pred_cost'], alpha=0.5, s=1)
    min_cost, max_cost = df['true_cost'].min(), df['true_cost'].max()
    plt.plot([min_cost, max_cost], [min_cost, max_cost], 'r--', label='Perfect Prediction')
    plt.xlabel('True Cost')
    plt.ylabel('Predicted Cost')
    plt.title('Predicted vs True Cost')
    plt.legend()
    
    # Plot 4: Parameter sensitivity heatmap
    plt.subplot(2, 2, 4)
    param_cols = [f'param_{i}' for i in range(13)]
    correlations = df[param_cols + ['true_cost']].corr()['true_cost'][:-1]
    plt.bar(range(13), np.abs(correlations))
    plt.xlabel('Parameter Index')
    plt.ylabel('|Correlation with Cost|')
    plt.title('Parameter Sensitivity Analysis')
    
    plt.tight_layout()
    plt.savefig('training_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df

def performance_metrics_table(df):
    """Generate performance metrics table for paper"""
    # Calculate key metrics
    relative_errors = np.abs(df['pred_cost'] - df['true_cost']) / df['true_cost']
    
    metrics = {
        'Mean Relative Error': f"{np.mean(relative_errors):.4f}",
        'Median Relative Error': f"{np.median(relative_errors):.4f}",
        'Std Relative Error': f"{np.std(relative_errors):.4f}",
        'Accuracy (< 5%)': f"{100 * np.mean(relative_errors < 0.05):.1f}%",
        'Accuracy (< 10%)': f"{100 * np.mean(relative_errors < 0.10):.1f}%",
        'R² Score': f"{stats.pearsonr(df['pred_cost'], df['true_cost'])[0]**2:.4f}",
        'Final Training Loss': f"{df['cost_loss'].iloc[-100:].mean():.4f}",
        'Training Samples': f"{len(df):,}"
    }
    
    print("Performance Metrics Table:")
    print("=" * 40)
    for metric, value in metrics.items():
        print(f"{metric:20}: {value}")
    
    return metrics

if __name__ == "__main__":
    # Analyze your results
    csv_file = "normal_hier_plant_surrogate_model.pt.csv"
    df = analyze_training_convergence(csv_file)
    metrics = performance_metrics_table(df)
