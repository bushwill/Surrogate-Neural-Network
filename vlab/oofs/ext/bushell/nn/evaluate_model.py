"""
Universal model evaluation script for hierarchical surrogate neural network research.
Compares multiple models fairly by using the minimum training samples across all models.
Groups samples into 100 bins for clean visualization regardless of dataset size.

Configure the CSV files to compare in the CSV_FILES list below.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os
import argparse
from pathlib import Path

# ============================================================================
# CONFIGURATION: Edit this list to specify which CSV files to compare
# ============================================================================

CSV_FILES = [
    "Normal Data/08-11-25 Run/surrogate_model1.pt.csv",
    "Normal Data/08-11-25 Run/surrogate_model2.pt.csv",
    "Normal Data/08-11-25 Run/mutant1_surrogate_model.pt.csv",
    "Normal Data/08-11-25 Run/mutant2_surrogate_model.pt.csv",
]

# Optional: Custom model names (if not provided, will use directory names)
CUSTOM_MODEL_NAMES = {
    CSV_FILES[0]: "Base Model 1",
    CSV_FILES[1]: "Base Model 2",
    CSV_FILES[2]: "Mutant Model 1",
    CSV_FILES[3]: "Mutant Model 2",
}

# Output settings
OUTPUT_DIRECTORY = "Normal Data/08-11-25 Run"
NUM_GROUPS = 100  # Number of groups to divide samples into

def load_and_process_csvs(csv_paths=None, custom_names=None):
    """Load CSVs and find minimum sample count for fair comparison"""
    if csv_paths is None:
        csv_paths = CSV_FILES
    if custom_names is None:
        custom_names = CUSTOM_MODEL_NAMES
        
    models_data = {}
    min_samples = float('inf')
    
    print(f"Looking for {len(csv_paths)} CSV file(s)...")
    
    for i, csv_path in enumerate(csv_paths):
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found, skipping...")
            continue
            
        # Load CSV
        df = pd.read_csv(csv_path)
        
        # Clean column names (remove spaces)
        df.columns = df.columns.str.strip()
        
        # Print column information for debugging
        print(f"Columns in {csv_path}: {list(df.columns)}")
        
        # Standardize column names for different model types
        column_mapping = {
            'avg_loss': 'cost_loss',  # Some models may use avg_loss instead of cost_loss
            'loss': 'cost_loss',
            'final_loss': 'cost_loss',
            'individual_loss': 'cost_loss'
        }
        
        # Apply column mapping if needed
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns and 'cost_loss' not in df.columns:
                df = df.rename(columns={old_name: new_name})
                print(f"  Mapped '{old_name}' to '{new_name}'")
        
        # Ensure required columns exist
        required_columns = ['pred_cost', 'true_cost']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Warning: {csv_path} missing required columns {missing_columns}, skipping...")
            continue
        
        # Create cost_loss if it doesn't exist (fallback to avg_loss or calculate from predictions)
        if 'cost_loss' not in df.columns:
            if 'avg_loss' in df.columns:
                df['cost_loss'] = df['avg_loss']
                print(f"  Using 'avg_loss' as 'cost_loss'")
            else:
                # Calculate a basic loss from predictions
                df['cost_loss'] = np.abs(df['pred_cost'] - df['true_cost']) / df['true_cost']
                print(f"  Calculated 'cost_loss' from predictions")
        
        # Create total_loss if it doesn't exist
        if 'total_loss' not in df.columns:
            df['total_loss'] = df['cost_loss']
            print(f"  Using 'cost_loss' as 'total_loss'")
        
        # Determine model name
        if csv_path in custom_names:
            model_name = custom_names[csv_path]
        elif len(csv_paths) > 1:
            model_name = Path(csv_path).parent.name
            if model_name == "." or model_name == "":
                model_name = Path(csv_path).stem
        else:
            model_name = f"Model_{i+1}"
            
        models_data[model_name] = df
        
        # Track minimum samples
        min_samples = min(min_samples, len(df))
        
        print(f"Loaded {model_name}: {len(df):,} samples")
    
    if not models_data:
        print("No valid CSV files found! Please check the CSV_FILES list in the script.")
        return {}, 0
    
    print(f"\nUsing {min_samples:,} samples for fair comparison")
    
    # Truncate all models to minimum sample count
    for model_name in models_data:
        models_data[model_name] = models_data[model_name].iloc[:min_samples].copy()
    
    return models_data, min_samples

def group_samples(df, num_groups=None):
    """Group samples into specified number of groups and calculate averages"""
    if num_groups is None:
        num_groups = NUM_GROUPS
        
    samples_per_group = len(df) // num_groups
    grouped_data = []
    
    for i in range(num_groups):
        start_idx = i * samples_per_group
        end_idx = start_idx + samples_per_group
        
        # Handle last group (include remaining samples)
        if i == num_groups - 1:
            end_idx = len(df)
        
        group = df.iloc[start_idx:end_idx]
        
        # Calculate group statistics - handle missing columns gracefully
        group_stats = {
            'group_number': i + 1,
            'start_sample': start_idx + 1,
            'end_sample': end_idx,
            'avg_pred_cost': group['pred_cost'].mean(),
            'avg_true_cost': group['true_cost'].mean(),
            'avg_relative_error': np.mean(np.abs(group['pred_cost'] - group['true_cost']) / group['true_cost']),
            'accuracy_1pct': 100 * np.mean(np.abs(group['pred_cost'] - group['true_cost']) / group['true_cost'] < 0.01),
            'accuracy_5pct': 100 * np.mean(np.abs(group['pred_cost'] - group['true_cost']) / group['true_cost'] < 0.05),
            'accuracy_10pct': 100 * np.mean(np.abs(group['pred_cost'] - group['true_cost']) / group['true_cost'] < 0.10),
            'sample_count': len(group)
        }
        
        # Add optional columns if they exist
        if 'cost_loss' in group.columns:
            group_stats['avg_cost_loss'] = group['cost_loss'].mean()
        else:
            group_stats['avg_cost_loss'] = group_stats['avg_relative_error']  # Use relative error as fallback
            
        if 'total_loss' in group.columns:
            group_stats['avg_total_loss'] = group['total_loss'].mean()
        else:
            group_stats['avg_total_loss'] = group_stats['avg_cost_loss']  # Use cost_loss as fallback
        
        grouped_data.append(group_stats)
    
    return pd.DataFrame(grouped_data)

def create_comparison_plots(models_data, output_dir=None):
    """Create comprehensive comparison plots"""
    if output_dir is None:
        output_dir = OUTPUT_DIRECTORY
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plot style
    plt.style.use('default')
    colors = plt.cm.Set1(np.linspace(0, 1, len(models_data)))
    
    # Create main comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # Process each model
    model_summaries = {}
    for (model_name, df), color in zip(models_data.items(), colors):
        grouped_df = group_samples(df, num_groups=NUM_GROUPS)
        model_summaries[model_name] = {
            'grouped_data': grouped_df,
            'full_data': df,
            'color': color
        }
    
    # Plot 1: Cost Loss Evolution
    ax1 = axes[0, 0]
    for model_name, data in model_summaries.items():
        grouped_df = data['grouped_data']
        ax1.plot(grouped_df['end_sample'], grouped_df['avg_cost_loss'], 
                label=model_name, color=data['color'], linewidth=2)
    ax1.set_xlabel('Training Samples')
    ax1.set_ylabel('Average Cost Loss')
    ax1.set_title('Cost Loss Evolution')
    ax1.legend()
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Relative Error Evolution
    ax2 = axes[0, 1]
    for model_name, data in model_summaries.items():
        grouped_df = data['grouped_data']
        ax2.plot(grouped_df['end_sample'], grouped_df['avg_relative_error'], 
                label=model_name, color=data['color'], linewidth=2)
    ax2.axhline(y=0.01, color='green', linestyle='--', alpha=0.7, label='1% Threshold')
    ax2.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='5% Threshold')
    ax2.axhline(y=0.10, color='orange', linestyle='--', alpha=0.7, label='10% Threshold')
    ax2.set_xlabel('Training Samples')
    ax2.set_ylabel('Average Relative Error')
    ax2.set_title('Relative Error Evolution')
    ax2.legend()
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Accuracy (< 5%) Evolution
    ax3 = axes[0, 2]
    for model_name, data in model_summaries.items():
        grouped_df = data['grouped_data']
        ax3.plot(grouped_df['end_sample'], grouped_df['accuracy_5pct'], 
                label=model_name, color=data['color'], linewidth=2)
    ax3.set_xlabel('Training Samples')
    ax3.set_ylabel('Accuracy < 5% (%)')
    ax3.set_title('Accuracy Evolution (< 5% Error)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 100)
    
    # Plot 4: Predicted vs True Cost (Final Group)
    ax4 = axes[1, 0]
    for model_name, data in model_summaries.items():
        df = data['full_data']
        grouped_df = data['grouped_data']
        # Use the final group from grouped data for consistency
        final_group_start = int(grouped_df.iloc[-1]['start_sample']) - 1  # Convert to 0-based indexing
        final_group_end = int(grouped_df.iloc[-1]['end_sample'])
        final_samples = df.iloc[final_group_start:final_group_end]
        ax4.scatter(final_samples['true_cost'], final_samples['pred_cost'], 
                   alpha=0.6, s=10, label=model_name, color=data['color'])
    
    # Perfect prediction line
    if len(model_summaries) > 0:
        all_costs = pd.concat([data['full_data'][['true_cost', 'pred_cost']] 
                              for data in model_summaries.values()])
        min_cost, max_cost = all_costs.min().min(), all_costs.max().max()
        ax4.plot([min_cost, max_cost], [min_cost, max_cost], 'k--', alpha=0.7, label='Perfect Prediction')
    
    ax4.set_xlabel('True Cost')
    ax4.set_ylabel('Predicted Cost')
    ax4.set_title(f'Predicted vs True Cost (Final Group)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Loss Distribution (Final Group)
    ax5 = axes[1, 1]
    max_relative_error = 0
    all_final_errors = []
    
    for model_name, data in model_summaries.items():
        df = data['full_data']
        grouped_df = data['grouped_data']
        # Use the final group from grouped data for consistency
        final_group_start = int(grouped_df.iloc[-1]['start_sample']) - 1  # Convert to 0-based indexing
        final_group_end = int(grouped_df.iloc[-1]['end_sample'])
        final_relative_errors = np.abs(df.iloc[final_group_start:final_group_end]['pred_cost'] - df.iloc[final_group_start:final_group_end]['true_cost']) / df.iloc[final_group_start:final_group_end]['true_cost']
        all_final_errors.extend(final_relative_errors.tolist())
        max_relative_error = max(max_relative_error, final_relative_errors.max())
        
        ax5.hist(final_relative_errors, bins=50, alpha=0.7, label=model_name, 
                color=data['color'], density=True)
    
    # Set x-axis limit based on actual data range
    # Use 95th percentile to avoid extreme outliers stretching the axis too much
    if all_final_errors:
        percentile_95 = np.percentile(all_final_errors, 95)
        # Use the smaller of max error or 95th percentile, but ensure we show at least up to 0.5
        x_limit = max(0.5, min(max_relative_error * 1.1, percentile_95 * 1.2))
        ax5.set_xlim(0, x_limit)
    else:
        ax5.set_xlim(0, 1)  # Fallback if no data
    
    ax5.set_xlabel('Relative Error')
    ax5.set_ylabel('Density')
    ax5.set_title(f'Relative Error Distribution (Final Group)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Parameter Sensitivity (if multiple parameters)
    ax6 = axes[1, 2]
    if len(model_summaries) == 1:
        # Single model: show parameter correlations
        model_name, data = list(model_summaries.items())[0]
        df = data['full_data']
        param_cols = [col for col in df.columns if col.startswith('param_')]
        if param_cols:
            correlations = df[param_cols + ['true_cost']].corr()['true_cost'][:-1]
            bars = ax6.bar(range(len(correlations)), np.abs(correlations), color=data['color'])
            ax6.set_xlabel('Parameter Index')
            ax6.set_ylabel('|Correlation with Cost|')
            ax6.set_title('Parameter Sensitivity')
            ax6.set_xticks(range(len(correlations)))
            ax6.grid(True, alpha=0.3)
        else:
            ax6.text(0.5, 0.5, 'No parameter data available', ha='center', va='center', transform=ax6.transAxes)
    else:
        # Multiple models: show final performance comparison using final group
        model_names = list(model_summaries.keys())
        final_accuracies = []
        final_rel_errors = []
        
        for data in model_summaries.values():
            df = data['full_data']
            grouped_df = data['grouped_data']
            # Use the final group from grouped data for consistency
            final_group_start = int(grouped_df.iloc[-1]['start_sample']) - 1  # Convert to 0-based indexing
            final_group_end = int(grouped_df.iloc[-1]['end_sample'])
            final_samples = df.iloc[final_group_start:final_group_end]
            
            # Calculate metrics for final group
            relative_errors = np.abs(final_samples['pred_cost'] - final_samples['true_cost']) / final_samples['true_cost']
            accuracy_1pct = 100 * np.mean(relative_errors < 0.01)
            accuracy_5pct = 100 * np.mean(relative_errors < 0.05)
            avg_rel_error = np.mean(relative_errors)
            
            final_accuracies.append(accuracy_5pct)
            final_rel_errors.append(avg_rel_error)
        
        x = np.arange(len(model_names))
        width = 0.35
        
        ax6_twin = ax6.twinx()
        bars1 = ax6.bar(x - width/2, final_accuracies, width, label='Accuracy < 5%', alpha=0.7)
        bars2 = ax6_twin.bar(x + width/2, final_rel_errors, width, label='Relative Error', alpha=0.7, color='red')
        
        ax6.set_xlabel('Model')
        ax6.set_ylabel('Accuracy < 5% (%)', color='blue')
        ax6_twin.set_ylabel('Relative Error', color='red')
        ax6.set_title('Final Performance Comparison (Final Group)')
        ax6.set_xticks(x)
        ax6.set_xticklabels(model_names, rotation=45, ha='right')
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return model_summaries

def generate_performance_table(model_summaries, output_dir=None):
    """Generate detailed performance metrics table"""
    if output_dir is None:
        output_dir = OUTPUT_DIRECTORY
        
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL PERFORMANCE ANALYSIS")
    print("="*80)
    
    performance_data = []
    
    for model_name, data in model_summaries.items():
        df = data['full_data']
        grouped_df = data['grouped_data']
        
        # Use the final group from grouped data to ensure consistency
        # This is the same data used in plots and final accuracy calculations
        final_group_start = int(grouped_df.iloc[-1]['start_sample']) - 1  # Convert to 0-based indexing
        final_group_end = int(grouped_df.iloc[-1]['end_sample'])
        final_samples = df.iloc[final_group_start:final_group_end]
        
        relative_errors = np.abs(final_samples['pred_cost'] - final_samples['true_cost']) / final_samples['true_cost']
        
        # Check if cost_loss column exists
        if 'cost_loss' in grouped_df.columns:
            final_cost_loss = grouped_df['avg_cost_loss'].iloc[-1]
            cost_loss_stability = np.std(grouped_df['avg_cost_loss'].iloc[-min(10, len(grouped_df)):])
        else:
            final_cost_loss = grouped_df['avg_relative_error'].iloc[-1]
            cost_loss_stability = np.std(grouped_df['avg_relative_error'].iloc[-min(10, len(grouped_df)):])
        
        metrics = {
            'Model': model_name,
            'Training Samples': f"{len(df):,}",
            'Final Cost Loss': f"{final_cost_loss:.6f}",
            'Final Relative Error': f"{grouped_df['avg_relative_error'].iloc[-1]:.4f}",
            'Final Accuracy < 1%': f"{grouped_df['accuracy_1pct'].iloc[-1]:.1f}%",
            'Final Accuracy < 5%': f"{grouped_df['accuracy_5pct'].iloc[-1]:.1f}%",
            'Final Accuracy < 10%': f"{grouped_df['accuracy_10pct'].iloc[-1]:.1f}%",
            'Mean Relative Error': f"{np.mean(relative_errors):.4f}",
            'Median Relative Error': f"{np.median(relative_errors):.4f}",
            'Std Relative Error': f"{np.std(relative_errors):.4f}",
            'R² Score': f"{stats.pearsonr(final_samples['pred_cost'], final_samples['true_cost'])[0]**2:.4f}",
            'Convergence Stability': f"{cost_loss_stability:.6f}"
        }
        
        performance_data.append(metrics)
        
        # Print individual model summary
        print(f"\n{model_name.upper()}")
        print("-" * len(model_name))
        for key, value in metrics.items():
            if key != 'Model':
                print(f"{key:25}: {value}")
    
    # Save detailed results
    results_df = pd.DataFrame(performance_data)
    results_df.to_csv(f'{output_dir}/model_performance_comparison.csv', index=False)

    return results_df

def main():
    """Main function - can be called with or without command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate and compare surrogate neural network models')
    parser.add_argument('csv_paths', nargs='*', help='Paths to model CSV files (optional, uses CSV_FILES if not provided)')
    parser.add_argument('--output_dir', default=None, help=f'Output directory for results (default: {OUTPUT_DIRECTORY})')
    parser.add_argument('--groups', type=int, default=None, help=f'Number of groups to divide samples into (default: {NUM_GROUPS})')
    
    args = parser.parse_args()
    
    # Use command line arguments if provided, otherwise use script configuration
    csv_paths = args.csv_paths if args.csv_paths else CSV_FILES
    output_dir = args.output_dir if args.output_dir else OUTPUT_DIRECTORY
    num_groups = args.groups if args.groups else NUM_GROUPS
    
    # Update global variables if overridden
    NUM_GROUPS = num_groups
    OUTPUT_DIRECTORY = output_dir
    
    print("Hierarchical Surrogate Neural Network Model Evaluation")
    print("="*60)
    print(f"Evaluating {len(csv_paths)} model(s)")
    print(f"Output directory: {output_dir}")
    print(f"Sample grouping: {num_groups} groups")
    
    # Load and process data
    models_data, min_samples = load_and_process_csvs(csv_paths)
    
    if not models_data:
        print("\nNo valid CSV files found!")
        print("Please check the CSV_FILES list in the script or provide valid file paths.")
        return
    
    # Create visualizations
    model_summaries = create_comparison_plots(models_data, output_dir)
    
    # Generate performance table
    results_df = generate_performance_table(model_summaries, output_dir)
    
    print(f"\nResults saved to {output_dir}/")
    print("- model_comparison.png: Comprehensive comparison plots")
    print("- model_performance_comparison.csv: Detailed metrics table")

def run_evaluation():
    """Simplified function to run evaluation with script configuration"""
    print("Hierarchical Surrogate Neural Network Model Evaluation")
    print("="*60)
    print(f"Using configured CSV files: {CSV_FILES}")
    print(f"Output directory: {OUTPUT_DIRECTORY}")
    print(f"Sample grouping: {NUM_GROUPS} groups")
    
    # Load and process data
    models_data, min_samples = load_and_process_csvs()
    
    if not models_data:
        print("\nNo valid CSV files found!")
        print("Please check the CSV_FILES list in the script.")
        return
    
    # Create visualizations
    model_summaries = create_comparison_plots(models_data)
    
    # Generate performance table
    results_df = generate_performance_table(model_summaries)
    
    print(f"\nResults saved to {OUTPUT_DIRECTORY}/")
    print("- model_comparison.png: Comprehensive comparison plots")
    print("- model_performance_comparison.csv: Detailed metrics table")
    
    return results_df

if __name__ == "__main__":
    # Check if running with command line arguments or using script configuration
    if len(os.sys.argv) == 1:
        # No command line arguments - use script configuration
        print("No command line arguments provided. Using script configuration...")
        print("\nTo customize:")
        print("1. Edit the CSV_FILES list in this script")
        print("2. Or run with: python evaluate_model.py model1.csv model2.csv")
        print("3. Or call run_evaluation() if importing as module")
        print("\n" + "-"*60)
        
        # Run with script configuration
        run_evaluation()
    else:
        # Command line arguments provided - use main() function
        main()
