"""
Script to read and analyze surrogate model training data from CSV files.
"""

import csv
import os
import matplotlib.pyplot as plt
import numpy as np

def read_surrogate_csv(csv_file):
    """Read surrogate model training data from CSV file"""
    if not os.path.exists(csv_file):
        print(f"CSV file not found: {csv_file}")
        return None
    
    data = {
        'run_numbers': [],
        'timestamps': [],
        'avg_losses': [],
        'avg_loss_changes': [],
        'losses': [],
        'pred_costs': [],
        'true_costs': [],
        'parameters': []
    }
    
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        print(f"CSV header: {header}")  # Debug: show header structure
        
        for row in reader:
            try:
                data['run_numbers'].append(int(row[0]))
                data['timestamps'].append(row[1])
                data['avg_losses'].append(float(row[2]))
                data['avg_loss_changes'].append(float(row[3]))
                data['losses'].append(float(row[4]))
                # Check the CSV header structure - costs might be in different columns
                if len(row) >= 10:  # New format with hierarchical loss components
                    data['pred_costs'].append(float(row[8]))  # pred_cost column
                    data['true_costs'].append(float(row[9]))  # true_cost column
                    # Parameters (columns 10 onwards)
                    params = [float(row[i]) for i in range(10, min(23, len(row)))]
                else:  # Old format
                    data['pred_costs'].append(float(row[5]))
                    data['true_costs'].append(float(row[6]))
                    # Parameters (columns 7-19)
                    params = [float(row[i]) for i in range(7, min(20, len(row)))]
                data['parameters'].append(params)
            except (ValueError, IndexError) as e:
                print(f"Error parsing row: {row}, error: {e}")
                continue
    
    return data

def analyze_surrogate_data(data):
    """Analyze the surrogate model training data"""
    if data is None:
        return
    
    print(f"Total training samples: {len(data['run_numbers'])}")
    print(f"Final average loss: {data['avg_losses'][-1]:.4f}")
    print(f"Final individual loss: {data['losses'][-1]:.4f}")
    
    # Calculate accuracy metrics
    pred_costs = np.array(data['pred_costs'])
    true_costs = np.array(data['true_costs'])
    
    # Print scale analysis
    print(f"\nSCALE ANALYSIS:")
    print(f"Predicted costs - Min: {np.min(pred_costs):.2f}, Max: {np.max(pred_costs):.2f}, Mean: {np.mean(pred_costs):.2f}, Std: {np.std(pred_costs):.2f}")
    print(f"True costs - Min: {np.min(true_costs):.2f}, Max: {np.max(true_costs):.2f}, Mean: {np.mean(true_costs):.2f}, Std: {np.std(true_costs):.2f}")
    print(f"Absolute errors - Min: {np.min(np.abs(pred_costs - true_costs)):.2f}, Max: {np.max(np.abs(pred_costs - true_costs)):.2f}, Mean: {np.mean(np.abs(pred_costs - true_costs)):.2f}")
    
    # Check for potential scale mismatch
    pred_range = np.max(pred_costs) - np.min(pred_costs)
    true_range = np.max(true_costs) - np.min(true_costs)
    scale_ratio = pred_range / (true_range + 1e-8)
    print(f"Scale ratio (pred/true range): {scale_ratio:.4f}")
    
    if scale_ratio > 10 or scale_ratio < 0.1:
        print("⚠️  WARNING: Significant scale mismatch detected!")
    
    relative_errors = np.abs(pred_costs - true_costs) / (np.abs(true_costs) + 1e-8)
    
    print(f"\nACCURACY METRICS:")
    print(f"Mean relative error: {np.mean(relative_errors):.4f}")
    print(f"Median relative error: {np.median(relative_errors):.4f}")
    print(f"Accuracy (rel_error < 0.01): {100 * np.mean(relative_errors < 0.01):.2f}%")
    print(f"Accuracy (rel_error < 0.1): {100 * np.mean(relative_errors < 0.1):.2f}%")
    
    # Additional percentile analysis
    print(f"95th percentile relative error: {np.percentile(relative_errors, 95):.4f}")
    print(f"99th percentile relative error: {np.percentile(relative_errors, 99):.4f}")

def plot_surrogate_training(data, save_plots=False):
    """Plot training progress"""
    if data is None:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot average loss over time
    axes[0, 0].plot(data['run_numbers'], data['avg_losses'])
    axes[0, 0].set_title('Average Loss Over Time')
    axes[0, 0].set_xlabel('Run Number')
    axes[0, 0].set_ylabel('Average Loss')
    axes[0, 0].grid(True)
    
    # Plot individual losses
    axes[0, 1].plot(data['run_numbers'], data['losses'])
    axes[0, 1].set_title('Individual Loss Over Time')
    axes[0, 1].set_xlabel('Run Number')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True)
    
    # Plot predicted vs true costs
    axes[1, 0].scatter(data['true_costs'], data['pred_costs'], alpha=0.5)
    min_cost = min(min(data['true_costs']), min(data['pred_costs']))
    max_cost = max(max(data['true_costs']), max(data['pred_costs']))
    axes[1, 0].plot([min_cost, max_cost], [min_cost, max_cost], 'r--', label='Perfect prediction')
    axes[1, 0].set_title('Predicted vs True Costs')
    axes[1, 0].set_xlabel('True Cost')
    axes[1, 0].set_ylabel('Predicted Cost')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot relative error over time
    pred_costs = np.array(data['pred_costs'])
    true_costs = np.array(data['true_costs'])
    relative_errors = np.abs(pred_costs - true_costs) / (np.abs(true_costs) + 1e-8)
    axes[1, 1].plot(data['run_numbers'], relative_errors)
    axes[1, 1].set_title('Relative Error Over Time')
    axes[1, 1].set_xlabel('Run Number')
    axes[1, 1].set_ylabel('Relative Error')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('surrogate_training_analysis.png', dpi=300, bbox_inches='tight')
        print("Plots saved to surrogate_training_analysis.png")
    
    plt.show()

if __name__ == "__main__":
    import sys

    directory = "Run 3 Data/"
    
    # Get CSV file from command line argument or use default
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = directory+"normal_hier_plant_surrogate_model.pt.csv"
    
    print(f"Reading surrogate model data from: {csv_file}")
    data = read_surrogate_csv(csv_file)
    
    if data:
        analyze_surrogate_data(data)
        plot_surrogate_training(data, save_plots=True)
    else:
        print("No data to analyze.")
