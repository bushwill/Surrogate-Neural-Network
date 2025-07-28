import csv
import matplotlib.pyplot as plt
import os

# List of CSV files to analyze
csv_files = [
    "simple_plant_surrogate_model.pt.csv",
    "normal_plant_surrogate_model.pt.csv", 
    "normal_batch_16_plant_surrogate_model.pt.csv",
    "normal_batch_32_plant_surrogate_model.pt.csv"
]

# Define colors for each line
colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

def process_csv_file(csv_file, batch_size=1000):
    """Process a single CSV file and return batch errors"""
    batch_errors = []
    current_batch = []
    
    if not os.path.exists(csv_file):
        print(f"Warning: {csv_file} not found, skipping...")
        return batch_errors
    
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            try:
                pred = float(row["pred_cost"])
                true = float(row["true_cost"])
                if true != 0:
                    rel_error = abs(pred - true) / abs(true)
                    current_batch.append(rel_error)
            except Exception:
                continue
            if (i + 1) % batch_size == 0 and current_batch:
                batch_avg = sum(current_batch) / len(current_batch)
                batch_errors.append(batch_avg)
                current_batch = []
    
    # Handle last batch if not empty
    if current_batch:
        batch_avg = sum(current_batch) / len(current_batch)
        batch_errors.append(batch_avg)
    
    return batch_errors

def calculate_last_1000_rel_error(csv_file):
    """Calculate average relative error percentage for the last 1000 samples in a CSV file"""
    rel_errors = []
    
    if not os.path.exists(csv_file):
        print(f"Warning: {csv_file} not found, skipping...")
        return None
    
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)
        
        # Get last 1000 rows (or all if less than 1000)
        last_rows = all_rows[-1000:] if len(all_rows) >= 1000 else all_rows
        
        for row in last_rows:
            try:
                pred = float(row["pred_cost"])
                true = float(row["true_cost"])
                if true != 0:
                    rel_error = abs(pred - true) / abs(true)
                    rel_errors.append(rel_error)
            except Exception:
                continue
    
    if rel_errors:
        return (sum(rel_errors) / len(rel_errors)) * 100  # Convert to percentage
    return None

# Process all CSV files
batch_size = 100
plt.figure(figsize=(12, 6))

for i, csv_file in enumerate(csv_files):
    batch_errors = process_csv_file(csv_file, batch_size)
    if batch_errors:  # Only plot if we have data
        # Create label from filename (remove .csv extension and make it readable)
        label = csv_file.replace('.pt.csv', '').replace('_', ' ').title()
        
        # Use different colors but same marker for each line
        color = colors[i % len(colors)]
        
        plt.plot([j * batch_size for j in range(1, len(batch_errors) + 1)], 
                [e * 100 for e in batch_errors], 
                marker='o', 
                label=label, 
                color=color,
                linewidth=2,
                markersize=4)

plt.xlabel("Sample #")
plt.ylabel("Average Relative Error (%)")
plt.title(f"Average Relative Error per {batch_size} Samples - Model Comparison")
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Calculate and print relative error percentage for last 1000 samples
print("Average Relative Error Percentage (Last 1000 samples):")
print("-" * 50)
for csv_file in csv_files:
    rel_error_pct = calculate_last_1000_rel_error(csv_file)
    if rel_error_pct is not None:
        model_name = csv_file.replace('.pt.csv', '').replace('_', ' ').title()
        print(f"{model_name}: {rel_error_pct:.2f}%")
    else:
        print(f"{csv_file}: No data available")
print("-" * 50)