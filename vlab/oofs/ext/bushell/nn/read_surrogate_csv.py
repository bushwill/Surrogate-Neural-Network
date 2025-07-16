import csv
import matplotlib.pyplot as plt

initial_csv_file = "surrogate_training_data.csv"
batch_csv_file = "batch_plant_surrogate_model.pt.csv"

# Read initial_csv_file
batch_size = 1000
batch_errors = []
current_batch = []

with open(initial_csv_file, "r") as f:
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

# Read batch_csv_file
batch_errors2 = []
current_batch2 = []

with open(batch_csv_file, "r") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        try:
            pred = float(row["pred_cost"])
            true = float(row["true_cost"])
            if true != 0:
                rel_error = abs(pred - true) / abs(true)
                current_batch2.append(rel_error)
        except Exception:
            continue
        if (i + 1) % batch_size == 0 and current_batch2:
            batch_avg = sum(current_batch2) / len(current_batch2)
            batch_errors2.append(batch_avg)
            current_batch2 = []

# Handle last batch if not empty
if current_batch2:
    batch_avg = sum(current_batch2) / len(current_batch2)
    batch_errors2.append(batch_avg)

plt.figure(figsize=(10, 5))
plt.plot([i * batch_size for i in range(1, len(batch_errors) + 1)], [e * 100 for e in batch_errors], marker='x', label='Initial CSV')
plt.plot([i * batch_size for i in range(1, len(batch_errors2) + 1)], [e * 100 for e in batch_errors2], marker='x', label='Batch CSV', color='orange')
plt.xlabel("Sample #")
plt.ylabel("Average Relative Error (%)")
plt.title(f"Average Relative Error per {batch_size} Samples")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()