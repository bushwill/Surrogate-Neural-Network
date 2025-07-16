import torch
import torch.nn as nn
import os
import sys
import time as t
import subprocess
import shutil
import csv
import numpy as np
from plant_comparison_nn import read_real_plants
from utils_nn import build_random_parameter_file, generate_and_evaluate

accuracy_threshold = 0.01

# Default statistics for normalization (replace with your own if needed)
param_mean = np.array([
    9.88270607, 2.9858148, 1.61202144, -0.26396205, 135.48622668,
    5.13301503, 0.49941818, 0.99277462, 90.09817002, 180.16946128,
    0.70191325, 0.89834544, 0.49940136
])
param_std = np.array([
    1.08647035, 0.091338525, 91.2438917, 4.20708716, 4.97262854,
    1.04900474, 0.00917093112, 0.100055085, 3.00745595, 3.04930688,
    0.0509022923, 0.0106940044, 0.0100063813
])
cost_mean = 65403.89308560747
cost_std = 7702.132079934675

def normalize(x, mean, std):
    return (np.array(x) - mean) / (std + 1e-8)

def denormalize(x, mean, std):
    return np.array(x) * (std + 1e-8) + mean

class PlantSurrogateNet(nn.Module):
    def __init__(self, input_dim=13, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        return self.net(x)

def clear_surrogate_dir():
    folder = "surrogate"
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
            
if __name__ == "__main__":
    import time
    
    # Get number of runs from command line, default to 1000
    if len(sys.argv) > 1:
        try:
            num_runs = int(sys.argv[1])
        except ValueError:
            print("Invalid argument for number of runs, using default 1000.")
            num_runs = 1000
    else:
        num_runs = 1000

    batch_size = 16  # You can adjust this

    # Read csv file
    model_name = "batch_plant_surrogate_model.pt"
    csv_file = model_name + ".csv"
    if os.path.exists(csv_file):
        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            existing_rows = list(reader)
            start_run = len(existing_rows) - 1  # subtract header
            # Read previous losses (skip header)
            prev_losses = []
            for row in existing_rows[1:]:
                try:
                    prev_losses.append(float(row[4]))  # Use loss column (index 4)
                except Exception:
                    pass
    else:
        start_run = 0
        prev_losses = []
    write_header = not os.path.exists(csv_file)
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["run #", "datetime", "avg_loss", "avg_loss_change", "loss", "pred_cost", "true_cost"] + [f"param_{i}" for i in range(13)])
            
    model = PlantSurrogateNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Always load if exists, but always train
    if os.path.exists(model_name):
        model.load_state_dict(torch.load(model_name))
        print(f"Loaded existing model from {model_name}")
    else:
        print(f"No existing model found at {model_name}, creating new model.")
    model.train()
    print("Reading real plants...")
    real_bp, real_ep = read_real_plants()
    print(f"Starting training with {num_runs} plants...")
    
    total_loss = sum(prev_losses)
    total_samples = len(prev_losses)
    avg_loss_change_history = []
    rel_error_history = []

    start_time = time.time()

    params_batch = []
    true_costs_batch = []
    sample_times = []  # Track the time taken for each sample

    for idx in range(num_runs):
        sample_start_time = time.time()  # Start timing the current sample
        clear_surrogate_dir()
        # 1. Generate random parameters and write to file
        params = build_random_parameter_file("surrogate_params.vset")
        params_batch.append(params)
        # 2. Get true cost from L-system
        true_cost = generate_and_evaluate("surrogate_params.vset", real_bp, real_ep)
        true_costs_batch.append(true_cost)

        # Validate true_cost to ensure it's within a reasonable range
        if not np.isfinite(true_cost) or true_cost < 0:
            print(f"Warning: Invalid true_cost value {true_cost}. Skipping this sample.")
            continue

        # Normalize the current sample's parameters and true cost
        params_np = np.array([params])
        true_cost_np = np.array([true_cost])
        params_tensor = torch.tensor(normalize(params_np, param_mean, param_std), dtype=torch.float32)
        true_cost_tensor = torch.tensor(normalize(true_cost_np, cost_mean, cost_std), dtype=torch.float32).unsqueeze(1)

        # Predict cost and calculate loss for the current sample
        pred_cost_tensor = model(params_tensor)
        loss = loss_fn(pred_cost_tensor, true_cost_tensor)

        # Denormalize prediction for reporting
        pred_cost_val = denormalize(pred_cost_tensor.detach().squeeze().numpy(), cost_mean, cost_std)
        rel_error = abs(pred_cost_val - true_cost) / (abs(true_cost) + 1e-8)
        rel_error_history.append(rel_error)
        if len(rel_error_history) > 1000:
            rel_error_history.pop(0)

        total_loss += (pred_cost_val - true_cost) ** 2
        total_samples += 1
        avg_loss = total_loss / total_samples
        timestamp = t.strftime("%Y-%m-%d %H:%M:%S")
        if total_samples > 1:
            prev_avg_loss = (total_loss - (pred_cost_val - true_cost) ** 2) / (total_samples - 1)
            avg_loss_change = avg_loss - prev_avg_loss
        else:
            avg_loss_change = 0.0
        avg_loss_change_history.append(avg_loss_change)
        if len(avg_loss_change_history) > 1000:
            avg_loss_change_history.pop(0)
        avg_loss_change_1000 = sum(avg_loss_change_history) / len(avg_loss_change_history)

        # Accuracy: percent of rel_error < 0.1 in last 1000 samples
        accuracy_1000 = 100.0 * sum(e < accuracy_threshold for e in rel_error_history) / len(rel_error_history)

        # Progress and ETA: use the current run's counter instead of overall samples_done
        current_run = idx + 1
        overall_sample = start_run + current_run
        percent = 100.0 * current_run / num_runs  # progress for this run only
        sample_time = time.time() - sample_start_time  # Time taken for the current sample
        sample_times.append(sample_time)
        avg_time_per_sample = sum(sample_times) / len(sample_times)
        samples_left = max(0, num_runs - current_run)  # remaining in current run

        if samples_left > 0:
            eta = samples_left * avg_time_per_sample
            eta_str = time.strftime('%H:%M:%S', time.gmtime(eta))
        else:
            eta_str = "00:00:00"  # finished
        
        # Print information for the current sample using overall_sample for display
        print(
            f"{timestamp} - Sample {overall_sample}, ({percent:.2f}%): "
            f"avg_avg_loss_change={avg_loss_change_1000:.4f}, "
            f"avg_loss={avg_loss:.4f}, rel_error={rel_error:.4f}, "
            f"acc_1000={accuracy_1000:.2f}%, ETA={eta_str}"
        )

        # Save the current sample's data to the CSV file using overall_sample
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [overall_sample, timestamp, f"{avg_loss:.4f}", f"{avg_loss_change:.4f}", f"{loss.item():.4f}", f"{pred_cost_val:.4f}", f"{true_cost:.4f}"] +
                [f"{p:.4f}" for p in params]
            )

        # Add the current sample to the batch for training
        if len(params_batch) < batch_size:
            params_batch.append(params)
            true_costs_batch.append(true_cost)

        # If the batch is full, update the model
        if len(params_batch) == batch_size or idx == num_runs - 1:
            params_np = np.array(params_batch)
            true_costs_np = np.array(true_costs_batch)
            params_tensor = torch.tensor(normalize(params_np, param_mean, param_std), dtype=torch.float32)
            true_costs_tensor = torch.tensor(normalize(true_costs_np, cost_mean, cost_std), dtype=torch.float32).unsqueeze(1)
            pred_costs_tensor = model(params_tensor)
            loss = loss_fn(pred_costs_tensor, true_costs_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            params_batch = []
            true_costs_batch = []
    print()  # Newline after progress bar
    torch.save(model.state_dict(), model_name)
    print(f"Trained and saved new model to {model_name}")
    print(f"Total samples: {total_samples}, Total loss: {total_loss:.4f}, Average loss: {avg_loss:.4f}")