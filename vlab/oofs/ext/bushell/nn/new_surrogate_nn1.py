"""
Simple surrogate neural network for plant cost prediction.
"""

import torch
import torch.nn as nn
import os
import sys
import time as t
import csv
from plant_comparison_nn import read_real_plants
from utils_nn import build_random_parameter_file, generate_and_evaluate

model_name = "simple_plant_surrogate_model.pt"
accuracy_threshold = 0.01

class PlantSurrogateNet(nn.Module):
    def __init__(self, input_dim=13, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softplus()  # use Softplus to ensure outputs are positive smoothly
        )
    def forward(self, x):
        return self.net(x)

def clear_surrogate_dir():
    folder = "surrogate"
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
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

    # Read csv file
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

    for idx in range(num_runs):
        iter_start_time = time.time()
        clear_surrogate_dir()
        # 1. Generate random parameters and write to file
        params = build_random_parameter_file("surrogate_params.vset")
        params_tensor = torch.tensor(params, dtype=torch.float32).unsqueeze(0)
        # 2. Get true cost from L-system
        true_cost = generate_and_evaluate("surrogate_params.vset", real_bp, real_ep)
        true_cost_tensor = torch.tensor([[true_cost]], dtype=torch.float32)
        # 3. Predict cost with surrogate
        pred_cost = model(params_tensor)
        # 4. Compute loss and update surrogate
        loss = loss_fn(pred_cost, true_cost_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_samples += 1
        avg_loss = total_loss / total_samples
        timestamp = t.strftime("%Y-%m-%d %H:%M:%S")
        if total_samples > 1:
            prev_avg_loss = (total_loss - loss.item()) / (total_samples - 1)
            avg_loss_change = avg_loss - prev_avg_loss
        else:
            avg_loss_change = 0.0
        avg_loss_change_history.append(avg_loss_change)
        if len(avg_loss_change_history) > 1000:
            avg_loss_change_history.pop(0)
        avg_loss_change_1000 = sum(avg_loss_change_history) / len(avg_loss_change_history)
        # Compute relative error
        pred_cost_val = pred_cost.item()
        rel_error = abs(pred_cost_val - true_cost) / (abs(true_cost) + 1e-8)
        rel_error_history.append(rel_error)
        if len(rel_error_history) > 1000:
            rel_error_history.pop(0)
        # Accuracy: percent of rel_error < 0.1 in last 1000 samples
        accuracy_1000 = 100.0 * sum(e < accuracy_threshold for e in rel_error_history) / len(rel_error_history)
        # Progress and ETA
        samples_done = idx + 1
        percent = 100.0 * samples_done / num_runs
        elapsed = time.time() - start_time
        samples_left = num_runs - samples_done
        avg_time_per_sample = elapsed / samples_done
        eta = samples_left * avg_time_per_sample
        eta_str = time.strftime('%H:%M:%S', time.gmtime(eta))
        # Print progress with flush
        sys.stdout.write(
            f"\rSample {start_run + idx + 1}, ({percent:.2f}%): "
            f"avg_loss={avg_loss:.4f}, "
            f"avg_avg_loss_change={avg_loss_change_1000:.4f}, "
            f"acc_1000={accuracy_1000:.2f}%, ETA={eta_str}      "
        )
        sys.stdout.flush()
        clear_surrogate_dir()
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            run_number = start_run + idx + 1
            writer.writerow(
                [run_number, timestamp, f"{avg_loss:.4f}", f"{avg_loss_change:.4f}", f"{loss.item():.4f}", f"{pred_cost.item():.4f}", f"{true_cost:.4f}"] +
                [f"{p:.4f}" for p in params]
            )
    print()  # Newline after progress bar
    torch.save(model.state_dict(), model_name)
    print(f"Trained and saved new model to {model_name}")
    print(f"Total samples: {total_samples}, Total loss: {total_loss:.4f}, Average loss: {avg_loss:.4f}")