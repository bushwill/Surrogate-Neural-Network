import torch
import torch.nn as nn
import os
import sys
import time as t
import subprocess
import shutil
import csv
import numpy as np
from plant_comparison_nn import calculate_cost, read_real_plants
from utils_nn import build_random_parameter_file

accuracy_threshold = 0.01


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

def generateSurrogatePlant(param_file):
        # setup call to lpfg
        # lpfg_command = "lpfg -w 306 256 lsystem.l view.v materials.mat -a anim.a contours.cset functions.fset functions.tset loop_parameters.vset > log.txt"
        lpfg_command = f"lpfg -w 306 256 lsystem.l view.v materials.mat contours.cset functions.fset functions.tset {param_file} > surrogate/lpfg_log.txt"

        if not os.path.exists("project"):
            os.system("g++ -o project -Wall -Wextra project.cpp -lm")
            
        if not os.path.exists("surrogate"):
            os.mkdir("surrogate")

        # run lpfg  
        process = subprocess.Popen(['bash', '-c', lpfg_command])
        process.wait()
        os.system(f"./project 2454 2056 leafposition.dat > surrogate/output.txt")
        shutil.move("leafposition.dat", f"./surrogate")
        
def read_syn_plant_surrogate(file_name="surrogate/output.txt"):
    f = open(file_name, "r")
    lines = f.readlines()
    day_temp = 0
    syn_bp = []
    syn_ep = []
    syn_bp_day = []
    syn_ep_day = []
    day = []

    for line in lines:
        temp = line.split(" ")
        if temp[0] == "Day:":
            day_temp = int(temp[1])
            if day_temp>2:
                syn_bp.append(syn_bp_day)
                syn_ep.append(syn_ep_day)
                syn_bp_day = []
                syn_ep_day = []
        if (temp[0] != "Day:") & (day_temp > 1):
            if temp[0] == "I":
                syn_bp_day.append([int(temp[3]), int(temp[2])])
                day.append(day_temp)
            else:
                syn_ep_day.append([int(temp[3]), int(temp[2])])
                day.append(day_temp)

    if day_temp == 27:
        syn_bp.append(syn_bp_day)
        syn_ep.append(syn_ep_day)

    f.close()

    return syn_bp, syn_ep

def generate_and_evaluate(param_file, real_bp, real_ep):
    # Run lpfg to generate the synthetic plant
    generateSurrogatePlant(param_file)
    # Read the synthetic plant's endpoints and branchpoints for the latest run
    syn_bp, syn_ep = read_syn_plant_surrogate()
    # Use the first (or only) day's data for cost calculation
    cost = 0
    for i in range(min(len(syn_bp), len(real_bp))):
        cost += calculate_cost(syn_bp[i], syn_ep[i], real_bp[i], real_ep[i])
    return cost

def clear_surrogate_dir():
    folder = "surrogate"
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
    csv_file = "surrogate_training_data.csv"
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
            
    model_path = "plant_surrogate_model.pt"
    model = PlantSurrogateNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Always load if exists, but always train
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded existing model from {model_path}")
    else:
        print(f"No existing model found at {model_path}, creating new model.")
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
            f"\r{timestamp} - Sample {start_run + idx + 1}, ({percent:.2f}%): "
            f"avg_avg_loss_change={avg_loss_change_1000:.4f}, "
            f"avg_loss={avg_loss:.4f}, rel_error={rel_error:.4f}, "
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
    torch.save(model.state_dict(), model_path)
    print(f"Trained and saved new model to {model_path}")