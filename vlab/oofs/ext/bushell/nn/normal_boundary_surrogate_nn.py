"""
Normalized surrogate neural network for plant cost prediction.
"""

import torch
import torch.nn as nn
import os
import sys
import time as t
import csv
from plant_comparison_nn import read_real_plants
from utils_nn import build_random_parameter_file, generate_and_evaluate, get_normalization_stats

model_name = "normal_boundary_plant_surrogate_model.pt"
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
        # Retrieve normalization stats for inputs and outputs
        self.input_mean, self.input_std, self.output_mean, self.output_std = get_normalization_stats()
        
    def forward(self, x):
        # Normalize inputs
        x_norm = (x - self.input_mean) / self.input_std
        out_norm = self.net(x_norm)
        # Denormalize outputs
        return out_norm * self.output_std + self.output_mean

def compute_boundary_penalty(params, param_min, param_max, penalty_weight=0.1, margin=0.1):
    """
    Compute soft boundary penalty to keep parameters within reasonable ranges
    while allowing some exploration beyond hard boundaries.
    """
    penalty = 0.0
    param_range = param_max - param_min
    
    for i, (p, p_min, p_max, p_range) in enumerate(zip(params, param_min, param_max, param_range)):
        # Create soft boundaries with margin
        soft_min = p_min - margin * p_range
        soft_max = p_max + margin * p_range
        
        # Penalty increases quadratically as we move away from acceptable range
        if p < soft_min:
            penalty += penalty_weight * ((soft_min - p) / p_range) ** 2
        elif p > soft_max:
            penalty += penalty_weight * ((p - soft_max) / p_range) ** 2
        # Small penalty for being near boundaries (encourages exploration inward)
        elif p < p_min + 0.05 * p_range or p > p_max - 0.05 * p_range:
            boundary_dist = min(p - p_min, p_max - p) / p_range
            penalty += penalty_weight * 0.1 * (0.05 - boundary_dist) ** 2 if boundary_dist < 0.05 else 0
    
    return penalty

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
            writer.writerow(["run #", "datetime", "avg_loss", "avg_loss_change", "loss", "pred_cost", "true_cost"] + [f"param_{i}" for i in range(13)] + ["boundary_penalty"])
            
    model = PlantSurrogateNet()
    initial_lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    loss_fn = nn.MSELoss()
    
    # Adaptive learning rate parameters
    lr_decay_factor = 0.95
    lr_decay_threshold = 0.1  # Decay when relative error drops below this
    lr_min = 1e-5  # Minimum learning rate
    lr_patience = 100  # Check every N samples for adaptation
    
    # Boundary constraint parameters
    param_min = torch.tensor([8.0, 2.8, -110.0, -4.0, 125.0, 3.0, 0.48, 0.8, 80.0, 170.0, 0.6, 0.88, 0.48])
    param_max = torch.tensor([12.0, 3.2, 110.0, 4.0, 145.0, 7.0, 0.52, 1.2, 100.0, 190.0, 0.8, 0.92, 0.52])
    boundary_penalty_weight = 0.05  # Weight for boundary penalty

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
        
        # 4. Add boundary penalty to encourage parameters within reasonable ranges
        boundary_penalty = compute_boundary_penalty(params, param_min, param_max, boundary_penalty_weight)
        boundary_penalty_tensor = torch.tensor([[boundary_penalty]], dtype=torch.float32)
        
        # 5. Compute total loss (surrogate loss + boundary penalty)
        surrogate_loss = loss_fn(pred_cost, true_cost_tensor)
        total_loss_val = surrogate_loss + boundary_penalty_tensor
        loss = total_loss_val  # For backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += surrogate_loss.item()  # Only track surrogate loss for statistics
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
        
        # Adaptive learning rate adjustment
        current_lr = optimizer.param_groups[0]['lr']
        if total_samples % lr_patience == 0 and total_samples > 1000:  # Check every lr_patience samples after warmup
            avg_rel_error_1000 = sum(rel_error_history) / len(rel_error_history)
            
            # Decay learning rate if model is performing well
            if avg_rel_error_1000 < lr_decay_threshold and current_lr > lr_min:
                new_lr = max(current_lr * lr_decay_factor, lr_min)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                print(f"\nLearning rate adapted: {current_lr:.6f} -> {new_lr:.6f} (avg_rel_error={avg_rel_error_1000:.4f})")
            
            # Increase learning rate if model is struggling (relative error > 0.2)
            elif avg_rel_error_1000 > 0.2 and current_lr < initial_lr:
                new_lr = min(current_lr / lr_decay_factor, initial_lr)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                print(f"\nLearning rate increased: {current_lr:.6f} -> {new_lr:.6f} (avg_rel_error={avg_rel_error_1000:.4f})")
        # Progress and ETA
        samples_done = idx + 1
        percent = 100.0 * samples_done / num_runs
        elapsed = time.time() - start_time
        samples_left = num_runs - samples_done
        avg_time_per_sample = elapsed / samples_done
        eta = samples_left * avg_time_per_sample
        eta_str = time.strftime('%H:%M:%S', time.gmtime(eta))
        # Print progress with flush (show boundary penalty info)
        boundary_info = f", boundary_penalty={boundary_penalty:.4f}" if boundary_penalty > 0.001 else ""
        sys.stdout.write(
            f"\rSample {start_run + idx + 1}, ({percent:.2f}%): "
            f"avg_loss={avg_loss:.4f}, "
            f"avg_avg_loss_change={avg_loss_change_1000:.4f}, "
            f"acc_1000={accuracy_1000:.2f}%, lr={current_lr:.6f}{boundary_info}, ETA={eta_str}      "
        )
        sys.stdout.flush()
        clear_surrogate_dir()
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            run_number = start_run + idx + 1
            writer.writerow(
                [run_number, timestamp, f"{avg_loss:.4f}", f"{avg_loss_change:.4f}", f"{surrogate_loss.item():.4f}", f"{pred_cost.item():.4f}", f"{true_cost:.4f}"] +
                [f"{p:.4f}" for p in params] + [f"{boundary_penalty:.4f}"]
            )
    print()  # Newline after progress bar
    torch.save(model.state_dict(), model_name)
    print(f"Trained and saved new model to {model_name}")
    print(f"Total samples: {total_samples}, Total loss: {total_loss:.4f}, Average loss: {avg_loss:.4f}")