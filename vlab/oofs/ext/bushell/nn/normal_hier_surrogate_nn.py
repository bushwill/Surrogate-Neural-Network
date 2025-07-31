"""
Hierarchical surrogate neural network for plant cost prediction.
Decomposes the problem into specialized modules:
1. Structure Generation Network (generates branch/end points from parameters)
2. Hungarian Assignment Network (learns optimal assignment patterns)
3. Cost Aggregation Network (combines assignments into final cost)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import time as t
import csv
import numpy as np
from plant_comparison_nn import read_real_plants, make_matrix, make_index
from utils_nn import build_random_parameter_file, generate_and_evaluate, get_normalization_stats

model_name = "normal_hier_plant_surrogate_model.pt"
accuracy_threshold = 0.05  # 5% accuracy threshold (more realistic than 1%)

class StructureGenerationNet(nn.Module):
    """Generates plant structure points (branch points and end points) from L-system parameters"""
    def __init__(self, input_dim=13, max_points=50):
        super().__init__()
        self.max_points = max_points
        
        # Shared feature extraction
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Branch point generation (x, y coordinates + existence probability)
        self.bp_net = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, max_points * 3)  # x, y, existence_prob for each point
        )
        
        # End point generation (x, y coordinates + existence probability)
        self.ep_net = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, max_points * 3)  # x, y, existence_prob for each point
        )
        
    def forward(self, x):
        features = self.feature_net(x)
        
        # Generate branch points
        bp_raw = self.bp_net(features).view(-1, self.max_points, 3)
        bp_coords = bp_raw[:, :, :2] * 200  # Scale coordinates to reasonable range
        bp_probs = torch.sigmoid(bp_raw[:, :, 2])  # Existence probabilities
        
        # Generate end points
        ep_raw = self.ep_net(features).view(-1, self.max_points, 3)
        ep_coords = ep_raw[:, :, :2] * 200  # Scale coordinates to reasonable range
        ep_probs = torch.sigmoid(ep_raw[:, :, 2])  # Existence probabilities
        
        return bp_coords, bp_probs, ep_coords, ep_probs

class HungarianAssignmentNet(nn.Module):
    """Learns to predict optimal assignment patterns and costs"""
    def __init__(self, max_points=50):
        super().__init__()
        self.max_points = max_points
        
        # Process pairs of structures (synthetic vs real)
        # Input: bp_syn, ep_syn, bp_real, ep_real each [batch_size, max_points, 2]
        # Flattened: [batch_size, max_points * 2 * 4] = [batch_size, max_points * 8]
        input_dim = max_points * 8
        
        self.structure_encoder = nn.Sequential(
            nn.Linear(input_dim, 256),  # Fixed input dimension
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Predict assignment matrix (soft assignment weights)
        self.assignment_net = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, max_points * max_points),  # Assignment matrix
            nn.Softmax(dim=-1)
        )
        
        # Predict assignment costs
        self.cost_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()
        )
        
    def forward(self, bp_syn, ep_syn, bp_real, ep_real):
        batch_size = bp_syn.size(0)
        
        # Debug: Print tensor shapes
        # print(f"bp_syn shape: {bp_syn.shape}, ep_syn shape: {ep_syn.shape}")
        # print(f"bp_real shape: {bp_real.shape}, ep_real shape: {ep_real.shape}")
        
        # Flatten and concatenate structures
        # Each tensor is [batch_size, max_points, 2], flatten to [batch_size, max_points * 2]
        bp_syn_flat = bp_syn.view(batch_size, -1)      # [batch_size, max_points * 2]
        ep_syn_flat = ep_syn.view(batch_size, -1)      # [batch_size, max_points * 2]  
        bp_real_flat = bp_real.view(batch_size, -1)    # [batch_size, max_points * 2]
        ep_real_flat = ep_real.view(batch_size, -1)    # [batch_size, max_points * 2]
        
        structure_features = torch.cat([
            bp_syn_flat, ep_syn_flat, bp_real_flat, ep_real_flat
        ], dim=1)  # [batch_size, max_points * 8]
        
        # Debug: Print concatenated tensor shape
        # print(f"structure_features shape: {structure_features.shape}")
        
        encoded = self.structure_encoder(structure_features)
        
        # Predict assignment matrix and total cost
        assignment_weights = self.assignment_net(encoded).view(batch_size, self.max_points, self.max_points)
        total_cost = self.cost_net(encoded)
        
        return assignment_weights, total_cost

class CostAggregationNet(nn.Module):
    """Aggregates costs across multiple days and assignment patterns"""
    def __init__(self, max_days=26):
        super().__init__()
        self.max_days = max_days
        
        # Process temporal sequence of costs
        self.temporal_net = nn.Sequential(
            nn.Linear(max_days, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()
        )
        
    def forward(self, daily_costs):
        # daily_costs: [batch_size, num_days]
        return self.temporal_net(daily_costs)

class HierarchicalPlantSurrogateNet(nn.Module):
    """Hierarchical network combining all modules"""
    def __init__(self, input_dim=13, max_points=50, max_days=26):
        super().__init__()
        self.structure_gen = StructureGenerationNet(input_dim, max_points)
        self.hungarian_net = HungarianAssignmentNet(max_points)
        self.cost_aggregator = CostAggregationNet(max_days)
        
        # Initialize normalization stats with reasonable defaults
        try:
            self.input_mean, self.input_std, self.output_mean, self.output_std = get_normalization_stats()
        except:
            # Fallback to reasonable defaults if normalization stats don't exist
            print("Warning: Could not load normalization stats, using defaults")
            self.input_mean = torch.zeros(input_dim)
            self.input_std = torch.ones(input_dim)
            self.output_mean = torch.tensor([0.0])
            self.output_std = torch.tensor([1.0])
        
        # Ensure tensors are properly shaped
        if len(self.input_mean.shape) == 0:
            self.input_mean = self.input_mean.unsqueeze(0).repeat(input_dim)
        if len(self.input_std.shape) == 0:
            self.input_std = self.input_std.unsqueeze(0).repeat(input_dim)
        if len(self.output_mean.shape) == 0:
            self.output_mean = self.output_mean.unsqueeze(0)
        if len(self.output_std.shape) == 0:
            self.output_std = self.output_std.unsqueeze(0)
        
    def forward(self, x, real_bp_batch=None, real_ep_batch=None):
        batch_size = x.size(0)
        
        # Normalize inputs (avoid division by zero)
        x_norm = (x - self.input_mean) / (self.input_std + 1e-8)
        
        # Generate synthetic plant structures
        bp_syn, bp_probs, ep_syn, ep_probs = self.structure_gen(x_norm)
        
        if real_bp_batch is None or real_ep_batch is None:
            # During inference, return structure predictions
            return bp_syn, bp_probs, ep_syn, ep_probs
        
        # Compute Hungarian assignment and costs for each day
        daily_costs = []
        
        for day in range(real_bp_batch.size(1)):  # Iterate over days
            bp_real_day = real_bp_batch[:, day, :, :]  # [batch_size, max_points, 2]
            ep_real_day = real_ep_batch[:, day, :, :]  # [batch_size, max_points, 2]
            
            # Get assignment and cost for this day
            assignment_weights, day_cost = self.hungarian_net(bp_syn, ep_syn, bp_real_day, ep_real_day)
            daily_costs.append(day_cost)
        
        # Stack daily costs and aggregate
        daily_costs_tensor = torch.stack(daily_costs, dim=1).squeeze(-1)  # [batch_size, num_days]
        
        # Pad or truncate to max_days
        if daily_costs_tensor.size(1) < 26:
            padding = torch.zeros(batch_size, 26 - daily_costs_tensor.size(1))
            daily_costs_tensor = torch.cat([daily_costs_tensor, padding], dim=1)
        else:
            daily_costs_tensor = daily_costs_tensor[:, :26]
        
        # Final cost aggregation
        final_cost = self.cost_aggregator(daily_costs_tensor)
        
        # Apply output scaling more carefully to avoid extreme values
        output_scale = self.output_std + 1e-8
        scaled_cost = final_cost * output_scale + self.output_mean
        
        # Clamp to reasonable range based on typical true cost values (50k-100k range)
        scaled_cost = torch.clamp(scaled_cost, min=40000.0, max=120000.0)
        
        return scaled_cost

def prepare_real_plant_batch(real_bp, real_ep, max_points=50):
    """Convert real plant data to fixed-size tensors for batch processing"""
    num_days = len(real_bp)
    
    # Initialize tensors
    bp_batch = torch.zeros(1, num_days, max_points, 2)
    ep_batch = torch.zeros(1, num_days, max_points, 2)
    
    for day in range(num_days):
        # Branch points
        bp_day = real_bp[day]
        if len(bp_day) > 0:
            bp_array = torch.tensor(bp_day[:max_points], dtype=torch.float32)
            bp_batch[0, day, :min(len(bp_day), max_points), :] = bp_array
        
        # End points
        ep_day = real_ep[day]
        if len(ep_day) > 0:
            ep_array = torch.tensor(ep_day[:max_points], dtype=torch.float32)
            ep_batch[0, day, :min(len(ep_day), max_points), :] = ep_array
    
    return bp_batch, ep_batch

def hierarchical_loss_function(pred_cost, true_cost, bp_syn, bp_probs, ep_syn, ep_probs, real_bp, real_ep):
    """Multi-component loss function for hierarchical training"""
    
    # Primary cost prediction loss - focus on relative error
    relative_error = torch.abs(pred_cost - true_cost) / (torch.abs(true_cost) + 1e-8)
    
    # Use Huber loss for robustness to outliers
    huber_loss = F.huber_loss(pred_cost / (true_cost + 1e-8), torch.ones_like(true_cost), delta=0.1)
    
    # Log-scale MSE for better handling of different scales
    log_pred = torch.log(torch.clamp(pred_cost, min=1e-8))
    log_true = torch.log(torch.clamp(true_cost, min=1e-8))
    log_mse_loss = F.mse_loss(log_pred, log_true)
    
    # Combine different loss components with adaptive weighting
    rel_error_loss = torch.mean(relative_error)
    cost_loss = 0.4 * rel_error_loss + 0.3 * huber_loss + 0.3 * log_mse_loss
    
    # Structure generation loss (encourage reasonable point distributions)
    structure_loss = 0.0
    
    # Existence probability regularization (prevent too many/few points)
    bp_count_target = torch.tensor([min(len(day_bp), 50) for day_bp in real_bp]).float().mean()
    ep_count_target = torch.tensor([min(len(day_ep), 50) for day_ep in real_ep]).float().mean()
    
    bp_count_pred = bp_probs.sum()
    ep_count_pred = ep_probs.sum()
    
    count_loss = F.mse_loss(bp_count_pred, bp_count_target) + F.mse_loss(ep_count_pred, ep_count_target)
    
    # Spatial distribution loss (encourage reasonable coordinate ranges)
    coord_regularization = 0.01 * (torch.var(bp_syn) + torch.var(ep_syn))
    
    # Reduce the weight of auxiliary losses when cost prediction is improving
    aux_weight = min(0.05, 0.5 / (1.0 + cost_loss.item()))  # Even lower weight for auxiliary losses
    
    total_loss = cost_loss + aux_weight * count_loss + 0.005 * coord_regularization
    
    return total_loss, cost_loss, count_loss, coord_regularization

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
            writer.writerow(["run #", "datetime", "avg_loss", "avg_loss_change", "total_loss", "cost_loss", "count_loss", "coord_reg", "pred_cost", "true_cost"] + [f"param_{i}" for i in range(13)])
            
    model = HierarchicalPlantSurrogateNet()
    initial_lr = 3e-4  # Optimal learning rate based on current performance
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-5)  # Use AdamW with weight decay
    
    # Adaptive learning rate parameters - more aggressive since model is working well
    lr_decay_factor = 0.85  # More aggressive decay
    lr_decay_threshold = 0.02  # Decay when relative error drops below 2%
    lr_min = 5e-6  # Minimum learning rate
    lr_patience = 50  # Check every 50 samples for adaptation

    # Always load if exists, but always train
    if os.path.exists(model_name):
        try:
            model.load_state_dict(torch.load(model_name))
            print(f"Loaded existing model from {model_name}")
        except Exception as e:
            print(f"Could not load existing model: {e}. Creating new model.")
    else:
        print(f"No existing model found at {model_name}, creating new model.")
    
    model.train()
    print("Reading real plants...")
    real_bp, real_ep = read_real_plants()
    real_bp_batch, real_ep_batch = prepare_real_plant_batch(real_bp, real_ep)
    print(f"Starting hierarchical training with {num_runs} plants...")
    
    # Collect normalization statistics during training
    all_true_costs = []
    all_params = []
    
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
        
        # Collect data for normalization
        all_true_costs.append(true_cost)
        all_params.append(params)
        
        # Update normalization statistics every 100 samples
        if total_samples > 0 and total_samples % 100 == 0:
            if len(all_true_costs) > 10:  # Need some data points
                cost_array = np.array(all_true_costs)
                param_array = np.array(all_params)
                
                # Update model normalization parameters
                model.output_mean = torch.tensor([np.mean(cost_array)], dtype=torch.float32)
                model.output_std = torch.tensor([np.std(cost_array) + 1e-8], dtype=torch.float32)
                model.input_mean = torch.tensor(np.mean(param_array, axis=0), dtype=torch.float32)
                model.input_std = torch.tensor(np.std(param_array, axis=0) + 1e-8, dtype=torch.float32)
        
        # 3. Forward pass through hierarchical model
        try:
            pred_cost = model(params_tensor, real_bp_batch, real_ep_batch)
        except RuntimeError as e:
            print(f"Error in forward pass: {e}")
            print(f"params_tensor shape: {params_tensor.shape}")
            print(f"real_bp_batch shape: {real_bp_batch.shape}")
            print(f"real_ep_batch shape: {real_ep_batch.shape}")
            
            # Check structure generation output
            with torch.no_grad():
                params_norm = (params_tensor - model.input_mean) / model.input_std
                bp_syn, bp_probs, ep_syn, ep_probs = model.structure_gen(params_norm)
                print(f"bp_syn shape: {bp_syn.shape}")
                print(f"ep_syn shape: {ep_syn.shape}")
            raise e
        
        # Get intermediate outputs for loss computation  
        with torch.no_grad():
            # Normalize inputs for structure generation
            params_norm = (params_tensor - model.input_mean) / model.input_std
            bp_syn, bp_probs, ep_syn, ep_probs = model.structure_gen(params_norm)
        
        # 4. Compute hierarchical loss
        total_loss_val, cost_loss, count_loss, coord_reg = hierarchical_loss_function(
            pred_cost, true_cost_tensor, bp_syn, bp_probs, ep_syn, ep_probs, real_bp, real_ep
        )
        
        # 5. Backpropagation with gradient clipping
        optimizer.zero_grad()
        total_loss_val.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 6. Update statistics
        total_loss += cost_loss.item()  # Track only cost loss for comparison
        total_samples += 1
        avg_loss = total_loss / total_samples
        timestamp = t.strftime("%Y-%m-%d %H:%M:%S")
        
        if total_samples > 1:
            prev_avg_loss = (total_loss - cost_loss.item()) / (total_samples - 1)
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
        if total_samples % lr_patience == 0 and total_samples > 500:  # Check every lr_patience samples after shorter warmup
            avg_rel_error_1000 = sum(rel_error_history) / len(rel_error_history)
            
            # Decay learning rate if model is performing well
            if avg_rel_error_1000 < lr_decay_threshold and current_lr > lr_min:
                new_lr = max(current_lr * lr_decay_factor, lr_min)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                print(f"\nLearning rate decreased: {current_lr:.6f} -> {new_lr:.6f} (avg_rel_error={avg_rel_error_1000:.4f})")
            
            # Increase learning rate if model is struggling (relative error > 0.5)
            elif avg_rel_error_1000 > 0.5 and current_lr < initial_lr:
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
        
        # Print progress with hierarchical loss components
        sys.stdout.write(
            f"\rSample {start_run + idx + 1}, ({percent:.2f}%): "
            f"avg_loss={avg_loss:.4f}, "
            f"total_loss={total_loss_val.item():.4f}, "
            f"cost_loss={cost_loss.item():.4f}, "
            f"count_loss={count_loss.item():.4f}, "
            f"acc_1000={accuracy_1000:.2f}%, lr={current_lr:.6f}, ETA={eta_str}      "
        )
        sys.stdout.flush()
        
        clear_surrogate_dir()
        
        # Write to CSV with hierarchical loss components
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            run_number = start_run + idx + 1
            writer.writerow(
                [run_number, timestamp, f"{avg_loss:.4f}", f"{avg_loss_change:.4f}", 
                 f"{total_loss_val.item():.4f}", f"{cost_loss.item():.4f}", f"{count_loss.item():.4f}", 
                 f"{coord_reg.item():.4f}", f"{pred_cost.item():.4f}", f"{true_cost:.4f}"] +
                [f"{p:.4f}" for p in params]
            )
    print()  # Newline after progress bar
    torch.save(model.state_dict(), model_name)
    print(f"Trained and saved hierarchical model to {model_name}")
    print(f"Total samples: {total_samples}, Total loss: {total_loss:.4f}, Average loss: {avg_loss:.4f}")
    
    # Print model architecture summary
    print("\nHierarchical Model Architecture:")
    print(f"- Structure Generation Net: {sum(p.numel() for p in model.structure_gen.parameters())} parameters")
    print(f"- Hungarian Assignment Net: {sum(p.numel() for p in model.hungarian_net.parameters())} parameters") 
    print(f"- Cost Aggregation Net: {sum(p.numel() for p in model.cost_aggregator.parameters())} parameters")
    print(f"- Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Optional: Save intermediate outputs for analysis
    if total_samples > 0:
        model.eval()
        with torch.no_grad():
            # Generate sample structure predictions
            sample_params = torch.tensor(params, dtype=torch.float32).unsqueeze(0)
            bp_syn, bp_probs, ep_syn, ep_probs = model.structure_gen(sample_params)
            
            print(f"\nSample structure generation:")
            print(f"- Predicted branch points: {bp_probs.sum().item():.1f}")
            print(f"- Predicted end points: {ep_probs.sum().item():.1f}")
            print(f"- Branch point coordinates range: [{bp_syn.min().item():.1f}, {bp_syn.max().item():.1f}]")
            print(f"- End point coordinates range: [{ep_syn.min().item():.1f}, {ep_syn.max().item():.1f}]")