"""
Hierarchical surrogate neural network for plant cost prediction.
Decomposes the problem into specialized modules:
1. Structure Generation Network (generates branch/end points from parameters)
2. Structure Processing Network (analyzes generated structures for cost prediction)
3. Cost Aggregation Network (combines daily costs into final cost)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import time as t
import csv
import numpy as np
from plant_comparison_nn import make_matrix, make_index
from utils_nn import build_random_parameter_file, get_normalization_stats

def calculate_intrinsic_cost(bp_data, ep_data):
    """
    Calculate cost based on intrinsic plant structure properties
    This replaces the need for real plant comparison data
    """
    if not bp_data or not ep_data:
        return 30000.0  # Reduced base cost for minimal structure
    
    total_cost = 0.0
    num_days = len(bp_data)
    
    for day in range(num_days):
        bp_day = bp_data[day] if day < len(bp_data) else []
        ep_day = ep_data[day] if day < len(ep_data) else []
        
        # Calculate structure complexity cost
        num_bp = len(bp_day)
        num_ep = len(ep_day)
        
        # Basic structure cost (more points = higher cost) - reduced multipliers
        structure_cost = (num_bp * 100) + (num_ep * 80)
        
        # Calculate spatial distribution cost
        if num_ep > 1:
            ep_array = np.array(ep_day)
            # Cost based on spatial spread (larger spread = higher cost)
            if ep_array.ndim == 2 and ep_array.shape[0] > 1:
                spread = np.max(ep_array, axis=0) - np.min(ep_array, axis=0)
                spread_cost = np.sum(spread) * 8  # Reduced multiplier
            else:
                spread_cost = 80.0
        else:
            spread_cost = 40.0
            
        # Calculate branching efficiency cost
        if num_bp > 0 and num_ep > 0:
            branch_ratio = num_ep / max(num_bp, 1)
            # Penalize inefficient branching patterns
            efficiency_cost = abs(branch_ratio - 2.0) * 150  # Reduced penalty
        else:
            efficiency_cost = 300.0
            
        daily_cost = structure_cost + spread_cost + efficiency_cost
        total_cost += daily_cost
    
    # Add growth progression cost (later days should generally have more structure)
    if num_days > 1:
        final_ep = len(ep_data[-1]) if ep_data[-1] else 0
        initial_ep = len(ep_data[0]) if ep_data[0] else 0
        growth_cost = max(0, (initial_ep - final_ep)) * 80  # Reduced penalty
        total_cost += growth_cost
    
    # Clamp to more reasonable range that allows learning
    return max(5000.0, min(300000.0, total_cost))

def generateSurrogatePlant(param_file):
    """Generate plant using L-system and return intrinsic cost"""
    # setup call to lpfg
    lpfg_command = f"lpfg -w 306 256 lsystem.l view.v materials.mat contours.cset functions.fset functions.tset {param_file} > surrogate/lpfg_log.txt"

    if not os.path.exists("project"):
        os.system("g++ -o project -Wall -Wextra project.cpp -lm")
        
    if not os.path.exists("surrogate"):
        os.makedirs("surrogate")
    
    # Run lpfg to generate the plant
    os.system(lpfg_command)
    
    # Read the generated plant data
    from plant_comparison_nn import read_syn_plant_surrogate
    syn_bp, syn_ep = read_syn_plant_surrogate()
    
    # Calculate cost based on intrinsic properties
    cost = calculate_intrinsic_cost(syn_bp, syn_ep)
    
    return cost

def generate_and_evaluate(param_file):
    """Generate and evaluate plant using only intrinsic cost calculation"""
    return generateSurrogatePlant(param_file)

model_name = "normal_hier_plant_surrogate_model.pt"
accuracy_threshold = 0.05

class StructureGenerationNet(nn.Module):
    # Generates plant structure points (branch points and end points) from L-system parameters
    def __init__(self, input_dim=13, max_points=50):
        super().__init__()
        self.max_points = max_points
        
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        self.bp_net = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, max_points * 3)  # x, y, existence_prob for each point
        )
        
        self.ep_net = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, max_points * 3)  # x, y, existence_prob for each point
        )
        
        # Apply proper weight initialization
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights to prevent constant outputs"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot initialization
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
    def forward(self, x):
        features = self.feature_net(x)
        
        # Generate branch points
        bp_raw = self.bp_net(features).view(-1, self.max_points, 3)
        # Use sigmoid to ensure coordinates are in [0, 1] then scale to realistic pixel range
        bp_coords = torch.sigmoid(bp_raw[:, :, :2]) * 500  # Scale to [0, 500] pixel range
        bp_probs = torch.sigmoid(bp_raw[:, :, 2])  # existence_prob
        
        # Generate end points  
        ep_raw = self.ep_net(features).view(-1, self.max_points, 3)
        # Use sigmoid to ensure coordinates are in [0, 1] then scale to realistic pixel range
        ep_coords = torch.sigmoid(ep_raw[:, :, :2]) * 500  # Scale to [0, 500] pixel range
        ep_probs = torch.sigmoid(ep_raw[:, :, 2])  # existence_prob
        
        return bp_coords, bp_probs, ep_coords, ep_probs

class StructureProcessingNet(nn.Module):
    # Processes generated structures to extract meaningful features for cost prediction
    def __init__(self, max_points=50):
        super().__init__()
        self.max_points = max_points
        
        # Process generated structures (bp and ep coordinates and probabilities)
        # Input: bp_coords, bp_probs, ep_coords, ep_probs
        input_dim = max_points * 6  # (x,y,prob) for both bp and ep
        
        self.structure_analyzer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Extract geometric features from structures
        self.geometry_net = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Extract topological features
        self.topology_net = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Combine features for daily cost prediction
        self.daily_cost_net = nn.Sequential(
            nn.Linear(128, 64),  # 64 + 64 = 128 from geometry + topology
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()
        )
        
        # Apply proper weight initialization
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights to prevent constant outputs"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot initialization
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
    def forward(self, bp_coords, bp_probs, ep_coords, ep_probs):
        batch_size = bp_coords.size(0)
        
        # Flatten structure data
        bp_flat = torch.cat([
            bp_coords.view(batch_size, -1),  # [batch_size, max_points * 2]
            bp_probs.view(batch_size, -1)    # [batch_size, max_points]
        ], dim=1)  # [batch_size, max_points * 3]
        
        ep_flat = torch.cat([
            ep_coords.view(batch_size, -1),  # [batch_size, max_points * 2]
            ep_probs.view(batch_size, -1)    # [batch_size, max_points]
        ], dim=1)  # [batch_size, max_points * 3]
        
        structure_features = torch.cat([bp_flat, ep_flat], dim=1)  # [batch_size, max_points * 6]
        
        # Analyze structure
        analyzed = self.structure_analyzer(structure_features)
        
        # Extract different types of features
        geometry_features = self.geometry_net(analyzed)
        topology_features = self.topology_net(analyzed)
        
        # Combine features
        combined_features = torch.cat([geometry_features, topology_features], dim=1)
        
        # Predict daily cost
        daily_cost = self.daily_cost_net(combined_features)
        
        return daily_cost

class CostAggregationNet(nn.Module):
    # Aggregates costs across multiple days and assignment patterns
    def __init__(self, max_days=26):
        super().__init__()
        self.max_days = max_days
        
        self.temporal_net = nn.Sequential(
            nn.Linear(max_days, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()
        )
        
        # Apply proper weight initialization
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights to prevent constant outputs"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot initialization with smaller variance for output layer
                if module == list(self.temporal_net.modules())[-2]:  # Last linear layer
                    nn.init.xavier_uniform_(module.weight, gain=0.1)  # Smaller initial outputs
                else:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
    def forward(self, daily_costs):
        # daily_costs: [batch_size, num_days]
        return self.temporal_net(daily_costs)

class HierarchicalPlantSurrogateNet(nn.Module):
    # Hierarchical network combining all modules
    def __init__(self, input_dim=13, max_points=50, max_days=26):
        super().__init__()
        self.structure_gen = StructureGenerationNet(input_dim, max_points)
        self.structure_processor = StructureProcessingNet(max_points)
        self.cost_aggregator = CostAggregationNet(max_days)
        
        try:
            self.input_mean, self.input_std, self.output_mean, self.output_std = get_normalization_stats()
        except:
            print("Warning: Could not load normalization stats, using defaults")
            self.input_mean = torch.zeros(input_dim)
            self.input_std = torch.ones(input_dim)
            self.output_mean = torch.tensor([0.0])
            self.output_std = torch.tensor([1.0])
        
        if len(self.input_mean.shape) == 0:
            self.input_mean = self.input_mean.unsqueeze(0).repeat(input_dim)
        if len(self.input_std.shape) == 0:
            self.input_std = self.input_std.unsqueeze(0).repeat(input_dim)
        if len(self.output_mean.shape) == 0:
            self.output_mean = self.output_mean.unsqueeze(0)
        if len(self.output_std.shape) == 0:
            self.output_std = self.output_std.unsqueeze(0)
            
        # Apply proper weight initialization to prevent constant outputs
        self._init_weights()
        
    def _init_weights(self):
        """Initialize all network weights properly"""
        # Initialize sub-networks
        for module in [self.structure_gen, self.structure_processor, self.cost_aggregator]:
            if hasattr(module, '_init_weights'):
                module._init_weights()
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Normalize inputs (avoid division by zero)
        x_norm = (x - self.input_mean) / (self.input_std + 1e-8)
        
        # Generate synthetic plant structures
        bp_syn, bp_probs, ep_syn, ep_probs = self.structure_gen(x_norm)
        
        # Process structures to generate multiple daily costs
        # Generate 26 daily costs by processing the same structure with slight variations
        daily_costs = []
        
        for day in range(26):  
            # Add temporal variation to structure processing
            # Slight perturbation based on day to simulate temporal growth patterns
            day_factor = torch.tensor(day / 25.0, dtype=torch.float32)  # 0 to 1
            
            # Apply temporal scaling to coordinates to simulate growth
            temporal_scale = 1.0 + day_factor * 0.5  # Scale from 1.0 to 1.5
            bp_coords_day = bp_syn * temporal_scale
            ep_coords_day = ep_syn * temporal_scale
            
            # Process structure for this day
            day_cost = self.structure_processor(bp_coords_day, bp_probs, ep_coords_day, ep_probs)
            daily_costs.append(day_cost)
        
        daily_costs_tensor = torch.stack(daily_costs, dim=1).squeeze(-1)  # [batch_size, num_days]
        
        assert daily_costs_tensor.size(1) == 26, f"Expected 26 days but got {daily_costs_tensor.size(1)} days"
        
        final_cost = self.cost_aggregator(daily_costs_tensor)
        
        # Better output scaling - don't force minimum to 10000
        output_scale = self.output_std + 1e-8
        scaled_cost = final_cost * output_scale + self.output_mean
        
        # Use softer clamping with reasonable bounds based on actual data range
        # Allow predictions to start low and learn the correct range
        scaled_cost = torch.clamp(scaled_cost, min=5000.0, max=300000.0)
        
        return scaled_cost

def simplified_loss_function(pred_cost, true_cost, bp_syn, bp_probs, ep_syn, ep_probs):
    # Simplified loss function focusing purely on cost prediction accuracy
    
    # Primary cost prediction loss - focus on getting the scale right
    # Use MSE loss for direct optimization
    mse_loss = F.mse_loss(pred_cost, true_cost)
    
    # Add relative error component for better scaling
    relative_error = torch.abs(pred_cost - true_cost) / (torch.abs(true_cost) + 1e-8)
    rel_error_loss = torch.mean(relative_error)
    
    # Combine with emphasis on MSE for direct cost matching
    cost_loss = 0.7 * mse_loss + 0.3 * rel_error_loss
    
    # Light regularization to prevent extreme structure outputs
    structure_regularization = 0.001 * (torch.var(bp_syn) + torch.var(ep_syn))
    
    # Focus primarily on cost prediction
    total_loss = cost_loss + structure_regularization
    
    return total_loss, cost_loss, structure_regularization

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
                    prev_losses.append(float(row[5]))  # Use cost_loss column (index 5) for consistency
                except Exception:
                    pass
    else:
        start_run = 0
        prev_losses = []
    write_header = not os.path.exists(csv_file)
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            # 1: run #, 2: datetime, 3: avg_loss, 4: avg_loss_change, 5: total_loss, 6: cost_loss, 7: structure_reg, 8: pred_cost, 9: true_cost, 10-22: params
            writer.writerow(["run #", "datetime", "avg_loss", "avg_loss_change", "total_loss", "cost_loss", "structure_reg", "pred_cost", "true_cost"] + [f"param_{i}" for i in range(13)])
            
    model = HierarchicalPlantSurrogateNet()
    initial_lr = 1e-3 
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, weight_decay=1e-4)
    
    # initialize learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=200, min_lr=1e-6
    )

    if os.path.exists(model_name):
        try:
            model.load_state_dict(torch.load(model_name))
            print(f"Loaded existing model from {model_name}")
        except Exception as e:
            print(f"Could not load existing model: {e}. Creating new model.")
    else:
        print(f"No existing model found at {model_name}, creating new model.")
    
    model.train()
    print(f"Starting hierarchical training with {num_runs} plants...")
    
    # Note: We no longer need real plant data for training!
    # The model is now a true surrogate that learns from parameters → cost pairs
    
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
        
        # 2. Get true cost from L-system (this is the only L-system call we need!)
        true_cost = generate_and_evaluate("surrogate_params.vset")
        true_cost_tensor = torch.tensor([[true_cost]], dtype=torch.float32)
        
        # Collect data for normalization
        all_true_costs.append(true_cost)
        all_params.append(params)
        
        # Update normalization statistics every 100 samples (but don't update during training)
        if total_samples == 100:  # Only update once early in training for stability
            if len(all_true_costs) > 10:  # Need some data points
                cost_array = np.array(all_true_costs)
                param_array = np.array(all_params)
                
                # Update model normalization parameters once
                model.output_mean = torch.tensor([np.mean(cost_array)], dtype=torch.float32)
                model.output_std = torch.tensor([np.std(cost_array) + 1e-8], dtype=torch.float32)
                model.input_mean = torch.tensor(np.mean(param_array, axis=0), dtype=torch.float32)
                model.input_std = torch.tensor(np.std(param_array, axis=0) + 1e-8, dtype=torch.float32)
                print(f"\nUpdated normalization stats at sample {total_samples}")
                print(f"Cost mean: {model.output_mean.item():.1f}, std: {model.output_std.item():.1f}")
        
        # 3. Forward pass through hierarchical model (no real plant data needed!)
        try:
            pred_cost = model(params_tensor)
        except RuntimeError as e:
            print(f"Error in forward pass: {e}")
            print(f"params_tensor shape: {params_tensor.shape}")
            
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
        
        # 4. Compute simplified loss (no real plant data needed!)
        total_loss_val, cost_loss, structure_reg = simplified_loss_function(
            pred_cost, true_cost_tensor, bp_syn, bp_probs, ep_syn, ep_probs
        )
        
        # 5. Backpropagation with gradient clipping
        optimizer.zero_grad()
        total_loss_val.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Step the learning rate scheduler
        scheduler.step(cost_loss.item())
        
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
        
        # Get current learning rate for display
        current_lr = optimizer.param_groups[0]['lr']
        
        # Progress and ETA
        samples_done = idx + 1
        percent = 100.0 * samples_done / num_runs
        elapsed = time.time() - start_time
        samples_left = num_runs - samples_done
        avg_time_per_sample = elapsed / samples_done
        eta = samples_left * avg_time_per_sample
        eta_str = time.strftime('%H:%M:%S', time.gmtime(eta))
        
        # Print progress with simplified loss components
        sys.stdout.write(
            f"\rSample {start_run + idx + 1}, ({percent:.2f}%): "
            f"avg_loss={avg_loss:.4f}, "
            f"total_loss={total_loss_val.item():.4f}, "
            f"cost_loss={cost_loss.item():.4f}, "
            f"acc_1000={accuracy_1000:.2f}%, lr={current_lr:.6f}, ETA={eta_str}      "
        )
        sys.stdout.flush()
        
        clear_surrogate_dir()
        
        # Write to CSV with simplified loss components
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            run_number = start_run + idx + 1
            writer.writerow(
                [run_number, timestamp, f"{avg_loss:.4f}", f"{avg_loss_change:.4f}", 
                 f"{total_loss_val.item():.4f}", f"{cost_loss.item():.4f}", 
                 f"{structure_reg.item():.4f}", f"{pred_cost.item():.4f}", f"{true_cost:.4f}"] +
                [f"{p:.4f}" for p in params]
            )
    print()  # Newline after progress bar
    torch.save(model.state_dict(), model_name)
    print(f"Trained and saved hierarchical model to {model_name}")
    print(f"Total samples: {total_samples}, Total loss: {total_loss:.4f}, Average loss: {avg_loss:.4f}")
    
    # Print model architecture summary
    print("\nHierarchical Model Architecture:")
    print(f"- Structure Generation Net: {sum(p.numel() for p in model.structure_gen.parameters())} parameters")
    print(f"- Structure Processing Net: {sum(p.numel() for p in model.structure_processor.parameters())} parameters") 
    print(f"- Cost Aggregation Net: {sum(p.numel() for p in model.cost_aggregator.parameters())} parameters")
    print(f"- Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Optional: Save intermediate outputs for analysis
    if total_samples > 0:
        model.eval()
        with torch.no_grad():
            # Generate sample structure predictions
            sample_params = torch.tensor(params, dtype=torch.float32).unsqueeze(0)
            # Normalize inputs for structure generation
            params_norm = (sample_params - model.input_mean) / model.input_std
            bp_syn, bp_probs, ep_syn, ep_probs = model.structure_gen(params_norm)
            
            print(f"\nSample structure generation:")
            print(f"- Predicted branch points: {bp_probs.sum().item():.1f}")
            print(f"- Predicted end points: {ep_probs.sum().item():.1f}")
            print(f"- Branch point coordinates range: [{bp_syn.min().item():.1f}, {bp_syn.max().item():.1f}]")
            print(f"- End point coordinates range: [{ep_syn.min().item():.1f}, {ep_syn.max().item():.1f}]")