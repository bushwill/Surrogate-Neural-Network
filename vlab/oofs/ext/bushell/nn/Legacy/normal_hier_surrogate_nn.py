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
from plant_comparison_nn import make_matrix, make_index, calculate_cost, read_real_plants
from utils_nn import (build_random_parameter_file, get_normalization_stats, generateSurrogatePlant, 
                      clear_surrogate_dir, setup_training_csv, 
                      log_training_step, print_training_progress, read_syn_plant_surrogate,
                      generate_and_evaluate)

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
            nn.LayerNorm(256),  # Use LayerNorm instead of BatchNorm1d
            nn.Linear(256, 256),
            nn.ReLU(),
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
            nn.Linear(128, 64),
            nn.ReLU(),
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
            self.input_mean, self.input_std, self.output_mean, self.output_std = get_normalization_stats("normal_hier_plant_surrogate_model.pt")
        except:
            print("Warning: Could not load normalization stats, using defaults")
            self.input_mean, self.input_std, self.output_mean, self.output_std = get_normalization_stats()
        
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
            # Exponential growth simulation: seed → mature plant over 26 days
            # Day 0: tiny seed (scale = 0.1)
            # Day 25: mature plant (scale = 1.0)
            # Uses exponential growth curve: scale = 0.1 * exp(growth_rate * day)
            growth_rate = np.log(10) / 25  # ln(10)/25 gives 10x growth over 25 days
            
            # Exponential growth: starts at 0.1x, grows to 1.0x by day 25
            temporal_scale = 0.1 * np.exp(growth_rate * day)
            temporal_scale = torch.tensor(temporal_scale, dtype=torch.float32)
            
            # Apply exponential growth scaling to coordinates
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

    # Setup training CSV and get previous state
    start_run, prev_losses, csv_file = setup_training_csv(model_name)
            
    model = HierarchicalPlantSurrogateNet()
    initial_lr = 1e-3 
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, weight_decay=1e-4)
    
    # initialize learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.9, patience=500, min_lr=1e-6
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
    
    # Load real plant data for comparison (just like the original model)
    real_bp, real_ep = read_real_plants()
    print(f"Loaded real plant data: {len(real_bp)} days")
    
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
        
        # 2. Get true cost from L-system using real plant comparison
        true_cost = generate_and_evaluate("surrogate_params.vset", real_bp, real_ep)
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
                print(f"Cost range in first 100 samples: [{np.min(cost_array):.1f}, {np.max(cost_array):.1f}]")
        
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
        
        # Get current learning rate before scheduler step
        prev_lr = optimizer.param_groups[0]['lr']
        
        # Step the learning rate scheduler
        scheduler.step(cost_loss.item())
        
        # Check if learning rate was changed by scheduler
        current_lr = optimizer.param_groups[0]['lr']
        if prev_lr != current_lr:
            print(f"\nLearning rate reduced by scheduler: {prev_lr:.6f} -> {current_lr:.6f} (cost_loss={cost_loss.item():.4f})")
        
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
        
        # Current learning rate already obtained above for change detection
        
        # Print progress with meaningful metrics
        print_training_progress(idx, num_runs, start_run, avg_loss, total_loss_val.item(), 
                               cost_loss.item(), accuracy_1000, current_lr, start_time,
                               rel_error=rel_error, pred_cost=pred_cost_val, true_cost=true_cost)
        
        clear_surrogate_dir()
        
        # Write to CSV with simplified loss components
        run_number = start_run + idx + 1
        log_training_step(csv_file, run_number, total_loss_val.item(), cost_loss.item(), 
                         structure_reg.item(), pred_cost.item(), true_cost, params, 
                         avg_loss, avg_loss_change)
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