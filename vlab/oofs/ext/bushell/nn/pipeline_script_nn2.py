"""
This script uses batch_plant_surrogate_model.pt to train an optimizer network
to optimize plant parameters, which are then evaluated against real plant data.
It saves the best parameters and their corresponding cost.
It doesn't work well. 
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
import shutil
from utils_nn import build_parameter_file, generate_and_evaluate_in_dir
from plant_comparison_nn import calculate_cost, read_real_plants
from surrogate_nn1 import PlantSurrogateNet

output_dir="temp_plant"
surrogate_model_path = "batch_plant_surrogate_model.pt"
optimizer_model_path = "optimizer_net.pt"

diversity_amount = 0.1

param_min = torch.tensor([8.0, 2.8, -110.0, -4.0, 125.0, 3.0, 0.48, 0.8, 80.0, 170.0, 0.6, 0.88, 0.48])
param_max = torch.tensor([12.0, 3.2, 110.0, 4.0, 145.0, 7.0, 0.52, 1.2, 100.0, 190.0, 0.8, 0.92, 0.52])

class OptimizerNet(nn.Module):
    def __init__(self, input_dim=1, output_dim=13):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Sigmoid()  # constrain to [0,1]
        )
        self.param_min = param_min
        self.param_max = param_max

    def forward(self, x):
        out = self.net(x)
        return out * (self.param_max - self.param_min) + self.param_min

def clear_dir(output_dir=output_dir):
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        os.makedirs(output_dir)
        
if __name__ == "__main__":
    surrogate = PlantSurrogateNet()
    surrogate.load_state_dict(torch.load(surrogate_model_path))
    surrogate.eval()

    num_restarts = 5  # Number of optimizer restarts
    best_cost = float('inf')
    best_params = None

    # Get number of epochs from command line argument, default to 5000
    if len(sys.argv) > 1:
        try:
            num_epochs = int(sys.argv[1])
        except ValueError:
            print("Invalid argument for number of epochs, using default 5000.")
            num_epochs = 5000
    else:
        num_epochs = 5000

    batch_size = 32  # Already defined earlier
    # Compute total number of samples = num_epochs * batch_size
    num_runs = num_epochs * batch_size

    for restart in range(num_restarts):
        optimizer_net = OptimizerNet(input_dim=1, output_dim=13)
        if os.path.exists(optimizer_model_path) and restart == 0:
            optimizer_net.load_state_dict(torch.load(optimizer_model_path))
            print(f"Loaded existing optimizer network from {optimizer_model_path}")
        else:
            print(f"Starting optimizer network from scratch (restart {restart+1}/{num_restarts}).")

        # Use consistent optimizer settings: lr=1e-2 and weight_decay=1e-5
        optimizer = torch.optim.Adam(optimizer_net.parameters(), lr=1e-2, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
        
        # Get number of epochs as before; and use the same noise level:
        param_noise_std = 0.02  # same as pipeline_script_nn3
        
        for idx in range(num_runs):
            optimizer.zero_grad()
            noise = torch.rand(batch_size, 1)
            params = optimizer_net(noise)
            # Add parameter noise and clamp to valid range
            params_noisy = params + torch.randn_like(params) * param_noise_std
            params_noisy = torch.max(torch.min(params_noisy, param_max), param_min)
            pred_cost = surrogate(params_noisy)
            loss = pred_cost.mean()
            diversity_loss = -params_noisy.var(dim=0).mean()
            total_loss = loss + diversity_amount * diversity_loss
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            if idx % max(1, num_runs // 100) == 0 or idx == num_runs - 1:
                percent = 100 * (idx + 1) / num_runs
                denorm_cost = denormalize(pred_cost.mean().detach().item(), cost_mean, cost_std)
                sys.stdout.write(f"\rRestart {restart+1}/{num_restarts} - Progress: {percent:.1f}% - surrogate cost (denormalized)= {denorm_cost:.4f}")
                sys.stdout.flush()

        # After training, get the optimized plant parameters for this restart
        noise = torch.rand(1, 1)
        optimized_params = optimizer_net(noise).detach().numpy().flatten()
        print(f"\nOptimized parameters (restart {restart+1}):", optimized_params)

        # Evaluate surrogate cost for this set
        param_file = f"optimized_params_restart_{restart+1}.vset"
        build_parameter_file(param_file, optimized_params)
        real_bp, real_ep = read_real_plants()
        clear_dir()
        real_cost = generate_and_evaluate_in_dir(param_file, real_bp, real_ep, output_dir, cost_fn=calculate_cost)
        print(f"Real cost for optimized parameters (restart {restart+1}): {real_cost:.4f}")
        clear_dir()

        if real_cost < best_cost:
            best_cost = real_cost
            best_params = optimized_params
            # Save the best model
            torch.save(optimizer_net.state_dict(), optimizer_model_path)

    print(f"\nBest real cost after {num_restarts} restarts: {best_cost:.4f}")
    print("Best optimized parameters:", best_params)
    # Write best parameters to a file
    build_parameter_file("optimized_params_best.vset", best_params)
