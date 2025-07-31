"""
Baseline comparison experiments for hierarchical surrogate neural network research.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error, r2_score
import time

class SimpleMLP(nn.Module):
    """Simple MLP baseline for comparison"""
    def __init__(self, input_dim=13, hidden_dims=[128, 64, 32]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class DeepCNN(nn.Module):
    """CNN baseline that processes parameter vector as 1D sequence"""
    def __init__(self, input_dim=13):
        super().__init__()
        # Treat parameters as 1D sequence
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        # x: [batch_size, 13] -> [batch_size, 1, 13]
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)

def run_baseline_experiments(params_data, cost_data, test_ratio=0.2):
    """Run all baseline experiments and compare with hierarchical model"""
    
    # Split data
    n_samples = len(params_data)
    n_test = int(n_samples * test_ratio)
    
    # Random split
    indices = np.random.permutation(n_samples)
    train_idx, test_idx = indices[:-n_test], indices[-n_test:]
    
    X_train, X_test = params_data[train_idx], params_data[test_idx]
    y_train, y_test = cost_data[train_idx], cost_data[test_idx]
    
    results = {}
    
    # 1. Simple MLP
    print("Training Simple MLP...")
    mlp = SimpleMLP()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    
    for epoch in range(1000):
        optimizer.zero_grad()
        pred = mlp(X_train_tensor)
        loss = nn.MSELoss()(pred, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # Evaluate MLP
    mlp.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test)
        y_pred_mlp = mlp(X_test_tensor).numpy()
    
    rel_error_mlp = np.abs(y_pred_mlp.flatten() - y_test) / y_test
    results['Simple MLP'] = {
        'mean_rel_error': np.mean(rel_error_mlp),
        'accuracy_5pct': np.mean(rel_error_mlp < 0.05) * 100,
        'r2_score': r2_score(y_test, y_pred_mlp.flatten())
    }
    
    # 2. Random Forest
    print("Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    
    rel_error_rf = np.abs(y_pred_rf - y_test) / y_test
    results['Random Forest'] = {
        'mean_rel_error': np.mean(rel_error_rf),
        'accuracy_5pct': np.mean(rel_error_rf < 0.05) * 100,
        'r2_score': r2_score(y_test, y_pred_rf)
    }
    
    # 3. Gaussian Process (on subset due to computational cost)
    print("Training Gaussian Process...")
    subset_size = min(1000, len(X_train))
    subset_idx = np.random.choice(len(X_train), subset_size, replace=False)
    
    gp = GaussianProcessRegressor(random_state=42)
    gp.fit(X_train[subset_idx], y_train[subset_idx])
    y_pred_gp = gp.predict(X_test)
    
    rel_error_gp = np.abs(y_pred_gp - y_test) / y_test
    results['Gaussian Process'] = {
        'mean_rel_error': np.mean(rel_error_gp),
        'accuracy_5pct': np.mean(rel_error_gp < 0.05) * 100,
        'r2_score': r2_score(y_test, y_pred_gp)
    }
    
    # Print comparison table
    print("\nBaseline Comparison Results:")
    print("=" * 60)
    print(f"{'Method':<20} {'Mean Rel Error':<15} {'Accuracy@5%':<12} {'R² Score':<10}")
    print("-" * 60)
    
    for method, metrics in results.items():
        print(f"{method:<20} {metrics['mean_rel_error']:<15.4f} "
              f"{metrics['accuracy_5pct']:<12.1f}% {metrics['r2_score']:<10.4f}")
    
    return results

def ablation_study():
    """Test importance of each hierarchical component"""
    # TODO: Implement versions of your model with components removed
    # 1. Remove Hungarian Assignment Net
    # 2. Remove temporal aggregation
    # 3. Use different loss function components
    pass

if __name__ == "__main__":
    # Load your training data
    # params_data = np.load("training_params.npy")  # Shape: [n_samples, 13]
    # cost_data = np.load("training_costs.npy")     # Shape: [n_samples,]
    
    # results = run_baseline_experiments(params_data, cost_data)
    print("Run this after loading your training data!")
