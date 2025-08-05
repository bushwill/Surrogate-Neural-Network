#!/usr/bin/env python3
"""
Quick test script to check coordinate generation ranges
"""
import torch
import sys
import os

# Add the path so we can import the model
sys.path.append('/home/pzu426/Surrogate Neural Network/vlab/oofs/ext/bushell/nn')

# Import the model
from normal_hier_surrogate_nn import HierarchicalPlantSurrogateNet
import numpy as np

def test_coordinate_generation():
    print("Testing coordinate generation in the new hierarchical model...")
    
    # Create model
    model = HierarchicalPlantSurrogateNet()
    model.eval()
    
    # Create some test parameters
    test_params = torch.randn(5, 13)  # 5 test samples, 13 parameters each
    
    with torch.no_grad():
        # Test structure generation
        print("\nTesting StructureGenerationNet...")
        x_norm = (test_params - model.input_mean) / (model.input_std + 1e-8)
        bp_coords, bp_probs, ep_coords, ep_probs = model.structure_gen(x_norm)
        
        print(f"BP coordinates shape: {bp_coords.shape}")
        print(f"BP coordinates range: [{bp_coords.min().item():.2f}, {bp_coords.max().item():.2f}]")
        print(f"BP probabilities range: [{bp_probs.min().item():.3f}, {bp_probs.max().item():.3f}]")
        print(f"Average BP probability: {bp_probs.mean().item():.3f}")
        
        print(f"\nEP coordinates shape: {ep_coords.shape}")
        print(f"EP coordinates range: [{ep_coords.min().item():.2f}, {ep_coords.max().item():.2f}]")  
        print(f"EP probabilities range: [{ep_probs.min().item():.3f}, {ep_probs.max().item():.3f}]")
        print(f"Average EP probability: {ep_probs.mean().item():.3f}")
        
        # Test full model
        print(f"\nTesting full model...")
        predicted_costs = model(test_params)
        print(f"Predicted costs: {predicted_costs.flatten().tolist()}")
        print(f"Cost range: [{predicted_costs.min().item():.1f}, {predicted_costs.max().item():.1f}]")
    
    print(f"\nModel architecture:")
    print(f"- Structure Generation Net: {sum(p.numel() for p in model.structure_gen.parameters())} parameters")
    print(f"- Structure Processing Net: {sum(p.numel() for p in model.structure_processor.parameters())} parameters")
    print(f"- Cost Aggregation Net: {sum(p.numel() for p in model.cost_aggregator.parameters())} parameters")
    print(f"- Total parameters: {sum(p.numel() for p in model.parameters())}")

if __name__ == "__main__":
    test_coordinate_generation()
