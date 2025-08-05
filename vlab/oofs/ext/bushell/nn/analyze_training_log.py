#!/usr/bin/env python3
"""
Analyze training log from hierarchical surrogate model to identify core issues
"""

import csv
import numpy as np

def analyze_training_log(csv_file):
    """Comprehensive analysis of training instability and performance issues"""
    
    # Load the data manually
    data = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for key in ['avg_loss', 'avg_loss_change', 'pred_cost', 'true_cost', 'structure_reg']:
                if key in row:
                    row[key] = float(row[key])
            data.append(row)
    
    print("=" * 60)
    print("HIERARCHICAL MODEL TRAINING ANALYSIS")
    print("=" * 60)
    
    # Basic statistics
    print(f"\n📊 BASIC STATISTICS:")
    print(f"Total epochs: {len(data)}")
    print(f"Training duration: {data[0]['datetime']} to {data[-1]['datetime']}")
    
    # Extract values for analysis
    avg_losses = [row['avg_loss'] for row in data]
    avg_loss_changes = [row['avg_loss_change'] for row in data]
    pred_costs = [row['pred_cost'] for row in data]
    true_costs = [row['true_cost'] for row in data]
    structure_regs = [row['structure_reg'] for row in data]
    
    # Loss analysis
    print(f"\n🔥 LOSS ANALYSIS:")
    print(f"Initial loss: {avg_losses[0]:,.0f}")
    print(f"Final loss: {avg_losses[-1]:,.0f}")
    print(f"Min loss: {min(avg_losses):,.0f}")
    print(f"Max loss: {max(avg_losses):,.0f}")
    
    loss_mean = sum(avg_losses) / len(avg_losses)
    loss_variance = sum((x - loss_mean)**2 for x in avg_losses) / len(avg_losses)
    loss_std = loss_variance**0.5
    print(f"Loss volatility (std/mean): {loss_std/loss_mean:.3f}")
    
    # Gradient explosion detection
    print(f"\n💥 GRADIENT EXPLOSION ANALYSIS:")
    large_changes = [abs(change) > 1e8 for change in avg_loss_changes]
    explosion_count = sum(large_changes)
    print(f"Large gradient changes (>100M): {explosion_count}/{len(data)} epochs")
    
    if explosion_count > 0:
        explosive_epochs = [i+1 for i, has_explosion in enumerate(large_changes) if has_explosion]
        print(f"Explosive epochs: {explosive_epochs[:10]}{'...' if len(explosive_epochs) > 10 else ''}")
    
    # Prediction accuracy
    print(f"\n🎯 PREDICTION ACCURACY:")
    pred_errors = [abs(pred - true) / true * 100 for pred, true in zip(pred_costs, true_costs)]
    
    pred_errors_sorted = sorted(pred_errors)
    n = len(pred_errors)
    median_error = pred_errors_sorted[n//2]
    mean_error = sum(pred_errors) / len(pred_errors)
    
    within_5_pct = sum(1 for err in pred_errors if err < 5) / len(pred_errors) * 100
    within_10_pct = sum(1 for err in pred_errors if err < 10) / len(pred_errors) * 100
    within_20_pct = sum(1 for err in pred_errors if err < 20) / len(pred_errors) * 100
    
    print(f"Mean prediction error: {mean_error:.1f}%")
    print(f"Median prediction error: {median_error:.1f}%")
    print(f"Predictions within 5% error: {within_5_pct:.1f}%")
    print(f"Predictions within 10% error: {within_10_pct:.1f}%")
    print(f"Predictions within 20% error: {within_20_pct:.1f}%")
    
    # Suspicious patterns
    print(f"\n🚨 SUSPICIOUS PATTERNS:")
    
    # Check for the infamous "10000" pattern
    constant_predictions = sum(1 for pred in pred_costs if abs(pred - 10000) < 0.01)
    constant_pct = constant_predictions / len(pred_costs) * 100
    print(f"Constant 10000 predictions: {constant_predictions}/{len(data)} epochs ({constant_pct:.1f}%)")
    
    if constant_pct > 80:
        print("🚨 CRITICAL: Model is outputting constant 10000 - this indicates:")
        print("   • Network not learning meaningful patterns")
        print("   • Possible gradient vanishing/explosion")
        print("   • Architecture or loss function issues")
        print("   • Model may be stuck in local minimum")
    
    # Check prediction variance
    pred_mean = sum(pred_costs) / len(pred_costs)
    pred_variance = sum((x - pred_mean)**2 for x in pred_costs) / len(pred_costs)
    pred_std = pred_variance**0.5
    
    if pred_std < pred_mean * 0.1:
        print(f"⚠️  Predictions are too stable (std={pred_std:.1f}, mean={pred_mean:.1f})")
    
    # Structure regularization trend
    print(f"\n🏗️  STRUCTURE REGULARIZATION:")
    struct_reg_start = structure_regs[0]
    struct_reg_end = structure_regs[-1]
    print(f"Structure reg: {struct_reg_start:.2f} → {struct_reg_end:.4f}")
    if struct_reg_end > 0:
        print(f"Reduction factor: {struct_reg_start/struct_reg_end:.0f}x")
    
    if struct_reg_end < 1:
        print("⚠️  Structure regularization became negligible")
    
    # Convergence analysis
    print(f"\n📈 CONVERGENCE ANALYSIS:")
    early_losses = avg_losses[:10]
    late_losses = avg_losses[-10:]
    
    early_loss_mean = sum(early_losses) / len(early_losses)
    late_loss_mean = sum(late_losses) / len(late_losses)
    
    improvement = (early_loss_mean - late_loss_mean) / early_loss_mean * 100
    print(f"Overall improvement: {improvement:.1f}%")
    
    # Check for oscillations
    abs_changes = [abs(change) for change in avg_loss_changes]
    abs_changes_sorted = sorted(abs_changes)
    threshold_90 = abs_changes_sorted[int(0.9 * len(abs_changes_sorted))]
    high_oscillation_epochs = sum(1 for change in abs_changes if change > threshold_90)
    oscillation_pct = high_oscillation_epochs / len(data) * 100
    print(f"High oscillation epochs: {high_oscillation_epochs}/{len(data)} ({oscillation_pct:.1f}%)")
    
    # Final diagnosis
    print(f"\n🔍 DIAGNOSIS:")
    issues = []
    
    if constant_pct > 50:
        issues.append("Constant prediction output (MAJOR)")
    if improvement < 10:
        issues.append("Poor convergence")
    if oscillation_pct > 30:
        issues.append("Training oscillations")
    if mean_error > 50:
        issues.append("Poor prediction accuracy")
    if explosion_count > len(data) * 0.1:
        issues.append("Frequent gradient explosions")
    
    if issues:
        print("Issues detected:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("No major issues detected")
    
    # Specific analysis for this case
    print(f"\n🔬 SPECIFIC ANALYSIS:")
    print(f"• Model predicts exactly 10000 in {constant_pct:.0f}% of cases")
    print(f"• True costs range from {min(true_costs):.0f} to {max(true_costs):.0f}")
    print(f"• Model is completely ignoring input parameters")
    print(f"• This suggests the network has collapsed to a trivial solution")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if constant_pct > 80:
        print("1. 🚨 URGENT: Complete architecture revision needed")
        print("   • The model has collapsed - not learning anything meaningful")
        print("   • Try the NEW hierarchical architecture we developed")
        print("   • Implement gradient clipping (max_norm=1.0)")
        print("   • Reduce learning rate significantly (try 1e-4 or 1e-5)")
        print("   • Add proper weight initialization")
    
    print("2. 🎯 Test the improved model:")
    print("   • Use the new StructureProcessingNet architecture")
    print("   • Apply the fixed coordinate scaling (sigmoid*500)")
    print("   • Use the intrinsic cost function")
    print("   • Train for more epochs with lower learning rate")
    
    print("3. 📊 Better monitoring:")
    print("   • Track individual loss components")
    print("   • Monitor gradient norms")
    print("   • Add early stopping based on validation loss")
    
    print("\n" + "=" * 60)
    return constant_pct, mean_error, improvement

if __name__ == "__main__":
    csv_file = "Run 6 Data/normal_hier_plant_surrogate_model.pt.csv"
    constant_pct, mean_error, improvement = analyze_training_log(csv_file)
