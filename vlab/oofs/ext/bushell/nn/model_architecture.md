# Hierarchical Plant Surrogate Neural Network Architecture

## Overview
The hierarchical model decomposes plant cost prediction into three specialized modules that work together to predict the cost of comparing synthetic L-system generated plants with real plant data.

## Model Structure

```
Input: L-system Parameters (13 dimensions)
    ↓
┌─────────────────────────────────────────────────────────────────┐
│                    HierarchicalPlantSurrogateNet                │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              1. StructureGenerationNet                      ││
│  │                                                             ││
│  │  Input: Normalized L-system params [batch_size, 13]        ││
│  │  ┌─────────────────────────────────────────────────────────┐││
│  │  │ Shared Feature Extraction:                              │││
│  │  │   Linear(13 → 128) → ReLU                              │││
│  │  │   Linear(128 → 128) → ReLU                             │││
│  │  │   Linear(128 → 64) → ReLU                              │││
│  │  └─────────────────────────────────────────────────────────┘││
│  │                    ↓                                        ││
│  │  ┌─────────────────┐           ┌─────────────────┐          ││
│  │  │  Branch Points  │           │   End Points    │          ││
│  │  │  Generation:    │           │   Generation:   │          ││
│  │  │  Linear(64→128) │           │  Linear(64→128) │          ││
│  │  │  ReLU           │           │  ReLU           │          ││
│  │  │  Linear(128→150)│           │  Linear(128→150)│          ││
│  │  │  [50×3 output]  │           │  [50×3 output]  │          ││
│  │  └─────────────────┘           └─────────────────┘          ││
│  │          ↓                              ↓                   ││
│  │  Branch Points:                 End Points:                 ││
│  │  - 50×2 coordinates            - 50×2 coordinates           ││
│  │  - 50×1 existence probs        - 50×1 existence probs      ││
│  └─────────────────────────────────────────────────────────────┘│
│                                ↓                                │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              2. HungarianAssignmentNet                      ││
│  │                                                             ││
│  │  Input: Synthetic + Real structures [batch_size, 400]      ││
│  │  (bp_syn + ep_syn + bp_real + ep_real flattened)           ││
│  │  ┌─────────────────────────────────────────────────────────┐││
│  │  │ Structure Encoder:                                      │││
│  │  │   Linear(400 → 256) → ReLU                             │││
│  │  │   Linear(256 → 256) → ReLU                             │││
│  │  │   Linear(256 → 128) → ReLU                             │││
│  │  └─────────────────────────────────────────────────────────┘││
│  │                    ↓                                        ││
│  │  ┌─────────────────┐           ┌─────────────────┐          ││
│  │  │  Assignment     │           │   Cost          │          ││
│  │  │  Matrix Net:    │           │   Prediction:   │          ││
│  │  │  Linear(128→256)│           │  Linear(128→64) │          ││
│  │  │  ReLU           │           │  ReLU           │          ││
│  │  │  Linear(256→2500)│          │  Linear(64→1)   │          ││
│  │  │  Softmax        │           │  Softplus       │          ││
│  │  │  [50×50 matrix] │           │  [daily cost]   │          ││
│  │  └─────────────────┘           └─────────────────┘          ││
│  └─────────────────────────────────────────────────────────────┘│
│                                ↓                                │
│         Applied for each of 26 days (real plant timeline)       │
│                                ↓                                │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │               3. CostAggregationNet                         ││
│  │                                                             ││
│  │  Input: Daily costs [batch_size, 26]                       ││
│  │  ┌─────────────────────────────────────────────────────────┐││
│  │  │ Temporal Aggregation:                                   │││
│  │  │   Linear(26 → 64) → ReLU                               │││
│  │  │   Linear(64 → 32) → ReLU                               │││
│  │  │   Linear(32 → 1) → Softplus                            │││
│  │  └─────────────────────────────────────────────────────────┘││
│  │                                                             ││
│  │  Output: Final aggregated cost [batch_size, 1]             ││
│  └─────────────────────────────────────────────────────────────┘│
│                                ↓                                │
│             Output scaling and clamping (40k-120k)              │
└─────────────────────────────────────────────────────────────────┘
                                ↓
                    Final Cost Prediction
```

## Module Details

### 1. StructureGenerationNet
**Purpose**: Converts L-system parameters into synthetic plant structure points
- **Input**: 13 L-system parameters (normalized)
- **Output**: 
  - Branch points: 50×2 coordinates + 50×1 existence probabilities
  - End points: 50×2 coordinates + 50×1 existence probabilities
- **Parameters**: ~52K parameters

### 2. HungarianAssignmentNet
**Purpose**: Learns optimal assignment patterns between synthetic and real plant structures
- **Input**: Flattened concatenation of synthetic and real structures (400 dimensions)
- **Output**: 
  - Assignment matrix: 50×50 soft assignment weights
  - Daily cost: Single cost value for structure comparison
- **Parameters**: ~350K parameters

### 3. CostAggregationNet
**Purpose**: Aggregates costs across multiple days (plant growth timeline)
- **Input**: 26 daily costs (one for each day of plant growth)
- **Output**: Final aggregated cost prediction
- **Parameters**: ~3K parameters

## Data Flow

1. **L-system Parameters** → Structure Generation → **Synthetic Plant Structures**
2. **Synthetic + Real Structures** → Hungarian Assignment → **Daily Costs** (×26 days)
3. **Daily Costs** → Cost Aggregation → **Final Cost Prediction**

## Key Features

- **Hierarchical Decomposition**: Separates structure generation, assignment, and aggregation
- **Temporal Processing**: Handles 26-day plant growth timeline
- **Soft Assignment**: Uses learnable assignment weights instead of hard Hungarian algorithm
- **Robust Loss**: Combines relative error, Huber loss, and log-scale MSE
- **Normalization**: Input/output scaling with fallback defaults
- **Gradient Clipping**: Prevents exploding gradients during training

## Total Model Size
- **Total Parameters**: ~405K parameters
- **Structure Generation**: ~52K (13%)
- **Hungarian Assignment**: ~350K (86%)
- **Cost Aggregation**: ~3K (1%)

## Training Configuration
- **Optimizer**: AdamW with weight decay (1e-5)
- **Learning Rate**: 3e-4 (adaptive scheduling)
- **Loss Function**: Multi-component (relative error + Huber + log MSE)
- **Accuracy Threshold**: 5% relative error
- **Output Range**: Clamped to 40k-120k (typical cost range)
