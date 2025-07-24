import torch
import torch.nn as nn

class PlantParamNet(nn.Module):
    def __init__(self, input_dim=16, output_dim=13):
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