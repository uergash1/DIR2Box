import torch
import torch.nn as nn


class GeometricBox(nn.Module):
    def __init__(self, hidden_dim):
        super(GeometricBox, self).__init__()
        self.W_center = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_offset = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x):
        center = self.W_center(x)
        offset = torch.relu(self.W_offset(x))
        return center, offset