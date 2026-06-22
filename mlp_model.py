"""
mlp_model.py
------------
PyTorch MLP architecture for the Enterprise Sentinel firewall. Kept in its own
module so the exact same class definition is used at train time and at load time
in the Streamlit app (torch.load of a state_dict needs the matching nn.Module).
"""

import torch
import torch.nn as nn

INPUT_DIM = 14  # number of engineered features


class PromptMLP(nn.Module):
    """Simple 3-hidden-layer feedforward classifier (2 logits)."""

    def __init__(self, input_dim: int = INPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        return self.net(x)
