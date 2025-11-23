import torch
import torch.nn as nn

class MLPLagResidual(nn.Module):
    """
    MLP residual — previsão de Δ(t)
    """
    def __init__(self, input_size, seq_len=30):
        super().__init__()
        self.seq_len = seq_len
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size * seq_len, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)
