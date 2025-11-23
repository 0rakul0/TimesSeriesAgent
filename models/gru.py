import torch
import torch.nn as nn

class GRUModel(nn.Module):
    """
    GRU normal — previsão absoluta
    """
    def __init__(self, input_size, hidden=128, layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.fc(last).squeeze(1)
