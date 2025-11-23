import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    """
    Transformer normal (SEM positional encoding)
    — Mesma arquitetura base usada no pipeline original
    """
    def __init__(self, input_size, hidden=128, heads=4, layers=2, dropout=0.1):
        super().__init__()

        self.embed = nn.Linear(input_size, hidden)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers
        )

        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        z = self.embed(x)
        out = self.transformer(z)
        last = out[:, -1, :]
        return self.fc(last).squeeze(1)
