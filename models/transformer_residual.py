import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerResidual(nn.Module):
    def __init__(self, input_size, hidden=128, heads=4, layers=2, dropout=0.1):
        super().__init__()

        self.embed = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.LayerNorm(hidden)
        )

        self.pos = PositionalEncoding(hidden)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dropout=dropout,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        z = self.embed(x)
        z = self.pos(z)
        out = self.encoder(z)
        last = out[:, -1, :]
        return self.fc(last).squeeze(1)  # Δ(t)
