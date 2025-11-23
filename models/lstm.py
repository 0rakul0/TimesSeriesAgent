import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """
    LSTM normal — previsão absoluta de preço.
    hidden: 128
    layers: 2
    dropout: 0.2
    seq_len: 30 (definido externamente)
    """
    def __init__(self, input_size, hidden=128, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)
        output: preço(t)
        """
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last).squeeze(1)
