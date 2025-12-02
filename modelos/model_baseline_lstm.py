import torch
import torch.nn as nn

class LSTMPrice(nn.Module):
    """
    Modelo LSTM univariado/multivariado para previsão de preços.
    Agora separado do script de treino.
    """

    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        out = out[:, -1, :]          # último timestep
        out = self.fc(out)
        return out

