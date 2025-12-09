import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Codificação posicional senoidal, padrão dos transformers.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class TransformerPrice(nn.Module):
    """
    Modelo Transformer Encoder para previsão de preços.
    Estrutura:
        - projeção inicial input → d_model
        - codificação posicional
        - N camadas TransformerEncoder
        - projeção final para 1 valor (preço futuro)
    """

    def __init__(
        self,
        input_size,
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.1
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x):
        """
        x shape: (batch, seq_len, input_size)
        """
        # projeção entrada → espaço do transformer
        x = self.input_proj(x)

        # codificação posicional
        x = self.pos_encoder(x)

        # transformer
        out = self.transformer_encoder(x)

        # pega o último timestep
        last = out[:, -1, :]

        # previsão final
        pred = self.fc_out(last)

        return pred
