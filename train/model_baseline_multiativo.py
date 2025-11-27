import torch
import torch.nn as nn


class MultiAssetLSTM(nn.Module):
    """
    LSTM Multi-Ativo com embedding para permitir que o modelo
    aprenda como cada ativo reage ao Brent e às próprias features.
    """

    def __init__(self, input_size, hidden_size, num_layers, n_ativos, emb_dim=6, dropout=0.1):
        super().__init__()

        # embedding categórico para ativos
        self.asset_emb = nn.Embedding(n_ativos, emb_dim)

        # LSTM recebe features + embedding
        self.lstm = nn.LSTM(
            input_size + emb_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, ativo_ids):
        """
        x: (batch, seq_len, input_size)
        ativo_ids: (batch,) -> índice inteiro do ativo
        """
        emb = self.asset_emb(ativo_ids)         # (batch, emb_dim)
        emb = emb.unsqueeze(1).repeat(1, x.size(1), 1)

        x_cat = torch.cat([x, emb], dim=-1)

        out, _ = self.lstm(x_cat)
        last = out[:, -1, :]                    # último passo da sequência

        return self.fc(last).squeeze(1)
