import torch
import torch.nn as nn

class ImpactPredictor(nn.Module):
    """
    Entrada:
      - emb (batch, emb_dim)
      - hist (batch, T) -> retornos percentuais (T = janela_atras+1)
    Saída:
      - preds (batch, horizon) -> retornos percentuais previstos D+1..D+horizon
    """

    def __init__(self, emb_dim=1536, hist_len=6, hist_hidden=64, mlp_hidden=128, horizon=4):
        super().__init__()
        self.hist_lstm = nn.LSTM(input_size=1, hidden_size=hist_hidden, num_layers=1, batch_first=True)
        self.emb_mlp = nn.Sequential(
            nn.Linear(emb_dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU()
        )
        self.final = nn.Sequential(
            nn.Linear(hist_hidden + mlp_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, horizon)
        )

    def forward(self, emb, hist):
        # emb: (B, emb_dim); hist: (B, T)
        h_in = hist.unsqueeze(-1)  # (B, T, 1)
        _, (h_n, _) = self.hist_lstm(h_in)
        h = h_n[-1]                # (B, hist_hidden)
        e = self.emb_mlp(emb)      # (B, mlp_hidden)
        z = torch.cat([h, e], dim=1)
        out = self.final(z)        # (B, horizon) RETURNS in percent units
        return out
