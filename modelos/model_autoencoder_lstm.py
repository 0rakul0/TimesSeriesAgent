import torch
import torch.nn as nn

class LSTMAutoencoderPrice(nn.Module):
    """
    LSTM Autoencoder + Forecast Head
    - Encoder: extrai representação latente
    - Decoder: reconstrói a sequência de entrada (autoencoder)
    - FC final: previsão do preço futuro
    """

    def __init__(
        self,
        input_size,
        hidden_size=128,
        latent_size=64,
        num_layers=2,
        dropout=0.1
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers

        # ---------------------------------------------------------
        # 1) ENCODER LSTM
        # ---------------------------------------------------------
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        # pro latent
        self.to_latent = nn.Linear(hidden_size, latent_size)

        # ---------------------------------------------------------
        # 2) DECODER LSTM (autoencoder)
        # ---------------------------------------------------------
        self.decoder = nn.LSTM(
            input_size=latent_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        self.reconstruct = nn.Linear(hidden_size, input_size)

        # ---------------------------------------------------------
        # 3) Cabeça de previsão (forecast head)
        # ---------------------------------------------------------
        self.fc_pred = nn.Linear(latent_size, 1)

    def forward(self, x):
        """
        x: (batch, seq_len, input_size)

        Retorna:
            pred: previsão do preço (batch, 1)
            recon: reconstrução da sequência (batch, seq_len, input_size)
        """

        # =========================================================
        # ENCODER
        # =========================================================
        enc_out, (h_n, c_n) = self.encoder(x)

        # usa apenas o último hidden state
        h_last = enc_out[:, -1, :]           # (batch, hidden_size)

        # projeção para espaço latente
        z = self.to_latent(h_last)           # (batch, latent_size)

        # =========================================================
        # FORECAST HEAD (usa apenas z)
        # =========================================================
        pred = self.fc_pred(z)               # (batch, 1)

        # =========================================================
        # DECODER (Autoencoder)
        # =========================================================
        # Replicar o vetor latente ao longo da sequência
        seq_len = x.size(1)
        z_repeat = z.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, latent_size)

        dec_out, _ = self.decoder(z_repeat)
        recon = self.reconstruct(dec_out)    # (batch, seq_len, input_size)

        return pred, recon
