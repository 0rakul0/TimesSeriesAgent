#!/usr/bin/env python3
"""
treinar_autoencoder.py
Treina o Autoencoder LSTM híbrido (reconstrução + previsão)
para qualquer ativo com CSV no formato dados_<ATIVO>_brent.csv.

Uso:
  python treinar_autoencoder.py
"""
import os, sys
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# importa o seu modelo AE
from modelos.model_lstm_autoencoder import LSTMAutoencoderPrice

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEQ_LEN = 15
EPOCHS  = 140
LR      = 1e-3
LAMBDA_RECON = 0.5   # peso da loss de reconstrução


# ============================================================
# Preparar dados
# ============================================================
def preparar_dados(csv_path, seq_len):
    df = pd.read_csv(csv_path, parse_dates=["Date"]).sort_values("Date").ffill().bfill()

    # Detecta ticker automaticamente
    tickers = [c.split("_")[1] for c in df.columns if c.startswith("Close_")]
    if not tickers:
        raise ValueError("Não foi possível identificar ticker no CSV.")
    ticker = tickers[0]  # ex: PRIO3.SA

    # Colunas do ativo + do Brent
    col_base = [
        f"Open_{ticker}",
        f"High_{ticker}",
        f"Low_{ticker}",
        f"Close_{ticker}",
        f"Volume_{ticker}",
        "Open_BZ=F", "High_BZ=F", "Low_BZ=F", "Close_BZ=F", "Volume_BZ=F"
    ]
    col_base = [c for c in col_base if c in df.columns]

    df = df.dropna(subset=[f"Close_{ticker}"])
    df_feat = df[col_base].astype(float)

    scaler = MinMaxScaler()
    arr = scaler.fit_transform(df_feat)

    X, y, Y_recon = [], [], []
    idx_close = df_feat.columns.get_loc(f"Close_{ticker}")

    for i in range(len(arr) - seq_len - 1):
        window = arr[i:i + seq_len]
        X.append(window)

        # forecast target (um passo à frente)
        y.append(arr[i + seq_len][idx_close])

        # reconstrução (a janela original)
        Y_recon.append(window)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)
    Y_recon = np.array(Y_recon, dtype=np.float32)

    return X, y, Y_recon, scaler, df_feat.columns.tolist()


# ============================================================
# Treinar Autoencoder
# ============================================================
def treinar(model, X, y_pred, y_recon):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_pred  = nn.MSELoss()
    loss_recon = nn.MSELoss()

    X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    y_pred_t = torch.tensor(y_pred, dtype=torch.float32).to(DEVICE)
    y_recon_t = torch.tensor(y_recon, dtype=torch.float32).to(DEVICE)

    for ep in range(1, EPOCHS + 1):
        opt.zero_grad()

        pred, recon = model(X_t)

        l1 = loss_pred(pred, y_pred_t)
        l2 = loss_recon(recon, y_recon_t)

        loss = l1 + LAMBDA_RECON * l2
        loss.backward()
        opt.step()

        if ep % 5 == 0 or ep == 1:
            print(f"Epoch {ep}/{EPOCHS} | Forecast={l1.item():.6f} | Recon={l2.item():.6f} | Total={loss.item():.6f}")


# ============================================================
# MAIN
# ============================================================
def treino_ae(csv, out):

    print("Carregando dados...")
    X, y_pred, Y_recon, scaler, cols = preparar_dados(csv, seq_len=SEQ_LEN)

    print(f"Treinando Autoencoder para out={out}")
    model = LSTMAutoencoderPrice(
        input_size=X.shape[2],
        hidden_size=128,
        latent_size=64,
        num_layers=2,
        dropout=0.1
    ).to(DEVICE)

    treinar(model, X, y_pred, Y_recon)

    ckpt = {
        "model_state": model.state_dict(),
        "scaler": scaler,
        "train_columns": cols,
        "seq_len": SEQ_LEN
    }
    torch.save(ckpt, out)
    print(f"✔ Modelo salvo em {out}")


if __name__ == "__main__":
    treino_ae(csv="../data/dados_petr4_brent.csv", out="../modelos/autoencoder_petr4.pt")
    treino_ae(csv="../data/dados_prio3_brent.csv", out="../modelos/autoencoder_prio3.pt")
    treino_ae(csv="../data/dados_exxo34_brent.csv", out="../modelos/autoencoder_exxo34.pt")
