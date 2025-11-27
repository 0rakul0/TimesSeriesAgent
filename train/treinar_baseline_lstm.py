"""
--csv ../data/dados_petr4_brent.csv --epochs 120 --seq-len 30

"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from modelos.model_baseline_lstm import LSTMPrice

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Preparar dataset
# -----------------------------
def preparar_dados(csv_path, seq_len=30):
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = df.sort_values("Date").ffill().bfill()

    # features numéricas
    cols = [c for c in df.columns if c not in ["Date", "Ativo"]]
    data = df[cols].values

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    # target: Close
    target_idx = [i for i,c in enumerate(cols) if "Close" in c][0]

    X, y = [], []
    for i in range(len(scaled) - seq_len):
        X.append(scaled[i:i+seq_len])
        y.append(scaled[i+seq_len, target_idx])

    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(-1)

    return X, y, scaler, cols, target_idx


# -----------------------------
# Treinar modelo baseline
# -----------------------------
def treinar(csv_path, seq_len=30, epochs=40, lr=1e-3):
    print(f"[INFO] Carregando dados de {csv_path}...")

    X, y, scaler, cols, target_idx = preparar_dados(csv_path, seq_len)

    model = LSTMPrice(input_size=X.shape[2]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()

        pred = model(X.to(DEVICE))
        loss = loss_fn(pred, y.to(DEVICE))
        loss.backward()
        opt.step()

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs}  Loss={loss.item():.6f}")

    # salvar checkpoint
    ativo = os.path.basename(csv_path).replace(".csv","").replace("dados_","")
    save_path = os.path.join(BASE_DIR, "modelos", f"lstm_{ativo}.pt")

    ckpt = {
        "model_state": model.state_dict(),
        "scaler": scaler,
        "train_columns": cols,
        "target_idx": target_idx,
        "seq_len": seq_len
    }
    torch.save(ckpt, save_path)

    print(f"✔ Modelo salvo em: {save_path}")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    csv = "../data/dados_prio3_brent.csv"
    epochs = 120
    seq_len = 30
    treinar(csv, seq_len, epochs)
