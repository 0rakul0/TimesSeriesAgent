#!/usr/bin/env python3
"""
treinar_petr4.py
Treina LSTM SINGLE-ASSET para PETR4 usando model_baseline_lstm.py

Uso:
  python treinar_ativo.py
  --csv ../data/dados_petr4_brent.csv --out ../models/lstm_petr4.pt
  --csv ../data/dados_prio3_brent.csv --out ../models/lstm_prio3.pt
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from modelos.model_baseline_lstm import LSTMPrice

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 30
EPOCHS = 80
LR = 1e-3
def preparar_dados(csv_path, seq_len=30):

    df = pd.read_csv(csv_path, parse_dates=["Date"]).sort_values("Date").ffill().bfill()

    # 1) Detectar automaticamente o ticker existente no CSV
    tickers = [c.split("_")[1] for c in df.columns if c.startswith("Close_")]
    if not tickers:
        raise ValueError("Não foi possível identificar ticker no CSV.")
    ticker = tickers[0]     # ex: PRIO3.SA

    # 2) Construir colunas base dinamicamente
    col_base = [
        f"Open_{ticker}",
        f"High_{ticker}",
        f"Low_{ticker}",
        f"Close_{ticker}",
        f"Volume_{ticker}",
        "Open_BZ=F", "High_BZ=F", "Low_BZ=F", "Close_BZ=F", "Volume_BZ=F"
    ]

    # 3) Filtrar somente as colunas presentes no CSV
    col_base = [c for c in col_base if c in df.columns]

    # 4) Dropar linhas onde o preço do ativo está ausente
    df = df.dropna(subset=[f"Close_{ticker}"])

    # 5) Criar matriz de features
    df_feat = df[col_base].astype(float)
    scaler = MinMaxScaler()
    arr = scaler.fit_transform(df_feat)

    # 6) Criar X,y com janelas
    X, y = [], []
    idx_close = df_feat.columns.get_loc(f"Close_{ticker}")

    for i in range(len(arr) - seq_len - 1):
        X.append(arr[i:i + seq_len])
        y.append(arr[i + seq_len][idx_close])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)

    return X, y, scaler, df_feat.columns.tolist()


def treinar(model, X, y):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    y_t = torch.tensor(y, dtype=torch.float32).to(DEVICE)

    for ep in range(1, EPOCHS+1):
        opt.zero_grad()
        pred = model(X_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        opt.step()

        if ep % 5 == 0 or ep == 1:
            print(f"Epoch {ep}/{EPOCHS} | Loss={loss.item():.6f}")

def main(csv, out):

    print("Carregando dados...")
    X, y, scaler, cols = preparar_dados(csv)

    print(f"Treinando LSTM para out={out}")
    model = LSTMPrice(input_size=X.shape[2]).to(DEVICE)
    treinar(model, X, y)

    ckpt = {
        "model_state": model.state_dict(),
        "scaler": scaler,
        "train_columns": cols,
        "seq_len": SEQ_LEN
    }
    torch.save(ckpt, out)
    print(f"✔ Modelo salvo em {out}")

if __name__ == "__main__":
    main(csv="../data/dados_petr4_brent.csv", out="../modelos/lstm_petr4.pt")
    main(csv="../data/dados_prio3_brent.csv", out="../modelos/lstm_prio3.pt")
