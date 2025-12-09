#!/usr/bin/env python3
"""
treinar_transformer_ativo.py
Treina Transformer SINGLE-ASSET para PETR4, PRIO3, EXXO34.

Uso:
  python treinar_transformer_ativo.py
  --csv ../data/dados_<ATIVO>_brent.csv --out ../modelos/transformer_<ATIVO>.pt
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

from modelos.model_transformer_price import TransformerPrice

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 15
EPOCHS = 160
LR = 1e-4          # Transformers convergem melhor com LR menor


# =====================================================================
# PREPARAÇÃO DOS DADOS (MESMO PADRÃO DO LSTM)
# =====================================================================
def preparar_dados(csv_path, seq_len):

    df = pd.read_csv(csv_path, parse_dates=["Date"]).sort_values("Date").ffill().bfill()

    # detectar ticker dinamicamente
    tickers = [c.split("_")[1] for c in df.columns if c.startswith("Close_")]
    if not tickers:
        raise ValueError("Ticker não encontrado no CSV!")
    ticker = tickers[0]     # ex: PRIO3.SA

    # colunas-base usadas em TODOS os ativos
    col_base = [
        f"Open_{ticker}",
        f"High_{ticker}",
        f"Low_{ticker}",
        f"Close_{ticker}",
        f"Volume_{ticker}",
        "Open_BZ=F", "High_BZ=F", "Low_BZ=F", "Close_BZ=F", "Volume_BZ=F"
    ]

    # filtrar apenas as colunas existentes no CSV
    col_base = [c for c in col_base if c in df.columns]

    # remover linhas sem preço principal
    df = df.dropna(subset=[f"Close_{ticker}"])

    # matriz de features
    df_feat = df[col_base].astype(float)
    scaler = MinMaxScaler()
    arr = scaler.fit_transform(df_feat)

    # criar janelas LSTM/Transformer X,y
    X, y = [], []
    idx_close = df_feat.columns.get_loc(f"Close_{ticker}")

    for i in range(len(arr) - seq_len - 1):
        X.append(arr[i:i + seq_len])
        y.append(arr[i + seq_len][idx_close])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)

    return X, y, scaler, df_feat.columns.tolist()


# =====================================================================
# TREINO
# =====================================================================
def treinar(model, X, y):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    y_t = torch.tensor(y, dtype=torch.float32).to(DEVICE)

    for ep in range(1, EPOCHS + 1):
        opt.zero_grad()
        pred = model(X_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        opt.step()

        if ep % 5 == 0 or ep == 1:
            print(f"Epoch {ep}/{EPOCHS} | Loss={loss.item():.6f}")


# =====================================================================
# TREINO TRANSFORMER
# =====================================================================
def treino_transformer(csv, out):
    print("Carregando dados...")
    X, y, scaler, cols = preparar_dados(csv, seq_len=SEQ_LEN)

    print(f"Treinando TRANSFORMER para out={out}")

    model = TransformerPrice(
        input_size=X.shape[2],
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.1
    ).to(DEVICE)

    treinar(model, X, y)

    ckpt = {
        "model_state": model.state_dict(),
        "scaler": scaler,
        "train_columns": cols,
        "seq_len": SEQ_LEN
    }
    torch.save(ckpt, out)
    print(f"✔ Modelo salvo em {out}")


# =====================================================================
# MAIN (para treinar todos automaticamente)
# =====================================================================
if __name__ == "__main__":
    treino_transformer(
        csv="../data/dados_petr4_brent.csv",
        out="../modelos/transformer_petr4.pt"
    )

    treino_transformer(
        csv="../data/dados_prio3_brent.csv",
        out="../modelos/transformer_prio3.pt"
    )

    treino_transformer(
        csv="../data/dados_exxo34_brent.csv",
        out="../modelos/transformer_exxo34.pt"
    )
