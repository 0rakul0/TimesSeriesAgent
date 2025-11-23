# model_baseline_lstm.py
"""
Treina um LSTM baseline que prevê o PREÇO (ex: Close_PETR4.SA).
Checkpoint salvo em ../models/lstm_baseline_price.pt contendo:
- model_state
- scaler
- target_col
- seq_len
- train_columns
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import random

# CONFIG
CAMINHO_DADOS = "../data/dados_combinados.csv"
MODELO_SAIDA = "../models/lstm_baseline_price.pt"
SEQ_LEN = 30
BATCH_SIZE = 64
EPOCHS = 40
LR = 5e-4
HIDDEN = 128
NUM_LAYERS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("../models", exist_ok=True)

class PriceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMPrice(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last).squeeze(1)

def preparar_dados():
    df = pd.read_csv(CAMINHO_DADOS, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index).normalize()
    df = df.sort_index()
    df = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill().fillna(0.0)
    # identifica coluna de preço alvo
    candidates = [c for c in df.columns if "PETR4" in c and "Close" in c]
    target_col = candidates[0] if candidates else df.columns[0]
    return df, target_col

def create_sequences(df_scaled, seq_len, target_col):
    arr = df_scaled.values.astype(np.float32)
    cols = list(df_scaled.columns)
    target_idx = cols.index(target_col)
    X, y, dates = [], [], []
    for i in range(len(arr) - seq_len):
        X.append(arr[i:i+seq_len, :])
        y.append(arr[i+seq_len, target_idx])
        dates.append(df_scaled.index[i+seq_len])
    return np.array(X), np.array(y), dates

def main():
    print("=== Treinamento Baseline (preço) ===")
    df, target_col = preparar_dados()
    nunique = df.nunique()
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        df = df.drop(columns=const_cols)

    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)

    X, y, dates = create_sequences(df_scaled, SEQ_LEN, target_col)
    print(f"[INFO] Sequências: X={X.shape}, y={y.shape}")

    n = len(X)
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    train_loader = DataLoader(PriceDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(PriceDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMPrice(X.shape[2], HIDDEN, NUM_LAYERS).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    for epoch in range(1, EPOCHS+1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        train_loss = np.mean(losses)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                vout = model(xb)
                val_losses.append(loss_fn(vout, yb).item())
        val_loss = np.mean(val_losses) if val_losses else float("nan")
        print(f"Epoch {epoch:02d} | Train {train_loss:.6e} | Val {val_loss:.6e}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "scaler": scaler,
                "target_col": target_col,
                "seq_len": SEQ_LEN,
                "train_columns": list(df.columns)
            }, MODELO_SAIDA)
            print("  -> Best model salvo.")
    print("Treino finalizado. Modelo salvo em", MODELO_SAIDA)

if __name__ == "__main__":
    main()
