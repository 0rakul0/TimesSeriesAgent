#!/usr/bin/env python3
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from modelos.impact_predictor import ImpactPredictor
from utils.embedding_manager import EmbeddingManager
import pandas as pd
from utils.impact_dataset import gerar_dataset_eventos
from dotenv import load_dotenv

load_dotenv()



def preparar_dataset(csv_path, eventos_dir, ativo, emb_mgr, janela=5, horizon=4, batch_size=64):
    df = pd.read_csv(csv_path, parse_dates=["Date"]).sort_values("Date").ffill().bfill()
    X_emb, X_hist, Y, ids = gerar_dataset_eventos(eventos_dir, df, emb_mgr, ativo=ativo,
                                                 janela_atras=janela, horizon=horizon)
    return X_emb, X_hist, Y, ids

def to_torch(X_emb, X_hist, Y):
    X_emb_t = torch.tensor(X_emb, dtype=torch.float32)
    X_hist_t = torch.tensor(X_hist, dtype=torch.float32)
    Y_t = torch.tensor(Y, dtype=torch.float32)
    return TensorDataset(X_emb_t, X_hist_t, Y_t)

def train(csv, eventos, ativo,
          janela=5, horizon=4, epochs=100, batch=64, lr=1e-3,
          mlp_hidden=128, patience=8):
    emb_mgr = EmbeddingManager()
    X_emb, X_hist, Y, ids = preparar_dataset(csv, eventos, ativo,
                                            emb_mgr, janela=janela, horizon=horizon)

    if len(X_emb) == 0:
        raise RuntimeError("Nenhum evento válido encontrado. Verifique os arquivos evento_*.json e datas.")

    X_train_e, X_val_e, X_train_h, X_val_h, y_train, y_val = train_test_split(
        X_emb, X_hist, Y, test_size=0.15, random_state=42
    )

    train_ds = to_torch(X_train_e, X_train_h, y_train)
    val_ds = to_torch(X_val_e, X_val_h, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImpactPredictor(emb_dim=X_emb.shape[1], hist_len=X_hist.shape[1],
                            mlp_hidden=mlp_hidden, horizon=horizon).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = 1e9
    patience = patience
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, xh, yb in train_loader:
            xb, xh, yb = xb.to(device), xh.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb, xh)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        # validação
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, xh, yb in val_loader:
                xb, xh, yb = xb.to(device), xh.to(device), yb.to(device)
                pred = model(xb, xh)
                val_losses.append(float(loss_fn(pred, yb).item()))

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        print(f"Epoch {epoch} Train={avg_train:.6f} Val={avg_val:.6f}")

        if avg_val < best_val:
            best_val = avg_val
            wait = 0
            # salvar
            save_path = os.path.join("../modelos", f"impact_predictor_{ativo}.pt")
            torch.save({
                "model_state": model.state_dict(),
                "emb_dim": X_emb.shape[1],
                "hist_len": X_hist.shape[1],
                "horizon": horizon
            }, save_path)
            print("✔ Checkpoint salvo:", save_path)
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break

if __name__ == "__main__":
    train("../data/dados_petr4_brent.csv", "../output_noticias", "PETR4",janela=5, horizon=4, epochs=100, batch=64, lr=1e-3, mlp_hidden=128, patience=8)
    train("../data/dados_prio3_brent.csv", "../output_noticias", "PRIO3",janela=5, horizon=4, epochs=100, batch=64, lr=1e-3, mlp_hidden=128, patience=8)

