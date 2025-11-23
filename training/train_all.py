#!/usr/bin/env python3
"""
train_all_v6.py

Treina os modelos (normais e residuais) corretamente:
- Modelos "normal" preveem preço(t) (como antes).
- Modelos "residual" preveem Δ(t) = preço(t) - preço(t-1).
  • O Δ é escalado com um scaler separado (scaler_delta).
  • Checkpoint salva: model_state, scaler (preço), scaler_delta (quando aplicável),
    seq_len, target_col, train_columns, arch, is_residual, timestamp.

Uso: execute direto no root/src/training ou via subprocess do avaliador.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from datetime import datetime, timezone
import plotly.graph_objects as go

# Config (ajuste se necessário)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(ROOT_DIR, "data", "dados_combinados.csv")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
SEQ_LEN = 30
BATCH_SIZE = 64
LR = 5e-4
EPOCHS = 40
PATIENCE = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Import modelos (assume arquivos em src/models/)
import sys
SRC_DIR = os.path.join(ROOT_DIR, "src")
sys.path.insert(0, SRC_DIR)

from models.lstm import LSTMModel
from models.lstm_residual import LSTMResidual
from models.gru import GRUModel
from models.gru_residual import GRUResidual
from models.transformer import TransformerModel
from models.transformer_residual import TransformerResidual
from models.mlp import MLPLag
from models.mlp_residual import MLPLagResidual

# ===== Dataset helper =====
class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ===== create sequences (common) =====
def create_sequences_from_array(arr, target_idx, seq_len):
    X, y = [], []
    for i in range(len(arr) - seq_len):
        X.append(arr[i:i+seq_len])
        y.append(arr[i+seq_len, target_idx])
    return np.array(X), np.array(y)

# ===== EarlyStopping =====
class EarlyStopping:
    def __init__(self, patience=PATIENCE, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.counter = 0
        self.stop = False

    def step(self, v):
        if v < self.best - self.min_delta:
            self.best = v
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

# ===== train loop =====
def train_model(model, Xtr, ytr, Xval, yval, name):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    dl_tr = DataLoader(SeqDataset(Xtr, ytr), batch_size=BATCH_SIZE, shuffle=True)
    dl_val = DataLoader(SeqDataset(Xval, yval), batch_size=BATCH_SIZE, shuffle=False)

    stopper = EarlyStopping()

    best_state = None
    best_rmse = float("inf")
    losses = []

    for ep in range(1, EPOCHS + 1):
        model.train()
        batch_losses = []
        for xb, yb in dl_tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            batch_losses.append(loss.item())

        train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        losses.append(train_loss)

        # validação
        model.eval()
        preds, reals = [], []
        with torch.no_grad():
            for xb, yb in dl_val:
                xb = xb.to(DEVICE)
                out = model(xb).cpu().numpy()
                preds.extend(out)
                reals.extend(yb.numpy())

        preds = np.array(preds)
        reals = np.array(reals)
        rmse = float(np.sqrt(mean_squared_error(reals, preds))) if len(reals) else float("inf")

        print(f"{name} | Ep {ep}/{EPOCHS} | TrainLoss={train_loss:.6f} | ValRMSE={rmse:.6f}")
        if rmse < best_rmse:
            best_rmse = rmse
            best_state = model.state_dict()

        stopper.step(rmse)
        if stopper.stop:
            print(f"⛔ Early stopping ({name}) at epoch {ep}")
            break

    # restaura melhor
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, losses, best_rmse

# ===== salvar checkpoint (padronizado) =====
def save_checkpoint(model, scaler_price, scaler_delta, train_cols, name, arch, is_residual=False):
    ckpt = {
        "model_state": model.state_dict(),
        "scaler": scaler_price,
        "scaler_delta": scaler_delta,   # None se não houver
        "seq_len": SEQ_LEN,
        "target_col": TARGET_COL,
        "train_columns": train_cols,
        "arch": arch,
        "is_residual": bool(is_residual),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    path = os.path.join(MODEL_DIR, f"{name}.pt")
    torch.save(ckpt, path)
    print("💾 Checkpoint salvo:", path)

# ===== main pipeline =====
def main():
    print("\n🚀 Treinando modelos TimesSeriesAgent v6 (com residuals corretos)")
    print("Carregando dados...")

    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index).normalize()
    df = df.ffill().bfill()

    numeric = df.select_dtypes(include=[np.number]).copy()
    train_cols = numeric.columns.tolist()

    # scaler para preços (usado pelo modelos "normais")
    scaler_price = MinMaxScaler()
    numeric_scaled = pd.DataFrame(
        scaler_price.fit_transform(numeric),
        index=numeric.index,
        columns=numeric.columns
    )

    # construir Δ series (price diff) para o target
    # delta_series indexed like numeric.index but first value will be NaN
    delta_series = numeric[TARGET_COL].diff().fillna(0.0)  # first diff = 0.0 safe fallback

    # scaler para delta (apenas uma coluna)
    scaler_delta = MinMaxScaler()
    delta_scaled = scaler_delta.fit_transform(delta_series.values.reshape(-1,1)).flatten()

    # criar sequências para modelos normais (preço) e residuais (delta)
    # Para simplificar: usaremos a mesma matriz de features (numeric_scaled) como input
    arr_price = numeric_scaled.values
    target_idx = train_cols.index(TARGET_COL)
    X_price, y_price, dates_price = [], [], []
    for i in range(len(arr_price) - SEQ_LEN):
        X_price.append(arr_price[i:i+SEQ_LEN])
        y_price.append(arr_price[i+SEQ_LEN, target_idx])
        dates_price.append(numeric_scaled.index[i+SEQ_LEN])
    X_price = np.array(X_price); y_price = np.array(y_price)

    # para residual, y_delta uses delta_scaled
    arr_delta = numeric_scaled.values  # features still the same inputs (you can add engineered deltas too)
    X_delta, y_delta, dates_delta = [], [], []
    for i in range(len(arr_delta) - SEQ_LEN):
        X_delta.append(arr_delta[i:i+SEQ_LEN])
        y_delta.append(delta_scaled[i+SEQ_LEN])  # index-aligned with y_price
        dates_delta.append(numeric_scaled.index[i+SEQ_LEN])
    X_delta = np.array(X_delta); y_delta = np.array(y_delta)

    # split 80/20
    split = int(0.8 * len(X_price))
    Xtr_price, Xval_price = X_price[:split], X_price[split:]
    ytr_price, yval_price = y_price[:split], y_price[split:]

    Xtr_delta, Xval_delta = X_delta[:split], X_delta[split:]
    ytr_delta, yval_delta = y_delta[:split], y_delta[split:]

    print("Sequências criadas:", X_price.shape, "-> train/val:", Xtr_price.shape, Xval_price.shape)

    # modelos a treinar (nome: (instance, arch, is_residual_flag))
    modelos = {
        "LSTM_normal": (LSTMModel(numeric.shape[1]), "lstm_normal", False),
        "LSTM_residual": (LSTMResidual(numeric.shape[1]), "lstm_residual", True),

        "GRU_normal": (GRUModel(numeric.shape[1]), "gru_normal", False),
        "GRU_residual": (GRUResidual(numeric.shape[1]), "gru_residual", True),

        "Transformer_normal": (TransformerModel(numeric.shape[1]), "transformer_normal", False),
        "Transformer_residual": (TransformerResidual(numeric.shape[1]), "transformer_residual", True),

        "MLP_normal": (MLPLag(numeric.shape[1]), "mlp_normal", False),
        "MLP_residual": (MLPLagResidual(numeric.shape[1]), "mlp_residual", True)
    }

    results = []

    for name, (model, arch, is_residual) in modelos.items():
        print(f"\n===== Treinando {name} (residual={is_residual}) =====")
        if is_residual:
            # treinar no espaço delta (ytr_delta)
            model, losses, rmse = train_model(model, Xtr_delta, ytr_delta, Xval_delta, yval_delta, name)
            # salvar checkpoint: incluir scaler_delta (o scaler de delta)
            save_checkpoint(model, scaler_price, scaler_delta, train_cols, name, arch, is_residual=True)
        else:
            model, losses, rmse = train_model(model, Xtr_price, ytr_price, Xval_price, yval_price, name)
            save_checkpoint(model, scaler_price, None, train_cols, name, arch, is_residual=False)

        # salvar learning curve
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=losses, mode="lines"))
        fig.update_layout(title=f"Learning Curve - {name}", xaxis_title="epoch", yaxis_title="loss")
        fig.write_html(os.path.join(OUTPUT_DIR, f"{name}_learning.html"))

        results.append((name, rmse))
        print(f"🏁 {name} finalizado — best val RMSE (train-space) = {rmse:.6f}")

    # ranking
    df_rank = pd.DataFrame(results, columns=["modelo", "rmse_val"])
    df_rank.to_csv(os.path.join(OUTPUT_DIR, "ranking_modelos_v6.csv"), index=False)
    print("\n🏁 Treino finalizado!")
    print(df_rank)


# ===== Constants (below) =====
# define TARGET_COL constant after function definitions to avoid forward name errors
TARGET_COL = "Close_PETR4.SA"  # ajuste caso seja outro nome na sua base

if __name__ == "__main__":
    main()
