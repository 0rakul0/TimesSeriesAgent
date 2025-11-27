import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from train.model_baseline_multiativo import MultiAssetLSTM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = "../data/dados_multiativo.csv"
SAVE_MODEL = "./modelos/lstm_multiativo.pt"

SEQ_LEN = 30
BATCH = 32
EPOCHS = 120
LR = 1e-3


# ==================== DATASET ====================
class PriceDataset(Dataset):
    def __init__(self, X, y, ativos):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.ativos = torch.tensor(ativos, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            self.X[idx],
            self.y[idx],
            self.ativos[idx]
        )


# ==================== CARREGAR E PREPARAR DADOS ====================
def preparar_dados(path):
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values(["Date", "Ativo"]).reset_index(drop=True)

    # Transformar Ativo em ID
    df["Ativo_ID"], ativos_uniques = pd.factorize(df["Ativo"])
    n_ativos = len(ativos_uniques)

    # target por ativo
    df["Target"] = df.groupby("Ativo")["Close"].shift(-1)
    df = df.dropna().reset_index(drop=True)

    features = [
        c for c in df.columns
        if c not in ["Date", "Ativo", "Target", "Ativo_ID"]
    ]

    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])

    seqs = []
    targets = []
    ativos = []

    for ativo in df["Ativo"].unique():
        dfa = df[df["Ativo"] == ativo].reset_index(drop=True)
        X_arr = dfa[features].values
        y_arr = dfa["Target"].values
        ativos_arr = dfa["Ativo_ID"].values

        for i in range(len(dfa) - SEQ_LEN):
            seqs.append(X_arr[i:i+SEQ_LEN])
            targets.append(y_arr[i+SEQ_LEN])
            ativos.append(ativos_arr[i+SEQ_LEN])

    return (
        np.array(seqs),
        np.array(targets),
        np.array(ativos),
        scaler,
        features,
        n_ativos
    )


# ==================== MAIN ====================
def main():
    print("Carregando e preparando dados multi-ativo...")
    X, y, ativos, scaler, features, n_ativos = preparar_dados(DATA_PATH)

    X_train, X_test, y_train, y_test, ativos_train, ativos_test = train_test_split(
        X, y, ativos, test_size=0.2, shuffle=False
    )

    train_ds = PriceDataset(X_train, y_train, ativos_train)
    test_ds  = PriceDataset(X_test,  y_test,  ativos_test)

    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH)

    model = MultiAssetLSTM(
        input_size=X.shape[2],
        hidden_size=128,
        num_layers=2,
        n_ativos=n_ativos
    ).to(DEVICE)

    optim = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    print("Treinando modelo multi-ativo...")
    for epoch in range(1, EPOCHS+1):
        model.train()
        losses = []

        for xb, yb, atb in train_dl:
            xb, yb, atb = xb.to(DEVICE), yb.to(DEVICE), atb.to(DEVICE)

            pred = model(xb, atb)
            loss = loss_fn(pred, yb)

            optim.zero_grad()
            loss.backward()
            optim.step()

            losses.append(loss.item())

        print(f"Epoch {epoch}/{EPOCHS} | Loss: {np.mean(losses):.5f}")

    # salvar checkpoint
    torch.save({
        "model_state": model.state_dict(),
        "scaler": scaler,
        "train_columns": features,
        "seq_len": SEQ_LEN,
        "n_ativos": n_ativos
    }, SAVE_MODEL)

    print(f"\n✔ Modelo multi-ativo salvo em {SAVE_MODEL}")


if __name__ == "__main__":
    main()
