#!/usr/bin/env python3
"""
Ensemble Híbrido com Confiança Adaptativa
-----------------------------------------
Combina:
  • LSTM
  • Ajuste Eval Adaptativo (seq_dk × confiança)
  • Ajuste Ridge (resíduos temporais)
Usa pesos ótimos para combinação final.

Gera:
  previsao_hibrida_ensemble_petr4.html
  previsao_hibrida_ensemble_prio3.html
"""

import os
import sys
import json
import glob
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from dotenv import load_dotenv

# Inicializar variáveis de ambiente
load_dotenv()

# ------------------ CONFIG ------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from utils.embedding_manager import EmbeddingManager
from modelos.model_baseline_lstm import LSTMPrice

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLUSTER_CSV = os.path.join(BASE_DIR, "data", "cluster_motivos.csv")
OUT_PETR4 = os.path.join(BASE_DIR, "img", "previsao_hibrida_ensemble_petr4.html")
OUT_PRIO3 = os.path.join(BASE_DIR, "img", "previsao_hibrida_ensemble_prio3.html")

SEQ_COLS = [f"seq_d{i}" for i in range(6)]


# ======================================================
# 1) Carregar modelo seguro (PyTorch 2.6)
# ======================================================
def carregar_modelo(model_path):
    torch.serialization.add_safe_globals([np.ndarray, MinMaxScaler, dict, list, tuple])
    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)

    model = LSTMPrice(len(ckpt["train_columns"])).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, ckpt["scaler"], ckpt["train_columns"], ckpt["seq_len"]


# ======================================================
# 2) Detectar coluna Close correta do ativo
# ======================================================
def detectar_coluna_close(train_cols):
    closes = [c for c in train_cols if c.startswith("Close_")]
    if closes:
        return closes[0]
    closes = [c for c in train_cols if "Close" in c]
    return closes[0]


# ======================================================
# 3) Previsão LSTM
# ======================================================
def prever_lstm(model, scaler, df, seq_len, train_cols):
    df2 = df.copy().sort_values("Date").reset_index(drop=True)
    target_col = detectar_coluna_close(train_cols)
    target_idx = train_cols.index(target_col)

    for c in train_cols:
        if c not in df2.columns:
            df2[c] = 0.0

    arr = scaler.transform(df2[train_cols])

    preds, reals, dates = [], [], []

    for i in range(len(df2) - seq_len):
        seq = arr[i:i + seq_len]
        seq_tensor = torch.tensor(seq[np.newaxis], dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            pred_s = model(seq_tensor).cpu().numpy().item()

        # inverse scaling
        zeros = np.zeros((1, len(train_cols)))
        zeros[0][target_idx] = pred_s
        inv = scaler.inverse_transform(zeros)[0][target_idx]

        preds.append(float(inv))
        reals.append(float(df2.loc[i + seq_len, target_col]))
        dates.append(df2.loc[i + seq_len, "Date"])

    return pd.DataFrame({"Date": pd.to_datetime(dates),
                         "Pred": preds,
                         "Real": reals}).set_index("Date")


# ======================================================
# 4) Extrair motivos
# ======================================================
def extrair_motivos(folder):
    noticias = {}
    for p in glob.glob(os.path.join(folder, "evento_*.json")):
        try:
            j = json.load(open(p, "r", encoding="utf-8"))
            d = pd.to_datetime(j["data"]).normalize()
            motivos = [m.strip() for m in j.get("motivos_identificados", []) if m.strip()]
            if motivos:
                noticias.setdefault(d, []).extend(motivos)
        except:
            pass
    return noticias


# ======================================================
# 5) Construir X + confiança adaptativa
# ======================================================
def construir_X_conf(dates, noticias, clusters_df, emb_mgr, emb_repr):
    N = len(dates)
    X = np.zeros((N, 6))
    conf = np.zeros(N)

    norms_repr = np.linalg.norm(emb_repr, axis=1)

    for i, dia in enumerate(pd.to_datetime(dates)):
        motivos = noticias.get(dia, [])
        if not motivos:
            continue

        sims_list = []

        for motivo in motivos:
            emb = emb_mgr.embed(motivo)
            norm_e = np.linalg.norm(emb)
            sims = (emb.ravel() @ emb_repr.T) / (norm_e * norms_repr + 1e-12)
            idx = int(np.argmax(sims))
            sims_list.append(float(sims[idx]))

            # cluster selected
            row = clusters_df.iloc[idx]
            for k in range(6):
                val = row[f"seq_d{k}"]
                if not pd.isna(val):
                    X[i, k] += float(val)

        conf[i] = np.mean(sims_list)

    return X, conf


# ======================================================
# 6) Ajuste EVAL adaptativo
# ======================================================
def ajuste_eval(pred_series, X, conf):
    seq_dec = X / 100.0
    impacto = seq_dec.sum(axis=1) * conf
    return pred_series * (1 + impacto)


# ======================================================
# 7) Ajuste Ridge
# ======================================================
def ajuste_ridge(pred_df, X):
    y = pred_df["Real"].values - pred_df["Pred"].values

    mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
    X2 = X[mask]
    y2 = y[mask]

    ridge = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100],
                    cv=TimeSeriesSplit(n_splits=5),
                    scoring="neg_mean_squared_error").fit(X2, y2)

    ajustes = X.dot(ridge.coef_)
    return pred_df["Pred"].values + ajustes


# ======================================================
# 8) Ensemble com pesos ótimos
# ======================================================
def combinar_ensemble(pred_lstm, pred_eval, pred_ridge, real):
    GRID = [
        (0.5, 0.3, 0.2),
        (0.6, 0.2, 0.2),
        (0.4, 0.4, 0.2),
        (0.3, 0.5, 0.2),
        (0.2, 0.6, 0.2)
    ]

    best_rmse = 1e9
    best_pred = None

    for w_lstm, w_eval, w_ridge in GRID:
        pred = (w_lstm * pred_lstm +
                w_eval * pred_eval +
                w_ridge * pred_ridge)

        rmse = np.sqrt(mean_squared_error(real, pred))
        if rmse < best_rmse:
            best_rmse = rmse
            best_pred = pred

    return best_pred


# ======================================================
# 9) Plot
# ======================================================
def plotar(pred_df, out, nome):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pred_df.index, y=pred_df["Real"], name="Real"))
    fig.add_trace(go.Scatter(x=pred_df.index, y=pred_df["Pred_LSTM"], name="LSTM"))
    fig.add_trace(go.Scatter(x=pred_df.index, y=pred_df["Pred_Ensemble"], name="Ensemble"))

    rmse = np.sqrt(mean_squared_error(pred_df["Real"], pred_df["Pred_Ensemble"]))
    r2 = r2_score(pred_df["Real"], pred_df["Pred_Ensemble"])

    fig.update_layout(
        title=f"{nome} Ensemble com Confiança Adaptativa | RMSE={rmse:.4f} | R²={r2:.4f}",
        template="plotly_white"
    )
    fig.write_html(out)
    print("✔ salvo:", out)


# ======================================================
# 10) Pipeline Final
# ======================================================
def rodar(csv, model_path, out, nome):
    print(f"\n==== {nome} ====\n")

    df = pd.read_csv(csv, parse_dates=["Date"]).sort_values("Date").ffill().bfill()

    model, scaler, train_cols, seq_len = carregar_modelo(model_path)
    pred_df = prever_lstm(model, scaler, df, seq_len, train_cols)

    pred_df["Pred_LSTM"] = pred_df["Pred"]

    emb_mgr = EmbeddingManager()
    clusters_df = pd.read_csv(CLUSTER_CSV)
    emb_repr = emb_mgr.embed_lote(clusters_df["frase_exemplo"].astype(str).tolist())

    noticias = extrair_motivos(os.path.join(BASE_DIR, "output_noticias"))

    X, conf = construir_X_conf(pred_df.index, noticias, clusters_df, emb_mgr, emb_repr)

    # Eval Adaptativo
    pred_eval = ajuste_eval(pred_df["Pred"], X, conf)

    # Ridge
    pred_ridge = ajuste_ridge(pred_df, X)

    # Ensemble
    pred_ens = combinar_ensemble(
        pred_df["Pred"].values,
        pred_eval,
        pred_ridge,
        pred_df["Real"].values
    )

    pred_df["Pred_Ensemble"] = pd.Series(pred_ens).ewm(span=3, adjust=False).mean()

    plotar(pred_df, out, nome)

    print("RMSE LSTM:", np.sqrt(mean_squared_error(pred_df["Real"], pred_df["Pred"])))
    print("RMSE Ensemble:", np.sqrt(mean_squared_error(pred_df["Real"], pred_df["Pred_Ensemble"])))


def main():
    rodar(
        csv=os.path.join(BASE_DIR, "data", "dados_petr4_brent.csv"),
        model_path=os.path.join(BASE_DIR, "modelos", "lstm_petr4.pt"),
        out=OUT_PETR4, nome="PETR4"
    )

    rodar(
        csv=os.path.join(BASE_DIR, "data", "dados_prio3_brent.csv"),
        model_path=os.path.join(BASE_DIR, "modelos", "lstm_prio3.pt"),
        out=OUT_PRIO3, nome="PRIO3"
    )


if __name__ == "__main__":
    main()
