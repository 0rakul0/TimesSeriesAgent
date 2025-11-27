#!/usr/bin/env python3
"""
Modelo Híbrido Ridge por Ativo (PETR4 / PRIO3)
----------------------------------------------

Fluxo:
 1. Carrega modelo LSTM treinado para cada ativo.
 2. Gera previsões base (Pred, Real).
 3. Constrói matriz X usando seq_d0..seq_d5 dos clusters.
 4. Treina RidgeCV sobre o resíduo (Real - Pred).
 5. Aplica correção Pred_Ajustado.
 6. Gera gráfico HTML.

Executa automaticamente:
 - PETR4
 - PRIO3
"""

import os
import sys
import glob
import json
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go
from dotenv import load_dotenv

# Inicializar variáveis de ambiente
load_dotenv()
# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from utils.embedding_manager import EmbeddingManager
from modelos.model_baseline_lstm import LSTMPrice

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_COLS = [f"seq_d{i}" for i in range(6)]

CLUSTERS_PATH = os.path.join(BASE_DIR, "data", "cluster_motivos.csv")
NEWS_PATH = os.path.join(BASE_DIR, "output_noticias")

OUT_PETR4 = os.path.join(BASE_DIR, "img", "previsao_hibrida_ridge_petr4.html")
OUT_PRIO3 = os.path.join(BASE_DIR, "img", "previsao_hibrida_ridge_prio3.html")

# ============================================================
# CHECKPOINT SAFE LOADER
# ============================================================

def safe_load_checkpoint(path):
    """Carregamento compatível com PyTorch 2.6+."""
    try:
        return torch.load(path, map_location=DEVICE)
    except Exception:
        # allowlist fallback
        import numpy as _np
        from sklearn.preprocessing import MinMaxScaler
        torch.serialization.add_safe_globals([
            _np.ndarray,
            _np._core.multiarray._reconstruct,
            MinMaxScaler,
            dict, list, tuple
        ])
        return torch.load(path, map_location=DEVICE, weights_only=False)

# ============================================================
# CARREGAR MODELO
# ============================================================

def carregar_modelo(path):
    ckpt = safe_load_checkpoint(path)

    model = LSTMPrice(input_size=len(ckpt["train_columns"])).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return (
        model,
        ckpt["scaler"],
        ckpt["train_columns"],
        ckpt["seq_len"]
    )

# ============================================================
# PREVISÃO LSTM
# ============================================================

def detectar_coluna_close(train_cols):
    closes = [c for c in train_cols if c.startswith("Close_")]
    if closes:
        return closes[0]
    closes = [c for c in train_cols if "Close" in c]
    if closes:
        return closes[0]
    raise KeyError("Nenhuma coluna Close encontrada no modelo.")

def prever(model, scaler, df, seq_len, train_cols):
    df2 = df.sort_values("Date").reset_index(drop=True)

    target_col = detectar_coluna_close(train_cols)
    target_idx = train_cols.index(target_col)

    # garantir todas colunas
    for c in train_cols:
        if c not in df2.columns:
            df2[c] = 0.0

    arr = scaler.transform(df2[train_cols])

    preds, reals, dates = [], [], []

    for i in range(len(df2) - seq_len):
        seq = arr[i:i+seq_len]
        seq_tensor = torch.tensor(seq[np.newaxis], dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            pred_scaled = model(seq_tensor).cpu().numpy().item()

        zeros = np.zeros((1, len(train_cols)))
        zeros[0, target_idx] = pred_scaled
        final = scaler.inverse_transform(zeros)[0, target_idx]

        preds.append(float(final))
        reals.append(float(df2.loc[i+seq_len, target_col]))
        dates.append(df2.loc[i+seq_len, "Date"])

    return pd.DataFrame({
        "Date": dates,
        "Pred": preds,
        "Real": reals
    }).set_index("Date")

# ============================================================
# NOTÍCIAS
# ============================================================

def extrair_motivos(folder):
    noticias = {}
    for p in sorted(glob.glob(os.path.join(folder, "evento_*.json"))):
        try:
            j = json.load(open(p, "r", encoding="utf-8"))
            d = pd.to_datetime(j["data"]).normalize()
            motivos = j.get("motivos_identificados", [])
            motivos = [m.strip() for m in motivos if m.strip()]
            if motivos:
                noticias.setdefault(d, []).extend(motivos)
        except:
            continue
    return noticias

# ============================================================
# CONSTRUIR X
# ============================================================

def construir_X(dates, noticias, clusters_df, emb_mgr, emb_repr):
    dates = pd.to_datetime(dates)
    N = len(dates)
    K = len(SEQ_COLS)
    X = np.zeros((N, K))

    emb_repr = np.asarray(emb_repr)
    norms_repr = np.linalg.norm(emb_repr, axis=1) + 1e-12

    for i, dt in enumerate(dates):
        for k in range(K):
            dtk = dt - pd.Timedelta(days=k)
            motivos = noticias.get(dtk, [])
            soma = 0.0
            for motivo in motivos:
                e = emb_mgr.embed(motivo).ravel()
                norm_e = np.linalg.norm(e) + 1e-12
                sims = (e @ emb_repr.T) / (norm_e * norms_repr)
                idx = int(np.argmax(sims))

                cluster_id = int(clusters_df.iloc[idx]["cluster"])
                row = clusters_df[clusters_df["cluster"] == cluster_id].iloc[0]

                val = row.get(f"seq_d{k}", None)
                if pd.notna(val):
                    soma += float(val)
            X[i, k] = soma

    return X

# ============================================================
# TREINAR RIDGE (resíduo)
# ============================================================

def treinar_ridge(X, y):
    mask = np.isfinite(y) & np.isfinite(X).all(axis=1)

    if mask.sum() < 10:
        raise ValueError("Poucas amostras válidas para Ridge.")

    tscv = TimeSeriesSplit(n_splits=5)
    ridge = RidgeCV(
        alphas=[0.01, 0.1, 1.0, 10, 100],
        cv=tscv,
        scoring="neg_mean_squared_error"
    )
    ridge.fit(X[mask], y[mask])

    return ridge, mask

# ============================================================
# APLICAR AJUSTE
# ============================================================

def aplicar_ajuste(pred_df, X, ridge_model, mask):
    ajustes = X.dot(ridge_model.coef_)

    pred_df = pred_df.copy()
    pred_df["Pred_Ajustado"] = pred_df["Pred"]

    idxs = np.where(mask)[0]
    pred_df.iloc[idxs, pred_df.columns.get_loc("Pred_Ajustado")] = (
        pred_df["Pred"].iloc[idxs] + ajustes[idxs]
    )

    pred_df["Pred_Ajustado_Suave"] = (
        pred_df["Pred_Ajustado"].ewm(span=3, adjust=False).mean()
    )

    return pred_df

# ============================================================
# PLOT
# ============================================================

def plotar(pred_df, out_path, nome):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pred_df.index, y=pred_df["Real"], name="Real", line=dict(color="black")))
    fig.add_trace(go.Scatter(x=pred_df.index, y=pred_df["Pred"], name="LSTM", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=pred_df.index, y=pred_df["Pred_Ajustado_Suave"],
                             name="Híbrido Ridge", line=dict(color="orange")))

    rmse = np.sqrt(mean_squared_error(pred_df["Real"], pred_df["Pred_Ajustado_Suave"]))
    r2 = r2_score(pred_df["Real"], pred_df["Pred_Ajustado_Suave"])

    fig.update_layout(
        title=f"{nome} — Híbrido Ridge | RMSE={rmse:.4f} | R²={r2:.4f}",
        template="plotly_white"
    )
    fig.write_html(out_path)

    print(f"\n✔ Gráfico salvo em: {out_path}")
    print(f"RMSE antes : {np.sqrt(mean_squared_error(pred_df['Real'], pred_df['Pred']))}")
    print(f"RMSE depois: {rmse}")

# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def rodar_modelo(csv_path, model_path, out_html, nome):
    print(f"\n==================== {nome} ====================")

    df = pd.read_csv(csv_path, parse_dates=["Date"]).sort_values("Date").ffill().bfill()
    model, scaler, train_cols, seq_len = carregar_modelo(model_path)

    pred_df = prever(model, scaler, df, seq_len, train_cols)

    emb_mgr = EmbeddingManager()
    clusters_df = pd.read_csv(CLUSTERS_PATH)
    emb_repr = emb_mgr.embed_lote(clusters_df["frase_exemplo"].astype(str).tolist())

    noticias = extrair_motivos(NEWS_PATH)
    X = construir_X(pred_df.index, noticias, clusters_df, emb_mgr, emb_repr)

    y = (pred_df["Real"] - pred_df["Pred"]).values

    try:
        ridge, mask = treinar_ridge(X, y)
    except ValueError as e:
        print("[WARN] Ridge não treinado:", e)
        pred_df["Pred_Ajustado"] = pred_df["Pred"]
        pred_df["Pred_Ajustado_Suave"] = pred_df["Pred"].ewm(span=3).mean()
        plotar(pred_df, out_html, nome)
        return

    pred_adj = aplicar_ajuste(pred_df, X, ridge, mask)
    plotar(pred_adj, out_html, nome)

# ============================================================
# MAIN
# ============================================================

def main():
    rodar_modelo(
        csv_path=os.path.join(BASE_DIR, "data", "dados_petr4_brent.csv"),
        model_path=os.path.join(BASE_DIR, "modelos", "lstm_petr4.pt"),
        out_html=OUT_PETR4,
        nome="PETR4"
    )

    rodar_modelo(
        csv_path=os.path.join(BASE_DIR, "data", "dados_prio3_brent.csv"),
        model_path=os.path.join(BASE_DIR, "modelos", "lstm_prio3.pt"),
        out_html=OUT_PRIO3,
        nome="PRIO3"
    )

if __name__ == "__main__":
    main()
