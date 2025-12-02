#!/usr/bin/env python3
"""
Comparação Final — Todos os Modelos

Modelos avaliados:
1) LSTM baseline
2) LSTM Autoencoder (AE)
3) Híbrido baseado em Clusters (retornos)
4) Híbrido com Impact Predictor supervisionado

Resultado:
- RMSE e R² de cada modelo
- Gráfico total com todas as curvas
"""

# ======================================================================
# IMPORTS
# ======================================================================

import os
import sys
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from modelos.model_baseline_lstm import LSTMPrice
from modelos.model_autoencoder_lstm import LSTMAutoencoderPrice

from eval.modelo_hibrido_eval_autoencoder_ativo import (
    prever,
    integrar_impacto,
    extrair_motivos,
    detectar_coluna_close
)

from eval.impact_integration import aplicar_impact_predictor_com_motivos
from utils.embedding_manager import EmbeddingManager

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================================
# 🔧 CARREGAR MODELO (detecção automática LSTM → AE)
# ======================================================================

def carregar_modelo(model_path):
    """
    Carrega checkpoint detectando automaticamente se é:
    - LSTM baseline (LSTMPrice)
    - Autoencoder (LSTMAutoencoderPrice)
    """

    import numpy as np
    from sklearn.preprocessing import MinMaxScaler

    torch.serialization.add_safe_globals([
        np.ndarray, MinMaxScaler, dict, list, tuple
    ])

    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    state_keys = ckpt["model_state"].keys()
    input_size = len(ckpt["train_columns"])

    # detectar AE ou baseline
    is_autoencoder = any("encoder." in k for k in state_keys) or any("decoder." in k for k in state_keys)

    if is_autoencoder:
        print(f"[LOAD] Autoencoder detectado → {os.path.basename(model_path)}")
        model = LSTMAutoencoderPrice(input_size=input_size)
    else:
        print(f"[LOAD] LSTM Baseline detectado → {os.path.basename(model_path)}")
        model = LSTMPrice(input_size=input_size)

    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()

    return model, ckpt["scaler"], ckpt["train_columns"], ckpt["seq_len"]


# ======================================================================
# 🔧 PREVISÃO LSTM BASELINE
# ======================================================================

def prever_baseline(model, scaler, df, seq_len, train_cols):
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
        xt = torch.tensor(seq[np.newaxis], dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            pred_s = model(xt)

        pred_s = float(pred_s.cpu().numpy().item())

        zeros = np.zeros((1, len(train_cols)))
        zeros[0, target_idx] = pred_s

        inv = scaler.inverse_transform(zeros)[0, target_idx]

        preds.append(inv)
        reals.append(df2.loc[i + seq_len, target_col])
        dates.append(df2.loc[i + seq_len, "Date"])

    return pd.DataFrame({"Date": dates, "Pred": preds, "Real": reals}).set_index("Date")


# ======================================================================
# 🔧 MÉTRICAS
# ======================================================================

def metrics(df, col):
    rmse = np.sqrt(mean_squared_error(df["Real"], df[col]))
    r2 = r2_score(df["Real"], df[col])
    return rmse, r2


# ======================================================================
# 🔥 COMPARAÇÃO FINAL
# ======================================================================

def comparar_tudo(csv_path, ativo, out_html):

    print(f"\n========= {ativo} =========")

    df = pd.read_csv(csv_path, parse_dates=["Date"]).sort_values("Date").ffill().bfill()

    # caminhos dos modelos
    path_base = os.path.join(BASE_DIR, "modelos", f"lstm_{ativo.lower()}.pt")
    path_ae = os.path.join(BASE_DIR, "modelos", f"lstm_ae_{ativo.lower()}_brent.pt")
    path_ip = os.path.join(BASE_DIR, "modelos", f"impact_predictor_{ativo}.pt")

    # ---------------------------------------------------
    # 1) BASELINE
    # ---------------------------------------------------
    print("📌 Prevendo baseline...")
    model_b, scaler_b, cols_b, seq_b = carregar_modelo(path_base)
    pred_base = prever_baseline(model_b, scaler_b, df, seq_b, cols_b)

    # ---------------------------------------------------
    # 2) AUTOENCODER (AE)
    # ---------------------------------------------------
    print("📌 Prevendo Autoencoder...")
    model_ae, scaler_ae, cols_ae, seq_ae = carregar_modelo(path_ae)
    pred_ae = prever(model_ae, scaler_ae, df, seq_ae, cols_ae)
    pred_ae["Pred_AE"] = pred_ae["Pred"]

    # preparação híbridos
    emb_mgr = EmbeddingManager()
    clusters_df = pd.read_csv(os.path.join(BASE_DIR, "data", "cluster_motivos.csv"))
    emb_repr = emb_mgr.embed_lote(clusters_df["frase_exemplo"].astype(str).tolist())
    motivos = extrair_motivos(os.path.join(BASE_DIR, "output_noticias"))

    # ---------------------------------------------------
    # 3) AE + CLUSTERS
    # ---------------------------------------------------
    print("📌 Aplicando híbrido baseado em clusters...")
    pred_cluster = integrar_impacto(
        pred_ae.copy(),
        motivos,
        emb_mgr,
        clusters_df,
        emb_repr,
        atv_label=ativo,
        scale=0.4,
        janela=5,
        horizon=4
    )
    pred_cluster.rename(columns={"Pred_Ajustado": "Pred_Clusters",
                                 "Pred_Hibrido": "Pred_Clusters"}, inplace=True)

    # ---------------------------------------------------
    # 4) AE + IMPACT PREDICTOR
    # ---------------------------------------------------
    if os.path.exists(path_ip):
        print("📌 Aplicando Impact Predictor supervisionado...")
        pred_ip = aplicar_impact_predictor_com_motivos(
            pred_ae.copy(),
            path_ip,
            motivos,
            emb_mgr=emb_mgr,
            janela=5,
            horizon=4,
            scale=1.0
        )
        pred_ip.rename(columns={"Pred_Hibrido": "Pred_Impact"}, inplace=True)
    else:
        print("⚠️ Impact Predictor não encontrado.")
        pred_ip = None

    # ======================================================================
    # MÉTRICAS
    # ======================================================================

    print("\n===== MÉTRICAS =====")

    rmse_b, r2_b = metrics(pred_base, "Pred")
    print(f"Baseline LSTM         → RMSE={rmse_b:.6f}  R²={r2_b:.6f}")

    rmse_ae, r2_ae = metrics(pred_ae, "Pred")
    print(f"Autoencoder LSTM      → RMSE={rmse_ae:.6f}  R²={r2_ae:.6f}")

    rmse_cl, r2_cl = metrics(pred_cluster, "Pred_Clusters")
    print(f"Híbrido Clusters      → RMSE={rmse_cl:.6f}  R²={r2_cl:.6f}")

    if pred_ip is not None:
        rmse_ip, r2_ip = metrics(pred_ip, "Pred_Impact")
        print(f"Híbrido ImpactPred    → RMSE={rmse_ip:.6f}  R²={r2_ip:.6f}")

    # ======================================================================
    # GRÁFICO FINAL
    # ======================================================================

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=pred_base.index, y=pred_base["Real"], name="Real", line=dict(color="black")
    ))
    fig.add_trace(go.Scatter(
        x=pred_base.index, y=pred_base["Pred"], name="Baseline LSTM", line=dict(color="blue")
    ))
    fig.add_trace(go.Scatter(
        x=pred_ae.index, y=pred_ae["Pred"], name="Autoencoder", line=dict(color="green")
    ))
    fig.add_trace(go.Scatter(
        x=pred_cluster.index, y=pred_cluster["Pred_Clusters"], name="Híbrido Clusters", line=dict(color="orange")
    ))

    if pred_ip is not None:
        fig.add_trace(go.Scatter(
            x=pred_ip.index, y=pred_ip["Pred_Impact"], name="Impact Predictor", line=dict(color="purple")
        ))

    fig.update_layout(
        title=f"Comparação Completa — {ativo}",
        template="plotly_white",
        height=900
    )

    fig.write_html(out_html)
    print("✔ Gráfico salvo em:", out_html)


# ======================================================================
# MAIN
# ======================================================================

def main():

    comparar_tudo(
        csv_path=os.path.join(BASE_DIR, "data", "dados_petr4_brent.csv"),
        ativo="PETR4",
        out_html=os.path.join(BASE_DIR, "img", "comparacao_total_petr4.html")
    )

    comparar_tudo(
        csv_path=os.path.join(BASE_DIR, "data", "dados_prio3_brent.csv"),
        ativo="PRIO3",
        out_html=os.path.join(BASE_DIR, "img", "comparacao_total_prio3.html")
    )


if __name__ == "__main__":
    main()
