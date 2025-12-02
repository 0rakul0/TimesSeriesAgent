#!/usr/bin/env python3
"""
Comparação entre:
- LSTM baseline (sem autoencoder)
- LSTM Autoencoder
- Híbrido (AE + notícias com impacto via retornos)

Ativo: PETR4 e PRIO3
"""

# ============================================================
# IMPORTS
# ============================================================

import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# modelos
from modelos.model_baseline_lstm import LSTMPrice
from modelos.model_autoencoder_lstm import LSTMAutoencoderPrice

# híbrido atualizado
from eval.modelo_hibrido_eval_autoencoder_ativo import (
    detectar_coluna_close,
    aplicar_seq_real_correction_retornos,
    extrair_motivos
)

# embeddings
from utils.embedding_manager import EmbeddingManager

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# CARREGAR MODELO
# ============================================================

def carregar_modelo(model_path):
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler

    torch.serialization.add_safe_globals([
        np.ndarray, MinMaxScaler, dict, list, tuple
    ])

    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)

    # escolhe a arquitetura automaticamente
    if "ae" in model_path.lower():
        model = LSTMAutoencoderPrice(input_size=len(ckpt["train_columns"]))
    else:
        model = LSTMPrice(input_size=len(ckpt["train_columns"]))

    model = model.to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, ckpt["scaler"], ckpt["train_columns"], ckpt["seq_len"]


# ============================================================
# FUNÇÃO DE PREVISÃO (BASELINE E AE)
# ============================================================

def prever(model, scaler, df, seq_len, train_cols, is_ae=False):
    df2 = df.copy().sort_values("Date").reset_index(drop=True)
    target_col = detectar_coluna_close(train_cols)
    target_idx = train_cols.index(target_col)

    # garantir features
    for c in train_cols:
        if c not in df2.columns:
            df2[c] = 0.0

    arr = scaler.transform(df2[train_cols])

    preds, reals, dates = [], [], []

    for i in range(len(df2) - seq_len):
        seq = arr[i:i+seq_len]
        xt = torch.tensor(seq[np.newaxis], dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            if is_ae:
                pred_s, _ = model(xt)
            else:
                pred_s = model(xt)

            pred_s = float(pred_s.cpu().numpy().item())

        # inverter escala
        zeros = np.zeros((1, len(train_cols)))
        zeros[0, target_idx] = pred_s
        p_inv = scaler.inverse_transform(zeros)[0, target_idx]

        preds.append(p_inv)
        reals.append(df2.loc[i + seq_len, target_col])
        dates.append(df2.loc[i + seq_len, "Date"])

    return pd.DataFrame({"Date": dates, "Pred": preds, "Real": reals}).set_index("Date")


# ============================================================
# FUNÇÃO PRINCIPAL DE COMPARAÇÃO
# ============================================================

def comparar(csv_path, model_base, model_ae, cluster_csv, out_html):

    print("\n📌 Carregando dados...")
    df = pd.read_csv(csv_path, parse_dates=["Date"]).sort_values("Date")

    print("📌 Carregando modelos...")
    model_b, scaler_b, col_b, seq_len_b = carregar_modelo(model_base)
    model_a, scaler_a, col_a, seq_len_a = carregar_modelo(model_ae)

    print("📌 Prevendo baseline...")
    pred_base = prever(model_b, scaler_b, df, seq_len_b, col_b, is_ae=False)

    print("📌 Prevendo Autoencoder...")
    pred_ae = prever(model_a, scaler_a, df, seq_len_a, col_a, is_ae=True)

    # ============================================================
    # ALINHAMENTO DAS PREVISÕES
    # ============================================================
    print("📌 Alinhando baseline e AE...")

    pred_df = pred_base.join(pred_ae["Pred"], how="inner", rsuffix="_AE")
    pred_df.rename(columns={"Pred_AE": "Pred_AE"}, inplace=True)
    pred_df.dropna(inplace=True)

    # híbrido começa com AE
    pred_df["Pred_Hibrido"] = pred_df["Pred_AE"]

    # ============================================================
    # APLICAR IMPACTO VIA RETORNOS
    # ============================================================

    print("📌 Aplicando impacto híbrido (retornos)...")

    clusters_df = pd.read_csv(cluster_csv)

    emb_mgr = EmbeddingManager()
    emb_repr = emb_mgr.embed_lote(clusters_df["frase_exemplo"].astype(str).tolist())
    motivos = extrair_motivos(os.path.join(BASE_DIR, "output_noticias"))

    pred_hib = aplicar_seq_real_correction_retornos(
        pred_df.copy(),
        motivos,
        emb_mgr,
        clusters_df,
        emb_repr,
        scale=0.4
    )

    pred_df["Pred_Hibrido"] = pred_hib["Pred_Ajustado"]
    pred_df.dropna(inplace=True)

    # ============================================================
    # MÉTRICAS
    # ============================================================

    rmse_b = np.sqrt(mean_squared_error(pred_df["Real"], pred_df["Pred"]))
    rmse_a = np.sqrt(mean_squared_error(pred_df["Real"], pred_df["Pred_AE"]))
    rmse_h = np.sqrt(mean_squared_error(pred_df["Real"], pred_df["Pred_Hibrido"]))

    r2_b = r2_score(pred_df["Real"], pred_df["Pred"])
    r2_a = r2_score(pred_df["Real"], pred_df["Pred_AE"])
    r2_h = r2_score(pred_df["Real"], pred_df["Pred_Hibrido"])

    print("\n=========== RESULTADOS ===========")
    print(f"RMSE LSTM Baseline:      {rmse_b:.6f}")
    print(f"RMSE LSTM Autoencoder:   {rmse_a:.6f}")
    print(f"RMSE Híbrido (Retornos): {rmse_h:.6f}")
    print("----------------------------------")
    print(f"R² LSTM Baseline:        {r2_b:.6f}")
    print(f"R² LSTM Autoencoder:     {r2_a:.6f}")
    print(f"R² Híbrido (Retornos):   {r2_h:.6f}")
    print("==================================")

    # ============================================================
    # GRÁFICO FINAL
    # ============================================================

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=pred_df.index, y=pred_df["Real"],
        name="Real", line=dict(color="black")
    ))
    fig.add_trace(go.Scatter(
        x=pred_df.index, y=pred_df["Pred"],
        name="LSTM Baseline", line=dict(color="blue")
    ))
    fig.add_trace(go.Scatter(
        x=pred_df.index, y=pred_df["Pred_AE"],
        name="LSTM Autoencoder", line=dict(color="green")
    ))
    fig.add_trace(go.Scatter(
        x=pred_df.index, y=pred_df["Pred_Hibrido"],
        name="Híbrido (AE + Notícias, Retornos)", line=dict(color="orange")
    ))

    fig.update_layout(
        title=f"Comparação dos Modelos – RMSE: Baseline={rmse_b:.4f} | AE={rmse_a:.4f} | Híbrido={rmse_h:.4f}",
        template="plotly_white",
        legend=dict(x=0.01, y=0.99)
    )

    fig.write_html(out_html)
    print("✔ Gráfico salvo em:", out_html)


# ============================================================
# MAIN
# ============================================================

def main():

    comparar(
        csv_path=os.path.join(BASE_DIR, "data", "dados_petr4_brent.csv"),
        model_base=os.path.join(BASE_DIR, "modelos", "lstm_petr4.pt"),
        model_ae=os.path.join(BASE_DIR, "modelos", "lstm_ae_petr4_brent.pt"),
        cluster_csv=os.path.join(BASE_DIR, "data", "cluster_motivos.csv"),
        out_html=os.path.join(BASE_DIR, "img", "comparacao_petr4.html")
    )

    comparar(
        csv_path=os.path.join(BASE_DIR, "data", "dados_prio3_brent.csv"),
        model_base=os.path.join(BASE_DIR, "modelos", "lstm_prio3.pt"),
        model_ae=os.path.join(BASE_DIR, "modelos", "lstm_ae_prio3_brent.pt"),
        cluster_csv=os.path.join(BASE_DIR, "data", "cluster_motivos.csv"),
        out_html=os.path.join(BASE_DIR, "img", "comparacao_prio3.html")
    )


if __name__ == "__main__":
    main()
