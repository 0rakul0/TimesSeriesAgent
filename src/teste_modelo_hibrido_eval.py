"""
modelo_hibrido_eval_final.py
------------------------------------------------------------
Versão FINAL integrada com o novo módulo de impacto:
 - Usa LSTM baseline para previsão técnica
 - Extrai motivos via JSON em output_noticias/
 - Detecta correspondências semânticas (opcional com embeddings)
 - Aplica impacto real com clusters + replicação D0..DH
 - Reset correto, não-cumulativo
 - Caps de segurança em alpha e impacto diário
------------------------------------------------------------
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import torch
import plotly.graph_objects as go
from datetime import timedelta
from sklearn.metrics import mean_squared_error, r2_score
from dotenv import load_dotenv

# ===============================
# Novo engine substituto
# ===============================
from final_impact_module import apply_impact_sequences

# ===============================
# CONFIGURAÇÃO
# ===============================
load_dotenv()

CAMINHO_DADOS = "../data/dados_combinados.csv"
PASTA_JSON = "../output_noticias"
MODELO_BASELINE = "../models/lstm_baseline_price.pt"

HTML_SAIDA = "../img/previsao_baseline_hibrido_vFinal.html"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# IMPORTAÇÃO DO MODELO BASELINE
# ===============================
def carregar_modelo(model_path):
    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)

    from model_baseline_lstm import LSTMPrice
    model = LSTMPrice(len(ckpt["train_columns"]), 128, 2).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return (
        model,
        ckpt["scaler"],
        ckpt["target_col"],
        ckpt["seq_len"],
        ckpt["train_columns"],
    )


# ===============================
# PREVISÃO TÉCNICA DO LSTM
# ===============================
def prever(model, scaler, df, seq_len, target_col, train_cols):
    df_al = df.copy()
    faltantes = [c for c in train_cols if c not in df_al.columns]
    for c in faltantes:
        df_al[c] = 0.0
    df_al = df_al[train_cols]

    arr = scaler.transform(df_al)
    target_idx = train_cols.index(target_col)

    preds_scaled = []
    reals_scaled = []
    dates = []

    for i in range(len(arr) - seq_len):
        seq = arr[i : i + seq_len, :]
        seq_tensor = torch.tensor(seq[np.newaxis, :, :], dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            pred_scaled = model(seq_tensor).cpu().numpy().item()

        preds_scaled.append(pred_scaled)
        reals_scaled.append(arr[i + seq_len, target_idx])
        dates.append(df_al.index[i + seq_len])

    # desfazer escala apenas do target
    def inverse(values):
        zeros = np.zeros((len(values), len(train_cols)))
        zeros[:, target_idx] = values
        return scaler.inverse_transform(zeros)[:, target_idx]

    df_out = pd.DataFrame(
        {
            "Date": dates,
            "Pred": inverse(np.array(preds_scaled)),
            "Real": inverse(np.array(reals_scaled)),
        }
    ).set_index("Date")

    print(f"[INFO] Previsões geradas: {len(df_out)} amostras.")
    return df_out


# ===============================
# EXTRAÇÃO DE MOTIVOS DOS JSONS
# ===============================
def extrair_motivos_ultimos_dias(pasta_json, janela_dias=30, ref_date=None):
    arquivos = glob.glob(os.path.join(pasta_json, "evento_*.json"))
    noticias = {}

    hoje = pd.to_datetime(ref_date).normalize() if ref_date else pd.Timestamp.today().normalize()
    inicio = hoje - pd.Timedelta(days=janela_dias)

    for path in arquivos:
        try:
            j = json.load(open(path, "r", encoding="utf-8"))
            data = pd.to_datetime(j["data"]).normalize()

            if not (inicio < data <= hoje):
                continue

            motivos = j.get("motivos_identificados", [])
            motivos = [str(m).strip() for m in motivos if m.strip()]

            if motivos:
                noticias.setdefault(data, []).extend(motivos)

        except Exception as e:
            print(f"[WARN] Falha ao ler {path}: {e}")

    return noticias


# ===============================
# PLOTAGEM FINAL
# ===============================
def plotar_final(pred_df, html_path):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=pred_df.index,
            y=pred_df["Real"],
            mode="lines",
            name="Preço Real",
            line=dict(color="black", width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=pred_df.index,
            y=pred_df["Pred"],
            mode="lines",
            name="Previsto (LSTM técnico)",
            line=dict(color="blue", width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=pred_df.index,
            y=pred_df["Pred_Impact"],
            mode="lines+markers",
            name="Previsto Ajustado (Notícias)",
            line=dict(color="orange", width=3),
        )
    )

    rmse = np.sqrt(mean_squared_error(pred_df["Real"], pred_df["Pred_Impact"]))
    r2 = r2_score(pred_df["Real"], pred_df["Pred_Impact"])

    fig.update_layout(
        title=f"Previsão Híbrida vFINAL — RMSE={rmse:.3f} | R²={r2:.3f}",
        xaxis_title="Data",
        yaxis_title="Preço (R$)",
        template="plotly_white",
        width=1200,
        height=600,
    )

    fig.write_html(html_path, auto_open=False)
    print(f"✅ Gráfico final salvo em {html_path}")


# ===============================
# MAIN
# ===============================
def main():
    print("\n🚀 Rodando versão FINAL do modelo híbrido...")

    # --- carregar base
    df = pd.read_csv(CAMINHO_DADOS, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index).normalize()
    df = df.ffill().bfill().fillna(0.0)

    # --- model baseline
    model, scaler, target_col, seq_len, cols = carregar_modelo(MODELO_BASELINE)

    # --- previsões
    pred_df = prever(model, scaler, df, seq_len, target_col, cols)

    # --- motivos
    motivos_por_data = extrair_motivos_ultimos_dias(
        PASTA_JSON,
        janela_dias=30,
        ref_date=pred_df.index[-1],
    )

    print(f"[INFO] Motivos carregados: {len(motivos_por_data)} dias com eventos.")

    # --- aplicar impacto REAL com o novo engine
    pred_corr, logs = apply_impact_sequences(
        pred_df,
        motivos_por_data,
        verbose=True,
        horizon=5,
        alpha_method="zscore",
        alpha_cap=(0.25, 4.0),
        max_pct_per_day=0.25,
    )

    # --- salvar versão CSV opcional
    pred_corr.to_csv("../data/previsao_impacto_final.csv")

    # --- plotar
    plotar_final(pred_corr, HTML_SAIDA)

    print("\n[RESULTADO FINAL]")
    rmse = np.sqrt(mean_squared_error(pred_corr["Real"], pred_corr["Pred_Impact"]))
    r2 = r2_score(pred_corr["Real"], pred_corr["Pred_Impact"])
    print(f"RMSE = {rmse:.4f} | R² = {r2:.4f}")

    print("\nLogs de aplicação armazenados (lista de dicts).")


if __name__ == "__main__":
    main()
