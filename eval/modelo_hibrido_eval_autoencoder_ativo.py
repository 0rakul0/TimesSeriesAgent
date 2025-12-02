#!/usr/bin/env python3
"""
Modelo Híbrido Avaliativo – Autoencoder + Impact Predictor (quando disponível)

Este arquivo:
- mantém a função antiga de impacto via clusters (aplicar_seq_real_correction_retornos)
- integra o Impact Predictor supervisionado (aplicar_impact_predictor_com_motivos) se houver checkpoint
- permanece compatível com seu pipeline (recon plots, previsões AE)
"""

# =====================================================================
# IMPORTS
# =====================================================================

import os
import sys
import glob
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from modelos.model_autoencoder_lstm import LSTMAutoencoderPrice
from utils.embedding_manager import EmbeddingManager

# importa a integração do predictor gerada separadamente
try:
    from eval.impact_integration import aplicar_impact_predictor_com_motivos
except Exception:
    aplicar_impact_predictor_com_motivos = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEQ_COLS = [f"seq_d{i}" for i in range(6)]
CLUSTER_CSV = os.path.join(BASE_DIR, "data", "cluster_motivos.csv")


# =====================================================================
# CARREGAR MODELO AE
# =====================================================================

def carregar_modelo(model_path):
    """
    Carrega checkpoint treinado com autoencoder
    """
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler

    torch.serialization.add_safe_globals([
        np.ndarray, MinMaxScaler, dict, list, tuple
    ])

    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)

    model = LSTMAutoencoderPrice(
        input_size=len(ckpt["train_columns"])
    ).to(DEVICE)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, ckpt["scaler"], ckpt["train_columns"], ckpt["seq_len"]


# =====================================================================
# DETECTAR COLUNA DE FECHAMENTO
# =====================================================================

def detectar_coluna_close(train_cols):
    closes = [c for c in train_cols if c.startswith("Close_")]
    if closes:
        return closes[0]
    closes = [c for c in train_cols if "Close" in c]
    if not closes:
        raise KeyError("Nenhuma coluna Close encontrada no checkpoint.")
    return closes[0]


# =====================================================================
# PREVISOR (usa AE)
# =====================================================================

def prever(model, scaler, df, seq_len, train_cols, salvar_recon_plot=None):
    """
    Retorna Pred e Real + reconstrói a última janela para plot
    """

    df2 = df.copy().sort_values("Date").reset_index(drop=True)
    target_col = detectar_coluna_close(train_cols)
    target_idx = train_cols.index(target_col)

    # garantir colunas
    for c in train_cols:
        if c not in df2.columns:
            df2[c] = 0.0

    arr = scaler.transform(df2[train_cols])

    preds, reals, dates = [], [], []
    recon_samples = []

    for i in range(len(df2) - seq_len):
        seq = arr[i:i + seq_len]
        seq_tensor = torch.tensor(seq[np.newaxis], dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            pred_s, recon_s = model(seq_tensor)
            pred_s = pred_s.cpu().numpy().item()
            recon_np = recon_s.cpu().numpy()[0]

        recon_samples.append((seq, recon_np))

        # inversão da escala
        zeros = np.zeros((1, len(train_cols)))
        zeros[0, target_idx] = pred_s
        inv = scaler.inverse_transform(zeros)[0, target_idx]

        preds.append(inv)
        reals.append(df2.loc[i + seq_len, target_col])
        dates.append(pd.to_datetime(df2.loc[i + seq_len, "Date"]))

    # gráfico de reconstrução
    if salvar_recon_plot and recon_samples:
        real_seq, recon_seq = recon_samples[-1]
        plt.figure(figsize=(10, 5))
        plt.plot(real_seq[:, target_idx], label="Real", color="black")
        plt.plot(recon_seq[:, target_idx], label="Recon AE", color="blue")
        plt.title("Reconstrução da Última Janela – Autoencoder")
        plt.legend()
        plt.grid(True)
        plt.savefig(salvar_recon_plot)
        plt.close()

    return pd.DataFrame({"Date": dates, "Pred": preds, "Real": reals}).set_index("Date")


# =====================================================================
# EXTRAÇÃO DE MOTIVOS
# =====================================================================

def extrair_motivos(pasta_json):
    """
    Lê arquivos evento_*.json contendo motivos.
    """
    arquivos = glob.glob(os.path.join(pasta_json, "evento_*.json"))
    noticias = {}

    for arq in arquivos:
        try:
            j = json.load(open(arq, "r", encoding="utf-8"))
            d = pd.to_datetime(j["data"]).normalize()
            motivos = j.get("motivos_identificados", [])
            motivos = [m.strip() for m in motivos if isinstance(m, str) and m.strip()]
            if motivos:
                noticias.setdefault(d, []).extend(motivos)
        except Exception:
            continue

    return noticias


# =====================================================================
# 💥 FUNÇÃO ANTIGA: IMPACTO VIA RETORNOS + CORRECTION LOOP (mantida)
# =====================================================================

def aplicar_seq_real_correction_retornos(
        pred_df,
        motivos_por_data,
        emb_mgr,
        clusters_df,
        emb_repr,
        sim_threshold=0.7,
        max_horizon=4,
        scale=0.4,
        beta_min=0.5,
        beta_max=1.6
):
    """
    ✔ Impacto aplicado em RETORNO, não em PREÇO
    ✔ Correction Loop Online
    ✔ Reconstrução de preço via retornos acumulados
    """

    df = pred_df.copy()
    # assume Pred_AE já presente como coluna Pred (quando chamada a partir do comparador)
    if "Pred_AE" not in df.columns:
        df["Pred_AE"] = df["Pred"].copy()
    df["Pred_Ajustado"] = df["Pred_AE"].copy()

    dias = df.index
    eventos = sorted(motivos_por_data.keys())

    for data_evt in eventos:

        motivos = motivos_por_data[data_evt]
        if not motivos:
            continue

        pos = dias.searchsorted(data_evt)
        if pos >= len(dias):
            continue

        # retornos previstos do AE
        df["Ret_AE"] = df["Pred_AE"].pct_change() * 100

        futuras = [d for d in eventos if d > data_evt]

        for motivo in motivos:

            # ---------------- EMBEDDING ----------------
            emb = emb_mgr.embed(motivo)
            norm_e = np.linalg.norm(emb)

            norms_repr = np.linalg.norm(emb_repr, axis=1)
            sims = (emb @ emb_repr.T).ravel() / (norm_e * norms_repr + 1e-12)

            idx_best = int(np.argmax(sims))
            best_sim = float(sims[idx_best])
            if best_sim < sim_threshold:
                continue

            # sequência do cluster
            row = clusters_df.iloc[idx_best]
            seq_raw = []
            for k in range(max_horizon + 1):
                col = f"seq_d{k}"
                seq_raw.append(
                    float(row[col]) if col in row and pd.notna(row[col]) else None
                )
            seq_corr = seq_raw.copy()

            # ---------- CORRECTION LOOP ----------
            for k in range(1, max_horizon + 1):

                idx = pos + k
                if idx >= len(dias):
                    break

                impacto = seq_corr[k]
                if impacto is None:
                    break

                # retorno real observado
                r_obs = df["Real"].pct_change().loc[dias[idx]] * 100

                if abs(seq_corr[k]) < 1e-9:
                    beta = 1.0
                else:
                    beta = r_obs / seq_corr[k]
                    beta = np.clip(beta, beta_min, beta_max)

                # aplica correção nos dias futuros
                for t in range(k + 1, len(seq_corr)):
                    if seq_corr[t] is not None:
                        seq_corr[t] *= beta

            # ---------- APLICAR IMPACTO EM RETORNO ----------
            for k in range(1, max_horizon + 1):

                idx = pos + k
                if idx >= len(dias):
                    break

                if any((f > data_evt and f <= dias[idx]) for f in futuras):
                    break

                impacto = seq_corr[k]
                if impacto is None:
                    break

                r_pred = df["Pred_AE"].pct_change().loc[dias[idx]] * 100
                r_new = r_pred + scale * best_sim * impacto

                P_prev = df["Pred_Ajustado"].loc[dias[idx - 1]]
                df.loc[dias[idx], "Pred_Ajustado"] = P_prev * (1 + r_new / 100)

    return df.drop(columns=["Ret_AE"], errors="ignore")


# =====================================================================
# PLOT FINAL
# =====================================================================

def plotar(pred_df, out_html, nome):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pred_df.index, y=pred_df["Real"], name="Real", line=dict(color="black")))
    fig.add_trace(go.Scatter(x=pred_df.index, y=pred_df["Pred"], name="AE-LSTM", line=dict(color="blue")))
    # Pred_Ajustado ou Pred_Hibrido, dependendo de qual foi gerado
    adj_col = "Pred_Ajustado" if "Pred_Ajustado" in pred_df.columns else "Pred_Hibrido" if "Pred_Hibrido" in pred_df.columns else "Pred"
    fig.add_trace(go.Scatter(x=pred_df.index, y=pred_df[adj_col], name="Híbrido (Notícias)", line=dict(color="orange")))

    rmse = np.sqrt(mean_squared_error(pred_df["Real"], pred_df[adj_col]))
    r2 = r2_score(pred_df["Real"], pred_df[adj_col])

    fig.update_layout(
        title=f"{nome} – Híbrido com Impacto | RMSE={rmse:.4f} | R²={r2:.4f}",
        template="plotly_white"
    )

    fig.write_html(out_html)
    print("✔ Salvo:", out_html)


# =====================================================================
# FUNÇÃO QUE ESCOLHE O MÉTODO DE INTEGRAÇÃO (Predictor ou Clusters)
# =====================================================================

def integrar_impacto(pred_df, motivos_por_data, emb_mgr, clusters_df, emb_repr,
                      atv_label="PETR4", impact_model_dir=None,
                      scale=1.0, janela=5, horizon=4, sim_threshold=0.7):
    """
    Tenta usar o Impact Predictor treinado (arquivo modelos/impact_predictor_<ATIVO>.pt).
    Se o checkpoint não existir ou a função de integração não estiver disponível,
    usa o método antigo baseado em clusters (aplicar_seq_real_correction_retornos).

    Retorna pred_df com coluna 'Pred_Hibrido' (quando predictor) ou 'Pred_Ajustado' (quando clusters).
    """
    # garantir colunas base
    if "Pred_AE" not in pred_df.columns:
        pred_df["Pred_AE"] = pred_df["Pred"].copy()
    pred_df["Pred_Hibrido"] = pred_df["Pred_AE"].copy()

    # construir caminho padrão do modelo de impacto
    if impact_model_dir is None:
        impact_model_dir = os.path.join(BASE_DIR, "modelos")

    model_name = f"impact_predictor_{atv_label}.pt"
    model_path = os.path.join(impact_model_dir, model_name)

    # se temos a função do predictor e o checkpoint existe, use-o
    if aplicar_impact_predictor_com_motivos is not None and os.path.exists(model_path):
        print(f"[INFO] Usando Impact Predictor: {model_path}")
        pred_df = aplicar_impact_predictor_com_motivos(
            pred_df,
            model_path,
            motivos_por_data,
            emb_mgr=emb_mgr,
            janela=janela,
            horizon=horizon,
            scale=scale
        )
        # garantir coluna padronizada
        if "Pred_Hibrido" not in pred_df.columns and "Pred_Ajustado" in pred_df.columns:
            pred_df["Pred_Hibrido"] = pred_df["Pred_Ajustado"].copy()
        return pred_df

    # caso contrário, usar método baseado em clusters (compatibilidade)
    print("[INFO] Impact Predictor não encontrado ou indisponível — usando método de clusters.")
    return aplicar_seq_real_correction_retornos(
        pred_df,
        motivos_por_data,
        emb_mgr,
        clusters_df,
        emb_repr,
        sim_threshold=sim_threshold,
        max_horizon=horizon,
        scale=scale
    )


# =====================================================================
# PIPELINE COMPLETO
# =====================================================================

def rodar_modelo(csv_path, model_path, out_html, out_recon, nome,
                 impact_model_dir=None, scale=0.4, janela=5, horizon=4):
    df = pd.read_csv(csv_path, parse_dates=["Date"]).sort_values("Date").ffill().bfill()

    model, scaler, train_cols, seq_len = carregar_modelo(model_path)

    pred_df = prever(
        model=model,
        scaler=scaler,
        df=df,
        seq_len=seq_len,
        train_cols=train_cols,
        salvar_recon_plot=out_recon
    )

    emb_mgr = EmbeddingManager()
    clusters_df = pd.read_csv(CLUSTER_CSV)
    emb_repr = emb_mgr.embed_lote(clusters_df["frase_exemplo"].astype(str).tolist())

    motivos_por_data = extrair_motivos(os.path.join(BASE_DIR, "output_noticias"))

    # tenta integrar o impacto com predictor; se não houver, usa clusters
    pred_df = integrar_impacto(
        pred_df,
        motivos_por_data,
        emb_mgr,
        clusters_df,
        emb_repr,
        atv_label=nome,
        impact_model_dir=impact_model_dir,
        scale=scale,
        janela=janela,
        horizon=horizon
    )

    plotar(pred_df, out_html, nome)


# =====================================================================
# MAIN
# =====================================================================

def main():
    # se quiser, altere impact_model_dir para apontar para outro diretório de checkpoints
    impact_model_dir = os.path.join(BASE_DIR, "modelos")

    rodar_modelo(
        csv_path=os.path.join(BASE_DIR, "data", "dados_petr4_brent.csv"),
        model_path=os.path.join(BASE_DIR, "modelos", "lstm_ae_petr4_brent.pt"),
        out_html=os.path.join(BASE_DIR, "img", "petr4_ae_hibrido_retornos.html"),
        out_recon=os.path.join(BASE_DIR, "img", "petr4_ae_recon.png"),
        nome="PETR4",
        impact_model_dir=impact_model_dir,
        scale=1.0,
        janela=5,
        horizon=4
    )

    rodar_modelo(
        csv_path=os.path.join(BASE_DIR, "data", "dados_prio3_brent.csv"),
        model_path=os.path.join(BASE_DIR, "modelos", "lstm_ae_prio3_brent.pt"),
        out_html=os.path.join(BASE_DIR, "img", "prio3_ae_hibrido_retornos.html"),
        out_recon=os.path.join(BASE_DIR, "img", "prio3_ae_recon.png"),
        nome="PRIO3",
        impact_model_dir=impact_model_dir,
        scale=1.0,
        janela=5,
        horizon=4
    )


if __name__ == "__main__":
    main()
