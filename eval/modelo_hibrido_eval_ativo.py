#!/usr/bin/env python3
"""
Modelo Híbrido Avaliativo por Ativo (PETR4 / PRIO3)
----------------------------------------------------

Este script cria um modelo híbrido que combina:

1) PREVISÃO LSTM (modelo base)
   - Apenas série temporal, sem notícias

2) AJUSTE DE IMPACTO POR EVENTOS (parte híbrida real)
   - Cada notícia é comparada com clusters de eventos passados usando embeddings
   - Se similaridade for alta, aplica-se a sequência real seq_d0..seq_d5 daquele cluster
   - Impacto é aplicado nos dias seguintes (D1, D2, …)
   - Interrompe caso um novo evento ocorra no mesmo intervalo temporal

Resultado:
Uma previsão mais realista e responsiva a notícias *sem pesos artificiais*,
usando apenas comportamento histórico real dos clusters.
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
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

# =====================================================================
# CONFIG
# =====================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from utils.embedding_manager import EmbeddingManager
from modelos.model_baseline_lstm import LSTMPrice

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# nomes das colunas seq_d0..seq_d5 no CSV dos clusters
SEQ_COLS = [f"seq_d{i}" for i in range(6)]
CLUSTER_CSV = os.path.join(BASE_DIR, "data", "cluster_motivos.csv")

# saída dos gráficos
OUT_PETR4 = os.path.join(BASE_DIR, "img", "previsao_hibrida_eval_petr4.html")
OUT_PRIO3 = os.path.join(BASE_DIR, "img", "previsao_hibrida_eval_prio3.html")

# =====================================================================
# CHECKPOINT SAFE LOADER (PyTorch 2.6+)
# =====================================================================

def carregar_modelo(model_path):
    """
    Carrega o modelo LSTM salvo no checkpoint (.pt).

    Resolve problemas do PyTorch 2.6+, adicionando tipos que podem ser
    desserializados com segurança (MinMaxScaler, numpy array etc).

    Retorna:
        model          -> modelo LSTM carregado
        scaler         -> MinMaxScaler usado no treino
        train_cols     -> colunas usadas como features no treino
        seq_len        -> tamanho da janela usada no treino
    """
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler

    # adiciona tipos permitidos na desserialização
    torch.serialization.add_safe_globals([
        np.ndarray,
        MinMaxScaler,
        dict,
        list,
        tuple
    ])

    # carrega checkpoint
    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)

    # recria o modelo
    model = LSTMPrice(input_size=len(ckpt["train_columns"])).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, ckpt["scaler"], ckpt["train_columns"], ckpt["seq_len"]



# =====================================================================
# PREVISÃO DO MODELO LSTM
# =====================================================================

def detectar_coluna_close(train_cols):
    """
    Identifica automaticamente qual coluna representa o preço de fechamento
    do ativo (Close_PETR4.SA, Close_PRIO3.SA, Close_BZ=F etc).

    Caso não encontre, tenta fallback para qualquer coluna contendo "Close".
    """
    closes = [c for c in train_cols if c.startswith("Close_")]
    if closes:
        return closes[0]

    closes = [c for c in train_cols if "Close" in c]
    if not closes:
        raise KeyError("Nenhuma coluna Close encontrada no checkpoint.")
    return closes[0]


def prever(model, scaler, df, seq_len, train_cols):
    """
    Gera previsões LSTM para um único ativo.

    Etapas:
    1) Ordena dataset
    2) Garante que todas features necessárias existam
    3) Escala dataset com o scaler do treino
    4) Roda o modelo LSTM janela por janela
    5) Inverte a escala para obter valores reais

    Retorna:
        DataFrame com índice Date contendo:
            Pred -> preço previsto pela LSTM
            Real -> preço real
    """
    df2 = df.copy().sort_values("Date").reset_index(drop=True)

    target_col = detectar_coluna_close(train_cols)
    target_idx = train_cols.index(target_col)

    # garante que todas features existam
    for c in train_cols:
        if c not in df2.columns:
            df2[c] = 0.0

    arr = scaler.transform(df2[train_cols])

    preds, reals, dates = [], [], []
    if len(df2) <= seq_len:
        return pd.DataFrame(columns=["Pred", "Real"])

    # rolling window
    for i in range(len(df2) - seq_len):
        seq = arr[i:i + seq_len]
        seq_tensor = torch.tensor(seq[np.newaxis], dtype=torch.float32).to(DEVICE)

        # previsão
        with torch.no_grad():
            pred_s = model(seq_tensor).cpu().numpy().item()

        # inversão da escala
        zeros = np.zeros((1, len(train_cols)))
        zeros[0, target_idx] = pred_s
        inv = scaler.inverse_transform(zeros)[0, target_idx]

        preds.append(inv)
        reals.append(df2.loc[i + seq_len, target_col])
        dates.append(pd.to_datetime(df2.loc[i + seq_len, "Date"]))

    return pd.DataFrame({"Date": dates, "Pred": preds, "Real": reals}).set_index("Date")



# =====================================================================
# EXTRAÇÃO DE MOTIVOS DO JSON
# =====================================================================

def extrair_motivos(pasta_json):
    """
    Lê arquivos evento_*.json contendo:
        - data
        - motivos_identificados: lista de textos

    Retorna:
        dict {data → lista de motivos(str)}
    """
    arquivos = glob.glob(os.path.join(pasta_json, "evento_*.json"))
    noticias = {}

    for arq in arquivos:
        try:
            j = json.load(open(arq, "r", encoding="utf-8"))
            d = pd.to_datetime(j["data"]).normalize()
            motivos = j.get("motivos_identificados", []) or []
            motivos = [m.strip() for m in motivos if isinstance(m,str) and m.strip()]
            if motivos:
                noticias.setdefault(d, []).extend(motivos)
        except:
            continue

    return noticias



# =====================================================================
# APLICAÇÃO REAL DA SEQUÊNCIA (CORE DO MODELO HÍBRIDO)
# =====================================================================

def aplicar_seq_real(
    pred_df,
    motivos_por_data,
    emb_mgr,
    clusters_df,
    emb_repr,
    sim_threshold=0.7,
    max_horizon=4,
    scale=0.4
):
    """
    Aplica impacto real encontrado nos clusters às previsões LSTM.

    Lógica:
    ------
    1) Para cada evento (data e motivos):
       - Gera embedding da frase do motivo.
       - Compara com embeddings dos clusters (frase_exemplo).
       - Seleciona o cluster mais parecido.

    2) Se a similaridade >= sim_threshold:
       - Obtém a sequência real seq_d0..seq_d5
       - Aplica efeito multiplicativo:
            Pred_Ajustado[dia+k] *= (1 + impacto%)

    3) Se um novo evento ocorre depois do evento atual,
       interrompe a sequência (regra realista).

    Parâmetros:
        sim_threshold -> similaridade mínima para aceitar um cluster
        max_horizon   -> máximo Dk aplicado (até D4 ou D5)
        scale         -> suavização global do impacto

    Retorna:
        pred_df com coluna Pred_Ajustado atualizada
    """
    df = pred_df.copy()

    if "Pred_Ajustado" not in df:
        df["Pred_Ajustado"] = df["Pred"].copy()

    dias = df.index
    eventos = sorted(motivos_por_data.keys())

    for data_evt in eventos:
        motivos = motivos_por_data[data_evt]
        if not motivos:
            continue

        # posição do evento no índice da previsão
        pos = dias.searchsorted(data_evt)
        if pos >= len(dias):
            continue

        # eventos futuros para a interrupção da sequência
        futuras = [d for d in eventos if d > data_evt]

        # processa todos os motivos daquela data
        for motivo in motivos:

            emb = emb_mgr.embed(motivo)
            norm_e = np.linalg.norm(emb)

            # similaridade com todos clusters
            norms_repr = np.linalg.norm(emb_repr, axis=1)
            sims = (emb.ravel() @ emb_repr.T) / (norm_e * norms_repr + 1e-12)

            idx_best = int(np.argmax(sims))
            best_sim = float(sims[idx_best])

            # rejeita motivos com pouca similaridade
            if best_sim < sim_threshold:
                continue

            # cluster selecionado
            cluster_id = int(clusters_df.iloc[idx_best]["cluster"])
            row = clusters_df[clusters_df["cluster"] == cluster_id].iloc[0]

            # extrai seq_d0..seq_dk
            seq = []
            for k in range(max_horizon + 1):
                col = f"seq_d{k}"
                seq.append(float(row[col]) if col in row and pd.notna(row[col]) else None)

            # aplica impacto progressivo
            for k, impacto in enumerate(seq):
                if impacto is None:
                    break

                idx = pos + k
                if idx >= len(dias):
                    break

                diaK = dias[idx]

                # interrupção se houver novo evento
                if any((f > data_evt and f <= diaK) for f in futuras):
                    break

                # impacto real do cluster (multiplicativo)
                ajuste = scale * best_sim * (impacto / 100.0)

                df.loc[diaK, "Pred_Ajustado"] *= (1.0 + ajuste)

                print(f"[SEQ] Evento={data_evt.date()} motivo='{motivo}' cluster={cluster_id} D{k}: {impacto:+.2f}% sim={best_sim:.3f}")

    return df



# =====================================================================
# PLOT
# =====================================================================

def plotar(pred_df, out_html, nome):
    """
    Salva gráfico interativo com:
        - preço real
        - previsão LSTM
        - previsão híbrida ajustada por notícias
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pred_df.index, y=pred_df["Real"], name="Real", line=dict(color="black")))
    fig.add_trace(go.Scatter(x=pred_df.index, y=pred_df["Pred"], name="LSTM", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=pred_df.index, y=pred_df["Pred_Ajustado"], name="Híbrido", line=dict(color="orange")))

    rmse = np.sqrt(mean_squared_error(pred_df["Real"], pred_df["Pred_Ajustado"]))
    r2 = r2_score(pred_df["Real"], pred_df["Pred_Ajustado"])

    fig.update_layout(
        title=f"{nome} Híbrido Real | RMSE={rmse:.4f} | R²={r2:.4f}",
        template="plotly_white"
    )

    fig.write_html(out_html)
    print("✔ Salvo em:", out_html)
    print("RMSE antes:", np.sqrt(mean_squared_error(pred_df["Real"], pred_df["Pred"])))
    print("RMSE depois:", rmse)



# =====================================================================
# ORQUESTRAÇÃO (PIPELINE COMPLETO)
# =====================================================================

def rodar_modelo(csv_path, model_path, out_html, nome):
    """
    Fluxo completo de execução para um ativo:

    1) Carrega CSV do ativo
    2) Gera previsão LSTM
    3) Carrega clusters e embeddings
    4) Extrai motivos dos eventos
    5) Aplica sequência real de impacto
    6) Gera gráfico final
    """
    print(f"\n=========== {nome} ===========")

    df = pd.read_csv(csv_path, parse_dates=["Date"]).sort_values("Date").ffill().bfill()

    model, scaler, train_cols, seq_len = carregar_modelo(model_path)
    pred_df = prever(model, scaler, df, seq_len, train_cols)

    emb_mgr = EmbeddingManager()
    clusters_df = pd.read_csv(CLUSTER_CSV)
    emb_repr = emb_mgr.embed_lote(clusters_df["frase_exemplo"].astype(str).tolist())

    motivos_por_data = extrair_motivos(os.path.join(BASE_DIR, "output_noticias"))

    pred_df = aplicar_seq_real(
        pred_df,
        motivos_por_data,
        emb_mgr,
        clusters_df,
        emb_repr
    )

    plotar(pred_df, out_html, nome)



# =====================================================================
# MAIN – Executa PETR4 e PRIO3
# =====================================================================

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
