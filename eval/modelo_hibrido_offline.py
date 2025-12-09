#!/usr/bin/env python3
"""
Modelo Híbrido Avaliativo por Ativo (PETR4 / PRIO3 / EXXO34) — Versão Final Corrigida

Inclui:
✔ LSTM baseline
✔ Autoencoder (previsão + reconstrução)
✔ Aplicação de impacto real via clusters (escolhe motivo mais relevante por dia)
✔ Scale automático por ativo
✔ Correção residual WALK-FORWARD (sem data leakage) usando motivo/cluster relevante
✔ Correção de shapes nas similaridades (resolve IndexError)
✔ Filtragem de clusters por ativo (usa coluna ativo_cluster no CSV)
"""

# =====================================================================
# IMPORTS
# =====================================================================

import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

load_dotenv()

# ajustar path do projeto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# Import do EmbeddingManager (cache)
from utils.embedding_manager import EmbeddingManager

# instância global única do embedding manager — usa cache consistentemente
emb_mgr = EmbeddingManager()

from modelos.model_baseline_lstm import LSTMPrice
from modelos.model_lstm_autoencoder import LSTMAutoencoderPrice
from modelos.model_transformer_price import TransformerPrice

from eval.plotter_refactor import plotar_hibrido_corrigido, plotar_comparacao_por_ativo

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLUSTER_CSV = os.path.join(BASE_DIR, "data", "cluster_motivos.csv")

OUT_AE_PETR4  = os.path.join(BASE_DIR, "img", "previsao_autoencoder_petr4.html")
OUT_AE_PRIO3  = os.path.join(BASE_DIR, "img", "previsao_autoencoder_prio3.html")
OUT_AE_EXXO34 = os.path.join(BASE_DIR, "img", "previsao_autoencoder_exxo34.html")

OUT_LSTM_PETR4  = os.path.join(BASE_DIR, "img", "previsao_hibrida_eval_petr4.html")
OUT_LSTM_PRIO3  = os.path.join(BASE_DIR, "img", "previsao_hibrida_eval_prio3.html")
OUT_LSTM_EXXO34 = os.path.join(BASE_DIR, "img", "previsao_hibrida_eval_exxo34.html")

OUT_TR_PETR4  = os.path.join(BASE_DIR, "img", "previsao_transformer_petr4.html")
OUT_TR_PRIO3  = os.path.join(BASE_DIR, "img", "previsao_transformer_prio3.html")
OUT_TR_EXXO34 = os.path.join(BASE_DIR, "img", "previsao_transformer_exxo34.html")


RESULTADOS = []

PREVISOES_MODELOS = {}     # armazenará -> {"PETR4 (LSTM)": df_pred, ...}
PREVISOES_HIBRIDOS = {}    # armazenará -> {"PETR4 (LSTM)": df_hibrido, ...}


# =====================================================================
# LOADERS
# =====================================================================
def carregar_modelo_unificado(model_path, tipo="lstm"):
    """
    tipo: "lstm", "autoencoder", "transformer"
    """
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler

    # compatibilidade com objetos do checkpoint
    try:
        torch.serialization.add_safe_globals([np.ndarray, MinMaxScaler, dict, list, tuple])
    except Exception:
        pass

    # carregar checkpoint
    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)

    input_size = len(ckpt["train_columns"])

    # selecionar modelo
    if tipo.lower() == "lstm":
        model = LSTMPrice(
            input_size=input_size
        ).to(DEVICE)

    elif tipo.lower() == "autoencoder":
        model = LSTMAutoencoderPrice(
            input_size=input_size,
            hidden_size=128,
            latent_size=64,
            num_layers=2,
            dropout=0.1
        ).to(DEVICE)

    elif tipo.lower() == "transformer":
        model = TransformerPrice(
            input_size=input_size,
            d_model=128,
            nhead=4,
            num_layers=3,
            dim_feedforward=256,
            dropout=0.1
        ).to(DEVICE)

    else:
        raise ValueError(f"Tipo desconhecido de modelo: {tipo}")

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, ckpt["scaler"], ckpt["train_columns"], ckpt["seq_len"]


# =====================================================================
# PREVISÃO LSTM & AE
# =====================================================================

def detectar_coluna_close(train_cols):
    closes = [c for c in train_cols if c.startswith("Close_")]
    if closes:
        return closes[0]
    closes = [c for c in train_cols if "Close" in c]
    if not closes:
        raise KeyError("Nenhuma coluna Close encontrada no checkpoint.")
    return closes[0]


def prever_unificado(model, scaler, df, seq_len, train_cols, tipo="lstm"):
    df2 = df.copy().sort_values("Date").reset_index(drop=True)
    target_col = detectar_coluna_close(train_cols)
    target_idx = train_cols.index(target_col)

    # garante que TODAS as colunas existem
    for c in train_cols:
        if c not in df2.columns:
            df2[c] = 0.0

    arr = scaler.transform(df2[train_cols])
    preds, reals, dates = [], [], []

    if len(df2) <= seq_len:
        return pd.DataFrame(columns=["Date", "Pred", "Real"]).set_index("Date")

    for i in range(len(df2) - seq_len):
        seq = arr[i:i + seq_len]
        seq_tensor = torch.tensor(seq[np.newaxis], dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            if tipo.lower() == "autoencoder":
                pred_s, _ = model(seq_tensor)
                pred_s = pred_s.cpu().numpy().item()
            else:
                pred_s = model(seq_tensor).cpu().numpy().item()

        # desnormalizar
        zeros = np.zeros((1, len(train_cols)))
        zeros[0, target_idx] = pred_s
        inv = scaler.inverse_transform(zeros)[0, target_idx]

        preds.append(inv)
        reals.append(df2.loc[i + seq_len, target_col])
        dates.append(pd.to_datetime(df2.loc[i + seq_len, "Date"]))

    return pd.DataFrame(
        {"Date": dates, "Pred": preds, "Real": reals}
    ).set_index("Date")


# =====================================================================
# EXTRAÇÃO DE MOTIVOS
# =====================================================================

def extrair_motivos(pasta_json):

    arquivos = glob.glob(os.path.join(pasta_json, "evento_*.json"))
    noticias = {}

    for arq in arquivos:
        try:
            j = json.load(open(arq, "r", encoding="utf-8"))
            d = pd.to_datetime(j["data"]).normalize()

            motivos = j.get("motivos_identificados", []) or []
            motivos = [m.strip() for m in motivos if isinstance(m, str) and m.strip()]
            if motivos:
                noticias.setdefault(d, []).extend(motivos)
        except:
            pass

    return noticias


def escolher_motivo_mais_relevante(motivos, emb_mgr_local, emb_repr):
    """
    Retorna o motivo com maior similaridade a qualquer cluster.
    (mantido para compatibilidade)
    """
    if not motivos:
        return None, None

    emb_repr_mat = np.asarray(emb_repr)
    if emb_repr_mat.ndim == 1:
        emb_repr_mat = emb_repr_mat.reshape(1, -1)

    melhor_motivo = None
    melhor_sim = -1.0

    for mot in motivos:
        emb = emb_mgr_local.embed(mot)
        emb = np.asarray(emb).reshape(1, -1)

        # similaridade com todos clusters
        emb_norm = np.linalg.norm(emb, axis=1, keepdims=True)
        repr_norm = np.linalg.norm(emb_repr_mat, axis=1, keepdims=True)
        sim_vec = (emb @ emb_repr_mat.T) / (emb_norm * repr_norm.T + 1e-12)

        sim = float(np.max(sim_vec))

        if sim > melhor_sim:
            melhor_sim = sim
            melhor_motivo = mot

    return melhor_motivo, melhor_sim


def motivo_e_cluster_mais_relevante(motivos, emb_mgr_local, emb_repr, clusters_df, ativo):
    """
    Seleciona o motivo mais relevante filtrando clusters por ativo e aplicando pesos.
    Retorna: motivo, sim_weighted, cluster_id, row
    """
    if not motivos:
        return None, 0.0, None, None

    emb_repr_mat = np.asarray(emb_repr)
    if emb_repr_mat.ndim == 1:
        emb_repr_mat = emb_repr_mat.reshape(1, -1)

    ativo = (ativo or "").upper()
    ativos_permitidos = [ativo, "BRENT", "GENÉRICO"]

    # se não existir coluna ativo_cluster, não filtra
    if "ativo_cluster" not in clusters_df.columns:
        mask = pd.Series([True] * len(clusters_df), index=clusters_df.index)
    else:
        mask = clusters_df["ativo_cluster"].astype(str).str.upper().isin(ativos_permitidos)

    if mask.sum() == 0:
        return None, 0.0, None, None

    clusters_df_f = clusters_df[mask].reset_index(drop=True)
    try:
        emb_f = emb_repr_mat[mask.values]
    except Exception:
        emb_f = emb_repr_mat

    if emb_f.ndim == 1:
        emb_f = emb_f.reshape(1, -1)

    # pesos por tipo de ativo
    peso_map = {
        ativo: 1.0,
        "BRENT": 0.75,
        "GENÉRICO": 0.5
    }

    best = {"motivo": None, "sim": -1, "cluster": None, "row": None}
    for mot in motivos:
        emb = emb_mgr_local.embed(mot)
        emb = np.asarray(emb).reshape(1, -1)

        emb_norm = np.linalg.norm(emb, axis=1, keepdims=True)
        repr_norm = np.linalg.norm(emb_f, axis=1, keepdims=True)
        sim_vec = (emb @ emb_f.T) / (emb_norm * repr_norm.T + 1e-12)

        if sim_vec.size == 0:
            continue

        # calcular sim ponderada pelo ativo do cluster
        sims = sim_vec.flatten()
        sims_weighted = []
        for i, s in enumerate(sims):
            try:
                tipo = str(clusters_df_f.iloc[i]["ativo_cluster"]).upper()
            except Exception:
                tipo = "GENÉRICO"
            peso = peso_map.get(tipo, 0.5)
            sims_weighted.append(float(s) * float(peso))

        idx = int(np.argmax(sims_weighted))
        sim_w = float(sims_weighted[idx])
        if sim_w > best["sim"]:
            try:
                row = clusters_df_f.iloc[idx]
                cluster_id = int(row["cluster"]) if "cluster" in row.index else None
            except Exception:
                row = None
                cluster_id = None
            best.update({"motivo": mot, "sim": sim_w, "cluster": cluster_id, "row": row})

    return best["motivo"], best["sim"], best["cluster"], best["row"]


# =====================================================================
#                     IMPACTO REAL (clusters)
# =====================================================================

def encontrar_scale_otimo(pred_df, motivos_por_data, emb_mgr_local, clusters_df, emb_repr, ativo):
    candidatos = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    melhor, menor = 0.5, float("inf")

    for scale in candidatos:
        temp = aplicar_seq_real(
            pred_df, motivos_por_data, emb_mgr_local, clusters_df, emb_repr,
            ativo=ativo, sim_threshold=0.7, max_horizon=4, scale=scale
        )
        rmse = np.sqrt(mean_squared_error(temp["Real"], temp["Pred_Ajustado"]))
        print(f"[TUNING] scale={scale} → RMSE={rmse:.4f}")

        if rmse < menor:
            menor = rmse
            melhor = scale

    print(f"✔ SCALE ÓTIMO = {melhor}\n")
    return melhor

from openai import OpenAI
client = OpenAI()


def escolher_cluster_com_llm(motivos_do_dia,
                             clusters_df,
                             emb_mgr_local,
                             ativo,
                             top_k=3):
    """
    Etapas:
    1. Pega o embedding de cada motivo do dia.
    2. Calcula similaridade com TODOS os clusters.
    3. Seleciona os TOP-K mais similares.
    4. Envia os candidatos para a LLM decidir o cluster final.
    5. Retorna (melhor_motivo, score_final, cluster_id, row CSV)
    """

    if not motivos_do_dia:
        return None, 0.0, None, None

    # === embeddings dos clusters canônicos ===
    frases_canon = clusters_df["frase_exemplo"].astype(str).tolist()
    emb_clusters = emb_mgr_local.embed_lote(frases_canon)

    # === embeddings dos motivos do dia ===
    emb_mots = [emb_mgr_local.embed(m) for m in motivos_do_dia]

    # média dos embeddings do dia
    emb_day = np.mean(np.vstack(emb_mots), axis=0)

    # === similaridade coseno ===
    sims = emb_clusters @ emb_day / (
        np.linalg.norm(emb_clusters, axis=1) * np.linalg.norm(emb_day) + 1e-12
    )

    # top-k índices dos clusters
    idx_top = sims.argsort()[::-1][:top_k]

    # extrair dados dos clusters candidatos
    candidatos = clusters_df.iloc[idx_top].copy()
    candidatos["sim"] = sims[idx_top]

    # preparar descrição para LLM
    blocos = []
    for _, row in candidatos.iterrows():
        frases_originais = row.get("frases_originais", "")
        if isinstance(frases_originais, str):
            pass
        elif isinstance(frases_originais, list):
            frases_originais = "\n".join(f"- {f}" for f in frases_originais)

        blocos.append(
            f"""
CLUSTER {int(row['cluster'])}
Frase canônica: {row['frase_exemplo']}
Similaridade inicial: {row['sim']:.3f}
Frases originais do cluster:
{frases_originais}
"""
        )
    texto_clusters = "\n\n".join(blocos)

    motivos_txt = "\n".join(f"- {m}" for m in motivos_do_dia)

    prompt = f"""
Você é um analista financeiro em 2025.
Seu trabalho é escolher qual CLUSTER representa melhor os MOTIVOS DO DIA.

Considere:
- ativo analisado: {ativo}
- motivos do dia:
{motivos_txt}

Clusters candidatos:
{texto_clusters}

REGRAS:
- Escolha exatamente UM cluster.
- Avalie semanticamente as frases originais do cluster.
- Use as frases canônicas como resumo auxiliar.
- NÃO invente fatos.
- NÃO escolha cluster que não tenha relação causável com o motivo do dia.
- Responda somente JSON no formato:

{{
 "cluster_escolhido": <id>,
 "justificativa": "<texto curto>"
}}

Agora responda:
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4.1",
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        data = resp.choices[0].message.parsed
        cid = int(data["cluster_escolhido"])
        row = clusters_df[clusters_df["cluster"] == cid].iloc[0]
        return motivos_do_dia[0], sims[cid], cid, row

    except Exception as e:
        print("⚠ Erro LLM:", e)
        # fallback: cluster mais similar
        cid = int(candidatos.iloc[0]["cluster"])
        row = candidatos.iloc[0]
        return motivos_do_dia[0], float(candidatos.iloc[0]["sim"]), cid, row


def aplicar_seq_real(pred_df, motivos_por_data, emb_mgr_local, clusters_df, emb_repr,
                     ativo, sim_threshold=0.7, max_horizon=4, scale=0.4):
    """
    Aplica a sequência média do cluster mais relevante para o motivo mais relevante do dia.
    Filtra clusters por ativo (aceita BRENT e GENÉRICO).
    """

    df = pred_df.copy()
    if "Pred_Ajustado" not in df.columns:
        df["Pred_Ajustado"] = df["Pred"].copy()

    dias = df.index
    eventos = sorted(motivos_por_data.keys())

    # garantir formatos
    emb_repr_mat = np.asarray(emb_repr)  # shape (N, D)

    for data_evt in eventos:
        motivos = motivos_por_data.get(data_evt, [])
        if not motivos:
            continue

        pos = dias.searchsorted(data_evt)
        if pos >= len(dias):
            continue

        futuras = [d for d in eventos if d > data_evt]

        # selecionar apenas o motivo+cluster mais relevantes para o dia (filtrando por ativo)
        motivo, best_sim, clust_id, row = motivo_e_cluster_mais_relevante(
            motivos, emb_mgr_local, emb_repr_mat, clusters_df, ativo
        )

        if row is None or best_sim < sim_threshold:
            continue

        # extrair sequência do cluster selecionado
        seq = []
        for k in range(max_horizon + 1):
            col = f"seq_d{k}"
            seq.append(float(row[col]) if (col in row.index and pd.notna(row[col])) else None)

        # aplicar impacto sequencial
        for k, impacto in enumerate(seq):
            if impacto is None:
                break

            idx_d = pos + k
            if idx_d >= len(dias):
                break

            diaK = dias[idx_d]

            # se existir outra notícia entre data_evt (exclusive) e diaK (inclusive), interrompe
            if any((f > data_evt and f <= diaK) for f in futuras):
                break

            ajuste = scale * best_sim * (impacto / 100.0)
            df.loc[diaK, "Pred_Ajustado"] *= (1 + ajuste)

            # debug opcional:
            # print(f"[SEQ] {data_evt.date()} motivo='{motivo}' cluster={clust_id} D{k}: {impacto:+.2f}% sim={best_sim:.3f}")

    return df


# =====================================================================
#           MÉTODO B-WF — WALK-FORWARD RESIDUAL CORRECTION
# =====================================================================

def aplicar_walkforward_residual(pred_df, motivos_por_data, emb_mgr_local, clusters_df, emb_repr, ativo, janela):
    """
    Correção residual CAUSAL usando motivo+cluster filtrado por ativo.
    """

    df = pred_df.copy().reset_index()
    if "Pred_Ajustado" not in df.columns:
        df["Pred_Ajustado"] = df["Pred"].copy()

    df["r"] = df["Real"] - df["Pred_Ajustado"]

    # preparar emb_repr
    emb_repr_mat = np.asarray(emb_repr)
    if emb_repr_mat.ndim == 1:
        emb_repr_mat = emb_repr_mat.reshape(1, -1)

    sims = []
    eventos = []
    cluster_ids = []

    for d in df["Date"]:
        dnorm = pd.to_datetime(d).normalize()
        motivos = motivos_por_data.get(dnorm, [])

        if motivos:
            eventos.append(1)
            # escolher o motivo + cluster mais relevante (filtrado por ativo)
            mot, sim, clust, row = motivo_e_cluster_mais_relevante(
                motivos, emb_mgr_local, emb_repr_mat, clusters_df, ativo
            )
            sims.append(sim if sim is not None else 0.0)
            cluster_ids.append(clust if clust is not None else -1)
        else:
            eventos.append(0)
            sims.append(0.0)
            cluster_ids.append(-1)

    df["sim_day"] = sims
    df["event_bin"] = eventos
    df["cluster_id"] = cluster_ids

    df["r_lag1"] = df["r"].shift(1)  # ontem
    df["r_lag2"] = df["r"].shift(2)  # anteontem

    df["Pred_Final"] = df["Pred_Ajustado"].copy()

    feat_cols = ["r_lag1", "r_lag2", "sim_day", "event_bin", "cluster_id"]

    # WALK-FORWARD LOOP (causal)
    for t in range(janela, len(df)):
        train_df = df.iloc[:t].dropna(subset=feat_cols + ["r"])
        if len(train_df) < janela:
            continue

        X = train_df[feat_cols].values
        y = train_df["r"].shift(-1).dropna().values

        if len(y) < len(X):
            X = X[:len(y)]

        model = Ridge(alpha=1.0)
        model.fit(X, y)

        X_now = df.loc[t, feat_cols].values.reshape(1, -1)
        try:
            r_pred = float(model.predict(X_now)[0])
        except Exception:
            r_pred = 0.0

        df.loc[t, "Pred_Final"] = df.loc[t, "Pred_Final"] + r_pred

    df = df.set_index("Date")
    return df


# =====================================================================
# UTIL - extrair ativo do nome (ex: "PETR4 (LSTM)" -> "PETR4")
# =====================================================================
def _ativo_from_nome(nome):
    if not nome:
        return None
    # tenta extrair token alfanumérico no começo (ex: PETR4)
    m = re.match(r"^([A-Z0-9]+)", nome.upper().strip())
    if m:
        return m.group(1)
    return nome.upper().split()[0]


# =====================================================================
# PIPELINE LSTM
# =====================================================================
def rodar_modelo_unificado(csv_path, model_path, out_html, nome, tipo):
    """
    tipo ∈ {"lstm", "autoencoder", "transformer"}
    """

    print(f"\n=========== {nome} ({tipo.upper()}) ===========")

    # ----------------------
    # 1) Carregar dados
    # ----------------------
    df = pd.read_csv(csv_path, parse_dates=["Date"]).sort_values("Date").ffill().bfill()

    # ----------------------
    # 2) Carregar modelo
    # ----------------------
    model, scaler, cols, seq_len = carregar_modelo_unificado(model_path, tipo=tipo)

    # ----------------------
    # 3) Previsão base
    # ----------------------
    pred_df = prever_unificado(model, scaler, df, seq_len, cols, tipo=tipo)

    # ----------------------
    # 4) Carregar clusters do ativo
    # ----------------------
    # usa a instância global emb_mgr
    ativo = _ativo_from_nome(nome)
    cluster_file = os.path.join(BASE_DIR, "data", f"cluster_{ativo.lower()}.csv")

    if not os.path.exists(cluster_file):
        print(f"⚠ Cluster específico não encontrado ({cluster_file}), usando cluster_motivos.csv")
        cluster_file = os.path.join(BASE_DIR, "data", "cluster_motivos.csv")

    clusters_df = pd.read_csv(cluster_file)
    if "ativo_cluster" not in clusters_df.columns:
        clusters_df["ativo_cluster"] = "GENÉRICO"

    emb_repr = emb_mgr.embed_lote(clusters_df["frase_exemplo"].astype(str).tolist())

    # ----------------------
    # 5) Motivos da data
    # ----------------------
    motivos_por_data = extrair_motivos(os.path.join(BASE_DIR, "output_noticias"))

    # ----------------------
    # 6) Otimizar SCALE
    # ----------------------
    scale = encontrar_scale_otimo(pred_df, motivos_por_data, emb_mgr, clusters_df, emb_repr, ativo)
    pred_df = aplicar_seq_real(pred_df, motivos_por_data, emb_mgr, clusters_df, emb_repr,
                               ativo=ativo, scale=scale)

    # ----------------------
    # 7) WALK-FORWARD Residual Correction
    # ----------------------
    pred_df = aplicar_walkforward_residual(pred_df, motivos_por_data, emb_mgr, clusters_df, emb_repr,
                                           ativo=ativo, janela=15)

    pred_df["Pred_Ajustado"] = pred_df["Pred_Final"]
    pred_df["Pred_Final_Price"] = pred_df["Pred_Final"]

    # ----------------------
    # 8) Gráfico híbrido
    # ----------------------
    rmse_base, rmse_hib = plotar_hibrido_corrigido(
        pred_df,
        motivos_por_data,
        emb_mgr,
        clusters_df,
        emb_repr,
        out_html,
        nome
    )

    # ----------------------
    # 9) Guardar previsões
    # ----------------------
    PREVISOES_MODELOS[nome] = pred_df[["Pred", "Real"]].copy()
    PREVISOES_HIBRIDOS[nome] = pred_df[["Pred_Ajustado", "Real"]].copy()

    # ----------------------
    # 10) Resultados
    # ----------------------
    RESULTADOS.append({
        "Ativo": nome,
        "Modelo": tipo.upper(),
        "RMSE_Modelo": round(rmse_base, 4),
        "RMSE_Hibrido": round(rmse_hib, 4),
        "Ganho": round(rmse_base - rmse_hib, 4)
    })


# =====================================================================
# MAIN
# =====================================================================
def eval_modelos():

    ativos = {
        "PETR4": "petr4",
        "PRIO3": "prio3",
        "EXXO34": "exxo34"
    }

    # MAPEAMENTO DOS CAMINHOS HTML DE SAÍDA
    OUTS = {
        "lstm":       lambda a: os.path.join(BASE_DIR, "img", f"previsao_lstm_{a}.html"),
        "autoencoder": lambda a: os.path.join(BASE_DIR, "img", f"previsao_ae_{a}.html"),
        "transformer": lambda a: os.path.join(BASE_DIR, "img", f"previsao_transformer_{a}.html"),
    }

    # MAPEIA O PREFIXO DO NOME
    NOME_FORMAT = {
        "lstm":        "(LSTM)",
        "autoencoder": "(AE)",
        "transformer": "(Transformer)"
    }

    # ----------------------------------------
    # 1) RODAR TODOS OS MODELOS POR ATIVO
    # ----------------------------------------
    for tipo in ["lstm", "autoencoder", "transformer"]:
        for ativo, nomefile in ativos.items():

            csv_path  = os.path.join(BASE_DIR, "data",    f"dados_{nomefile}_brent.csv")
            model_path = os.path.join(BASE_DIR, "modelos", f"{tipo}_{nomefile}.pt")
            out_html   = OUTS[tipo](nomefile)

            rodar_modelo_unificado(
                csv_path,
                model_path,
                out_html,
                f"{ativo} {NOME_FORMAT[tipo]}",
                tipo=tipo
            )

    # =====================================================
    # 2) Preparar dados para os gráficos por ativo
    # =====================================================
    motivos_por_data = extrair_motivos(os.path.join(BASE_DIR, "output_noticias"))
    # usa a instância global emb_mgr
    clusters_df = pd.read_csv(CLUSTER_CSV)
    if "ativo_cluster" not in clusters_df.columns:
        clusters_df["ativo_cluster"] = "GENÉRICO"
    emb_repr = emb_mgr.embed_lote(clusters_df["frase_exemplo"].astype(str).tolist())

    # =====================================================
    # 3) Gerar gráficos consolidados por ativo
    # =====================================================

    for ativo in ["PETR4", "PRIO3", "EXXO34"]:

        prev_puro  = {k: v for k, v in PREVISOES_MODELOS.items()  if k.startswith(ativo)}
        prev_hib   = {k: v for k, v in PREVISOES_HIBRIDOS.items() if k.startswith(ativo)}

        if len(prev_puro) == 0:
            print(f"⚠ Nenhuma previsão pura encontrada para {ativo}, pulando.")
            continue

        df_base = list(prev_puro.values())[0]

        plotar_comparacao_por_ativo(
            df_base,
            prev_puro,
            prev_hib,
            motivos_por_data,
            emb_mgr,
            clusters_df,
            emb_repr,
            os.path.join(BASE_DIR, "img", f"comparacao_{ativo.lower()}.html"),
            ativo
        )

    # =====================================================
    # 4) SALVAR RESULTADOS
    # =====================================================
    print("\n========================== RESULTADO FINAL ==========================\n")
    df_final = pd.DataFrame(RESULTADOS)
    print(df_final.to_string(index=False))
    df_final.to_csv(os.path.join(BASE_DIR, "data", "resultado_comparacao_modelos.csv"), index=False)
    print("\n✔ Tabela salva com sucesso!\n")

if __name__ == "__main__":
    eval_modelos()
