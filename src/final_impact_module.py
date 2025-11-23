# final_impact_module.py
"""
Módulo final de aplicação de impacto de notícias (não-cumulativo, baseado em biblioteca histórica).
- detecta cluster por motivos (usando embeddings, se disponíveis)
- busca ref_seq em impact_library.json
- escala por alpha (métodos: 'zscore' ou 'ratio'), com caps
- cria mapa de eventos onde eventos posteriores sobrescrevem anteriores
- aplica sequência (D0..DH) multiplicativa sobre Pred:
      Pred_impactado = Pred * (1 + seq_k/100)
Retorna df com coluna 'Pred_Impact' (cópia de Pred se nada aplicado) e registros de aplicação.
"""
from typing import Dict, List, Tuple, Optional
import os
import json
import numpy as np
from datetime import timedelta
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Optional dependencies to call OpenAI embeddings if available in env:
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False

# Configs (ajustáveis)
BASE_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
IMPACT_LIB_PATH = "../data/impact_library.json"
EMB_PATH = "../data/embeddings_frases.npy"
EMB_META = "../data/embeddings_frases_meta.csv"
OPENAI_MODEL = "text-embedding-3-small"



def _load_impact_library(path=IMPACT_LIB_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"impact_library.json não encontrado: {path}")
    return json.load(open(path, "r", encoding="utf-8"))


def _load_embeddings_meta():
    if not os.path.exists(EMB_PATH) or not os.path.exists(EMB_META):
        return None, None
    emb = np.load(EMB_PATH)
    meta = pd.read_csv(EMB_META)
    return emb, meta


def _create_embedding(text, client):
    r = client.embeddings.create(model=OPENAI_MODEL, input=[text])
    return np.array(r.data[0].embedding).reshape(1, -1)


def detectar_cluster_por_motivos(motivos, emb_hist=None, meta_hist=None, openai_client=None, cluster_forcado=None):
    if cluster_forcado is not None:
        return cluster_forcado, "forçado", 1.0

    frase = " ".join(motivos).lower()

    # Regras simples para petróleo / brent
    if any(k in frase for k in ["petróleo", "petroleo", "brent", "oil"]):
        if any(w in frase for w in ["alta", "aumenta", "demanda", "procura", "aperta"]):
            return "posi_brent", "regra petróleo +", 0.9
        if any(w in frase for w in ["queda", "desacelera", "fraca", "reduz"]):
            return "neg_brent", "regra petróleo -", 0.9

    # Regras simples para PETR4 / Petrobras
    if any(k in frase for k in ["petrobras", "petr4"]):
        if any(w in frase for w in ["alta", "ganho", "positivo", "recupera"]):
            return "posi_petr4", "regra PETR4 +", 0.9
        if any(w in frase for w in ["queda", "perda", "negativo", "recuo"]):
            return "neg_petr4", "regra PETR4 -", 0.9

    # Fallback: usar embeddings históricos se disponíveis e se OpenAI client for passado
    if emb_hist is not None and meta_hist is not None and openai_client:
        try:
            emb = _create_embedding(" ".join(motivos), openai_client)
            sims = cosine_similarity(emb, emb_hist)[0]
            idx = int(np.argmax(sims))
            cluster = meta_hist.iloc[idx].get("cluster", "outros")
            motivo_ref = meta_hist.iloc[idx].get("motivo", "")
            sim = float(sims[idx])
            return cluster, motivo_ref, sim
        except Exception:
            pass

    return "outros", "", 0.0


def compute_alpha(r_new0, r_ref0, hist_std=None, method="zscore", alpha_cap=(0.25, 4.0)):
    if r_ref0 == 0 or np.isnan(r_ref0):
        alpha = 1.0
    else:
        if method == "ratio":
            alpha = r_new0 / r_ref0
        else:
            if hist_std is None or hist_std == 0:
                alpha = r_new0 / r_ref0
            else:
                z_new = r_new0 / hist_std
                z_ref = r_ref0 / hist_std
                alpha = 1.0 if z_ref == 0 else z_new / z_ref

    amin, amax = alpha_cap
    return max(amin, min(amax, float(alpha)))


def apply_impact_sequences(pred_df, motivos_por_data, verbose=True, horizon=5,
                           alpha_method="zscore", alpha_cap=(0.25, 4.0),
                           max_pct_per_day=0.25, openai_client=None):
    """
    pred_df: DataFrame com colunas 'Pred' e 'Real', index = pregões (datetime)
    motivos_por_data: dict {date -> {"motivos":[...], "cluster_forcado": optional}}
    Retorna: (df_adjusted, logs)
    """

    df = pred_df.copy().sort_index()
    lib = _load_impact_library()
    emb_hist, meta_hist = _load_embeddings_meta()

    trading_index = df.index
    mapa = {}
    logs = []

    # inicializa coluna metadata como object (permite dicts)
    df["Pred_Impact"] = df["Pred"].copy()
    df["Impact_Metadata"] = [None] * len(df)
    df["Impact_Metadata"] = df["Impact_Metadata"].astype(object)

    for k, info in motivos_por_data.items():
        dt = pd.to_datetime(k).normalize()
        # aceitar dois formatos: info pode ser lista de motivos (compatibilidade) ou dict
        if isinstance(info, dict):
            motivos = info.get("motivos", [])
            cluster_forcado = info.get("cluster_forcado", None)
        else:
            motivos = info
            cluster_forcado = None

        pos = trading_index.searchsorted(dt)
        if pos >= len(trading_index):
            if verbose:
                print(f"[WARN] Evento {dt.date()} depois do último pregão disponível. Pulando.")
            continue
        data_evento = trading_index[pos]

        cluster, motivo_ref, sim = detectar_cluster_por_motivos(
            motivos, emb_hist, meta_hist, openai_client=openai_client, cluster_forcado=cluster_forcado
        )

        if cluster not in lib:
            if verbose:
                print(f"[WARN] Cluster '{cluster}' não tem sequência no impact_library.json. Pulando evento {data_evento.date()}.")
            continue

        ref_seq = np.array(lib[cluster]["ref_seq"], dtype=float)
        r_ref0 = float(ref_seq[0]) if len(ref_seq) > 0 else 0.0

        idx_evt = trading_index.get_loc(data_evento)
        if idx_evt == 0:
            if verbose:
                print(f"[WARN] Evento em primeiro pregão ({data_evento.date()}), sem dia anterior para calcular r_new0. Pulando.")
            continue

        prev_day = trading_index[idx_evt - 1]

        # usar Pred para extrapolar (compatível com simulação futura)
        preco_h = float(df.at[data_evento, "Pred"])
        preco_ant = float(df.at[prev_day, "Pred"])
        r_new0 = ((preco_h - preco_ant) / preco_ant) * 100.0 if preco_ant != 0 else r_ref0

        # hist_std opcional: estimativa simples usando retornos percentuais de 'Pred'
        try:
            serie_rets = df["Pred"].pct_change().dropna() * 100.0
            hist_std = float(serie_rets.expanding().std().at[data_evento]) if data_evento in serie_rets.index else None
        except Exception:
            hist_std = None

        alpha = compute_alpha(r_new0, r_ref0, hist_std=hist_std, method=alpha_method, alpha_cap=alpha_cap)
        seq_applied = (alpha * ref_seq).tolist()

        cap = max_pct_per_day * 100.0
        seq_applied = [max(-cap, min(cap, float(v))) for v in seq_applied]

        # construir mapa (eventos posteriores sobrescrevem)
        for d in range(min(horizon + 1, len(seq_applied))):
            future_day = data_evento + timedelta(days=d)
            pos_k = trading_index.searchsorted(future_day)
            if pos_k >= len(trading_index):
                break
            dia_k = trading_index[pos_k]
            mapa[dia_k] = {
                "cluster": cluster,
                "k": d,
                "seq_percent": seq_applied[d],
                "motivos": motivos,
                "alpha": alpha,
                "motivo_ref": motivo_ref,
                "sim": sim
            }

        logs.append({
            "data_evento": str(data_evento.date()),
            "cluster": cluster,
            "alpha": alpha,
            "seq0": seq_applied[0] if seq_applied else None
        })

        if verbose:
            print(f"[APPLY] Evento {data_evento.date()} cluster={cluster} alpha={alpha:.3f} seq0={seq_applied[0] if seq_applied else None}")

    # aplicar mapa (atribuições escalares com .at para evitar alinhamento)
    for dia, meta in mapa.items():
        if dia not in df.index:
            continue
        df.at[dia, "Pred_Impact"] = float(df.at[dia, "Pred"]) * (1.0 + float(meta["seq_percent"]) / 100.0)
        df.at[dia, "Impact_Metadata"] = meta

    if verbose:
        print(f"[RESULT] Impacto aplicado em {len(mapa)} pregões.")

    return df, logs