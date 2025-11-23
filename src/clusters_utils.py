import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# ============================================================
# BASE DIRS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_EVENTOS = os.path.join(BASE_DIR, "output_noticias")

ARQ_CLUSTER = os.path.join(DATA_DIR, "frases_impacto_clusters.csv")
ARQ_EMB = os.path.join(DATA_DIR, "embeddings_frases.npy")
ARQ_META = os.path.join(DATA_DIR, "embeddings_frases_meta.csv")
ARQ_LIB = os.path.join(DATA_DIR, "impact_library.json")

OPENAI_MODEL = "text-embedding-3-small"

# Quantos dias replicar?
HORIZON = 5


# ============================================================
# 1) Embeddings e clusters
# ============================================================

def carregar_embeddings_historicos():
    if not os.path.exists(ARQ_EMB):
        raise FileNotFoundError(f"❌ Não encontrei embeddings: {ARQ_EMB}")

    if not os.path.exists(ARQ_META):
        raise FileNotFoundError(f"❌ Não encontrei meta frases: {ARQ_META}")

    emb = np.load(ARQ_EMB)
    meta = pd.read_csv(ARQ_META)

    df_clusters = pd.read_csv(ARQ_CLUSTER)[["motivo", "cluster"]].drop_duplicates()
    meta = meta.merge(df_clusters, on="motivo", how="left")
    meta["cluster"].fillna("outros", inplace=True)

    return emb, meta


def gerar_embedding(texto: str):
    resp = client.embeddings.create(
        model=OPENAI_MODEL,
        input=[texto]
    )
    return np.array(resp.data[0].embedding).reshape(1, -1)


def detectar_cluster_por_embeddings(motivos: list, emb_hist, meta_hist):
    if not motivos:
        return "outros", "", 0.0

    frase = " ".join(motivos)
    emb_frase = gerar_embedding(frase)
    sims = cosine_similarity(emb_frase, emb_hist)[0]
    idx = int(np.argmax(sims))

    return (
        meta_hist.iloc[idx]["cluster"],
        meta_hist.iloc[idx]["motivo"],
        float(sims[idx])
    )


# ============================================================
# 2) Biblioteca de impacto histórico
# ============================================================

def extrair_sequencia(df, date, ativo, horizon=HORIZON):
    if ativo == "PETR4":
        col = "Close_PETR4.SA"
    else:
        col = "Close_BZ=F"

    if date not in df.index:
        return None

    seq = []

    # D0
    prev = date - pd.Timedelta(days=1)
    while prev not in df.index:
        prev -= pd.Timedelta(days=1)
        if prev < df.index.min():
            return None

    r0 = (df.loc[date, col] - df.loc[prev, col]) / df.loc[prev, col]
    seq.append(float(r0))

    # D1..DH
    current = date
    for k in range(1, horizon + 1):
        next_day = current + pd.Timedelta(days=1)
        while next_day not in df.index:
            next_day += pd.Timedelta(days=1)
            if next_day > df.index.max():
                return None

        rk = (df.loc[next_day, col] - df.loc[current, col]) / df.loc[current, col]
        seq.append(float(rk))

        current = next_day

    return seq


def construir_biblioteca_impacto():
    """Ler eventos e construir biblioteca de sequências reais por cluster."""
    df_prices = pd.read_csv(os.path.join(DATA_DIR, "dados_combinados.csv"),
                            index_col=0, parse_dates=True)
    df_prices.index = pd.to_datetime(df_prices.index).normalize()

    eventos = sorted([f for f in os.listdir(OUT_EVENTOS) if f.startswith("evento_")])

    biblioteca = {}

    for fname in eventos:
        j = json.load(open(os.path.join(OUT_EVENTOS, fname), "r", encoding="utf-8"))

        if "cluster" not in j:
            continue

        cluster = j["cluster"]
        ativo = j["ativo"]
        data = pd.to_datetime(j["data"])

        seq = extrair_sequencia(df_prices, data, ativo)
        if not seq:
            continue

        biblioteca.setdefault(cluster, []).append(seq)

    # agregação por cluster
    agg = {}
    for c, seqs in biblioteca.items():
        arr = np.array(seqs)
        ref = np.median(arr, axis=0).tolist()
        agg[c] = {
            "ref_seq": ref,
            "n": len(seqs)
        }

    json.dump(agg, open(ARQ_LIB, "w", encoding="utf-8"),
              indent=4, ensure_ascii=False)

    return agg


# ============================================================
# 3) Aplicação da sequência (Z-score scaling)
# ============================================================

def aplicar_impacto_zscore(r_new0, ref_seq, r_ref0, alpha_cap=(0.25, 4.0)):
    if r_ref0 == 0:
        alpha = 1.0
    else:
        alpha = r_new0 / r_ref0

    alpha = max(alpha_cap[0], min(alpha_cap[1], alpha))
    seq = (alpha * np.array(ref_seq)).tolist()

    return seq, alpha


def obter_impacto_real(motivos, r_new0, ativo):
    """Retorna: cluster, seq_aplicada (retornos), seq_ref, alpha."""
    emb_hist, meta_hist = carregar_embeddings_historicos()

    cluster, motivo_ref, sim = detectar_cluster_por_embeddings(
        motivos, emb_hist, meta_hist
    )

    if not os.path.exists(ARQ_LIB):
        raise FileNotFoundError("❌ impact_library.json não encontrado. Rode construir_biblioteca_impacto().")

    lib = json.load(open(ARQ_LIB, "r", encoding="utf-8"))

    if cluster not in lib:
        return {
            "cluster": cluster,
            "motivo_referência": motivo_ref,
            "similaridade": sim,
            "sequencia_replicada": [],
            "sequencia_referencia": [],
            "alpha": 1.0
        }

    ref_seq = lib[cluster]["ref_seq"]
    r_ref0 = ref_seq[0]

    seq_aplicada, alpha = aplicar_impacto_zscore(r_new0, ref_seq, r_ref0)

    return {
        "cluster": cluster,
        "motivo_referência": motivo_ref,
        "similaridade": sim,
        "sequencia_replicada": seq_aplicada,
        "sequencia_referencia": ref_seq,
        "alpha": alpha
    }
