import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

ARQ_EMB = os.path.join(DATA_DIR, "embeddings_frases.npy")
ARQ_META = os.path.join(DATA_DIR, "embeddings_frases_meta.csv")
ARQ_LIB = os.path.join(DATA_DIR, "impact_library.json")
OPENAI_MODEL = "text-embedding-3-small"


# ==============================================
# CARREGAR EMBEDDINGS
# ==============================================
def carregar_embeddings():
    emb = np.load(ARQ_EMB)
    meta = pd.read_csv(ARQ_META)
    return emb, meta


def gerar_embedding(texto):
    resp = client.embeddings.create(
        model=OPENAI_MODEL,
        input=[texto]
    )
    return np.array(resp.data[0].embedding).reshape(1, -1)


def detectar_cluster(motivos, emb_hist, meta):
    frase = " ".join(motivos)
    emb = gerar_embedding(frase)
    sims = cosine_similarity(emb, emb_hist)[0]
    idx = int(np.argmax(sims))
    cluster = meta.iloc[idx]["cluster"]
    motivo = meta.iloc[idx]["motivo"]
    sim = float(sims[idx])
    return cluster, motivo, sim


# ==============================================
# APLICAR SEQUÊNCIA REAL (IMPACTO REPLICADO)
# ==============================================
def aplicar_impacto_real(pred_df, motivos_por_data, horizon=5):
    """
    Para cada dia com notícia:
    1️⃣ detecta cluster → pega ref_seq
    2️⃣ escala ref_seq pelo impacto observado (alpha)
    3️⃣ aplica nos próximos N dias
    """

    emb_hist, meta = carregar_embeddings()
    lib = json.load(open(ARQ_LIB, "r", encoding="utf-8"))

    df = pred_df.copy()
    df["Pred_Cluster"] = df["Pred"].copy()

    for data_evento, motivos in motivos_por_data.items():
        try:
            data_evento = pd.to_datetime(data_evento)
        except:
            continue

        cluster, motivo_ref, sim = detectar_cluster(motivos, emb_hist, meta)

        if cluster not in lib:
            print(f"[SKIP] Cluster '{cluster}' sem ref_seq")
            continue

        ref_seq = np.array(lib[cluster]["ref_seq"])  # D0..D5
        r_ref0 = ref_seq[0]

        # Dia da previsão equivalente ao real D0
        if data_evento not in df.index:
            continue

        r_new0 = df.loc[data_evento, "Real"]

        # escala
        alpha = r_new0 / r_ref0 if r_ref0 != 0 else 1.0
        seq_adj = alpha * ref_seq

        print("\n================= IMPACTO REAL =================")
        print("Data evento:", data_evento.date())
        print("Motivos:", motivos)
        print("Cluster detectado:", cluster)
        print("Motivo referência:", motivo_ref)
        print("Similaridade:", sim)
        print("Alpha (escala):", alpha)
        print("Seq replicada:", seq_adj.tolist())

        # Aplicar no modelo – D0..D5
        for k in range(len(seq_adj)):
            dia_k = data_evento + pd.Timedelta(days=k)
            if dia_k in df.index:
                df.loc[dia_k, "Pred_Cluster"] = df.loc[data_evento, "Pred"] * (1 + seq_adj[k] / 100)

    return df
