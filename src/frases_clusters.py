import os
import json
import numpy as np
import pandas as pd
from glob import glob
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# =========================================
# CONFIG
# =========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVENTS_DIR = os.path.join(BASE_DIR, "output_noticias")
OUT_CSV = os.path.join(BASE_DIR, "data", "cluster_motivos.csv")
EMBED_PATH = os.path.join(BASE_DIR, "data", "embeddings_frases.npy")


# =========================================
# Gerar embedding da frase
# =========================================
def embed_text(texto):
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=texto
    )
    return resp.data[0].embedding


# =========================================
# Carregar todos os eventos e motivos
# =========================================
def carregar_eventos():
    motivos = []
    seqs = []
    sentimentos = []

    arquivos = sorted(glob(os.path.join(EVENTS_DIR, "evento_*.json")))

    for path in arquivos:
        with open(path, "r", encoding="utf-8") as f:
            j = json.load(f)

        sentimento = j.get("sentimento_do_mercado", "neutro")
        seq_dict = j.get("seq", {})

        # cada seq p/ cada motivo
        for ativo, seq in seq_dict.items():
            for frase in j.get("motivos_identificados", []):
                motivos.append(frase)
                seqs.append(seq)
                sentimentos.append(sentimento)

    return motivos, seqs, sentimentos


# =========================================
# Geração de embeddings com alinhamento
# =========================================
def gerar_embeddings(motivos, save_path):
    embeds = []

    print(f"📌 Gerando embeddings para {len(motivos)} frases...")

    for frase in motivos:
        emb = embed_text(frase)
        embeds.append(emb)

    embeds = np.array(embeds)
    np.save(save_path, embeds)
    print("✔ Embeddings salvos em:", save_path)

    return embeds


# =========================================
# Clusterizar com SIMILARIDADE COSENO
# =========================================
def clusterizar_motivos_cosine(embeds, n_clusters=50):
    print("📌 Clusterizando usando similaridade do coseno...")

    # Agglomerative com métrica COSENO
    cluster = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='cosine',
        linkage='average'
    )

    labels = cluster.fit_predict(embeds)
    return labels


# =========================================
# Padronizar sequências
# =========================================
def padronizar_seqs(seqs):
    max_len = max(len(s) for s in seqs)
    pad = []
    for s in seqs:
        diff = max_len - len(s)
        pad.append(s + [None] * diff)
    return np.array(pad, dtype=object), max_len


# =========================================
# Média por cluster + sentimento probabilístico
# =========================================
def gerar_media_por_cluster(motivos, sentimentos, seqs, labels):

    df = pd.DataFrame({
        "motivo": motivos,
        "sentimento": sentimentos,
        "seq": seqs,
        "cluster": labels
    })

    linhas = []

    for cl in sorted(df["cluster"].unique()):
        grupo = df[df["cluster"] == cl]

        seq_pad, max_len = padronizar_seqs(grupo["seq"].tolist())

        # média dos seqs
        seq_media = []
        for col in range(max_len):
            col_vals = [row[col] for row in seq_pad if row[col] is not None]
            seq_media.append(np.mean(col_vals) if col_vals else None)

        # distribuição do sentimento
        total = len(grupo)
        p_pos = sum(grupo["sentimento"] == "positivo") / total
        p_neg = sum(grupo["sentimento"] == "negativo") / total
        p_neu = sum(grupo["sentimento"] == "neutro") / total

        linhas.append({
            "cluster": cl,
            "frase_exemplo": grupo.iloc[0]["motivo"],
            "n_eventos": len(grupo),
            "p_positivo": p_pos,
            "p_negativo": p_neg,
            "p_neutro": p_neu,
            **{f"seq_d{i}": seq_media[i] if i < len(seq_media) else None for i in range(max_len)}
        })

    return pd.DataFrame(linhas)


# =========================================
# MAIN
# =========================================
def gerar_cluster_motivos():

    print("📌 Carregando motivos e sequências...")
    motivos, seqs, sentimentos = carregar_eventos()

    print("📌 Gerando embeddings alinhados...")
    embeds = gerar_embeddings(motivos, EMBED_PATH)

    print("📌 Aplicando clusterização...")
    labels = clusterizar_motivos_cosine(embeds)

    print("📌 Calculando médias...")
    df_final = gerar_media_por_cluster(motivos, sentimentos, seqs, labels)

    print("📌 Salvando CSV final...")
    df_final.to_csv(OUT_CSV, index=False)

    print("✔ Arquivo salvo em:", OUT_CSV)


if __name__ == "__main__":
    gerar_cluster_motivos()
