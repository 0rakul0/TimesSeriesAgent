"""
frases_clusters_v12.py
--------------------------------------------------
Gera clusters semânticos dos motivos das notícias,
com base no impacto_real (% real) produzido pelo agente_noticia.

Agora usa impacto_real em vez de peso_de_correcao (-1..1).
Projeta frases com PCA e UMAP, gera gráficos e salva embeddings.
"""

import os
import json
from glob import glob
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import umap
import plotly.graph_objects as go

# =====================
# CONFIGURAÇÕES
# =====================
PASTA_EVENTOS = "../output_noticias"
ARQUIVO_SAIDA = "../data/frases_impacto_clusters.csv"
ARQUIVO_EMBEDDINGS = "../data/embeddings_frases.npy"
ARQUIVO_EMBEDDINGS_META = "../data/embeddings_frases_meta.csv"
HTML_3D = "../img/graficos_motivos/frases_impacto_clusters.html"
HTML_UMAP_INTERATIVO = "../img/graficos_motivos/frases_impacto_clusters_umap_interativo.html"
HTML_HEATMAP = "../img/graficos_motivos/heatmap_clusters.html"
HTML_RADAR = "../img/graficos_motivos/radar_clusters.html"
RESUMO_CSV = "../data/resumo_clusters.csv"

OPENAI_MODEL = "text-embedding-3-small"
RECLASS_THRESHOLD = 0.65
RECLASS_AMBOS = 0.60

load_dotenv()
client = OpenAI()

def carregar_frases_com_peso(pasta):
    arquivos = sorted(glob(os.path.join(pasta, "evento_*.json")))
    frases = []

    for path in arquivos:
        try:
            with open(path, "r", encoding="utf-8") as f:
                j = json.load(f)
        except:
            continue

        data = j.get("data")
        ativo = str(j.get("ativo") or "").upper().strip()
        motivos = j.get("motivos_identificados") or []

        # impacto individual por ativo
        if ativo in ["PETR4", "BRENT", "PRIO3"]:
            peso = j.get("impacto_d0")
            if peso is None:
                continue
            for m in motivos:
                frases.append({
                    "data": data,
                    "ativo": ativo,
                    "motivo": m.strip(),
                    "peso": float(peso)
                })

        elif ativo == "AMBOS":
            for nome_ativo, chave in [
                ("PETR4", "impacto_d0_PETR4"),
                ("BRENT", "impacto_d0_BRENT"),
                ("PRIO3", "impacto_d0_PRIO3")
            ]:
                peso = j.get(chave)
                if peso is not None:
                    for m in motivos:
                        frases.append({
                            "data": data,
                            "ativo": nome_ativo,
                            "motivo": m.strip(),
                            "peso": float(peso)
                        })

    df = pd.DataFrame(frases)
    df = df.dropna(subset=["motivo"]).reset_index(drop=True)

    if df.empty:
        raise RuntimeError("Nenhuma frase carregada.")

    return df


def gerar_ou_carregar_embeddings(frases):
    if os.path.exists(ARQUIVO_EMBEDDINGS) and os.path.exists(ARQUIVO_EMBEDDINGS_META):
        meta = pd.read_csv(ARQUIVO_EMBEDDINGS_META)
        if len(meta) == len(frases):
            return np.load(ARQUIVO_EMBEDDINGS)

    embs = []
    for i in tqdm(range(0, len(frases), 64)):
        batch = frases[i:i+64]
        resp = client.embeddings.create(model=OPENAI_MODEL, input=batch)
        for item in resp.data:
            embs.append(item.embedding)

    emb = np.array(embs, dtype=np.float32)
    np.save(ARQUIVO_EMBEDDINGS, emb)

    return emb


def reclassificar_outros_semanticamente(df, emb_matrix):
    clusters_main = [
        "posi_petr4", "neg_petr4",
        "posi_brent", "neg_brent",
        "posi_prio3", "neg_prio3"
    ]

    # calcular embeddings médios
    medias = {}
    for c in clusters_main:
        idxs = df.index[df["cluster"] == c].tolist()
        if idxs:
            medias[c] = emb_matrix[idxs].mean(axis=0)

    idx_outros = df.index[df["cluster"] == "outros"].tolist()

    for i in idx_outros:
        vec = emb_matrix[i].reshape(1, -1)
        sims = {
            c: cosine_similarity(vec, medias[c].reshape(1, -1))[0][0]
            for c in medias
        }
        c1 = max(sims, key=sims.get)
        df.at[i, "cluster"] = c1

    return df


def definir_cluster(row):
    p, a = row["peso"], row["ativo"]
    if np.isnan(p):
        return "outros"
    if a == "PETR4":
        return "posi_petr4" if p > 0 else "neg_petr4"
    if a == "BRENT":
        return "posi_brent" if p > 0 else "neg_brent"
    if a == "PRIO3":
        return "posi_prio3" if p > 0 else "neg_prio3"
    return "outros"


def main():
    df = carregar_frases_com_peso(PASTA_EVENTOS)

    # cluster base pelo impacto
    df["cluster"] = df.apply(definir_cluster, axis=1)

    # agregação por motivo
    agrupado = (
        df.groupby(["motivo", "cluster", "data", "ativo"])
          .agg(peso_medio=("peso", "mean"), freq=("motivo", "count"))
          .reset_index()
    )

    # gerar embeddings
    frases = agrupado["motivo"].tolist()
    emb = gerar_ou_carregar_embeddings(frases)

    # reclassificar semanticamente
    agrupado = reclassificar_outros_semanticamente(agrupado, emb)

    # salvar tabelas principais
    agrupado.to_csv(ARQUIVO_SAIDA, index=False, encoding="utf-8-sig")

    meta = agrupado[["motivo", "cluster"]]
    meta.to_csv(ARQUIVO_EMBEDDINGS_META, index=False, encoding="utf-8-sig")

    resumo = agrupado.groupby(["cluster"]).agg(
        num_frases=("motivo", "count"),
        impacto_medio=("peso_medio", "mean")
    ).reset_index()

    resumo.to_csv(RESUMO_CSV, index=False, encoding="utf-8-sig")

    print("Processo concluído!")


if __name__ == "__main__":
    main()
