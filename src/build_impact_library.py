"""
build_impact_library.py

Cria a "biblioteca de impacto" a partir dos arquivos gerados por frases_clusters_v12.py:
 - ../data/frases_impacto_clusters.csv
 - ../data/embeddings_frases.npy
 - ../data/embeddings_frases_meta.csv

Saídas:
 - ../data/impact_library.json   (estrutura principal: clusters -> stats, exemplos, centroid)
 - ../data/impact_library_embeddings.npy  (centroids / protótipos por cluster)
 - ../data/impact_library_meta.csv
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.metrics.pairwise import cosine_similarity

# --- Configurações / caminhos ---
INPUT_CSV = "../data/frases_impacto_clusters.csv"
EMB_FILE = "../data/embeddings_frases.npy"
EMB_META = "../data/embeddings_frases_meta.csv"

OUT_LIB_JSON = "../data/impact_library.json"
OUT_LIB_EMB = "../data/impact_library_embeddings.npy"
OUT_LIB_META = "../data/impact_library_meta.csv"

# --- Parâmetros ---
MIN_EXAMPLES_PER_CLUSTER = 3   # para registrar exemplos representativos


# --------------------
# Funções essenciais
# --------------------

def load_inputs(csv_path: str = INPUT_CSV, emb_path: str = EMB_FILE, meta_path: str = EMB_META):
    """
    Carrega dataframe, embeddings e meta (motivo->cluster) gerados pelo passo anterior.
    Retorna (df, emb, meta_df).
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} não encontrado. Rode frases_clusters_v12 primeiro.")
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"{emb_path} não encontrado. Rode frases_clusters_v12 primeiro.")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"{meta_path} não encontrado. Rode frases_clusters_v12 primeiro.")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    emb = np.load(emb_path)
    meta = pd.read_csv(meta_path, encoding="utf-8-sig")

    # Sanity: garantir mesma ordem / tamanho
    if len(df) != len(emb):
        # em algumas versões meta tinha apenas motivo/cluster; aqui usamos df order
        # tentar alinhar pelo motivo se meta estiver disponível
        if "motivo" in meta.columns and len(meta) == len(emb):
            # meta corresponde à ordem dos embeddings; reconstruir df a partir da meta
            df_meta = meta.copy()
            df_meta = df_meta.rename(columns={"motivo": "motivo_meta"})  # apenas evitar conflito
        else:
            raise RuntimeError("Comprimento de embeddings e csv diferente; verifique arquivos.")
    return df, emb, meta


def compute_cluster_centroids(df: pd.DataFrame, emb: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Calcula o centróide (média) do embedding para cada cluster.
    Retorna dicionário cluster -> centroid (numpy array).
    """
    centroids = {}
    for cluster in sorted(df["cluster"].unique()):
        idxs = df.index[df["cluster"] == cluster].tolist()
        if not idxs:
            continue
        centroid = emb[idxs].mean(axis=0)
        centroids[cluster] = centroid
    return centroids


def compute_cluster_stats(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Calcula estatísticas por cluster: frequência, impacto médio, std, exemplos.
    Retorna dicionário com stats por cluster.
    """
    stats = {}
    groups = df.groupby("cluster")
    for name, g in groups:
        impacto_medio = float(g["peso_medio"].mean())
        impacto_std = float(g["peso_medio"].std(ddof=0)) if len(g) > 1 else 0.0
        freq = int(g.shape[0])
        # Exemplo: pegar top-N motivos por frequência e ordená-los
        exemplos = (g.groupby("motivo").agg(freq=("motivo", "count"),
                                            impacto_media=("peso_medio", "mean"))
                      .reset_index()
                      .sort_values(["freq", "impacto_media"], ascending=[False, False])
                      .head(10).to_dict(orient="records"))
        stats[name] = {
            "freq": freq,
            "impacto_medio": impacto_medio,
            "impacto_std": impacto_std,
            "exemplos": exemplos
        }
    return stats


def build_impact_library(df: pd.DataFrame, emb: np.ndarray) -> Dict[str, Any]:
    """
    Constrói a biblioteca: para cada cluster guarda centroid, stats, e exemplos representativos.
    Retorna a estrutura (dict) pronta para serializar em JSON.
    """
    centroids = compute_cluster_centroids(df, emb)
    stats = compute_cluster_stats(df)

    # montar entrada por cluster
    library = {}
    for cluster, centroid in centroids.items():
        info = stats.get(cluster, {})
        # transformar centroid em lista (JSON serializable) e normalizar escala se desejar
        library[cluster] = {
            "centroid": centroid.tolist(),
            "freq": info.get("freq", 0),
            "impacto_medio": info.get("impacto_medio", 0.0),
            "impacto_std": info.get("impacto_std", 0.0),
            "exemplos": info.get("exemplos", [])
        }
    # metadados gerais
    library_meta = {
        "n_clusters": len(library),
        "clusters": list(library.keys())
    }
    return {"meta": library_meta, "library": library}


def save_library(lib: Dict[str, Any], centroids: Dict[str, np.ndarray],
                 lib_path: str = OUT_LIB_JSON, emb_out: str = OUT_LIB_EMB, meta_out: str = OUT_LIB_META):
    """
    Salva a biblioteca em JSON e os centroids em .npy para uso rápido.
    Também salva um CSV meta com cluster, freq, impacto_medio.
    """
    os.makedirs(os.path.dirname(lib_path), exist_ok=True)
    with open(lib_path, "w", encoding="utf-8") as f:
        json.dump(lib, f, indent=2, ensure_ascii=False)

    # salvar centroids como matriz (clusters ordenados)
    clusters = list(lib["library"].keys())
    cent_mat = np.vstack([np.array(lib["library"][c]["centroid"], dtype=np.float32) for c in clusters])
    np.save(emb_out, cent_mat)

    # salvar meta csv
    rows = []
    for c in clusters:
        rows.append({
            "cluster": c,
            "freq": lib["library"][c]["freq"],
            "impacto_medio": lib["library"][c]["impacto_medio"],
            "impacto_std": lib["library"][c]["impacto_std"]
        })
    pd.DataFrame(rows).to_csv(meta_out, index=False, encoding="utf-8-sig")

    print(f"Impact library salva em: {lib_path}")
    print(f"Centroids salvos em: {emb_out}")
    print(f"Meta salvo em: {meta_out}")


def build_and_save(csv_path: str = INPUT_CSV, emb_path: str = EMB_FILE):
    """
    Fluxo principal curto para chamada externa.
    """
    df, emb, meta = load_inputs(csv_path, emb_path, EMB_META)
    lib_struct = build_impact_library(df, emb)
    save_library(lib_struct, centroids=None)  # centroids implícitos dentro do lib_struct


# --------------------
# Execução direta
# --------------------
if __name__ == "__main__":
    build_and_save()
