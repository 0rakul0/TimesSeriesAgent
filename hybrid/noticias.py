"""
noticias.py
- carregar_base_frases(): carrega CSV de frases + embeddings (com tqdm + mmap)
- buscar_similares(): gera embedding da frase e retorna top-k similares
- extrair_motivos_ultimos_dias(): carrega motivos de JSONs na pasta de notícias
Config: espera CSV_FRASES e EMB_PATH no mesmo nível do projeto (../data/...)
"""

import os
import json
import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
_openai_client = OpenAI()

# caminhos padrão (ajuste se necessário)
CSV_FRASES = "../data/frases_impacto_clusters.csv"
EMB_PATH = "../models/embeddings_frases.npy"

# ------------------------
# Carregamento/geração de embeddings
# ------------------------
def carregar_base_frases(csv_path: str = CSV_FRASES, emb_path: str = EMB_PATH, openai_client=None):
    """
    Carrega CSV de frases (colunas esperadas: motivo, peso_medio, data) e embeddings.
    Se emb_path existir -> carrega via np.load(mmap_mode='r') e copia para RAM com tqdm.
    Se NÃO existir -> gera embeddings via OpenAI (text-embedding-3-large) e salva.
    Retorna: meta_df (DataFrame), emb_matrix (np.ndarray)
    """
    if openai_client is None:
        openai_client = _openai_client

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV de frases não encontrado: {csv_path}")

    meta_df = pd.read_csv(csv_path, parse_dates=["data"])
    meta_df["motivo"] = meta_df["motivo"].astype(str)

    if os.path.exists(emb_path):
        print("[INFO] Arquivo de embeddings encontrado. Carregando com mmap + tqdm...")
        emb_mmap = np.load(emb_path, mmap_mode="r")
        emb_matrix = np.empty((len(emb_mmap), emb_mmap.shape[1]), dtype=emb_mmap.dtype)
        for i in tqdm(range(len(emb_mmap)), desc="Carregando embeddings", ncols=80):
            emb_matrix[i] = emb_mmap[i]
        print(f"[INFO] Embeddings carregados: {emb_matrix.shape}")
        return meta_df, emb_matrix

    # gerar embeddings (uma única vez)
    print("[INFO] Embeddings não encontrados. Gerando via API (uma vez)...")
    embeddings = []
    motivos = meta_df["motivo"].tolist()

    for motivo in tqdm(motivos, desc="Gerando embeddings", ncols=80):
        try:
            resp = openai_client.embeddings.create(model="text-embedding-3-large", input=motivo)
            emb = resp.data[0].embedding
        except Exception as e:
            print("[WARN] Falha ao gerar embedding, usando zero-vector para motivo:", motivo[:80], "...", e)
            emb = [0.0] * 1536
        embeddings.append(emb)

    emb_matrix = np.array(embeddings, dtype=np.float32)
    try:
        np.save(emb_path, emb_matrix)
        print(f"[INFO] Embeddings salvos em {emb_path}")
    except Exception as e:
        print("[WARN] Falha ao salvar embeddings:", e)

    return meta_df, emb_matrix


# ------------------------
# Buscar similares
# ------------------------
def buscar_similares(motivo: str, meta_df: pd.DataFrame, emb_matrix: np.ndarray,
                     openai_client=None, top_k: int = 8, sim_threshold: float = 0.55):
    """
    Retorna lista de tuplas (motivo_base, peso_medio, similarity) para os top_k similares
    Usa embeddings OpenAI se disponível (gera embedding da frase de entrada).
    """
    if openai_client is None:
        openai_client = _openai_client

    motivos_base = meta_df["motivo"].astype(str).tolist()

    try:
        resp = openai_client.embeddings.create(model="text-embedding-3-small", input=motivo)
        emb = np.array(resp.data[0].embedding).reshape(1, -1)
        sims = cosine_similarity(emb, emb_matrix)[0]
        idxs = np.argsort(sims)[::-1][:top_k]

        resultados = []
        for i in idxs:
            s = float(sims[i])
            if s >= sim_threshold:
                resultados.append((motivos_base[i], float(meta_df.iloc[i]["peso_medio"]), s))
        return resultados

    except Exception:
        # fallback textual fuzzy (simples)
        from difflib import get_close_matches, SequenceMatcher
        matches = get_close_matches(motivo, motivos_base, n=top_k, cutoff=0.45)
        resultados = []
        for m in matches:
            idx = motivos_base.index(m)
            s = SequenceMatcher(None, motivo, m).ratio()
            resultados.append((m, float(meta_df.iloc[idx]["peso_medio"]), float(s)))
        return resultados


# ------------------------
# Extrair motivos das notícias (JSONs)
# ------------------------
def extrair_motivos_ultimos_dias(pasta_json: str, janela_dias: int = 30, ref_date=None):
    """
    Lê arquivos JSON na pasta 'pasta_json' com padrão evento_*.json
    Cada JSON esperado ter chaves: data (YYYY-MM-DD ou ISO), motivos_identificados (lista)
    Retorna: dict {data(normalized): [motivo1, motivo2, ...], ...}
    """
    arquivos = glob.glob(os.path.join(pasta_json, "evento_*.json"))
    if ref_date is None:
        hoje = pd.Timestamp.today().normalize()
    else:
        hoje = pd.to_datetime(ref_date).normalize()

    inicio = hoje - pd.Timedelta(days=janela_dias)
    noticias = {}

    for path in arquivos:
        try:
            with open(path, "r", encoding="utf-8") as f:
                j = json.load(f)
            data_str = j.get("data") or os.path.basename(path).split("_")[-1].replace(".json", "")
            data = pd.to_datetime(data_str).normalize()
            if not (inicio < data <= hoje):
                continue
            motivos = j.get("motivos_identificados") or []
            motivos = [str(m).strip() for m in motivos if str(m).strip()]
            if motivos:
                noticias.setdefault(data, []).extend(motivos)
        except Exception:
            continue

    return noticias

