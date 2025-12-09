# frases_clusters.py — CLUSTERIZAÇÃO HÍBRIDA COM FRASE CANÔNICA (VERSÃO FINAL CACHEADA)

import os
import json
import numpy as np
import pandas as pd
from glob import glob
from sklearn.cluster import AgglomerativeClustering
from dotenv import load_dotenv

# ============================
# IMPORTANTE: EmbeddingManager com CACHE
# ============================
from utils.embedding_manager import EmbeddingManager

load_dotenv()
emb_mgr = EmbeddingManager()

# ============================
# CONFIG
# ============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVENTS_DIR = os.path.join(BASE_DIR, "output_noticias")
OUT_DIR = os.path.join(BASE_DIR, "data")
OUT_CSV = os.path.join(OUT_DIR, "cluster_motivos.csv")

N_CLUSTERS_POR_ATIVO = 50
MIN_MOTIVOS_POR_ATIVO = 100
TOP_CENTER = 50
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

# ============================
# EMBEDDING (usando cache!)
# ============================
def embed_text(texto: str):
    """
    Usa EmbeddingManager → garante:
      - cache local persistente
      - sem chamadas repetidas à API
      - consistência com MVP e modelo híbrido
    Retorna vetor shape (1536,)
    """
    try:
        emb = emb_mgr.embed(texto)
        return emb.reshape(-1)  # torna 1D
    except Exception as e:
        print("⚠ Erro ao gerar embedding:", e)
        return np.zeros((1536,), dtype=float)

# ============================
# Oversample de frases (permanece igual)
# ============================
def gerar_variacoes_via_openai(frase, n=5):
    from openai import OpenAI
    client = OpenAI()

    prompt = f"""
Gere {n} variações naturais da frase abaixo, mantendo o mesmo significado.
Retorne APENAS uma lista JSON de strings.

Frase original:
"{frase}"
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=200,
        )
        txt = resp.choices[0].message.content

        import re, ast
        m = re.search(r"\[.*\]", txt, flags=re.S)
        if m:
            return ast.literal_eval(m.group(0))

    except Exception as e:
        print("⚠ Erro variações:", e)

    return [frase]

# ============================
# FRASE CANÔNICA DO CLUSTER
# ============================
def gerar_frase_canonica_cluster(frases_cluster, ativo):
    frases_cluster = [f.strip() for f in frases_cluster if isinstance(f, str) and f.strip()]
    if not frases_cluster:
        return "Evento relevante no mercado."

    # 1) embeddings (cacheado)
    embeds = np.vstack([embed_text(f) for f in frases_cluster])
    centroide = embeds.mean(axis=0)

    sims = []
    for frase, emb in zip(frases_cluster, embeds):
        sim = float(emb @ centroide / (np.linalg.norm(emb) * np.linalg.norm(centroide) + 1e-12))
        sims.append((sim, frase))

    frases_centrais = [s[1] for s in sorted(sims, key=lambda x: x[0], reverse=True)[:TOP_CENTER]]
    frases_txt = "\n".join(f"- {f}" for f in frases_centrais)

    # 2) gerar resumo canônico via LLM
    from openai import OpenAI
    client = OpenAI()

    prompt = f"""
Você é um analista financeiro em 2025.

As frases abaixo representam o núcleo semântico deste cluster.
Resuma o conceito central em UMA frase curta, objetiva e atemporal.

NÃO invente novos temas.
Use somente o que está implícito nas frases.
Mantenha o foco econômico e de mercado.

Frases principais:
{frases_txt}

Retorne APENAS a frase final.
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4.1",
            temperature=0.2,
            max_tokens=80,
            messages=[{"role": "user", "content": prompt}]
        )
        frase_llm = resp.choices[0].message.content.strip()
    except:
        frase_llm = frases_centrais[0]

    # 3) validação semântica
    emb_canon = embed_text(frase_llm)
    sim_canon = float(
        emb_canon @ centroide / (np.linalg.norm(emb_canon) * np.linalg.norm(centroide) + 1e-12)
    )

    if sim_canon < 0.70:
        return frases_centrais[0]

    return frase_llm

# ============================
# Carregar eventos JSON
# ============================
def carregar_eventos():
    arquivos = sorted(glob(os.path.join(EVENTS_DIR, "evento_*.json")))

    motivos, seqs, sentimentos, ativos = [], [], [], []

    for p in arquivos:
        try:
            j = json.load(open(p, "r", encoding="utf-8"))
        except:
            continue

        ativo = str(j.get("ativo", "GENÉRICO")).upper()
        seq = j.get("seq", {}).get(ativo, [])

        frases = j.get("motivos_identificados", []) or []
        sent = j.get("sentimento_do_mercado", "neutro")

        for f in frases:
            f = f.strip()
            if f:
                motivos.append(f)
                seqs.append(seq)
                sentimentos.append(sent)
                ativos.append(ativo)

    return motivos, seqs, sentimentos, ativos

# ============================
# Oversample por ativo
# ============================
def oversample_motivos_por_ativo(motivos, seqs, sentimentos, ativos):
    df = pd.DataFrame({"motivo": motivos, "seq": seqs, "sentimento": sentimentos, "ativo": ativos})

    out_motivos, out_seqs, out_sent, out_ativos = [], [], [], []

    for ativo, g in df.groupby("ativo"):
        frases = g["motivo"].unique().tolist()

        # base
        out_motivos.extend(frases)
        out_seqs.extend([g[g["motivo"] == f]["seq"].iloc[0] for f in frases])
        out_sent.extend([g[g["motivo"] == f]["sentimento"].iloc[0] for f in frases])
        out_ativos.extend([ativo] * len(frases))

        # oversample (gera variações reais)
        if len(frases) < MIN_MOTIVOS_POR_ATIVO:
            need = MIN_MOTIVOS_POR_ATIVO - len(frases)
            per_frase = max(1, int(np.ceil(need / len(frases))))

            print(f"[OVERSAMPLE] ativo={ativo} → {need} novas frases")

            for f in frases:
                novas = gerar_variacoes_via_openai(f, n=per_frase)
                for v in novas:
                    out_motivos.append(v)
                    out_seqs.append(g[g["motivo"] == f]["seq"].iloc[0])
                    out_sent.append(g[g["motivo"] == f]["sentimento"].iloc[0])
                    out_ativos.append(ativo)

    return out_motivos, out_seqs, out_sent, out_ativos

# ============================
# Clusterização final
# ============================
def clusterizar_por_ativo(motivos, seqs, sentimentos, ativos):
    df = pd.DataFrame({"motivo": motivos, "seq": seqs, "sentimento": sentimentos, "ativo": ativos})

    resultados = []

    for ativo, g in df.groupby("ativo"):
        frases = g["motivo"].tolist()
        if not frases:
            continue

        print(f"[CLUSTER] ativo={ativo} | motivos={len(frases)}")

        # Embeddings consistentes e cacheados
        embeds = np.vstack([embed_text(f) for f in frases])

        # numero de clusters
        k = min(N_CLUSTERS_POR_ATIVO, max(1, len(frases) // 2))

        cluster = AgglomerativeClustering(
            n_clusters=k,
            metric="cosine",
            linkage="average"
        )
        labels = cluster.fit_predict(embeds)

        g["cluster"] = labels

        # gerar clusters finais
        for c in sorted(g["cluster"].unique()):
            grp = g[g["cluster"] == c]
            frases_cluster = grp["motivo"].tolist()

            frase_canon = gerar_frase_canonica_cluster(frases_cluster, ativo)

            seqs_raw = grp["seq"].tolist()
            max_len = max((len(s) if isinstance(s, list) else 0) for s in seqs_raw)

            seq_avg = []
            for i in range(max_len):
                vals = [s[i] for s in seqs_raw if isinstance(s, list) and len(s) > i and s[i] is not None]
                seq_avg.append(np.mean(vals) if vals else None)

            resultados.append({
                "cluster": int(c),
                "frase_exemplo": frase_canon,
                "ativo_cluster": ativo,
                "n_eventos": len(grp),
                "frases_originais": frases_cluster,
                **{f"seq_d{i}": seq_avg[i] for i in range(len(seq_avg))}
            })

        # salvar CSV por ativo
        df_ativo = pd.DataFrame([r for r in resultados if r["ativo_cluster"] == ativo])
        df_ativo.to_csv(os.path.join(OUT_DIR, f"cluster_{ativo.lower()}.csv"), index=False)
        print(" → salvo cluster_", ativo.lower())

    # salvar CSV combinado
    pd.DataFrame(resultados).to_csv(OUT_CSV, index=False)
    print("Clusters combinados salvos em:", OUT_CSV)

    return pd.DataFrame(resultados)


# ============================
# MAIN
# ============================
def gerar_cluster_motivos():
    print("Carregando eventos…")
    motivos, seqs, sentimentos, ativos = carregar_eventos()
    print(f"{len(motivos)} motivos carregados")

    print("Oversample…")
    motivos2, seqs2, sent2, ativos2 = oversample_motivos_por_ativo(
        motivos, seqs, sentimentos, ativos
    )

    print("Clusterizando…")
    df_final = clusterizar_por_ativo(motivos2, seqs2, sent2, ativos2)

    print("Finalizado.")
    return df_final


if __name__ == "__main__":
    gerar_cluster_motivos()
