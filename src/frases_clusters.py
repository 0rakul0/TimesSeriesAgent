import json
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.cluster import AgglomerativeClustering

from utils.embedding_manager import EmbeddingManager
from utils.project_paths import DATA_DIR, OUTPUT_NOTICIAS_DIR

load_dotenv()

PROMPT_VERSION = "news-cluster-v1"
N_CLUSTERS_POR_ATIVO = 50
MIN_MOTIVOS_POR_ATIVO = 100
TOP_CENTER = 50
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
emb_mgr = EmbeddingManager()
_openai_client = None


def get_openai_client():
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


def embed_text(texto: str):
    try:
        return np.asarray(emb_mgr.embed(texto)).reshape(-1)
    except Exception as exc:
        print("[WARN] Erro ao gerar embedding:", exc)
        return np.zeros((1536,), dtype=float)


def gerar_variacoes_via_openai(frase, n=5):
    prompt = f"""
Gere {n} variacoes naturais da frase abaixo, mantendo o mesmo significado.
Retorne apenas uma lista JSON de strings.

Frase original:
"{frase}"
"""

    try:
        resp = get_openai_client().chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=200,
        )
        txt = resp.choices[0].message.content or ""

        import ast
        import re

        match = re.search(r"\[.*\]", txt, flags=re.S)
        if match:
            return ast.literal_eval(match.group(0))
    except Exception as exc:
        print("[WARN] Erro variacoes:", exc)

    return [frase]


def gerar_frase_representativa_cluster(frases_cluster, ativo):
    frases_cluster = [frase.strip() for frase in frases_cluster if isinstance(frase, str) and frase.strip()]
    if not frases_cluster:
        return "Evento relevante no mercado."

    embeds = np.vstack([embed_text(frase) for frase in frases_cluster])
    centroide = embeds.mean(axis=0)

    sims = []
    for frase, emb in zip(frases_cluster, embeds):
        sim = float(emb @ centroide / (np.linalg.norm(emb) * np.linalg.norm(centroide) + 1e-12))
        sims.append((sim, frase))

    frases_centrais = [item[1] for item in sorted(sims, key=lambda x: x[0], reverse=True)[:TOP_CENTER]]
    frases_txt = "\n".join(f"- {frase}" for frase in frases_centrais)

    prompt = f"""
Voce e um analista financeiro em 2025.
As frases abaixo representam o nucleo semantico de um cluster do ativo {ativo}.
Resuma o conceito central em uma frase curta, objetiva e atemporal.

Nao invente novos temas.
Use somente o que esta implicito nas frases.

Frases principais:
{frases_txt}

Retorne apenas a frase final.
"""

    try:
        resp = get_openai_client().chat.completions.create(
            model="gpt-4.1",
            temperature=0.2,
            max_tokens=80,
            messages=[{"role": "user", "content": prompt}],
        )
        frase_llm = (resp.choices[0].message.content or "").strip()
    except Exception:
        frase_llm = frases_centrais[0]

    emb_canon = embed_text(frase_llm)
    sim_canon = float(emb_canon @ centroide / (np.linalg.norm(emb_canon) * np.linalg.norm(centroide) + 1e-12))
    return frase_llm if sim_canon >= 0.70 else frases_centrais[0]


def carregar_eventos():
    motivos, seqs, sentimentos, ativos, datas = [], [], [], [], []

    for arquivo in sorted(OUTPUT_NOTICIAS_DIR.glob("evento_*.json")):
        try:
            with arquivo.open("r", encoding="utf-8") as handle:
                registro = json.load(handle)
        except Exception:
            continue

        ativo = str(registro.get("ativo", "GENERICO")).upper()
        seq = (registro.get("seq", {}) or {}).get(ativo, [])
        frases = registro.get("motivos_identificados", []) or []
        sentimento = registro.get("sentimento_do_mercado", "neutro")
        data = pd.to_datetime(registro.get("data")).normalize()

        for frase in frases:
            frase = frase.strip()
            if frase:
                motivos.append(frase)
                seqs.append(seq)
                sentimentos.append(sentimento)
                ativos.append(ativo)
                datas.append(data)

    return motivos, seqs, sentimentos, ativos, datas


def oversample_motivos_por_ativo(motivos, seqs, sentimentos, ativos, datas):
    df = pd.DataFrame(
        {
            "motivo": motivos,
            "seq": seqs,
            "sentimento": sentimentos,
            "ativo": ativos,
            "data_evento": datas,
        }
    )

    out_motivos, out_seqs, out_sent, out_ativos, out_datas = [], [], [], [], []

    for ativo, grupo in df.groupby("ativo"):
        frases = grupo["motivo"].unique().tolist()

        for frase in frases:
            linha = grupo[grupo["motivo"] == frase].iloc[0]
            out_motivos.append(frase)
            out_seqs.append(linha["seq"])
            out_sent.append(linha["sentimento"])
            out_ativos.append(ativo)
            out_datas.append(linha["data_evento"])

        if len(frases) < MIN_MOTIVOS_POR_ATIVO:
            need = MIN_MOTIVOS_POR_ATIVO - len(frases)
            per_frase = max(1, int(np.ceil(need / max(len(frases), 1))))

            print(f"[OVERSAMPLE] ativo={ativo} -> {need} novas frases")
            for frase in frases:
                linha = grupo[grupo["motivo"] == frase].iloc[0]
                for variacao in gerar_variacoes_via_openai(frase, n=per_frase):
                    out_motivos.append(variacao)
                    out_seqs.append(linha["seq"])
                    out_sent.append(linha["sentimento"])
                    out_ativos.append(ativo)
                    out_datas.append(linha["data_evento"])

    return out_motivos, out_seqs, out_sent, out_ativos, out_datas


def clusterizar_por_ativo(motivos, seqs, sentimentos, ativos, datas):
    df = pd.DataFrame(
        {
            "motivo": motivos,
            "seq": seqs,
            "sentimento": sentimentos,
            "ativo": ativos,
            "data_evento": datas,
        }
    )

    resultados = []

    for ativo, grupo in df.groupby("ativo"):
        frases = grupo["motivo"].tolist()
        if not frases:
            continue

        print(f"[CLUSTER] ativo={ativo} | motivos={len(frases)}")
        embeds = np.vstack([embed_text(frase) for frase in frases])
        k = min(N_CLUSTERS_POR_ATIVO, max(1, len(frases) // 2))

        cluster = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
        labels = cluster.fit_predict(embeds)
        grupo = grupo.copy()
        grupo["cluster"] = labels

        for cluster_id in sorted(grupo["cluster"].unique()):
            subset = grupo[grupo["cluster"] == cluster_id].copy()
            frases_cluster = subset["motivo"].tolist()
            frase_representativa = gerar_frase_representativa_cluster(frases_cluster, ativo)

            seqs_raw = subset["seq"].tolist()
            max_len = max((len(seq) if isinstance(seq, list) else 0) for seq in seqs_raw)
            seq_avg = []
            for i in range(max_len):
                valores = [seq[i] for seq in seqs_raw if isinstance(seq, list) and len(seq) > i and seq[i] is not None]
                seq_avg.append(float(np.mean(valores)) if valores else None)

            datas_cluster = sorted(pd.to_datetime(subset["data_evento"]).dt.strftime("%Y-%m-%d").tolist())
            resultados.append(
                {
                    "cluster": int(cluster_id),
                    "frase_exemplo": frase_representativa,
                    "ativo_cluster": ativo,
                    "n_eventos": int(len(subset)),
                    "n_motivos_unicos": int(subset["motivo"].nunique()),
                    "prompt_version": PROMPT_VERSION,
                    "first_event_date": datas_cluster[0] if datas_cluster else None,
                    "last_event_date": datas_cluster[-1] if datas_cluster else None,
                    "event_dates": json.dumps(datas_cluster, ensure_ascii=False),
                    "frases_originais": json.dumps(frases_cluster, ensure_ascii=False),
                    **{f"seq_d{i}": seq_avg[i] for i in range(len(seq_avg))},
                }
            )

        df_ativo = pd.DataFrame([row for row in resultados if row["ativo_cluster"] == ativo])
        df_ativo.to_csv(DATA_DIR / f"cluster_{ativo.lower()}.csv", index=False)
        print(f"[OK] salvo cluster_{ativo.lower()}.csv")

    df_final = pd.DataFrame(resultados)
    df_final.to_csv(DATA_DIR / "cluster_motivos.csv", index=False)
    print("[OK] Clusters combinados salvos em:", DATA_DIR / "cluster_motivos.csv")
    return df_final


def gerar_cluster_motivos():
    print("Carregando eventos...")
    motivos, seqs, sentimentos, ativos, datas = carregar_eventos()
    print(f"{len(motivos)} motivos carregados")

    print("Oversample...")
    motivos2, seqs2, sent2, ativos2, datas2 = oversample_motivos_por_ativo(motivos, seqs, sentimentos, ativos, datas)

    print("Clusterizando...")
    df_final = clusterizar_por_ativo(motivos2, seqs2, sent2, ativos2, datas2)

    print("Finalizado.")
    return df_final


if __name__ == "__main__":
    gerar_cluster_motivos()
