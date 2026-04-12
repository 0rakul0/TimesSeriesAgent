from __future__ import annotations

import os
import re
from dataclasses import dataclass

import pandas as pd
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.project_paths import BASE_DIR, DATA_DIR


@dataclass(frozen=True)
class KnowledgeDocument:
    title: str
    content: str


def _read_text(path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin1")


def _chunk_markdown(title: str, text: str, max_chars: int = 1400) -> list[KnowledgeDocument]:
    text = text.strip()
    if not text:
        return []

    sections = re.split(r"(?m)^##\s+", text)
    docs: list[KnowledgeDocument] = []
    if len(sections) == 1:
        sections = [text]

    for idx, section in enumerate(sections, start=1):
        clean = section.strip()
        if not clean:
            continue

        if "\n" in clean:
            header, body = clean.split("\n", 1)
        else:
            header, body = f"Trecho {idx}", clean

        body = body.strip()
        if len(body) <= max_chars:
            docs.append(KnowledgeDocument(title=f"{title} · {header.strip()}", content=body))
            continue

        for part_idx in range(0, len(body), max_chars):
            chunk_number = part_idx // max_chars + 1
            docs.append(
                KnowledgeDocument(
                    title=f"{title} · {header.strip()} ({chunk_number})",
                    content=body[part_idx: part_idx + max_chars].strip(),
                )
            )

    return docs


def _results_summary() -> str:
    path = DATA_DIR / "resultado_comparacao_modelos.csv"
    if not path.exists():
        return ""

    df = pd.read_csv(path)
    linhas = []
    for _, row in df.sort_values("RMSE_Hibrido").iterrows():
        linhas.append(
            (
                f"{row['Ativo']}: modelo {row['Modelo']} com RMSE base {row['RMSE_Modelo']:.4f}, "
                f"RMSE hibrido {row['RMSE_Hibrido']:.4f}, ganho {row['Ganho_Hibrido']:.4f} "
                f"e scale {row['Scale_Selecionado']:.2f}."
            )
        )
    return "\n".join(linhas)


def build_documents(asset_code: str, event_detail: dict | None, cluster_match: dict | None, projection: dict) -> list[KnowledgeDocument]:
    docs: list[KnowledgeDocument] = []

    docs.extend(_chunk_markdown("README", _read_text(BASE_DIR / "README.md")))
    docs.extend(_chunk_markdown("Perguntas de Pesquisa", _read_text(DATA_DIR / "perguntas_pesquisa.md")))

    summary = _results_summary()
    if summary:
        docs.append(KnowledgeDocument(title="Comparacao de Modelos", content=summary))

    if event_detail:
        docs.append(
            KnowledgeDocument(
                title=f"Evento Atual · {asset_code}",
                content=(
                    f"Data: {pd.to_datetime(event_detail['data']).strftime('%Y-%m-%d')}\n"
                    f"Sentimento: {event_detail.get('sentimento_do_mercado', 'neutro')}\n"
                    f"Motivos: {', '.join(event_detail.get('motivos_identificados', []))}\n"
                    f"Resumo: {event_detail.get('o_que_houve', '')}\n"
                    f"Fontes: {', '.join(event_detail.get('fontes', []))}"
                ),
            )
        )

    if cluster_match:
        seq_lines = ", ".join(
            f"{label}: {value:.2f}%"
            for label, value in cluster_match.get("seq_map", {}).items()
            if pd.notna(value)
        )
        docs.append(
            KnowledgeDocument(
                title=f"Cluster Atual · {asset_code}",
                content=(
                    f"Cluster: {cluster_match.get('cluster_id')}\n"
                    f"Frase representativa: {cluster_match.get('frase_exemplo', '')}\n"
                    f"Motivo de referencia: {cluster_match.get('motivo_referencia', '')}\n"
                    f"Similaridade: {cluster_match.get('similaridade', 0.0):.3f}\n"
                    f"Eventos no cluster: {cluster_match.get('n_eventos', 0)}\n"
                    f"Comportamento medio: {seq_lines}"
                ),
            )
        )

    docs.append(
        KnowledgeDocument(
            title=f"Projecao Atual · {asset_code}",
            content=(
                f"Modelo selecionado: {projection['modelo']}\n"
                f"Scale: {projection['scale']:.2f}\n"
                f"Ultimo fechamento: {projection['last_close']:.2f}\n"
                f"Previsoes base: {projection['preds_base']}\n"
                f"Previsoes ajustadas: {projection['preds_ajustadas']}\n"
                f"Maximo projetado: {projection['maximo_projetado']:.2f}"
            ),
        )
    )

    return docs


def retrieve_documents(question: str, documents: list[KnowledgeDocument], top_k: int = 4) -> list[KnowledgeDocument]:
    if not documents:
        return []

    corpus = [doc.content for doc in documents] + [question]
    vectorizer = TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(corpus)
    doc_matrix = matrix[:-1]
    query_vector = matrix[-1]
    sims = cosine_similarity(query_vector, doc_matrix).flatten()
    order = sims.argsort()[::-1][:top_k]
    return [documents[idx] for idx in order]


def _fallback_answer(question: str, retrieved: list[KnowledgeDocument]) -> str:
    if not retrieved:
        return "Nao encontrei contexto suficiente na base local para responder essa pergunta."

    lines = [f"Pergunta: {question}", "", "Pontos mais relevantes da base:"]
    for doc in retrieved[:3]:
        preview = doc.content.replace("\n", " ").strip()
        preview = preview[:260] + ("..." if len(preview) > 260 else "")
        lines.append(f"- {doc.title}: {preview}")
    return "\n".join(lines)


def answer_question(question: str, asset_code: str, event_detail: dict | None, cluster_match: dict | None, projection: dict) -> dict:
    documents = build_documents(asset_code, event_detail, cluster_match, projection)
    retrieved = retrieve_documents(question, documents)

    if not os.getenv("OPENAI_API_KEY"):
        return {
            "answer": _fallback_answer(question, retrieved),
            "sources": [doc.title for doc in retrieved],
        }

    prompt_context = "\n\n".join(f"[{doc.title}]\n{doc.content}" for doc in retrieved)
    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.2,
            max_tokens=350,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Responda em portugues do Brasil. Use apenas o contexto fornecido, "
                        "seja objetivo e cite os titulos das fontes no fim."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Contexto:\n{prompt_context}\n\nPergunta:\n{question}",
                },
            ],
        )
        answer = (response.choices[0].message.content or "").strip()
    except Exception:
        answer = _fallback_answer(question, retrieved)

    return {
        "answer": answer,
        "sources": [doc.title for doc in retrieved],
    }
