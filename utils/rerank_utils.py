from __future__ import annotations

import ast
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def parse_jsonish_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    text = str(value).strip()
    if not text:
        return []

    for parser in (ast.literal_eval,):
        try:
            parsed = parser(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except (ValueError, SyntaxError):
            continue

    return [text]


def token_overlap_score(query: str, text: str) -> float:
    q_tokens = {token for token in normalize_text(query).split() if token}
    t_tokens = {token for token in normalize_text(text).split() if token}
    if not q_tokens or not t_tokens:
        return 0.0
    return len(q_tokens & t_tokens) / len(q_tokens)


def lexical_similarity_scores(query: str, texts: Iterable[str]) -> np.ndarray:
    texts = [normalize_text(text) for text in texts]
    query = normalize_text(query)
    if not texts or not query:
        return np.zeros((len(texts),), dtype=float)

    word_vectorizer = TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2))
    char_vectorizer = TfidfVectorizer(lowercase=True, strip_accents="unicode", analyzer="char_wb", ngram_range=(3, 5))

    word_matrix = word_vectorizer.fit_transform(texts + [query])
    char_matrix = char_vectorizer.fit_transform(texts + [query])
    word_scores = cosine_similarity(word_matrix[-1], word_matrix[:-1]).flatten()
    char_scores = cosine_similarity(char_matrix[-1], char_matrix[:-1]).flatten()
    overlap_scores = np.asarray([token_overlap_score(query, text) for text in texts], dtype=float)

    scores = 0.45 * word_scores + 0.40 * char_scores + 0.15 * overlap_scores
    return np.clip(scores, 0.0, 1.0)


def recency_scores(dates: Iterable[object]) -> np.ndarray:
    values = pd.to_datetime(list(dates), errors="coerce")
    if len(values) == 0:
        return np.zeros((0,), dtype=float)

    valid = values[~pd.isna(values)]
    if len(valid) == 0:
        return np.zeros((len(values),), dtype=float)

    min_date = valid.min()
    max_date = valid.max()
    span_days = max((max_date - min_date).days, 1)

    scores = []
    for value in values:
        if pd.isna(value):
            scores.append(0.0)
        else:
            scores.append((value - min_date).days / span_days)
    return np.asarray(scores, dtype=float)


def hybrid_rerank_scores(
    query: str,
    texts: Iterable[str],
    semantic_scores: Iterable[float],
    dates: Iterable[object] | None = None,
    semantic_weight: float = 0.70,
    lexical_weight: float = 0.25,
    recency_weight: float = 0.05,
) -> np.ndarray:
    texts = list(texts)
    semantic = np.asarray(list(semantic_scores), dtype=float)
    if len(texts) != len(semantic):
        raise ValueError("texts e semantic_scores devem ter o mesmo tamanho.")

    lexical = lexical_similarity_scores(query, texts)
    recency_input = [None] * len(texts) if dates is None else list(dates)
    recency = recency_scores(recency_input)

    total = semantic_weight + lexical_weight + recency_weight
    if total <= 0:
        total = 1.0

    scores = (
        semantic_weight * np.clip(semantic, 0.0, 1.0)
        + lexical_weight * lexical
        + recency_weight * recency
    ) / total

    return np.clip(scores, 0.0, 1.0)


def select_semantic_pool(
    semantic_scores: Iterable[float],
    top_k: int,
    pool_size: int,
    semantic_margin: float = 0.06,
) -> np.ndarray:
    semantic = np.asarray(list(semantic_scores), dtype=float)
    if semantic.size == 0:
        return np.asarray([], dtype=int)

    order = np.argsort(semantic)[::-1]
    best = float(semantic[order[0]])
    pool = [idx for idx in order if semantic[idx] >= best - semantic_margin]

    min_pool = min(len(order), max(top_k, pool_size))
    if len(pool) < min_pool:
        pool = order[:min_pool].tolist()
    else:
        pool = pool[:pool_size]

    return np.asarray(pool, dtype=int)
