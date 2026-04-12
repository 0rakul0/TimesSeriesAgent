import numpy as np

from utils.rerank_utils import hybrid_rerank_scores, lexical_similarity_scores


def test_lexical_similarity_prioritizes_exact_theme():
    scores = lexical_similarity_scores(
        "cortes da opep e oferta global",
        [
            "realizacao de lucros e ajuste tecnico",
            "cortes da opep e preocupacoes com oferta global de petroleo",
        ],
    )
    assert scores[1] > scores[0]


def test_hybrid_rerank_can_overcome_small_semantic_gap_with_lexical_signal():
    scores = hybrid_rerank_scores(
        query="fechamento de ormuz e choque de oferta",
        texts=[
            "choque de oferta de petroleo por tensoes em ormuz",
            "otimismo com resultados trimestrais da empresa",
        ],
        semantic_scores=np.array([0.77, 0.79]),
        dates=["2026-03-15", "2026-03-15"],
        semantic_weight=0.60,
        lexical_weight=0.35,
        recency_weight=0.05,
    )
    assert scores[0] > scores[1]
