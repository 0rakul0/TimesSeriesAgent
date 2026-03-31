from pathlib import Path


def test_project_paths_point_to_repo_root():
    from utils.project_paths import BASE_DIR, DATA_DIR, MODELOS_DIR, OUTPUT_NOTICIAS_DIR

    assert BASE_DIR.name == "TimesSeriesAgent"
    assert DATA_DIR.parent == BASE_DIR
    assert MODELOS_DIR.parent == BASE_DIR
    assert OUTPUT_NOTICIAS_DIR.parent == BASE_DIR


def test_embedding_manager_uses_repo_paths():
    from utils.embedding_manager import EmbeddingManager
    from utils.project_paths import BASE_DIR

    manager = EmbeddingManager()

    assert Path(manager.base_dir) == BASE_DIR
    assert manager.cache_path.parent.name == "data"
    assert manager.emb_npy_path.parent.name == "modelos"


def test_core_modules_import_without_runtime_side_effects():
    import src.main
    import src.mvp
    import src.pipeline_online

    assert src.main.app.title == "TimesSeriesAgent Online API"
    assert callable(src.mvp.executar_demo)
    assert callable(src.pipeline_online.run_online)


def test_historical_retrieval_respects_cutoff_date():
    import numpy as np
    import pandas as pd

    from eval import modelo_hibrido_offline as mh

    original_emb_mgr = mh.emb_mgr

    class StubEmb:
        def embed(self, text):
            value = 1.0 if "oil" in text.lower() else 0.0
            return np.array([[value, 0.0]])

    mh.emb_mgr = StubEmb()
    try:
        biblioteca = pd.DataFrame(
            [
                {"date": pd.Timestamp("2025-01-10"), "ativo": "PETR4", "motivo": "oil up", "seq": [1.0, 0.5], "embedding": np.array([1.0, 0.0])},
                {"date": pd.Timestamp("2025-02-10"), "ativo": "PETR4", "motivo": "oil down", "seq": [0.8, 0.3], "embedding": np.array([1.0, 0.0])},
                {"date": pd.Timestamp("2025-03-10"), "ativo": "PETR4", "motivo": "future event", "seq": [9.0, 9.0], "embedding": np.array([1.0, 0.0])},
            ]
        )

        contexto = mh.recuperar_impacto_historico(
            motivos=["oil shock"],
            cutoff_date=pd.Timestamp("2025-03-01"),
            biblioteca_eventos=biblioteca,
            ativo="PETR4",
            top_k=5,
            sim_threshold=0.1,
        )

        assert contexto is not None
        assert contexto["n_historico"] == 2
        assert "2025-03-10" not in contexto["datas_referencia"]
    finally:
        mh.emb_mgr = original_emb_mgr


def test_threshold_sensitivity_table_counts_asset_events(tmp_path):
    import pandas as pd

    from src.agent_noticia import analisar_limiares_evento

    csv_path = tmp_path / "dados_teste.csv"
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2025-01-01", periods=5, freq="D"),
            "Close_TESTE.SA": [100, 102, 99, 101, 104],
            "Close_BZ=F": [50, 50.2, 50.1, 50.0, 50.3],
        }
    )
    df.to_csv(csv_path, index=False)

    tabela = analisar_limiares_evento(
        {"TESTE.SA": csv_path},
        limiares=[1.5, 2.0, 2.5],
        incluir_brent=False,
        salvar_csv=False,
    )

    assert list(tabela["Limiar_Evento_Pct"]) == [1.5, 2.0, 2.5]
    assert list(tabela["Qtd_Eventos_Ativo"]) == [4, 3, 2]
    assert list(tabela["Qtd_Eventos_Considerados"]) == [4, 3, 2]


def test_article_answers_generates_expected_columns(tmp_path):
    import pandas as pd

    from eval.article_answers import (
        gerar_casos_piora,
        gerar_eventos_vs_nao_eventos,
        gerar_mapa_perguntas_pesquisa,
        gerar_metricas_complementares,
        gerar_robustez_anual,
    )

    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2026-01-04"]),
            "Ativo": ["PETR4"] * 4,
            "Modelo": ["LSTM"] * 4,
            "Real": [10.0, 11.0, 10.5, 12.0],
            "Pred_Base": [9.8, 10.8, 10.9, 11.0],
            "Pred_Hibrido": [10.1, 11.1, 10.6, 11.7],
            "Hybrid_Better": [True, True, True, True],
            "Event_Day": [True, False, True, False],
            "Year": [2025, 2025, 2025, 2026],
        }
    )
    df["ErroAbs_Base"] = (df["Real"] - df["Pred_Base"]).abs()
    df["ErroAbs_Hibrido"] = (df["Real"] - df["Pred_Hibrido"]).abs()

    metricas = gerar_metricas_complementares(df, output_dir=tmp_path)
    eventos = gerar_eventos_vs_nao_eventos(df, output_dir=tmp_path)
    robustez = gerar_robustez_anual(df, output_dir=tmp_path)
    piora_resumo, piora_detalhes = gerar_casos_piora(df, output_dir=tmp_path)
    mapa = gerar_mapa_perguntas_pesquisa(tmp_path / "perguntas_pesquisa.md")

    assert "MAE_Hibrido" in metricas.columns
    assert "Pct_Dias_Hibrido_Melhor" in eventos.columns
    assert "Ano" in robustez.columns
    assert "Qtd_Dias_Piora" in piora_resumo.columns
    assert isinstance(piora_detalhes, pd.DataFrame)
    assert mapa.exists()
    assert "RQ1" in mapa.read_text(encoding="utf-8")
