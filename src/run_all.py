"""
Pipeline principal do TimesSeriesAgent.

Fluxo:
1. Baixa dados e faz decomposicao sazonal
2. Gera series ativo x Brent
3. Executa analise exploratoria
4. Detecta eventos e sequencias
5. Gera clusters semanticos
6. Treina modelos base
7. Avalia o modelo hibrido
8. Gera tabelas auxiliares e mapa de perguntas de pesquisa
"""

from eval.article_answers import gerar_respostas_artigo
from eval.modelo_hibrido_offline import eval_modelos
from src.agent_noticia import detectar_eventos, remover_jsons_sem_motivos
from src.correlacao_ativos import juntar_e_correlacionar_lado_a_lado
from src.frases_clusters import gerar_cluster_motivos
from src.seq_eventos import gerar_sequencias_eventos
from src.serietemporal import baixar_dados_acao, decomposicao_sazonal
from src.variacia_ativos import comparar_ativos_interativo
from train.trainar_ae_ativo import treino_ae
from train.treinar_ativo import treino_lstm
from train.treinar_transformrs import treino_transformer
from utils.project_paths import DATA_DIR, MODELOS_DIR

TICKERS = ["PRIO3.SA", "PETR4.SA", "EXXO34.SA", "BZ=F"]
ATIVOS_A = ["PETR4.SA", "PRIO3.SA", "EXXO34.SA"]
ATIVO_B = "BZ=F"

CSV_FONTE = {
    "PETR4.SA": DATA_DIR / "dados_acao_PETR4.SA_5y.csv",
    "PRIO3.SA": DATA_DIR / "dados_acao_PRIO3.SA_5y.csv",
    "EXXO34.SA": DATA_DIR / "dados_acao_EXXO34.SA_5y.csv",
}
CSV_B = DATA_DIR / "dados_acao_BZ=F_5y.csv"
COMPARACAO = {
    "PETR4.SA": DATA_DIR / "dados_petr4_brent.csv",
    "PRIO3.SA": DATA_DIR / "dados_prio3_brent.csv",
    "EXXO34.SA": DATA_DIR / "dados_exxo34_brent.csv",
}


def etapa_dados():
    print("\n=== ETAPA 1 - BAIXANDO DADOS + DECOMPOSICAO ===")
    for ticker in TICKERS:
        dados = baixar_dados_acao(ticker)
        decomposicao_sazonal(dados, ticker)
    print("[OK] Dados prontos.\n")


def etapa_correlacao():
    print("\n=== ETAPA 2 - CORRELACAO ENTRE ATIVOS ===")
    for ativo_a in ATIVOS_A:
        caminhos = {ativo_a: str(CSV_FONTE[ativo_a]), ATIVO_B: str(CSV_B)}
        saida = COMPARACAO[ativo_a]
        juntar_e_correlacionar_lado_a_lado(caminhos, salvar=str(saida))
    print("[OK] Correlacao concluida.\n")


def etapa_comparacoes():
    print("\n=== ETAPA 3 - COMPARACOES DE ATIVOS ===")
    for ticker, csv_path in COMPARACAO.items():
        comparar_ativos_interativo(
            caminho_csv=str(csv_path),
            ticker_a=ticker,
            ticker_b=ATIVO_B,
            janela_rolling=30,
            limiar=5.0,
        )
    print("[OK] Comparacoes concluidas.\n")


def etapa_eventos():
    print("\n=== ETAPA 4 - DETECCAO DE EVENTOS ===")
    for ticker, caminho in COMPARACAO.items():
        detectar_eventos(ticker, str(caminho))
    remover_jsons_sem_motivos()
    gerar_sequencias_eventos()
    print("[OK] Eventos detectados.\n")


def etapa_clusterizacao():
    print("\n=== ETAPA 5 - CLUSTERIZACAO DE FRASES ===")
    gerar_cluster_motivos()
    print("[OK] Clusterizacao concluida.\n")


def etapa_treinamento():
    print("\n=== ETAPA 6 - TREINAMENTO DE MODELOS ===")
    treino_lstm(str(DATA_DIR / "dados_petr4_brent.csv"), str(MODELOS_DIR / "lstm_petr4.pt"))
    treino_lstm(str(DATA_DIR / "dados_prio3_brent.csv"), str(MODELOS_DIR / "lstm_prio3.pt"))
    treino_lstm(str(DATA_DIR / "dados_exxo34_brent.csv"), str(MODELOS_DIR / "lstm_exxo34.pt"))

    treino_ae(str(DATA_DIR / "dados_petr4_brent.csv"), str(MODELOS_DIR / "autoencoder_petr4.pt"))
    treino_ae(str(DATA_DIR / "dados_prio3_brent.csv"), str(MODELOS_DIR / "autoencoder_prio3.pt"))
    treino_ae(str(DATA_DIR / "dados_exxo34_brent.csv"), str(MODELOS_DIR / "autoencoder_exxo34.pt"))

    treino_transformer(str(DATA_DIR / "dados_petr4_brent.csv"), str(MODELOS_DIR / "transformer_petr4.pt"))
    treino_transformer(str(DATA_DIR / "dados_prio3_brent.csv"), str(MODELOS_DIR / "transformer_prio3.pt"))
    treino_transformer(str(DATA_DIR / "dados_exxo34_brent.csv"), str(MODELOS_DIR / "transformer_exxo34.pt"))
    print("[OK] Modelos treinados.\n")


def etapa_avaliacao():
    print("\n=== ETAPA 7 - AVALIACAO DOS MODELOS HIBRIDOS ===")
    eval_modelos()
    gerar_respostas_artigo()
    print("[OK] Avaliacao e tabelas auxiliares concluidas.\n")


def run_pipeline():
    etapa_dados()
    etapa_correlacao()
    etapa_comparacoes()
    etapa_eventos()
    etapa_clusterizacao()
    etapa_treinamento()
    # etapa_avaliacao()
    print("\n[OK] Pipeline completo finalizado com sucesso.\n")


if __name__ == "__main__":
    run_pipeline()
