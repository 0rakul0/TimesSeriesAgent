"""
PIPELINE OFICIAL – TimesSeriesAgent
Fluxo completo:
1) Baixar dados + decomposição sazonal
2) Correlação entre ativos
3) Comparação de séries
4) Detecção de eventos
5) Clusterização semântica de motivos
6) Treinamento (LSTM, AE e Transformer)
7) Avaliação dos modelos híbridos
"""

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

# ============================================================
# CONFIGURAÇÕES GERAIS
# ============================================================

TICKERS = ["PRIO3.SA", "PETR4.SA", "EXXO34.SA", "BZ=F"]

ATIVOS_A = ["PETR4.SA", "PRIO3.SA", "EXXO34.SA"]

CSV_FONTE = [
    "dados_acao_PETR4.SA_5y.csv",
    "dados_acao_PRIO3.SA_5y.csv",
    "dados_acao_EXXO34.SA_5y.csv",
]

CSV_SAIDA = [
    "../data/dados_petr4_brent.csv",
    "../data/dados_prio3_brent.csv",
    "../data/dados_exxo34_brent.csv",
]

ATIVO_B = "BZ=F"
CSV_B = "dados_acao_BZ=F_5y.csv"

COMPARACAO = {
    "PETR4.SA": "../data/dados_petr4_brent.csv",
    "PRIO3.SA": "../data/dados_prio3_brent.csv",
    "EXXO34.SA": "../data/dados_exxo34_brent.csv"
}


# ============================================================
# 1) BAIXAR DADOS + DECOMPOSIÇÃO
# ============================================================

def etapa_dados():
    print("\n=== ETAPA 1 — BAIXANDO DADOS + DECOMPOSIÇÃO ===")
    for ticker in TICKERS:
        dados = baixar_dados_acao(ticker)
        decomposicao_sazonal(dados, ticker)
    print("✔ Dados prontos.\n")


# ============================================================
# 2) CORRELAÇÃO ENTRE ATIVOS
# ============================================================

def etapa_correlacao():
    print("\n=== ETAPA 2 — CORRELAÇÃO ENTRE ATIVOS ===")
    for fonte, saida, ativo_a in zip(CSV_FONTE, CSV_SAIDA, ATIVOS_A):
        print(f"\n🔵 {ativo_a} x {ATIVO_B}")
        caminhos = {
            ativo_a: fonte,
            ATIVO_B: CSV_B
        }
        juntar_e_correlacionar_lado_a_lado(caminhos, salvar=saida)
    print("✔ Correlação concluída.\n")


# ============================================================
# 3) COMPARAÇÕES / MOTIFS / DISCORDS
# ============================================================

def etapa_comparacoes():
    print("\n=== ETAPA 3 — COMPARAÇÕES DE ATIVOS ===")
    for ticker, csv in COMPARACAO.items():
        comparar_ativos_interativo(
            caminho_csv=csv,
            ticker_a=ticker,
            ticker_b=ATIVO_B,
            janela_rolling=30,
            limiar=5.0
        )
    print("✔ Comparações concluídas.\n")


# ============================================================
# 4) DETECÇÃO DE EVENTOS
# ============================================================

def etapa_eventos():
    print("\n=== ETAPA 4 — DETECÇÃO DE EVENTOS ===")

    for ticker, caminho in COMPARACAO.items():
        detectar_eventos(ticker, caminho)

    remover_jsons_sem_motivos()

    gerar_sequencias_eventos()
    print("✔ Eventos detectados.\n")


# ============================================================
# 5) CLUSTERIZAÇÃO DE FRASES (semântica)
# ============================================================

def etapa_clusterizacao():
    print("\n=== ETAPA 5 — CLUSTERIZAÇÃO DE FRASES ===")
    gerar_cluster_motivos()
    print("✔ Clusterização concluída.\n")


# ============================================================
# 6) TREINAMENTO LSTM + AE
# ============================================================

def etapa_treinamento():
    print("\n=== ETAPA 6 — TREINAMENTO DE MODELOS ===")

    # LSTM
    treino_lstm("../data/dados_petr4_brent.csv", "../modelos/lstm_petr4.pt")
    treino_lstm("../data/dados_prio3_brent.csv", "../modelos/lstm_prio3.pt")
    treino_lstm("../data/dados_exxo34_brent.csv", "../modelos/lstm_exxo34.pt")

    # AutoEncoder
    treino_ae("../data/dados_petr4_brent.csv", "../modelos/autoencoder_petr4.pt")
    treino_ae("../data/dados_prio3_brent.csv", "../modelos/autoencoder_prio3.pt")
    treino_ae("../data/dados_exxo34_brent.csv", "../modelos/autoencoder_exxo34.pt")

    # Transformrs
    treino_transformer("../data/dados_petr4_brent.csv","../modelos/transformer_petr4.pt")
    treino_transformer("../data/dados_prio3_brent.csv","../modelos/transformer_prio3.pt")
    treino_transformer("../data/dados_exxo34_brent.csv","../modelos/transformer_exxo34.pt")

    print("🎉 Modelos treinados!\n")


# ============================================================
# 7) AVALIAÇÃO HÍBRIDA
# ============================================================

def etapa_avaliacao():
    print("\n=== ETAPA 7 — AVALIAÇÃO DOS MODELOS HÍBRIDOS ===")
    eval_modelos()
    print("✔ Avaliação concluída.\n")


# ============================================================
# EXECUÇÃO COMPLETA
# ============================================================

def run_pipeline():
    etapa_dados()
    etapa_correlacao()
    etapa_comparacoes()
    etapa_eventos()
    # etapa_clusterizacao()
    # etapa_treinamento()
    etapa_avaliacao()

    print("\n🎯 PIPELINE COMPLETA FINALIZADA COM SUCESSO!\n")


if __name__ == "__main__":
    run_pipeline()
