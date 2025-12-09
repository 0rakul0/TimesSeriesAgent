"""
Análise de séries temporais: PRIO3 x BRENT e PETR4 x BRENT
Decomposição, combinação e motifs/discords.
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

# ============================================================
# 1) BAIXAR DADOS
# ============================================================
def baixar_dados_acao(ticker: str, periodo: str = "5y") -> pd.DataFrame:
    caminho = f'../data/dados_acao_{ticker}_{periodo}.csv'
    acao = yf.Ticker(ticker)

    # ---------------------------------------------------------
    # 1) Se o arquivo não existir → baixa tudo
    # ---------------------------------------------------------
    if not os.path.exists(caminho):
        dados = acao.history(period=periodo)
        dados.to_csv(caminho)
        return dados

    # ---------------------------------------------------------
    # 2) Se existir → carregar CSV existente
    # ---------------------------------------------------------
    dados_antigos = pd.read_csv(caminho, index_col=0, parse_dates=True)

    if dados_antigos.empty:
        dados = acao.history(period=periodo)
        dados.to_csv(caminho)
        return dados

    ultima_data_local = dados_antigos.index.max().date()

    # ---------------------------------------------------------
    # 3) Baixar desde a última data até hoje
    # ---------------------------------------------------------
    hoje = pd.Timestamp.today().normalize().date()

    if ultima_data_local >= hoje:
        # Nada para atualizar
        return dados_antigos

    # Yahoo usa strings de data
    dados_novos = acao.history(start=str(ultima_data_local), end=str(hoje))

    # Remover possível duplicata da última linha
    dados_novos = dados_novos[dados_novos.index.date > ultima_data_local]

    # ---------------------------------------------------------
    # 4) Concatenar e salvar sobrescrevendo
    # ---------------------------------------------------------
    dados_atualizados = pd.concat([dados_antigos, dados_novos])
    dados_atualizados = dados_atualizados[~dados_atualizados.index.duplicated(keep='last')]

    dados_atualizados.to_csv(caminho)

    return dados_atualizados


# ============================================================
# 2) DECOMPOSIÇÃO SAZONAL
# ============================================================
def decomposicao_sazonal(dados: pd.DataFrame, ticker: str, coluna="Close", freq=252):
    serie_temporal = dados[coluna]
    stl = STL(serie_temporal, period=freq)
    result = stl.fit()

    print(f"\n📊 Decomposição — {ticker}")
    print(f"• Força Tendência: {np.var(result.trend)/np.var(serie_temporal):.2%}")
    print(f"• Força Sazonal:  {np.var(result.seasonal)/np.var(serie_temporal):.2%}")

    return result

# ============================================================
# 4) EXECUÇÃO PRINCIPAL
# ============================================================
if __name__ == "__main__":
    tickers = ["PRIO3.SA", "PETR4.SA", "EXXO34.SA", "BZ=F"]

    # --- Baixar e decompor somente PRIO3, PETR4 e BRENT ---
    for ticker in tickers:
        dados = baixar_dados_acao(ticker)
        decomposicao_sazonal(dados, ticker)

