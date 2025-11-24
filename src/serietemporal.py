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
from correlacao_ativos import analisar_motifs_discords, plotar_series_temporais


# ============================================================
# 1) BAIXAR DADOS
# ============================================================
def baixar_dados_acao(ticker: str, periodo: str = "5y") -> pd.DataFrame:
    caminho = f'../data/dados_acao_{ticker}_{periodo}.csv'
    acao = yf.Ticker(ticker)

    if not os.path.exists(caminho):
        dados = acao.history(period=periodo)
        dados.to_csv(caminho)
    else:
        dados = pd.read_csv(caminho, index_col=0, parse_dates=True)

    return dados


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

    tickers = ["PRIO3.SA", "PETR4.SA", "BZ=F"]

    # --- Baixar e decompor somente PRIO3, PETR4 e BRENT ---
    for ticker in tickers:
        dados = baixar_dados_acao(ticker)
        decomposicao_sazonal(dados, ticker)

