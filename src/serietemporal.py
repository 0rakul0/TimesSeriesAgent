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
# 3) COMBINAR BASES
# ============================================================
def carregar_e_preparar(caminho, ticker):
    df = pd.read_csv(caminho, index_col=0, parse_dates=True)

    df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
    df["Date"] = df.index.date  # cria coluna explícita
    df = df.set_index("Date")   # transforma Date em índice

    colunas = ["Open", "High", "Low", "Close", "Volume", "Dividends"]
    renomeadas = {c: f"{c}_{ticker}" for c in colunas}

    return df.rename(columns=renomeadas)[list(renomeadas.values())]


def juntar_series(caminhos_csv: dict, salvar):
    series = []
    for ticker, arq in caminhos_csv.items():
        df = carregar_e_preparar(f"../data/{arq}", ticker)
        series.append(df)

    dados = pd.concat(series, axis=1, join="outer").sort_index()
    dados = dados.dropna(how="all")

    # garante Date como primeira coluna no CSV
    dados_reset = dados.copy()
    dados_reset["Date"] = dados_reset.index
    colunas = ["Date"] + [c for c in dados_reset.columns if c != "Date"]
    dados_reset[colunas].to_csv(salvar, index=False)

    print(f"💾 Arquivo salvo em: {salvar}  — {dados.shape}")
    return dados


# ============================================================
# 4) EXECUÇÃO PRINCIPAL
# ============================================================
if __name__ == "__main__":

    tickers = ["PRIO3.SA", "PETR4.SA", "BZ=F"]

    # --- Baixar e decompor somente PRIO3, PETR4 e BRENT ---
    for ticker in tickers:
        dados = baixar_dados_acao(ticker)
        decomposicao_sazonal(dados, ticker)

    # ======================================================
    # 🔵 1) PRIO3 x BRENT
    # ======================================================
    caminhos_prio = {
        "PRIO3.SA": "dados_acao_PRIO3.SA_5y.csv",
        "BZ=F": "dados_acao_BZ=F_5y.csv",
    }

    dados_prio = juntar_series(caminhos_prio, salvar="../data/dados_prio3_brent.csv")

    plotar_series_temporais(dados_prio, titulo="PRIO3 x BRENT", normalizar=True)

    dados_prio = pd.read_csv("../data/dados_prio3_brent.csv", index_col=0, parse_dates=True)

    analisar_motifs_discords(
        dados_prio,
        ticker="PRIO3.SA",
        janela=60,
        n_motifs=5,
        limite_volume=30_000_000
    )

    analisar_motifs_discords(
        dados_prio,
        ticker="BZ=F",
        janela=60,
        n_motifs=5,
        limite_volume=50_000
    )


    # ======================================================
    # 🔵 2) PETR4 x BRENT
    # ======================================================
    caminhos_petr = {
        "PETR4.SA": "dados_acao_PETR4.SA_5y.csv",
        "BZ=F": "dados_acao_BZ=F_5y.csv",
    }

    dados_petr = juntar_series(caminhos_petr, salvar="../data/dados_petr4_brent.csv")

    plotar_series_temporais(dados_petr, titulo="PETR4 x BRENT", normalizar=True)

    dados_petr = pd.read_csv("../data/dados_petr4_brent.csv", index_col=0, parse_dates=True)

    analisar_motifs_discords(
        dados_petr,
        ticker="PETR4.SA",
        janela=60,
        n_motifs=5,
        limite_volume=50_000_000
    )

    analisar_motifs_discords(
        dados_petr,
        ticker="BZ=F",
        janela=60,
        n_motifs=5,
        limite_volume=50_000
    )
