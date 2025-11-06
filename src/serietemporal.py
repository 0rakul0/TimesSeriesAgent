"""
Análise de séries temporais e simulações baseadas em Monte Carlo e impacto de notícias.
"""
import os
import numpy as np
import pandas as pd
import yfinance as yf
from collections import OrderedDict
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import STL

# ===============================
# Funções principais
# ===============================
def baixar_dados_acao(ticker: str, periodo: str = "5y") -> pd.DataFrame:
    """
    Baixa dados históricos de uma ação usando yfinance.
    """
    acao = yf.Ticker(ticker)
    if not os.path.exists(f'../data/dados_acao_{ticker}.csv'):
        dados = acao.history(period=periodo)
    else:
        dados = pd.read_csv(f'../data/dados_acao_{ticker}.csv', index_col=0, parse_dates=True)
    dados.to_csv(f'../data/dados_acao_{ticker}.csv')
    return dados


def decomposicao_sazonal(dados: pd.DataFrame, coluna: str = "Close", freq: int = 252):
    """
    Realiza a decomposição sazonal de uma série temporal usando STL (não corta bordas).
    """
    serie_temporal = dados[coluna]

    # --- Decomposição STL ---
    stl = STL(serie_temporal, period=freq)
    resultado = stl.fit()

    trend = resultado.trend
    seasonal = resultado.seasonal
    resid = resultado.resid

    # ============================
    # MÉTRICAS QUANTITATIVAS
    # ============================
    var_total = np.var(serie_temporal)
    var_trend = np.var(trend)
    var_seasonal = np.var(seasonal)

    forca_tendencia = var_trend / var_total
    forca_sazonal = var_seasonal / var_total
    amplitude_media = (seasonal.max() - seasonal.min()).mean()

    impacto_dividendos = np.nan
    dividendos = dados[dados["Dividends"] > 0]
    if not dividendos.empty:
        variacoes = []
        for idx in dividendos.index:
            try:
                preco_antes = dados.loc[idx, coluna]
                prox_idx = dados.index.get_loc(idx) + 5
                if prox_idx < len(dados):
                    preco_depois = dados.iloc[prox_idx][coluna]
                    variacoes.append((preco_depois - preco_antes) / preco_antes * 100)
            except Exception:
                continue
        if variacoes:
            impacto_dividendos = np.mean(variacoes)

    print("\n📊 MÉTRICAS DE DECOMPOSIÇÃO -", ticker)
    print(f"Força da Tendência:     {forca_tendencia:.2%}")
    print(f"Força da Sazonalidade:  {forca_sazonal:.2%}")
    print(f"Amplitude Média Sazonal: {amplitude_media:.2f}")
    if not np.isnan(impacto_dividendos):
        print(f"Impacto Médio dos Dividendos (5 dias): {impacto_dividendos:.2f}%")
    else:
        print("Impacto Médio dos Dividendos: não calculado (faltam dados)")

    # ============================
    # GRÁFICO PRINCIPAL (STL)
    # ============================
    fig_stl = resultado.plot()
    plt.suptitle(f"Decomposição Sazonal - {ticker}", fontsize=14)
    plt.tight_layout()

    if not dividendos.empty:
        eixos = fig_stl.axes
        for ax in eixos:
            for idx in dividendos.index:
                ax.axvline(x=idx, color="purple", linestyle="--", alpha=0.7)
        eixos[-1].legend(["Dividendo"], loc="upper left")

    plt.savefig(f"../img/decomposicao_sazonal_{ticker}.png", bbox_inches="tight")
    plt.show()

    # ============================
    # GRÁFICO RESUMO AUTOMÁTICO
    # ============================
    plt.figure(figsize=(12, 6))
    plt.plot(dados.index, serie_temporal, label="Preço Fechamento", color="blue", linewidth=2)
    plt.plot(dados.index, trend, label="Tendência", color="orange", linestyle="--", linewidth=2)
    plt.plot(dados.index, seasonal + trend.mean(), label="Sazonalidade (ajustada)", color="green", linestyle=":",
             linewidth=1.5)

    # Linhas verticais de dividendos
    if not dividendos.empty:
        for idx in dividendos.index:
            plt.axvline(x=idx, color="purple", linestyle="--", alpha=0.6)
        plt.scatter(dividendos.index, dados.loc[dividendos.index, coluna],
                    color="purple", label="Dividendos", zorder=5)

    plt.title(f"📈 Resumo de Decomposição - {ticker}")
    plt.xlabel("Data")
    plt.ylabel("Preço")
    plt.legend(loc="upper left")
    plt.grid(alpha=0.3)

    # Painel de métricas na parte inferior
    texto_metricas = (
        f"Força da Tendência: {forca_tendencia:.2%}\n"
        f"Força da Sazonalidade: {forca_sazonal:.2%}\n"
        f"Amplitude Média Sazonal: {amplitude_media:.2f}\n"
        f"Impacto Médio dos Dividendos (5 dias): "
        f"{impacto_dividendos:.2f}%" if not np.isnan(impacto_dividendos)
        else "Impacto Médio dos Dividendos: não calculado"
    )

    plt.figtext(0.02, -0.05, texto_metricas, fontsize=10, ha="left", va="top",
                bbox=dict(facecolor="whitesmoke", edgecolor="lightgray", boxstyle="round,pad=0.5"))

    plt.tight_layout()
    plt.savefig(f"../img/resumo_decomposicao_{ticker}.png", bbox_inches="tight")
    plt.show()

    return resultado



def gerar_noticias_sinteticas(caminho_csv: str, datas_referencia: pd.Series):
    """
    Gera um CSV sintético de notícias com base nas datas do dataset do yfinance.
    Cada dia pode ter entre 1 e 4 notícias.
    """
    np.random.seed(42)  # Reprodutibilidade

    registros = []
    for data in datas_referencia:
        num_noticias = np.random.randint(1, 5)
        for _ in range(num_noticias):
            polaridade = np.random.choice([-1, 0, 1], p=[0.2, 0.2, 0.6])
            peso_impacto = np.round(np.random.uniform(0.1, 1.0), 2)
            categoria = np.random.choice(["Econômica", "Financeira", "Política", "Setorial", "Mercado"])
            fonte = np.random.choice(["Reuters", "Valor Econômico", "Estadão", "Bloomberg", "CNN Brasil"])
            titulo = f"Notícia {np.random.randint(1000,9999)}"
            registros.append({
                "data": data,
                "titulo": titulo,
                "polaridade": polaridade,
                "peso_impacto": peso_impacto,
                "categoria": categoria,
                "fonte": fonte
            })

    df = pd.DataFrame(registros)
    Path(caminho_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(caminho_csv, index=False)
    print(f"[INFO] CSV sintético criado em: {caminho_csv} ({len(df)} notícias geradas)")
    return df

def carregar_noticias_sinteticas(datas_referencia: pd.Series):
    """
    Carrega o CSV de notícias sintéticas ou o cria, caso não exista.
    """
    caminho_csv = Path("../util/noticias_sinteticas.csv")

    if not caminho_csv.exists():
        print("[INFO] Arquivo de notícias não encontrado. Gerando novo...")
        return gerar_noticias_sinteticas(caminho_csv, datas_referencia)

    print("[INFO] Lendo notícias existentes de:", caminho_csv)
    noticias = pd.read_csv(caminho_csv, parse_dates=["data"])
    return noticias


def simulacao_baseada_em_noticias(dados: pd.DataFrame, noticias: pd.DataFrame,
                                  coluna: str = "Close", dias: int = 30, peso_volatilidade: float = 0.3):
    """
    Simula o preço futuro de uma ação ponderando o impacto de notícias positivas ou negativas.
    """
    retornos = np.log(1 + dados[coluna].pct_change().dropna())
    media_hist = retornos.mean()
    desvio_hist = retornos.std()
    preco_atual = dados[coluna].iloc[-1]
    precos = [preco_atual]

    for dia in range(dias):
        if dia < len(noticias):
            polaridade = noticias.iloc[dia]['polaridade']
            impacto = noticias.iloc[dia]['peso_impacto']
        else:
            polaridade, impacto = 0, 0

        choque_noticia = polaridade * impacto * desvio_hist * 2
        choque_aleatorio = np.random.normal(media_hist, desvio_hist) * peso_volatilidade
        variacao = choque_noticia + choque_aleatorio
        novo_preco = precos[-1] * np.exp(variacao)
        precos.append(novo_preco)

    return pd.Series(precos, name="Simulação")

# ===============================
# Execução principal revisada
# ===============================
if __name__ == "__main__":
    if __name__ == "__main__":
        tickers = ["PETR4.SA", "BZ=F", "USDBRL=X"]

        # --- Baixar dados da ação (2 anos por padrão) ---
        for ticker in tickers:
            dados_acao = baixar_dados_acao(ticker, periodo="5y")
            decomposicao_sazonal(dados_acao)

            # noticias = carregar_noticias_sinteticas(dados_acao.index)

            # sim = simulacao_baseada_em_noticias(dados_acao, noticias)
            # print("✅ Simulação baseada em notícias concluída.")
