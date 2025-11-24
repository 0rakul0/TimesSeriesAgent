import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import stumpy
import pandas as pd


def analisar_motifs_discords(dados, ticker, janela, n_motifs, limite_volume):
    """
    Analisa padrões (motifs) e anomalias (discords) no preço médio e volume negociado.
    Se existir uma coluna 'Dividends_{ticker}', marca os pagamentos de dividendos nos gráficos.
    """

    # 1️⃣ Verificação de colunas
    cols = [f"High_{ticker}", f"Low_{ticker}", f"Close_{ticker}", f"Volume_{ticker}"]
    if f"Dividends_{ticker}" in dados.columns:
        has_dividends = True
        cols.append(f"Dividends_{ticker}")
    else:
        has_dividends = False

    for c in cols:
        if c not in dados.columns:
            raise ValueError(f"❌ Coluna {c} não encontrada no DataFrame.")

    df = dados[cols].dropna(subset=[f"Close_{ticker}"]).copy()
    df["Preco_Medio"] = df[[f"High_{ticker}", f"Low_{ticker}", f"Close_{ticker}"]].mean(axis=1)

    # 2️⃣ Calcula Matrix Profile
    mp = stumpy.stump(df["Preco_Medio"].values, m=janela)
    mp_dist = mp[:, 0]
    motif_indices = np.argsort(mp_dist)[:n_motifs]
    discord_idx = np.argmax(mp_dist)

    # 3️⃣ Filtra volumes acima do limite
    volume_col = f"Volume_{ticker}"
    picos_volume = df[df[volume_col] > limite_volume]

    # 4️⃣ Dividendos (se houver)
    if has_dividends:
        div_col = f"Dividends_{ticker}"
        df_div = df[df[div_col] > 0][[div_col]]
    else:
        df_div = pd.DataFrame()

    # --- 🖼️ Gráfico estático (matplotlib) ---
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.set_title(f"Motifs, Discords, Volumes e Dividendos - {ticker}")
    ax1.plot(df.index, df["Preco_Medio"], color="tab:blue", linewidth=1.5, label="Preço Médio")

    # Motifs numerados
    for i, idx in enumerate(motif_indices, start=1):
        trecho = df["Preco_Medio"].iloc[idx:idx + janela]
        ax1.plot(df.index[idx:idx + janela], trecho, linewidth=3, alpha=0.8, label=f"Motif {i}")
        meio = idx + janela // 2
        ax1.text(df.index[meio], df["Preco_Medio"].iloc[meio],
                 str(i), color="black", fontsize=9, fontweight="bold",
                 ha="center", va="bottom",
                 bbox=dict(facecolor="white", edgecolor="gray", boxstyle="circle,pad=0.3"))

    # Discord
    ax1.plot(df.index[discord_idx:discord_idx + janela],
             df["Preco_Medio"].iloc[discord_idx:discord_idx + janela],
             color="red", linewidth=3, label="Discord")

    # Eixo de volume
    ax2 = ax1.twinx()
    ax2.bar(df.index, df[volume_col], color="lightgray", alpha=0.4, label="Volume Total")
    ax2.bar(picos_volume.index, picos_volume[volume_col],
            color="orange", alpha=0.8, label=f"Volume > {limite_volume:,}")

    # Dividendos (se houver)
    if has_dividends and not df_div.empty:
        ax1.scatter(df_div.index, df.loc[df_div.index, "Preco_Medio"],
                    color="purple", s=80, marker="v", label="Dividendos", zorder=5)

    # Legenda e layout
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"../img/motifs_discords_dividends_{ticker}.png", bbox_inches="tight")
    plt.close(fig)

    # --- 🌐 Gráfico interativo (Plotly) ---
    fig_plotly = go.Figure()

    # Preço médio
    fig_plotly.add_trace(go.Scatter(x=df.index, y=df["Preco_Medio"],
                                    mode="lines", name="Preço Médio", line=dict(color="blue")))

    # Motifs
    for i, idx in enumerate(motif_indices, start=1):
        trecho = df["Preco_Medio"].iloc[idx:idx + janela]
        fig_plotly.add_trace(go.Scatter(x=df.index[idx:idx + janela], y=trecho,
                                        mode="lines", name=f"Motif {i}", line=dict(width=3)))

    # Discord
    fig_plotly.add_trace(go.Scatter(x=df.index[discord_idx:discord_idx + janela],
                                    y=df["Preco_Medio"].iloc[discord_idx:discord_idx + janela],
                                    mode="lines", name="Discord", line=dict(color="red", width=3)))

    # Volume normalizado
    fig_plotly.add_trace(go.Bar(x=df.index,
                                y=df[volume_col] / df[volume_col].max() * df["Preco_Medio"].max(),
                                name="Volume Normalizado", opacity=0.3))

    # Dividendos interativos
    if has_dividends and not df_div.empty:
        fig_plotly.add_trace(go.Scatter(
            x=df_div.index,
            y=df.loc[df_div.index, "Preco_Medio"],
            mode="markers",
            name="Dividendos",
            marker=dict(symbol="triangle-down", size=10, color="purple"),
            text=[f"Dividendo: {v:.2f}" for v in df_div[f'Dividends_{ticker}']],
            hovertemplate="%{x}<br>Preço: %{y:.2f}<br>%{text}"
        ))

    # Layout final
    fig_plotly.update_layout(
        title=f"Motifs, Discords, Volumes e Dividendos - {ticker}",
        xaxis_title="Data",
        yaxis_title="Preço Médio",
        template="plotly_white",
        legend=dict(x=1.02, y=1)
    )
    fig_plotly.write_html(f"../img/motifs_discords_dividends_{ticker}.html")

    print(f"✅ Gráficos (PNG e HTML) salvos para {ticker} com marcação de dividendos")
    return df, motif_indices, discord_idx, picos_volume


def plotar_series_temporais(dados, ticker="GERAL",
                              titulo="Séries Temporais - Ativos",
                              normalizar=True):
    """
    Plota séries temporais e salva PNG e HTML com nomes dependentes do ticker.
    """
    if normalizar:
        dados_plot = dados / dados.iloc[0] * 100
        ylabel = "Índice Normalizado (base 100)"
        sufixo = "series_temporais_normalizadas"
    else:
        dados_plot = dados
        ylabel = "Preço de Fechamento"
        sufixo = "series_temporais_bruto"

    # Nome final dos arquivos
    arq_png = f"../img/{ticker}_{sufixo}.png"
    arq_html = f"../img/{ticker}_{sufixo}.html"

    # --- Matplotlib (estático) ---
    plt.figure(figsize=(12, 6))
    for coluna in dados_plot.columns:
        plt.plot(dados_plot.index, dados_plot[coluna], label=coluna, linewidth=1.8)
    plt.title(titulo)
    plt.ylabel(ylabel)
    plt.xlabel("Data")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(arq_png, bbox_inches="tight")
    plt.close()

    # --- Plotly (interativo) ---
    fig_plotly = go.Figure()
    for coluna in dados_plot.columns:
        fig_plotly.add_trace(go.Scatter(
            x=dados_plot.index,
            y=dados_plot[coluna],
            mode="lines",
            name=coluna,
        ))
    fig_plotly.update_layout(
        title=titulo,
        xaxis_title="Data",
        yaxis_title=ylabel,
        template="plotly_white"
    )
    fig_plotly.write_html(arq_html)

    print(f"✅ Arquivos salvos:")
    print("   ├─", arq_png)
    print("   └─", arq_html)




def juntar_e_correlacionar_lado_a_lado(caminhos_csv, salvar="../data/dados_combinados.csv"):
    """
    Junta séries temporais padronizadas e salva no mesmo formato de `juntar_series`.
    - Mantém Date como primeira coluna
    - Salva sem índice (index=False)
    - Renomeia colunas por ticker
    """
    series = []

    for ticker, caminho in caminhos_csv.items():
        df = pd.read_csv(f"../data/{caminho}", parse_dates=True, index_col=0)

        # Padroniza índice
        df.index = pd.to_datetime(df.index, utc=True)
        df.index = df.index.tz_convert(None).date

        # Colunas obrigatórias
        colunas_esperadas = ["Open", "High", "Low", "Close", "Volume", "Dividends"]
        faltando = [c for c in colunas_esperadas if c not in df.columns]
        if faltando:
            raise ValueError(f"❌ Colunas faltando em {ticker}: {faltando}")

        # Renomear colunas com sufixo do ticker
        renomeadas = {col: f"{col}_{ticker}" for col in colunas_esperadas}
        df = df.rename(columns=renomeadas)
        df = df[list(renomeadas.values())]

        series.append(df)

    # Junta tudo
    dados = pd.concat(series, axis=1, join="outer").sort_index()
    dados = dados.dropna(how="all")

    # ----- 🔥 Parte igual ao juntar_series -----

    # cria Date como coluna
    dados_reset = dados.copy()
    dados_reset["Date"] = dados.index  # mesma saída que juntar_series

    # reorganizar colunas: Date primeiro
    colunas = ["Date"] + [c for c in dados_reset.columns if c != "Date"]

    # salvar mesmo formato
    dados_reset[colunas].to_csv(salvar, index=False)

    print(f"💾 Arquivo salvo (formato juntar_series): {salvar}")

    # ----- 🔍 Extra: correlação -----
    corr = dados.corr(numeric_only=True)
    print("\n📊 Correlação entre séries:")
    print(corr)

    return dados, corr

if __name__ == "__main__":
    # ======================================================
    # 🔵 1) PRIO3 x BRENT
    # ======================================================
    caminhos_prio = {
        "PRIO3.SA": "dados_acao_PRIO3.SA_5y.csv",
        "BZ=F": "dados_acao_BZ=F_5y.csv",
    }

    dados_prio = juntar_e_correlacionar_lado_a_lado(caminhos_prio, salvar="../data/dados_prio3_brent.csv")

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

    dados_petr = juntar_e_correlacionar_lado_a_lado(caminhos_petr, salvar="../data/dados_petr4_brent.csv")

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
