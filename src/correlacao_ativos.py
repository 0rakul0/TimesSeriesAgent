# ============================================================
#  correlação_ativos.py — versão organizada e revisada
# ============================================================

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import stumpy
import os

# ============================================================
# 1) FUNÇÃO: Motifs & Discords
# ============================================================
def analisar_motifs_discords(dados, ticker, janela, n_motifs, limite_volume):
    """
    Identifica motifs, discords, picos de volume e dividendos.
    Gera gráficos estáticos (PNG) e interativos (HTML).
    """

    # --- valida colunas ---
    cols = [f"High_{ticker}", f"Low_{ticker}", f"Close_{ticker}", f"Volume_{ticker}"]
    col_div = f"Dividends_{ticker}"

    has_dividends = col_div in dados.columns
    if has_dividends:
        cols.append(col_div)

    for c in cols:
        if c not in dados.columns:
            raise ValueError(f"❌ Coluna {c} não encontrada no DataFrame.")

    # --- preparar dataframe ---
    df = dados[cols].dropna(subset=[f"Close_{ticker}"]).copy()
    df["Preco_Medio"] = df[[f"High_{ticker}", f"Low_{ticker}", f"Close_{ticker}"]].mean(axis=1)

    # --- matrix profile ---
    mp = stumpy.stump(df["Preco_Medio"].values, m=janela)
    mp_dist = mp[:, 0]

    motif_indices = np.argsort(mp_dist)[:n_motifs]
    discord_idx = np.argmax(mp_dist)

    volume_col = f"Volume_{ticker}"
    picos_volume = df[df[volume_col] > limite_volume]

    df_div = df[df[col_div] > 0][[col_div]] if has_dividends else pd.DataFrame()

    # =====================================================
    #  Gráfico Estático (PNG)
    # =====================================================
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.set_title(f"Motifs, Discords, Volumes e Dividendos - {ticker}")
    ax1.plot(df.index, df["Preco_Medio"], color="tab:blue", linewidth=1.5, label="Preço Médio")

    # motifs
    for i, idx in enumerate(motif_indices, start=1):
        trecho = df["Preco_Medio"].iloc[idx:idx + janela]
        ax1.plot(df.index[idx:idx + janela], trecho, linewidth=3, alpha=0.8, label=f"Motif {i}")

        meio = idx + janela // 2
        ax1.text(df.index[meio], df["Preco_Medio"].iloc[meio],
                 str(i), color="black", fontsize=9, fontweight="bold",
                 ha="center", va="bottom",
                 bbox=dict(facecolor="white", edgecolor="gray", boxstyle="circle,pad=0.3"))

    # discord
    ax1.plot(df.index[discord_idx:discord_idx + janela],
             df["Preco_Medio"].iloc[discord_idx:discord_idx + janela],
             color="red", linewidth=3, label="Discord")

    # volume
    ax2 = ax1.twinx()
    ax2.bar(df.index, df[volume_col], color="lightgray", alpha=0.4)
    ax2.bar(picos_volume.index, picos_volume[volume_col], color="orange", alpha=0.8)

    # dividendos
    if has_dividends and not df_div.empty:
        ax1.scatter(df_div.index, df.loc[df_div.index, "Preco_Medio"],
                    color="purple", s=80, marker="v", label="Dividendos", zorder=5)

    # legenda
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left", bbox_to_anchor=(1.02, 0.5))

    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_png = f"../img/motifs_discords_dividends_{ticker}.png"
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    # =====================================================
    #  Gráfico Interativo (HTML)
    # =====================================================
    fig_plotly = go.Figure()

    fig_plotly.add_trace(go.Scatter(
        x=df.index, y=df["Preco_Medio"], mode="lines",
        name="Preço Médio", line=dict(color="blue")
    ))

    # motifs
    for i, idx in enumerate(motif_indices, start=1):
        trecho = df["Preco_Medio"].iloc[idx:idx + janela]
        fig_plotly.add_trace(go.Scatter(
            x=df.index[idx:idx + janela], y=trecho,
            mode="lines", name=f"Motif {i}", line=dict(width=3)
        ))

    # discord
    fig_plotly.add_trace(go.Scatter(
        x=df.index[discord_idx:discord_idx + janela],
        y=df["Preco_Medio"].iloc[discord_idx:discord_idx + janela],
        mode="lines", name="Discord", line=dict(color="red", width=3)
    ))

    # volume normalizado
    fig_plotly.add_trace(go.Bar(
        x=df.index,
        y=df[volume_col] / df[volume_col].max() * df["Preco_Medio"].max(),
        name="Volume Normalizado",
        opacity=0.3
    ))

    # dividendos
    if has_dividends and not df_div.empty:
        fig_plotly.add_trace(go.Scatter(
            x=df_div.index,
            y=df.loc[df_div.index, "Preco_Medio"],
            mode="markers",
            name="Dividendos",
            marker=dict(symbol="triangle-down", size=10, color="purple"),
        ))

    fig_plotly.update_layout(
        title=f"Motifs, Discords, Volumes e Dividendos - {ticker}",
        xaxis_title="Data",
        yaxis_title="Preço Médio",
        template="plotly_white",
        legend=dict(x=1.02, y=1),
    )

    out_html = f"../img/motifs_discords_dividends_{ticker}.html"
    fig_plotly.write_html(out_html)

    print(f"✅ Motifs/Discords salvos para {ticker}:")
    print("   ├─", out_png)
    print("   └─", out_html)

    return df, motif_indices, discord_idx, picos_volume


# ============================================================
# 2) FUNÇÃO: Plot séries temporais
# ============================================================
def plotar_series_temporais(dados, ticker, titulo, normalizar=True):

    if normalizar:
        dados_plot = dados / dados.iloc[0] * 100
        ylabel = "Índice Normalizado (base 100)"
        sufixo = "series_temporais_normalizadas"
    else:
        dados_plot = dados
        ylabel = "Preço"
        sufixo = "series_temporais_bruto"

    out_png = f"../img/{ticker}_{sufixo}.png"
    out_html = f"../img/{ticker}_{sufixo}.html"

    # gráfico estático
    plt.figure(figsize=(12, 6))
    for coluna in dados_plot.columns:
        plt.plot(dados_plot.index, dados_plot[coluna], linewidth=1.8, label=coluna)

    plt.title(titulo)
    plt.xlabel("Data")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

    # gráfico interativo
    fig = go.Figure()
    for coluna in dados_plot.columns:
        fig.add_trace(go.Scatter(x=dados_plot.index, y=dados_plot[coluna], mode="lines", name=coluna))

    fig.update_layout(title=titulo, xaxis_title="Data", yaxis_title=ylabel, template="plotly_white")
    fig.write_html(out_html)

    print("✅ Séries salvas:")
    print("   ├─", out_png)
    print("   └─", out_html)


# ============================================================
# 3) FUNÇÃO: Merge + Correlação
# ============================================================
def juntar_e_correlacionar_lado_a_lado(caminhos_fonte, salvar):
    """
    Junta arquivos crus, renomeia colunas com sufixos,
    e salva CSV combinado.
    """
    series = []

    for ticker, caminho in caminhos_fonte.items():

        df = pd.read_csv(f"../data/{caminho}", parse_dates=True, index_col=0)

        df.index = pd.to_datetime(df.index, utc=True)
        df.index = df.index.tz_convert(None).date

        base_cols = ["Open", "High", "Low", "Close", "Volume", "Dividends"]
        faltando = [c for c in base_cols if c not in df.columns]

        if faltando:
            raise ValueError(f"❌ Colunas faltando em {ticker}: {faltando}")

        renomeadas = {col: f"{col}_{ticker}" for col in base_cols}
        df = df.rename(columns=renomeadas)
        df = df[list(renomeadas.values())]

        series.append(df)

    # unir
    dados = pd.concat(series, axis=1, join="outer").sort_index()
    dados = dados.dropna(how="all")

    dados_reset = dados.copy()
    dados_reset["Date"] = dados.index
    colunas = ["Date"] + [c for c in dados_reset.columns if c != "Date"]
    dados_reset[colunas].to_csv(salvar, index=False)

    print(f"💾 CSV combinado salvo em: {salvar}")

    corr = dados.corr(numeric_only=True)
    print("\n📊 Correlação:")
    print(corr)

    return dados, corr


# ============================================================
# 4) MAIN — processo completo
# ============================================================
if __name__ == "__main__":

    ativos_a = ["PETR4.SA", "PRIO3.SA", "EXXO34.SA"]

    caminhos_fonte = [
        "dados_acao_PETR4.SA_5y.csv",
        "dados_acao_PRIO3.SA_5y.csv",
        "dados_acao_EXXO34.SA_5y.csv"
    ]

    caminhos_saida = [
        "../data/dados_petr4_brent.csv",
        "../data/dados_prio3_brent.csv",
        "../data/dados_exxo34_brent.csv"
    ]

    ativo_b = "BZ=F"

    limites_volume = {
        "PETR4.SA": 50_000_000,
        "PRIO3.SA": 30_000_000,
        "EXXO34.SA": 5_000_000,
        "BZ=F": 50_000,
    }

    janela = 60
    n_motifs = 5

    # LOOP PRINCIPAL
    for fonte, salva, ticker_a in zip(caminhos_fonte, caminhos_saida, ativos_a):

        print("\n" + "=" * 80)
        print(f"🔵 PROCESSANDO: {ticker_a}  x  {ativo_b}")
        print("=" * 80)

        caminhos_locais = {
            ticker_a: fonte,
            ativo_b: "dados_acao_BZ=F_5y.csv",
        }

        dados, corr = juntar_e_correlacionar_lado_a_lado(caminhos_locais, salvar=salva)

        plotar_series_temporais(
            dados,
            ticker=ticker_a.replace(".SA", ""),
            titulo=f"{ticker_a} x {ativo_b}",
            normalizar=True
        )


    print("\n\n🎉 Processo concluído com sucesso!")
