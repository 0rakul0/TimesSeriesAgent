import pandas as pd
import plotly.graph_objects as go


def comparar_ativos_interativo(caminho_csv: str,
                               ticker_a: str,
                               ticker_b: str,
                               janela_rolling: int = 30,
                               limiar: float = 5.0):
    """
    Compara dinamicamente dois ativos usando retornos percentuais, correlação móvel
    e detecção de eventos extremos (> limiar%).
    """

    print(f"\n🔍 Comparando: {ticker_a}  x  {ticker_b}")

    df = pd.read_csv(caminho_csv, parse_dates=["Date"])
    df = df.set_index("Date")
    df.index = pd.to_datetime(df.index).normalize()

    col_a = f"Close_{ticker_a}"
    col_b = f"Close_{ticker_b}"

    # Verifica colunas necessárias
    if col_a not in df.columns:
        raise ValueError(f"❌ Coluna não encontrada no CSV: {col_a}")
    if col_b not in df.columns:
        raise ValueError(f"❌ Coluna não encontrada no CSV: {col_b}")

    # --- Cálculo de retornos ---
    df[f"Ret_{ticker_a}"] = df[col_a].pct_change(fill_method=None) * 100
    df[f"Ret_{ticker_b}"] = df[col_b].pct_change(fill_method=None) * 100

    df = df.dropna(subset=[f"Ret_{ticker_a}", f"Ret_{ticker_b}"])
    corr = df[f"Ret_{ticker_a}"].corr(df[f"Ret_{ticker_b}"])

    df["Corr_rolling"] = (
        df[f"Ret_{ticker_a}"]
        .rolling(window=janela_rolling)
        .corr(df[f"Ret_{ticker_b}"])
    )

    # --- Detectar eventos acima do limiar ---
    eventos = df[df[f"Ret_{ticker_a}"].abs() > limiar]
    eventos = eventos[[f"Ret_{ticker_a}", f"Ret_{ticker_b}"]]

    print(f"\n⚡ Eventos detectados ({ticker_a} variação > {limiar}%):")
    if eventos.empty:
        print("Nenhum evento encontrado.")
    else:
        for data, linha in eventos.iterrows():
            print(f"📅 {data.date()} | {ticker_a}: {linha[f'Ret_{ticker_a}']:.2f}% | "
                  f"{ticker_b}: {linha[f'Ret_{ticker_b}']:.2f}%")

    # ============================================================
    # 🔵 Gráfico 1 — Retorno %
    # ============================================================
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=df.index,
        y=df[f"Ret_{ticker_a}"],
        mode='lines',
        name=f'{ticker_a} - Retorno (%)'
    ))
    fig1.add_trace(go.Scatter(
        x=df.index,
        y=df[f"Ret_{ticker_b}"],
        mode='lines',
        name=f'{ticker_b} - Retorno (%)'
    ))

    # Eventos
    fig1.add_trace(go.Scatter(
        x=eventos.index,
        y=eventos[f"Ret_{ticker_a}"],
        mode='markers',
        name='Eventos relevantes',
        marker=dict(color='red', size=9, symbol='star')
    ))

    fig1.update_layout(
        title=f"Variação Percentual Diária - {ticker_a} vs {ticker_b}<br>"
              f"Correlação Geral: {corr:.3f}",
        xaxis_title="Data",
        yaxis_title="Retorno (%)",
        template="plotly_white"
    )
    fig1.write_html(f"../img/retornos_{ticker_a}_vs_{ticker_b}.html")


    # ============================================================
    # 🔵 Gráfico 2 — Correlação móvel
    # ============================================================
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df.index,
        y=df["Corr_rolling"],
        mode='lines',
        name=f"Correlação {janela_rolling} dias"
    ))

    fig2.add_hline(y=0, line_dash="dot")

    fig2.update_layout(
        title=f"Correlação Móvel ({janela_rolling} dias) - {ticker_a} vs {ticker_b}",
        xaxis_title="Data",
        yaxis_title="Correlação",
        template="plotly_white"
    )
    fig2.write_html(f"../img/correlacao_{ticker_a}_vs_{ticker_b}.html")

    print("✅ Gráficos gerados com sucesso.")
    return df, eventos, corr


# Execução direta
if __name__ == "__main__":
    df, eventos, corr = comparar_ativos_interativo(
        caminho_csv="../data/dados_petr4_brent.csv",
        ticker_a="PETR4.SA",
        ticker_b="BZ=F",
        janela_rolling=30,
        limiar=5.0
    )

    df, eventos, corr = comparar_ativos_interativo(
        caminho_csv="../data/dados_prio3_brent.csv",
        ticker_a="PRIO3.SA",
        ticker_b="BZ=F",
        janela_rolling=30,
        limiar=5.0
    )
