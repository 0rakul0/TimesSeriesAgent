import pandas as pd
import plotly.graph_objects as go

def comparar_petr4_brent_interativo(caminho_csv: str = "../data/dados_combinados.csv", janela_rolling: int = 30):
    df = pd.read_csv(caminho_csv, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index).normalize()

    col_petr = "Close_PETR4.SA"
    col_brent = "Close_BZ=F"

    if col_petr not in df.columns or col_brent not in df.columns:
        raise ValueError("❌ Colunas de fechamento não encontradas no CSV.")

    df["Ret_PETR4"] = df[col_petr].pct_change(fill_method=None) * 100
    df["Ret_BZ"] = df[col_brent].pct_change(fill_method=None) * 100
    df = df.dropna(subset=["Ret_PETR4", "Ret_BZ"])
    corr = df["Ret_PETR4"].corr(df["Ret_BZ"])

    df["Corr_rolling"] = df["Ret_PETR4"].rolling(window=janela_rolling).corr(df["Ret_BZ"])

    # --- Gráfico 1: Variação percentual ---
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df.index, y=df["Ret_PETR4"], mode='lines', name='PETR4 - Retorno (%)'))
    fig1.add_trace(go.Scatter(x=df.index, y=df["Ret_BZ"], mode='lines', name='Brent (BZ=F) - Retorno (%)'))
    fig1.update_layout(
        title=f"Variação Percentual Diária - PETR4 vs Brent<br>Correlação Geral: {corr:.3f}",
        xaxis_title="Data",
        yaxis_title="Variação % Diária",
        template="plotly_white"
    )
    fig1.write_html("../img/correlacao_petr4_brent_retorno.html")

    # --- Gráfico 2: Correlação móvel ---
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df.index, y=df["Corr_rolling"], mode='lines', name=f"Correlação {janela_rolling} dias"))
    fig2.add_hline(y=0, line_dash="dot")
    fig2.update_layout(
        title=f"Correlação Móvel ({janela_rolling} dias) - PETR4 vs Brent",
        xaxis_title="Data",
        yaxis_title="Correlação",
        template="plotly_white"
    )
    fig2.write_html("../img/correlacao_rolling_petr4_brent.html")

    print("✅ Gráficos interativos salvos em ../img/")
    return df, corr

if __name__ == "__main__":
    df, corr = comparar_petr4_brent_interativo("../data/dados_combinados.csv", janela_rolling=30)
