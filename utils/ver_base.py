import pandas as pd

#
# df = pd.read_csv("../data/dados_prio3_brent.csv", parse_dates=["Date"])

# import pandas as pd
import plotly.graph_objects as go

# Carrega arquivo CSV enviado
df = pd.read_csv("../data/resultado_comparacao_modelos.csv")

# Cria gráfico com Plotly GO
fig = go.Figure()

fig.add_trace(go.Bar(
    x=df["Ativo"],
    y=df["Ganho"],
    text=df["Ganho"].round(3),
    textposition="outside",
    marker_color="#1f77b4"
))

fig.update_layout(
    title="Ganho do Modelo Híbrido por Ativo",
    xaxis_title="Ativo",
    yaxis_title="Ganho (RMSE Modelo - RMSE Híbrido)",
    template="plotly_white",
    height=500,
)

fig.show()
