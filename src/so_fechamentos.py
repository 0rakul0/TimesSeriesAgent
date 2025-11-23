import pandas as pd


df = pd.read_csv("../data/dados_combinados.csv", index_col=0, parse_dates=True)
df.index = pd.to_datetime(df.index).normalize()

df = df.sort_index()


print(df.info())