from teste_modelo_hibrido_eval import carregar_modelo, prever
import pandas as pd

df = pd.read_csv("../data/dados_combinados.csv", index_col=0, parse_dates=True)

model, scaler, target_col, seq_len, cols = carregar_modelo("../models/lstm_baseline_price.pt")

pred = prever(model, scaler, df, seq_len, target_col, cols)

print(pred.tail())
