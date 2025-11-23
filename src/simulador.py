# simulador_previsao_futura.py (VERSÃO FINAL COMPLETA)
# -------------------------------------------------------
# Pipeline limpo, robusto e 100% funcional:
# 1. Carrega dados reais (dados_combinados.csv)
# 2. Gera dias futuros replicando o último valor (opção A)
# 3. Previsão iterativa multi-step (walk-forward) usando LSTM
# 4. Injeta evento sintético usando D0 = último dia real
# 5. Aplica impacto via final_impact_module.apply_impact_sequences
# 6. Salva CSV + HTML
# -------------------------------------------------------

import os
import numpy as np
import pandas as pd
from datetime import timedelta
import torch

# importa funções do seu projeto
from teste_modelo_hibrido_eval import carregar_modelo, prever, plotar_final
from final_impact_module import apply_impact_sequences

CAMINHO_DADOS = "../data/dados_combinados.csv"
MODELO = "../models/lstm_baseline_price.pt"
CSV_OUT = "../data/simulacao_futura_A.csv"
HTML_OUT = "../img/simulacao_futura_A.html"

DIAS_FUTUROS = 3
MOTIVO_EVENTO = ["alta demanda do petróleo"]
CLUSTER_FORCADO = "posi_brent"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
#  Carregar dados reais
# ---------------------------------------------------------------------------
def carregar_dados(caminho):
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")
    df = pd.read_csv(caminho, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index).normalize()
    return df


# ---------------------------------------------------------------------------
#  Gerar datas futuras (opção A = replicar última linha inteira)
# ---------------------------------------------------------------------------
def gerar_datas_futuras_por_replicar_ultimo(df, dias=3):
    df = df.copy().sort_index()
    last_idx = df.index[-1]
    last_row = df.iloc[-1]

    new_dates = [last_idx + timedelta(days=i) for i in range(1, dias + 1)]
    extra = pd.DataFrame([last_row.values] * dias, columns=df.columns, index=new_dates)
    extra.index = pd.to_datetime(extra.index).normalize()

    return pd.concat([df, extra])


# ---------------------------------------------------------------------------
#  Inverter escala somente do target
# ---------------------------------------------------------------------------
def inverse_scaler_target(scaler, values, target_idx, train_cols):
    zeros = np.zeros((len(values), len(train_cols)), dtype=float)
    zeros[:, target_idx] = values
    return scaler.inverse_transform(zeros)[:, target_idx]


# ---------------------------------------------------------------------------
#  Previsão iterativa multi-step (walk-forward)
# ---------------------------------------------------------------------------
def iterative_forecast(model, scaler, df_full, seq_len, target_col, train_cols, steps, device):
    df_local = df_full.copy()

    # garante colunas
    for c in train_cols:
        if c not in df_local.columns:
            df_local[c] = 0.0
    df_local = df_local[train_cols]

    arr = scaler.transform(df_local)  # array normalizado (N, C)
    cols = list(train_cols)
    target_idx = cols.index(target_col)

    # última posição real
    if "Real" in df_full.columns:
        real_mask = ~df_full["Real"].isna()
        if real_mask.any():
            last_real_pos = np.where(real_mask.values)[0][-1]
        else:
            last_real_pos = len(arr) - steps - 1
    else:
        last_real_pos = len(arr) - steps - 1

    start_seq_pos = max(0, last_real_pos - seq_len + 1)
    seq_window = arr[start_seq_pos:start_seq_pos + seq_len, :].astype(np.float32).copy()

    preds_scaled = []

    model.eval()
    model.to(device)
    with torch.no_grad():
        for _ in range(steps):
            seq_tensor = torch.tensor(seq_window[np.newaxis, :, :], dtype=torch.float32).to(device)
            pred_scaled = model(seq_tensor).cpu().numpy().item()
            preds_scaled.append(pred_scaled)

            new_row = np.zeros((1, arr.shape[1]), dtype=np.float32)
            new_row[0, target_idx] = pred_scaled

            seq_window = np.vstack([seq_window[1:, :], new_row])

    preds_inv = inverse_scaler_target(scaler, np.array(preds_scaled), target_idx, train_cols)
    future_dates = list(df_full.index[-steps:])

    return pd.DataFrame({"Pred": preds_inv, "Real": [np.nan]*steps}, index=future_dates)


# ---------------------------------------------------------------------------
#  MAIN
# ---------------------------------------------------------------------------
def main():
    print("[INFO] Carregando dados reais...")
    df = carregar_dados(CAMINHO_DADOS)
    ultima_real = df.index[-1]
    print(f"[INFO] Último pregão real: {ultima_real.date()}")

    print("[INFO] Carregando modelo baseline...")
    model, scaler, target_col, seq_len, train_cols = carregar_modelo(MODELO)

    print(f"[INFO] Gerando {DIAS_FUTUROS} dias futuros...")
    df_ext = gerar_datas_futuras_por_replicar_ultimo(df, dias=DIAS_FUTUROS)

    # Previsão iterativa para os dias futuros
    print("[INFO] Prevendo dias futuros (iterativo walk-forward)...")
    pred_future = iterative_forecast(
        model, scaler, df_ext, seq_len, target_col, train_cols,
        steps=DIAS_FUTUROS, device=DEVICE
    )

    # Construir DataFrame final de previsão
    pred_df = pd.DataFrame(index=df_ext.index)
    pred_df["Pred"] = np.nan
    pred_df["Real"] = np.nan

    # preencher Real nos dados históricos
    if target_col in df_ext.columns:
        pred_df.loc[df.index, "Real"] = df_ext.loc[df.index, target_col]

    # inserir previsões futuras
    for dt, row in pred_future.iterrows():
        pred_df.at[dt, "Pred"] = row["Pred"]

    # Aplicar impacto (D0 = última data real)
    motivos_por_data = {
        ultima_real: {
            "motivos": MOTIVO_EVENTO,
            "cluster_forcado": CLUSTER_FORCADO
        }
    }

    print(f"[INFO] Aplicando impacto do evento sintético em D0={ultima_real.date()}...")
    pred_final, logs = apply_impact_sequences(
        pred_df,
        motivos_por_data,
        verbose=True,
        horizon=5,
        alpha_method="zscore",
        alpha_cap=(0.25, 4.0),
        max_pct_per_day=0.25
    )

    # Salvar CSV
    os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
    pred_final.to_csv(CSV_OUT)
    print(f"[INFO] CSV salvo em: {CSV_OUT}")

    # Plotar HTML
    try:
        pred_plot = pred_final.dropna(how="all", subset=["Pred", "Pred_Impact"])
        plotar_final(pred_plot, HTML_OUT)
        print(f"[INFO] HTML salvo em: {HTML_OUT}")
    except Exception as e:
        print(f"[WARN] Falha ao gerar HTML: {e}")

    # LOGS
    print("\n=== Últimas linhas ===")
    print(pred_final.tail(10))

    print("\n=== Logs do impacto ===")
    for l in logs:
        print(l)


if __name__ == "__main__":
    main()