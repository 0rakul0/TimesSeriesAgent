#!/usr/bin/env python3
"""
avaliador_dinamico.py — Versão final dinâmica v6

✔ Detecta automaticamente ROOT_DIR
✔ Auto-treina se não houver modelos
✔ Carrega checkpoints com weights_only=False
✔ Aceita modelos residuais USING scaler_delta
✔ Reconstrói preço(t) = preço(t-1) + Δ(t)
✔ Avalia todos os modelos
✔ Seleção DINÂMICA dos modelos que receberão o híbrido:
    • melhor modelo geral
    • melhor modelo por família (LSTM, GRU, MLP, Transformer)
    • melhor residual vs melhor normal
    • filtros de qualidade
✔ Aplica híbrido D+1 nos selecionados
✔ Gera CSV + HTML final
"""

import os
import glob
import sys
import subprocess
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# DETECTAR ROOT DO PROJETO (100% compatível com sua estrutura atual)
# -----------------------------------------------------------------------------
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))         # TimesSeriesAgent/evaluation
ROOT_DIR = os.path.abspath(os.path.join(EVAL_DIR, ".."))      # TimesSeriesAgent
SRC_DIR  = os.path.join(ROOT_DIR, "src")                      # TimesSeriesAgent/src

print("[INFO] ROOT_DIR detectado:", ROOT_DIR)

DATA_PATH  = os.path.join(ROOT_DIR, "data", "dados_combinados.csv")
MODEL_DIR  = os.path.join(ROOT_DIR, "models")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
PASTA_JSON = os.path.join(ROOT_DIR, "output_noticias")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# IMPORT DAS ARQUITETURAS (modelos v6)
# -----------------------------------------------------------------------------
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from models.lstm import LSTMModel
from models.lstm_residual import LSTMResidual
from models.gru import GRUModel
from models.gru_residual import GRUResidual
from models.transformer import TransformerModel
from models.transformer_residual import TransformerResidual
from models.mlp import MLPLag
from models.mlp_residual import MLPLagResidual

from hybrid.noticias import carregar_base_frases, extrair_motivos_ultimos_dias
from hybrid.hibrido import calcular_pesos_por_noticia, aplicar_correcao_d_plus_1

# Permitir o MinMaxScaler no pickle (PyTorch 2.6)
from torch.serialization import add_safe_globals
from sklearn.preprocessing import MinMaxScaler
add_safe_globals([MinMaxScaler])


# -----------------------------------------------------------------------------
# Função para reconstruir modelo conforme atributo arch
# -----------------------------------------------------------------------------
def reconstruir_por_arch(arch, input_size, seq_len):
    arch = (arch or "").lower()
    if "lstm" in arch and "residual" in arch:
        return LSTMResidual(input_size)
    if "lstm" in arch:
        return LSTMModel(input_size)
    if "gru" in arch and "residual" in arch:
        return GRUResidual(input_size)
    if "gru" in arch:
        return GRUModel(input_size)
    if "transformer" in arch and "residual" in arch:
        return TransformerResidual(input_size)
    if "transformer" in arch:
        return TransformerModel(input_size)
    if "mlp" in arch and "residual" in arch:
        return MLPLagResidual(input_size, seq_len)
    if "mlp" in arch:
        return MLPLag(input_size, seq_len)
    return LSTMModel(input_size)


# -----------------------------------------------------------------------------
# Previsão full, com reconstrução residual Δ(t)
# -----------------------------------------------------------------------------
def prever_full(model, ckpt, df_numeric, seq_len):
    train_cols = ckpt["train_columns"]
    target_col = ckpt["target_col"]
    scaler_price = ckpt["scaler"]
    scaler_delta = ckpt.get("scaler_delta", None)
    is_residual = bool(ckpt.get("is_residual", False))

    df2 = df_numeric.copy()
    for c in train_cols:
        if c not in df2.columns:
            df2[c] = 0.0
    df2 = df2[train_cols]

    arr = scaler_price.transform(df2)
    tgt_idx = train_cols.index(target_col)

    preds_scaled = []
    reals_scaled = []
    dates = []

    model.eval()
    with torch.no_grad():
        for i in range(len(arr) - seq_len):
            seq = arr[i:i+seq_len]
            seq_t = torch.tensor(seq[np.newaxis,:,:], dtype=torch.float32).to(DEVICE)
            out = model(seq_t).cpu().numpy().item()
            preds_scaled.append(out)
            reals_scaled.append(arr[i+seq_len, tgt_idx])
            dates.append(df2.index[i+seq_len])

    preds_scaled = np.array(preds_scaled)
    reals_scaled = np.array(reals_scaled)

    # inversão normal
    def inv_price(values_scaled):
        zeros = np.zeros((len(values_scaled), len(train_cols)))
        zeros[:, tgt_idx] = values_scaled
        return scaler_price.inverse_transform(zeros)[:, tgt_idx]

    df_pred = pd.DataFrame({
        "Date": dates,
        "Pred": inv_price(preds_scaled) if not is_residual else preds_scaled,
        "Real": inv_price(reals_scaled)
    }).set_index("Date")


    # reconstrução residual
    if is_residual:
        if scaler_delta is None:
            raise RuntimeError("Modelo residual mas checkpoint não contém scaler_delta")

        # inverter Δ
        delta = scaler_delta.inverse_transform(preds_scaled.reshape(-1,1)).flatten()

        # reconstrução preço(t)
        prev_real = []
        for dt in df_pred.index:
            pos = df_numeric.index.get_loc(dt)
            prev_real.append(df_numeric.iloc[max(0, pos-1)][target_col])

        df_pred["Delta_pred"] = delta
        df_pred["Real_prev"] = prev_real
        df_pred["Pred"] = delta + np.array(prev_real)

    return df_pred


# -----------------------------------------------------------------------------
# Seleção DINÂMICA dos modelos para híbrido
# -----------------------------------------------------------------------------
def selecionar_modelos_dinamicamente(resultados):
    """
    resultados[nome] = {
        "df": df_pred,
        "rmse_tecnico": rmse,
        "arch": arch,
        "df_hibrido": None,
        "rmse_hibrido": None
    }
    """

    # 1. Ranking geral
    ranking = sorted(resultados.items(), key=lambda kv: kv[1]["rmse_tecnico"])
    best_overall_name, best_overall_info = ranking[0]

    best_rmse = best_overall_info["rmse_tecnico"]
    threshold = best_rmse * 2.0     # limite de qualidade

    print("\n🏆 Melhor modelo geral:", best_overall_name, "| RMSE:", best_rmse)

    selecionados = set()
    selecionados.add(best_overall_name)

    # 2. Separar por famílias
    familias = {
        "lstm": [],
        "gru": [],
        "mlp": [],
        "transformer": [],
        "residual": [],
        "normal": []
    }

    for nome, info in resultados.items():
        arch = (info["arch"] or "").lower()
        rmse = info["rmse_tecnico"]

        if rmse > threshold:  # modelo ruim → ignora
            continue

        if "lstm" in arch:
            familias["lstm"].append((nome, rmse))
        if "gru" in arch:
            familias["gru"].append((nome, rmse))
        if "transformer" in arch:
            familias["transformer"].append((nome, rmse))
        if "mlp" in arch:
            familias["mlp"].append((nome, rmse))
        if "residual" in arch:
            familias["residual"].append((nome, rmse))
        else:
            familias["normal"].append((nome, rmse))

    # 3. Selecionar melhor de cada família
    for fam, lst in familias.items():
        if not lst:
            continue
        best_fam = min(lst, key=lambda x: x[1])[0]
        selecionados.add(best_fam)

    selecionados = list(selecionados)
    print("\n🤖 Modelos selecionados para híbrido (dinâmico):", selecionados)

    return selecionados


# -----------------------------------------------------------------------------
# Plot comparativo
# -----------------------------------------------------------------------------
def plot_comparativo(resultados, modo, out_html):
    fig = go.Figure()
    first = next(iter(resultados.values()))
    fig.add_trace(go.Scatter(x=first["df"].index, y=first["df"]["Real"],
                             name="Real", line=dict(color="black", width=3)))

    for nome, info in resultados.items():
        df = info["df"]
        fig.add_trace(go.Scatter(x=df.index, y=df["Pred"],
                                 name=f"{nome} (Técnico {info['rmse_tecnico']:.4f})"))

        if info["df_hibrido"] is not None:
            dfh = info["df_hibrido"]
            fig.add_trace(go.Scatter(x=dfh.index, y=dfh["Pred_Ajustado"],
                                     name=f"{nome} (Híbrido {info['rmse_hibrido']:.4f})",
                                     line=dict(width=3, dash="dash")))

    fig.update_layout(title=f"Comparação de Modelos — Modo {modo}",
                      template="plotly_white", width=1400, height=700)

    fig.write_html(out_html)
    print("\n📊 HTML salvo em:", out_html)


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    print("\n🚀 Avaliador Dinâmico v6 (FULL)\n")

    # 1. Procurar modelos
    files = sorted(glob.glob(os.path.join(MODEL_DIR, "*.pt")))
    if not files:
        print("Nenhum modelo encontrado. Treinando automaticamente...")
        train_script = os.path.join(SRC_DIR, "training", "train_all_v6.py")
        r = subprocess.run([sys.executable, train_script], cwd=ROOT_DIR)
        if r.returncode != 0:
            raise RuntimeError("Treinamento falhou.")
        files = sorted(glob.glob(os.path.join(MODEL_DIR, "*.pt")))

    print(f"\n📌 {len(files)} modelos encontrados:")
    for f in files:
        print(" -", os.path.basename(f))

    # 2. Carregar dados originais
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index).normalize()
    df = df.ffill().bfill()

    # 3. Carregar embeddings
    meta_df, emb_matrix = carregar_base_frases()

    resultados = {}

    # 4. Avaliação técnica
    for path in files:
        nome = os.path.basename(path).replace(".pt", "")
        print(f"\n=== Carregando {nome}")

        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        input_size = len(ckpt["train_columns"])
        seq_len = ckpt["seq_len"]
        arch = ckpt.get("arch", None)

        model = reconstruir_por_arch(arch, input_size, seq_len)
        try:
            model.load_state_dict(ckpt["model_state"])
        except Exception:
            model.load_state_dict(ckpt["model_state"], strict=False)
        model.to(DEVICE).eval()

        df_pred = prever_full(model, ckpt, df, seq_len)
        rmse_tecnico = float(np.sqrt(mean_squared_error(df_pred["Real"], df_pred["Pred"])))

        resultados[nome] = {
            "df": df_pred,
            "rmse_tecnico": rmse_tecnico,
            "arch": arch,
            "df_hibrido": None,
            "rmse_hibrido": None
        }

        print(f"{nome} → RMSE técnico: {rmse_tecnico:.4f}")

    # 5. Seleção dinâmicca dos modelos
    modelos_hibrido = selecionar_modelos_dinamicamente(resultados)

    # 6. Aplicar híbrido FULL
    print("\n===== Híbrido FULL =====")
    lista_jsons = glob.glob(os.path.join(PASTA_JSON, "evento_*.json"))
    if lista_jsons:
        datas = [pd.to_datetime(os.path.basename(f).split("_")[-1].replace(".json",""))
                 for f in lista_jsons]
        dias_total = (max(datas) - min(datas)).days
    else:
        dias_total = len(df)

    for nome in modelos_hibrido:
        print(f"\n[HÍBRIDO] {nome}")
        info = resultados[nome]
        df_pred = info["df"]

        noticias = extrair_motivos_ultimos_dias(PASTA_JSON, dias_total, df_pred.index[-1])
        pesos = calcular_pesos_por_noticia(noticias, meta_df, emb_matrix)

        df_corr = aplicar_correcao_d_plus_1(df_pred, pesos, noticias, janela_test=len(df_pred))
        rmse_h = float(np.sqrt(mean_squared_error(df_corr["Real"], df_corr["Pred_Ajustado"])))

        resultados[nome]["df_hibrido"] = df_corr
        resultados[nome]["rmse_hibrido"] = rmse_h

        print(f"  → RMSE híbrido: {rmse_h:.4f}")

    # 7. Exportar CSV + HTML
    csv_out = os.path.join(OUTPUT_DIR, "avaliacao_dinamica_v6.csv")
    pd.DataFrame([
        {
            "modelo": n,
            "arch": info["arch"],
            "rmse_tecnico": info["rmse_tecnico"],
            "rmse_hibrido": info["rmse_hibrido"],
        }
        for n,info in resultados.items()
    ]).to_csv(csv_out, index=False)

    print("\n📄 CSV salvo em:", csv_out)

    html_out = os.path.join(OUTPUT_DIR, "comparacao_dinamica_v6_full.html")
    plot_comparativo(resultados, "full", html_out)

    print("\n🎯 Avaliação Dinâmica FINALIZADA!\n")


if __name__ == "__main__":
    main()
