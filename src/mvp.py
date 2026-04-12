from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch

from eval.modelo_hibrido_offline import (
    RERANK_CONFIG_BY_MODEL,
    carregar_modelo_unificado,
    motivo_e_cluster_mais_relevante,
)
from utils.embedding_manager import EmbeddingManager
from utils.project_paths import DATA_DIR, IMG_DIR, MODELOS_DIR, ensure_runtime_dirs

ensure_runtime_dirs()

DEFAULT_CSV_PETR4 = DATA_DIR / "dados_petr4_brent.csv"
CLUSTER_CSV = DATA_DIR / "cluster_motivos.csv"
RESULTADOS_CSV = DATA_DIR / "resultado_comparacao_modelos.csv"

emb_mgr = EmbeddingManager()


def _model_device(model):
    return next(model.parameters()).device


def detectar_coluna_close(df, ativo="PETR4"):
    ativo = (ativo or "").lower()
    cols = list(df.columns)
    target = f"close_{ativo}"

    for coluna in cols:
        if coluna.lower() == target:
            return coluna

    for candidato in ["close", "adj close", "close_price", "preco_fechamento", "fechamento"]:
        for coluna in cols:
            if coluna.lower() == candidato:
                return coluna

    for coluna in cols:
        if "close" in coluna.lower():
            return coluna

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        for coluna in numeric_cols[::-1]:
            if "volume" not in coluna.lower():
                return coluna
        return numeric_cols[-1]

    raise KeyError(f"Nenhuma coluna Close encontrada. Colunas disponiveis: {cols}")


def escolher_melhor_modelo(ativo="PETR4"):
    if not RESULTADOS_CSV.exists():
        raise FileNotFoundError(f"{RESULTADOS_CSV} nao encontrado.")

    df = pd.read_csv(RESULTADOS_CSV)
    df_ativo = df[df["Ativo"].astype(str).str.contains(ativo, na=False)]
    if df_ativo.empty:
        raise ValueError(f"Nenhum modelo encontrado para {ativo} em {RESULTADOS_CSV}")

    best_row = df_ativo.sort_values("RMSE_Hibrido").iloc[0]
    modelo_tipo = str(best_row["Modelo"]).lower()
    model_path = MODELOS_DIR / f"{modelo_tipo}_{ativo.lower()}.pt"
    print(f"[OK] Melhor modelo para {ativo}: {modelo_tipo.upper()}")
    return modelo_tipo, model_path


def carregar_janela_base(csv_path, seq_len):
    df = pd.read_csv(csv_path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    df = df[df["Date"].notna()].copy()
    if df.empty:
        raise RuntimeError("CSV vazio ou sem coluna Date valida.")

    df = df.ffill().bfill()
    janela = df.tail(seq_len).reset_index(drop=True)
    if len(janela) < seq_len:
        faltam = seq_len - len(janela)
        primeiro = janela.iloc[0] if not janela.empty else df.iloc[0]
        datas_pad = pd.bdate_range(end=pd.to_datetime(primeiro["Date"]) - timedelta(days=1), periods=faltam)
        pad = []
        for data in datas_pad:
            row = primeiro.copy()
            row["Date"] = data
            pad.append(row)
        janela = pd.concat([pd.DataFrame(pad), janela], ignore_index=True)

    return janela.reset_index(drop=True)


def _denormalizar_target(pred_scaled, scaler, train_cols, target_idx):
    zeros = np.zeros((1, len(train_cols)))
    zeros[0, target_idx] = pred_scaled
    return float(scaler.inverse_transform(zeros)[0, target_idx])


def _preparar_proxima_linha(base_row, train_cols, pred_price):
    proxima = base_row.copy()
    for coluna in train_cols:
        coluna_lower = coluna.lower()
        if any(token in coluna_lower for token in ["open_", "high_", "low_", "close_"]) and "bz=f" not in coluna_lower:
            proxima[coluna] = pred_price
    return proxima


def prever_proximos_3(model_path, csv_path, modelo_tipo):
    model, scaler, train_cols, seq_len = carregar_modelo_unificado(str(model_path), tipo=modelo_tipo)
    target_col = next(col for col in train_cols if col.startswith("Close_"))
    target_idx = train_cols.index(target_col)

    df_window = carregar_janela_base(csv_path, seq_len)
    for coluna in train_cols:
        if coluna not in df_window.columns:
            df_window[coluna] = 0.0

    preds = []
    dates = []
    working = df_window.copy()
    last_row = working.iloc[-1].copy()

    for passo in range(3):
        arr = scaler.transform(working[train_cols].astype(float))
        seq_tensor = torch.tensor(arr[np.newaxis], dtype=torch.float32).to(_model_device(model))

        with torch.no_grad():
            if modelo_tipo.lower() == "autoencoder":
                pred_scaled, _ = model(seq_tensor)
                pred_scaled = float(pred_scaled.cpu().numpy().item())
            else:
                pred_scaled = float(model(seq_tensor).cpu().numpy().item())

        pred_price = _denormalizar_target(pred_scaled, scaler, train_cols, target_idx)
        preds.append(pred_price)

        next_date = pd.bdate_range(start=pd.to_datetime(last_row["Date"]) + timedelta(days=1), periods=1)[0]
        dates.append(next_date)

        new_row = _preparar_proxima_linha(last_row, train_cols, pred_price)
        new_row["Date"] = next_date
        working = pd.concat([working.iloc[1:], pd.DataFrame([new_row])], ignore_index=True)
        last_row = new_row

    hist_arr = scaler.transform(df_window[train_cols].astype(float))
    hist_seq = hist_arr[np.newaxis]
    hist_tensor = torch.tensor(hist_seq, dtype=torch.float32).to(_model_device(model))
    with torch.no_grad():
        if modelo_tipo.lower() == "autoencoder":
            hist_pred_scaled, _ = model(hist_tensor)
            hist_pred_scaled = float(hist_pred_scaled.cpu().numpy().item())
        else:
            hist_pred_scaled = float(model(hist_tensor).cpu().numpy().item())

    last_hist_pred = _denormalizar_target(hist_pred_scaled, scaler, train_cols, target_idx)
    df_pred_full = pd.DataFrame({"Date": dates, "Pred": preds})

    return preds, df_pred_full, last_hist_pred


def obter_cluster_de_motivo(motivos, ativo="PETR4", modelo_tipo=None):
    clusters_df = pd.read_csv(CLUSTER_CSV)
    emb_repr = emb_mgr.embed_lote(clusters_df["frase_exemplo"].astype(str).tolist())
    rerank_config = None
    if modelo_tipo:
        rerank_config = RERANK_CONFIG_BY_MODEL.get(str(modelo_tipo).lower(), RERANK_CONFIG_BY_MODEL["default"])
    motivo, sim, cluster_id, row = motivo_e_cluster_mais_relevante(
        motivos,
        emb_mgr,
        emb_repr,
        clusters_df,
        ativo,
        rerank_config=rerank_config,
    )
    if row is None:
        row = pd.Series(dtype=object)
    return motivo, sim, cluster_id, row


def ajustar_previsao_com_impacto(preds, cluster_row, sim, scale=0.4):
    impactos = []
    for k in range(1, 4):
        col = f"seq_d{k}"
        if isinstance(cluster_row, pd.Series) and col in cluster_row.index and pd.notna(cluster_row[col]):
            impactos.append(float(cluster_row[col]) / 100.0)
        else:
            impactos.append(0.0)
    preds_adj = [pred * (1 + scale * float(sim) * impacto) for pred, impacto in zip(preds, impactos)]
    return preds_adj, impactos


def construir_datas_e_historico(df, col_close):
    df_valid = df.dropna(subset=[col_close]).sort_values("Date").reset_index(drop=True)
    if df_valid.empty:
        raise RuntimeError("Dados vazios ou coluna close sem valores.")

    ultimos5 = df_valid.tail(5).copy()
    if len(ultimos5) < 5:
        last_date = df_valid["Date"].max()
        days = pd.bdate_range(end=last_date, periods=5)
        candidate = df_valid.set_index(pd.to_datetime(df_valid["Date"]).dt.normalize())
        precos = []
        for data in days:
            if data in candidate.index:
                valor = candidate.loc[data, col_close]
                if isinstance(valor, pd.Series):
                    valor = valor.iloc[-1]
                precos.append(float(valor))
            else:
                precos.append(np.nan)
        datas_hist = list(days)
        precos_hist = precos
    else:
        datas_hist = list(pd.to_datetime(ultimos5["Date"]).dt.normalize())
        precos_hist = list(ultimos5[col_close].astype(float))

    d0 = datas_hist[-1]
    futuros = list(pd.bdate_range(start=d0 + timedelta(days=1), periods=3))
    return datas_hist + futuros, precos_hist, d0


def plotar(datas_total, precos_hist, preds_base_full, preds_adj_full, impactos, show=True):
    labels = [pd.to_datetime(data).strftime("%d/%m/%Y") for data in datas_total]
    labels_focus = labels[-4:]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=labels[: len(precos_hist)],
            y=precos_hist,
            mode="lines+markers",
            name="Historico",
            line=dict(color="gray", width=2),
            marker=dict(size=6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=labels_focus,
            y=preds_base_full,
            mode="lines+markers",
            name="Previsao Base",
            line=dict(color="blue", width=3),
            marker=dict(size=8),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=labels_focus,
            y=preds_adj_full,
            mode="lines+markers",
            name="Previsao Ajustada",
            line=dict(color="red", width=3),
            marker=dict(size=8),
        )
    )
    fig.add_trace(
        go.Bar(
            x=labels_focus[1:],
            y=[impacto * 100 for impacto in impactos],
            name="Impacto (%)",
            marker_color=["green" if impacto >= 0 else "red" for impacto in impactos],
            opacity=0.45,
            yaxis="y2",
        )
    )

    fig.update_layout(
        title="Comparacao entre previsao base e previsao ajustada por noticias",
        xaxis=dict(title="Datas"),
        yaxis=dict(title="Preco"),
        yaxis2=dict(title="Impacto (%)", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        margin=dict(l=60, r=180, t=80, b=80),
        template="plotly_white",
        height=700,
    )
    fig.update_xaxes(tickangle=-20)

    if show:
        fig.show()
    return fig


def executar_demo(retornar_html=False, ativo="PETR4", csv_path=None, show_plot=True, motivos=None):
    csv_path = str(csv_path or DEFAULT_CSV_PETR4)

    print("\n====================================")
    print("       TimesSeriesAgent - MVP")
    print("====================================\n")

    modelo_tipo, model_path = escolher_melhor_modelo(ativo)
    preds, df_pred_full, last_hist_pred = prever_proximos_3(model_path, csv_path, modelo_tipo)
    print("Previsao base (D+1..D+3):", preds)

    motivos = motivos or ["crise politica no pais"]
    motivo_sel, sim, cluster_id, row = obter_cluster_de_motivo(motivos, ativo, modelo_tipo=modelo_tipo)
    preds_adj, impactos = ajustar_previsao_com_impacto(preds, row, sim, scale=0.4)

    print("Motivo selecionado:", motivo_sel)
    print("Similaridade:", sim)
    print("Cluster:", cluster_id)
    print("Impactos (D+1..D+3):", impactos)
    print("Previsao ajustada (D+1..D+3):", preds_adj)

    df = pd.read_csv(csv_path, parse_dates=["Date"]).sort_values("Date")
    col_close = detectar_coluna_close(df, ativo)
    datas_total, precos_hist, _ = construir_datas_e_historico(df, col_close)

    ultimo_preco_real = float(precos_hist[-1])
    preds_base_full = [ultimo_preco_real] + [float(valor) for valor in preds]
    preds_adj_full = [ultimo_preco_real] + [float(valor) for valor in preds_adj]

    fig = plotar(datas_total, precos_hist, preds_base_full, preds_adj_full, impactos, show=show_plot)

    resultado = {
        "fig": fig,
        "html": None,
        "preds_base": preds,
        "preds_ajustadas": preds_adj,
        "last_hist_pred": last_hist_pred,
        "motivo": motivo_sel,
        "similaridade": sim,
        "cluster_id": cluster_id,
    }

    if retornar_html:
        caminho_html = IMG_DIR / f"mvp_online_{ativo.lower()}.html"
        fig.write_html(caminho_html)
        print(f"[OK] HTML salvo em: {caminho_html}")
        resultado["html"] = str(caminho_html)

    return resultado


if __name__ == "__main__":
    executar_demo(retornar_html=True, ativo="PETR4", csv_path=str(DEFAULT_CSV_PETR4), show_plot=False)
