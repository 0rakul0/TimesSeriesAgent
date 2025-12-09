# src/mvp.py — MVP final (versão revisada e pronta para FastAPI)
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta

from utils.embedding_manager import EmbeddingManager
from eval.modelo_hibrido_offline import (
    carregar_modelo_unificado,
    prever_unificado,
    motivo_e_cluster_mais_relevante,
)

# ---------------------------
# CONFIG
# ---------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CSV_PETR4 = os.path.join(BASE_DIR, "data", "dados_petr4_brent.csv")
CLUSTER_CSV = os.path.join(BASE_DIR, "data", "cluster_motivos.csv")
RESULTADOS_CSV = os.path.join(BASE_DIR, "data", "resultado_comparacao_modelos.csv")
MODEL_DIR = os.path.join(BASE_DIR, "modelos")
IMG_DIR = os.path.join(BASE_DIR, "img")
os.makedirs(IMG_DIR, exist_ok=True)

# reuse single EmbeddingManager (cache)
emb_mgr = EmbeddingManager()


# ---------------------------
# Aux helpers
# ---------------------------
def detectar_coluna_close(df, ativo="PETR4"):
    """Detecta coluna 'close' com heurísticas seguras."""
    ativo = (ativo or "").lower()
    cols = list(df.columns)

    # 1) explicit Close_{ativo}
    target = f"close_{ativo}"
    for c in cols:
        if c.lower() == target:
            return c

    # 2) common names (case-insensitive)
    candidates = ["close", "adj close", "close_price", "preco_fechamento", "fechamento"]
    for cand in candidates:
        for c in cols:
            if c.lower() == cand:
                return c

    # 3) any column containing 'close'
    for c in cols:
        if "close" in c.lower():
            return c

    # 4) last numeric column as fallback (warn)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        # prefer columns that look like price (not Volume)
        for c in numeric_cols[::-1]:
            if "volume" not in c.lower():
                return c
        return numeric_cols[-1]

    raise KeyError(f"Nenhuma coluna 'Close' encontrada no CSV. Colunas disponíveis: {cols}")


def escolher_melhor_modelo(ativo="PETR4"):
    """Escolhe o melhor modelo para o ativo a partir do CSV de resultados (RMSE_Hibrido)."""
    if not os.path.exists(RESULTADOS_CSV):
        raise FileNotFoundError(f"{RESULTADOS_CSV} não encontrado.")
    df = pd.read_csv(RESULTADOS_CSV)
    df_ativo = df[df["Ativo"].str.contains(ativo, na=False)]
    if df_ativo.empty:
        raise ValueError(f"Nenhum modelo encontrado para {ativo} em {RESULTADOS_CSV}")

    best_row = df_ativo.sort_values("RMSE_Hibrido").iloc[0]
    modelo_tipo = best_row["Modelo"].lower()
    nomefile = ativo.lower()
    model_path = os.path.join(MODEL_DIR, f"{modelo_tipo}_{nomefile}.pt")
    if not os.path.exists(model_path):
        # fallback: try generic name pattern
        alt = os.path.join(MODEL_DIR, f"{modelo_tipo}_{nomefile}.pt")
        model_path = alt  # keep as-is; carregar_modelo_unificado vai falhar com mensagem clara se ausente

    print(f"✔ Melhor modelo para {ativo}: {modelo_tipo.upper()}")
    return modelo_tipo, model_path


def carregar_5_com_padding(csv_path, seq_len):
    """
    Carrega os últimos 5 pregões (linhas) do CSV e garante seq_len+1 linhas adicionando
    padding com business days se necessário, mais 1 linha extra (próximo pregão útil).
    """
    df = pd.read_csv(csv_path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)

    # keep only rows with a Date (defensive)
    df = df[df["Date"].notna()].copy()
    if df.empty:
        raise RuntimeError("CSV vazio ou sem coluna Date válida.")

    df5 = df.tail(5).reset_index(drop=True)

    # if already have seq_len+1 inside last 5, return the slice + extra business day row
    if len(df5) >= seq_len + 1:
        out = df5.tail(seq_len + 1).reset_index(drop=True)
        # extra row: next business day after last
        last_date = pd.to_datetime(out["Date"].iloc[-1])
        next_bd = pd.bdate_range(start=last_date + timedelta(days=1), periods=1)[0]
        extra = out.iloc[[-1]].copy()
        extra["Date"] = next_bd
        return pd.concat([out, extra], ignore_index=True)

    falta = max(0, seq_len - len(df5))
    primeira = df5.iloc[0] if len(df5) > 0 else df.iloc[0]

    pad_rows = []
    # build padding using business days before the first available row in df5
    first_date = pd.to_datetime(primeira["Date"])
    dates_pad = pd.bdate_range(end=first_date - timedelta(days=1), periods=falta)
    # dates_pad are in ascending order (oldest .. newest)
    for d in dates_pad:
        r = primeira.copy()
        r["Date"] = d
        pad_rows.append(r)
    padding = pd.DataFrame(pad_rows) if pad_rows else pd.DataFrame(columns=df.columns)

    df_final = pd.concat([padding, df5], ignore_index=True)

    # adicionar linha extra: próximo pregão útil após última data
    last_date = pd.to_datetime(df_final["Date"].iloc[-1])
    next_bd = pd.bdate_range(start=last_date + timedelta(days=1), periods=1)[0]
    extra = df_final.iloc[[-1]].copy()
    extra["Date"] = next_bd

    df_final = pd.concat([df_final, extra], ignore_index=True).reset_index(drop=True)
    return df_final


def prever_proximos_3(model_path, csv_path, modelo_tipo):
    """
    Retorna (preds_list_of_3, df_pred_full)
    """
    model, scaler, cols, seq_len = carregar_modelo_unificado(model_path, tipo=modelo_tipo)
    df_pad = carregar_5_com_padding(csv_path, seq_len)
    df_pred = prever_unificado(model, scaler, df_pad, seq_len, cols, tipo=modelo_tipo)
    if df_pred.empty:
        raise RuntimeError("Previsão vazia (df_pred). Verifique seq_len/padding.")
    pred_ultimo = float(df_pred["Pred"].iloc[-1])
    return [pred_ultimo, pred_ultimo, pred_ultimo], df_pred


def obter_cluster_de_motivo(motivos, ativo="PETR4"):
    clusters_df = pd.read_csv(CLUSTER_CSV)
    emb_repr = emb_mgr.embed_lote(clusters_df["frase_exemplo"].astype(str).tolist())
    motivo, sim, clust_id, row = motivo_e_cluster_mais_relevante(motivos, emb_mgr, emb_repr, clusters_df, ativo)
    if row is None:
        row = pd.Series(dtype=object)
    return motivo, sim, clust_id, row


def aplicar_residuo_extrapolativo(df_pred_full, num_last=5):
    """
    Média dos resíduos (Real - Pred) nos últimos num_last pontos disponíveis.
    """
    if df_pred_full is None or df_pred_full.empty:
        return 0.0
    df = df_pred_full.dropna(subset=["Real", "Pred"]).copy()
    if df.empty:
        return 0.0
    last = df.tail(num_last)
    residues = (last["Real"] - last["Pred"]).astype(float)
    if len(residues) == 0:
        return 0.0
    return float(residues.mean())


def ajustar_previsao_com_impacto(preds, cluster_row, sim, scale=0.4):
    impactos = []
    for k in range(1, 4):
        col = f"seq_d{k}"
        if isinstance(cluster_row, pd.Series) and (col in cluster_row.index) and pd.notna(cluster_row[col]):
            impactos.append(float(cluster_row[col]) / 100.0)
        else:
            impactos.append(0.0)
    preds_adj = [p * (1 + scale * float(sim) * imp) for p, imp in zip(preds, impactos)]
    return preds_adj, impactos


def construir_datas_e_historico(df, col_close):
    """
    Retorna datas_total (D-4..D0 + D1..D3), precos_hist (5 valores), D0 timestamp.
    Usa bdate_range para futuros D1..D3 e para padding histórico se necessário.
    """
    df_valid = df.dropna(subset=[col_close]).sort_values("Date").reset_index(drop=True)
    if df_valid.empty:
        raise RuntimeError("Dados vazios ou coluna close sem valores.")

    # pegar exatamente os últimos 5 pregões fechados (D-4..D0)
    ultimos5 = df_valid.tail(5).copy()
    # se tiver menos que 5, preencher com bdate_range antes do primeiro disponível
    if len(ultimos5) < 5:
        last_date = df_valid["Date"].max()
        days = pd.bdate_range(end=last_date, periods=5)
        # tentar alinhar preços quando possível; caso contrário NaN
        candidate = df_valid.set_index(pd.to_datetime(df_valid["Date"]).dt.normalize())
        precos = []
        for d in days:
            if d in candidate.index:
                precos.append(float(candidate.loc[d, col_close]))
            else:
                precos.append(np.nan)
        datas_hist = list(days)
        precos_hist = precos
        D0 = datas_hist[-1]
        futuros = pd.bdate_range(start=D0 + timedelta(days=1), periods=3)
        datas_total = datas_hist + list(futuros)
        return datas_total, precos_hist, D0

    datas_hist = list(pd.to_datetime(ultimos5["Date"]).dt.normalize())
    precos_hist = list(ultimos5[col_close].astype(float))
    D0 = datas_hist[-1]

    futuros = pd.bdate_range(start=D0 + timedelta(days=1), periods=3)
    datas_total = datas_hist + list(futuros)
    return datas_total, precos_hist, D0

# ---------------------------
# Plot final (retorna fig)
# ---------------------------
def plotar(datas_total, precos_hist, preds_base_full, preds_adj_full, impactos, show=True):
    """
    datas_total: [D-4..D0, D1, D2, D3]
    precos_hist: 5 valores (D-4..D0)
    preds_base_full: 4 valores [D0, D1, D2, D3]
    preds_adj_full: 4 valores [D0, D1, D2, D3]
    impactos: 3 valores (D1..D3)
    """
    labels = [pd.to_datetime(d).strftime("%d/%m/%Y") for d in datas_total]
    n_hist = len(precos_hist)
    labels_focus = labels[-4:]  # D0..D3

    fig = go.Figure()

    # HISTÓRICO (D-4..D0)
    fig.add_trace(go.Scatter(
        x=labels[:n_hist],
        y=precos_hist,
        mode="lines+markers",
        name="Histórico (última semana)",
        line=dict(color="gray", width=2),
        marker=dict(size=6),
        hoverinfo="x+y"
    ))

    # PREVISÃO BASE (D0..D3)
    fig.add_trace(go.Scatter(
        x=labels_focus,
        y=preds_base_full,
        mode="lines+markers",
        name="Previsão Base",
        line=dict(color="blue", width=3),
        marker=dict(size=8),
        hoverinfo="x+y"
    ))

    # PREVISÃO AJUSTADA (D0..D3)
    fig.add_trace(go.Scatter(
        x=labels_focus,
        y=preds_adj_full,
        mode="lines+markers",
        name="Previsão Ajustada",
        line=dict(color="red", width=3),
        marker=dict(size=8),
        hoverinfo="x+y"
    ))

    # BARRAS IMPACTO (%)
    labels_impact = labels_focus[1:]  # D1, D2, D3
    fig.add_trace(go.Bar(
        x=labels_impact,
        y=[i * 100 for i in impactos],
        name="Impacto (%)",
        marker_color=["green" if i >= 0 else "red" for i in impactos],
        opacity=0.45,
        yaxis="y2",
        hoverinfo="x+y"
    ))

    # ============
    # AJUSTE VISUAL
    # ============
    fig.update_layout(
        title="Comparação Previsão Base vs Ajustada (Impacto das notícias)",
        xaxis=dict(title="Datas"),
        yaxis=dict(title="Preço (R$)"),
        yaxis2=dict(title="Impacto (%)", overlaying="y", side="right", showgrid=False),

        # 🟦 LEGENDA COMPLETAMENTE FORA DO GRÁFICO
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,                 # deslocado para fora
            bgcolor="rgba(255,255,255,0.7)"
        ),

        # Maior margem direita para caber a legenda
        margin=dict(l=60, r=180, t=80, b=80),

        template="plotly_white",
        height=700
    )

    fig.update_xaxes(tickangle=-20)

    if show:
        fig.show()

    return fig


# ---------------------------
# EXECUÇÃO MVP (agora parametrizável)
# ---------------------------
def executar_demo(retornar_html=False, ativo="PETR4", csv_path=None, show_plot=True):
    """
    executar_demo:
      - retornar_html: se True, salva HTML e retorna caminho
      - ativo: "PETR4", "PRIO3", ...
      - csv_path: caminho do CSV do ativo (se None, usa default PETR4)
      - show_plot: se True, chama fig.show()
    Retorna: None (se retornar_html False) ou caminho para HTML (str) se True.
    """
    if csv_path is None:
        csv_path = DEFAULT_CSV_PETR4

    print("\n====================================")
    print("       🚀 TimesSeriesAgent – MVP")
    print("====================================\n")

    # 1) escolher melhor modelo
    modelo_tipo, model_path = escolher_melhor_modelo(ativo)

    # 2) previsões base (retorna preds e df_pred_full)
    preds, df_pred_full = prever_proximos_3(model_path, csv_path, modelo_tipo)
    print("Previsão base (raw):", preds)

    # 3) calcular resíduo médio dos últimos N pontos e usar para extrapolar
    mean_resid = aplicar_residuo_extrapolativo(df_pred_full, num_last=5)
    print(f"Média do resíduo (Real - Pred) nos últimos pontos: {mean_resid:.6f}")

    # extrapolação simples (D1 gets +1*resid, D2 +2*resid ...)
    preds_with_resid = [float(p + mean_resid * (i + 1)) for i, p in enumerate(preds)]
    print("Previsões com resíduo aplicado (D1..D3):", preds_with_resid)

    # 4) cluster/motivos e impacto
    motivos = [
        "crise politica no país",
    ]
    motivo_sel, sim, cluster_id, row = obter_cluster_de_motivo(motivos, ativo)
    preds_adj, impactos = ajustar_previsao_com_impacto(preds_with_resid, row, sim, scale=0.4)

    print("Motivo selecionado:", motivo_sel)
    print("Similaridade:", sim)
    print("Impactos (D1..D3):", impactos)
    print("Previsão ajustada (D1..D3):", preds_adj)

    # 5) construir datas e pegar histórico
    df = pd.read_csv(csv_path, parse_dates=["Date"]).sort_values("Date")
    col_close = detectar_coluna_close(df, ativo)
    datas_total, precos_hist, D0 = construir_datas_e_historico(df, col_close)

    # garantir precos_hist tem 5 elementos (se faltar, preencher à esquerda com NaN)
    if len(precos_hist) < 5:
        last5 = df[col_close].dropna().tail(5).astype(float).tolist()
        while len(last5) < 5:
            last5.insert(0, np.nan)
        precos_hist = last5

    # 6) montar arrays para plot: D0 + D1..D3
    ultimo_preco_real = float(precos_hist[-1])
    preds_base_full = [ultimo_preco_real] + [float(x) for x in preds_with_resid]
    preds_adj_full = [ultimo_preco_real] + [float(x) for x in preds_adj]

    # 7) plot (retorna fig)
    fig = plotar(datas_total, precos_hist, preds_base_full, preds_adj_full, impactos, show=show_plot)

    # 8) salvar HTML se pedido
    if retornar_html:
        caminho_html = os.path.join(IMG_DIR, f"mvp_online_{ativo.lower()}.html")
        fig.write_html(caminho_html)
        print(f"✔ HTML salvo em: {caminho_html}")
        return {"html": caminho_html, "fig": fig}

    print("✔ MVP final executado.")
    return {"fig": fig}


if __name__ == "__main__":
    # execução local de teste
    executar_demo(retornar_html=True, ativo="PETR4", csv_path=DEFAULT_CSV_PETR4, show_plot=False)
