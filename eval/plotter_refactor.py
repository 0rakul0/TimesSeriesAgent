import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import mean_squared_error  # import já no topo


# =====================================================================
# UTILIDADES DE SIMILARIDADE / CLUSTERS
# =====================================================================

def calc_sim(emb, emb_repr):
    """
    emb: vetor 1D ou 2D (1,x)
    emb_repr: matriz (N, dim)
    retorna array (N,)
    """
    emb = np.asarray(emb).reshape(1, -1)
    M = np.asarray(emb_repr)
    if M.ndim == 1:
        M = M.reshape(1, -1)
    num = emb @ M.T
    denom = (np.linalg.norm(emb, axis=1, keepdims=True) * np.linalg.norm(M, axis=1) + 1e-12)
    return (num / denom).flatten()


def alinhar_motivos_por_posicoes(index_dates, motivos_por_data):
    dates = np.asarray(pd.to_datetime(index_dates).normalize()).astype("datetime64[D]")
    date_to_pos = {d: i for i, d in enumerate(dates)}
    out = {}
    for d, motivos in motivos_por_data.items():
        dnorm = np.datetime64(pd.to_datetime(d).normalize(), "D")
        pos = date_to_pos.get(dnorm, None)
        if pos is not None:
            out.setdefault(pos, []).extend(motivos)
    return out


def escolher_melhor_cluster_por_dia(motivos, emb_mgr, emb_repr, clusters_df):
    emb_repr_mat = np.asarray(emb_repr)
    if emb_repr_mat.ndim == 1:
        emb_repr_mat = emb_repr_mat.reshape(1, -1)
    best_sim = 0.0
    best_idx = -1
    best_mot = None
    for mot in motivos:
        emb = emb_mgr.embed(mot)
        if emb is None:
            continue
        sims = calc_sim(emb, emb_repr_mat)
        if sims.size == 0:
            continue
        k = int(np.argmax(sims))
        s = float(sims[k])
        if s > best_sim:
            best_sim = s
            best_idx = k
            best_mot = mot
    if best_idx >= 0 and best_idx < len(clusters_df):
        row = clusters_df.iloc[best_idx]
        return row, best_sim, best_mot
    return None, 0.0, None


# =====================================================================
# GRÁFICO HÍBRIDO ORIGINAL (mantido)
# =====================================================================
def plotar_hibrido_corrigido(
        df,
        motivos_por_data,
        emb_mgr,
        clusters_df,
        emb_repr,
        out_path,
        nome="Híbrido",
        top_k=20):

    import plotly.graph_objects as go
    import plotly.express as px
    from sklearn.metrics import mean_squared_error
    import numpy as np
    import pandas as pd

    df = df.copy().sort_index()
    N = len(df)
    if N == 0:
        raise ValueError("DataFrame vazio para plotagem.")

    # =============================================================
    # POSICIONAR MOTIVOS NO EIXO
    # =============================================================
    motivos_pos = alinhar_motivos_por_posicoes(df.index, motivos_por_data)
    event_positions = np.array(sorted(motivos_pos.keys()), dtype=int) if motivos_pos else np.array([], dtype=int)

    # =============================================================
    # DELTA (RETORNO) SEQUÊNCIA APLICADA
    # =============================================================
    if "Pred_Return" in df.columns and "Pred_Return_Ajust" in df.columns:
        delta_ret = (df["Pred_Return_Ajust"] - df["Pred_Return"]).values
    else:
        delta_ret = np.zeros(N)

    evt_clusters = []
    evt_sims = []
    evt_best_rows = []
    evt_best_motivos = []
    evt_seq_aplicada = []

    # =============================================================
    # IDENTIFICAR CLUSTER + MELHOR MOTIVO POR EVENTO
    # =============================================================
    for pos in event_positions:
        motivos = motivos_pos.get(int(pos), [])
        if not motivos:
            evt_clusters.append(None)
            evt_sims.append(0)
            evt_best_rows.append(None)
            evt_best_motivos.append(None)
            evt_seq_aplicada.append(np.zeros(N))
            continue

        row, sim, mot_sel = escolher_melhor_cluster_por_dia(
            motivos, emb_mgr, emb_repr, clusters_df
        )

        if row is None:
            evt_clusters.append(None)
            evt_sims.append(0)
            evt_best_rows.append(None)
            evt_best_motivos.append(None)
            evt_seq_aplicada.append(np.zeros(N))
            continue

        # ID do cluster
        try:
            cid = int(row["cluster"])
        except:
            cid = int(row.name)

        evt_clusters.append(cid)
        evt_sims.append(sim)
        evt_best_rows.append(row)
        evt_best_motivos.append(mot_sel)

        # Construir sequência aplicada
        seq_ap = np.zeros(N)
        seq_vals = []
        seq_len = 0

        for k in range(10):
            col = f"seq_d{k}"
            if col in row and pd.notna(row[col]):
                seq_vals.append(float(row[col]) / 100.0)
                seq_len += 1
            else:
                break

        future = [p for p in event_positions if p > pos]
        next_pos = future[0] if future else N
        slice_end = min(next_pos, pos + seq_len)

        seq_obs = delta_ret[pos:slice_end]
        seq_ap[pos:slice_end] = seq_obs
        evt_seq_aplicada.append(seq_ap)

    # =============================================================
    # MAPA DE CORES PARA CLUSTERS
    # =============================================================
    palette = px.colors.qualitative.Set3
    unique_clusters = [c for c in evt_clusters if c is not None]
    cluster_color_map = {cid: palette[i % len(palette)] for i, cid in enumerate(sorted(set(unique_clusters)))}

    # =============================================================
    # INICIAR FIGURA
    # =============================================================
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df.index, y=df["Real"], name="Real",
                             line=dict(color="black", width=2)))

    fig.add_trace(go.Scatter(x=df.index, y=df["Pred"], name="Base (Pred)",
                             line=dict(color="gray", width=2, dash="dash")))

    if "Pred_Final_Price" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["Pred_Final_Price"],
                                 name="Híbrido (final)",
                                 line=dict(color="blue", width=2)))

    # =============================================================
    # ESTRELA NO D0
    # =============================================================
    star_x, star_y = [], []
    for i, pos in enumerate(event_positions):
        if evt_seq_aplicada[i][pos] != 0:
            star_x.append(df.index[pos])
            yv = df["Pred_Final_Price"].iloc[pos] if "Pred_Final_Price" in df.columns else df["Pred"].iloc[pos]
            star_y.append(yv)

    if len(star_x):
        fig.add_trace(go.Scatter(
            x=star_x, y=star_y,
            mode="markers+text",
            text=["*"] * len(star_x),
            textposition="top center",
            marker=dict(size=14, color="black"),
            name="D0 (*)"
        ))

    # =============================================================
    # GERAR BLOCOS DE EVENTOS
    # =============================================================
    blocks = []
    if len(event_positions):
        last = evt_clusters[0]
        start = event_positions[0]
        ev_idx = 0

        for i in range(1, len(event_positions)):
            cid = evt_clusters[i]
            pos = event_positions[i]

            if cid != last:
                blocks.append((start, event_positions[i] - 1, last, ev_idx))
                start = pos
                last = cid
                ev_idx = i

        blocks.append((start, min(start + 4, N - 1), last, ev_idx))

    # =============================================================
    # DESENHAR BLOCOS + MARCADORES (AGRUPADOS NA LEGENDA)
    # =============================================================
    clusters_already_in_legend = set()

    for (bstart, bend, cid, ev_idx) in blocks:

        if cid is None:
            continue

        row = evt_best_rows[ev_idx]
        motivo = evt_best_motivos[ev_idx]
        sim_val = evt_sims[ev_idx]

        if row is None:
            continue

        frase = row["frase_exemplo"] if "frase_exemplo" in row else f"Cluster {cid}"
        border = cluster_color_map.get(cid, "blue")

        # Legenda agrupada
        show_legend = cid not in clusters_already_in_legend
        if show_legend:
            clusters_already_in_legend.add(cid)

        legend_name = f"Cluster {cid} — {frase}"

        # Fundo do bloco
        fig.add_shape(
            type="rect",
            x0=df.index[bstart],
            x1=df.index[bend],
            y0=df["Real"].min(),
            y1=df["Real"].max(),
            fillcolor="rgba(0,116,217,0.12)" if row["seq_d0"] > 0 else "rgba(222,45,38,0.12)",
            opacity=0.25,
            layer="below",
            line=dict(color=border, width=2)
        )

        # Marcador inicial do bloco
        fig.add_trace(go.Scatter(
            x=[df.index[bstart]],
            y=[df["Real"].iloc[bstart]],
            mode="markers",
            marker=dict(size=10, color=border),
            name=legend_name,
            legendgroup=f"cluster_{cid}",
            showlegend=show_legend,
            hovertemplate=f"<b>Cluster {cid}</b><br>{frase}<br>Motivo: {motivo}<extra></extra>"
        ))

        # ====================================================
        # MARCADORES DA SEQUÊNCIA
        # ====================================================
        seq_vals = []
        for k in range(10):
            col = f"seq_d{k}"
            if col in row and pd.notna(row[col]):
                seq_vals.append((k, float(row[col]) / 100.0))
            else:
                break

        seq_x, seq_y, seq_text = [], [], []

        for (k, val) in seq_vals:
            pos_idx = bstart + k
            if pos_idx >= len(df):
                break

            seq_x.append(df.index[pos_idx])
            yv = df["Pred_Final_Price"].iloc[pos_idx] if "Pred_Final_Price" in df.columns else df["Pred"].iloc[pos_idx]
            seq_y.append(yv)

            impacto_pct = val * 100

            seq_text.append(
                f"<b>Motivo:</b> {motivo}<br>"
                f"<b>Cluster:</b> {cid} — {frase}<br>"
                f"<b>Step:</b> D{k}<br>"
                f"<b>Impacto original:</b> {impacto_pct:+.2f}%<br>"
                f"<b>Similaridade:</b> {sim_val:.3f}<br>"
                f"<b>Data:</b> {df.index[pos_idx].date()}"
            )

        if seq_x:
            fig.add_trace(go.Scatter(
                x=seq_x,
                y=seq_y,
                mode="markers+lines",
                marker=dict(size=7, color=border),
                line=dict(color=border, width=1),
                name=legend_name,
                legendgroup=f"cluster_{cid}",
                showlegend=False,
                hovertemplate="%{text}<extra></extra>",
                text=seq_text
            ))

    # =============================================================
    # MÉTRICAS
    # =============================================================
    final_price_col = "Pred_Final_Price" if "Pred_Final_Price" in df.columns else \
                      ("Pred_Ajustado" if "Pred_Ajustado" in df.columns else "Pred")

    rmse_b = np.sqrt(mean_squared_error(df["Real"], df["Pred"]))
    rmse_h = np.sqrt(mean_squared_error(df["Real"], df[final_price_col]))

    fig.update_layout(
        title=f"{nome} | RMSE Base={rmse_b:.4f} | RMSE Híbrido={rmse_h:.4f}",
        template="plotly_white",
        height=900
    )

    fig.write_html(out_path)
    print(f"Plot salvo em: {out_path}")

    return rmse_b, rmse_h


# =====================================================================
# GRÁFICOS SIMPLES – modelos puros e híbridos
# =====================================================================

def plotar_modelos_puros(prev_dict, out_path):
    """
    prev_dict: { nome_modelo: df[['Pred','Real']] }
    """
    fig = go.Figure()

    first_df = list(prev_dict.values())[0]
    fig.add_trace(go.Scatter(
        x=first_df.index,
        y=first_df["Real"],
        name="Real",
        line=dict(color="black", width=3)
    ))

    palette = px.colors.qualitative.Set1

    for i, (nome, dfp) in enumerate(prev_dict.items()):
        fig.add_trace(go.Scatter(
            x=dfp.index,
            y=dfp["Pred"],
            name=f"{nome} - Pred",
            line=dict(color=palette[i % len(palette)], width=2)
        ))

    fig.update_layout(
        title="Comparação dos Modelos PUROS (LSTM / AE / Transformer)",
        xaxis_title="Data",
        yaxis_title="Preço",
        height=700,
        template="plotly_white",
        legend=dict(itemsizing="constant")
    )

    fig.write_html(out_path)
    print(f"Gráfico de modelos puros salvo em: {out_path}")


def plotar_modelos_hibridos(prev_dict, out_path):
    """
    prev_dict: { nome_modelo: df[['Pred_Ajustado','Real']] }
    """
    fig = go.Figure()

    first_df = list(prev_dict.values())[0]
    fig.add_trace(go.Scatter(
        x=first_df.index,
        y=first_df["Real"],
        name="Real",
        line=dict(color="black", width=3)
    ))

    palette = px.colors.qualitative.Set2

    for i, (nome, dfp) in enumerate(prev_dict.items()):
        fig.add_trace(go.Scatter(
            x=dfp.index,
            y=dfp["Pred_Ajustado"],
            name=f"{nome} - Híbrido",
            line=dict(color=palette[i % len(palette)], width=2)
        ))

    fig.update_layout(
        title="Comparação dos Modelos HÍBRIDOS (LSTM / AE / Transformer)",
        xaxis_title="Data",
        yaxis_title="Preço",
        height=700,
        template="plotly_white",
        legend=dict(itemsizing="constant")
    )

    fig.write_html(out_path)
    print(f"Gráfico de modelos híbridos salvo em: {out_path}")


# =====================================================================
# NOVO — GRÁFICO FINAL COMPLETO POR ATIVO (o que você pediu)
# =====================================================================
def plotar_comparacao_por_ativo(df_base,
                                modelos_puros: dict,
                                modelos_hibridos: dict,
                                motivos_por_data,
                                emb_mgr,
                                clusters_df,
                                emb_repr,
                                out_path,
                                ativo):

    fig = go.Figure()

    # ============================================================
    # 1) Plotar série real
    # ============================================================
    df_real = df_base.copy()
    fig.add_trace(go.Scatter(
        x=df_real.index,
        y=df_real["Real"],
        name=f"{ativo} - Real",
        line=dict(color="black", width=3)
    ))

    # ============================================================
    # 2) Plotar modelos puros
    # ============================================================
    palette_puro = px.colors.qualitative.Set1

    for i, (nome_modelo, df) in enumerate(modelos_puros.items()):
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["Pred"],
            name=f"{nome_modelo} (puro)",
            line=dict(color=palette_puro[i % len(palette_puro)], width=2, dash="dot")
        ))

    # ============================================================
    # 3) Plotar modelos híbridos
    # ============================================================
    palette_hibrido = px.colors.qualitative.Set2

    for i, (nome_modelo, df) in enumerate(modelos_hibridos.items()):
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["Pred_Ajustado"],
            name=f"{nome_modelo} (híbrido)",
            line=dict(color=palette_hibrido[i % len(palette_hibrido)], width=2)
        ))

    # ============================================================
    # 4) Plotar blocos e eventos (reaproveitando sua lógica)
    # ============================================================

    motivos_pos = alinhar_motivos_por_posicoes(df_real.index, motivos_por_data)
    event_positions = np.array(sorted(motivos_pos.keys()), dtype=int) if motivos_pos else []

    emb_repr_mat = np.asarray(emb_repr)
    if emb_repr_mat.ndim == 1:
        emb_repr_mat = emb_repr_mat.reshape(1, -1)

    N = len(df_real)

    evt_clusters = []
    evt_best_rows = []

    for pos in event_positions:
        motivos = motivos_pos.get(int(pos), [])
        row, sim, mot_sel = escolher_melhor_cluster_por_dia(
            motivos, emb_mgr, emb_repr, clusters_df
        )
        if row is None:
            evt_clusters.append(None)
            evt_best_rows.append(None)
        else:
            cid = int(row["cluster"]) if "cluster" in row.index else row.name
            evt_clusters.append(cid)
            evt_best_rows.append(row)

    # blocos por clusters consecutivos
    blocks = []
    if len(event_positions):
        last_cid = evt_clusters[0]
        start = event_positions[0]
        ev_idx = 0

        for i in range(1, len(event_positions)):
            cid = evt_clusters[i]
            pos = event_positions[i]

            if cid != last_cid:
                blocks.append((start, event_positions[i] - 1, last_cid, ev_idx))
                start = pos
                last_cid = cid
                ev_idx = i

        blocks.append((start, min(start + 4, N - 1), last_cid, ev_idx))

    # desenhar blocos
    for (s, e, cid, ev_idx) in blocks:

        if cid is None:
            continue

        if cid not in clusters_df.index:
            continue

        row = clusters_df.loc[cid]
        seq0 = float(row["seq_d0"]) if "seq_d0" in row and pd.notna(row["seq_d0"]) else 0

        fill = "rgba(0,116,217,0.12)" if seq0 > 0 else "rgba(222,45,38,0.12)"

        fig.add_shape(
            type="rect",
            x0=df_real.index[s],
            x1=df_real.index[e],
            y0=df_real["Real"].min(),
            y1=df_real["Real"].max(),
            fillcolor=fill,
            opacity=0.25,
            layer="below",
            line=dict(color="rgba(0,0,0,0)")
        )

        # marcador do início
        fig.add_trace(go.Scatter(
            x=[df_real.index[s]],
            y=[df_real["Real"].iloc[s]],
            name=f"Cluster {cid} (evento)",
            mode="markers",
            marker=dict(size=10, color="red"),
            showlegend=False
        ))

    # ============================================================
    # 5) Layout final
    # ============================================================
    fig.update_layout(
        title=f"Comparação Completa - {ativo}<br>Modelos Puros + Híbridos + Eventos",
        height=900,
        template="plotly_white",
        legend=dict(traceorder="normal")
    )

    fig.write_html(out_path)
    print(f"[OK] Comparação completa salva em {out_path}")
