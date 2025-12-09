import numpy as np
import pandas as pd


# ===============================================================
# 1) Similaridade (correto)
# ===============================================================
def calc_sim(emb, emb_repr):
    """
    emb: vetor (1, dim)
    emb_repr: matriz (N, dim)
    retorna array (N,)
    """
    emb = np.asarray(emb).reshape(1, -1)
    M = np.asarray(emb_repr)

    if M.ndim == 1:
        M = M.reshape(1, -1)

    num = emb @ M.T
    denom = np.linalg.norm(emb, axis=1, keepdims=True) * np.linalg.norm(M, axis=1) + 1e-12
    return (num / denom).flatten()


# ===============================================================
# 2) Alinhar motivos à timeline
# ===============================================================
def alinhar_motivos_pred_dates(pred_dates, motivos_dict):
    dates_arr = np.asarray(pred_dates).astype("datetime64[D]")
    date_to_pos = {d: i for i, d in enumerate(dates_arr)}
    out = {}

    for d, motivos in motivos_dict.items():
        try:
            dnorm = np.datetime64(pd.to_datetime(d).normalize(), 'D')
        except:
            dnorm = np.datetime64(d, 'D')

        pos = date_to_pos.get(dnorm)
        if pos is not None:
            out.setdefault(pos, []).extend(motivos)

    return out


# ===============================================================
# 3) Selecionar cluster mais relevante POR DIA
# ===============================================================
def escolher_cluster_do_dia(motivos, emb_mgr, emb_repr, clusters_df, ativo,
                            sim_threshold=0.7, max_horizon=4):
    """
    Retorna:
      cluster_id, seq (array float), best_sim
      ou (-1, None, 0.0) se nada for relevante
    """

    if not motivos:
        return -1, None, 0.0

    emb_repr_np = np.asarray(emb_repr)
    if emb_repr_np.ndim == 1:
        emb_repr_np = emb_repr_np.reshape(1, -1)

    # ---------------------------------------------------------
    # Filtragem por ativo
    # ---------------------------------------------------------
    ativo = (ativo or "").upper()
    ativos_validos = [ativo, "BRENT", "GENÉRICO"]

    if "ativo_cluster" in clusters_df.columns:
        mask = clusters_df["ativo_cluster"].astype(str).str.upper().isin(ativos_validos)
        df_f = clusters_df[mask].reset_index(drop=True)
        emb_f = emb_repr_np[mask.values]
    else:
        df_f = clusters_df.copy()
        emb_f = emb_repr_np.copy()

    if len(df_f) == 0:
        return -1, None, 0.0

    # ---------------------------------------------------------
    # Pesos por origem
    # ---------------------------------------------------------
    peso_map = {
        ativo: 1.0,
        "BRENT": 0.75,
        "GENÉRICO": 0.5
    }

    best_sim = -1
    best_cluster_id = -1
    best_seq = None

    # ---------------------------------------------------------
    # Avalia todos motivos, pega o cluster mais forte
    # ---------------------------------------------------------
    for mot in motivos:
        emb_m = emb_mgr.embed(mot)
        sims = calc_sim(emb_m, emb_f)
        if sims.size == 0:
            continue

        for i, s in enumerate(sims):
            tipo = str(df_f.iloc[i]["ativo_cluster"]).upper()
            peso = peso_map.get(tipo, 0.5)
            sim_p = float(s) * peso       # similaridade ponderada

            if sim_p > best_sim and sim_p >= sim_threshold:
                row = df_f.iloc[i]

                seq_vals = []
                for h in range(max_horizon + 1):
                    col = f"seq_d{h}"
                    if col in row and pd.notna(row[col]):
                        seq_vals.append(float(row[col]) / 100.0)
                    else:
                        break

                if len(seq_vals) == 0:
                    continue

                best_sim = sim_p
                best_seq = np.array(seq_vals, float)
                best_cluster_id = int(row["cluster"]) if "cluster" in row else i

    return best_cluster_id, best_seq, best_sim


# ===============================================================
# 4) Matriz de impactos cumulativos
# ===============================================================
def build_impacts_matrix_returns(pred_dates, motivos_dict, emb_mgr, clusters_df,
                                 emb_repr, ativo, max_horizon=4, sim_threshold=0.7):
    """
    Retorna:
      event_positions (E,)
      impacts_cumsum  (E,N)
      event_dates     (E,)
      impacts_raw     (E,N)
      event_cluster_ids (E,)
    """

    N = len(pred_dates)
    pred_dates_arr = np.asarray(pred_dates).astype("datetime64[D]")

    motivos_pos = alinhar_motivos_pred_dates(pred_dates_arr, motivos_dict)
    if not motivos_pos:
        return (
            np.array([], int),
            np.zeros((0, N)),
            np.array([], 'datetime64[D]'),
            np.zeros((0, N)),
            np.array([], int)
        )

    # ordenar eventos por posição na timeline
    eventos = sorted(list(motivos_pos.items()), key=lambda x: x[0])

    E = len(eventos)
    impacts_raw = np.zeros((E, N), float)
    cluster_ids = []

    # datas futuras de eventos (para cortar impacto)
    future_dates = [pred_dates_arr[e[0]] for e in eventos]

    for j, (pos, motivos_dia) in enumerate(eventos):

        cluster_id, seq_vals, sim_val = escolher_cluster_do_dia(
            motivos_dia, emb_mgr, emb_repr, clusters_df,
            ativo, sim_threshold, max_horizon
        )

        cluster_ids.append(cluster_id)

        if seq_vals is None:
            continue

        # aplicar seq vals D0..Dh, interrompendo se houver outro evento
        for h, impact_h in enumerate(seq_vals):
            idx = pos + h
            if idx >= N:
                break

            # se houver outro evento entre pos e idx (exclusive/inclusive), parar
            for dnext in future_dates:
                if dnext > pred_dates_arr[pos] and dnext <= pred_dates_arr[idx]:
                    idx = None
                    break
            if idx is None:
                break

            impacts_raw[j, pos + h] = impact_h * sim_val

    # cumulativo por evento
    impacts_cumsum = np.cumsum(impacts_raw, axis=1)

    return (
        np.array([e[0] for e in eventos], int),
        impacts_cumsum,
        np.array([pred_dates_arr[e[0]] for e in eventos], 'datetime64[D]'),
        impacts_raw,
        np.array(cluster_ids, int)
    )


# ===============================================================
# 5) Aplicar impactos aos preços
# ===============================================================
def apply_impacts_to_returns(pred_prices, impacts_cumsum, event_positions, scale=0.4):
    N = len(pred_prices)
    pred_prices = np.asarray(pred_prices, float)

    # cálculo dos retornos originais
    pred_returns = np.zeros(N)
    if N > 1:
        pred_returns[1:] = pred_prices[1:] / pred_prices[:-1] - 1.0

    # impacto total no tempo
    total_impact = np.zeros(N)

    if event_positions.size > 0:
        order = np.argsort(event_positions)
        ev_sorted = event_positions[order]
        imp_sorted = impacts_cumsum[order]

        counts = np.searchsorted(ev_sorted, np.arange(N), side='right')
        mask = counts > 0

        # CORREÇÃO AQUI:
        cols = np.arange(N, dtype=int)

        total_impact[mask] = imp_sorted[counts[mask] - 1, cols[mask]]

    # aplicar escala
    pred_returns_adj = pred_returns + scale * total_impact

    # reconstruir preços
    p_adj = np.zeros_like(pred_prices)
    p_adj[0] = pred_prices[0]
    for t in range(1, N):
        p_adj[t] = p_adj[t - 1] * (1 + pred_returns_adj[t])

    return pred_returns_adj, p_adj


# ===============================================================
# 6) Ridge analítico
# ===============================================================
def ridge_closed_form(X, y, alpha=1.0):
    XT_X = X.T @ X
    A = XT_X + alpha * np.eye(XT_X.shape[0])
    XT_y = X.T @ y

    try:
        beta = np.linalg.solve(A, XT_y)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(A, XT_y, rcond=None)[0]

    return beta
