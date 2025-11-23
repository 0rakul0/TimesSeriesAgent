"""
hibrido.py
- calcular_pesos_por_noticia(): transforma motivos -> pesos por data usando buscar_similares
- aplicar_correcao_d_plus_1(): aplica ajuste multiplicativo D+1 sobre Pred (não cumulativo),
    com meia_vida exponencial e propagação limitada
- função é compatível tanto para modelos absolutos quanto para modelos residuais (aplica sobre Pred fornecido)
"""

import numpy as np
import pandas as pd
from datetime import timedelta
from .noticias import buscar_similares, carregar_base_frases, extrair_motivos_ultimos_dias

# ------------------------
# Calcular pesos por data
# ------------------------
def calcular_pesos_por_noticia(noticias_dict: dict, meta_df: pd.DataFrame, emb_matrix: np.ndarray,
                               top_k: int = 8, sim_threshold: float = 0.55):
    """
    noticias_dict: {date: [motivo1, motivo2, ...], ...}
    meta_df, emb_matrix: base de motivos e embeddings (carregue com carregar_base_frases)
    Retorna: {date: peso_medio}
    """
    pesos_por_data = {}
    for data, motivos in noticias_dict.items():
        pesos = []
        for motivo in motivos:
            sims = buscar_similares(motivo, meta_df, emb_matrix, top_k=top_k, sim_threshold=sim_threshold)
            if not sims:
                continue
            numer = sum(p * s for (_, p, s) in sims)
            denom = sum(s for (_, _, s) in sims)
            if denom > 0:
                peso_medio = numer / denom
                pesos.append(peso_medio)
        if pesos:
            pesos_por_data[data] = float(np.mean(pesos))
    return pesos_por_data


# ------------------------
# Aplicar correção D+1 (global ou janela)
# ------------------------
def aplicar_correcao_d_plus_1(
    pred_df: pd.DataFrame,
    pesos_por_data: dict,
    motivos_por_data: dict = None,
    janela_test: int = 30,
    meia_vida: float = 4.0,
    dias_propagacao: int = 3
):
    """
    pred_df: DataFrame index=Date com colunas ['Pred','Real']
    pesos_por_data: {date_evento: peso}
    motivos_por_data: optional dict {date:[motivos]}
    janela_test: aplica correção apenas nos últimos N dias (se = len(pred_df) aplica em toda a série)
    meia_vida: decaimento exponencial
    dias_propagacao: número de dias após o evento que ainda são impactados
    RETORNA: pred_df ajustado (coluna Pred_Ajustado)
    """

    df = pred_df.copy()
    df["Pred_Ajustado"] = df["Pred"].copy()

    dias_uteis = df.index
    # define índices alvo
    if janela_test >= len(dias_uteis):
        test_index = dias_uteis
    else:
        test_index = dias_uteis[-janela_test:]

    # ordenar eventos por data
    noticias_ord = sorted(pesos_por_data.items(), key=lambda x: pd.to_datetime(x[0]))

    # mapear cada evento para o pregão mais próximo (ajusta fim de semana)
    eventos_processados = []
    for data_raw, peso_raw in noticias_ord:
        dt = pd.to_datetime(data_raw)
        pos = dias_uteis.searchsorted(dt)
        if pos >= len(dias_uteis):
            continue
        data_evento = dias_uteis[pos]
        peso = np.clip(float(peso_raw), -1.0, 1.0)
        eventos_processados.append((data_evento, peso))

    # criar mapa dia -> (evento_data, peso, k)
    mapa = {}
    for data_evento, peso in eventos_processados:
        idx_evento = dias_uteis.get_loc(data_evento)
        for k in range(dias_propagacao + 1):
            idx_k = idx_evento + k
            if idx_k >= len(dias_uteis):
                break
            dia_k = dias_uteis[idx_k]
            if dia_k not in test_index:
                continue
            # sobrescreve se houver evento mais recente — isso implementa reset por nova notícia
            mapa[dia_k] = (data_evento, peso, k)

    # aplicar ajustes (não cumulativos)
    for dia, (data_evento, peso, k) in mapa.items():
        alpha_k = float(np.exp(-k / meia_vida))
        # normaliza peso — observação: seu pipeline original dividia por 100, mantive forma porém coerente:
        peso_suave = float(np.tanh((peso / 100.0) * alpha_k))
        # multiplicativo (mantive sua lógica multiplicativa)
        df.loc[dia, "Pred_Ajustado"] = df.loc[dia, "Pred"] * (1.0 + peso_suave)

        # imprimir motivo apenas no D0 (opcional)
        motivo_txt = motivos_por_data.get(pd.to_datetime(data_evento), []) if motivos_por_data is not None else []
        if k == 0:
            print(f"[AJUSTE] Evento {data_evento.date()} -> dia {dia.date()} | k={k} | peso={peso:.4f} | suave={peso_suave:.6f} | motivos={motivo_txt}")
        else:
            print(f"[AJUSTE] Evento {data_evento.date()} -> dia {dia.date()} | propagação k={k} | suave={peso_suave:.6f}")

    return df


# ------------------------
# utilitário: aplicar fluxo completo (carrega meta/emb se necessário)
# ------------------------
def aplicar_hibrido_fluxo(pred_df, pasta_noticias="../output_noticias", csv_frases="../data/frases_impacto_clusters.csv", emb_path="../data/embeddings_frases.npy", janela_test=None):
    """
    Conveniência: carrega meta/emb, extrai motivos de pasta_noticias para o período do pred_df,
    calcula pesos e aplica correção D+1.
    janela_test: se None -> len(pred_df) (aplica em toda série)
    Retorna: pred_corr_df, pesos_por_data
    """
    meta_df, emb_matrix = carregar_base_frases(csv_frases, emb_path)
    # definir janela de extração = tamanho da série
    if janela_test is None:
        janela_test = len(pred_df)
    # extrair noticias para todo histórico disponível (baseia-se na última data pred_df)
    dias_total = (pred_df.index[-1] - pred_df.index[0]).days
    motivos = extrair_motivos_ultimos_dias(pasta_noticias, dias_total, ref_date=pred_df.index[-1])
    pesos = calcular_pesos_por_noticia(motivos, meta_df, emb_matrix)
    pred_corr = aplicar_correcao_d_plus_1(pred_df, pesos_por_data=pesos, motivos_por_data=motivos, janela_test=janela_test)
    return pred_corr, pesos
