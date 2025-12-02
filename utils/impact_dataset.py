import os
import numpy as np
import pandas as pd

def gerar_dataset_eventos(eventos_dir, df_prices, emb_mgr,
                          ativo="PETR4",
                          janela_atras=5,
                          horizon=4):
    """
    Gera dataset supervisionado para treinar o Impact Predictor.

    Entradas:
      - eventos_dir: pasta com evento_*.json
      - df_prices: DataFrame de preços com coluna Date index/col e col Close_<ativo>
      - emb_mgr: instancia de EmbeddingManager (com método embed que retorna vetor 1xD)
      - ativo: string (ex: "PETR4" ou "PRIO3" ou "BRENT")
      - janela_atras: quantos dias antes incluir (ex: 5 -> D-5..D0)
      - horizon: quantos dias prever (ex: 4 -> prever D+1..D+4)

    Retorna:
      X_emb (N, emb_dim),
      X_hist (N, janela_atras+1, n_features_hist) -> aqui n_features_hist=1 (retorno) por padrão
      Y (N, horizon) -> retornos em percentuais (ex: 0.5 = 0.5%)
      ids (lista) -> (data_evt, index_evento) para referência
    """

    import glob, json
    arquivos = glob.glob(os.path.join(eventos_dir, "evento_*.json"))
    X_emb = []
    X_hist = []
    Y = []
    ids = []

    # garante índice datelike
    df = df_prices.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
        df = df.set_index("Date")
    else:
        df.index = pd.to_datetime(df.index).normalize()

    close_col = [c for c in df.columns if c.startswith(f"Close_{ativo}") or c.endswith(ativo)]
    if not close_col:
        # fallback: qualquer Close
        close_col = [c for c in df.columns if "Close" in c]
    close_col = close_col[0]

    for arq in arquivos:
        try:
            j = json.load(open(arq, "r", encoding="utf-8"))
            data_evt = pd.to_datetime(j["data"]).normalize()
            motivos = j.get("motivos_identificados", []) or []
            if ativo not in j.get("seq", {}) and j.get("ativo","") != ativo and j.get("ativo","")!="AMBOS":
                continue

            # pegar seq real para o ativo (se houver) - usamos para construir targets Y
            seq_real = j.get("seq", {}).get(ativo, None)
            if seq_real is None or len(seq_real) < horizon+1:
                # pula se não temos D0..D+horizon
                continue

            # Embedding da notícia - junta motivos numa frase
            texto = " ".join(motivos) if motivos else j.get("o_que_houve","")
            emb = emb_mgr.embed(texto).reshape(-1)  # vetor 1-D

            # localizar posição do evento no df
            if data_evt not in df.index:
                # avança pro próximo pregão
                search_pos = df.index.searchsorted(data_evt)
                if search_pos >= len(df.index):
                    continue
                data_evt = df.index[search_pos]

            idx = df.index.get_loc(data_evt)
            if idx - janela_atras < 0 or idx + horizon >= len(df.index):
                continue

            # janela histórica D-(janela_atras) ... D0 (incluindo D0)
            hist_idx = df.index[idx - janela_atras: idx + 1]
            closes = df.loc[hist_idx, close_col].values
            # converter para retornos percentuais (D-? -> D-?+1), aqui representamos cada dia como retorno relativo ao anterior
            # para manter consistência, representamos a janela como retornos em relação ao dia anterior, do mesmo comprimento (janela_atras+1)
            # compute pct_change with fill 0 for first
            rets = pd.Series(closes).pct_change().fillna(0).values * 100.0  # em %
            # target Y: seq_real[1: horizon+1] (D+1..D+horizon) já estão em % no seu evento?
            # assume que seq_real já está em % (como seu arquivo mostra). Se não estiver, adapte.
            y = seq_real[1:horizon+1]
            if len(y) < horizon:
                continue

            X_emb.append(emb)
            X_hist.append(rets)   # shape (janela_atras+1,)
            Y.append(np.array(y[:horizon], dtype=float))
            ids.append((data_evt.strftime("%Y-%m-%d"), os.path.basename(arq)))

        except Exception as e:
            # ignora erro de parsing
            continue

    X_emb = np.vstack(X_emb) if X_emb else np.zeros((0, emb_mgr.embed("a").shape[1]))
    X_hist = np.array(X_hist)  # (N, janela+1)
    Y = np.array(Y)            # (N, horizon)
    return X_emb, X_hist, Y, ids
