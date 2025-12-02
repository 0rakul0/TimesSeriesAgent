import os
import numpy as np
import pandas as pd
import torch

from utils.embedding_manager import EmbeddingManager
from modelos.impact_predictor import ImpactPredictor


def aplicar_impact_predictor_com_motivos(
    pred_df,
    model_path,
    motivos_por_data,
    emb_mgr=None,
    janela=5,
    horizon=4,
    scale=1.0,
    device=None
):
    """
    Aplica o Impact Predictor supervisionado às previsões do modelo AE.

    Parâmetros:
    -----------
    pred_df : DataFrame
        Deve conter colunas:
           Pred_AE (preço previsto pelo Autoencoder)
           Real    (preço real)

    model_path : str
        Caminho do checkpoint impact_predictor_<ATIVO>.pt

    motivos_por_data : dict
        { Timestamp -> [motivos strings] }

    janela : int
        Quantos dias para trás usar (D-janela ... D0)

    horizon : int
        Quantos dias para frente prever (D+1..D+horizon)

    scale : float
        Multiplicador do retorno previsto (ajuste fino)

    Retorna:
    --------
    pred_df com coluna Pred_Hibrido
    """

    # escolher device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # carregar checkpoint
    ckpt = torch.load(model_path, map_location=device)

    emb_dim = ckpt.get("emb_dim")
    hist_len = ckpt.get("hist_len")
    horizon = ckpt.get("horizon", horizon)

    # instanciar modelo
    model = ImpactPredictor(
        emb_dim=emb_dim,
        hist_len=hist_len,
        horizon=horizon
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    # embedding manager
    if emb_mgr is None:
        emb_mgr = EmbeddingManager()

    # copia dataframe
    df = pred_df.copy()

    # garantir Pred_AE
    if "Pred_AE" not in df.columns:
        df["Pred_AE"] = df["Pred"].copy()

    df["Pred_Hibrido"] = df["Pred_AE"].copy()

    # percorre eventos
    for data_evt, motivos in motivos_por_data.items():
        data_evt = pd.to_datetime(data_evt).normalize()

        # se a data não estiver no índice, procurar o próximo pregão
        if data_evt not in df.index:
            pos = df.index.searchsorted(data_evt)
            if pos >= len(df.index):
                continue
            data_evt = df.index[pos]

        idx = df.index.get_loc(data_evt)

        # checar se temos janela suficiente
        if idx - janela < 0:
            continue

        # texto do embedding
        if isinstance(motivos, (list, tuple)):
            texto = " ".join(motivos)
        else:
            texto = str(motivos)

        emb = emb_mgr.embed(texto).astype(np.float32).reshape(1, -1)

        # janela histórica (D-janela ... D0)
        hist_idx = df.index[idx - janela : idx + 1]
        closes = df.loc[hist_idx, "Pred_AE"].values

        # retornos percentuais
        rets = pd.Series(closes).pct_change().fillna(0).values
        rets = rets.astype(np.float32).reshape(1, -1)

        xb = torch.tensor(emb, dtype=torch.float32).to(device)
        xh = torch.tensor(rets, dtype=torch.float32).to(device)

        # prever impactos
        with torch.no_grad():
            pred_rets = model(xb, xh).cpu().numpy().reshape(-1)

        # aplicar impacto nos próximos dias
        for k in range(1, horizon + 1):
            idx_k = idx + k
            if idx_k >= len(df.index):
                break

            # retorno previsto pelo AE
            r_base = (df["Pred_AE"].iloc[idx_k] / df["Pred_AE"].iloc[idx_k - 1] - 1) * 100

            # retorno ajustado
            r_new = r_base + scale * float(pred_rets[k - 1])

            # reconstruir preço ajustado
            P_prev = df["Pred_Hibrido"].iloc[idx_k - 1]
            df.at[df.index[idx_k], "Pred_Hibrido"] = P_prev * (1 + r_new / 100.0)

    return df
