#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from eval.plotter_refactor import plotar_comparacao_por_ativo, plotar_hibrido_corrigido
from modelos.model_baseline_lstm import LSTMPrice
from modelos.model_lstm_autoencoder import LSTMAutoencoderPrice
from modelos.model_transformer_price import TransformerPrice
from utils.embedding_manager import EmbeddingManager
from utils.project_paths import DATA_DIR, IMG_DIR, MODELOS_DIR, OUTPUT_NOTICIAS_DIR, ensure_runtime_dirs

ensure_runtime_dirs()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_DIR = DATA_DIR / "experiment_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

emb_mgr = EmbeddingManager()

RESULTADOS = []
PREVISOES_MODELOS = {}
PREVISOES_HIBRIDOS = {}


def carregar_modelo_unificado(model_path, tipo="lstm"):
    from sklearn.preprocessing import MinMaxScaler

    try:
        torch.serialization.add_safe_globals([np.ndarray, MinMaxScaler, dict, list, tuple])
    except Exception:
        pass

    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    input_size = len(ckpt["train_columns"])

    if tipo.lower() == "lstm":
        model = LSTMPrice(input_size=input_size).to(DEVICE)
    elif tipo.lower() == "autoencoder":
        model = LSTMAutoencoderPrice(
            input_size=input_size,
            hidden_size=128,
            latent_size=64,
            num_layers=2,
            dropout=0.1,
        ).to(DEVICE)
    elif tipo.lower() == "transformer":
        model = TransformerPrice(
            input_size=input_size,
            d_model=128,
            nhead=4,
            num_layers=3,
            dim_feedforward=256,
            dropout=0.1,
        ).to(DEVICE)
    else:
        raise ValueError(f"Tipo desconhecido de modelo: {tipo}")

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["scaler"], ckpt["train_columns"], ckpt["seq_len"]


def detectar_coluna_close(train_cols):
    closes = [col for col in train_cols if col.startswith("Close_")]
    if closes:
        return closes[0]
    closes = [col for col in train_cols if "Close" in col]
    if not closes:
        raise KeyError("Nenhuma coluna Close encontrada no checkpoint.")
    return closes[0]


def prever_unificado(model, scaler, df, seq_len, train_cols, tipo="lstm"):
    df2 = df.copy().sort_values("Date").reset_index(drop=True)
    target_col = detectar_coluna_close(train_cols)
    target_idx = train_cols.index(target_col)

    for col in train_cols:
        if col not in df2.columns:
            df2[col] = 0.0

    arr = scaler.transform(df2[train_cols])
    preds, reals, dates = [], [], []

    if len(df2) <= seq_len:
        return pd.DataFrame(columns=["Date", "Pred", "Real"]).set_index("Date")

    for i in range(len(df2) - seq_len):
        seq = arr[i:i + seq_len]
        seq_tensor = torch.tensor(seq[np.newaxis], dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            if tipo.lower() == "autoencoder":
                pred_s, _ = model(seq_tensor)
                pred_s = pred_s.cpu().numpy().item()
            else:
                pred_s = model(seq_tensor).cpu().numpy().item()

        zeros = np.zeros((1, len(train_cols)))
        zeros[0, target_idx] = pred_s
        inv = scaler.inverse_transform(zeros)[0, target_idx]

        preds.append(inv)
        reals.append(df2.loc[i + seq_len, target_col])
        dates.append(pd.to_datetime(df2.loc[i + seq_len, "Date"]))

    return pd.DataFrame({"Date": dates, "Pred": preds, "Real": reals}).set_index("Date")


def extrair_motivos(pasta_json):
    noticias = {}
    for arquivo in sorted(Path(pasta_json).glob("evento_*.json")):
        try:
            with arquivo.open("r", encoding="utf-8") as handle:
                registro = json.load(handle)
            data = pd.to_datetime(registro["data"]).normalize()
            motivos = [mot.strip() for mot in registro.get("motivos_identificados", []) if isinstance(mot, str) and mot.strip()]
            if motivos:
                noticias.setdefault(data, []).extend(motivos)
        except Exception:
            continue
    return noticias


def carregar_biblioteca_eventos(pasta_json):
    rows = []
    for arquivo in sorted(Path(pasta_json).glob("evento_*.json")):
        try:
            with arquivo.open("r", encoding="utf-8") as handle:
                registro = json.load(handle)
        except Exception:
            continue

        data = pd.to_datetime(registro.get("data")).normalize()
        ativo = str(registro.get("ativo", "GENERICO")).upper()
        seq_map = registro.get("seq", {}) or {}
        seq = seq_map.get(ativo) or []
        if not seq:
            continue

        motivos = [mot.strip() for mot in registro.get("motivos_identificados", []) if isinstance(mot, str) and mot.strip()]
        for motivo in motivos:
            rows.append(
                {
                    "date": data,
                    "ativo": ativo,
                    "motivo": motivo,
                    "seq": list(seq),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["date", "ativo", "motivo", "seq", "embedding"])

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df["embedding"] = [np.asarray(emb_mgr.embed(motivo)).reshape(-1) for motivo in df["motivo"]]
    return df


def _ativo_from_nome(nome):
    token = str(nome).upper().strip().split()[0]
    return token


def _event_library_filter(df, ativo, cutoff_date):
    ativos_validos = {ativo.upper(), "BRENT", "GENERICO"}
    return df[(df["date"] < cutoff_date) & (df["ativo"].isin(ativos_validos))].copy()


def _similaridade_coseno(vetor, matriz):
    vetor = np.asarray(vetor).reshape(1, -1)
    matriz = np.asarray(matriz)
    if matriz.ndim == 1:
        matriz = matriz.reshape(1, -1)
    num = vetor @ matriz.T
    den = np.linalg.norm(vetor, axis=1, keepdims=True) * np.linalg.norm(matriz, axis=1) + 1e-12
    return (num / den).flatten()


def recuperar_impacto_historico(motivos, cutoff_date, biblioteca_eventos, ativo, top_k=5, sim_threshold=0.70):
    if not motivos:
        return None

    historico = _event_library_filter(biblioteca_eventos, ativo, cutoff_date)
    if historico.empty:
        return None

    candidatos = []
    emb_matrix = np.vstack(historico["embedding"].values)

    for motivo in motivos:
        emb = np.asarray(emb_mgr.embed(motivo)).reshape(-1)
        sims = _similaridade_coseno(emb, emb_matrix)
        if sims.size == 0:
            continue

        hist = historico.copy()
        hist["sim"] = sims
        hist["motivo_atual"] = motivo
        candidatos.append(hist)

    if not candidatos:
        return None

    candidatos_df = pd.concat(candidatos, ignore_index=True)
    candidatos_df = candidatos_df[candidatos_df["sim"] >= sim_threshold]
    if candidatos_df.empty:
        return None

    candidatos_df = candidatos_df.sort_values(["sim", "date"], ascending=[False, False]).head(top_k).reset_index(drop=True)
    max_len = max(len(seq) for seq in candidatos_df["seq"])
    seq_avg = []
    for i in range(max_len):
        valores = [seq[i] for seq in candidatos_df["seq"] if len(seq) > i and seq[i] is not None]
        seq_avg.append(float(np.average(valores, weights=candidatos_df.loc[[idx for idx, seq in enumerate(candidatos_df["seq"]) if len(seq) > i and seq[i] is not None], "sim"])) if valores else np.nan)

    return {
        "motivo_atual": str(candidatos_df.iloc[0]["motivo_atual"]),
        "similaridade": float(candidatos_df["sim"].mean()),
        "n_historico": int(len(candidatos_df)),
        "seq_media": seq_avg,
        "datas_referencia": [pd.to_datetime(d).strftime("%Y-%m-%d") for d in candidatos_df["date"].tolist()],
        "motivos_referencia": candidatos_df["motivo"].tolist(),
    }


def motivo_e_cluster_mais_relevante(motivos, emb_mgr_local, emb_repr, clusters_df, ativo):
    if not motivos:
        return None, 0.0, None, None

    emb_repr_mat = np.asarray(emb_repr)
    if emb_repr_mat.ndim == 1:
        emb_repr_mat = emb_repr_mat.reshape(1, -1)

    ativo = (ativo or "").upper()
    permitidos = [ativo, "BRENT", "GENERICO", "GENÉRICO"]
    if "ativo_cluster" in clusters_df.columns:
        mask = clusters_df["ativo_cluster"].astype(str).str.upper().isin(permitidos)
        clusters_df_f = clusters_df[mask].reset_index(drop=True)
        emb_f = emb_repr_mat[mask.values] if len(clusters_df_f) == len(mask[mask]) else emb_repr_mat
    else:
        clusters_df_f = clusters_df.reset_index(drop=True)
        emb_f = emb_repr_mat

    if clusters_df_f.empty:
        return None, 0.0, None, None

    best = {"motivo": None, "sim": -1.0, "cluster": None, "row": None}
    for motivo in motivos:
        emb = np.asarray(emb_mgr_local.embed(motivo)).reshape(1, -1)
        sims = _similaridade_coseno(emb, emb_f)
        idx = int(np.argmax(sims))
        sim = float(sims[idx])
        if sim > best["sim"]:
            row = clusters_df_f.iloc[idx]
            cluster_id = int(row["cluster"]) if "cluster" in row else idx
            best.update({"motivo": motivo, "sim": sim, "cluster": cluster_id, "row": row})

    return best["motivo"], best["sim"], best["cluster"], best["row"]


def aplicar_seq_real(pred_df, motivos_por_data, biblioteca_eventos, ativo, sim_threshold=0.7, max_horizon=4, scale=0.4, decision_log=None):
    df = pred_df.copy()
    df["Pred_Ajustado"] = df.get("Pred_Ajustado", df["Pred"]).astype(float)
    dias = df.index
    eventos = sorted(motivos_por_data.keys())

    for data_evt in eventos:
        motivos = motivos_por_data.get(data_evt, [])
        if not motivos:
            continue

        pos = dias.searchsorted(data_evt)
        if pos >= len(dias):
            continue

        contexto = recuperar_impacto_historico(motivos, data_evt, biblioteca_eventos, ativo, sim_threshold=sim_threshold)
        if contexto is None:
            if decision_log is not None:
                decision_log.append(
                    {
                        "event_date": pd.to_datetime(data_evt).strftime("%Y-%m-%d"),
                        "ativo": ativo,
                        "motivos": json.dumps(motivos, ensure_ascii=False),
                        "selected_motivo": None,
                        "avg_similarity": 0.0,
                        "historical_events_used": 0,
                        "reference_dates": "[]",
                        "reference_motivos": "[]",
                        "applied_scale": scale,
                        "source": "seq",
                        "status": "no_historical_match",
                    }
                )
            continue

        futuras = [d for d in eventos if d > data_evt]
        seq = contexto["seq_media"]

        if decision_log is not None:
            decision_log.append(
                {
                    "event_date": pd.to_datetime(data_evt).strftime("%Y-%m-%d"),
                    "ativo": ativo,
                    "motivos": json.dumps(motivos, ensure_ascii=False),
                    "selected_motivo": contexto["motivo_atual"],
                    "avg_similarity": contexto["similaridade"],
                    "historical_events_used": contexto["n_historico"],
                    "reference_dates": json.dumps(contexto["datas_referencia"], ensure_ascii=False),
                    "reference_motivos": json.dumps(contexto["motivos_referencia"], ensure_ascii=False),
                    "applied_scale": scale,
                    "source": "seq",
                    "status": "applied",
                }
            )

        for k, impacto in enumerate(seq[: max_horizon + 1]):
            if pd.isna(impacto):
                break
            idx_d = pos + k
            if idx_d >= len(dias):
                break
            dia_k = dias[idx_d]
            if any((f > data_evt and f <= dia_k) for f in futuras):
                break
            ajuste = scale * contexto["similaridade"] * (float(impacto) / 100.0)
            df.loc[dia_k, "Pred_Ajustado"] *= (1 + ajuste)

    return df


def aplicar_walkforward_residual(pred_df, motivos_por_data, biblioteca_eventos, ativo, janela, base_col="Pred_Ajustado", decision_log=None):
    df = pred_df.copy().reset_index()
    df["Base_Residual"] = df[base_col].astype(float)
    df["r"] = df["Real"] - df["Base_Residual"]

    sims, eventos, hist_counts = [], [], []
    for data in df["Date"]:
        dnorm = pd.to_datetime(data).normalize()
        motivos = motivos_por_data.get(dnorm, [])
        contexto = recuperar_impacto_historico(motivos, dnorm, biblioteca_eventos, ativo) if motivos else None
        eventos.append(1 if motivos else 0)
        sims.append(float(contexto["similaridade"]) if contexto else 0.0)
        hist_counts.append(int(contexto["n_historico"]) if contexto else 0)
        if motivos and decision_log is not None:
            decision_log.append(
                {
                    "event_date": pd.to_datetime(dnorm).strftime("%Y-%m-%d"),
                    "ativo": ativo,
                    "motivos": json.dumps(motivos, ensure_ascii=False),
                    "selected_motivo": contexto["motivo_atual"] if contexto else None,
                    "avg_similarity": float(contexto["similaridade"]) if contexto else 0.0,
                    "historical_events_used": int(contexto["n_historico"]) if contexto else 0,
                    "reference_dates": json.dumps(contexto["datas_referencia"], ensure_ascii=False) if contexto else "[]",
                    "reference_motivos": json.dumps(contexto["motivos_referencia"], ensure_ascii=False) if contexto else "[]",
                    "applied_scale": None,
                    "source": "residual",
                    "status": "feature_only" if contexto else "no_historical_match",
                }
            )

    df["sim_day"] = sims
    df["event_bin"] = eventos
    df["hist_count"] = hist_counts
    df["r_lag1"] = df["r"].shift(1)
    df["r_lag2"] = df["r"].shift(2)
    df["Pred_Final"] = df["Base_Residual"].copy()

    feat_cols = ["r_lag1", "r_lag2", "sim_day", "event_bin", "hist_count"]

    for t in range(janela, len(df)):
        train_df = df.iloc[:t].dropna(subset=feat_cols + ["r"])
        if len(train_df) < janela:
            continue

        X = train_df[feat_cols].values
        y = train_df["r"].shift(-1).dropna().values
        if len(y) < len(X):
            X = X[: len(y)]

        model = Ridge(alpha=1.0)
        model.fit(X, y)

        X_now = df.loc[t, feat_cols].values.reshape(1, -1)
        try:
            r_pred = float(model.predict(X_now)[0])
        except Exception:
            r_pred = 0.0
        df.loc[t, "Pred_Final"] = df.loc[t, "Base_Residual"] + r_pred

    return df.set_index("Date")


def encontrar_scale_otimo(pred_df, motivos_por_data, biblioteca_eventos, ativo):
    candidatos = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    melhor, menor = 0.4, float("inf")

    for scale in candidatos:
        temp = aplicar_seq_real(pred_df, motivos_por_data, biblioteca_eventos, ativo=ativo, sim_threshold=0.7, max_horizon=4, scale=scale)
        rmse = np.sqrt(mean_squared_error(temp["Real"], temp["Pred_Ajustado"]))
        if rmse < menor:
            menor = rmse
            melhor = scale
    return melhor


def _rmse(real, pred):
    return float(np.sqrt(mean_squared_error(real, pred)))


def _save_decision_log(log_rows, ativo, tipo):
    if not log_rows:
        return None
    out_path = LOG_DIR / f"decision_log_{ativo.lower()}_{tipo}.csv"
    pd.DataFrame(log_rows).to_csv(out_path, index=False)
    return out_path


def _save_prediction_details(ativo, tipo, pred_base, seq_only, residual_only, hybrid, motivos_por_data):
    event_dates = {pd.to_datetime(data).normalize() for data in motivos_por_data.keys()}
    index = hybrid.index
    df = pd.DataFrame(
        {
            "Date": index,
            "Ativo": ativo,
            "Modelo": tipo.upper(),
            "Real": hybrid["Real"].astype(float).values,
            "Pred_Base": pred_base.reindex(index)["Pred"].astype(float).values,
            "Pred_SeqOnly": seq_only.reindex(index)["Pred_Ajustado"].astype(float).values,
            "Pred_ResidualOnly": residual_only.reindex(index)["Pred_Final"].astype(float).values,
            "Pred_Hibrido": hybrid["Pred_Final"].astype(float).values,
        }
    )
    df["ErroAbs_Base"] = (df["Real"] - df["Pred_Base"]).abs()
    df["ErroAbs_Hibrido"] = (df["Real"] - df["Pred_Hibrido"]).abs()
    df["Hybrid_Better"] = df["ErroAbs_Hibrido"] < df["ErroAbs_Base"]
    df["Event_Day"] = df["Date"].apply(lambda data: pd.to_datetime(data).normalize() in event_dates)
    df["Year"] = pd.to_datetime(df["Date"]).dt.year

    out_path = LOG_DIR / f"prediction_details_{ativo.lower()}_{tipo}.csv"
    df.to_csv(out_path, index=False)
    return out_path


def rodar_modelo_unificado(csv_path, model_path, out_html, nome, tipo):
    print(f"\n=========== {nome} ({tipo.upper()}) ===========")

    ativo = _ativo_from_nome(nome)
    df = pd.read_csv(csv_path, parse_dates=["Date"]).sort_values("Date").ffill().bfill()
    model, scaler, cols, seq_len = carregar_modelo_unificado(model_path, tipo=tipo)
    pred_df = prever_unificado(model, scaler, df, seq_len, cols, tipo=tipo)

    motivos_por_data = extrair_motivos(OUTPUT_NOTICIAS_DIR)
    biblioteca_eventos = carregar_biblioteca_eventos(OUTPUT_NOTICIAS_DIR)

    decision_log = []
    scale = encontrar_scale_otimo(pred_df, motivos_por_data, biblioteca_eventos, ativo)

    seq_only = aplicar_seq_real(pred_df, motivos_por_data, biblioteca_eventos, ativo=ativo, scale=scale, decision_log=decision_log)
    residual_only = aplicar_walkforward_residual(pred_df, motivos_por_data, biblioteca_eventos, ativo=ativo, janela=15, base_col="Pred", decision_log=decision_log)
    hybrid = aplicar_walkforward_residual(seq_only, motivos_por_data, biblioteca_eventos, ativo=ativo, janela=15, base_col="Pred_Ajustado", decision_log=decision_log)
    hybrid["Pred_Ajustado"] = hybrid["Pred_Final"]
    hybrid["Pred_Final_Price"] = hybrid["Pred_Final"]

    _save_decision_log(decision_log, ativo, tipo)
    _save_prediction_details(ativo, tipo, pred_df, seq_only, residual_only, hybrid, motivos_por_data)

    rmse_base = _rmse(pred_df["Real"], pred_df["Pred"])
    rmse_seq = _rmse(seq_only["Real"], seq_only["Pred_Ajustado"])
    rmse_residual = _rmse(residual_only["Real"], residual_only["Pred_Final"])
    rmse_hib = _rmse(hybrid["Real"], hybrid["Pred_Final"])

    PREVISOES_MODELOS[nome] = pred_df[["Pred", "Real"]].copy()
    PREVISOES_HIBRIDOS[nome] = hybrid[["Pred_Ajustado", "Real"]].copy()

    clusters_df = pd.read_csv(DATA_DIR / "cluster_motivos.csv") if (DATA_DIR / "cluster_motivos.csv").exists() else pd.DataFrame(columns=["cluster", "frase_exemplo", "ativo_cluster"])
    emb_repr = emb_mgr.embed_lote(clusters_df["frase_exemplo"].astype(str).tolist()) if not clusters_df.empty else np.zeros((0, 1536))
    plotar_hibrido_corrigido(hybrid, motivos_por_data, emb_mgr, clusters_df, emb_repr, out_html, nome)

    RESULTADOS.append(
        {
            "Ativo": nome,
            "Modelo": tipo.upper(),
            "RMSE_Modelo": round(rmse_base, 4),
            "RMSE_SeqOnly": round(rmse_seq, 4),
            "RMSE_ResidualOnly": round(rmse_residual, 4),
            "RMSE_Hibrido": round(rmse_hib, 4),
            "Ganho_Hibrido": round(rmse_base - rmse_hib, 4),
            "Scale_Selecionado": round(scale, 2),
        }
    )


def eval_modelos():
    ativos = {"PETR4": "petr4", "PRIO3": "prio3", "EXXO34": "exxo34"}
    outs = {
        "lstm": lambda a: IMG_DIR / f"previsao_lstm_{a}.html",
        "autoencoder": lambda a: IMG_DIR / f"previsao_ae_{a}.html",
        "transformer": lambda a: IMG_DIR / f"previsao_transformer_{a}.html",
    }
    nome_format = {"lstm": "(LSTM)", "autoencoder": "(AE)", "transformer": "(Transformer)"}

    for tipo in ["lstm", "autoencoder", "transformer"]:
        for ativo, nomefile in ativos.items():
            rodar_modelo_unificado(
                str(DATA_DIR / f"dados_{nomefile}_brent.csv"),
                str(MODELOS_DIR / f"{tipo}_{nomefile}.pt"),
                str(outs[tipo](nomefile)),
                f"{ativo} {nome_format[tipo]}",
                tipo=tipo,
            )

    motivos_por_data = extrair_motivos(OUTPUT_NOTICIAS_DIR)
    clusters_df = pd.read_csv(DATA_DIR / "cluster_motivos.csv") if (DATA_DIR / "cluster_motivos.csv").exists() else pd.DataFrame(columns=["cluster", "frase_exemplo"])
    emb_repr = emb_mgr.embed_lote(clusters_df["frase_exemplo"].astype(str).tolist()) if not clusters_df.empty else np.zeros((0, 1536))

    for ativo in ["PETR4", "PRIO3", "EXXO34"]:
        prev_puro = {k: v for k, v in PREVISOES_MODELOS.items() if k.startswith(ativo)}
        prev_hib = {k: v for k, v in PREVISOES_HIBRIDOS.items() if k.startswith(ativo)}
        if not prev_puro:
            continue
        df_base = list(prev_puro.values())[0]
        plotar_comparacao_por_ativo(
            df_base,
            prev_puro,
            prev_hib,
            motivos_por_data,
            emb_mgr,
            clusters_df,
            emb_repr,
            str(IMG_DIR / f"comparacao_{ativo.lower()}.html"),
            ativo,
        )

    df_final = pd.DataFrame(RESULTADOS)
    print(df_final.to_string(index=False))
    df_final.to_csv(DATA_DIR / "resultado_comparacao_modelos.csv", index=False)
    df_final.to_csv(DATA_DIR / "resultado_ablation_modelos.csv", index=False)


if __name__ == "__main__":
    eval_modelos()
