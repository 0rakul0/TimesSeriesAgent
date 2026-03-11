from pathlib import Path

import numpy as np
import pandas as pd

from utils.project_paths import DATA_DIR


LOG_DIR = DATA_DIR / "experiment_logs"


def _load_prediction_details():
    arquivos = sorted(LOG_DIR.glob("prediction_details_*.csv"))
    if not arquivos:
        raise FileNotFoundError("Nenhum prediction_details_*.csv encontrado em data/experiment_logs. Rode eval/modelo_hibrido_offline.py primeiro.")
    return pd.concat([pd.read_csv(arquivo, parse_dates=["Date"]) for arquivo in arquivos], ignore_index=True)


def _directional_accuracy(real, pred):
    real = pd.Series(real).astype(float)
    pred = pd.Series(pred).astype(float)
    real_diff = np.sign(real.diff())
    pred_diff = np.sign(pred.diff())
    mask = real_diff.notna() & pred_diff.notna()
    if mask.sum() == 0:
        return np.nan
    return float((real_diff[mask] == pred_diff[mask]).mean())


def gerar_metricas_complementares(df):
    rows = []
    for (ativo, modelo), grupo in df.groupby(["Ativo", "Modelo"]):
        rows.append(
            {
                "Ativo": ativo,
                "Modelo": modelo,
                "MAE_Base": float((grupo["Real"] - grupo["Pred_Base"]).abs().mean()),
                "MAE_Hibrido": float((grupo["Real"] - grupo["Pred_Hibrido"]).abs().mean()),
                "Directional_Accuracy_Base": _directional_accuracy(grupo["Real"], grupo["Pred_Base"]),
                "Directional_Accuracy_Hibrido": _directional_accuracy(grupo["Real"], grupo["Pred_Hibrido"]),
                "Pct_Dias_Hibrido_Melhor": float(grupo["Hybrid_Better"].mean()),
            }
        )
    out = pd.DataFrame(rows).sort_values(["Ativo", "Modelo"]).reset_index(drop=True)
    out.to_csv(DATA_DIR / "analise_metricas_complementares.csv", index=False)
    return out


def gerar_eventos_vs_nao_eventos(df):
    rows = []
    for (ativo, modelo, event_day), grupo in df.groupby(["Ativo", "Modelo", "Event_Day"]):
        rows.append(
            {
                "Ativo": ativo,
                "Modelo": modelo,
                "Event_Day": bool(event_day),
                "Qtd_Observacoes": int(len(grupo)),
                "MAE_Base": float((grupo["Real"] - grupo["Pred_Base"]).abs().mean()),
                "MAE_Hibrido": float((grupo["Real"] - grupo["Pred_Hibrido"]).abs().mean()),
                "Pct_Dias_Hibrido_Melhor": float(grupo["Hybrid_Better"].mean()),
            }
        )
    out = pd.DataFrame(rows).sort_values(["Ativo", "Modelo", "Event_Day"]).reset_index(drop=True)
    out.to_csv(DATA_DIR / "analise_eventos_vs_nao_eventos.csv", index=False)
    return out


def gerar_robustez_anual(df):
    rows = []
    for (ativo, modelo, year), grupo in df.groupby(["Ativo", "Modelo", "Year"]):
        rows.append(
            {
                "Ativo": ativo,
                "Modelo": modelo,
                "Ano": int(year),
                "Qtd_Observacoes": int(len(grupo)),
                "RMSE_Base": float(np.sqrt(np.mean((grupo["Real"] - grupo["Pred_Base"]) ** 2))),
                "RMSE_Hibrido": float(np.sqrt(np.mean((grupo["Real"] - grupo["Pred_Hibrido"]) ** 2))),
                "MAE_Base": float((grupo["Real"] - grupo["Pred_Base"]).abs().mean()),
                "MAE_Hibrido": float((grupo["Real"] - grupo["Pred_Hibrido"]).abs().mean()),
                "Pct_Dias_Hibrido_Melhor": float(grupo["Hybrid_Better"].mean()),
            }
        )
    out = pd.DataFrame(rows).sort_values(["Ativo", "Modelo", "Ano"]).reset_index(drop=True)
    out.to_csv(DATA_DIR / "analise_robustez_anual.csv", index=False)
    return out


def gerar_casos_piora(df):
    piora = df[df["ErroAbs_Hibrido"] > df["ErroAbs_Base"]].copy()
    piora["Deterioracao_Abs"] = piora["ErroAbs_Hibrido"] - piora["ErroAbs_Base"]

    resumo = (
        piora.groupby(["Ativo", "Modelo"])
        .agg(
            Qtd_Dias_Piora=("Date", "count"),
            Deterioracao_Media=("Deterioracao_Abs", "mean"),
            Deterioracao_Max=("Deterioracao_Abs", "max"),
        )
        .reset_index()
        .sort_values(["Ativo", "Modelo"])
    )
    resumo.to_csv(DATA_DIR / "analise_piora_hibrido_resumo.csv", index=False)

    detalhes = piora.sort_values(["Ativo", "Modelo", "Deterioracao_Abs"], ascending=[True, True, False]).reset_index(drop=True)
    detalhes.to_csv(DATA_DIR / "analise_piora_hibrido_detalhes.csv", index=False)
    return resumo, detalhes


def gerar_respostas_artigo():
    df = _load_prediction_details()
    metricas = gerar_metricas_complementares(df)
    eventos = gerar_eventos_vs_nao_eventos(df)
    robustez = gerar_robustez_anual(df)
    piora_resumo, _ = gerar_casos_piora(df)

    print("\nMetricas complementares:\n")
    print(metricas.to_string(index=False))
    print("\nEventos vs nao-eventos:\n")
    print(eventos.to_string(index=False))
    print("\nRobustez anual:\n")
    print(robustez.to_string(index=False))
    print("\nResumo de piora do hibrido:\n")
    print(piora_resumo.to_string(index=False))


if __name__ == "__main__":
    gerar_respostas_artigo()
