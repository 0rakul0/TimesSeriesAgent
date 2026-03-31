from pathlib import Path

import numpy as np
import pandas as pd

from utils.project_paths import DATA_DIR


LOG_DIR = DATA_DIR / "experiment_logs"
RQ_OUTPUT = DATA_DIR / "perguntas_pesquisa.md"
RQ_CATALOG = [
    {
        "id": "RQ1",
        "pergunta": "Modelos hibridos com recuperacao semantica de eventos reduzem o erro de previsao em relacao aos modelos puramente quantitativos?",
        "por_que_importa": "Essa e a pergunta principal do projeto, porque testa se a camada informacional agrega valor preditivo real.",
        "evidencias": [
            "resultado_comparacao_modelos.csv",
            "analise_metricas_complementares.csv",
        ],
    },
    {
        "id": "RQ2",
        "pergunta": "O ganho do hibrido se concentra em dias com evento ou tambem aparece em dias sem noticia relevante?",
        "por_que_importa": "Ajuda a separar utilidade informacional de um simples efeito medio de suavizacao do erro.",
        "evidencias": [
            "analise_eventos_vs_nao_eventos.csv",
        ],
    },
    {
        "id": "RQ3",
        "pergunta": "Eventos semanticamente semelhantes apresentam padroes recorrentes de propagacao temporal apos o choque inicial?",
        "por_que_importa": "Valida a hipotese de que o componente de noticias pode ser reutilizado como biblioteca historica de eventos comparaveis, e nao apenas como contextualizacao textual.",
        "evidencias": [
            "cluster_motivos.csv",
            "cluster_*.csv",
            "output_noticias/evento_*.json",
        ],
    },
    {
        "id": "RQ4",
        "pergunta": "O desempenho do metodo e robusto entre ativos, arquiteturas base e anos distintos?",
        "por_que_importa": "Evita concluir a partir de um unico ativo, modelo ou recorte temporal favoravel.",
        "evidencias": [
            "resultado_comparacao_modelos.csv",
            "analise_robustez_anual.csv",
        ],
    },
    {
        "id": "RQ5",
        "pergunta": "Em quais situacoes o ajuste semantico piora a previsao e quais falhas operacionais explicam esses casos?",
        "por_que_importa": "Essa pergunta transforma erros do hibrido em aprendizado metodologico para o artigo.",
        "evidencias": [
            "analise_piora_hibrido_resumo.csv",
            "analise_piora_hibrido_detalhes.csv",
            "data/experiment_logs/prediction_details_*.csv",
        ],
    },
    {
        "id": "RQ6",
        "pergunta": "Quao sensivel e a deteccao de eventos ao limiar adotado para variacao diaria?",
        "por_que_importa": "Garante que a construcao da base semantica nao dependa de uma heuristica arbitraria nao auditada.",
        "evidencias": [
            "analise_limiares_evento.csv",
        ],
    },
]


def _resolve_output_dir(output_dir: Path | None = None) -> Path:
    path = Path(output_dir) if output_dir is not None else DATA_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_prediction_details(log_dir: Path | None = None):
    base_log_dir = Path(log_dir) if log_dir is not None else LOG_DIR
    arquivos = sorted(base_log_dir.glob("prediction_details_*.csv"))
    if not arquivos:
        raise FileNotFoundError(
            f"Nenhum prediction_details_*.csv encontrado em {base_log_dir}. "
            "Rode eval/modelo_hibrido_offline.py primeiro."
        )
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


def gerar_metricas_complementares(df, output_dir: Path | None = None):
    out_dir = _resolve_output_dir(output_dir)
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
    out.to_csv(out_dir / "analise_metricas_complementares.csv", index=False)
    return out


def gerar_eventos_vs_nao_eventos(df, output_dir: Path | None = None):
    out_dir = _resolve_output_dir(output_dir)
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
    out.to_csv(out_dir / "analise_eventos_vs_nao_eventos.csv", index=False)
    return out


def gerar_robustez_anual(df, output_dir: Path | None = None):
    out_dir = _resolve_output_dir(output_dir)
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
    out.to_csv(out_dir / "analise_robustez_anual.csv", index=False)
    return out


def gerar_casos_piora(df, output_dir: Path | None = None):
    out_dir = _resolve_output_dir(output_dir)
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
    resumo.to_csv(out_dir / "analise_piora_hibrido_resumo.csv", index=False)

    detalhes = piora.sort_values(["Ativo", "Modelo", "Deterioracao_Abs"], ascending=[True, True, False]).reset_index(drop=True)
    detalhes.to_csv(out_dir / "analise_piora_hibrido_detalhes.csv", index=False)
    return resumo, detalhes


def gerar_mapa_perguntas_pesquisa(output_path: Path | None = None):
    destino = Path(output_path) if output_path is not None else RQ_OUTPUT
    destino.parent.mkdir(parents=True, exist_ok=True)

    linhas = [
        "# Perguntas de pesquisa do TimesSeriesAgent",
        "",
        "Este arquivo organiza as perguntas de pesquisa que o projeto ja consegue responder ou que devem orientar a escrita do artigo.",
        "",
    ]

    for item in RQ_CATALOG:
        linhas.extend(
            [
                f"## {item['id']}",
                "",
                f"**Pergunta.** {item['pergunta']}",
                "",
                f"**Por que importa.** {item['por_que_importa']}",
                "",
                "**Artefatos que ajudam a responder.**",
            ]
        )
        for evidencia in item["evidencias"]:
            linhas.append(f"- `{evidencia}`")
        linhas.append("")

    destino.write_text("\n".join(linhas), encoding="utf-8")
    return destino


def gerar_respostas_artigo(log_dir: Path | None = None, output_dir: Path | None = None):
    out_dir = _resolve_output_dir(output_dir)
    df = _load_prediction_details(log_dir=log_dir)
    metricas = gerar_metricas_complementares(df, output_dir=out_dir)
    eventos = gerar_eventos_vs_nao_eventos(df, output_dir=out_dir)
    robustez = gerar_robustez_anual(df, output_dir=out_dir)
    piora_resumo, _ = gerar_casos_piora(df, output_dir=out_dir)
    mapa_path = gerar_mapa_perguntas_pesquisa(out_dir / "perguntas_pesquisa.md")

    print("\nMetricas complementares:\n")
    print(metricas.to_string(index=False))
    print("\nEventos vs nao-eventos:\n")
    print(eventos.to_string(index=False))
    print("\nRobustez anual:\n")
    print(robustez.to_string(index=False))
    print("\nResumo de piora do hibrido:\n")
    print(piora_resumo.to_string(index=False))
    print(f"\nMapa de perguntas de pesquisa salvo em: {mapa_path}")


if __name__ == "__main__":
    gerar_respostas_artigo()
