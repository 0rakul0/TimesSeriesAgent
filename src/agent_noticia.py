import json
import os
import re
import argparse
import sys
from typing import List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from tavily import TavilyClient

if __package__ in (None, ""):
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.project_paths import DATA_DIR, OUTPUT_NOTICIAS_DIR, ensure_runtime_dirs

load_dotenv()
ensure_runtime_dirs()

LIMIAR_VARIACAO = 1.9
LIMIARES_SENSIBILIDADE = [1.5, 2.0, 2.5]

_openai_client = None
_tavily_client = None


def get_openai_client():
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


def get_tavily_client():
    global _tavily_client
    if _tavily_client is None:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError("TAVILY_API_KEY nao configurada.")
        _tavily_client = TavilyClient(api_key=api_key)
    return _tavily_client


EMPRESAS = {
    "PETR4.SA": "Petrobras",
    "PRIO3.SA": "PetroRio",
    "EXXO34.SA": "ExxonMobil",
    "BZ=F": "petroleo Brent",
    "BRENT": "petroleo Brent",
}


class EventoNoticia(BaseModel):
    data: str
    ativo: str
    retorno_no_dia: float
    fechamento: float
    sentimento_do_mercado: str = Field(default="neutro")
    o_que_houve: str = ""
    motivos_identificados: List[str] = Field(default_factory=list)
    fontes: List[str] = Field(default_factory=list)


def _safe_extract_json(texto: str) -> dict:
    try:
        return json.loads(texto)
    except Exception:
        match = re.search(r"\{.*\}", texto, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
    raise ValueError("O modelo nao retornou JSON valido.")


def _sem_evento_relevante(texto: str) -> bool:
    if not texto:
        return True

    texto = texto.lower()
    padroes = [
        "sem evento",
        "nao houve",
        "nenhuma noticia",
        "movimento geral",
        "macro",
        "nao ha registro",
        "fatores macroeconomicos",
    ]
    return any(padrao in texto for padrao in padroes)


def escolher_motivo_principal(motivos: List[str]) -> List[str]:
    if not motivos or len(motivos) == 1:
        return motivos

    prompt = f"""
Dentre os motivos abaixo, escolha apenas aquele que representa a causa principal
do movimento do ativo no mercado. Responda somente com o texto exato do motivo.

MOTIVOS:
{json.dumps(motivos, ensure_ascii=False, indent=2)}
"""

    try:
        resposta = get_openai_client().chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        escolha = (resposta.choices[0].message.content or "").strip()
        for motivo in motivos:
            if escolha.lower() in motivo.lower():
                return [motivo]
    except Exception as exc:
        print(f"[WARN] Erro ao escolher motivo principal: {exc}")

    return [motivos[0]]


def coletar_noticias_tavily(ativo: str, data_iso: str):
    empresa = EMPRESAS.get(ativo, ativo)
    query = f"noticias {empresa} {ativo} {data_iso} petroleo brent"

    resposta = get_tavily_client().search(
        query=query,
        include_raw_content=True,
        search_depth="advanced",
        max_results=4,
    )

    textos = []
    fontes = []
    for resultado in resposta.get("results", []):
        if resultado.get("content"):
            textos.append(f"{resultado['title']}\n{resultado['content']}")
            fontes.append(resultado.get("url", ""))

    return "\n\n".join(textos), fontes


def consultar_chatgpt_evento(ativo: str, data_iso: str, retorno: float, fechamento: float):
    prompt = f"""
Explique o que ocorreu com o ativo {ativo} no dia {data_iso}.
Use apenas fatos reais. Se nao houver evento relevante, diga claramente.

Retorne somente JSON:

{{
  "data": "{data_iso}",
  "ativo": "{ativo}",
  "retorno_no_dia": {retorno},
  "fechamento": {fechamento},
  "sentimento_do_mercado": "<positivo|negativo|neutro>",
  "o_que_houve": "<maximo 3 frases>",
  "motivos_identificados": ["<mot1>", "<mot2>"],
  "fontes": ["Valor Economico", "Reuters"]
}}
"""

    resposta = get_openai_client().chat.completions.create(
        model="gpt-4.1",
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    dados = _safe_extract_json(resposta.choices[0].message.content or "")
    dados["retorno_no_dia"] = retorno
    dados["fechamento"] = fechamento
    dados["ativo"] = ativo
    dados["data"] = data_iso

    return EventoNoticia(**dados)


def consultar_evento_hibrido(ativo, data_iso, retorno, fechamento):
    evento = consultar_chatgpt_evento(ativo, data_iso, retorno, fechamento)

    if evento.o_que_houve and not _sem_evento_relevante(evento.o_que_houve):
        return evento

    print("[INFO] GPT nao encontrou evento claro; usando Tavily.")

    try:
        texto, fontes = coletar_noticias_tavily(ativo, data_iso)
    except Exception as exc:
        print(f"[WARN] Tavily indisponivel: {exc}")
        return evento

    if not texto:
        return evento

    empresa = EMPRESAS.get(ativo, ativo)
    prompt = f"""
Explique o que ocorreu com o ativo {ativo} ({empresa}) no dia {data_iso}
usando exclusivamente as noticias abaixo.

Retorne somente JSON.

NOTICIAS:
{texto}
"""

    resposta = get_openai_client().chat.completions.create(
        model="gpt-4.1",
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    dados = _safe_extract_json(resposta.choices[0].message.content or "")
    dados["retorno_no_dia"] = retorno
    dados["fechamento"] = fechamento
    dados["ativo"] = ativo
    dados["data"] = data_iso
    dados["fontes"] = dados.get("fontes") or fontes
    dados["motivos_identificados"] = escolher_motivo_principal(
        dados.get("motivos_identificados", [])
    )

    return EventoNoticia(**dados)


def calcular_zscore(retorno, std):
    if std is None or std <= 0 or np.isnan(std):
        return 0.0
    return float(retorno / std)


def nome_arquivo_evento(ticker: str, data_iso: str):
    ticker_limpo = ticker.replace(".SA", "").replace("=F", "")
    return f"evento_{ticker_limpo}_{data_iso}.json"


def carregar_base_eventos(ticker, caminho_dados):
    df = pd.read_csv(caminho_dados)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    df[f"Ret_{ticker}"] = df[f"Close_{ticker}"].pct_change(fill_method=None) * 100
    df["Ret_BZ"] = df["Close_BZ=F"].pct_change(fill_method=None) * 100

    std_a = df[f"Ret_{ticker}"].expanding().std()
    std_b = df["Ret_BZ"].expanding().std()
    return df, std_a, std_b


def identificar_datas_evento(df, ticker, limiar=LIMIAR_VARIACAO, incluir_brent=True):
    eps = 1e-9
    eventos_a = df[df[f"Ret_{ticker}"].abs() > (limiar + eps)]
    eventos_b = df[df["Ret_BZ"].abs() > (limiar + eps)]

    if incluir_brent:
        return sorted(set(eventos_a.index) | set(eventos_b.index)), eventos_a, eventos_b

    return sorted(eventos_a.index), eventos_a, eventos_b


def detectar_eventos(ticker, caminho_dados, limiar=LIMIAR_VARIACAO, incluir_brent=True):
    print(f"\n[INFO] Rodando eventos para {ticker} com limiar {limiar:.2f}%...")

    df, std_a, std_b = carregar_base_eventos(ticker, caminho_dados)
    datas, eventos_a, eventos_b = identificar_datas_evento(
        df,
        ticker,
        limiar=limiar,
        incluir_brent=incluir_brent,
    )

    for data_evt in datas:
        row = df.loc[data_evt]
        data_iso = data_evt.strftime("%Y-%m-%d")
        ativo = ticker if data_evt in eventos_a.index else "BRENT"

        out = OUTPUT_NOTICIAS_DIR / nome_arquivo_evento(ativo, data_iso)
        if out.exists():
            continue

        ret = row[f"Ret_{ticker}"] if ativo == ticker else row["Ret_BZ"]
        fech = row[f"Close_{ticker}"] if ativo == ticker else row["Close_BZ=F"]

        evento = consultar_evento_hibrido(ativo, data_iso, ret, fech)
        registro = evento.model_dump()
        registro["ativo"] = ativo.replace(".SA", "")
        registro["impacto_d0"] = ret
        registro["zscore_d0"] = calcular_zscore(
            ret,
            std_a.loc[data_evt] if ativo == ticker else std_b.loc[data_evt],
        )

        with out.open("w", encoding="utf-8") as file:
            json.dump(registro, file, indent=4, ensure_ascii=False)

        print(f"[OK] Salvo: {out}")


def analisar_limiares_evento(caminhos_csv, limiares=LIMIARES_SENSIBILIDADE, incluir_brent=False, salvar_csv=True):
    registros = []

    for ticker, caminho_csv in caminhos_csv.items():
        df, _, _ = carregar_base_eventos(ticker, str(caminho_csv))
        for limiar in limiares:
            datas, eventos_a, eventos_b = identificar_datas_evento(
                df,
                ticker,
                limiar=limiar,
                incluir_brent=incluir_brent,
            )
            registros.append(
                {
                    "Ativo": ticker.replace(".SA", ""),
                    "Limiar_Evento_Pct": float(limiar),
                    "Qtd_Eventos_Ativo": int(len(eventos_a)),
                    "Qtd_Eventos_Brent": int(len(eventos_b)),
                    "Qtd_Eventos_Considerados": int(len(datas)),
                    "Primeira_Data_Evento": eventos_a.index.min().strftime("%Y-%m-%d") if len(eventos_a) else None,
                    "Ultima_Data_Evento": eventos_a.index.max().strftime("%Y-%m-%d") if len(eventos_a) else None,
                }
            )

    df_resultado = pd.DataFrame(registros).sort_values(["Ativo", "Limiar_Evento_Pct"]).reset_index(drop=True)

    if salvar_csv:
        nome_arquivo = "analise_limiares_evento_com_brent.csv" if incluir_brent else "analise_limiares_evento.csv"
        out_csv = DATA_DIR / nome_arquivo
        df_resultado.to_csv(out_csv, index=False)
        print(f"[OK] Analise de limiares salva em: {out_csv}")

    return df_resultado


def remover_jsons_sem_motivos(caminho_saida=OUTPUT_NOTICIAS_DIR):
    print("\n[INFO] Limpando eventos sem motivos_identificados...")

    arquivos = [arquivo for arquivo in caminho_saida.iterdir() if arquivo.suffix == ".json"]
    removidos = 0

    for arquivo in arquivos:
        try:
            with arquivo.open("r", encoding="utf-8") as file:
                dados = json.load(file)

            if not dados.get("motivos_identificados"):
                arquivo.unlink()
                removidos += 1
                print(f"[DEL] Removido {arquivo.name}")
        except Exception as exc:
            print(f"[WARN] Erro ao ler {arquivo}: {exc}. Removendo arquivo.")
            arquivo.unlink(missing_ok=True)
            removidos += 1

    print(f"[OK] Limpeza concluida. Arquivos removidos: {removidos}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deteccao de eventos e analise de sensibilidade por limiar.")
    parser.add_argument(
        "--analisar-limiares",
        action="store_true",
        help="Gera tabela com quantidade de eventos detectados por limiar sem consultar APIs externas.",
    )
    parser.add_argument(
        "--incluir-brent",
        action="store_true",
        help="Inclui tambem eventos detectados no Brent na contagem consolidada.",
    )
    args = parser.parse_args()

    caminhos = {
        "PETR4.SA": DATA_DIR / "dados_petr4_brent.csv",
        "PRIO3.SA": DATA_DIR / "dados_prio3_brent.csv",
        "EXXO34.SA": DATA_DIR / "dados_exxo34_brent.csv",
    }

    if args.analisar_limiares:
        tabela = analisar_limiares_evento(caminhos, incluir_brent=args.incluir_brent)
        print("\nResumo de sensibilidade por limiar:\n")
        print(tabela.to_string(index=False))
        raise SystemExit(0)

    for ticker, nome_csv in caminhos.items():
        detectar_eventos(ticker, str(nome_csv.resolve()))

    remover_jsons_sem_motivos()
