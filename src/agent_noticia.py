import os
import re
import json
import pandas as pd
from datetime import datetime
import numpy as np
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from openai import OpenAI
from tavily import TavilyClient

load_dotenv()

CAMINHO_SAIDA = "../output_noticias"
LIMIAR_VARIACAO = 2.0

os.makedirs(CAMINHO_SAIDA, exist_ok=True)

client = OpenAI()
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# ============================================================
# MAPA PARA NOME DE EMPRESA
# ============================================================
EMPRESAS = {
    "PETR4.SA": "Petrobras",
    "PRIO3.SA": "PetroRio",
    "VALE3.SA": "Vale",
    "BZ=F": "petróleo Brent"
}


# ============================================================
# MODELO JSON
# ============================================================
class EventoNoticia(BaseModel):
    data: str
    ativo: str
    retorno_no_dia: float
    fechamento: float
    sentimento_do_mercado: str = Field(default="neutro")
    o_que_houve: str = ""
    motivos_identificados: List[str] = Field(default_factory=list)
    fontes: List[str] = Field(default_factory=list)


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================
def _safe_extract_json(texto: str) -> dict:
    """Extrai JSON mesmo se o modelo devolver texto extra."""
    try:
        return json.loads(texto)
    except Exception:
        pass

    # Tenta achar objetos JSON no meio do texto
    m = re.search(r"\{.*\}", texto, flags=re.DOTALL)
    if m:
        return json.loads(m.group(0))

    raise ValueError("O modelo não retornou JSON válido.")


def _sem_evento_relevante(txt: str) -> bool:
    if not txt:
        return True
    txt = txt.lower()
    padroes = [
        "sem evento",
        "não houve",
        "nenhuma notícia",
        "movimento geral",
        "macro",
        "não há registro",
        "fatores macroeconômicos"
    ]
    return any(p in txt for p in padroes)


# ============================================================
# BUSCA TAVILY
# ============================================================
def coletar_noticias_tavily(ativo: str, data_iso: str):
    empresa = EMPRESAS.get(ativo, ativo)

    query = f"notícias {empresa} {ativo} {data_iso} petróleo brent"

    resp = tavily.search(
        query=query,
        include_raw_content=True,
        search_depth="advanced",
        max_results=4
    )

    textos = []
    fontes = []

    for r in resp.get("results", []):
        if r.get("content"):
            textos.append(f"{r['title']}\n{r['content']}")
            fontes.append(r.get("url", ""))

    return "\n\n".join(textos), fontes


# ============================================================
# GPT PURO
# ============================================================
def consultar_chatgpt_evento(ativo: str, data_iso: str, retorno: float, fechamento: float):

    prompt = f"""
Explique o que ocorreu com o ativo {ativo} no dia {data_iso}.
Use apenas fatos reais. Se não houver evento, diga claramente.

Retorne SOMENTE JSON:

{{
  "data": "{data_iso}",
  "ativo": "{ativo}",
  "retorno_no_dia": {retorno},
  "fechamento": {fechamento},
  "sentimento_do_mercado": "<positivo|negativo|neutro>",
  "o_que_houve": "<máximo 3 frases>",
  "motivos_identificados": ["<mot1>", "<mot2>"],
  "fontes": ["Valor Econômico", "Reuters"]
}}
"""

    resp = client.chat.completions.create(
        model="gpt-4.1",
        response_format={"type": "json_object"},   # MODELO ACEITA
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    # ATENÇÃO: modelo retorna conteúdo JSON em .content
    dados = json.loads(resp.choices[0].message.content)

    # Blindagem
    dados["retorno_no_dia"] = retorno
    dados["fechamento"] = fechamento
    dados["ativo"] = ativo
    dados["data"] = data_iso

    return EventoNoticia(**dados)



# ============================================================
# HÍBRIDO GPT + TAVILY
# ============================================================
def consultar_evento_hibrido(ativo, data_iso, retorno, fechamento):

    # 1) GPT puro primeiro
    evento = consultar_chatgpt_evento(ativo, data_iso, retorno, fechamento)

    # Se GPT escreveu algo minimamente útil → aceitar
    if evento.o_que_houve and len(evento.o_que_houve.strip()) > 10:
        return evento

    print("🟡 GPT não encontrou um evento claro — usando Tavily...")

    # 2) Fallback Tavily
    texto, fontes = coletar_noticias_tavily(ativo, data_iso)
    if not texto:
        print("🔴 Tavily não encontrou nada — utilizando resposta do GPT.")
        return evento

    # 3) GPT com notícias reais
    empresa = EMPRESAS.get(ativo, ativo)

    prompt = f"""
Explique o que ocorreu com o ativo {ativo} ({empresa}) no dia {data_iso}
usando EXCLUSIVAMENTE as notícias abaixo.

Retorne SOMENTE JSON.

NOTÍCIAS:
{texto}
"""

    resp = client.chat.completions.create(
        model="gpt-4.1",
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    dados = json.loads(resp.choices[0].message.content)
    dados["retorno_no_dia"] = retorno
    dados["fechamento"] = fechamento
    dados["ativo"] = ativo
    dados["data"] = data_iso

    return EventoNoticia(**dados)



# ============================================================
# Z-SCORE
# ============================================================
def calcular_zscore(retorno, std):
    if std is None or std <= 0 or np.isnan(std):
        return 0.0
    return float(retorno / std)


def nome_arquivo_evento(ticker: str, data_iso: str):
    ticker_limpo = ticker.replace(".SA", "").replace("=F", "")
    return f"evento_{ticker_limpo}_{data_iso}.json"


# ============================================================
# DETECTAR EVENTOS
# ============================================================
def detectar_eventos(ticker, CAMINHO_DADOS):

    print(f"\n🚀 Rodando eventos para {ticker}...")

    df = pd.read_csv(CAMINHO_DADOS)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    df[f"Ret_{ticker}"] = df[f"Close_{ticker}"].pct_change(fill_method=None) * 100
    df["Ret_BZ"] = df["Close_BZ=F"].pct_change(fill_method=None) * 100

    std_a = df[f"Ret_{ticker}"].expanding().std()
    std_b = df["Ret_BZ"].expanding().std()

    eventos_a = df[df[f"Ret_{ticker}"].abs() > LIMIAR_VARIACAO]
    eventos_b = df[df["Ret_BZ"].abs() > LIMIAR_VARIACAO]

    datas = sorted(set(eventos_a.index) | set(eventos_b.index))

    for data_evt in datas:

        row = df.loc[data_evt]
        data_iso = data_evt.strftime("%Y-%m-%d")

        if data_evt in eventos_a.index and data_evt in eventos_b.index:
            ativo = "AMBOS"
        elif data_evt in eventos_a.index:
            ativo = ticker
        else:
            ativo = "BRENT"

        nome = nome_arquivo_evento(ativo, data_iso)
        out = os.path.join(CAMINHO_SAIDA, nome)

        if os.path.exists(out):
            continue

        print(f"\n📅 {data_iso} | Ativo detectado: {ativo}")

        if ativo == "AMBOS":

            evento_a = consultar_evento_hibrido(ticker, data_iso, row[f"Ret_{ticker}"], row[f"Close_{ticker}"])
            evento_b = consultar_evento_hibrido("BZ=F", data_iso, row["Ret_BZ"], row["Close_BZ=F"])

            registro = {
                "data": data_iso,
                "ativo": "AMBOS",
                "retorno_no_dia": {
                    ticker: row[f"Ret_{ticker}"],
                    "BRENT": row["Ret_BZ"]
                },
                "fechamento": {
                    ticker: row[f"Close_{ticker}"],
                    "BRENT": row["Close_BZ=F"]
                },
                "motivos_identificados": list(
                    dict.fromkeys(evento_a.motivos_identificados + evento_b.motivos_identificados)),
                "fontes": list(dict.fromkeys(evento_a.fontes + evento_b.fontes)),
                f"impacto_d0_{ticker}": row[f"Ret_{ticker}"],
                "impacto_d0_BRENT": row["Ret_BZ"],
                f"zscore_{ticker}": calcular_zscore(row[f"Ret_{ticker}"], std_a.loc[data_evt]),
                "zscore_brent": calcular_zscore(row["Ret_BZ"], std_b.loc[data_evt]),
                "o_que_houve": f"{ticker}: {evento_a.o_que_houve}\nBRENT: {evento_b.o_que_houve}"
            }

            # 🔥 NORMALIZAR TICKERS (REMOVER .SA)
            def normalize(x):
                return x.replace(".SA", "").replace("=F", "")

            # corrigir o retorno e fechamento
            registro["retorno_no_dia"] = {normalize(k): v for k, v in registro["retorno_no_dia"].items()}
            registro["fechamento"] = {normalize(k): v for k, v in registro["fechamento"].items()}

            # corrigir nomes dos impactos
            novo_registro = {}
            for k, v in registro.items():
                if "impacto_d0_" in k:
                    novo_registro["impacto_d0_" + normalize(k[12:])] = v
                else:
                    novo_registro[k] = v
            registro = novo_registro

        else:
            ret = row[f"Ret_{ticker}"] if ativo == ticker else row["Ret_BZ"]
            fech = row[f"Close_{ticker}"] if ativo == ticker else row["Close_BZ=F"]

            evento = consultar_evento_hibrido(ativo, data_iso, ret, fech)

            registro = evento.model_dump()
            registro["ativo"] = ativo.replace(".SA", "")

            registro["impacto_d0"] = ret
            registro["zscore_d0"] = calcular_zscore(
                ret,
                std_a.loc[data_evt] if ativo == ticker else std_b.loc[data_evt]
            )


        with open(out, "w", encoding="utf-8") as f:
            json.dump(registro, f, indent=4, ensure_ascii=False)

        print(f"✅ Salvo: {out}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    CAMINHOS = {
        "PETR4.SA": "../data/dados_petr4_brent.csv",
        "PRIO3.SA": "../data/dados_prio3_brent.csv",
    }

    for ticker, caminho in CAMINHOS.items():
        detectar_eventos(ticker, caminho)
