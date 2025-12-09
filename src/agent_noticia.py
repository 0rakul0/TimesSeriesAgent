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
LIMIAR_VARIACAO = 1.9

os.makedirs(CAMINHO_SAIDA, exist_ok=True)

client = OpenAI()
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# ============================================================
# MAPA PARA NOME DE EMPRESA
# ============================================================
EMPRESAS = {
    "PETR4.SA": "Petrobras",
    "PRIO3.SA": "PetroRio",
    "EXXO34.SA": "ExxonMobil",
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
# FUNÇÃO: Escolher o motivo principal entre vários
# ============================================================
def escolher_motivo_principal(motivos: List[str]) -> List[str]:
    """
    Dado um conjunto de motivos, usa GPT para escolher somente o mais importante.
    Retorna uma lista com apenas 1 motivo.
    """

    # Se só há 1 motivo, nada a fazer
    if not motivos or len(motivos) == 1:
        return motivos

    prompt = f"""
Dentre os motivos abaixo, escolha apenas aquele que representa a causa PRINCIPAL 
do movimento do ativo no mercado. Responda somente com o texto exato do motivo.

MOTIVOS:
{json.dumps(motivos, ensure_ascii=False, indent=2)}
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",  # versão mais barata e ideal para tarefa simples
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        escolha = resp.choices[0].message.content.strip()

        # Garantir que seja exatamente um dos motivos originais
        for m in motivos:
            if escolha.lower() in m.lower():
                return [m]

        # fallback: retorna o primeiro
        return [motivos[0]]

    except Exception as e:
        print(f"⚠ Erro ao escolher motivo principal: {e}")
        return [motivos[0]]


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

    dados["motivos_identificados"] = escolher_motivo_principal(dados.get("motivos_identificados", []))

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

        if data_evt in eventos_a.index:
            ativo = ticker
        else:
            ativo = "BRENT"

        nome = nome_arquivo_evento(ativo, data_iso)
        out = os.path.join(CAMINHO_SAIDA, nome)

        if os.path.exists(out):
            continue

        print(f"\n📅 {data_iso} | Ativo detectado: {ativo}")

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
# FUNÇÃO: REMOVER JSONs QUE NÃO TÊM MOTIVOS IDENTIFICADOS
# ============================================================
def remover_jsons_sem_motivos(caminho_saida=CAMINHO_SAIDA):
    print("\n🧹 Limpando eventos inválidos (sem motivos_identificados)...")

    arquivos = [
        os.path.join(caminho_saida, f)
        for f in os.listdir(caminho_saida)
        if f.endswith(".json")
    ]

    removidos = 0

    for arq in arquivos:
        try:
            with open(arq, "r", encoding="utf-8") as f:
                dados = json.load(f)

            # Se o campo não existe ou está vazio → remover
            if "motivos_identificados" not in dados or not dados["motivos_identificados"]:
                print(f"🗑 Removendo {os.path.basename(arq)} (sem motivos)")
                os.remove(arq)
                removidos += 1

        except Exception as e:
            # Caso arquivo corrompido → remove também
            print(f"⚠ Erro ao ler {arq}: {e}. Removendo arquivo.")
            os.remove(arq)
            removidos += 1

    print(f"✔ Limpeza concluída. Arquivos removidos: {removidos}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    CAMINHOS = {
        "PETR4.SA": "../data/dados_petr4_brent.csv",
        "PRIO3.SA": "../data/dados_prio3_brent.csv",
        "EXXO34.SA": "../data/dados_exxo34_brent.csv",
    }

    for ticker, caminho in CAMINHOS.items():
        detectar_eventos(ticker, caminho)

    remover_jsons_sem_motivos()

