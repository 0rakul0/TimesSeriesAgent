import os
import pandas as pd
import json
from babel.dates import format_date
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from tavily import TavilyClient
from dotenv import load_dotenv
import os

# Carrega variáveis do .env
load_dotenv()


LIMIAR_VARIACAO = 5.0  # % limite para disparar busca de notícias
CAMINHO_DADOS = "../data/dados_combinados.csv"
CAMINHO_SAIDA_EVENTOS = "../ouput_noticias"


class ResumoNoticias(BaseModel):
    data: str = Field(..., description="Data do evento analisado")
    contexto: str = Field(..., description="Contexto político e econômico do evento")
    acontecimento: str = Field(..., description="O que ocorreu com a Petrobras")
    impacto: str = Field(..., description="Impacto sobre a empresa e o mercado")
    fontes: str = Field(..., description="Principais fontes de informação (se houver)")


# ======================================
# 1️⃣ Configuração do cliente tavily e LLM
# ======================================
# Lê a chave da Tavily
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
llm = ChatOllama(model="llama3.2", base_url="http://localhost:11434")


def resumir_trecho(trecho: str, max_tokens: int = 100):
    prompt = ChatPromptTemplate.from_template("""
    Resuma o seguinte trecho de notícia em até {max_tokens} tokens, mantendo os pontos principais:
    {input}
    """)
    chain = prompt | llm
    response = chain.invoke({"input": trecho, "max_tokens": max_tokens})
    return response.content

# ======================================
# 🕵️ 1️⃣ Agente Coletor (usando Tavily)
# ======================================
def coletar_noticias(ativo, data_evento: str):
    """
    Coleta e resume notícias relevantes para o ativo (PETR4 ou Brent) na data especificada.
    """
    if ativo == "PETR4":
        procura = "Petrobras PETR4"
    else:
        procura = "preço do petróleo Brent"

    query = f"notícias sobre {procura} em {data_evento}"
    print(f"🗞️ Buscando notícias: {query}")

    resp = tavily_client.search(
        query=query,
        start_date=data_evento,
        search_depth="advanced",
        max_results=3,
        include_answer=True,
    )

    if not resp.get("results"):
        return "Nenhuma notícia encontrada."

    noticias = []
    for doc in resp["results"]:
        titulo = doc.get("title", "Sem título")
        url = doc.get("url", "")
        trecho = doc.get("content", "")

        # ✅ Resumo automático do trecho
        resumo_trecho = resumir_trecho(trecho)

        noticias.append({
            "titulo": titulo,
            "url": url,
            "resumo": resumo_trecho
        })

    # Retorna formato limpo e compacto
    return "\n\n".join(
        f"Título: {n['titulo']}\nLink: {n['url']}\nResumo: {n['resumo']}"
        for n in noticias
    )

# ======================================
# 😊 2️⃣ Agente Analisador de Sentimentos
# ======================================
def analisar_sentimento(texto: str, preco_atual: float, data: str):
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", f"""
        Você é um analista financeiro especializado no mercado de ações brasileiro.
        Analise as notícias sobre a Petrobras (PETR4) publicadas em {data} e:
        1️⃣ Classifique o sentimento geral (POSITIVO, NEGATIVO ou NEUTRO)
        2️⃣ Estime o impacto percentual no preço do ativo
        3️⃣ Projete o preço futuro considerando o preço atual de R${preco_atual:.2f}
        4️⃣ Justifique brevemente o raciocínio.
        Formato:
        ---
        Sentimento: <positivo|negativo|neutro>
        Impacto estimado: <percentual>
        Preço projetado: R$<valor>
        Justificativa: <texto curto>
        ---
        """),
        ("user", "{input}")
    ])
    chain = prompt_template | llm
    response = chain.invoke({"input": texto})
    return response.content


# ======================================
# 🧾 3️⃣ Agente Resumidor
# ======================================
def resumir_noticias(texto: str, data: str):
    prompt = ChatPromptTemplate.from_template("""
    Você é um jornalista econômico especializado em mercado financeiro brasileiro.
    Resuma as notícias sobre a Petrobras (PETR4) em {data} no formato JSON:
    {{
        "data": "{data}",
        "contexto": "<descrição do contexto político e econômico>",
        "acontecimento": "<o que aconteceu>",
        "impacto": "<impacto sobre a empresa e o mercado>",
        "fontes": "<principais fontes citadas>"
    }}
    Notícias:
    {input}
    """)
    chain = prompt | llm.with_structured_output(ResumoNoticias)
    response = chain.invoke({"input": texto, "data": data})
    return response

def detectar_eventos(caminho_csv: str = CAMINHO_DADOS, limiar: float = LIMIAR_VARIACAO):
    df = pd.read_csv(caminho_csv, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index).normalize()

    col_petr = "Close_PETR4.SA"
    col_brent = "Close_BZ=F"

    # ✅ Calcula variações percentuais
    df["Ret_PETR4"] = df[col_petr].pct_change() * 100
    df["Ret_BZ"] = df[col_brent].pct_change() * 100

    # 🔍 Detecta eventos relevantes
    eventos_petr = df[abs(df["Ret_PETR4"]) > limiar].copy()
    eventos_petr["origem_evento"] = "PETR4"

    eventos_brent = df[abs(df["Ret_BZ"]) > limiar].copy()
    eventos_brent["origem_evento"] = "Brent"

    # Junta e trata duplicatas (eventos simultâneos)
    eventos = pd.concat([eventos_petr, eventos_brent]).sort_index()

    # Se no mesmo dia ocorrer evento em ambos, marca como “Ambos”
    eventos = (
        eventos.groupby(eventos.index)
        .agg({
            col_petr: "first",
            col_brent: "first",
            "Ret_PETR4": "first",
            "Ret_BZ": "first",
            "origem_evento": lambda x: "Ambos" if len(set(x)) > 1 else list(x)[0],
        })
    )

    if eventos.empty:
        print("✅ Nenhum evento relevante encontrado.")
        return

    registros = []
    os.makedirs(CAMINHO_SAIDA_EVENTOS, exist_ok=True)

    for data_evento, linha in eventos.iterrows():
        preco_petr = linha[col_petr]
        preco_brent = linha[col_brent]
        ret_petr = linha["Ret_PETR4"]
        ret_brent = linha["Ret_BZ"]
        origem = linha["origem_evento"]

        # 🗓️ Formata a data no estilo "22 de fevereiro de 2021"
        data_formatada = format_date(data_evento, format="d 'de' MMMM 'de' y", locale='pt_BR')

        print(f"\n🚨 Evento detectado em {data_formatada} ({origem}):")
        print(f"   • PETR4: variação de {ret_petr:.2f}% (preço: R${preco_petr:.2f})")
        print(f"   • Brent: variação de {ret_brent:.2f}% (preço: US${preco_brent:.2f})")

        # 1️⃣ Coleta de notícias
        if origem == "PETR4":
            noticias = coletar_noticias("PETR4", data_evento.strftime("%Y-%m-%d"))
        elif origem == "Brent":
            noticias = coletar_noticias("Brent", data_evento.strftime("%Y-%m-%d"))
        else:  # origem == "Ambos"
            noticias_petr = coletar_noticias("PETR4", data_evento.strftime("%Y-%m-%d"))
            noticias_brent = coletar_noticias("Brent", data_evento.strftime("%Y-%m-%d"))
            noticias = {
                "Petrobras": noticias_petr,
                "Brent": noticias_brent
            }

        # 2️⃣ Análise de sentimento
        analise = analisar_sentimento(noticias, preco_petr, data_formatada)

        # 3️⃣ Resumo (pode retornar objeto BaseModel)
        resumo = resumir_noticias(noticias, data_formatada)
        resumo_dict = resumo.model_dump() if hasattr(resumo, "model_dump") else {"resumo": resumo}

        # 4️⃣ Registro estruturado
        registro = {
            "data": data_evento.strftime("%Y-%m-%d"),
            "data_formatada": data_formatada,
            "origem_evento": origem,
            "preco_petr4": preco_petr,
            "preco_brent": preco_brent,
            "variacao_petr4": ret_petr,
            "variacao_brent": ret_brent,
            "analise": analise,
            "noticias": noticias,
            **resumo_dict
        }
        registros.append(registro)

        # 💾 Salva JSON parcial por rodada
        nome_arquivo_json = os.path.join(
            CAMINHO_SAIDA_EVENTOS,
            f"evento_{data_evento.strftime('%Y-%m-%d')}_{origem}.json"
        )
        with open(nome_arquivo_json, "w", encoding="utf-8") as f:
            json.dump(registro, f, ensure_ascii=False, indent=4)
        print(f"💾 Evento salvo: {nome_arquivo_json}")

    # 💾 Salvar CSV final consolidado
    df_eventos = pd.DataFrame(registros)
    caminho_csv_final = os.path.join(CAMINHO_SAIDA_EVENTOS, "eventos_consolidados.csv")
    df_eventos.to_csv(caminho_csv_final, index=False)
    print(f"\n✅ Eventos consolidados salvos em {caminho_csv_final}")

# ======================================
# 🚀 Função Principal
# ======================================
def main():
    detectar_eventos()


if __name__ == "__main__":
    main()
