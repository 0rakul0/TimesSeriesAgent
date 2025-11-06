"""
aqui fica todas as ferramentas relacionadas a noticias, busca, resumo, etc.
o link alvo é o https://www.infomoney.com.br/cotacoes/b3/acao/petrobras-petr4/
"""

import requests
from bs4 import BeautifulSoup
from langchain.tools import tool
from pydantic import BaseModel, Field
from langchain_core.utils.function_calling import convert_to_openai_function


class RetnornoBuscaNoticias(BaseModel):
    noticias: list[dict] = Field(
        ...,
        description="Lista de notícias encontradas, cada notícia deve conter 'titulo', 'link' e 'resumo'."
    )

# ferramenta
@tool(args_schema=RetnornoBuscaNoticias)
def busca_noticias_tool(query: str, max_results: int = 5) -> list[dict]:
    """
    aqui ele vai buscar alguma coisa relacionada a noticias no site alvo
    :param query:
    :param max_results:
    :return:
    """


# retorno da ferramenta
tools = [busca_noticias_tool]
tools_json = [convert_to_openai_function(tool) for tool in tools]
tool_run = {tool.name: tool for tool in tools}