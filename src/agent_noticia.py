"""
nesse script, implementamos um agente simples que coleta notícias de um site de notícias.
Usamos a biblioteca requests para fazer requisições HTTP e BeautifulSoup para analisar o HTML e extrair as notícias.
"""

# agente 1: Coletor de notícias
# agente 2: Analisador de sentimentos
# agente 3: Resumidor de notícias


import requests
from bs4 import BeautifulSoup as bs

url = "https://economia.uol.com.br/cotacoes/noticias/redacao/2021/02/22/acoes-petrobras-bolsa-de-valores.htm"

def coletar_noticias(url):
    response = requests.get(url)

    print(response.content)


if __name__ == "__main__":
    noticias = coletar_noticias(url)
    for noticia in noticias:
        print(f"Título: {noticia['titulo']}\nLink: {noticia['link']}\n")