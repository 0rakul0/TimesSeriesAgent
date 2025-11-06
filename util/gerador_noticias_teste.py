import numpy as np
import pandas as pd
import random

def gerar_dataset_noticias(dados_acao: pd.DataFrame, arquivo_saida="noticias_sinteticas.csv"):
    """
    Gera um dataset de notícias sintéticas compatível com as datas do yfinance.
    Pode haver múltiplas notícias por dia (1 a 4).
    """
    np.random.seed(42)
    random.seed(42)

    datas = dados_acao.index
    fontes = ["Reuters", "Valor Econômico", "Exame", "Estadão", "Bloomberg", "Infomoney"]
    categorias = ["mercado", "produção", "governança", "ambiental", "financeira", "internacional"]

    manchetes_pos = [
        "Empresa anuncia aumento de produção",
        "Licença ambiental concedida para novo campo",
        "Lucro líquido supera expectativas",
        "Expansão internacional aprovada",
        "Novo contrato bilionário fechado"
    ]

    manchetes_neg = [
        "Produção interrompida devido a greve",
        "Processo judicial afeta projeções",
        "Queda de receita surpreende analistas",
        "Problemas técnicos em plataforma offshore",
        "Multa milionária aplicada por órgão regulador"
    ]

    manchetes_neutras = [
        "Relatório trimestral divulgado ao mercado",
        "Reunião de acionistas marcada para próxima semana",
        "Empresa mantém guidance para o ano",
        "Executivo comenta perspectivas do setor",
        "Mercado aguarda decisão de órgão regulador"
    ]

    linhas = []
    for data in datas:
        n_noticias = np.random.randint(1, 5)  # de 1 a 4 notícias por dia
        for _ in range(n_noticias):
            polaridade = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
            if polaridade == 1:
                titulo = random.choice(manchetes_pos)
            elif polaridade == -1:
                titulo = random.choice(manchetes_neg)
            else:
                titulo = random.choice(manchetes_neutras)

            peso_impacto = round(np.random.uniform(0.1, 1.0), 2)
            categoria = random.choice(categorias)
            fonte = random.choice(fontes)
            resumo = f"Notícia sobre {categoria} publicada pela {fonte}."

            linhas.append({
                "data": data,
                "titulo": titulo,
                "fonte": fonte,
                "polaridade": polaridade,
                "peso_impacto": peso_impacto,
                "categoria": categoria,
                "resumo": resumo
            })

    df_noticias = pd.DataFrame(linhas)
    df_noticias.to_csv(arquivo_saida, index=False)
    print(f"✅ Dataset de notícias salvo em: {arquivo_saida} ({len(df_noticias)} registros)")
    return df_noticias


# Exemplo de uso:
if __name__ == "__main__":
    import yfinance as yf
    dados = yf.Ticker("PETR4.SA").history(period="5y")
    df_noticias = gerar_dataset_noticias(dados)
    print(df_noticias.head(10))
