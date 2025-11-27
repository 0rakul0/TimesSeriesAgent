import pandas as pd


def ver_base(ativo: str):
    """
    Função para ver a base de dados combinada.
    """
    CAMINHO_DADOS = f"../data/dados_{ativo}_brent.csv"
    df = pd.read_csv(CAMINHO_DADOS, index_col=0, parse_dates=True)

    print(df.info())


if __name__ == "__main__":
    for i in ["PETR4", "PRIO3"]:
        ver_base(str(i).lower())