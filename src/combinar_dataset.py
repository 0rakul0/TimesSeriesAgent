import pandas as pd

IN_PETR4 = "../data/dados_petr4_brent.csv"
IN_PRIO3 = "../data/dados_prio3_brent.csv"
OUT = "../data/dados_multiativo.csv"

def preparar():
    df1 = pd.read_csv(IN_PETR4, parse_dates=["Date"])
    df2 = pd.read_csv(IN_PRIO3, parse_dates=["Date"])

    df1["Ativo"] = "PETR4.SA"
    df2["Ativo"] = "PRIO3.SA"

    # Unir
    df = pd.concat([df1, df2], ignore_index=True).sort_values(["Date", "Ativo"])

    # Criar colunas genéricas baseadas no ativo
    df["Open"] = df.apply(lambda r: r.get(f"Open_{r['Ativo']}", None), axis=1)
    df["High"] = df.apply(lambda r: r.get(f"High_{r['Ativo']}", None), axis=1)
    df["Low"] = df.apply(lambda r: r.get(f"Low_{r['Ativo']}", None), axis=1)
    df["Close"] = df.apply(lambda r: r.get(f"Close_{r['Ativo']}", None), axis=1)
    df["Volume"] = df.apply(lambda r: r.get(f"Volume_{r['Ativo']}", None), axis=1)
    df["Dividends"] = df.apply(lambda r: r.get(f"Dividends_{r['Ativo']}", None), axis=1)

    # Reordenar colunas
    keep = [
        "Date", "Ativo", "Open", "High", "Low", "Close", "Volume", "Dividends",
        "Open_BZ=F", "High_BZ=F", "Low_BZ=F", "Close_BZ=F",
        "Volume_BZ=F", "Dividends_BZ=F"
    ]
    df = df[keep]

    df = df.dropna(subset=["Close"])

    df.to_csv(OUT, index=False)
    print("✔ Base multi-ativo corrigida salva em:", OUT)


if __name__ == "__main__":
    preparar()
