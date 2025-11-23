import os
import json
import pandas as pd
import numpy as np
from glob import glob

# ==========================================
# CONFIG
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
EVENTOS_DIR = os.path.join(BASE_DIR, "output_noticias")

ARQ_DADOS = os.path.join(DATA_DIR, "dados_combinados.csv")

# ==========================================
# FUNÇÃO AUXILIAR
# ==========================================

def calcular_zscore(valor, std):
    if std is None or std <= 0 or np.isnan(std):
        std = 0.0001
    return float(valor / std)


# ==========================================
# CARREGAR BASE DE PREÇOS
# ==========================================

df = pd.read_csv(ARQ_DADOS)

if df.columns[0] != "Date":
    df.columns = ["Date"] + list(df.columns[1:])

df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)

df["Ret_PETR4"] = df["Close_PETR4.SA"].pct_change(fill_method=None) * 100
df["Ret_BZ"] = df["Close_BZ=F"].pct_change(fill_method=None) * 100

std_petr4 = df["Ret_PETR4"].expanding().std()
std_brent = df["Ret_BZ"].expanding().std()


# ==========================================
# PROCESSAR JSONS
# ==========================================

def corrigir_eventos():
    arquivos = glob(os.path.join(EVENTOS_DIR, "evento_*.json"))
    print(f"Encontrados {len(arquivos)} eventos para corrigir.\n")

    for path in arquivos:
        with open(path, "r", encoding="utf-8") as f:
            j = json.load(f)

        data_str = j["data"]
        ativo = j["ativo"].upper().strip()
        data = pd.to_datetime(data_str)

        if data not in df.index:
            print(f"⚠ Data fora do range {data_str}, pulando.")
            continue

        # =====================================================
        # CASO ESPECIAL — EVENTO COM ATIVO = "AMBOS"
        # =====================================================
        if ativo == "AMBOS":

            r_petr4 = df.loc[data, "Ret_PETR4"]
            r_brent = df.loc[data, "Ret_BZ"]

            f_petr4 = float(df.loc[data, "Close_PETR4.SA"])
            f_brent = float(df.loc[data, "Close_BZ=F"])

            z_petr4 = calcular_zscore(r_petr4, std_petr4.loc[data])
            z_brent = calcular_zscore(r_brent, std_brent.loc[data])

            j_corrigido = {
                "data": data_str,
                "ativo": "AMBOS",
                "retorno_no_dia": {
                    "PETR4": r_petr4,
                    "BRENT": r_brent
                },
                "fechamento": {
                    "PETR4": f_petr4,
                    "BRENT": f_brent
                },
                "sentimento_do_mercado": j.get("sentimento_do_mercado", "neutro"),
                "o_que_houve": j.get("o_que_houve", ""),
                "motivos_identificados": j.get("motivos_identificados", []),
                "fontes": j.get("fontes", []),

                "impacto_d0_PETR4": r_petr4,
                "impacto_d0_BRENT": r_brent,
                "zscore_petr4": z_petr4,
                "zscore_brent": z_brent
            }

            with open(path, "w", encoding="utf-8") as f:
                json.dump(j_corrigido, f, indent=4, ensure_ascii=False)

            print(f"✔ Corrigido (AMBOS): {os.path.basename(path)}")
            continue

        # =====================================================
        # EVENTOS PETR4 / BRENT INDIVIDUAIS
        # =====================================================

        if ativo == "PETR4":
            retorno = df.loc[data, "Ret_PETR4"]
            fechamento = float(df.loc[data, "Close_PETR4.SA"])
            z = calcular_zscore(retorno, std_petr4.loc[data])

        elif ativo == "BRENT":
            retorno = df.loc[data, "Ret_BZ"]
            fechamento = float(df.loc[data, "Close_BZ=F"])
            z = calcular_zscore(retorno, std_brent.loc[data])

        else:
            print(f"⚠ Ativo desconhecido em {path}, pulando.")
            continue

        j_corrigido = {
            "data": data_str,
            "ativo": ativo,
            "retorno_no_dia": retorno,
            "fechamento": fechamento,
            "sentimento_do_mercado": j.get("sentimento_do_mercado", "neutro"),
            "o_que_houve": j.get("o_que_houve", ""),
            "motivos_identificados": j.get("motivos_identificados", []),
            "fontes": j.get("fontes", []),
            "impacto_d0": retorno,
            "zscore_d0": z
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(j_corrigido, f, indent=4, ensure_ascii=False)

        print(f"✔ Corrigido: {os.path.basename(path)}")

    print("\n🎉 Correção concluída!")


if __name__ == "__main__":
    corrigir_eventos()
