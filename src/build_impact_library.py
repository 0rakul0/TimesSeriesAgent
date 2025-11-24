"""
===========================================
Construção da Biblioteca de Impacto (Impact Library)
===========================================

Este script processa dados de preços e eventos de mercado para gerar uma
biblioteca histórica de impactos (“impact_library.json”). Essa biblioteca
é utilizada pelo modelo híbrido para replicar o efeito de notícias reais
nos preços futuros de cada ativo.

O algoritmo executa as seguintes etapas principais:

1) CARREGAR MÚLTIPLAS BASES DE PREÇOS AUTOMATICAMENTE
   - Lê todos os CSVs definidos na lista CSV_FILES.
   - Junta todos em um único DataFrame, alinhando por data.
   - Identifica automaticamente quais ativos existem nas colunas
     (ex.: PETR4.SA, BZ=F, PRIO3.SA).

2) IDENTIFICAR E CARREGAR EVENTOS
   - Lê todos os arquivos evento_*.json em output_noticias/.
   - Extrai datas de eventos e motivos identificados.

3) EXTRAÇÃO DA SEQUÊNCIA REAL DE IMPACTO (D0 → Dn)
   - Para cada evento e para cada ativo relevante:
       • extrai a sequência de retornos reais a partir do dia da notícia
       • segue pelos próximos dias (até HORIZON)
       • interrompe caso outra notícia ocorra (sem sobreposição)
       • respeita apenas dias úteis presentes no DataFrame

4) AGRUPAMENTO EM CLUSTERS
   - Cada evento é atribuído a um cluster simples baseado no ativo
     e no sinal do impacto no D0 (positivo, negativo ou neutro).
   - Exemplo de clusters:
        posi_petr4, neg_petr4, posi_brent, neg_brent, posi_prio3, ...

5) AGREGAÇÃO DAS SEQUÊNCIAS
   - Para cada cluster:
       • junta todas as sequências extraídas
       • normaliza comprimento (preenchimento para alinhamento)
       • calcula a mediana por dia (ref_seq)
   - Essa "ref_seq" é o padrão histórico que será usado pelo modelo
     para replicar impactos futuros.

6) SALVAR A BIBLIOTECA FINAL
   - Gera o arquivo impact_library.json com:
       • sequência de referência (ref_seq)
       • número de eventos no cluster
       • exemplos utilizados na geração

-------------------------------------------
Funções principais
-------------------------------------------

detectar_ativos(df):
    Identifica automaticamente quais ativos existem no DataFrame de preços.

extrair_seq_sem_sobreposicao(df, data_evt, ativo, eventos_datas, horizon):
    Extrai a sequência real de retornos após o evento, parando se houver
    outra notícia subsequente.

derivar_cluster(ativo, impacto_d0):
    Atribui o evento a um cluster simples baseado no ativo e no sinal
    do impacto do dia da notícia.

build_library():
    Orquestra todo o processo:
       - lê CSVs
       - carrega eventos
       - extrai sequências por ativo
       - agrupa em clusters
       - gera a biblioteca final de impacto.

===========================================
"""


import os
import json
import numpy as np
import pandas as pd
from glob import glob

# =========================================
# CONFIG
# =========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
EVENTS_DIR = os.path.join(BASE_DIR, "output_noticias")

# 🆕 VÁRIOS CSVs AUTOMÁTICOS
CSV_FILES = [
    os.path.join(DATA_DIR, "dados_petr4_brent.csv"),
    os.path.join(DATA_DIR, "dados_prio3_brent.csv")
]

OUT_FILE = os.path.join(DATA_DIR, "impact_library.json")

HORIZON = 5


# =========================================
# 1. Identificar ativos presentes no CSV
# =========================================

def detectar_ativos(df):
    ativos = set()

    for col in df.columns:
        if col.startswith("Close_"):
            ticker = col.replace("Close_", "")
            ativos.add(ticker)

    return sorted(ativos)


# =========================================
# 2. Extrair sequência sem sobreposição
# =========================================

def extrair_seq_sem_sobreposicao(df, data_evt, ativo, eventos_datas, horizon=HORIZON):
    col = f"Close_{ativo}"

    if col not in df.columns:
        print(f"[ERRO] Coluna não encontrada: {col}")
        return None

    data_evt = pd.to_datetime(data_evt).normalize()
    eventos_datas = sorted([pd.to_datetime(x).normalize() for x in eventos_datas])

    # próximo evento após este
    proximos = [d for d in eventos_datas if d > data_evt]
    if proximos:
        proximo_evento = proximos[0]
    else:
        proximo_evento = df.index.max()

    seq = []
    dia_atual = data_evt

    for k in range(horizon + 1):

        if dia_atual > proximo_evento:
            break

        while dia_atual not in df.index:
            dia_atual += pd.Timedelta(days=1)
            if dia_atual > proximo_evento or dia_atual > df.index.max():
                break

        if dia_atual not in df.index or dia_atual > proximo_evento:
            break

        dia_anterior = dia_atual - pd.Timedelta(days=1)
        while dia_anterior not in df.index:
            dia_anterior -= pd.Timedelta(days=1)
            if dia_anterior < df.index.min():
                return None

        ret = (df.loc[dia_atual, col] - df.loc[dia_anterior, col]) / df.loc[dia_anterior, col] * 100
        seq.append(float(ret))

        dia_atual += pd.Timedelta(days=1)

    return seq


# =========================================
# 3. Cluster simples
# =========================================

def derivar_cluster(ativo: str, impacto_d0: float) -> str:

    base = ativo.replace(".SA", "").replace("=F", "").lower()

    if impacto_d0 == 0 or pd.isna(impacto_d0):
        return f"outros_{base}"

    return f"posi_{base}" if impacto_d0 > 0 else f"neg_{base}"


# =========================================
# 4. Construir biblioteca
# =========================================

def build_library():

    print("📌 Iniciando construção da biblioteca...")

    # 👉 JUNTA TODOS OS CSVs EM UM ÚNICO DF
    dfs = []
    for f in CSV_FILES:
        if os.path.exists(f):
            d = pd.read_csv(f, index_col=0, parse_dates=True)
            d.index = pd.to_datetime(d.index).normalize()
            dfs.append(d)
            print("✔ Lido:", f)

    if not dfs:
        print("❌ Nenhum CSV encontrado.")
        return

    # 🔥 merge automático de todos os ativos
    df = pd.concat(dfs, axis=1).sort_index()
    df = df.loc[:, ~df.columns.duplicated()]  # remove duplicatas

    # identificar ativos reais no DF
    ativos = detectar_ativos(df)
    print("\n📌 Ativos detectados:", ativos)

    # carregar eventos
    files = sorted(glob(os.path.join(EVENTS_DIR, "evento_*.json")))

    if not files:
        print("❌ Nenhum evento encontrado.")
        return

    eventos = []
    datas_eventos = []

    for path in files:
        j = json.load(open(path, "r", encoding="utf-8"))
        eventos.append(j)
        datas_eventos.append(pd.to_datetime(j["data"]).normalize())

    datas_eventos = sorted(list(set(datas_eventos)))

    clusters = {}

    # =========================================
    # PROCESSAR EVENTOS
    # =========================================
    for j in eventos:

        data_evt = pd.to_datetime(j["data"]).normalize()
        ativo_evt = j["ativo"].upper()

        # Evento múltiplo?
        if ativo_evt == "AMBOS":
            ativos_evento = ativos  # todos os ativos existentes
        else:
            ativos_evento = [ativo_evt]

        # para cada ativo
        for atv in ativos_evento:

            seq = extrair_seq_sem_sobreposicao(df, data_evt, atv, datas_eventos)

            if not seq:
                print(f"[WARN] Sequência vazia para {atv} em {data_evt.date()}")
                continue

            # impacto d0 vem do JSON corrigido
            impacto = None
            if ativo_evt == "AMBOS":
                impacto = (
                    j.get("impacto_d0_PETR4") if atv.startswith("PETR4") else
                    j.get("impacto_d0_BRENT") if atv.startswith("BZ") else
                    j.get("impacto_d0")  # fallback
                )
            else:
                impacto = j.get("impacto_d0", 0)

            cluster = derivar_cluster(atv, impacto)

            clusters.setdefault(cluster, []).append({
                "data": str(data_evt.date()),
                "ativo": atv,
                "motivos": j.get("motivos_identificados", []),
                "seq": seq,
                "d0": seq[0]
            })

    # =========================================
    # AGREGAR E SALVAR
    # =========================================

    lib = {}

    for c, items in clusters.items():
        arr = np.array([it["seq"] for it in items], dtype=object)
        max_len = max(len(x) for x in arr)
        arr_fixed = np.array([x + [0.0] * (max_len - len(x)) for x in arr], dtype=float)

        lib[c] = {
            "ref_seq": np.median(arr_fixed, axis=0).tolist(),
            "n_events": len(items),
            "examples": items[:5]
        }

    json.dump(lib, open(OUT_FILE, "w", encoding="utf-8"), indent=4, ensure_ascii=False)

    print("\n✅ Biblioteca criada com sucesso!")
    print("→", OUT_FILE)


if __name__ == "__main__":
    build_library()
