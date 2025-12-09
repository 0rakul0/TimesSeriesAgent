"""
===========================================================
Processamento de Eventos – Extração da Sequência Real (D0→D5)
===========================================================

Este script lê os eventos gerados pelo sistema de notícias
(arquivos evento_*.json) e adiciona a cada evento a sequência
real de retornos do ativo nos dias seguintes, formando o vetor:

    seq = [D0, D1, D2, D3, D4, D5]

onde:
- D0 é o retorno do ativo no dia da notícia;
- D1–D5 são os retornos dos cinco pregões seguintes;
- a sequência é interrompida automaticamente caso exista
  outra notícia que afete o mesmo ativo antes do D5.

O objetivo final é que cada evento JSON contenha uma sequência
realista de impacto futuro, permitindo calcular médias, clusters,
padrões de propagação e uso no modelo híbrido.

-----------------------------------------------------------
Descrição das Funções
-----------------------------------------------------------

escolher_csv_para_ativo(ativo)
    (NÃO USADA PARA BRENT)
    Para ativos PETR4 e PRIO3, retorna o caminho do CSV que
    contém o histórico desse ativo. Para BRENT, usamos outra
    lógica (multi-DataFrame).

calc_retorno(df, col, dia, dia_ant)
    Calcula o retorno percentual entre dois dias consecutivos:
        (close_dia - close_dia_anterior) / close_dia_anterior * 100
    Se faltar dado em algum dos dias, retorna None.

extrair_seq(df, data_evt, col, outras_datas)
    Extrai a sequência D0→D5 para um ativo específico.
    - começa no dia da notícia (D0)
    - avança um pregão por vez
    - interrompe se encontrar outra notícia futura (cortes)
    - retorna uma lista de retornos reais

limpar_evento(j)
    Remove campos redundantes do JSON de evento, mantendo apenas:
        data
        ativo
        retorno_no_dia
        motivos_identificados
        sentimento_do_mercado
        fontes
        o_que_houve
    A função garante que o evento fique limpo e padronizado,
    mantendo apenas informações essenciais + seq.

-----------------------------------------------------------
Lógica Principal (gerar_sequencias_eventos)
-----------------------------------------------------------

1) Carrega todos os CSVs de ativos (PETR4, PRIO3).
2) Identifica automaticamente quais DataFrames possuem BRENT.
   (o BRENT aparece em vários CSVs)
3) Carrega todos os eventos e extrai suas datas para uso no corte.
4) Para cada evento:
       - identifica se é PETR4, PRIO3, BRENT ou AMBOS
       - se for AMBOS, processa cada ativo envolvido separadamente
       - extrai a sequência real D0→D5 de cada ativo
       - interrompe a sequência se aparecer outra notícia
       - gera j["seq"] = { ativo : [D0,D1,...] }
       - limpa o evento, removendo campos obsoletos
       - salva o evento atualizado no arquivo original
5) Todos os JSONs são atualizados automaticamente
   com as sequências reais e estrutura padronizada.

Resultado:
Cada arquivo evento_*.json passa a conter:

{
    "data": "AAAA-MM-DD",
    "ativo": "PETR4" | "PRIO3" | "BRENT" | "AMBOS",
    "retorno_no_dia": ...,
    "motivos_identificados": [...],
    "sentimento_do_mercado": "...",
    "fontes": [...],
    "o_que_houve": "...",
    "seq": {
        "PETR4": [...],
        "BRENT": [...],
        "PRIO3": [...]
    }
}

===========================================================
"""


import os
import json
import pandas as pd
import numpy as np
from glob import glob

# =========================================
# CONFIG
# =========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
EVENTS_DIR = os.path.join(BASE_DIR, "output_noticias")

# CSVs existentes (adicione aqui quando tiver novos)
CSV_MAP = {
    "PETR4": os.path.join(DATA_DIR, "dados_petr4_brent.csv"),
    "PRIO3": os.path.join(DATA_DIR, "dados_prio3_brent.csv"),
    "EXXO34": os.path.join(DATA_DIR, "dados_exxo34_brent.csv"),
}

# Mapa de colunas corretas
COL_MAP = {
    "PETR4": "Close_PETR4.SA",
    "PRIO3": "Close_PRIO3.SA",
    "EXXO34": "Close_EXXO34.SA",
    "BRENT": "Close_BZ=F"
}

HORIZON = 5

# =========================================
# PRÉ-CARREGAR TODOS OS CSVs EM MEMÓRIA
# =========================================
DF_MAP = {}

for ativo, path in CSV_MAP.items():
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index).normalize()
        DF_MAP[ativo] = df

# localizar onde o BRENT aparece
BRENT_DFS = []
for nome, df in DF_MAP.items():
    if "Close_BZ=F" in df.columns:
        BRENT_DFS.append(df)

if BRENT_DFS:
    DF_MAP["BRENT"] = BRENT_DFS  # lista de vários DataFrames


# =========================================
# Função: extrair retorno diário
# =========================================
def calc_retorno(df, col, dia, dia_ant):
    try:
        close = df.loc[dia, col]
        close_ant = df.loc[dia_ant, col]
        if pd.isna(close) or pd.isna(close_ant):
            return None
        return float((close - close_ant) / close_ant * 100)
    except:
        return None


# =========================================
# Função: gerar seq D0→D5
# =========================================
def extrair_seq(df, data_evt, col, outras_datas):
    seq = []
    dia_atual = data_evt

    for i in range(HORIZON + 1):

        if i > 0 and dia_atual in outras_datas:
            break

        while dia_atual not in df.index:
            dia_atual += pd.Timedelta(days=1)
            if dia_atual > df.index.max():
                break

        if dia_atual not in df.index:
            break

        dia_ant = dia_atual - pd.Timedelta(days=1)
        while dia_ant not in df.index:
            dia_ant -= pd.Timedelta(days=1)
            if dia_ant < df.index.min():
                return seq if seq else None

        ret = calc_retorno(df, col, dia_atual, dia_ant)
        if ret is None:
            return seq if seq else None

        seq.append(ret)
        dia_atual += pd.Timedelta(days=1)

    return seq


# =========================================
# Limpeza do evento
# =========================================
def limpar_evento(j):
    CAMPOS_ESSENCIAIS = {
        "data",
        "ativo",
        "retorno_no_dia",
        "motivos_identificados",
        "sentimento_do_mercado",
        "fontes",
        "o_que_houve",
    }

    novo = {}

    for k in CAMPOS_ESSENCIAIS:
        if k in j:
            novo[k] = j[k]

    return novo


# =========================================
# MAIN PROCESS
# =========================================
def gerar_sequencias_eventos():
    arquivos = sorted(glob(os.path.join(EVENTS_DIR, "evento_*.json")))

    if not arquivos:
        print("Nenhum evento encontrado.")
        return

    # datas de todos os eventos (para corte da seq)
    datas_eventos = [
        pd.to_datetime(json.load(open(path, "r", encoding="utf-8"))["data"]).normalize()
        for path in arquivos
    ]
    datas_eventos = sorted(datas_eventos)

    for path in arquivos:
        with open(path, "r", encoding="utf-8") as f:
            j = json.load(f)

        data_evt = pd.to_datetime(j["data"]).normalize()
        ativo_evt = j["ativo"].upper()

        seq_final = {}

        # caso AMBOS → múltiplos ativos
        ativos_relacionados = [ativo_evt]

        for ativo in ativos_relacionados:

            col_close = COL_MAP.get(ativo)
            if col_close is None:
                print(f"[ERRO] Ativo sem coluna mapeada: {ativo}")
                continue

            # caso BRENT → múltiplos DFs
            if ativo == "BRENT":

                if "BRENT" not in DF_MAP:
                    print(f"[ERRO] Não encontrei DFs para BRENT")
                    seq_final[ativo] = []
                    continue

                seq = None
                for df in DF_MAP["BRENT"]:
                    if data_evt in df.index:
                        seq = extrair_seq(df, data_evt, col_close, datas_eventos)
                        if seq:
                            break

                seq_final[ativo] = seq if seq else []
                continue

            # PETR4 / PRIO3 → DF único
            df = DF_MAP.get(ativo)
            if df is None:
                print(f"[ERRO] DF não carregado para ativo {ativo}")
                seq_final[ativo] = []
                continue

            if col_close not in df.columns:
                print(f"[ERRO] Coluna {col_close} não encontrada no DF de {ativo}")
                seq_final[ativo] = []
                continue

            seq = extrair_seq(df, data_evt, col_close, datas_eventos)
            seq_final[ativo] = seq if seq else []

        # limpar json
        j_limpo = limpar_evento(j)

        # adicionar seq
        j_limpo["seq"] = seq_final

        # salvar
        with open(path, "w", encoding="utf-8") as f:
            json.dump(j_limpo, f, indent=4, ensure_ascii=False)

        print(f"✔ Seq added & cleaned: {ativo_evt} {data_evt.date()} → {seq_final}")

    print("\n🎯 Finalizado: todos os eventos atualizados com seq e JSON limpo.")


if __name__ == "__main__":
    gerar_sequencias_eventos()
