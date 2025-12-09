# pipeline_online.py
import os
import pandas as pd
import yfinance as yf
from datetime import datetime
from src.agent_noticia import detectar_eventos, remover_jsons_sem_motivos
from src.seq_eventos import gerar_sequencias_eventos
from src.mvp import executar_demo  # MVP retorna gráfico automaticamente


# -------------------------------------------------------
# 1) Atualizar CSV com preço real do dia
# -------------------------------------------------------
def atualizar_csv_d0(csv_path: str, ticker: str):
    print("\n=== Atualizando dados do ativo ===")

    df = pd.read_csv(csv_path, parse_dates=["Date"]).sort_values("Date")
    ultimo = df["Date"].max().normalize()

    dados = yf.download(ticker, period="5d")
    dados = dados.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]

    novos = dados[dados["Date"] > ultimo]

    if novos.empty:
        print("Nenhum novo dia disponível.")
        return False

    df2 = pd.concat([df, novos], ignore_index=True)
    df2.to_csv(csv_path, index=False)
    print(f"✔ CSV atualizado com {len(novos)} novos registros.")
    return True


# -------------------------------------------------------
# 2) Rodar agente de notícias
# -------------------------------------------------------
def etapa_noticias(ticker: str, csv_path: str):
    print("\n=== Detectando eventos nas notícias ===")
    detectar_eventos(ticker, csv_path)
    remover_jsons_sem_motivos()
    gerar_sequencias_eventos()
    print("✔ Eventos atualizados.")


# -------------------------------------------------------
# 3) Executar MVP e retornar o HTML do gráfico
# -------------------------------------------------------
def etapa_previsao():
    print("\n=== Rodando previsão MVP online ===")
    caminho_html = executar_demo(retornar_html=True)  # ajuste necessário no MVP
    print("✔ Previsão gerada.")
    return caminho_html


# -------------------------------------------------------
# Função geral que pode ser chamada via FastAPI
# -------------------------------------------------------
def run_online(csv_path: str, ticker: str):
    print("\n====================================")
    print("   🚀 TimesSeriesAgent – MODO ONLINE")
    print("====================================")

    atualizar_csv_d0(csv_path, ticker)
    etapa_noticias(ticker, csv_path)
    html_plot = etapa_previsao()

    print("\n🎯 Pipeline online finalizado!\n")
    return html_plot
