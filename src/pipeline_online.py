import pandas as pd
import yfinance as yf

from src.agent_noticia import detectar_eventos, remover_jsons_sem_motivos
from src.mvp import executar_demo
from src.seq_eventos import gerar_sequencias_eventos


def atualizar_csv_d0(csv_path: str, ticker: str):
    print("\n=== Atualizando dados do ativo ===")

    df = pd.read_csv(csv_path, parse_dates=["Date"]).sort_values("Date")
    ultimo = df["Date"].max().normalize()

    dados = yf.download(ticker, period="5d", auto_adjust=False, progress=False)
    if dados.empty:
        print("Nenhum dado novo retornado pelo Yahoo Finance.")
        return False

    dados = dados.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
    novos = dados[dados["Date"] > ultimo]

    if novos.empty:
        print("Nenhum novo dia disponivel.")
        return False

    df2 = pd.concat([df, novos], ignore_index=True)
    df2.to_csv(csv_path, index=False)
    print(f"[OK] CSV atualizado com {len(novos)} novos registros.")
    return True


def etapa_noticias(ticker: str, csv_path: str):
    print("\n=== Detectando eventos nas noticias ===")
    detectar_eventos(ticker, csv_path)
    remover_jsons_sem_motivos()
    gerar_sequencias_eventos()
    print("[OK] Eventos atualizados.")


def ticker_para_ativo(ticker: str) -> str:
    return ticker.replace(".SA", "").replace("=F", "")


def etapa_previsao(ticker: str, csv_path: str):
    print("\n=== Rodando previsao MVP online ===")
    ativo = ticker_para_ativo(ticker)
    resultado = executar_demo(
        retornar_html=True,
        ativo=ativo,
        csv_path=csv_path,
        show_plot=False,
    )
    print("[OK] Previsao gerada.")
    return resultado["html"]


def run_online(csv_path: str, ticker: str):
    print("\n====================================")
    print("   TimesSeriesAgent - MODO ONLINE")
    print("====================================")

    atualizar_csv_d0(csv_path, ticker)
    etapa_noticias(ticker, csv_path)
    html_plot = etapa_previsao(ticker, csv_path)

    print("\n[OK] Pipeline online finalizado.\n")
    return html_plot
