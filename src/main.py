from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from pipeline_online import run_online
from src.mvp import executar_demo


app = FastAPI(
    title="TimesSeriesAgent Online API",
    description="API do sistema híbrido com previsão D+1..D+3",
    version="2.0"
)


@app.get("/")
async def root():
    return {"status": "online", "msg": "TimesSeriesAgent API ativa!"}


@app.get("/run")
async def run_predict(
    ticker: str = Query("PETR4.SA"),
    csv_path: str = Query("../data/dados_petr4_brent.csv")
):
    """
    Executa o pipeline ONLINE:
    1) Atualiza CSV
    2) Busca notícias novas
    3) Gera sequências
    4) Roda MVP
    Retorna: gráfico HTML gerado
    """
    html_path = run_online(csv_path, ticker)
    return FileResponse(html_path, media_type="text/html")


# atalhos
@app.get("/run/petr4")
async def run_petr4():
    html_path = run_online("../data/dados_petr4_brent.csv", "PETR4.SA")
    return FileResponse(html_path, media_type="text/html")


@app.get("/run/prio3")
async def run_prio3():
    html_path = run_online("../data/dados_prio3_brent.csv", "PRIO3.SA")
    return FileResponse(html_path, media_type="text/html")


@app.get("/run/exxo34")
async def run_exxo34():
    html_path = run_online("../data/dados_exxo34_brent.csv", "EXXO34.SA")
    return FileResponse(html_path, media_type="text/html")


@app.get("/predict_and_plot/petr4")
def predict_plot(ativo: str = "PETR4", csv_path: str = None):
    res = executar_demo(retornar_html=True, ativo=ativo, csv_path=csv_path, show_plot=False)
    return {"html_path": res["html"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
