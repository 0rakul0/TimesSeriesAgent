from fastapi import FastAPI, Query
from fastapi.responses import FileResponse

from src.mvp import executar_demo
from src.pipeline_online import run_online
from utils.project_paths import DATA_DIR


app = FastAPI(
    title="TimesSeriesAgent Online API",
    description="API do sistema hibrido com previsao D+1..D+3",
    version="2.1",
)


@app.get("/")
async def root():
    return {"status": "online", "msg": "TimesSeriesAgent API ativa"}


@app.get("/run")
async def run_predict(
    ticker: str = Query("PETR4.SA"),
    csv_path: str = Query(str(DATA_DIR / "dados_petr4_brent.csv")),
):
    html_path = run_online(csv_path, ticker)
    return FileResponse(html_path, media_type="text/html")


@app.get("/run/petr4")
async def run_petr4():
    html_path = run_online(str(DATA_DIR / "dados_petr4_brent.csv"), "PETR4.SA")
    return FileResponse(html_path, media_type="text/html")


@app.get("/run/prio3")
async def run_prio3():
    html_path = run_online(str(DATA_DIR / "dados_prio3_brent.csv"), "PRIO3.SA")
    return FileResponse(html_path, media_type="text/html")


@app.get("/run/exxo34")
async def run_exxo34():
    html_path = run_online(str(DATA_DIR / "dados_exxo34_brent.csv"), "EXXO34.SA")
    return FileResponse(html_path, media_type="text/html")


@app.get("/predict_and_plot")
def predict_plot(ativo: str = "PETR4", csv_path: str | None = None):
    if csv_path is None:
        csv_path = str(DATA_DIR / f"dados_{ativo.lower()}_brent.csv")
    resultado = executar_demo(retornar_html=True, ativo=ativo, csv_path=csv_path, show_plot=False)
    return {"html_path": resultado["html"]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
