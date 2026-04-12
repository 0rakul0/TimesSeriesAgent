# Dashboard Streamlit

O app em `dashbord/` organiza a visualizacao operacional do TimesSeriesAgent em duas colunas:

- coluna esquerda com serie intradiaria, mascara de 50 pregoes, cards de minima/maxima/projecao e chat com a base de conhecimento;
- coluna direita com resumo do evento, cluster associado, noticias semelhantes e comportamento medio do cluster.

## Como rodar

Na raiz do projeto:

```bash
streamlit run dashbord/app.py
```

Se quiser instalar dependencias antes:

```bash
uv sync
```

ou

```bash
pip install -e .
```
