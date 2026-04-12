# TimesSeriesAgent

TimesSeriesAgent e um pipeline de pesquisa para previsao de series temporais financeiras com enriquecimento por noticias. O projeto combina:

- modelos base de previsao de preco (`LSTM`, `LSTM Autoencoder` e `Transformer`)
- deteccao automatica de eventos relevantes no preco
- busca e estruturacao de noticias relacionadas
- clusterizacao semantica dos motivos identificados
- ajuste hibrido da previsao com sequencias historicas observadas de impacto

O repositorio foi organizado para servir como base experimental de artigo. O foco aqui e reprodutibilidade, rastreabilidade metodologica e facilidade de reexecucao dos experimentos.

## Objetivo cientifico

A hipotese central do projeto e que parte dos movimentos abruptos de preco nao deve ser tratada apenas como ruido. Em certos casos, esses movimentos podem estar associados a eventos informacionais recorrentes, cuja propagacao temporal pode ser observada, agrupada e reutilizada como ajuste residual em modelos de previsao.

Em termos práticos, o pipeline tenta responder:

1. quando um evento relevante acontece em um ativo ou no Brent;
2. quais noticias ajudam a contextualizar esse evento;
3. como eventos semanticamente semelhantes se propagaram nos dias seguintes;
4. se usar esse historico melhora a previsao frente ao modelo puro.

## Perguntas de pesquisa

Se voce quiser resumir o projeto em uma unica pergunta de pesquisa, a melhor formulacao hoje e:

**"A recuperacao de eventos financeiros semanticamente semelhantes, com suas sequencias historicas de impacto, melhora a previsao de curto prazo de ativos de oleo e gas em relacao a modelos puramente quantitativos?"**

Para um artigo, vale separar isso em perguntas menores e testaveis:

1. **RQ1.** O modelo hibrido reduz RMSE e MAE em relacao aos baselines `LSTM`, `Autoencoder` e `Transformer`?
2. **RQ2.** O ganho do hibrido aparece principalmente em dias com evento relevante ou tambem em dias sem evento?
3. **RQ3.** Eventos semanticamente semelhantes apresentam padroes recorrentes de propagacao temporal `D0..D5`?
4. **RQ4.** O efeito e robusto entre ativos, arquiteturas e anos diferentes?
5. **RQ5.** Em quais situacoes o ajuste semantico piora a previsao?
6. **RQ6.** Quao sensivel e a base de eventos ao limiar usado na deteccao de movimentos relevantes?

Em termos de maturidade experimental, o repositorio ja responde diretamente `RQ1`, `RQ2`, `RQ4`, `RQ5` e `RQ6`. A `RQ3` tambem esta coberta pela infraestrutura atual, mas ainda exige interpretacao qualitativa cuidadosa dos clusters e dos JSONs de eventos.

## Escopo atual

Ativos cobertos no experimento:

- `PETR4.SA`
- `PRIO3.SA`
- `EXXO34.SA`
- `BZ=F` como proxy de Brent

Principais saidas geradas:

- series sincronizadas ativo x Brent em `data/`
- eventos estruturados em `output_noticias/`
- clusters semanticos em `data/cluster_*.csv`
- pesos de modelos em `modelos/*.pt`
- graficos de avaliacao em `img/`
- tabelas consolidadas em `data/`

## Estrutura do repositorio

```text
TimesSeriesAgent/
|-- data/                  # bases de entrada e artefatos tabulares
|-- eval/                  # avaliacao offline e scripts auxiliares para o artigo
|-- img/                   # visualizacoes HTML/PNG
|-- modelos/               # definicoes de modelos e checkpoints
|-- output_noticias/       # eventos JSON gerados pelo pipeline de noticias
|-- src/                   # pipeline principal, API e MVP
|-- tests/                 # smoke tests de reproducao/import
|-- train/                 # scripts de treino
|-- utils/                 # paths compartilhados, embeddings e utilitarios
|-- .env_template          # modelo das chaves de ambiente
|-- pyproject.toml         # manifesto de dependencias
`-- README.md
```

## Arquitetura do pipeline

### 1. Coleta e preparacao de dados

Arquivo principal: [src/serietemporal.py](https://github.com/0rakul0/TimesSeriesAgent/blob/master/src/serietemporal.py)

Responsabilidades:

- baixar historico via Yahoo Finance
- sincronizar o ativo com o Brent
- realizar decomposicao sazonal
- salvar CSVs de trabalho para treino e avaliacao

### 2. Analise exploratoria

Arquivo principal: [src/correlacao_ativos.py](https://github.com/0rakul0/TimesSeriesAgent/blob/master/src/correlacao_ativos.py)

Responsabilidades:

- correlacao movel
- comparacao entre ativo e Brent
- identificacao de motifs e discords
- visualizacoes auxiliares

### 3. Deteccao de eventos e noticias

Arquivo principal: [src/agent_noticia.py](https://github.com/0rakul0/TimesSeriesAgent/blob/master/src/agent_noticia.py)

Responsabilidades:

- detectar dias com variacao absoluta acima do limiar
- consultar LLM para estruturar uma hipotese explicativa do evento
- usar Tavily como fallback quando o modelo nao encontra contexto suficiente
- salvar um JSON por evento

### 4. Sequencia real de impacto

Arquivo principal: [src/seq_eventos.py](https://github.com/0rakul0/TimesSeriesAgent/blob/master/src/seq_eventos.py)

Responsabilidades:

- calcular a sequencia real `D0..D5` apos um evento
- interromper a propagacao quando um novo evento aparece
- padronizar os JSONs de eventos

### 5. Embeddings e clusterizacao semantica

Arquivo principal: [src/frases_clusters.py](https://github.com/0rakul0/TimesSeriesAgent/blob/master/src/frases_clusters.py)

Responsabilidades:

- gerar embeddings para os motivos dos eventos
- ampliar semanticamente frases quando necessario
- agrupar motivos semelhantes
- calcular uma frase representativa e a sequencia media por cluster

### 6. Treino dos modelos base

Arquivos principais:

- [train/treinar_ativo.py](https://github.com/0rakul0/TimesSeriesAgent/blob/master/train/treinar_ativo.py)
- [train/trainar_ae_ativo.py](https://github.com/0rakul0/TimesSeriesAgent/blob/master/train/trainar_ae_ativo.py)
- [train/treinar_transformrs.py](https://github.com/0rakul0/TimesSeriesAgent/blob/master/train/treinar_transformrs.py)

Responsabilidades:

- treinar os modelos base por ativo
- serializar checkpoints com `state_dict`, `scaler`, colunas e `seq_len`

### 7. Avaliacao hibrida

Arquivo principal: [eval/modelo_hibrido_offline.py](https://github.com/0rakul0/TimesSeriesAgent/blob/master/eval/modelo_hibrido_offline.py)

Responsabilidades:

- carregar os modelos treinados
- recuperar impacto historico com corte temporal
- aplicar ajuste por sequencia semantica
- aplicar correcao residual walk-forward
- exportar comparacoes, ablacoes e logs

### 8. MVP e API

Arquivos principais:

- [src/mvp.py](https://github.com/0rakul0/TimesSeriesAgent/blob/master/src/mvp.py)
- [src/pipeline_online.py](https://github.com/0rakul0/TimesSeriesAgent/blob/master/src/pipeline_online.py)
- [src/main.py](https://github.com/0rakul0/TimesSeriesAgent/blob/master/src/main.py)

Responsabilidades:

- escolher o melhor modelo para um ativo
- produzir previsao recursiva `D+1..D+3`
- aplicar ajuste por impacto semantico
- servir o resultado por FastAPI

## Requisitos

### Python

- Python `3.12+`

### Dependencias

As dependencias estao declaradas em [pyproject.toml](https://github.com/0rakul0/TimesSeriesAgent/blob/master/pyproject.toml). O ambiente inclui, entre outras:

- `torch`
- `pandas`
- `numpy`
- `scikit-learn`
- `statsmodels`
- `stumpy`
- `plotly`
- `fastapi`
- `openai`
- `tavily-python`

## Configuracao do ambiente

### 1. Criar o arquivo `.env`

Use [`.env_template`](https://github.com/0rakul0/TimesSeriesAgent/blob/master/.env_template) como base:

```bash
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

### 2. Instalar dependencias

Com `uv`:

```bash
uv sync
```

Ou com `pip`:

```bash
pip install -e .
```

Para desenvolvimento:

```bash
pip install -e .[dev]
```

## Como reproduzir os experimentos

### Pipeline offline completo

```bash
python -m src.run_all
```

Esse comando executa:

1. download e preparacao dos dados
2. correlacao e analise exploratoria
3. deteccao de eventos
4. geracao das sequencias reais
5. clusterizacao semantica
6. treino dos modelos
7. avaliacao hibrida
8. geracao das tabelas auxiliares do artigo e do mapa de perguntas de pesquisa

### Etapas individuais

```bash
python src/serietemporal.py
python src/correlacao_ativos.py
python src/agent_noticia.py
python src/seq_eventos.py
python src/frases_clusters.py
python train/treinar_ativo.py
python train/trainar_ae_ativo.py
python train/treinar_transformrs.py
python eval/modelo_hibrido_offline.py
```

### Tabela de sensibilidade do limiar

Para isolar a etapa de deteccao de eventos e contar quantos eventos candidatos sao capturados sob diferentes limiares:

```bash
python src/agent_noticia.py --analisar-limiares
```

Saida gerada:

- [data/analise_limiares_evento.csv](https://github.com/0rakul0/TimesSeriesAgent/blob/master/data/analise_limiares_evento.csv)

### Tabelas auxiliares para o artigo

Depois de rodar a avaliacao offline, o projeto pode gerar tabelas auxiliares para responder perguntas metodologicas:

```bash
python eval/article_answers.py
```

Arquivos previstos:

- `data/resultado_ablation_modelos.csv`
- `data/analise_metricas_complementares.csv`
- `data/analise_eventos_vs_nao_eventos.csv`
- `data/analise_robustez_anual.csv`
- `data/analise_piora_hibrido_resumo.csv`
- `data/analise_piora_hibrido_detalhes.csv`
- `data/perguntas_pesquisa.md`

Observacao importante:

No estado atual do repositório, a tabela de limiares e a tabela consolidada de comparacao de modelos ja estao preenchidas com dados reais do projeto. Algumas tabelas auxiliares adicionais precisam ser regeneradas a partir de uma execucao completa da avaliacao offline antes de serem usadas como evidencia do artigo.

### MVP local

```bash
python src/mvp.py
```

### API

```bash
uvicorn src.main:app --reload
```

Rotas uteis:

- `GET /`
- `GET /run`
- `GET /run/petr4`
- `GET /run/prio3`
- `GET /run/exxo34`
- `GET /predict_and_plot?ativo=PETR4`

## Reprodutibilidade

Mudancas recentes feitas para reduzir fragilidade experimental:

- paths centrais agora partem da raiz do projeto via [utils/project_paths.py](https://github.com/0rakul0/TimesSeriesAgent/blob/master/utils/project_paths.py)
- o manifesto de dependencias foi alinhado com os imports reais
- o MVP passou a gerar previsao recursiva real para `D+1..D+3`
- o pipeline online agora respeita o ativo solicitado
- inicializacoes de cliente OpenAI/Tavily foram deixadas lazy
- smoke tests foram adicionados em [tests/test_reproducibility_smoke.py](https://github.com/0rakul0/TimesSeriesAgent/blob/master/tests/test_reproducibility_smoke.py)
- a avaliacao hibrida passou a usar recuperacao historica com corte temporal estrito
- a avaliacao exporta logs de decisao em `data/experiment_logs/`

Para reproduzir o artigo com rigor, registre:

- hash do commit
- versao do Python
- versao das dependencias
- data da coleta via Yahoo Finance
- modelo OpenAI usado para embeddings e estruturacao textual de eventos
- datas de execucao do pipeline

## Estado atual do projeto

Hoje o projeto esta em um estado bem melhor para pesquisa reproduzivel do que a versao inicial, mas ainda deve ser entendido como um pipeline experimental de artigo, nao como sistema de producao.

Pontos fortes:

- setup e imports estao consistentes
- a API e o MVP respeitam o ativo solicitado
- a previsao curta `D+1..D+3` do MVP e recursiva de verdade
- a avaliacao offline registra ablacoes e logs auditaveis
- a recuperacao de impacto usada na avaliacao hibrida respeita corte temporal
- os clusters semanticos agora carregam metadados uteis para auditoria

Pontos que ainda devem ser tratados como limitacao experimental:

- a clusterizacao semantica continua sendo gerada offline sobre a base de eventos disponivel
- o pipeline depende de servicos externos, entao parte do processo nao e completamente deterministica
- ainda nao ha uma bateria extensa de testes quantitativos ou estatisticos no repositorio
- o uso do sistema continua mais adequado a analise academica do que a tomada de decisao operacional

Arquivos centrais para entender a versao atual:

- [utils/project_paths.py](https://github.com/0rakul0/TimesSeriesAgent/blob/master/utils/project_paths.py)
- [src/frases_clusters.py](https://github.com/0rakul0/TimesSeriesAgent/blob/master/src/frases_clusters.py)
- [eval/modelo_hibrido_offline.py](https://github.com/0rakul0/TimesSeriesAgent/blob/master/eval/modelo_hibrido_offline.py)
- [eval/article_answers.py](https://github.com/0rakul0/TimesSeriesAgent/blob/master/eval/article_answers.py)
- [tests/test_reproducibility_smoke.py](https://github.com/0rakul0/TimesSeriesAgent/blob/master/tests/test_reproducibility_smoke.py)

## FAQ cientifico

### Qual e exatamente a principal contribuicao do artigo em relacao a trabalhos que ja combinam noticias e preco?

A contribuicao principal nao e apenas adicionar texto ao modelo, mas modelar eventos informacionais como padroes recorrentes de impacto ao longo do horizonte curto. O sistema recupera historicos semanticamente semelhantes e usa esse conhecimento como ajuste residual orientado por eventos comparaveis sobre modelos quantitativos base.

### Como este trabalho se posiciona em relacao ao repositorio `stockpredictionai` de Boris Banushev?

O repositorio [`stockpredictionai`](https://github.com/borisbanushev/stockpredictionai), apresentado por Boris Banushev em 2019, e uma referencia util de implementacao publica para previsao de acoes com arquiteturas profundas e combinacao de multiplas fontes de informacao. Ele ajuda a mostrar uma linha aplicada em que preco, indicadores, ativos correlatos e texto entram como sinais para o modelo preditivo.

O posicionamento deste projeto e diferente. Aqui o texto nao entra apenas como *feature* de sentimento ou contexto agregado: as noticias sao convertidas em eventos estruturados, depois em pares *motivo-sequencia*, e por fim reutilizadas via recuperacao semantica sob corte temporal estrito. Isso aproxima o trabalho de uma memoria explicita e auditavel de eventos, em vez de um modelo unico ponta a ponta com fusao direta de atributos.

Se voce quiser citar esse repositorio no artigo, uma forma segura e trata-lo como implementacao publica relacionada, nao como artigo cientifico revisado por pares. Uma referencia BibTeX adequada e:

```bibtex
@misc{banushev2019stockpredictionai,
  author       = {Boris Banushev},
  title        = {stockpredictionai: Using the latest advancements in AI to predict stock market movements},
  year         = {2019},
  howpublished = {\url{https://github.com/borisbanushev/stockpredictionai}},
  note         = {GitHub repository. Accessed: 2026-03-27}
}
```

### O que diferencia a abordagem de uma simples variavel de sentimento adicionada ao modelo?

Sentimento e tratado na literatura, em geral, como um sinal contemporaneo agregado. Aqui o foco esta em recuperar eventos semanticamente semelhantes e aplicar sua sequencia temporal media de impacto, preservando defasagem e horizonte do efeito.

### Como o metodo reduz vazamento de informacao futura no regime walk-forward e quais limites permanecem para interpretacao causal?

Na avaliacao offline atual, a recuperacao de impacto filtra apenas eventos historicos com data anterior ao ponto previsto. Alem disso, a correcao residual e treinada em regime walk-forward, usando apenas informacao observada ate o instante corrente. Isso ajuda a reduzir vazamento de informacao futura, mas nao constitui identificacao causal formal.

### Como foi definido o limiar de 2% para detectar eventos? Houve analise de sensibilidade?

O limiar de `2.0%` foi inicialmente adotado como heuristica para capturar movimentos diarios relevantes sem inflar demais o numero de eventos. O repositorio agora inclui uma rotina isolada para analise de sensibilidade em [src/agent_noticia.py](https://github.com/0rakul0/TimesSeriesAgent/blob/master/src/agent_noticia.py), permitindo comparar `1.5%`, `2.0%` e `2.5%` sem chamar APIs externas.

Comando:

```bash
python src/agent_noticia.py --analisar-limiares
```

Saida:

- [data/analise_limiares_evento.csv](https://github.com/0rakul0/TimesSeriesAgent/blob/master/data/analise_limiares_evento.csv)

Contagem atual de eventos por ativo:

| Ativo | 1.5% | 2.0% | 2.5% |
|---|---:|---:|---:|
| PETR4 | 453 | 320 | 212 |
| PRIO3 | 610 | 469 | 354 |
| EXXO34 | 458 | 301 | 184 |

Esses resultados mostram o trade-off esperado: limiares menores aumentam cobertura de eventos, enquanto limiares maiores priorizam movimentos mais extremos. Para o artigo, `2.0%` pode ser defendido como um ponto intermediario razoavel entre sensibilidade e seletividade.

### Como os clusters semanticos foram validados?

No estado atual, a validacao e principalmente qualitativa e baseada em coerencia semantica, tamanho minimo dos grupos e auditabilidade das frases representativas e datas cobertas. Isso deve ser apresentado como etapa exploratoria estruturada, nao como validacao semantica definitiva.

### Por que PETR4, PRIO3, EXXO34 e Brent?

Esses ativos foram escolhidos por estarem expostos ao setor de oleo e gas e a choques macro, geopoliticos e de commodity. Isso favorece observar a difusao de eventos informacionais com boa interpretacao economica.

### O metodo generaliza para outros setores?

Conceitualmente sim, mas isso ainda precisa ser demonstrado empiricamente. A generalizacao deve ser tratada como hipotese plausivel, nao como conclusao fechada.

### Como o impacto historico medio `D0-D4` foi calculado e por que esse horizonte foi escolhido?

O impacto medio e calculado a partir das sequencias reais observadas apos eventos historicos semanticamente semelhantes. O horizonte curto foi escolhido para captar a propagacao imediata do evento, reduzindo o risco de confundir esse efeito com dinamicas de prazo mais longo.

### O que acontece quando duas noticias importantes ocorrem em sequencia ou se sobrepoem?

O pipeline interrompe a propagacao do impacto quando um novo evento relevante surge dentro da janela. Isso evita acumular sequencias de forma ingenua quando ha sobreposicao informacional.

### Como foi reduzido o risco de ruido ou alucinacao do modelo de linguagem?

O pipeline usa estrutura JSON, prompts controlados e fallback de busca. Alem disso, a avaliacao atual depende de recuperacao historica auditavel e logs de decisao, o que ajuda a inspecionar casos em que a camada semantica pode ter sido inadequada.

### O ganho de RMSE vem mais da parte semantica, da correcao residual, ou da combinacao das duas? Ha ablacão?

O repositorio ja exporta a tabela de ablacao em [data/resultado_ablation_modelos.csv](https://github.com/0rakul0/TimesSeriesAgent/blob/master/data/resultado_ablation_modelos.csv), mas ela deve ser regenerada a partir de uma execucao completa da avaliacao offline antes de ser usada como evidencia final do artigo. A infraestrutura para responder essa pergunta pelo codigo ja existe.

### Como o metodo se comporta em periodos sem noticias relevantes?

O script [eval/article_answers.py](https://github.com/0rakul0/TimesSeriesAgent/blob/master/eval/article_answers.py) foi criado para responder isso pela comparacao entre dias com evento e dias sem evento. Antes de usar essa resposta no artigo, rode novamente a avaliacao offline completa e gere [data/analise_eventos_vs_nao_eventos.csv](https://github.com/0rakul0/TimesSeriesAgent/blob/master/data/analise_eventos_vs_nao_eventos.csv).

### Por que RMSE foi escolhida como metrica principal?

RMSE foi usada por penalizar erros maiores e ser comum em previsao de series temporais. Ainda assim, o projeto tambem suporta geracao de metricas complementares, como `MAE`, `directional accuracy` e percentual de dias em que o hibrido supera o modelo base, por meio de [eval/article_answers.py](https://github.com/0rakul0/TimesSeriesAgent/blob/master/eval/article_answers.py).

### Os resultados sao estatisticamente robustos ou podem depender muito do periodo 2020-2025?

Eles podem depender do recorte temporal. O projeto agora suporta uma quebra anual de desempenho via [eval/article_answers.py](https://github.com/0rakul0/TimesSeriesAgent/blob/master/eval/article_answers.py), mas essa tabela deve ser regenerada com uma execucao completa da avaliacao offline antes de ser usada no texto do artigo.

### Qual e o custo computacional e operacional do pipeline completo?

O custo vem de tres partes: coleta de dados, chamadas a LLM/embeddings e treino dos modelos. A etapa semantica e a mais sensivel operacionalmente, porque depende de APIs externas e de latencia e estabilidade dessas chamadas.

### Qual parte do sistema mais contribui para auditabilidade e interpretacao economica?

A principal fonte de auditabilidade e a recuperacao historica de eventos semanticamente semelhantes com suas sequencias medias de impacto. Os logs experimentais ajudam a mostrar quais referencias foram usadas em cada ajuste, o que tambem melhora a interpretacao economica dos resultados.

### O metodo continua funcionando se a busca de noticias falhar ou trouxer noticias irrelevantes?

O pipeline nao colapsa completamente, porque o modelo base continua existindo. O que se perde e qualidade no ajuste informacional. Isso deve ser tratado como limitacao importante.

### Como o projeto lida com eventos macro que afetam simultaneamente varios ativos?

O uso do Brent e a filtragem por ativo ajudam a capturar parte desse efeito compartilhado. Ainda assim, eventos macro multissetoriais continuam sendo um caso desafiador e merecem extensao especifica.

### Em que situacoes o modelo hibrido tende a piorar a previsao?

O projeto agora suporta a geracao de tabelas de piora do hibrido por meio de [eval/article_answers.py](https://github.com/0rakul0/TimesSeriesAgent/blob/master/eval/article_answers.py), o que permite localizar os casos em que o ajuste informacional atrapalha. Em geral, isso tende a ocorrer quando o evento corrente nao encontra um bom historico comparavel, quando a noticia foi mal interpretada semanticamente ou quando o regime de mercado mudou.

### O uso do termo "RAG" e conceitual ou arquitetural no sentido estrito?

No artigo, o uso e principalmente conceitual. A ideia central e separar recuperacao de conhecimento historico e inferencia preditiva. Nao se trata de um RAG textual classico no sentido estrito de LLM com contexto recuperado para geracao livre.

### Se o artigo fosse expandido, qual seria o proximo experimento indispensavel?

O experimento mais importante seria uma avaliacao temporal ainda mais rigorosa da base semantica, reconstruindo ou congelando os artefatos por janela de tempo, alem de incluir sensibilidade de hiperparametros e validacao em outros setores.

## Evidencia quantitativa ja disponivel

A tabela consolidada de comparacao dos modelos ja existe em [data/resultado_comparacao_modelos.csv](https://github.com/0rakul0/TimesSeriesAgent/blob/master/data/resultado_comparacao_modelos.csv).

Resumo atual:

- RMSE medio dos modelos base: `1.6248`
- RMSE medio dos modelos hibridos: `0.8537`
- ganho medio absoluto: `0.7711`

Esta versao do repositorio inclui uma variante conservadora de reranqueamento no nucleo do pipeline. Ela atua apenas como desempate dentro de um pool semantico curto e, na configuracao atual, preserva o ganho do hibrido em todos os nove cenarios avaliados.

Detalhe por ativo e arquitetura:

| Ativo | Modelo | RMSE Modelo | RMSE Hibrido | Ganho |
|---|---|---:|---:|---:|
| PETR4 | LSTM | 0.9632 | 0.4650 | 0.4982 |
| PRIO3 | LSTM | 1.5592 | 0.8530 | 0.7062 |
| EXXO34 | LSTM | 2.0610 | 1.1624 | 0.8985 |
| PETR4 | AE | 1.0216 | 0.4678 | 0.5539 |
| PRIO3 | AE | 1.6077 | 0.8578 | 0.7499 |
| EXXO34 | AE | 2.1269 | 1.1744 | 0.9525 |
| PETR4 | Transformer | 1.0858 | 0.4752 | 0.6106 |
| PRIO3 | Transformer | 1.3836 | 0.8394 | 0.5442 |
| EXXO34 | Transformer | 2.8142 | 1.3883 | 1.4259 |

Esses numeros sustentam a afirmacao central de que, no recorte atualmente avaliado, o componente hibrido melhora consistentemente o desempenho em todos os ativos e arquiteturas testados. Em dias marcados como evento, a reducao media de MAE permanece em `48.1%`; em dias sem evento, ela fica em `53.4%`.

## Testes

Rodar smoke tests:

```bash
pytest
```

Esses testes nao validam as conclusoes quantitativas do artigo. Eles validam o minimo necessario para:

- importar os modulos principais
- verificar consistencia de paths
- detectar regressao basica de setup

## Observacoes importantes

### Dependencias externas

Algumas etapas dependem de servicos externos e, portanto, nao sao 100% deterministicas:

- Yahoo Finance para dados de mercado
- OpenAI para estruturacao textual de eventos e embeddings
- Tavily para busca de noticias

### Artefatos de pesquisa

Por padrao, o repositorio agora ignora:

- checkpoints treinados (`modelos/*.pt`)
- HTMLs e imagens gerados em `img/`
- JSONs de eventos em `output_noticias/`
- caches locais e arquivos de ambiente

Se voce quiser congelar um experimento para submissao do artigo, o ideal e publicar os artefatos finais em release, pacote suplementar ou repositorio com DOI, em vez de manter tudo no branch principal de desenvolvimento.

### Limitacoes atuais

- a previsao online usa recursao simples para variaveis exogenas futuras nao observadas
- a camada semantica depende da qualidade das noticias encontradas e da consistencia da resposta da LLM
- o sistema foi pensado para pesquisa, auditabilidade e interpretacao economica, nao para trading em producao

## Autores

- Jefferson Silva dos Anjos
- Luiz Jose Henrique Nogaroli Cavalcante
- Eduardo Soares Ogasawara

CEFET/RJ

## Uso

Uso academico e de pesquisa.
