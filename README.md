# TimesSeriesAgent

## Visão Geral

Este repositório implementa um **pipeline híbrido para previsão de séries temporais financeiras**, integrando modelos de *deep learning* (LSTM, Autoencoder LSTM e Transformer) com **análise semântica de notícias** e **modelagem explícita da propagação temporal de eventos informacionais**.

O projeto é a base computacional do artigo:

**"Da Notícia ao Preço: Mapeamento da Propagação de Eventos e Ajuste Residual em Modelos de Séries Temporais"**

O objetivo central é demonstrar que variações de mercado induzidas por notícias **não são ruído**, mas seguem **padrões recorrentes de propagação**, que podem ser aprendidos, armazenados e reutilizados como **ajustes residuais causais** em modelos preditivos.

---

## Arquitetura Geral

O pipeline completo é composto pelas seguintes etapas:

1. **Coleta e sincronização de dados financeiros** (ativos + Brent)
2. **Análise exploratória** (correlação móvel, motifs e discords)
3. **Detecção automática de eventos** por limiar de retorno
4. **Agente inteligente de notícias** (GPT + Tavily)
5. **Extração semântica de motivos**
6. **Geração de embeddings e clusterização semântica**
7. **Construção de sequências reais de impacto (D0 → D4/D5)**
8. **Treinamento de modelos base** (LSTM, AE, Transformer)
9. **Aplicação do modelo híbrido** com:

   * ajuste por sequência histórica de impacto
   * correção residual *walk-forward* (causal)
10. **Avaliação quantitativa e visualização interpretável**

---

## Estrutura de Pastas

```
TimesSeriesAgent/
│
├── data/                      # Bases processadas e artefatos finais
│   ├── dados_*_brent.csv      # Séries sincronizadas (ativo + Brent)
│   ├── cluster_motivos.csv    # Clusters semânticos globais
│   ├── cluster_<ativo>.csv    # Clusters específicos por ativo
│   ├── embeddings_frases.npy # Embeddings persistidos
│   ├── embeddings_frases_meta.csv
│   └── resultado_comparacao_modelos.csv
│
├── modelos/                   # Definições e pesos dos modelos
│   ├── model_baseline_lstm.py
│   ├── model_lstm_autoencoder.py
│   ├── model_transformer_price.py
│   ├── lstm_<ativo>.pt
│   ├── lstm_ae_<ativo>.pt
│   └── transformer_<ativo>.pt
│
├── train/                     # Scripts de treinamento
│   ├── treinar_ativo.py
│   ├── treinar_ae_ativo.py
│   └── treinar_transformer_ativo.py
│
├── src/                       # Pipeline de dados e notícias
│   ├── serietemporal.py       # Download e decomposição STL
│   ├── correlacao_ativos.py   # Merge, correlação, motifs, discords
│   ├── variacia_ativos.py     # Comparações exploratórias
│   ├── agent_noticia.py       # Agente GPT + Tavily
│   ├── seq_eventos.py         # Extração D0→D5 real
│   └── frases_clusters.py     # Clusterização semântica
│
├── utils/
│   └── embedding_manager.py   # Gerenciamento inteligente de embeddings
│
├── eval/
│   ├── modelo_hibrido_offline.py  # Avaliação final híbrida
│   └── plotter_refactor.py        # Visualizações interpretáveis
│
├── output_noticias/           # Eventos JSON estruturados
│   └── evento_<ativo>_<data>.json
│
├── img/                       # Gráficos PNG / HTML
│
├── artigo/
│   └── artigo_latex.tex       # Artigo científico final
│
└── README.md
```

---

## Descrição dos Principais Componentes

### 1. Coleta e Pré-processamento

**Arquivo:** `src/serietemporal.py`

* Download incremental via *Yahoo Finance*
* Sincronização ativo × Brent
* Decomposição STL (tendência e sazonalidade)

---

### 2. Análise Exploratória

**Arquivo:** `src/correlacao_ativos.py`

* Correlação móvel (rolling correlation)
* Identificação de *motifs* e *discords* (STUMPY)
* Visualizações estáticas e interativas

---

### 3. Detecção de Eventos e Agente de Notícias

**Arquivo:** `src/agent_noticia.py`

* Gatilho por retorno absoluto ≥ 2%
* Consulta híbrida:

  * GPT puro
  * Fallback Tavily + GPT
* Saída estruturada em JSON com:

  * motivos
  * sentimento
  * fontes

---

### 4. Sequência Real de Impacto

**Arquivo:** `src/seq_eventos.py`

* Extração automática de retornos:

```
[D0, D1, D2, D3, D4, D5]
```

* Interrupção causal se novo evento ocorrer

---

### 5. Embeddings e Clusterização Semântica

**Arquivo:** `src/frases_clusters.py`

* Embeddings OpenAI (`text-embedding-3-small`)
* Oversampling semântico
* Clusterização por ativo
* Geração de **frase canônica** por cluster

---

### 6. Modelos Base

Local: `modelos/`

* `model_baseline_lstm.py`
* `model_lstm_autoencoder.py`
* `model_transformer_price.py`

Treinamento:

* `train/treinar_ativo.py`
* `train/treinar_ae_ativo.py`
* `train/treinar_transformer_ativo.py`

---

### 7. Modelo Híbrido com Correção Residual

**Arquivo:** `eval/modelo_hibrido_offline.py`

Inclui:

* Aplicação da sequência média do cluster
* Escala por similaridade semântica
* Interrupção causal
* Correção residual *walk-forward* (Ridge)

---

### 8. Avaliação e Visualização

**Arquivo:** `eval/plotter_refactor.py`

* Gráficos comparativos
* Blocos de eventos
* Interpretação visual do impacto informacional

---

## Execução do Pipeline (Resumo)

```bash
# 1. Baixar e preparar dados
python src/serietemporal.py
python src/correlacao_ativos.py

# 2. Detectar eventos e notícias
python src/agent_noticia.py
python src/seq_eventos.py

# 3. Clusterizar motivos
python src/frases_clusters.py

# 4. Treinar modelos
python train/treinar_ativo.py
python train/treinar_ae_ativo.py
python train/treinar_transformer_ativo.py

# 5. Avaliação híbrida final
python eval/modelo_hibrido_offline.py
```

---

## Requisitos (requirements.txt)

```
numpy
pandas
scikit-learn
torch
yfinance
statsmodels
stumpy
matplotlib
plotly
tqdm
python-dotenv
openai
tavily-python
pydantic
```

---

## Contribuição Científica

Este projeto:

* Introduz **propagação temporal explícita de notícias**
* Reutiliza impactos históricos como **memória informacional**
* Integra NLP + Deep Learning de forma causal
* Aumenta acurácia **e interpretabilidade**

---

## Autores

* Jefferson Silva dos Anjos
* Luiz José Henrique Nogaroli Cavalcante
* Eduardo Soares Ogasawara

CEFET/RJ

---

## Licença

Uso acadêmico e de pesquisa.
