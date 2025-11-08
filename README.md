# 🧠 Análise de Eventos PETR4 e Brent com IA

Este projeto automatiza a **detecção de eventos relevantes** nas ações da **Petrobras (PETR4)** e no **petróleo Brent**, analisando variações abruptas nos preços e buscando **notícias e contexto** sobre os acontecimentos com auxílio de **modelos de linguagem (LLMs)** e da **API Tavily** para busca de notícias.

---

## 🚀 Visão Geral do Fluxo

O pipeline realiza automaticamente as seguintes etapas:

1. **Detecta variações significativas** nos preços da PETR4 e do Brent.
2. **Coleta notícias** relacionadas ao evento via API Tavily.
3. **Resume as notícias** usando uma LLM (Llama 3.2 via Ollama).
4. **Analisa o sentimento** e o possível **impacto de mercado**.
5. **Produz um resumo estruturado** em formato JSON.
6. **Salva todos os resultados** em arquivos JSON individuais e um CSV consolidado.

---

## 📊 Estrutura do Projeto

```
.
├── data/
│   └── dados_combinados.csv      # Dados históricos de preços PETR4 e Brent
├── output_noticias/
│   ├── evento_2021-02-22_PETR4.json
│   ├── evento_2021-03-09_Ambos.json
│   └── eventos_consolidados.csv
├── main.py                        # Script principal (este)
└── README.md                      # Este arquivo
```

---

## 🧩 Componentes Principais

### 1️⃣ **Detecção de Eventos**

Função: `detectar_eventos()`

* Lê o arquivo `dados_combinados.csv`
* Calcula as variações diárias em porcentagem (`pct_change * 100`)
* Identifica dias com variação superior a `LIMIAR_VARIACAO` (default = 5%)
* Classifica a origem do evento:

  * `PETR4` → variação na ação da Petrobras
  * `Brent` → variação no preço do petróleo
  * `Ambos` → ambos variaram significativamente no mesmo dia

---

### 2️⃣ **Coleta de Notícias (Agente Coletor)**

Função: `coletar_noticias(ativo, data_evento)`

* Consulta a **API Tavily** com busca avançada de notícias.
* Retorna até 3 resultados relevantes para a data.
* Cada notícia é resumida via **LLM** (função `resumir_trecho`) para manter o conteúdo conciso.

Exemplo de retorno:

```text
Título: Petrobras troca de presidente
Link: https://g1.globo.com/economia/noticia/...
Resumo: O presidente da Petrobras foi substituído após pressões do governo devido à alta nos combustíveis.
```

---

### 3️⃣ **Análise de Sentimento (Agente Financeiro)**

Função: `analisar_sentimento(texto, preco_atual, data)`

* Utiliza uma LLM para:

  * Classificar o **sentimento** (positivo, negativo ou neutro);
  * Estimar o **impacto percentual** no preço;
  * Projetar um **preço futuro** hipotético;
  * Fornecer uma **justificativa curta** baseada nas notícias.

Exemplo de saída:

```
Sentimento: NEGATIVO  
Impacto estimado: -7%  
Preço projetado: R$25.10  
Justificativa: As notícias indicam intervenção política e substituição de diretoria, o que preocupa investidores.
```

---

### 4️⃣ **Resumo Estruturado (Agente Jornalista)**

Função: `resumir_noticias(texto, data)`

* A LLM gera um **resumo jornalístico estruturado** com os campos:

  * `data`
  * `contexto` (econômico e político)
  * `acontecimento` (o que ocorreu)
  * `impacto` (sobre empresa e mercado)
  * `fontes` (principais referências)

Exemplo de retorno:

```json
{
  "data": "22 de fevereiro de 2021",
  "contexto": "Tensão política devido à interferência do governo na Petrobras.",
  "acontecimento": "Troca do presidente da estatal após divergências sobre política de preços.",
  "impacto": "Queda acentuada das ações da empresa e desvalorização no mercado.",
  "fontes": "Reuters, G1, Valor Econômico"
}
```

---

### 5️⃣ **Geração dos Arquivos de Saída**

* Para cada evento detectado:

  * Cria um arquivo JSON individual em `output_noticias/`
  * Adiciona o evento a um **CSV consolidado** (`eventos_consolidados.csv`)

---

## 🧠 Diagrama de Fluxo de Dados (Grafo)

```mermaid
graph TD

A[📈 CSV de preços PETR4 e Brent] --> B[🧮 detectar_eventos()]
B -->|variação > 5%| C[📰 coletar_noticias()]
C --> D[✂️ resumir_trecho()]
D --> E[🤖 analisar_sentimento()]
E --> F[🗞️ resumir_noticias()]
F --> G[💾 salvar JSON individual]
G --> H[📊 consolidar CSV final]
```

---

## ⚙️ Tecnologias Utilizadas

| Componente                         | Descrição                                  |
| ---------------------------------- | ------------------------------------------ |
| **Python**                         | Linguagem principal do projeto             |
| **Pandas**                         | Manipulação e análise de dados             |
| **Babel**                          | Formatação de datas em português           |
| **Tavily API**                     | Busca automatizada de notícias             |
| **LangChain + Ollama (Llama 3.2)** | Modelos de linguagem para resumo e análise |
| **Pydantic**                       | Estruturação dos dados de saída em JSON    |

---

## 🧾 Execução

### 1️⃣ Configurar dependências:

```bash
pip install pandas babel tavily langchain_ollama langchain_core pydantic
```

### 2️⃣ Iniciar o servidor Ollama (caso local):

```bash
ollama serve
ollama pull llama3.2
```

### 3️⃣ Rodar o script principal:

```bash
python main.py
```

---