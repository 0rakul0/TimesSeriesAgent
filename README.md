# 📘 **TimesSeriesAgent — Documentação Oficial (README)**

*Sistema completo para previsão temporal com impacto de notícias reais*

---

# 📌 **Visão Geral do Projeto**

O **TimesSeriesAgent** é um pipeline completo de previsão temporal que combina:

* séries históricas (ativos e commodities)
* eventos de notícias reais
* agrupamento semântico por embeddings
* geração automática da sequência de impacto (D0→D5)
* modelos tradicionais (LSTM/GRU/MLP)
* modelo híbrido (previsão + impacto de evento)

O objetivo é gerar previsões temporalmente realistas incorporando o efeito de notícias relevantes no preço dos ativos.

---

# 🧱 **Arquitetura Geral do Sistema**

```
1 ─ Preparação das Séries
2 ─ Pipeline de Notícias e Eventos
3 ─ Modelos de Previsão (Simples e Híbrido)
```

---

# 1️⃣ **Preparação das Séries (FASE 1)**

Scripts responsáveis por preparar, combinar e analisar séries históricas.

### **1.1 serietemporal.py**

* Junta séries históricas:

  * PETR4 + BRENT
  * PRIO3 + BRENT
* Normalização
* Criação de CSVs combinados (`dados_*.csv`)

### **1.2 correlacao_ativos.py**

Utilitário para:

* analisar correlação entre ativos
* identificar dependências e sincronização

### **1.3 variancia_ativos.py**

Ferramentas auxiliares:

* cálculo de retornos
* desvio padrão
* volatilidade diária/móvel

> Fase 1 prepara toda a base numérica necessária para as etapas seguintes.

---

# 2️⃣ **Pipeline de Notícias e Eventos (FASE 2)**

Esta fase coleta, interpreta e transforma notícias em eventos estruturados, extraindo *impacto temporal real* dos ativos.

---

## **2.1 agente_noticia.py**

Extrai informação relevante de uma notícia:

* motivo principal
* sentimento do mercado
* quais ativos foram afetados
* o que houve
* fontes
* resumos

Resultado → arquivos:

```
output_noticias/evento_*.json
```

---

## **2.2 frases_cluster.py**

Agrupa motivos de eventos por similaridade semântica.

* usa embeddings
* detecta eventos “semelhantes”
* cria clusters temáticos
* permite calcular média de impacto por motivo

---

## **2.3 gerar_seq_eventos.py**

O coração do pipeline de eventos.

Para cada `evento_*.json`:

1. identifica o ativo (ou ativos, no caso de AMBOS)
2. lê automaticamente o CSV correspondente
3. calcula D0 → D5 reais:

   ```
   seq = [ret_D0, ret_D1, ..., ret_D5]
   ```
4. paralisa a sequência se outro evento acontecer antes
5. adiciona `seq` ao JSON
6. limpa o arquivo removendo campos redundantes
7. mantém somente:

   * data
   * ativo(s)
   * retorno_no_dia
   * motivos_identificados
   * sentimento_do_mercado
   * fontes
   * o_que_houve
   * seq

Exemplo final:

```json
{
  "data": "2021-11-30",
  "ativo": "AMBOS",
  "retorno_no_dia": {"PRIO3": -2.63, "BRENT": -3.90},
  "motivos_identificados": [...],
  "sentimento_do_mercado": "negativo",
  "seq": {
    "PRIO3": [-2.63, -1.12, 0.55],
    "BRENT": [-3.90, -2.20, -1.00, 0.85]
  }
}
```

> Fase 2 transforma eventos brutos em eventos com impacto temporal real.

---

# 3️⃣ **Modelos de Previsão (FASE 3)**

---

## **3.1 modelos tradicionais (LSTM / GRU / MLP)**

Estes scripts treinam modelos preditivos usando apenas:

* séries históricas
* janelas fixas
* normalização

Eles geram a **previsão base** usada no modelo híbrido.

---

## **3.2 modelo híbrido (modelo_hibrido_eval.py)**

Combina previsão + impacto de eventos:

```
previsao_final(t) = previsao_modelo(t) + impacto_evento(t)
```

O impacto pode vir de:

* seq real do evento
* média das seqs de eventos semelhantes
* clustering semântico via embeddings

O resultado é um modelo que:

* entende padrões históricos
* reage a notícias reais
* replica choques de mercado
* simula propagação temporal de impacto

---

# 🧩 **Estrutura dos Arquivos de Evento**

Cada evento final contém:

```json
{
  "data": "AAAA-MM-DD",
  "ativo": "PETR4" | "PRIO3" | "BRENT" | "AMBOS",
  "retorno_no_dia": 2.15,
  "motivos_identificados": [...],
  "sentimento_do_mercado": "positivo",
  "fontes": [...],
  "o_que_houve": "...",
  "seq": {
    "PETR4": [...],
    "BRENT": [...]
  }
}
```

---

# 🔥 **Fluxo Geral do Sistema**

```
[ séries históricas ]      → serietemporal
         ↓
[ CSVs combinados ]        → variancia + correlação
         ↓
[ notícias ]               → agente_noticia
         ↓
[ eventos brutos ]         → frases_cluster
         ↓
[ seq D0→D5 reais ]        → gerar_seq_eventos
         ↓
[ base de impacto ]
         ↓
[ previsão base ]          → modelos LSTM/GRU
         ↓
[ modelo híbrido ]         → previsão final ajustada por eventos
```

---

# 📎 **Estrutura do Repositório (sugestão)**

```
TimesSeriesAgent/
│
├── data/
│   ├── dados_petr4_brent.csv
│   ├── dados_prio3_brent.csv
│   └── ...
│
├── output_noticias/
│   ├── evento_2021-11-30_PRIO3.json
│   └── ...
│
├── src/
│   ├── serietemporal.py
│   ├── correlacao_ativos.py
│   ├── variancia_ativos.py
│   ├── agente_noticia.py
│   ├── frases_cluster.py
│   ├── gerar_seq_eventos.py
│   ├── modelo_baseline_lstm.py
│   └── modelo_hibrido_eval.py
│
├── README.md
└── requirements.txt
```