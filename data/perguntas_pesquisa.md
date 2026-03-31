# Perguntas de pesquisa do TimesSeriesAgent

Este arquivo organiza as perguntas de pesquisa que o projeto ja consegue responder ou que devem orientar a escrita do artigo.

## RQ1

**Pergunta.** Modelos hibridos com recuperacao semantica de eventos reduzem o erro de previsao em relacao aos modelos puramente quantitativos?

**Por que importa.** Essa e a pergunta principal do projeto, porque testa se a camada informacional agrega valor preditivo real.

**Artefatos que ajudam a responder.**
- `resultado_comparacao_modelos.csv`
- `analise_metricas_complementares.csv`

## RQ2

**Pergunta.** O ganho do hibrido se concentra em dias com evento ou tambem aparece em dias sem noticia relevante?

**Por que importa.** Ajuda a separar utilidade informacional de um simples efeito medio de suavizacao do erro.

**Artefatos que ajudam a responder.**
- `analise_eventos_vs_nao_eventos.csv`

## RQ3

**Pergunta.** Eventos semanticamente semelhantes apresentam padroes recorrentes de propagacao temporal apos o choque inicial?

**Por que importa.** Valida a hipotese de que o componente de noticias pode ser reutilizado como biblioteca historica de eventos comparaveis, e nao apenas como contextualizacao textual.

**Artefatos que ajudam a responder.**
- `cluster_motivos.csv`
- `cluster_*.csv`
- `output_noticias/evento_*.json`

## RQ4

**Pergunta.** O desempenho do metodo e robusto entre ativos, arquiteturas base e anos distintos?

**Por que importa.** Evita concluir a partir de um unico ativo, modelo ou recorte temporal favoravel.

**Artefatos que ajudam a responder.**
- `resultado_comparacao_modelos.csv`
- `analise_robustez_anual.csv`

## RQ5

**Pergunta.** Em quais situacoes o ajuste semantico piora a previsao e quais falhas operacionais explicam esses casos?

**Por que importa.** Essa pergunta transforma erros do hibrido em aprendizado metodologico para o artigo.

**Artefatos que ajudam a responder.**
- `analise_piora_hibrido_resumo.csv`
- `analise_piora_hibrido_detalhes.csv`
- `data/experiment_logs/prediction_details_*.csv`

## RQ6

**Pergunta.** Quao sensivel e a deteccao de eventos ao limiar adotado para variacao diaria?

**Por que importa.** Garante que a construcao da base semantica nao dependa de uma heuristica arbitraria nao auditada.

**Artefatos que ajudam a responder.**
- `analise_limiares_evento.csv`
