# Cronograma do Projeto 2: Temporal RAG para CRM (6 semanas)

A lógica é entregar um pipeline end-to-end simples cedo e sofisticar depois. O benchmark vem antes da otimização, porque sem ele não é possível provar que as melhorias funcionam.

## Semana 1: Fundações e dados sintéticos

**Objetivo:** ter uma timeline de clientes realista e reproduzível.

- Definir a pergunta central e o escopo (o que o sistema responde e o que não responde).
- Desenhar o esquema bitemporal (`valid_time` e `recorded_time`) no Postgres.
- Escrever o gerador sintético com seed fixa: tickets, e-mails, chamadas, faturas, mudanças de plano, NPS, notas de vendedor.
- Injetar fatos que se contradizem ao longo do tempo (endereço, plano, contato) e PII sintética (nomes, CPFs inválidos, telefones).
- Documentar as premissas do gerador.
- Subir Docker Compose com Postgres + pgvector e a estrutura do repositório (pytest, lint, CI básico).

**Entregável:** `make seed` gera de 200 a 500 clientes com timelines, e um README inicial com as premissas.

## Semana 2: Benchmark (o coração do projeto)

**Objetivo:** ter o gabarito antes de construir o sistema.

- Gerar de 150 a 300 perguntas programaticamente a partir da timeline, com resposta exata e IDs dos eventos-fonte.
- Cobrir as categorias: lookup, ordenação temporal, as-of, detecção de mudança, agregação, multi-hop, não respondíveis e adversariais (prompt injection em e-mail, pedido de dados de outro cliente).
- Separar um conjunto de desenvolvimento e um de teste final, que não é tocado até o fim.
- Anotar à mão uma amostra de 30 a 50 perguntas, para calibrar o LLM-as-judge depois.
- Implementar o harness de avaliação: um comando roda tudo e gera o relatório.

**Entregável:** `make eval` roda contra um sistema "dummy" e produz a tabela de métricas.

## Semana 3: Retrieval (baseline e depois híbrido)

**Objetivo:** recuperar a evidência certa na janela certa.

- Pipeline de ingestão: chunking por evento, por janela de sessão e resumos hierárquicos (diário/mensal).
- Gerar embeddings e indexar (pgvector, mais `tsvector` para o lado lexical).
- **Baseline:** busca vetorial pura com filtro por cliente.
- Evoluir para busca híbrida (BM25 + vetorial com RRF), filtro de janela temporal, decaimento por recência e reranker.
- Medir recall@k, MRR, nDCG e acurácia temporal em cada variante.
- Se sobrar tempo, comparar com Qdrant em uma ablação.

**Entregável:** tabela de ablação de retrieval (denso vs híbrido, com e sem filtro temporal, com e sem reranker).

## Semana 4: Agente em LangGraph

**Objetivo:** orquestrar o raciocínio temporal com controle e rastreabilidade.

- Implementar os nós: classificar intenção temporal, extrair janelas, recuperar, verificar suficiência (com loop de re-retrieval), gerar resposta com citações, verificar grounding.
- Criar a tool SQL para agregações ("quantos tickets em 90 dias?") e o roteamento entre RAG e SQL.
- Adicionar abstenção explícita quando a evidência é insuficiente.
- Configurar checkpointing e um ponto de aprovação humana para ações sensíveis.
- Primeira rodada completa do benchmark com o agente.

**Entregável:** o agente responde de ponta a ponta, com IDs de eventos-fonte em cada resposta, e há uma primeira tabela de resultados.

## Semana 5: Privacidade, segurança e observabilidade

**Objetivo:** tornar o sistema defensável.

- Mascaramento de PII antes de chamar o LLM (Presidio ou regras próprias).
- Row-level security no Postgres para limitar o que o agente enxerga.
- Defesas contra prompt injection e testes contra as perguntas adversariais do benchmark.
- Logs sem PII e trilha de auditoria.
- Traces com Langfuse, LangSmith ou OpenTelemetry, com latência p50/p95 por nó e custo por consulta.
- Testes automatizados de vazamento de PII e de isolamento entre clientes, rodando no CI.

**Entregável:** relatório de segurança com taxa de bloqueio dos ataques e painel de latência e custo.

## Semana 6: Ablações finais, análise de erros e publicação

**Objetivo:** transformar o trabalho em um portfólio que converte.

- Rodar as ablações finais no conjunto de teste: com e sem filtro temporal, denso vs híbrido, com e sem reranker, modelo pequeno vs grande.
- Calibrar o LLM-as-judge contra as anotações manuais e reportar a concordância.
- **Análise de erros:** categorizar onde o sistema falha e por quê (esta seção vale mais do que um número bonito).
- Escrever o README (problema, decisões de design, resultados, limitações, reprodução em um comando), com diagrama do grafo e um GIF ou vídeo de 2 minutos.
- Escrever o post técnico com uma lição só, por exemplo "por que RAG falha em perguntas temporais".

**Entregável:** repositório final polido, post publicado e demo gravada.

## Pontos de controle

| Fim da semana | Você deve ter |
| --- | --- |
| 1 | Dados sintéticos reproduzíveis |
| 2 | Benchmark e harness de avaliação rodando |
| 3 | Retrieval medido, com ablação |
| 4 | Agente completo com primeira avaliação |
| 5 | PII, segurança e observabilidade |
| 6 | Resultados finais, README e post |

## Se o tempo apertar

Corte primeiro a comparação com Qdrant, os resumos hierárquicos e o aprovador humano. Nunca corte o benchmark, a tabela de ablação nem a análise de erros, porque são eles que sustentam o projeto.
