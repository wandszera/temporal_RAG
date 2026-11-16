# Aprendizados e limitações

## Este é um projeto de aprendizado

O objetivo foi **aprender** a construir e, principalmente, a **avaliar** um sistema de recuperação com consciência temporal, seguindo um plano de estudo de 6 semanas (ver seção 2). Isso significa:

- Os dados são **sintéticos**, gerados por mim, com seed fixa. Nenhum dado real de cliente foi usado.
- O sistema **não usa LLM**: as respostas são montadas por modelos de texto a partir dos eventos recuperados.
- Os resultados são de um benchmark **criado por mim**, com as mesmas premissas dos dados. Não são comparáveis a benchmarks públicos.
- Não há implantação, usuários, monitoramento real nem revisão externa.
- Parte do plano de estudo **não foi feita** (seção 2).

Quem avalia este repositório deve ler o código e os documentos como o registro de um exercício de aprendizado, e não como evidência de que a técnica funciona em produção.

## 1. O que o projeto me ensinou

1. **Medir antes de comemorar.** A primeira versão reportava 100% porque o avaliador dava ao agente a categoria e a data do gabarito e a checagem era permissiva. Depois de corrigir (checagem estrita, modo cego), o resultado inicial caiu para 75,6%.
2. **Um benchmark circular engana.** Mesmo com checagem estrita, 100% nas perguntas do gerador não significava generalização: em redações novas das mesmas perguntas, o agente começou em 41,7%. Separar redações de ajuste e de verificação foi o que expôs isso.
3. **Cuidado com o conjunto de verificação.** Ao olhar os erros de um conjunto, ele deixa de ser independente. Por isso criei novas redações (v4–v5) antes de corrigir mais regras, e documentei o que ainda pode estar contaminado.
4. **Filtrar por tempo importa mais que o algoritmo de ranking.** Na ablação, o filtro temporal subiu o recall@5 de 0,50–0,60 para 0,81–0,88, mais que a escolha entre TF-IDF, BM25, LSA e híbrido. O híbrido melhorou MRR e nDCG, mas não o recall@5.
5. **Modelo bitemporal evita vazamento do futuro.** Separar "quando foi verdade" de "quando o sistema soube" permite perguntar o que o sistema sabia numa data.
6. **Ambiguidade no gabarito é um bug.** Comparar "depois de 08/12" por instante e por dia dava respostas diferentes para o mesmo evento; definir a regra por dia resolveu.
7. **Abstenção e bloqueio precisam de teste próprio.** Perguntas sem resposta e pedidos adversariais são categorias do benchmark, com métrica de acerto própria.
8. **Regras funcionam até a primeira redação inesperada.** Palavras-chave e expressões regulares cobrem os modelos conhecidos e falham em linguagem livre. Isso motivou o extrator de intenção plugável.
9. **Testar a camada de LLM sem chamar um LLM.** A validação estrita da saída, o fallback e o prompt foram testados com um dublê de respostas fixas; isso valida o encanamento, mas **não** mede o comportamento de nenhum modelo.

## 2. Plano de estudo (6 semanas) versus o que foi feito

Legenda: feito, parcial, não feito.

| Semana | Item do plano | Situação |
| :--- | :--- | :--- |
| 1 | Esquema bitemporal | Feito (SQLite; tabelas e índices) |
| 1 | Gerador sintético com seed fixa | Feito (planos, tickets, faturas, NPS, atualização cadastral, e-mails adversariais); sem ligações |
| 1 | Docker Compose com Postgres + pgvector | Parcial: a imagem sobe, mas o código não usa a extensão nem foi testado com Postgres |
| 1 | CI básico | Não feito |
| 2 | Perguntas geradas por programa, com IDs das fontes | Feito (320) |
| 2 | Categorias: lookup, ordenação, as-of, mudança, agregação, multi-hop, sem resposta, adversariais | Feito; injeção dentro de um e-mail do corpus não é avaliada |
| 2 | Split de desenvolvimento e de teste | Feito (por cliente); teste nunca avaliado |
| 2 | Anotação manual de 30 a 50 perguntas | Não feito |
| 2 | Harness de avaliação em um comando | Feito |
| 3 | Chunking por evento | Feito (um evento por documento) |
| 3 | Resumos hierárquicos, janelas de sessão | Não feito |
| 3 | Embeddings no pgvector, `tsvector` | Não feito |
| 3 | Baseline vetorial puro e busca híbrida com RRF | Feito com LSA; embeddings neurais medidos à parte na ablação (não superaram os lexicais) |
| 3 | Filtro de janela temporal, decaimento por recência | Feito (decaimento sem efeito nos dados) |
| 3 | Reranker | Não feito |
| 3 | recall@k, MRR, nDCG e tabela de ablação | Feito |
| 4 | Nós do LangGraph: classificar, extrair janelas, recuperar, suficiência com repetição, resposta com citações | Feito |
| 4 | Verificar fundamentação da resposta | Não feito |
| 4 | Tool SQL para agregações | Parcial: agregação em Python, não em SQL |
| 4 | Abstenção explícita | Feito |
| 4 | Checkpointing e aprovação humana | Não feito |
| 5 | Mascaramento de PII | Feito com regras (e-mail, CPF, telefone) |
| 5 | Row-level security no Postgres | Não feito |
| 5 | Defesas contra injeção e testes adversariais | Parcial: guarda por padrões e testes |
| 5 | Traces (Langfuse, LangSmith, OpenTelemetry) e latência por nó | Não feito; só latência p50/p95 por rota |
| 5 | Testes de vazamento e isolamento no CI | Parcial: testes locais, sem CI |
| 6 | Ablações finais no conjunto de teste | Não feito |
| 6 | LLM-as-judge calibrado | Não feito |
| 6 | Análise de erros | Parcial: por categoria, em `AVALIACAO.md` |
| 6 | README com diagrama, resultados e limitações | Feito |
| 6 | GIF ou vídeo de demonstração, post técnico | Não feito |

## 3. Limitações

**Dados e avaliação**
- Dados sintéticos de um único gerador, 250 clientes, com ruído muito menor que o de um CRM real.
- Benchmark e regras do agente têm o mesmo autor.
- Poucas perguntas por categoria; sem intervalos de confiança.
- O conjunto de teste nunca foi usado; os 75% em redações novas ainda podem estar levemente otimistas.

**Sistema**
- Sem LLM e sem reranker; embeddings neurais só na ablação, fora do agente.
- Extrator de intenção por regras é frágil a redações novas; categorias `aggregation`, `change_detection` e `contradiction` ficaram em 50% na verificação.
- Agregações são filtros em Python, não SQL.
- Caminho PostgreSQL e uso do pgvector não testados.
- A API e a interface web não foram executadas na última rodada de documentação.

**Segurança**
- Chave de API global, sem autorização por cliente nem row-level security; CORS aberto na demonstração. Ver [`SEGURANCA.md`](SEGURANCA.md).

## 4. Próximos passos naturais

1. Escrever à mão (ou com revisão humana) um conjunto de perguntas de linguagem livre e avaliá-lo **uma vez** no split `test`.
2. Rodar o extrator de LLM (`--extractor llm`) com uma função de completar texto, medir custo e latência e comparar com as regras na verificação.
3. Testar um reranker e outros modelos de embedding na ablação (o MiniLM não superou os lexicais).
4. Usar PostgreSQL com pgvector e `tsvector` de verdade e comparar com a versão em memória.
5. Medir injeção dentro dos dados (e-mails adversariais) quando houver um LLM gerando a resposta.
6. Adicionar CI, intervalos de confiança e uma análise de erros mais detalhada por categoria.
