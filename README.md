# Temporal RAG para CRM

> **Projeto de aprendizado.** Foi construído para estudar como sistemas de recuperação e geração (RAG) lidam com perguntas que dependem do tempo ("qual era o plano deste cliente em março de 2023?"). Não é um produto, não foi validado em produção, **usa apenas dados sintéticos** e **não chama nenhum LLM**. Os números abaixo vêm de um benchmark criado por mim mesmo e devem ser lidos com as ressalvas da seção "Como ler os resultados".

## O problema

Em um CRM, os fatos mudam: o cliente troca de plano, de endereço, de contato. Um RAG comum busca por similaridade de texto e tende a devolver o fato mais parecido ou mais recente, não o fato que era verdadeiro na data perguntada. Isso gera três erros típicos:

1. **Inversão temporal:** responde com o plano atual quando a pergunta era sobre o passado.
2. **Vazamento do futuro:** usa informação que o sistema só registrou depois da data perguntada.
3. **Falta de abstenção:** inventa uma resposta para uma data em que o cliente nem existia.

## O que o projeto explora

- **Modelo bitemporal:** cada evento tem o tempo em que o fato foi verdadeiro no mundo (`valid_from`/`valid_to`) e o tempo em que o CRM o registrou (`recorded_at`).
- **Gerador sintético** reprodutível de clientes e linhas do tempo (seed fixa).
- **Agente em LangGraph** com nós explícitos: classificar a pergunta, checar limites temporais e segurança, rotear, recuperar, refletir sobre a suficiência da evidência e responder citando eventos.
- **Avaliação cuidadosa:** checagem estrita de respostas, modo "cego", divisão dev/teste por cliente, perguntas parafraseadas e ablação de recuperação (recall@k, MRR, nDCG).
- **Extrator de intenção plugável** (regras ou LLM), com validação da saída e fallback. Nenhum LLM foi executado neste projeto.

## Resultados em resumo

Tudo abaixo vem de `seed 42`, 250 clientes, 6.428 eventos, split `dev`. O split `test` **não foi usado**.

| O que foi medido | Naive RAG | Temporal RAG |
| :--- | :---: | :---: |
| Perguntas do benchmark (254), acurácia estrita | 3,1% | 100% |
| Perguntas **parafraseadas usadas no ajuste** (1.016) | 5,3% | 100% |
| Perguntas **parafraseadas de verificação** (508) | 11,0% | **75,0%** |

**O número que importa é o 75%:** o 100% mede o quanto o agente cobre os modelos de pergunta que o próprio gerador cria, e as regras do agente foram escritas olhando esses modelos. Detalhes, ablação de recuperação e limitações estão em [`docs/AVALIACAO.md`](docs/AVALIACAO.md).

## Como ler os resultados

- Dados sintéticos e benchmark vêm da mesma pessoa e do mesmo gerador: o sistema nunca enfrentou dados reais nem perguntas escritas por terceiros.
- Há 31 a 32 perguntas por categoria, então um erro vale cerca de 3 pontos percentuais.
- A versão inicial deste README reportava 100% com uma checagem frouxa e dando ao agente a resposta da categoria; isso foi corrigido (ver [`docs/AVALIACAO.md`](docs/AVALIACAO.md), seção "Histórico de correções").
- Latências são de uma máquina de desenvolvimento e servem só para comparação entre variantes.

## Início rápido

Testado com Python 3.10 em Linux (scikit-learn 1.7, numpy 2.2, pydantic 2.14, SQLAlchemy 2.0, FastAPI 0.162, LangGraph 1.2.22). O banco padrão é SQLite, sem Docker.

```bash
python -m venv .venv
source .venv/bin/activate            # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt pytest httpx
export PYTHONPATH=src                # PowerShell: $env:PYTHONPATH = "src"

python -m temporal_rag.cli seed      # gera dados, benchmark e paráfrases em data/
python -m temporal_rag.cli eval      # agentes no benchmark (split dev)
python -m temporal_rag.cli eval --blind
python -m temporal_rag.cli eval-paraphrase
python -m temporal_rag.cli ablate    # recuperação: recall@k, MRR, nDCG
python -m temporal_rag.cli compare   # exemplo lado a lado no terminal
python -m temporal_rag.cli serve     # API + interface em http://127.0.0.1:8000
python -m pytest -q tests/           # 40 testes
```

Atalhos em `Makefile` (`make seed`, `make eval`, `make eval-para`, `make ablate`, `make test`, `make run`) para quem tem `make`. `make clean` apaga `data/`. Para gerar os dados em outra pasta, defina `TEMPORAL_RAG_DATA_DIR`.

## Documentação

| Documento | Conteúdo |
| :--- | :--- |
| [`docs/ARQUITETURA.md`](docs/ARQUITETURA.md) | Modelo bitemporal, fluxo de dados, grafo do LangGraph, módulos e decisões de design |
| [`docs/AVALIACAO.md`](docs/AVALIACAO.md) | Benchmark, métricas, resultados completos, ablação, paráfrases, histórico de correções e ameaças à validade |
| [`docs/SEGURANCA.md`](docs/SEGURANCA.md) | Guarda de entrada, mascaramento de PII, API key, métricas e limites |
| [`docs/APRENDIZADOS_E_LIMITACOES.md`](docs/APRENDIZADOS_E_LIMITACOES.md) | O que foi e o que não foi feito do plano de estudo, lições aprendidas e próximos passos |

## Estrutura do código

```
temporal_RAG/
├── src/temporal_rag/
│   ├── models/schema.py           # Modelos Pydantic (eventos bitemporais, perguntas, respostas)
│   ├── storage/db.py              # Persistência (SQLite por padrão; PostgreSQL via DATABASE_URL, não testado)
│   ├── generator/synthetic.py     # Gerador sintético determinístico
│   ├── benchmark/
│   │   ├── dataset_generator.py   # Perguntas com gabarito, multi-hop, adversariais, split dev/teste
│   │   └── paraphrases.py         # 6 redações livres por pergunta
│   ├── retrieval/
│   │   ├── naive_retriever.py     # Baseline por similaridade
│   │   ├── temporal_retriever.py  # Filtro bitemporal e decaimento
│   │   └── strategies.py          # TF-IDF, BM25, LSA, híbrido RRF (usado na ablação)
│   ├── agent/
│   │   ├── langgraph_agent.py     # Grafo do agente temporal
│   │   ├── intent.py              # Extrator de intenção (regras ou LLM plugável)
│   │   ├── temporal_parser.py     # Datas, tickets, classificação por regras
│   │   └── agent.py               # Baseline e alias do agente temporal
│   ├── eval/                      # evaluator.py, metrics.py, ablation.py
│   ├── security.py                # Guarda de entrada e mascaramento de PII
│   ├── server/app.py              # API FastAPI e interface web
│   └── cli.py
├── tests/                         # 40 testes
├── docs/                          # Documentação
├── docker-compose.yml             # Postgres 16 + pgvector (imagem; a extensão não é usada pelo código)
└── Makefile
```

## Limitações principais

- Sem LLM: as respostas são montadas por modelos de texto a partir dos eventos recuperados.
- Sem embeddings neurais nem reranker: o "denso" da ablação é LSA (TF-IDF + SVD).
- O pgvector está disponível na imagem do Docker, mas **não é usado** pelo código.
- Agregações são filtros em Python sobre os eventos, não consultas SQL.
- Sem autorização por cliente nem row-level security.

A lista completa e o que fica para depois estão em [`docs/APRENDIZADOS_E_LIMITACOES.md`](docs/APRENDIZADOS_E_LIMITACOES.md).
