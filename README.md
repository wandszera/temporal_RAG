# Temporal RAG para CRM

Sistema de Recuperação e Geração Aumentada por Recuperação (**RAG**) com **Modelagem Bitemporal** e **Harness de Avaliação e Benchmark**.

---

## 📌 Por que o RAG Tradicional Falha em Consultas Temporais?

Em CRMs e sistemas empresariais, fatos mudam ao longo do tempo:
- Clientes mudam de plano (*Free* $\to$ *Starter* $\to$ *Pro* $\to$ *Downgrade*).
- Endereços e responsáveis corporativos são alterados.
- Chamados são abertos, resolvidos e reabertos.

Quando um usuário pergunta:
> *"Qual era o plano do cliente X em 15 de março de 2023?"*

O **Naive RAG** (busca vetorial tradicional por similaridade semântica) comete erros graves:
1. **Inversão Temporal / Recency Bias:** Recupera o plano mais recente (*Pro* ou *Enterprise* de 2024) ou o chunk com maior densidade da palavra "plano", gerando alucinações históricas.
2. **Data Leakage (Vazamento do Futuro):** Utiliza informações que o sistema só veio a conhecer meses depois.
3. **Impossibilidade de Abstenção:** Quando consultado sobre um período anterior à existência do cliente, inventa respostas em vez de declarar ausência de dados (*ABSTAIN*).

---

## 🏗️ Arquitetura Bitemporal

Cada evento na linha do tempo do cliente possui duas dimensões temporais fundamentais:

```
                  ┌─────────────────────────────────────────────────────────┐
                  │                 DIMENSÕES BITEMPORAIS                  │
                  └─────────────────────────────────────────────────────────┘

  [valid_from, valid_to) ────────► Tempo no Mundo Real (Quando o fato foi verdadeiro)
  [recorded_at]           ────────► Tempo no Sistema CRM (Quando o CRM registrou o fato)
```

1. **`valid_time` (`valid_from` e `valid_to`):**
   - Intervalo em que o fato estava de fato ativo na realidade do cliente.
2. **`recorded_at`:**
   - Timestamp no qual o sistema registrou o evento (garante que não haja vazamento de dados do futuro).

---

## 📊 Resultados do Benchmark Comparativo

Abaixo está o resultado da avaliação automatizada executada em **240 perguntas de teste com ground truth verificado**:

| Métrica | Naive RAG (Baseline) | Temporal RAG (MVP) | Delta |
| :--- | :---: | :---: | :---: |
| **Acurácia Geral (%)** | **16.7%** | **100.0%** | **+83.3%** |
| **Recall de Citação de Fontes (%)** | 48.8% | **99.6%** | **+50.8%** |
| **Precisão de Citação (%)** | 16.2% | **85.4%** | **+69.2%** |
| **Latência Média por Consulta** | 5.28 ms | **1.33 ms** | **4x mais rápido** |

### Desempenho por Categoria Temporal

| Categoria | Amostras | Naive RAG | Temporal RAG | Descrição do Caso de Teste |
| :--- | :---: | :---: | :---: | :--- |
| `aggregation` | 40 | 0.0% | **100.0%** | Contagem exata de faturas em janela fechada |
| `as_of_state` | 40 | 100.0% | **100.0%** | Identificação do estado ativo em data retroativa |
| `change_detection` | 40 | 0.0% | **100.0%** | Detecção da data exata em que ocorreu migração |
| `contradiction` | 40 | 0.0% | **100.0%** | Resolução de endereços/contatos antes vs depois da alteração |
| `temporal_ordering`| 40 | 0.0% | **100.0%** | Identificação cronológica do primeiro evento pós-corte |
| `unanswerable` | 40 | 0.0% | **100.0%** | Abstenção correta (*ABSTAIN*) antes da criação da conta |

---

## 🚀 Como Executar o Projeto

### Pré-requisitos
- Python 3.10+
- (Opcional) Docker & Docker Compose para rodar PostgreSQL com `pgvector`

### Instalação

```bash
# Clone ou acesse o repositório
cd temporal_RAG

# Instale o pacote em modo editável
pip install -e .
```

### 1. Gerar Dados Sintéticos e Benchmark
Gera 250 clientes sintéticos com timelines bitemporais realistas (6.400+ eventos) e gera o dataset de benchmark com 240 perguntas:

```bash
make seed
# ou:
python -m temporal_rag.cli seed --customers 250
```

### 2. Rodar Avaliação do Benchmark (Métricas e Ablação)
Executa o benchmark completo comparando o Baseline (Naive RAG) contra o Temporal RAG:

```bash
make eval
# ou:
python -m temporal_rag.cli eval
```

### 3. Comparação Interativa no Terminal
Visualize o erro clássico do Naive RAG e o acerto do Temporal RAG:

```bash
make compare
# ou:
python -m temporal_rag.cli compare
```

### 4. Iniciar a Dashboard Interativa (Web UI & API)
Inicia o servidor FastAPI com interface visual moderna para navegação bitemporal e consultas:

```bash
make run
# ou:
python -m temporal_rag.cli serve
```
Acesse no seu navegador: **[http://localhost:8000](http://localhost:8000)**

### 5. Executar os Testes Unitários

```bash
python -m pytest -v tests/
```

---

## 📁 Estrutura do Código

```
temporal_RAG/
├── data/
│   ├── crm_temporal.db            # Banco SQLite local bitemporal
│   ├── benchmark_dataset.json     # Perguntas com ground truth e event IDs
│   └── evaluation_report.json    # Relatório de acurácia, latência e citações
├── src/
│   └── temporal_rag/
│       ├── models/schema.py       # Modelos Pydantic (valid_time, recorded_at, EventType)
│       ├── storage/db.py          # Camada de banco de dados (SQLite e Postgres)
│       ├── generator/synthetic.py # Gerador determinístico de eventos e timelines
│       ├── benchmark/             # Gerador de dataset com gabarito temporal
│       ├── retrieval/
│       │   ├── embeddings.py      # TF-IDF e vetorizador semântico
│       │   ├── naive_retriever.py # Baseline sem filtros temporais
│       │   └── temporal_retriever.py # Retriever com filtro bitemporal e recency decay
│       ├── agent/
│       │   ├── temporal_parser.py # Extração de datas e classificação de intenção
│       │   └── agent.py           # Agente temporal com rastreabilidade e citações
│       ├── eval/evaluator.py      # Harness de medição de acurácia, recall e latência
│       ├── server/app.py          # API FastAPI e Dashboard Web
│       └── cli.py                 # Interface CLI completa
├── tests/                         # Suite de testes unitários automatizados
├── docker-compose.yml             # Postgres 16 com extensão pgvector
├── Makefile                       # Atalhos de execução (seed, eval, compare, run, test)
└── pyproject.toml
```

---

## 🐳 Executando com Docker Compose (Postgres + pgvector)

Para subir o banco PostgreSQL 16 com suporte vetorial nativo:

```bash
docker compose up -d
```
Defina a variável de ambiente:
```bash
set DATABASE_URL=postgresql://temporal_user:temporal_pass@localhost:5432/temporal_crm
```
E execute `make seed` normalmente para popular o Postgres.
