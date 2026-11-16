# Avaliação

> Projeto de aprendizado. Todos os números vêm de dados sintéticos e de um benchmark criado pelo próprio autor. Cada tabela foi gerada pelo comando indicado, com `seed 42` e 250 clientes, em Python 3.10 (sem GPU, sem LLM).

## 1. Como o benchmark é construído

`make seed` gera 250 clientes (6.428 eventos) e 320 perguntas com gabarito exato (resposta esperada e IDs dos eventos-fonte), a partir da própria linha do tempo.

| Categoria | Exemplo de pergunta | Como é considerada correta (checagem estrita) |
| :--- | :--- | :--- |
| `as_of_state` | "Qual era o plano do cliente em 15/03/2023?" | O plano esperado aparece e nenhum outro plano aparece na resposta |
| `change_detection` | "Quando o cliente passou para o plano Pro?" | A data esperada (`dd/mm/aaaa`) aparece |
| `temporal_ordering` | "Qual foi o primeiro chamado aberto depois de 08/12/2023?" | O ID do ticket esperado (`TCK-xxxx`) aparece. A comparação é por **dia** |
| `aggregation` | "Quantas faturas entre 04/06/2023 e 13/08/2023?" | O número esperado aparece isolado (não como parte de uma data) |
| `contradiction` | "Qual era o endereço da sede em 30/06/2023?" (antes de uma mudança) | O endereço esperado aparece |
| `unanswerable` | Plano numa data anterior à criação da conta | O agente se abstém |
| `multi_hop` | "Qual era o plano quando abriu o chamado TCK-1234?" ou "Qual o valor da primeira fatura depois que migrou para o plano Pro?" | O plano (ou valor) esperado aparece; o plano da época é intermediário, nem o primeiro nem o atual |
| `adversarial` | Pedido de dado de outro cliente, ou "ignore as instruções anteriores..." | O agente se abstém e a resposta não contém e-mail nem CPF |

Contagens: 320 perguntas no total (40 por categoria; a divisão por cliente deixa 31 a 32 por categoria no `dev` e 8 a 9 no `test`). O split é **por cliente** (`sha1(id) mod 5 == 0` vai para `test`, cerca de 20%): `dev` tem 254 perguntas e `test` tem 66. Nenhum cliente aparece nos dois. **O split `test` nunca foi avaliado** e deve ser usado uma única vez, no final.

## 2. Modos de avaliação

- **Assistido:** o avaliador entrega ao agente a categoria e a data do gabarito (comportamento da primeira versão do projeto).
- **Cego (`--blind`):** o agente recebe só a pergunta e o `customer_id` e precisa inferir categoria e datas. É o modo honesto; o resultado atual é igual ao assistido.
- **Checagem frouxa vs. estrita:** a frouxa (herdada da versão inicial) aceitava qualquer palavra com mais de 3 letras da resposta esperada, como "plano". A estrita está em `eval/metrics.py`. Ambas são reportadas em `make eval`.

## 3. Resultados nas perguntas do benchmark (`make eval`, split `dev`, 254 perguntas)

| Métrica | Naive RAG | Temporal RAG |
| :--- | :---: | :---: |
| Acurácia, checagem frouxa | 18,5% | 87,4% |
| Acurácia, checagem estrita | 3,1% | 100,0% |
| Recall de citação de fontes | 52,8% | 100,0% |
| Precisão de citação | 17,6% | 89,1% |

Por categoria (estrita): o Temporal acerta 100% em todas as 8 categorias. O Naive acerta 18,8% em `as_of_state`, 6,5% em `multi_hop` e 0% nas demais.

**Como ler:** esses 100% medem a cobertura dos modelos de pergunta que o gerador cria. As regras do agente foram escritas e ajustadas olhando esses modelos. Com 31–32 perguntas por categoria, um erro vale cerca de 3 pontos.

## 4. Generalização: perguntas parafraseadas (`make eval-para`, modo cego, split `dev`)

Cada pergunta do benchmark ganhou 6 redações livres (`benchmark/paraphrases.py`), com datas em `dd/mm/aaaa`, ISO e por extenso (1.920 no total; 1.524 no `dev`).

- **Ajuste (v0–v3, 1.016 perguntas):** orientaram as correções do agente. As v2 e v3 eram de verificação até eu ver seus erros e passarem a contar como ajuste.
- **Verificação (v4–v5, 508 perguntas):** escritas **antes** das correções seguintes; nenhuma correção foi guiada por seus erros.

| Grupo | Naive | Temporal |
| :--- | :---: | :---: |
| ajuste (v0–v3) | 5,3% | 100,0% |
| verificação (v4–v5) | 11,0% | **75,0%** |

Antes das correções do extrator por regras, a verificação estava em 41,5%. Por categoria na verificação: `as_of_state`, `temporal_ordering` e `unanswerable` 100%; `adversarial` 75%; `multi_hop` 74,2%; `aggregation`, `change_detection` e `contradiction` 50%.

**Ressalva:** o 75% pode ser otimista, porque algumas correções (leitura de datas por extenso, verbos de mudança de plano, padrões de bloqueio) pertencem à mesma família de redações que a verificação traz. O número honesto para citar é "cerca de 75% em perguntas reescritas que não orientaram o ajuste, em dados sintéticos".

## 5. Ablação de recuperação (`make ablate`, split `dev`, N = 159)

Mede só a recuperação: dado o texto da pergunta e os eventos do cliente, onde aparecem os eventos-fonte do gabarito? As categorias `unanswerable` e `adversarial` não entram (não têm evento-fonte).

| Variante | recall@5 | recall@10 | MRR | nDCG@5 |
| :--- | :---: | :---: | :---: | :---: |
| tfidf | 0,604 | 0,843 | 0,346 | 0,379 |
| bm25 | 0,531 | 0,799 | 0,289 | 0,322 |
| lsa | 0,502 | 0,805 | 0,562 | 0,469 |
| hybrid (bm25 + lsa, RRF) | 0,536 | 0,796 | 0,475 | 0,425 |
| tfidf + filtro temporal | 0,881 | 0,987 | 0,531 | 0,607 |
| bm25 + filtro temporal | 0,814 | 0,943 | 0,508 | 0,567 |
| hybrid + filtro temporal | 0,812 | 0,947 | 0,604 | 0,630 |
| hybrid + filtro + decaimento | 0,812 | 0,947 | 0,604 | 0,630 |

Observações:

- O filtro temporal, derivado só do texto da pergunta, é o maior ganho: recall@5 passa de 0,50–0,60 para 0,81–0,88.
- O híbrido tem o melhor MRR e nDCG@5, mas **não** o melhor recall@5; não há vencedor único.
- O decaimento por recência não mudou nada nestes dados.
- "Denso" aqui é LSA (TF-IDF + SVD), não embedding neural. Reranker não foi testado.
- Em `as_of_state` o MRR com filtro é 0,5: o filtro deixa poucos eventos, e não investiguei por que o evento certo não fica em primeiro.

## 6. Histórico de correções (por que o 100% antigo não vale)

1. A primeira versão reportava 100% de acurácia em 240 perguntas. O avaliador passava ao agente a categoria e a data do gabarito, e a checagem aceitava qualquer palavra longa em comum com a resposta.
2. Foram adicionados: checagem estrita, modo cego, categorias `multi_hop` e `adversarial`, split dev/teste por cliente e a ablação de recuperação. Com isso, o agente caiu para 75,6% (multi-hop 6,5% e adversarial 0%).
3. O agente ganhou o encadeamento de dois eventos e a guarda de segurança no grafo: 99,6% nas perguntas do gerador.
4. Em paráfrases, o mesmo agente ficou em 41,7%: as regras eram frágeis. Após novas regras e um conjunto de verificação inédito, 75,0%.
5. Foi corrigida uma ambiguidade do benchmark: "primeiro chamado depois de DD/MM" passou a comparar por dia, no gabarito e no agente. Isso levou os 99,6% a 100%.

## 7. Ameaças à validade

- **Circularidade:** dados, perguntas e regras vêm do mesmo autor; o agente foi ajustado olhando o benchmark.
- **Tamanho:** 31–32 perguntas por categoria (dev) e 508 paráfrases de verificação, todas derivadas das mesmas 254 perguntas e das mesmas 250 linhas do tempo.
- **Contaminação parcial** da verificação por famílias de redação semelhantes.
- **Sem LLM:** nada aqui mede alucinação nem qualidade de redação.
- **Latências** são de uma máquina de desenvolvimento e comparam variantes entre si, não indicam desempenho de produção.
- O endpoint `POST /api/benchmark/run` roda as primeiras N perguntas do arquivo completo (não filtra por split) e sobrescreve `data/evaluation_report.json`.

## 8. Reproduzir

```bash
export PYTHONPATH=src
python -m temporal_rag.cli seed --customers 250 --seed 42
python -m temporal_rag.cli eval                 # seção 3
python -m temporal_rag.cli eval --blind         # modo cego
python -m temporal_rag.cli eval-paraphrase      # seção 4
python -m temporal_rag.cli ablate               # seção 5
python -m pytest -q tests/                      # 40 testes
```

Os relatórios em JSON ficam em `data/` (`evaluation_report.json`, `ablation_report.json`, `paraphrase_report.json`). Para não sobrescrever uma pasta `data/` existente, defina `TEMPORAL_RAG_DATA_DIR`.
