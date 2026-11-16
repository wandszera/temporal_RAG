# Segurança, privacidade e observabilidade

> Projeto de aprendizado: estes controles demonstram ideias básicas e **não** tornam o sistema pronto para produção. Os dados são sintéticos.

## O que existe

| Controle | Onde | O que faz |
| :--- | :--- | :--- |
| Guarda de entrada | `security.query_is_safe`, usada na rota `/api/query` e no nó `check_temporal_bounds` do agente | Bloqueia pedidos de override ("ignore/desconsidere/esqueça ... instruções/regras/dito antes/acima"), "sem restrições", "todos os clientes" e qualquer `CUST_xxxx` diferente do cliente da sessão |
| Sinal `unsafe` do extrator | `agent/intent.py` | Um extrator de LLM pode marcar a pergunta como insegura; o agente então bloqueia antes de recuperar |
| Mascaramento de PII | `security.redact_pii`, `public_response` | Troca e-mail, CPF e telefone por marcadores na resposta e na trilha; não devolve o conteúdo bruto dos eventos recuperados |
| API key opcional | `TEMPORAL_RAG_API_KEY` | Se definida, todas as rotas `/api/` exigem o cabeçalho `X-API-Key` (comparação em tempo constante). Sem a variável, o modo é demonstração local, sem autenticação |
| Métricas de latência | `GET /api/metrics` | Contagem, p50 e p95 por rota, em memória (últimas 1.000 chamadas). Não guarda perguntas nem respostas. O custo de tokens é zero porque não há LLM |

## Testes relacionados

`tests/test_security.py` e `tests/test_benchmark_v2.py` verificam que a guarda barra todas as perguntas adversariais geradas, que parafraseados de override são bloqueados, e que a resposta pública remove PII e omite eventos brutos. `tests/test_intent.py` verifica o bloqueio quando o extrator marca `unsafe`.

## Limites conhecidos

- A chave de API é única e global: não há identidade por usuário nem autorização por cliente. O `customer_id` da requisição é confiado.
- As rotas de listagem de clientes, detalhes e eventos expõem PII sintética a quem tem a chave (ou a qualquer pessoa, no modo demonstração).
- Não há row-level security no banco.
- A guarda é baseada em expressões regulares: uma redação inesperada passa. Em perguntas parafraseadas não vistas, o bloqueio de `adversarial` ficou em 75% (ver `AVALIACAO.md`).
- O mascaramento cobre padrões comuns (e-mail, CPF, telefone); não substitui uma ferramenta de detecção de PII.
- Injeção **dentro dos dados** (e-mails sintéticos com texto adversarial no corpus) não é avaliada: o agente atual não passa o texto dos eventos a nenhum LLM, então o risco ainda não existe, mas também não foi medido.
- A configuração de CORS da API de demonstração permite qualquer origem (`allow_origins=["*"]`) e deve ser restringida fora de demonstração.
- A API e a interface web **não foram executadas nesta sessão de documentação**; só as funções de segurança foram testadas diretamente.
