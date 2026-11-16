"""Controles de entrada e saída para a demonstração do CRM."""

import re

from temporal_rag.models.schema import AgentResponse

CUSTOMER_ID = re.compile(r"\bCUST_\d+\b", re.IGNORECASE)
EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
CPF = re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b")
PHONE = re.compile(r"(?<!\d)(?:\+?55\s*)?\(?\d{2}\)?[\s.-]*9?\d{4}[\s.-]*\d{4}(?!\d)")
INJECTION = re.compile(
    r"(?:ignore|ignorar|desconsidere|desconsiderar|esqueça|esqueca|esquecer)"
    r".{0,60}(?:instruções|instructions|regras|dito antes|anterior|anteriormente|acima)"
    r"|sem restrições|sem restricoes|todos os clientes"
    r"|(?:system prompt|prompt de sistema|developer message|mensagem de desenvolvedor)",
    re.IGNORECASE | re.DOTALL,
)


def query_is_safe(query: str, customer_id: str) -> bool:
    """Bloqueia instruções de override e IDs de outro cliente na pergunta."""
    ids = {match.upper() for match in CUSTOMER_ID.findall(query)}
    return not INJECTION.search(query) and ids <= {customer_id.upper()}


def redact_pii(value: str) -> str:
    for pattern, label in ((EMAIL, "[EMAIL]"), (CPF, "[CPF]"), (PHONE, "[TELEFONE]")):
        value = pattern.sub(label, value)
    return value


def public_response(response: AgentResponse) -> dict:
    """Evita expor conteúdo bruto dos eventos e PII nas respostas da API."""
    data = response.model_dump(mode="json", exclude={"retrieved_events"})
    data["query"] = redact_pii(data["query"])
    data["answer"] = redact_pii(data["answer"])
    data["reasoning_trace"] = [redact_pii(step) for step in data["reasoning_trace"]]
    return data
