"""Parser de intenção temporal e extração de datas a partir de perguntas."""

import re
from datetime import datetime
from typing import Optional, Tuple
from dateutil import parser as dateutil_parser
from temporal_rag.models.schema import BenchmarkCategory


class TemporalIntentParser:
    """Extrai entidades temporais, cliente e intenção da consulta."""

    # Expressões regulares para datas brasileiras (dd/mm/yyyy) e ISO (yyyy-mm-dd)
    DATE_BR_REGEX = re.compile(r"(\b\d{1,2}/\d{1,2}/\d{4}\b)")
    DATE_ISO_REGEX = re.compile(r"(\b\d{4}-\d{1,2}-\d{1,2}\b)")
    CUSTOMER_ID_REGEX = re.compile(r"\b(CUST_\d{4})\b", re.IGNORECASE)
    MONTHS_PT = {
        "janeiro": 1, "fevereiro": 2, "março": 3, "marco": 3, "abril": 4, "maio": 5, "junho": 6,
        "julho": 7, "agosto": 8, "setembro": 9, "outubro": 10, "novembro": 11, "dezembro": 12,
    }
    DATE_TEXT_REGEX = re.compile(
        r"\b(\d{1,2})\s+de\s+(" + "|".join(MONTHS_PT) + r")\s+de\s+(\d{4})\b", re.IGNORECASE
    )
    TICKET_ID_REGEX = re.compile(r"\b(TCK-\d+)\b", re.IGNORECASE)

    @classmethod
    def extract_customer_id(cls, text: str) -> Optional[str]:
        match = cls.CUSTOMER_ID_REGEX.search(text)
        if match:
            return match.group(1).upper()
        return None

    @classmethod
    def extract_ticket_id(cls, text: str) -> Optional[str]:
        match = cls.TICKET_ID_REGEX.search(text)
        return match.group(1).upper() if match else None

    @classmethod
    def is_multi_hop(cls, text: str) -> bool:
        """Perguntas que ligam dois eventos: ticket -> plano vigente, ou mudança de plano -> próxima fatura."""
        lower = text.lower()
        ticket_to_plan = cls.extract_ticket_id(text) is not None and "plano" in lower
        mentions_plan = "plano" in lower or re.search(r"\b(free|starter|pro|enterprise)\b", lower) is not None
        plan_to_invoice = "fatura" in lower and mentions_plan and any(
            w in lower for w in ["depois", "após", "seguinte", "posterior"]
        ) and any(w in lower for w in ["migrou", "mudou", "alterou", "troca", "mudança", "upgrade"])
        return ticket_to_plan or plan_to_invoice

    @classmethod
    def extract_dates(cls, text: str) -> list[datetime]:
        found_dates = []
        
        # Procura por dd/mm/yyyy
        for m in cls.DATE_BR_REGEX.finditer(text):
            try:
                dt = datetime.strptime(m.group(1), "%d/%m/%Y")
                found_dates.append(dt)
            except Exception:
                pass

        # Procura por yyyy-mm-dd
        for m in cls.DATE_ISO_REGEX.finditer(text):
            try:
                dt = datetime.strptime(m.group(1), "%Y-%m-%d")
                found_dates.append(dt)
            except Exception:
                pass

        # Procura por datas por extenso: "15 de março de 2023"
        for m in cls.DATE_TEXT_REGEX.finditer(text):
            try:
                found_dates.append(datetime(int(m.group(3)), cls.MONTHS_PT[m.group(2).lower()], int(m.group(1))))
            except Exception:
                pass

        return found_dates

    @classmethod
    def classify_intent(cls, text: str) -> BenchmarkCategory:
        lower = text.lower()
        if cls.is_multi_hop(text):
            return BenchmarkCategory.MULTI_HOP
        if any(w in lower for w in ["quantas", "quantos", "total de", "soma", "volume", "número de"]):
            return BenchmarkCategory.AGGREGATION
        if any(w in lower for w in ["quando", "em qual data", "data em que", "momento em que", "dia da migração",
                                         "data da troca", "data da migração", "data da mudança", "dia da troca", "dia da mudança"]):
            return BenchmarkCategory.CHANGE_DETECTION
        if any(w in lower for w in ["primeiro", "primeira", "após", "depois de", "subsequente", "a partir de", "próximo"]):
            return BenchmarkCategory.TEMPORAL_ORDERING
        if any(w in lower for w in ["endereço", "sede", "mudou de", "conflitante", "localização", "instalado"]):
            return BenchmarkCategory.CONTRADICTION
        return BenchmarkCategory.AS_OF_STATE
