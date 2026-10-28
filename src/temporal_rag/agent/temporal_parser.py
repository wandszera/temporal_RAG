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

    @classmethod
    def extract_customer_id(cls, text: str) -> Optional[str]:
        match = cls.CUSTOMER_ID_REGEX.search(text)
        if match:
            return match.group(1).upper()
        return None

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

        return found_dates

    @classmethod
    def classify_intent(cls, text: str) -> BenchmarkCategory:
        lower = text.lower()
        if any(w in lower for w in ["quantas", "quantos", "total de", "soma", "volume"]):
            return BenchmarkCategory.AGGREGATION
        if any(w in lower for w in ["quando", "em qual data", "data em que", "momento em que"]):
            return BenchmarkCategory.CHANGE_DETECTION
        if any(w in lower for w in ["primeiro", "primeira", "após", "depois de", "subsequente"]):
            return BenchmarkCategory.TEMPORAL_ORDERING
        if any(w in lower for w in ["endereço", "sede", "mudou de", "conflitante"]):
            return BenchmarkCategory.CONTRADICTION
        return BenchmarkCategory.AS_OF_STATE
