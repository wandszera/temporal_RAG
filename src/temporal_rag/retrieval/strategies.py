"""Variantes de recuperação para a ablação: TF-IDF, BM25, LSA, híbrido (RRF), filtro temporal e decaimento."""

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from temporal_rag.agent.temporal_parser import TemporalIntentParser
from temporal_rag.config import settings
from temporal_rag.models.schema import BenchmarkCategory, TimelineEvent

TOKEN = re.compile(r"\w+", re.UNICODE)
RRF_K = 60


def tokenize(text: str) -> List[str]:
    stripped = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return TOKEN.findall(stripped.lower())


def event_text(event: TimelineEvent) -> str:
    return f"{event.title}. {event.content}"


@dataclass(frozen=True)
class Variant:
    name: str
    scorer: str  # "tfidf" | "bm25" | "lsa" | "hybrid"
    temporal_filter: bool = False
    decay: bool = False


DEFAULT_VARIANTS: Tuple[Variant, ...] = (
    Variant("tfidf", "tfidf"),
    Variant("bm25", "bm25"),
    Variant("lsa", "lsa"),
    Variant("hybrid (bm25+lsa, RRF)", "hybrid"),
    Variant("tfidf + filtro temporal", "tfidf", temporal_filter=True),
    Variant("bm25 + filtro temporal", "bm25", temporal_filter=True),
    Variant("hybrid + filtro temporal", "hybrid", temporal_filter=True),
    Variant("hybrid + filtro + decaimento", "hybrid", temporal_filter=True, decay=True),
)


class Bm25:
    """BM25 Okapi com estatísticas (idf, tamanho médio) calculadas sobre o corpus inteiro."""

    def __init__(self, corpus_tokens: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        n_docs = len(corpus_tokens)
        doc_freq: Counter = Counter()
        for tokens in corpus_tokens:
            doc_freq.update(set(tokens))
        self.idf: Dict[str, float] = {
            term: math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)) for term, df in doc_freq.items()
        }
        self.avg_len = sum(len(t) for t in corpus_tokens) / max(n_docs, 1)

    def scores(self, query_tokens: Sequence[str], docs_tokens: Sequence[Sequence[str]]) -> np.ndarray:
        result = np.zeros(len(docs_tokens))
        for i, tokens in enumerate(docs_tokens):
            freqs = Counter(tokens)
            length = len(tokens) or 1
            total = 0.0
            for term in set(query_tokens):
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * length / self.avg_len)
                total += self.idf.get(term, 0.0) * tf * (self.k1 + 1) / denom
            result[i] = total
        return result


def infer_filters(query: str) -> Dict[str, Optional[datetime]]:
    """Deriva filtros apenas do texto da pergunta (sem usar categoria ou data do gabarito)."""
    dates = TemporalIntentParser.extract_dates(query)
    intent = TemporalIntentParser.classify_intent(query)
    filters: Dict[str, Optional[datetime]] = {"as_of": None, "start": None, "end": None}
    if len(dates) >= 2:
        filters["start"], filters["end"] = min(dates[:2]), max(dates[:2])
    elif len(dates) == 1 and intent in (BenchmarkCategory.AS_OF_STATE, BenchmarkCategory.CONTRADICTION):
        filters["as_of"] = dates[0]
    return filters


def apply_temporal_filter(events: List[TimelineEvent], filters: Dict[str, Optional[datetime]]) -> List[TimelineEvent]:
    kept = events
    if filters["as_of"] is not None:
        ref = filters["as_of"]
        # válido no mundo real na data E já registrado no CRM (sem vazamento do futuro)
        kept = [e for e in kept if e.is_valid_as_of(ref) and e.was_recorded_by(ref + timedelta(days=1))]
    if filters["start"] is not None and filters["end"] is not None:
        kept = [e for e in kept if filters["start"] <= e.valid_from <= filters["end"]]
    return kept


def _minmax(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    span = values.max() - values.min()
    return (values - values.min()) / span if span > 0 else np.zeros_like(values)


def _rank_positions(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(len(values), dtype=int)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


class RetrievalLab:
    """Ajusta TF-IDF, LSA e BM25 no corpus inteiro e ordena eventos de um cliente por variante."""

    def __init__(self, corpus_events: Sequence[TimelineEvent], lsa_components: int = 48, seed: int = 42):
        texts = [event_text(e) for e in corpus_events]
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, strip_accents="unicode", lowercase=True)
        matrix = self.tfidf.fit_transform(texts)
        components = max(2, min(lsa_components, matrix.shape[0] - 1, matrix.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=components, random_state=seed)
        self.svd.fit(matrix)
        self.bm25 = Bm25([tokenize(t) for t in texts])

    def _scores(self, scorer: str, query: str, events: List[TimelineEvent]) -> np.ndarray:
        texts = [event_text(e) for e in events]
        if scorer == "bm25":
            return self.bm25.scores(tokenize(query), [tokenize(t) for t in texts])
        doc_vecs = self.tfidf.transform(texts)
        query_vec = self.tfidf.transform([query])
        if scorer == "tfidf":
            return (doc_vecs @ query_vec.T).toarray().ravel()
        if scorer == "lsa":
            d = self.svd.transform(doc_vecs)
            q = self.svd.transform(query_vec)
            norms = np.linalg.norm(d, axis=1) * (np.linalg.norm(q) or 1.0)
            return (d @ q.ravel()) / np.where(norms == 0, 1.0, norms)
        if scorer == "hybrid":
            bm = self._scores("bm25", query, events)
            ls = self._scores("lsa", query, events)
            return 1.0 / (RRF_K + _rank_positions(bm)) + 1.0 / (RRF_K + _rank_positions(ls))
        raise ValueError(f"scorer desconhecido: {scorer}")

    def rank(self, variant: Variant, events: List[TimelineEvent], query: str) -> List[str]:
        """IDs dos eventos ordenados do mais para o menos relevante."""
        filters = infer_filters(query)
        pool = apply_temporal_filter(events, filters) if variant.temporal_filter else list(events)
        if not pool:
            return []
        scores = self._scores(variant.scorer, query, pool)
        if variant.decay:
            ref = filters["as_of"] or filters["end"]
            if ref is not None:
                base = _minmax(scores)
                days = np.array([abs((e.valid_from - ref).total_seconds()) / 86400.0 for e in pool])
                scores = 0.7 * base + 0.3 * np.exp(-settings.recency_decay_lambda * days)
        order = np.argsort(-scores, kind="stable")
        return [pool[i].id for i in order]
