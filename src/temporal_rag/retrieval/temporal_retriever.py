"""Retriever Temporal bitemporal com filtros de validade e decaimento por recência."""

import math
from datetime import datetime
from typing import List, Optional
from temporal_rag.config import settings
from temporal_rag.models.schema import EventType, RetrievalResult, TimelineEvent
from temporal_rag.retrieval.embeddings import TextVectorizer
from temporal_rag.storage.db import Database


class TemporalRetriever:
    """Retriever avançado com consciência temporal, filtragem bitemporal e time-decay."""

    def __init__(self, db: Database, decay_lambda: float = settings.recency_decay_lambda):
        self.db = db
        self.vectorizer = TextVectorizer()
        self.decay_lambda = decay_lambda

    def retrieve(
        self,
        customer_id: str,
        query: str,
        as_of_valid_time: Optional[datetime] = None,
        as_of_recorded_time: Optional[datetime] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        event_type: Optional[EventType] = None,
        top_k: int = 5,
        filter_valid_interval: bool = False,
    ) -> List[RetrievalResult]:
        # Busca eventos considerando as restrições bitemporais
        all_events = self.db.get_all_events(customer_id)
        if not all_events:
            return []

        filtered_events: List[TimelineEvent] = []
        for evt in all_events:
            # Filtro 1: Não vazar dados gravados no futuro
            if as_of_recorded_time is not None and not evt.was_recorded_by(as_of_recorded_time):
                continue

            # Filtro 2: Validade estrita na data de referência (se for consulta pontual de estado)
            if filter_valid_interval and as_of_valid_time is not None:
                if not evt.is_valid_as_of(as_of_valid_time):
                    continue

            # Filtro 3: Janela temporal de início e fim
            if start_date is not None and evt.valid_from < start_date:
                continue
            if end_date is not None and evt.valid_from > end_date:
                continue

            # Filtro 4: Tipo de evento
            if event_type is not None and evt.event_type != event_type:
                continue

            filtered_events.append(evt)

        if not filtered_events:
            return []

        # Vetorização e cálculo de similaridade textual
        texts = [f"{e.title}. {e.content}" for e in filtered_events]
        similarities = self.vectorizer.compute_similarity(query, texts)

        results: List[RetrievalResult] = []
        ref_time = as_of_valid_time or datetime.now()

        for evt, sim in zip(filtered_events, similarities):
            sim_score = float(sim)
            # Decaimento temporal: calcula a distância em dias até o tempo de referência
            delta_days = abs((evt.valid_from - ref_time).total_seconds()) / 86400.0
            penalty = math.exp(-self.decay_lambda * delta_days)
            
            # Combinação ponderada de relevância semântica e proximidade temporal
            final_score = (0.7 * sim_score + 0.3 * penalty) if sim_score > 0 else (0.1 * penalty)

            results.append(
                RetrievalResult(
                    event=evt,
                    score=sim_score,
                    temporal_penalty=penalty,
                    final_score=final_score,
                )
            )

        # Ordena descendentemente pelo final_score
        results.sort(key=lambda r: r.final_score, reverse=True)
        return results[:top_k]
