"""Retriever Baseline (Naive RAG) - Sem restrições temporais."""

from typing import List, Optional
from temporal_rag.models.schema import RetrievalResult, TimelineEvent
from temporal_rag.retrieval.embeddings import TextVectorizer
from temporal_rag.storage.db import Database


class NaiveRetriever:
    """Baseline RAG que busca puramente por similaridade textual, ignorando bitemporalidade."""

    def __init__(self, db: Database):
        self.db = db
        self.vectorizer = TextVectorizer()

    def retrieve(
        self,
        customer_id: str,
        query: str,
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        events = self.db.get_all_events(customer_id)
        if not events:
            return []

        # Representa o texto de cada evento
        texts = [f"{e.title}. {e.content}" for e in events]
        similarities = self.vectorizer.compute_similarity(query, texts)

        # Associa cada evento ao seu score de similaridade
        results = [
            RetrievalResult(event=evt, score=float(sim), final_score=float(sim))
            for evt, sim in zip(events, similarities)
        ]

        # Ordena descendentemente pelo score
        results.sort(key=lambda r: r.final_score, reverse=True)
        return results[:top_k]
