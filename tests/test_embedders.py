"""Embeddings plugáveis: testados com um embedder determinístico (sem modelo neural)."""

import zlib
from datetime import datetime, timedelta

import numpy as np
import pytest

from temporal_rag.models.schema import TimelineEvent
from temporal_rag.retrieval.embedders import Embedder, build_embedder
from temporal_rag.retrieval.strategies import EMBEDDING_VARIANTS, RetrievalLab


def hash_encode(texts):
    """Saco de palavras com hashing: determinístico e sem dependências."""
    out = np.zeros((len(texts), 64))
    for i, text in enumerate(texts):
        for word in text.lower().split():
            out[i, zlib.crc32(word.encode()) % 64] += 1.0
    return out


def make_events():
    base = datetime(2024, 1, 1)
    titles = ["Troca de plano para Premium", "Chamado de suporte aberto", "Endereço atualizado"]
    return [
        TimelineEvent(
            id=f"E{i}", customer_id="C1", event_type="note", title=title, content=title,
            valid_from=base + timedelta(days=i), recorded_at=base + timedelta(days=i),
        )
        for i, title in enumerate(titles)
    ]


def test_embedder_normalizes_and_caches():
    calls = []
    emb = Embedder(lambda t: (calls.append(list(t)), hash_encode(t))[1])
    first = emb.embed(["plano premium", "suporte"])
    emb.embed(["plano premium"])
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0)
    assert len(calls) == 1


def test_dense_variants_rank_relevant_event_first():
    events = make_events()
    lab = RetrievalLab(events, embedder=Embedder(hash_encode))
    for variant in EMBEDDING_VARIANTS:
        if variant.temporal_filter:
            continue
        assert lab.rank(variant, events, "troca de plano Premium")[0] == "E0"


def test_dense_requires_embedder():
    events = make_events()
    with pytest.raises(RuntimeError):
        RetrievalLab(events).rank(EMBEDDING_VARIANTS[0], events, "plano")


def test_build_embedder_rejects_bad_spec():
    with pytest.raises(ValueError):
        build_embedder("sem-dois-pontos")
