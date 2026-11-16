"""Testes comparativos entre Naive Retriever e Temporal Retriever."""

from datetime import datetime
import pytest
from temporal_rag.models.schema import EventType, TimelineEvent
from temporal_rag.retrieval.naive_retriever import NaiveRetriever
from temporal_rag.retrieval.temporal_retriever import TemporalRetriever
from temporal_rag.storage.db import Database


@pytest.fixture
def populated_db(tmp_path):
    db = Database(db_url=f"sqlite:///{tmp_path / 'retriever_test.db'}")
    cust_id = "CUST_TEST"
    events = [
        TimelineEvent(
            id="EVT_P1",
            customer_id=cust_id,
            event_type=EventType.PLAN_CHANGE,
            title="Plano Starter",
            content="Cliente migrou para o plano Starter com mensalidade de 199 reais.",
            metadata={"new_plan": "Starter"},
            valid_from=datetime(2023, 1, 1),
            valid_to=datetime(2023, 6, 1),
            recorded_at=datetime(2023, 1, 1),
        ),
        TimelineEvent(
            id="EVT_P2",
            customer_id=cust_id,
            event_type=EventType.PLAN_CHANGE,
            title="Plano Enterprise",
            content="Cliente fez upgrade para o plano Enterprise com mensalidade de 1499 reais.",
            metadata={"new_plan": "Enterprise"},
            valid_from=datetime(2023, 6, 1),
            valid_to=None,
            recorded_at=datetime(2023, 6, 1),
        ),
    ]
    db.save_events(events)
    return db


def test_temporal_retriever_respects_valid_interval(populated_db):
    retriever = TemporalRetriever(populated_db)

    # Consulta com corte temporal em Março de 2023
    results = retriever.retrieve(
        customer_id="CUST_TEST",
        query="qual é o meu plano?",
        as_of_valid_time=datetime(2023, 3, 15),
        filter_valid_interval=True,
    )

    assert len(results) == 1
    assert results[0].event.id == "EVT_P1"
    assert results[0].event.metadata.get("new_plan") == "Starter"
