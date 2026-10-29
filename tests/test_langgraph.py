"""Testes unitários e de integração para o agente LangGraph."""

from datetime import datetime
import pytest
from temporal_rag.agent.langgraph_agent import LangGraphTemporalAgent
from temporal_rag.models.schema import BenchmarkCategory, Customer, EventType, TimelineEvent
from temporal_rag.storage.db import Database


@pytest.fixture
def test_db(tmp_path):
    db_file = tmp_path / "langgraph_test.db"
    db = Database(db_url=f"sqlite:///{db_file}")

    cust = Customer(
        id="CUST_0010",
        name="Tech Corp",
        email="contato@techcorp.com",
        phone="11999998888",
        company="Tech Corp Brasil",
        document_cpf="111.222.333-44",
        created_at=datetime(2023, 5, 1),
        initial_plan="Starter",
        current_plan="Pro",
        address="Av. Brasil, 500, SP",
    )
    db.save_customer(cust)

    events = [
        TimelineEvent(
            id="EVT_P1",
            customer_id="CUST_0010",
            event_type=EventType.PLAN_CHANGE,
            title="Plano Starter",
            content="Plano Starter ativo",
            metadata={"new_plan": "Starter"},
            valid_from=datetime(2023, 5, 1),
            valid_to=datetime(2023, 9, 1),
            recorded_at=datetime(2023, 5, 1),
        ),
        TimelineEvent(
            id="EVT_P2",
            customer_id="CUST_0010",
            event_type=EventType.PLAN_CHANGE,
            title="Plano Pro",
            content="Plano Pro ativo",
            metadata={"new_plan": "Pro"},
            valid_from=datetime(2023, 9, 1),
            valid_to=None,
            recorded_at=datetime(2023, 9, 1),
        ),
        TimelineEvent(
            id="EVT_INV1",
            customer_id="CUST_0010",
            event_type=EventType.INVOICE,
            title="Fatura Maio",
            content="Fatura paga de R$ 199",
            metadata={"status": "pago", "amount": 199},
            valid_from=datetime(2023, 5, 30),
            valid_to=datetime(2023, 5, 30),
            recorded_at=datetime(2023, 5, 30),
        ),
        TimelineEvent(
            id="EVT_INV2",
            customer_id="CUST_0010",
            event_type=EventType.INVOICE,
            title="Fatura Junho",
            content="Fatura paga de R$ 199",
            metadata={"status": "pago", "amount": 199},
            valid_from=datetime(2023, 6, 30),
            valid_to=datetime(2023, 6, 30),
            recorded_at=datetime(2023, 6, 30),
        ),
    ]
    db.save_events(events)
    return db


def test_langgraph_mermaid_diagram(test_db):
    agent = LangGraphTemporalAgent(test_db)
    mermaid = agent.get_graph_mermaid()
    assert isinstance(mermaid, str)
    assert "parse_and_classify" in mermaid
    assert "check_temporal_bounds" in mermaid
    assert "sql_aggregator" in mermaid
    assert "temporal_retriever" in mermaid
    assert "check_sufficiency" in mermaid
    assert "synthesize_answer" in mermaid
    assert "abstain" in mermaid


def test_langgraph_bounds_abstention(test_db):
    agent = LangGraphTemporalAgent(test_db)
    # Pergunta sobre janeiro de 2023 (conta criada em maio de 2023)
    resp = agent.answer(
        query="Qual era o meu plano em 15/01/2023?",
        customer_id="CUST_0010",
        as_of=datetime(2023, 1, 15),
    )
    assert resp.abstained is True
    assert "ABSTAIN" in resp.answer
    assert any("[LangGraph::Bounds]" in step for step in resp.reasoning_trace)


def test_langgraph_sql_aggregator_routing(test_db):
    agent = LangGraphTemporalAgent(test_db)
    resp = agent.answer(
        query="Quantas faturas foram emitidas entre 01/05/2023 e 31/07/2023?",
        customer_id="CUST_0010",
        category=BenchmarkCategory.AGGREGATION,
    )
    assert resp.abstained is False
    assert "2 fatura(s)" in resp.answer
    assert len(resp.cited_event_ids) == 2
    assert any("[LangGraph::SQLTool]" in step for step in resp.reasoning_trace)


def test_langgraph_as_of_state_retrieval(test_db):
    agent = LangGraphTemporalAgent(test_db)
    # Consulta em julho de 2023 (plano Starter ativo até setembro)
    resp = agent.answer(
        query="Qual era o meu plano contratado em 15/07/2023?",
        customer_id="CUST_0010",
        as_of=datetime(2023, 7, 15),
        category=BenchmarkCategory.AS_OF_STATE,
    )
    assert resp.abstained is False
    assert "Starter" in resp.answer
    assert "EVT_P1" in resp.cited_event_ids
    assert any("[LangGraph::Retrieve]" in step for step in resp.reasoning_trace)
