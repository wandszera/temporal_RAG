"""Testes para o agente temporal e parser de intenção."""

from datetime import datetime
import pytest
from temporal_rag.agent.agent import TemporalAgent
from temporal_rag.agent.temporal_parser import TemporalIntentParser
from temporal_rag.models.schema import BenchmarkCategory, Customer
from temporal_rag.storage.db import Database


def test_intent_parser():
    q1 = "Quantas faturas foram pagas pelo cliente CUST_0012 entre 01/01/2024 e 31/03/2024?"
    assert TemporalIntentParser.extract_customer_id(q1) == "CUST_0012"
    assert TemporalIntentParser.classify_intent(q1) == BenchmarkCategory.AGGREGATION

    dates = TemporalIntentParser.extract_dates(q1)
    assert len(dates) == 2
    assert dates[0] == datetime(2024, 1, 1)
    assert dates[1] == datetime(2024, 3, 31)


def test_agent_abstains_on_date_before_account_creation(tmp_path):
    db = Database(db_url=f"sqlite:///{tmp_path / 'agent_test.db'}")
    cust = Customer(
        id="CUST_0001",
        name="Cliente Futuro",
        email="futuro@teste.com",
        phone="1199999999",
        company="Futuro Tech",
        document_cpf="000.000.000-00",
        created_at=datetime(2023, 6, 1),
        initial_plan="Pro",
    )
    db.save_customer(cust)

    agent = TemporalAgent(db)
    # Pergunta sobre janeiro de 2023 (antes de junho de 2023)
    resp = agent.answer(
        query="Qual era o meu plano em 15/01/2023?",
        customer_id="CUST_0001",
        as_of=datetime(2023, 1, 15),
    )

    assert resp.abstained is True
    assert "ABSTAIN" in resp.answer
    assert resp.confidence == 1.0
