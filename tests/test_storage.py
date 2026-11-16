"""Testes para camada de banco e consultas bitemporais."""

from datetime import datetime
import pytest
from temporal_rag.models.schema import Customer, EventType, TimelineEvent
from temporal_rag.storage.db import Database


@pytest.fixture
def test_db(tmp_path):
    db_file = tmp_path / "test_crm.db"
    return Database(db_url=f"sqlite:///{db_file}")


def test_customer_crud(test_db):
    cust = Customer(
        id="CUST_9999",
        name="Teste Silva",
        email="teste@empresa.com",
        phone="(11) 99999-9999",
        company="Empresa Teste",
        document_cpf="123.456.789-00",
        created_at=datetime(2023, 1, 15, 10, 0),
        initial_plan="Starter",
        current_plan="Pro",
        address="Rua Teste, 100",
    )
    test_db.save_customer(cust)

    fetched = test_db.get_customer("CUST_9999")
    assert fetched is not None
    assert fetched.name == "Teste Silva"
    assert fetched.initial_plan == "Starter"


def test_bitemporal_event_filtering(test_db):
    cust_id = "CUST_0001"
    evt1 = TimelineEvent(
        id="EVT_001",
        customer_id=cust_id,
        event_type=EventType.PLAN_CHANGE,
        title="Plano Starter",
        content="Plano Starter ativo",
        valid_from=datetime(2023, 1, 1),
        valid_to=datetime(2023, 6, 1),
        recorded_at=datetime(2023, 1, 1, 12, 0),
    )
    evt2 = TimelineEvent(
        id="EVT_002",
        customer_id=cust_id,
        event_type=EventType.PLAN_CHANGE,
        title="Plano Pro",
        content="Plano Pro ativo",
        valid_from=datetime(2023, 6, 1),
        valid_to=None,
        recorded_at=datetime(2023, 6, 1, 12, 0),
    )
    test_db.save_events([evt1, evt2])

    # Consulta As-Of em Março de 2023: deve retornar apenas o evt1
    march_events = test_db.get_bitemporal_events(
        customer_id=cust_id,
        as_of_valid_time=datetime(2023, 3, 15),
    )
    assert len(march_events) == 1
    assert march_events[0].id == "EVT_001"

    # Consulta As-Of em Agosto de 2023: deve retornar apenas o evt2
    aug_events = test_db.get_bitemporal_events(
        customer_id=cust_id,
        as_of_valid_time=datetime(2023, 8, 1),
    )
    assert len(aug_events) == 1
    assert aug_events[0].id == "EVT_002"
