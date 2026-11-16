"""Testes para o gerador de dados sintéticos bitemporais."""

from temporal_rag.generator.synthetic import SyntheticDataGenerator
from temporal_rag.models.schema import EventType


def test_synthetic_data_generator_determinism():
    gen1 = SyntheticDataGenerator(seed=42)
    cust1, evts1 = gen1.generate_customers_and_timelines(count=5)

    gen2 = SyntheticDataGenerator(seed=42)
    cust2, evts2 = gen2.generate_customers_and_timelines(count=5)

    assert len(cust1) == len(cust2) == 5
    assert len(evts1) == len(evts2)
    assert cust1[0].id == cust2[0].id
    assert cust1[0].name == cust2[0].name
    assert evts1[0].id == evts2[0].id


def test_bitemporal_intervals_in_plan_changes():
    gen = SyntheticDataGenerator(seed=123)
    customers, events = gen.generate_customers_and_timelines(count=3)

    for cust in customers:
        cust_events = [e for e in events if e.customer_id == cust.id]
        plan_events = [e for e in cust_events if e.event_type == EventType.PLAN_CHANGE]
        
        # Cada cliente deve ter pelo menos um plano inicial registrado
        assert len(plan_events) >= 1

        # Verifica consistência cronológica dos planos
        for i in range(len(plan_events) - 1):
            curr_p = plan_events[i]
            next_p = plan_events[i + 1]
            assert curr_p.valid_to == next_p.valid_from
            assert curr_p.valid_from < curr_p.valid_to

        # O último plano em vigor deve ter valid_to == None
        assert plan_events[-1].valid_to is None
