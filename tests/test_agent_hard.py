"""O agente LangGraph deve resolver multi-hop e bloquear adversariais nas perguntas do benchmark v2."""

import pytest

from temporal_rag.agent.agent import TemporalAgent
from temporal_rag.agent.temporal_parser import TemporalIntentParser
from temporal_rag.benchmark.dataset_generator import BenchmarkDatasetGenerator
from temporal_rag.eval.metrics import strict_correct
from temporal_rag.generator.synthetic import SyntheticDataGenerator
from temporal_rag.models.schema import BenchmarkCategory
from temporal_rag.storage.db import Database


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    db = Database(db_url=f"sqlite:///{tmp_path_factory.mktemp('hard') / 'hard.db'}")
    customers, events = SyntheticDataGenerator(seed=11).generate_customers_and_timelines(count=60)
    for c in customers:
        db.save_customer(c)
    db.save_events(events)
    samples = BenchmarkDatasetGenerator(customers, events).generate_hard_samples(limit_per_category=10)
    return TemporalAgent(db), samples


@pytest.mark.parametrize("category", [BenchmarkCategory.MULTI_HOP, BenchmarkCategory.ADVERSARIAL])
@pytest.mark.parametrize("blind", [True, False])
def test_agent_solves_hard_categories(setup, category, blind):
    agent, samples = setup
    chosen = [s for s in samples if s.category == category]
    assert chosen
    for s in chosen:
        response = agent.answer(
            s.question,
            customer_id=s.customer_id,
            as_of=None if blind else s.as_of,
            category=None if blind else s.category,
        )
        assert strict_correct(s, response), (s.question, response.answer)
        if category == BenchmarkCategory.MULTI_HOP:
            assert set(response.cited_event_ids) == set(s.expected_event_ids)


def test_multi_hop_intent_classification():
    parse = TemporalIntentParser.classify_intent
    assert parse("Qual era o plano quando ele abriu o chamado TCK-1234?") == BenchmarkCategory.MULTI_HOP
    assert parse("Qual o valor da primeira fatura depois que ele migrou para o plano Pro?") == BenchmarkCategory.MULTI_HOP
    assert parse("Quantas faturas foram emitidas entre 01/02/2023 e 01/05/2023?") == BenchmarkCategory.AGGREGATION
    assert parse("Qual era o plano do cliente em 10/03/2023?") == BenchmarkCategory.AS_OF_STATE
