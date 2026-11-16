"""Benchmark difícil (multi-hop, adversarial), split dev/teste, métricas e ablação de recuperação."""

from datetime import datetime

import pytest

from temporal_rag.benchmark.dataset_generator import BenchmarkDatasetGenerator
from temporal_rag.eval.ablation import run_ablation, select_samples
from temporal_rag.eval.metrics import ndcg_at_k, recall_at_k, reciprocal_rank, strict_correct
from temporal_rag.generator.synthetic import SyntheticDataGenerator
from temporal_rag.models.schema import AgentResponse, BenchmarkCategory, BenchmarkSample, EventType, TimelineEvent
from temporal_rag.retrieval.strategies import Bm25, DEFAULT_VARIANTS, RetrievalLab, Variant, infer_filters, tokenize
from temporal_rag.security import query_is_safe
from temporal_rag.storage.db import Database


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    db = Database(db_url=f"sqlite:///{tmp_path_factory.mktemp('v2') / 'v2.db'}")
    customers, events = SyntheticDataGenerator(seed=7).generate_customers_and_timelines(count=60)
    for c in customers:
        db.save_customer(c)
    db.save_events(events)
    gen = BenchmarkDatasetGenerator(customers, events)
    return db, customers, events, gen.generate_full_benchmark(limit_per_category=10)


def _sample(category, expected, **kw):
    return BenchmarkSample(
        question_id="Q", customer_id="CUST_0001", category=category, question="q",
        as_of=datetime(2025, 1, 1), expected_answer=expected, expected_event_ids=kw.get("ids", []),
    )


def _resp(answer, abstained=False):
    return AgentResponse(query="q", customer_id="CUST_0001", answer=answer, confidence=1.0, abstained=abstained)


def test_ranking_metrics_known_values():
    ranked = ["a", "b", "c", "d"]
    assert recall_at_k(ranked, ["b", "z"], 2) == 0.5
    assert reciprocal_rank(ranked, ["c"]) == pytest.approx(1 / 3)
    assert reciprocal_rank(ranked, ["z"]) == 0.0
    assert ndcg_at_k(["a", "b"], ["a", "b"], 5) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "a"], ["a"], 5) < 1.0
    assert recall_at_k(ranked, [], 5) == 0.0


def test_strict_check_rejects_lenient_false_positives():
    plan = _sample(BenchmarkCategory.AS_OF_STATE, "Plano Starter")
    assert strict_correct(plan, _resp("O plano é Starter."))
    assert not strict_correct(plan, _resp("O plano é Pro."))  # a checagem frouxa aceitaria por causa de "plano"
    assert not strict_correct(plan, _resp("Mudou de Starter para Pro."))  # dois planos competindo

    agg = _sample(BenchmarkCategory.AGGREGATION, "3 fatura(s)")
    assert strict_correct(agg, _resp("Foram 3 faturas entre 05/01/2023 e 05/04/2023."))
    assert not strict_correct(agg, _resp("Foram 4 faturas entre 03/01/2023 e 05/04/2023."))

    adv = _sample(BenchmarkCategory.ADVERSARIAL, "BLOCK")
    assert strict_correct(adv, _resp("Pedido bloqueado.", abstained=True))
    assert not strict_correct(adv, _resp("Bloqueado, mas o e-mail é ana@empresa.com.br", abstained=True))
    assert not strict_correct(adv, _resp("O plano é Pro."))


def test_split_is_deterministic_and_disjoint(world):
    _, _, _, samples = world
    by_customer = {}
    for s in samples:
        by_customer.setdefault(s.customer_id, set()).add(s.split)
    assert all(len(v) == 1 for v in by_customer.values())
    assert BenchmarkDatasetGenerator.split_for_customer("CUST_0001") == BenchmarkDatasetGenerator.split_for_customer("CUST_0001")
    assert {s.split for s in samples} == {"dev", "test"}


def test_hard_samples_are_valid_multi_hop_and_adversarial(world):
    _, customers, events, samples = world
    by_id = {e.id: e for e in events}
    multi = [s for s in samples if s.category == BenchmarkCategory.MULTI_HOP]
    assert multi and all(len(s.expected_event_ids) == 2 for s in multi)
    assert all(e in by_id for s in multi for e in s.expected_event_ids)
    adversarial = [s for s in samples if s.category == BenchmarkCategory.ADVERSARIAL]
    assert adversarial
    # a camada de segurança da API deve barrar todas as perguntas adversariais geradas
    assert all(not query_is_safe(s.question, s.customer_id) for s in adversarial)


def test_multi_hop_plan_is_intermediate(world):
    _, _, events, samples = world
    plans = {}
    for e in events:
        if e.event_type == EventType.PLAN_CHANGE:
            plans.setdefault(e.customer_id, []).append(e)
    for s in samples:
        if s.category == BenchmarkCategory.MULTI_HOP and s.question_id.startswith("Q_MHA"):
            plan_event_id = s.expected_event_ids[1]
            ordered = [p.id for p in sorted(plans[s.customer_id], key=lambda p: p.valid_from)]
            assert plan_event_id not in (ordered[0], ordered[-1])


def test_bm25_prefers_matching_document():
    docs = [tokenize("fatura mensal do plano pro"), tokenize("falha de autenticação via sso"), tokenize("pesquisa nps nota 9")]
    bm25 = Bm25(docs)
    scores = bm25.scores(tokenize("problema de autenticação sso"), docs)
    assert scores.argmax() == 1


def test_infer_filters_uses_only_question_text():
    assert infer_filters("Qual era o plano em 15/03/2023?")["as_of"] == datetime(2023, 3, 15)
    both = infer_filters("Quantas faturas entre 01/02/2023 e 01/05/2023?")
    assert both["start"] == datetime(2023, 2, 1) and both["end"] == datetime(2023, 5, 1)
    assert infer_filters("Qual era o plano quando abriu o chamado TCK-1234?") == {"as_of": None, "start": None, "end": None}


def test_temporal_filter_removes_events_not_valid_as_of(world):
    db, customers, events, _ = world
    lab = RetrievalLab(events)
    cust = customers[0]
    cust_events = db.get_all_events(cust.id)
    plans = [e for e in cust_events if e.event_type == EventType.PLAN_CHANGE]
    target = plans[0]
    day = target.valid_from.replace(hour=12)
    ranked = lab.rank(Variant("f", "bm25", temporal_filter=True), cust_events, f"Qual era o plano em {day.strftime('%d/%m/%Y')}?")
    valid_ids = {e.id for e in cust_events if e.is_valid_as_of(day) and e.was_recorded_by(day)}
    assert set(ranked) <= valid_ids


def test_ablation_report_structure_and_bounds(world):
    db, _, _, samples = world
    chosen = select_samples(samples, "dev")
    assert all(s.category not in (BenchmarkCategory.UNANSWERABLE, BenchmarkCategory.ADVERSARIAL) for s in chosen)
    report = run_ablation(db, samples, split="dev", save=False)
    assert len(report["variants"]) == len(DEFAULT_VARIANTS)
    for row in report["variants"]:
        assert 0.0 <= row["mrr"] <= 1.0 and 0.0 <= row["recall@5"] <= row["recall@10"] <= 1.0
