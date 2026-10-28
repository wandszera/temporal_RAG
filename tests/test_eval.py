"""Testes para o pipeline de avaliação de benchmark."""

from datetime import datetime
from temporal_rag.benchmark.dataset_generator import BenchmarkDatasetGenerator
from temporal_rag.eval.evaluator import BenchmarkEvaluator
from temporal_rag.generator.synthetic import SyntheticDataGenerator
from temporal_rag.storage.db import Database


def test_evaluator_run_end_to_end(tmp_path):
    db_file = tmp_path / "eval_test.db"
    db = Database(db_url=f"sqlite:///{db_file}")

    gen = SyntheticDataGenerator(seed=99)
    customers, events = gen.generate_customers_and_timelines(count=10)
    for c in customers:
        db.save_customer(c)
    db.save_events(events)

    bench_gen = BenchmarkDatasetGenerator(customers, events)
    samples = bench_gen.generate_benchmark(limit_per_category=2)
    assert len(samples) > 0

    evaluator = BenchmarkEvaluator(db)
    report = evaluator.run_benchmark(samples)

    assert "naive_rag" in report["results"]
    assert "temporal_rag" in report["results"]
    assert report["results"]["temporal_rag"]["overall_accuracy"] >= report["results"]["naive_rag"]["overall_accuracy"]
