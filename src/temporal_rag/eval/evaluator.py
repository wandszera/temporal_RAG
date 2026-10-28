"""Módulo de avaliação e benchmark comparativo (Naive RAG vs Temporal RAG)."""

import json
import sys
import time
from typing import Dict, List, Optional
from tabulate import tabulate

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from temporal_rag.agent.agent import BaselineAgent, TemporalAgent
from temporal_rag.config import settings
from temporal_rag.models.schema import BenchmarkCategory, BenchmarkSample
from temporal_rag.storage.db import Database


class BenchmarkEvaluator:
    """Executa a avaliação automatizada e gera as métricas comparativas."""

    def __init__(self, db: Database):
        self.db = db
        self.baseline_agent = BaselineAgent(db)
        self.temporal_agent = TemporalAgent(db)

    def evaluate_model(
        self, agent_name: str, agent_instance, samples: List[BenchmarkSample]
    ) -> Dict:
        category_stats = {}
        total_time_ms = 0.0

        for cat in BenchmarkCategory:
            category_stats[cat.value] = {
                "total": 0,
                "correct": 0,
                "citation_hits": 0,
                "citation_total_expected": 0,
                "citation_total_retrieved": 0,
            }

        for sample in samples:
            cat_key = sample.category.value
            category_stats[cat_key]["total"] += 1

            start_t = time.perf_counter()
            response = agent_instance.answer(
                query=sample.question,
                customer_id=sample.customer_id,
                as_of=sample.as_of,
                category=sample.category,
            )
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            total_time_ms += elapsed_ms

            # 1. Verifica acurácia da resposta
            is_correct = False
            expected_lower = sample.expected_answer.lower()
            resp_lower = response.answer.lower()

            if sample.category == BenchmarkCategory.UNANSWERABLE:
                # Pergunta sem resposta válida: espera-se abstenção
                is_correct = response.abstained or "abstain" in resp_lower or "não existia" in resp_lower
            elif sample.category == BenchmarkCategory.AGGREGATION:
                # Extrai dígitos esperados e compara
                expected_digits = "".join(filter(str.isdigit, expected_lower))
                is_correct = expected_digits in resp_lower if expected_digits else (expected_lower in resp_lower)
            else:
                is_correct = (expected_lower in resp_lower) or (
                    any(word in resp_lower for word in expected_lower.split() if len(word) > 3)
                )

            if is_correct:
                category_stats[cat_key]["correct"] += 1

            # 2. Métricas de Citação de Eventos
            expected_set = set(sample.expected_event_ids)
            retrieved_set = set(response.cited_event_ids)
            
            category_stats[cat_key]["citation_total_expected"] += len(expected_set)
            category_stats[cat_key]["citation_total_retrieved"] += len(retrieved_set)
            category_stats[cat_key]["citation_hits"] += len(expected_set.intersection(retrieved_set))

        # Calcula agregações
        total_samples = len(samples)
        total_correct = sum(cs["correct"] for cs in category_stats.values())
        overall_accuracy = (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0

        total_hits = sum(cs["citation_hits"] for cs in category_stats.values())
        total_exp = sum(cs["citation_total_expected"] for cs in category_stats.values())
        total_ret = sum(cs["citation_total_retrieved"] for cs in category_stats.values())

        citation_recall = (total_hits / total_exp * 100.0) if total_exp > 0 else 100.0
        citation_precision = (total_hits / total_ret * 100.0) if total_ret > 0 else 100.0
        avg_latency = total_time_ms / total_samples if total_samples > 0 else 0.0

        breakdown = {}
        for cat_name, cs in category_stats.items():
            if cs["total"] > 0:
                breakdown[cat_name] = {
                    "count": cs["total"],
                    "accuracy": round((cs["correct"] / cs["total"]) * 100.0, 1),
                }

        return {
            "model": agent_name,
            "total_questions": total_samples,
            "overall_accuracy": round(overall_accuracy, 1),
            "citation_recall": round(citation_recall, 1),
            "citation_precision": round(citation_precision, 1),
            "avg_latency_ms": round(avg_latency, 2),
            "category_breakdown": breakdown,
        }

    def run_benchmark(self, samples: List[BenchmarkSample]) -> Dict:
        print(f"\n🚀 Iniciando Benchmark com {len(samples)} perguntas...")

        print("-> Avaliando Baseline (Naive RAG)...")
        naive_metrics = self.evaluate_model("Naive RAG (Baseline)", self.baseline_agent, samples)

        print("-> Avaliando Temporal RAG...")
        temporal_metrics = self.evaluate_model("Temporal RAG", self.temporal_agent, samples)

        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_samples": len(samples),
            "results": {
                "naive_rag": naive_metrics,
                "temporal_rag": temporal_metrics,
            },
        }

        with open(settings.evaluation_report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        self.print_summary_table(naive_metrics, temporal_metrics)
        return report

    def print_summary_table(self, naive: Dict, temporal: Dict):
        summary_rows = [
            ["Acurácia Geral (%)", f"{naive['overall_accuracy']}%", f"{temporal['overall_accuracy']}%"],
            ["Recall de Citação de Fontes (%)", f"{naive['citation_recall']}%", f"{temporal['citation_recall']}%"],
            ["Precisão de Citação (%)", f"{naive['citation_precision']}%", f"{temporal['citation_precision']}%"],
            ["Latência Média (ms)", f"{naive['avg_latency_ms']} ms", f"{temporal['avg_latency_ms']} ms"],
        ]

        print("\n" + "=" * 65)
        print("📊 TABELA COMPARATIVA DE PERFORMANCE (BENCHMARK TEMPORAL)")
        print("=" * 65)
        print(tabulate(summary_rows, headers=["Métrica", "Naive RAG (Baseline)", "Temporal RAG"], tablefmt="github"))
        print("\n" + "-" * 65)
        print("📈 DETALHAMENTO POR CATEGORIA TEMPORAL (Acurácia %)")
        print("-" * 65)

        cat_rows = []
        all_cats = sorted(list(set(list(naive["category_breakdown"].keys()) + list(temporal["category_breakdown"].keys()))))
        for cat in all_cats:
            naive_acc = naive["category_breakdown"].get(cat, {}).get("accuracy", 0.0)
            temp_acc = temporal["category_breakdown"].get(cat, {}).get("accuracy", 0.0)
            count = temporal["category_breakdown"].get(cat, {}).get("count", 0)
            cat_rows.append([cat, count, f"{naive_acc}%", f"{temp_acc}%"])

        print(tabulate(cat_rows, headers=["Categoria", "Amostras", "Naive RAG", "Temporal RAG"], tablefmt="github"))
        print("=" * 65 + "\n")
