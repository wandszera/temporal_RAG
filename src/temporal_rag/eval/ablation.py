"""Ablação de recuperação: compara variantes por recall@k, MRR e nDCG@k contra os eventos-fonte do gabarito."""

import json
import time
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

from tabulate import tabulate

from temporal_rag.config import settings
from temporal_rag.eval.metrics import average, ndcg_at_k, recall_at_k, reciprocal_rank
from temporal_rag.models.schema import BenchmarkCategory, BenchmarkSample
from temporal_rag.retrieval.embedders import Embedder
from temporal_rag.retrieval.strategies import DEFAULT_VARIANTS, EMBEDDING_VARIANTS, RetrievalLab, Variant
from temporal_rag.storage.db import Database

# Categorias sem evento-fonte a recuperar (abstenção / bloqueio) não entram na ablação.
EXCLUDED = {BenchmarkCategory.UNANSWERABLE, BenchmarkCategory.ADVERSARIAL}


def select_samples(samples: Sequence[BenchmarkSample], split: str) -> List[BenchmarkSample]:
    chosen = [s for s in samples if s.category not in EXCLUDED and s.expected_event_ids]
    if split != "all":
        chosen = [s for s in chosen if s.split == split]
    return chosen


def run_ablation(
    db: Database,
    samples: Sequence[BenchmarkSample],
    split: str = "dev",
    variants: Sequence[Variant] = DEFAULT_VARIANTS,
    ks: Sequence[int] = (5, 10),
    save: bool = True,
    embedder: Optional[Embedder] = None,
) -> Dict:
    chosen = select_samples(samples, split)
    customers = db.list_customers(limit=100000)
    events_by_customer = {c.id: db.get_all_events(c.id) for c in customers}
    corpus = [e for events in events_by_customer.values() for e in events]
    lab = RetrievalLab(corpus, embedder=embedder)
    if embedder is not None:
        variants = tuple(variants) + EMBEDDING_VARIANTS

    rows = []
    per_category: Dict[str, Dict[str, float]] = {}
    for variant in variants:
        recalls = {k: [] for k in ks}
        rr_all: List[float] = []
        ndcg_all: List[float] = []
        by_cat: Dict[str, List[float]] = defaultdict(list)
        start = time.perf_counter()
        for sample in chosen:
            ranked = lab.rank(variant, events_by_customer.get(sample.customer_id, []), sample.question)
            for k in ks:
                recalls[k].append(recall_at_k(ranked, sample.expected_event_ids, k))
            rr = reciprocal_rank(ranked, sample.expected_event_ids)
            rr_all.append(rr)
            ndcg_all.append(ndcg_at_k(ranked, sample.expected_event_ids, ks[0]))
            by_cat[sample.category.value].append(rr)
        elapsed_ms = (time.perf_counter() - start) * 1000.0 / max(len(chosen), 1)
        rows.append({
            "variant": variant.name,
            **{f"recall@{k}": round(average(recalls[k]), 3) for k in ks},
            "mrr": round(average(rr_all), 3),
            f"ndcg@{ks[0]}": round(average(ndcg_all), 3),
            "ms_per_query": round(elapsed_ms, 2),
        })
        per_category[variant.name] = {c: round(average(v), 3) for c, v in sorted(by_cat.items())}

    report = {
        "split": split,
        "n_samples": len(chosen),
        "seed": settings.seed,
        "embedder": embedder.name if embedder else None,
        "ks": list(ks),
        "variants": rows,
        "mrr_by_category": per_category,
    }
    if save:
        with open(settings.ablation_report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    return report


def print_ablation(report: Dict) -> None:
    print(f"\nAblação de recuperação — split={report['split']}, N={report['n_samples']}, seed={report['seed']}")
    print(tabulate([list(r.values()) for r in report["variants"]], headers=list(report["variants"][0].keys()), tablefmt="github"))
    cats = sorted({c for v in report["mrr_by_category"].values() for c in v})
    rows = [[name] + [by_cat.get(c, "-") for c in cats] for name, by_cat in report["mrr_by_category"].items()]
    print("\nMRR por categoria")
    print(tabulate(rows, headers=["variante"] + cats, tablefmt="github"))
