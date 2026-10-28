"""Gerador de benchmark e ground truth para avaliação temporal."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
from temporal_rag.config import settings
from temporal_rag.models.schema import (
    BenchmarkCategory,
    BenchmarkSample,
    Customer,
    EventType,
    TimelineEvent,
)


class BenchmarkDatasetGenerator:
    def __init__(self, customers: List[Customer], events: List[TimelineEvent]):
        self.customers = {c.id: c for c in customers}
        # Agrupa eventos por cliente
        self.events_by_customer = {}
        for ev in events:
            self.events_by_customer.setdefault(ev.customer_id, []).append(ev)
        for cid in self.events_by_customer:
            self.events_by_customer[cid].sort(key=lambda e: e.valid_from)

    def generate_benchmark(self, limit_per_category: int = 40) -> List[BenchmarkSample]:
        samples: List[BenchmarkSample] = []
        counter = 1

        def next_qid(cat: BenchmarkCategory) -> str:
            nonlocal counter
            qid = f"Q_{cat.value[:3].upper()}_{counter:04d}"
            counter += 1
            return qid

        # 1. AS_OF_STATE: Qual era o plano em uma data específica no passado?
        as_of_samples = 0
        for customer in self.customers.values():
            if as_of_samples >= limit_per_category:
                break
            plan_events = [
                e for e in self.events_by_customer.get(customer.id, [])
                if e.event_type == EventType.PLAN_CHANGE
            ]
            if len(plan_events) >= 2:
                # Escolhe um plano intermediário
                chosen_event = plan_events[0]
                # Pega uma data no meio do período de validade
                if chosen_event.valid_to:
                    mid_delta = (chosen_event.valid_to - chosen_event.valid_from) / 2
                    query_date = chosen_event.valid_from + mid_delta
                else:
                    query_date = chosen_event.valid_from + timedelta(days=15)

                expected_plan = chosen_event.metadata.get("new_plan")
                sample = BenchmarkSample(
                    question_id=next_qid(BenchmarkCategory.AS_OF_STATE),
                    customer_id=customer.id,
                    category=BenchmarkCategory.AS_OF_STATE,
                    question=f"Qual era o plano do cliente {customer.name} ({customer.id}) na data {query_date.strftime('%d/%m/%Y')}?",
                    as_of=query_date,
                    expected_answer=f"Plano {expected_plan}",
                    expected_event_ids=[chosen_event.id],
                    description=f"Plano no passado ({expected_plan}), diferente do plano final.",
                )
                samples.append(sample)
                as_of_samples += 1

        # 2. CHANGE_DETECTION: Quando ocorreu a mudança de plano?
        change_samples = 0
        for customer in self.customers.values():
            if change_samples >= limit_per_category:
                break
            plan_events = [
                e for e in self.events_by_customer.get(customer.id, [])
                if e.event_type == EventType.PLAN_CHANGE
            ]
            if len(plan_events) >= 2:
                # Mudança para o segundo plano
                target_change = plan_events[1]
                target_plan = target_change.metadata.get("new_plan")
                change_date = target_change.valid_from
                sample = BenchmarkSample(
                    question_id=next_qid(BenchmarkCategory.CHANGE_DETECTION),
                    customer_id=customer.id,
                    category=BenchmarkCategory.CHANGE_DETECTION,
                    question=f"Em qual data o cliente {customer.name} ({customer.id}) alterou seu plano para {target_plan}?",
                    as_of=datetime(2025, 1, 1),
                    expected_answer=change_date.strftime("%d/%m/%Y"),
                    expected_event_ids=[target_change.id],
                    description="Detecção temporal da data exata de mudança de estado.",
                )
                samples.append(sample)
                change_samples += 1

        # 3. TEMPORAL_ORDERING: Primeiro ticket após uma data
        order_samples = 0
        for customer in self.customers.values():
            if order_samples >= limit_per_category:
                break
            tickets = [
                e for e in self.events_by_customer.get(customer.id, [])
                if e.event_type == EventType.TICKET and "aberto" in e.metadata.get("status", "")
            ]
            if len(tickets) >= 2:
                reference_date = tickets[0].valid_from + timedelta(days=1)
                subsequent_tickets = [t for t in tickets if t.valid_from > reference_date]
                if subsequent_tickets:
                    first_after = subsequent_tickets[0]
                    sample = BenchmarkSample(
                        question_id=next_qid(BenchmarkCategory.TEMPORAL_ORDERING),
                        customer_id=customer.id,
                        category=BenchmarkCategory.TEMPORAL_ORDERING,
                        question=f"Qual foi o primeiro chamado/ticket aberto pelo cliente {customer.name} ({customer.id}) após {reference_date.strftime('%d/%m/%Y')}?",
                        as_of=datetime(2025, 1, 1),
                        expected_answer=first_after.title,
                        expected_event_ids=[first_after.id],
                        description="Ordenação cronológica após data de corte.",
                    )
                    samples.append(sample)
                    order_samples += 1

        # 4. AGGREGATION: Quantas faturas foram emitidas em determinado período?
        agg_samples = 0
        for customer in self.customers.values():
            if agg_samples >= limit_per_category:
                break
            invoices = [
                e for e in self.events_by_customer.get(customer.id, [])
                if e.event_type == EventType.INVOICE
            ]
            if len(invoices) >= 3:
                start_range = invoices[0].valid_from - timedelta(days=5)
                end_range = invoices[2].valid_from + timedelta(days=5)
                matching = [inv for inv in invoices if start_range <= inv.valid_from <= end_range]
                count = len(matching)
                sample = BenchmarkSample(
                    question_id=next_qid(BenchmarkCategory.AGGREGATION),
                    customer_id=customer.id,
                    category=BenchmarkCategory.AGGREGATION,
                    question=f"Quantas faturas foram emitidas para o cliente {customer.name} ({customer.id}) entre {start_range.strftime('%d/%m/%Y')} e {end_range.strftime('%d/%m/%Y')}?",
                    as_of=end_range + timedelta(days=1),
                    expected_answer=f"{count} fatura(s)",
                    expected_event_ids=[inv.id for inv in matching],
                    description="Contagem temporal de faturas em janela fechada.",
                )
                samples.append(sample)
                agg_samples += 1

        # 5. CONTRADICTION: Endereço do cliente antes vs depois de mudança
        contra_samples = 0
        for customer in self.customers.values():
            if contra_samples >= limit_per_category:
                break
            updates = [
                e for e in self.events_by_customer.get(customer.id, [])
                if e.event_type == EventType.CONTACT_UPDATE
            ]
            if updates:
                upd = updates[0]
                # Pergunta antes da mudança
                date_before = upd.valid_from - timedelta(days=10)
                old_addr = upd.metadata.get("old_address")
                sample = BenchmarkSample(
                    question_id=next_qid(BenchmarkCategory.CONTRADICTION),
                    customer_id=customer.id,
                    category=BenchmarkCategory.CONTRADICTION,
                    question=f"Qual era o endereço da sede do cliente {customer.name} ({customer.id}) em {date_before.strftime('%d/%m/%Y')}?",
                    as_of=date_before,
                    expected_answer=old_addr,
                    expected_event_ids=[],
                    description="Resolução de endereço anterior à mudança.",
                )
                samples.append(sample)
                contra_samples += 1

        # 6. UNANSWERABLE: Pergunta sobre período anterior ao cadastro (abstenção)
        unans_samples = 0
        for customer in self.customers.values():
            if unans_samples >= limit_per_category:
                break
            before_creation = customer.created_at - timedelta(days=180)
            sample = BenchmarkSample(
                question_id=next_qid(BenchmarkCategory.UNANSWERABLE),
                customer_id=customer.id,
                category=BenchmarkCategory.UNANSWERABLE,
                question=f"Qual era o plano ativo do cliente {customer.name} ({customer.id}) em {before_creation.strftime('%d/%m/%Y')}?",
                as_of=before_creation,
                expected_answer="ABSTAIN",
                expected_event_ids=[],
                description="Abstenção esperada: cliente ainda não existia no CRM.",
            )
            samples.append(sample)
            unans_samples += 1

        return samples

    def save_dataset(self, samples: List[BenchmarkSample], file_path: Optional[Path] = None):
        target = file_path or settings.benchmark_file
        data = [s.model_dump(mode="json") for s in samples]
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_dataset(cls, file_path: Optional[Path] = None) -> List[BenchmarkSample]:
        target = file_path or settings.benchmark_file
        with open(target, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return [BenchmarkSample(**item) for item in raw]
