"""Agente de Raciocínio Temporal e Baseline."""

import re
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
from temporal_rag.agent.temporal_parser import TemporalIntentParser
from temporal_rag.models.schema import (
    AgentResponse,
    BenchmarkCategory,
    EventType,
    TimelineEvent,
)
from temporal_rag.retrieval.naive_retriever import NaiveRetriever
from temporal_rag.retrieval.temporal_retriever import TemporalRetriever
from temporal_rag.storage.db import Database


class BaselineAgent:
    """Agente Naive: recupera por similaridade de texto e responde com o chunk de maior score."""

    def __init__(self, db: Database):
        self.db = db
        self.retriever = NaiveRetriever(db)

    def answer(
        self,
        query: str,
        customer_id: Optional[str] = None,
        as_of: Optional[datetime] = None,
        category: Optional[BenchmarkCategory] = None,
    ) -> AgentResponse:
        cid = customer_id or TemporalIntentParser.extract_customer_id(query)
        if not cid:
            return AgentResponse(
                query=query,
                customer_id="",
                answer="Cliente não identificado na consulta.",
                confidence=0.0,
                abstained=True,
            )

        customer = self.db.get_customer(cid)
        if not customer:
            return AgentResponse(
                query=query,
                customer_id=cid,
                answer=f"Cliente {cid} não encontrado no banco de dados.",
                confidence=0.0,
                abstained=True,
            )

        # Busca puramente semântica sem filtros de validade temporal
        retrieval_results = self.retriever.retrieve(cid, query, top_k=3)
        if not retrieval_results:
            return AgentResponse(
                query=query,
                customer_id=cid,
                answer="Nenhuma informação encontrada para responder à pergunta.",
                confidence=0.0,
                abstained=True,
            )

        top_event = retrieval_results[0].event
        cited_ids = [r.event.id for r in retrieval_results]

        # Gera resposta simples baseada no conteúdo do evento mais similar
        content = top_event.content
        if "plano" in query.lower():
            # Tenta extrair menção de plano do evento
            plan_name = top_event.metadata.get("new_plan") or customer.current_plan or customer.initial_plan
            answer_text = f"O plano é {plan_name} (informação extraída do evento {top_event.id})."
        else:
            answer_text = f"Com base no evento mais recente encontrado: {content}"

        return AgentResponse(
            query=query,
            customer_id=cid,
            as_of=as_of,
            answer=answer_text,
            confidence=0.6,
            cited_event_ids=cited_ids,
            retrieved_events=[r.event for r in retrieval_results],
            abstained=False,
            reasoning_trace=["Recuperação textual direta sem filtro bitemporal."],
        )


from temporal_rag.agent.langgraph_agent import LangGraphTemporalAgent

# TemporalAgent é acionado através do grafo compilado do LangGraph
TemporalAgent = LangGraphTemporalAgent


class LegacyRuleBasedTemporalAgent:
    """Implementação direta baseada em regras (mantida para benchmark e ablação)."""

    def __init__(self, db: Database):
        self.db = db
        self.retriever = TemporalRetriever(db)

    def answer(
        self,
        query: str,
        customer_id: Optional[str] = None,
        as_of: Optional[datetime] = None,
        category: Optional[BenchmarkCategory] = None,
    ) -> AgentResponse:
        trace: List[str] = []
        
        # 1. Identifica cliente
        cid = customer_id or TemporalIntentParser.extract_customer_id(query)
        if not cid:
            return AgentResponse(
                query=query,
                customer_id="",
                answer="Cliente não identificado na consulta.",
                confidence=0.0,
                abstained=True,
                reasoning_trace=["Falha na identificação do customer_id."],
            )

        customer = self.db.get_customer(cid)
        if not customer:
            return AgentResponse(
                query=query,
                customer_id=cid,
                answer=f"Cliente {cid} não encontrado no banco de dados.",
                confidence=0.0,
                abstained=True,
                reasoning_trace=[f"Cliente {cid} inexiste."],
            )

        # 2. Extrai datas da consulta e classifica intenção
        dates_in_query = TemporalIntentParser.extract_dates(query)
        intent = category or TemporalIntentParser.classify_intent(query)
        ref_date = as_of or (dates_in_query[0] if dates_in_query else datetime.now())

        trace.append(f"Intenção classificada: {intent.value}. Data de referência: {ref_date.strftime('%d/%m/%Y')}.")

        # 3. Verificação de abstenção temporal (pergunta anterior à criação do cliente)
        if ref_date.date() < customer.created_at.date():
            trace.append(
                f"Data consultada ({ref_date.strftime('%d/%m/%Y')}) é anterior à criação da conta ({customer.created_at.strftime('%d/%m/%Y')})."
            )
            return AgentResponse(
                query=query,
                customer_id=cid,
                as_of=ref_date,
                answer="ABSTAIN: O cliente ainda não existia no CRM na data consultada (conta criada em "
                + customer.created_at.strftime("%d/%m/%Y")
                + ").",
                confidence=1.0,
                abstained=True,
                reasoning_trace=trace,
            )

        all_cust_events = self.db.get_all_events(cid)

        # 4. Tratamento por intenção temporal
        # 4.1 AGGREGAÇÃO (ex: contagem de faturas ou tickets no intervalo)
        if intent == BenchmarkCategory.AGGREGATION:
            start_date = dates_in_query[0] if len(dates_in_query) >= 1 else (ref_date - timedelta(days=90))
            end_date = dates_in_query[1] if len(dates_in_query) >= 2 else ref_date
            if start_date > end_date:
                start_date, end_date = end_date, start_date

            target_type = EventType.INVOICE if "fatura" in query.lower() else EventType.TICKET
            matching_events = [
                e for e in all_cust_events
                if e.event_type == target_type and start_date <= e.valid_from <= end_date
            ]
            count = len(matching_events)
            item_name = "fatura(s)" if target_type == EventType.INVOICE else "chamado(s)"
            trace.append(f"Agregação executada no intervalo [{start_date.strftime('%d/%m/%Y')}, {end_date.strftime('%d/%m/%Y')}]: {count} encontrados.")

            return AgentResponse(
                query=query,
                customer_id=cid,
                as_of=ref_date,
                answer=f"Foram identificadas {count} {item_name} emitidas entre {start_date.strftime('%d/%m/%Y')} e {end_date.strftime('%d/%m/%Y')}.",
                confidence=0.98,
                cited_event_ids=[e.id for e in matching_events],
                retrieved_events=matching_events,
                abstained=False,
                reasoning_trace=trace,
            )

        # 4.2 DETECÇÃO DE MUDANÇA (ex: quando o plano mudou para Pro?)
        if intent == BenchmarkCategory.CHANGE_DETECTION:
            plan_matches = [
                e for e in all_cust_events
                if e.event_type == EventType.PLAN_CHANGE
            ]
            # Identifica qual plano está sendo perguntado
            target_plan = None
            for p in ["Free", "Starter", "Pro", "Enterprise"]:
                if p.lower() in query.lower():
                    target_plan = p
                    break

            if target_plan:
                matching_changes = [
                    e for e in plan_matches
                    if e.metadata.get("new_plan", "").lower() == target_plan.lower()
                ]
                if matching_changes:
                    change_event = matching_changes[0]
                    date_str = change_event.valid_from.strftime("%d/%m/%Y")
                    trace.append(f"Mudança para {target_plan} localizada no evento {change_event.id} em {date_str}.")
                    return AgentResponse(
                        query=query,
                        customer_id=cid,
                        as_of=ref_date,
                        answer=f"O cliente alterou o plano para {target_plan} na data {date_str}.",
                        confidence=0.95,
                        cited_event_ids=[change_event.id],
                        retrieved_events=[change_event],
                        abstained=False,
                        reasoning_trace=trace,
                    )

        # 4.3 ORDENAÇÃO TEMPORAL (ex: primeiro chamado após data X)
        if intent == BenchmarkCategory.TEMPORAL_ORDERING:
            cutoff = dates_in_query[0] if dates_in_query else ref_date
            subsequent = [
                e for e in all_cust_events
                if e.valid_from > cutoff and e.event_type == EventType.TICKET and "aberto" in e.metadata.get("status", "")
            ]
            if subsequent:
                subsequent.sort(key=lambda e: e.valid_from)
                first_evt = subsequent[0]
                trace.append(f"Primeiro evento após {cutoff.strftime('%d/%m/%Y')}: {first_evt.id}.")
                return AgentResponse(
                    query=query,
                    customer_id=cid,
                    as_of=ref_date,
                    answer=f"O primeiro chamado aberto após {cutoff.strftime('%d/%m/%Y')} foi '{first_evt.title}' em {first_evt.valid_from.strftime('%d/%m/%Y')}.",
                    confidence=0.95,
                    cited_event_ids=[first_evt.id],
                    retrieved_events=[first_evt],
                    abstained=False,
                    reasoning_trace=trace,
                )

        # 4.4 RESOLUÇÃO DE CONTRADIÇÃO / ENDEREÇO
        if intent == BenchmarkCategory.CONTRADICTION or "endereço" in query.lower() or "sede" in query.lower():
            address_updates = [
                e for e in all_cust_events
                if e.event_type == EventType.CONTACT_UPDATE
            ]
            if address_updates:
                upd = address_updates[0]
                if ref_date < upd.valid_from:
                    # Antes da mudança
                    addr = upd.metadata.get("old_address") or customer.address
                    trace.append(f"Consulta na data {ref_date.strftime('%d/%m/%Y')} anterior à mudança de endereço em {upd.valid_from.strftime('%d/%m/%Y')}.")
                    return AgentResponse(
                        query=query,
                        customer_id=cid,
                        as_of=ref_date,
                        answer=f"Na data {ref_date.strftime('%d/%m/%Y')}, o endereço corporativo era: {addr}.",
                        confidence=0.95,
                        cited_event_ids=[upd.id],
                        retrieved_events=[upd],
                        abstained=False,
                        reasoning_trace=trace,
                    )
                else:
                    addr = upd.metadata.get("new_address")
                    trace.append(f"Consulta na data posterior à alteração de endereço.")
                    return AgentResponse(
                        query=query,
                        customer_id=cid,
                        as_of=ref_date,
                        answer=f"O endereço atualizado da sede é: {addr}.",
                        confidence=0.95,
                        cited_event_ids=[upd.id],
                        retrieved_events=[upd],
                        abstained=False,
                        reasoning_trace=trace,
                    )

        # 4.5 AS_OF_STATE (padrão bitemporal)
        # Filtra estritamente eventos válidos na data solicitada e gravados até a data
        valid_events = [
            e for e in all_cust_events
            if e.is_valid_as_of(ref_date) and e.was_recorded_by(ref_date + timedelta(days=1))
        ]

        # Se for pergunta sobre plano
        if "plano" in query.lower():
            plan_evts = [e for e in valid_events if e.event_type == EventType.PLAN_CHANGE]
            if plan_evts:
                active_plan_evt = plan_evts[-1]
                plan_name = active_plan_evt.metadata.get("new_plan")
                trace.append(f"Plano ativo em {ref_date.strftime('%d/%m/%Y')}: {plan_name} (Evento {active_plan_evt.id}).")
                return AgentResponse(
                    query=query,
                    customer_id=cid,
                    as_of=ref_date,
                    answer=f"Na data {ref_date.strftime('%d/%m/%Y')}, o cliente estava no Plano {plan_name}.",
                    confidence=0.98,
                    cited_event_ids=[active_plan_evt.id],
                    retrieved_events=[active_plan_evt],
                    abstained=False,
                    reasoning_trace=trace,
                )

        # Caso geral com busca temporal filtrada
        results = self.retriever.retrieve(
            customer_id=cid,
            query=query,
            as_of_valid_time=ref_date,
            as_of_recorded_time=ref_date + timedelta(days=1),
            top_k=3,
        )

        if not results:
            return AgentResponse(
                query=query,
                customer_id=cid,
                as_of=ref_date,
                answer="Nenhuma informação temporalmente válida encontrada para esta data.",
                confidence=0.3,
                abstained=True,
                reasoning_trace=trace,
            )

        top_event = results[0].event
        trace.append(f"Evento com maior pontuação temporal: {top_event.id} ({top_event.title}).")
        
        return AgentResponse(
            query=query,
            customer_id=cid,
            as_of=ref_date,
            answer=f"Para a data {ref_date.strftime('%d/%m/%Y')}: {top_event.content}",
            confidence=0.9,
            cited_event_ids=[r.event.id for r in results],
            retrieved_events=[r.event for r in results],
            abstained=False,
            reasoning_trace=trace,
        )
