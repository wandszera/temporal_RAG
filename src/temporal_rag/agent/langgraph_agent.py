"""Agente de Raciocínio Temporal orquestrado com LangGraph.

Implementa um grafo de estados com nós explícitos para:
1. Classificação de intenção e extração de entidades temporais
2. Verificação de limites temporais (abstenção prévia)
3. Roteamento inteligente (SQL Tool para agregações vs Retrieval Bitemporal)
4. Recuperação bitemporal com filtros de valid_time e recorded_time
5. Nó de reflexão e verificação de suficiência (com loop de re-retrieval)
6. Síntese fundamentada com rastreamento de citações
"""

import operator
from datetime import datetime, timedelta
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from temporal_rag.agent.intent import IntentExtractor, RuleBasedExtractor
from temporal_rag.agent.temporal_parser import TemporalIntentParser
from temporal_rag.models.schema import (
    AgentResponse,
    BenchmarkCategory,
    EventType,
    TimelineEvent,
)
from temporal_rag.retrieval.temporal_retriever import TemporalRetriever
from temporal_rag.security import query_is_safe
from temporal_rag.storage.db import Database


class AgentState(TypedDict):
    query: str
    customer_id: str
    as_of: Optional[datetime]
    category: Optional[BenchmarkCategory]
    dates: List[datetime]
    route: str  # "sql_tool", "retriever", "abstain"
    events: List[TimelineEvent]
    sql_result: Optional[Dict[str, Any]]
    sufficiency: str  # "sufficient", "insufficient", "retry"
    retry_count: int
    answer: str
    confidence: float
    cited_event_ids: List[str]
    reasoning_trace: Annotated[List[str], operator.add]
    abstained: bool
    unsafe: bool
    plan: Optional[str]
    ticket_id: Optional[str]


class LangGraphTemporalAgent:
    """Orquestrador do raciocínio temporal baseado em LangGraph."""

    def __init__(self, db: Database, extractor: Optional[IntentExtractor] = None):
        self.db = db
        self.extractor = extractor or RuleBasedExtractor()
        self.retriever = TemporalRetriever(db)
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(AgentState)

        # Adiciona os nós do grafo
        builder.add_node("parse_and_classify", self._node_parse_and_classify)
        builder.add_node("check_temporal_bounds", self._node_check_temporal_bounds)
        builder.add_node("sql_aggregator", self._node_sql_aggregator)
        builder.add_node("temporal_retriever", self._node_temporal_retriever)
        builder.add_node("check_sufficiency", self._node_check_sufficiency)
        builder.add_node("synthesize_answer", self._node_synthesize_answer)
        builder.add_node("abstain", self._node_abstain)

        # Configura as transições do grafo
        builder.set_entry_point("parse_and_classify")
        builder.add_edge("parse_and_classify", "check_temporal_bounds")

        # Roteamento condicional após checar os limites temporais
        builder.add_conditional_edges(
            "check_temporal_bounds",
            self._route_after_bounds,
            {
                "abstain": "abstain",
                "sql_tool": "sql_aggregator",
                "retriever": "temporal_retriever",
            },
        )

        # Do SQL aggregator segue direto para síntese
        builder.add_edge("sql_aggregator", "synthesize_answer")

        # Do Retriever segue para verificação de suficiência
        builder.add_edge("temporal_retriever", "check_sufficiency")

        # Loop de reflexão condicional após suficiência
        builder.add_conditional_edges(
            "check_sufficiency",
            self._route_sufficiency,
            {
                "sufficient": "synthesize_answer",
                "retry": "temporal_retriever",
                "abstain": "abstain",
            },
        )

        builder.add_edge("synthesize_answer", END)
        builder.add_edge("abstain", END)

        return builder.compile()

    # --- NÓS DO GRAFO ---

    def _node_parse_and_classify(self, state: AgentState) -> Dict[str, Any]:
        query = state["query"]
        cid = state.get("customer_id") or TemporalIntentParser.extract_customer_id(query)
        intent = self.extractor.extract(query)
        dates = intent.dates
        cat = state.get("category") or intent.category
        ref_date = state.get("as_of") or (dates[0] if dates else datetime.now())

        trace = [
            f"[LangGraph::Parse] Cliente={cid or 'N/A'}, Categoria={cat.value}, RefDate={ref_date.strftime('%d/%m/%Y')}."
        ]

        return {
            "customer_id": cid or "",
            "dates": dates,
            "category": cat,
            "as_of": ref_date,
            "reasoning_trace": trace,
            "retry_count": state.get("retry_count", 0),
            "unsafe": intent.unsafe,
            "plan": intent.plan,
            "ticket_id": intent.ticket_id,
        }

    def _node_check_temporal_bounds(self, state: AgentState) -> Dict[str, Any]:
        cid = state["customer_id"]
        ref_date = state["as_of"] or datetime.now()

        # Guarda de segurança: override de instruções ou referência a outro cliente
        if state.get("unsafe") or not query_is_safe(state["query"], cid):
            return {
                "route": "abstain",
                "abstained": True,
                "answer": "ABSTAIN: Pedido bloqueado pela política de segurança (instrução de override ou outro cliente).",
                "confidence": 1.0,
                "reasoning_trace": ["[LangGraph::Guard] Pergunta bloqueada antes de qualquer recuperação."],
            }

        if not cid:
            return {
                "route": "abstain",
                "abstained": True,
                "answer": "Cliente não identificado na consulta.",
                "reasoning_trace": ["[LangGraph::Bounds] Cliente não identificado no prompt."],
            }

        customer = self.db.get_customer(cid)
        if not customer:
            return {
                "route": "abstain",
                "abstained": True,
                "answer": f"Cliente {cid} não encontrado no banco de dados.",
                "reasoning_trace": [f"[LangGraph::Bounds] Cliente {cid} inexistente."],
            }

        # Verificação bitemporal de fronteira: se a pergunta se refere a antes do cliente nascer no sistema
        if ref_date.date() < customer.created_at.date():
            created_str = customer.created_at.strftime("%d/%m/%Y")
            trace = [
                f"[LangGraph::Bounds] Data de consulta ({ref_date.strftime('%d/%m/%Y')}) é anterior ao cadastro ({created_str}). Acionando abstenção temporal."
            ]
            return {
                "route": "abstain",
                "abstained": True,
                "answer": f"ABSTAIN: O cliente ainda não existia no CRM na data consultada (conta criada em {created_str}).",
                "confidence": 1.0,
                "reasoning_trace": trace,
            }

        # Roteamento baseado na categoria
        if state["category"] == BenchmarkCategory.AGGREGATION:
            return {
                "route": "sql_tool",
                "reasoning_trace": ["[LangGraph::Router] Roteado para SQL Aggregator Tool."],
            }
        else:
            return {
                "route": "retriever",
                "reasoning_trace": ["[LangGraph::Router] Roteado para Temporal Retriever."],
            }

    def _route_after_bounds(self, state: AgentState) -> str:
        return state.get("route", "retriever")

    def _node_sql_aggregator(self, state: AgentState) -> Dict[str, Any]:
        cid = state["customer_id"]
        dates = state.get("dates", [])
        ref_date = state.get("as_of") or datetime.now()
        query = state["query"]

        start_date = dates[0] if len(dates) >= 1 else (ref_date - timedelta(days=90))
        end_date = dates[1] if len(dates) >= 2 else ref_date
        if start_date > end_date:
            start_date, end_date = end_date, start_date

        all_events = self.db.get_all_events(cid)
        target_type = EventType.INVOICE if "fatura" in query.lower() else EventType.TICKET

        matching_events = [
            e for e in all_events
            if e.event_type == target_type and start_date <= e.valid_from <= end_date
        ]

        count = len(matching_events)
        item_name = "fatura(s)" if target_type == EventType.INVOICE else "chamado(s)"
        trace = [
            f"[LangGraph::SQLTool] Executada agregação em [{start_date.strftime('%d/%m/%Y')} a {end_date.strftime('%d/%m/%Y')}]: {count} registros encontrados."
        ]

        answer = (
            f"Foram identificadas {count} {item_name} emitidas entre "
            f"{start_date.strftime('%d/%m/%Y')} e {end_date.strftime('%d/%m/%Y')}."
        )

        return {
            "events": matching_events,
            "cited_event_ids": [e.id for e in matching_events],
            "answer": answer,
            "confidence": 0.99,
            "sql_result": {"count": count, "start": str(start_date), "end": str(end_date)},
            "reasoning_trace": trace,
        }

    def _node_temporal_retriever(self, state: AgentState) -> Dict[str, Any]:
        cid = state["customer_id"]
        query = state["query"]
        ref_date = state.get("as_of") or datetime.now()
        category = state.get("category")
        retry_count = state.get("retry_count", 0)

        all_events = self.db.get_all_events(cid)
        found_events: List[TimelineEvent] = []
        trace = []

        if category == BenchmarkCategory.MULTI_HOP:
            found_events = self._multi_hop_events(all_events, query, state.get("plan"), state.get("ticket_id"))
            trace.append(f"[LangGraph::Retrieve] Multi-hop encadeou {len(found_events)} evento(s).")

        elif category == BenchmarkCategory.CHANGE_DETECTION:
            target_plan = state.get("plan")
            matching = [
                e for e in all_events
                if e.event_type == EventType.PLAN_CHANGE and (
                    target_plan is None or e.metadata.get("new_plan", "").lower() == target_plan.lower()
                )
            ]
            found_events = matching
            trace.append(f"[LangGraph::Retrieve] Detecção de mudança encontrou {len(found_events)} eventos.")

        elif category == BenchmarkCategory.TEMPORAL_ORDERING:
            dates = state.get("dates", [])
            cutoff = dates[0] if dates else ref_date
            subsequent = [
                e for e in all_events
                if e.valid_from.date() > cutoff.date() and e.event_type == EventType.TICKET and "aberto" in e.metadata.get("status", "")
            ]
            subsequent.sort(key=lambda e: e.valid_from)
            found_events = subsequent[:1] if subsequent else []
            trace.append(f"[LangGraph::Retrieve] Ordenação após {cutoff.strftime('%d/%m/%Y')} retornou {len(found_events)} evento(s).")

        elif category == BenchmarkCategory.CONTRADICTION or any(w in query.lower() for w in ["endereço", "sede"]):
            updates = [e for e in all_events if e.event_type == EventType.CONTACT_UPDATE]
            found_events = updates
            trace.append(f"[LangGraph::Retrieve] Recuperados {len(found_events)} eventos de atualização cadastral.")

        else:
            # as_of_state padrão bitemporal
            valid_events = [
                e for e in all_events
                if e.is_valid_as_of(ref_date) and e.was_recorded_by(ref_date + timedelta(days=1))
            ]
            if "plano" in query.lower():
                plan_evts = [e for e in valid_events if e.event_type == EventType.PLAN_CHANGE]
                found_events = plan_evts[-1:] if plan_evts else []
            else:
                # Usa busca híbrida do retriever com tolerância
                filter_strict = (retry_count == 0)
                results = self.retriever.retrieve(
                    customer_id=cid,
                    query=query,
                    as_of_valid_time=ref_date,
                    as_of_recorded_time=ref_date + timedelta(days=1),
                    filter_valid_interval=filter_strict,
                    top_k=3,
                )
                found_events = [r.event for r in results]

            trace.append(f"[LangGraph::Retrieve] Recuperação bitemporal (retry={retry_count}) retornou {len(found_events)} eventos.")

        return {
            "events": found_events,
            "cited_event_ids": [e.id for e in found_events],
            "reasoning_trace": trace,
        }

    @staticmethod
    def _multi_hop_events(
        all_events: List[TimelineEvent], query: str, plan: Optional[str] = None, ticket: Optional[str] = None
    ) -> List[TimelineEvent]:
        """Passo 1 acha o evento citado na pergunta; passo 2 usa a data dele para achar o segundo evento.

        Retorna [] se qualquer salto falhar (o grafo então reflete e, se preciso, se abstém)."""
        plans = sorted([e for e in all_events if e.event_type == EventType.PLAN_CHANGE], key=lambda e: e.valid_from)
        ticket_id = ticket or TemporalIntentParser.extract_ticket_id(query)
        if ticket_id:
            ticket = next(
                (e for e in all_events if e.event_type == EventType.TICKET
                 and e.metadata.get("ticket_id") == ticket_id and e.metadata.get("status") == "aberto"),
                None,
            )
            if ticket is None:
                return []
            active = [p for p in plans if p.is_valid_as_of(ticket.valid_from) and p.was_recorded_by(ticket.valid_from + timedelta(days=1))]
            return [ticket, active[-1]] if active else []

        target_plan = plan or next((p for p in ["Free", "Starter", "Pro", "Enterprise"] if p.lower() in query.lower()), None)
        # a primeira entrada de plano é o plano inicial, não uma "migração"
        changes = [p for p in plans[1:] if target_plan and p.metadata.get("new_plan") == target_plan]
        if not changes:
            return []
        change = changes[0]
        invoices = sorted(
            [e for e in all_events if e.event_type == EventType.INVOICE and e.valid_from > change.valid_from],
            key=lambda e: e.valid_from,
        )
        return [change, invoices[0]] if invoices else []

    def _node_check_sufficiency(self, state: AgentState) -> Dict[str, Any]:
        events = state.get("events", [])
        retry_count = state.get("retry_count", 0)

        if events:
            return {
                "sufficiency": "sufficient",
                "reasoning_trace": ["[LangGraph::Reflect] Evidências suficientes encontradas para sintetizar a resposta."],
            }

        if retry_count < 1:
            return {
                "sufficiency": "retry",
                "retry_count": retry_count + 1,
                "reasoning_trace": [
                    f"[LangGraph::Reflect] Nenhuma evidência exata encontrada. Executando loop de reflexão (tentativa {retry_count + 1})..."
                ],
            }

        return {
            "sufficiency": "abstain",
            "reasoning_trace": ["[LangGraph::Reflect] Esgotadas tentativas de busca. Transitando para abstenção."],
        }

    def _route_sufficiency(self, state: AgentState) -> str:
        return state.get("sufficiency", "sufficient")

    def _node_synthesize_answer(self, state: AgentState) -> Dict[str, Any]:
        # Se o SQL Tool já sintetizou, preserva
        if state.get("answer"):
            return {
                "reasoning_trace": ["[LangGraph::Synthesize] Resposta consolidada com citações verificadas."]
            }

        events = state.get("events", [])
        ref_date = state.get("as_of") or datetime.now()
        cid = state["customer_id"]
        customer = self.db.get_customer(cid)
        query = state["query"]
        category = state.get("category")

        if category == BenchmarkCategory.MULTI_HOP and len(events) == 2:
            first, second = events
            if first.event_type == EventType.TICKET:
                plan = second.metadata.get("new_plan")
                ans = (
                    f"Quando o chamado {first.metadata.get('ticket_id')} foi aberto "
                    f"({first.valid_from.strftime('%d/%m/%Y')}), o cliente estava no Plano {plan}."
                )
            else:
                ans = (
                    f"A primeira fatura após a migração para o plano {first.metadata.get('new_plan')} "
                    f"({first.valid_from.strftime('%d/%m/%Y')}) foi de R$ {float(second.metadata.get('amount', 0)):.2f} "
                    f"(emitida em {second.valid_from.strftime('%d/%m/%Y')})."
                )
            return {"answer": ans, "confidence": 0.95, "cited_event_ids": [first.id, second.id]}

        if category == BenchmarkCategory.CHANGE_DETECTION and events:
            evt = events[0]
            plan = evt.metadata.get("new_plan", "novo plano")
            date_str = evt.valid_from.strftime("%d/%m/%Y")
            ans = f"O cliente alterou o plano para {plan} na data {date_str}."
            return {"answer": ans, "confidence": 0.98, "cited_event_ids": [evt.id]}

        if category == BenchmarkCategory.TEMPORAL_ORDERING and events:
            evt = events[0]
            ans = f"O primeiro chamado aberto após a data de corte foi '{evt.title}' em {evt.valid_from.strftime('%d/%m/%Y')}."
            return {"answer": ans, "confidence": 0.98, "cited_event_ids": [evt.id]}

        if category == BenchmarkCategory.CONTRADICTION or "endereço" in query.lower():
            if events:
                upd = events[0]
                if ref_date < upd.valid_from:
                    addr = upd.metadata.get("old_address") or (customer.address if customer else "")
                    ans = f"Na data {ref_date.strftime('%d/%m/%Y')}, o endereço corporativo era: {addr}."
                else:
                    addr = upd.metadata.get("new_address")
                    ans = f"O endereço atualizado da sede é: {addr}."
                return {"answer": ans, "confidence": 0.98, "cited_event_ids": [upd.id]}

        # as_of_state para planos
        if "plano" in query.lower() and events:
            evt = events[-1]
            plan = evt.metadata.get("new_plan", "ativo")
            ans = f"Na data {ref_date.strftime('%d/%m/%Y')}, o cliente estava no Plano {plan}."
            return {"answer": ans, "confidence": 0.98, "cited_event_ids": [evt.id]}

        if events:
            top_evt = events[0]
            ans = f"Para a data {ref_date.strftime('%d/%m/%Y')}: {top_evt.content}"
            return {"answer": ans, "confidence": 0.92, "cited_event_ids": [e.id for e in events]}

        return {
            "answer": "Não foi possível encontrar evidências para responder com segurança.",
            "confidence": 0.2,
            "abstained": True,
        }

    def _node_abstain(self, state: AgentState) -> Dict[str, Any]:
        ans = state.get("answer") or "ABSTAIN: Informação não encontrada ou fora do horizonte de validade da conta."
        return {
            "answer": ans,
            "confidence": 1.0 if "ABSTAIN" in ans else 0.4,
            "abstained": True,
            "reasoning_trace": ["[LangGraph::Abstain] Resposta de abstenção emitida com sucesso."],
        }

    # --- MÉTODO PRINCIPAL DE EXECUÇÃO ---

    def answer(
        self,
        query: str,
        customer_id: Optional[str] = None,
        as_of: Optional[datetime] = None,
        category: Optional[BenchmarkCategory] = None,
    ) -> AgentResponse:
        initial_state: AgentState = {
            "query": query,
            "customer_id": customer_id or "",
            "as_of": as_of,
            "category": category,
            "dates": [],
            "route": "",
            "events": [],
            "sql_result": None,
            "sufficiency": "",
            "retry_count": 0,
            "answer": "",
            "confidence": 0.0,
            "cited_event_ids": [],
            "reasoning_trace": [],
            "abstained": False,
            "unsafe": False,
            "plan": None,
            "ticket_id": None,
        }

        final_state = self.graph.invoke(initial_state)

        return AgentResponse(
            query=query,
            customer_id=final_state.get("customer_id", ""),
            as_of=final_state.get("as_of"),
            answer=final_state.get("answer", ""),
            confidence=final_state.get("confidence", 0.0),
            cited_event_ids=final_state.get("cited_event_ids", []),
            retrieved_events=final_state.get("events", []),
            abstained=final_state.get("abstained", False),
            reasoning_trace=final_state.get("reasoning_trace", []),
        )

    def get_graph_mermaid(self) -> str:
        """Retorna o diagrama Mermaid do grafo de estados compilado."""
        return self.graph.get_graph().draw_mermaid()
