"""Servidor FastAPI e UI Interativa para Temporal RAG com orquestração LangGraph."""

from datetime import datetime
import os
import secrets
from collections import defaultdict
from time import perf_counter
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from temporal_rag.agent.agent import BaselineAgent, TemporalAgent
from temporal_rag.benchmark.dataset_generator import BenchmarkDatasetGenerator
from temporal_rag.eval.evaluator import BenchmarkEvaluator
from temporal_rag.storage.db import Database
from temporal_rag.security import public_response, query_is_safe

app = FastAPI(title="Temporal RAG para CRM", version="0.1.0")

_latencies = defaultdict(list)


@app.middleware("http")
async def api_controls(request: Request, call_next):
    start = perf_counter()
    key = os.getenv("TEMPORAL_RAG_API_KEY")
    if key and request.url.path.startswith("/api/"):
        supplied = request.headers.get("x-api-key", "")
        if not secrets.compare_digest(supplied, key):
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "Não autorizado"}, status_code=401)
    response = await call_next(request)
    if request.url.path.startswith("/api/"):
        _latencies[request.url.path].append((perf_counter() - start) * 1000)
        _latencies[request.url.path] = _latencies[request.url.path][-1000:]
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db = Database()
baseline_agent = BaselineAgent(db)
temporal_agent = TemporalAgent(db)


class QueryRequest(BaseModel):
    customer_id: str
    query: str
    as_of: Optional[str] = None
    mode: str = "both"  # "temporal", "naive", "both"


@app.get("/api/customers")
def get_customers(limit: int = 50):
    return db.list_customers(limit=limit)


@app.get("/api/customers/{customer_id}")
def get_customer(customer_id: str):
    customer = db.get_customer(customer_id)
    if not customer:
        raise HTTPException(status_code=404, detail="Cliente não encontrado")
    return customer


@app.get("/api/customers/{customer_id}/events")
def get_customer_events(customer_id: str):
    events = db.get_all_events(customer_id)
    return events


@app.get("/api/agent/graph")
def get_agent_graph():
    """Retorna o diagrama Mermaid e detalhes dos nós do agente LangGraph."""
    return {
        "mermaid": temporal_agent.get_graph_mermaid(),
        "nodes": [
            {"id": "parse_and_classify", "title": "1. Parse & Classify", "desc": "Extrai cliente, datas e intenção temporal da pergunta."},
            {"id": "check_temporal_bounds", "title": "2. Temporal Bounds", "desc": "Valida se a data consultada antecede a criação da conta (aciona abstenção prévia)."},
            {"id": "sql_aggregator", "title": "3. SQL Aggregator Tool", "desc": "Executa contagens analíticas em janelas fechadas (faturas/tickets)."},
            {"id": "temporal_retriever", "title": "4. Temporal Retriever", "desc": "Aplica filtros bitemporais (valid_from <= T < valid_to e recorded_at <= T)."},
            {"id": "check_sufficiency", "title": "5. Sufficiency & Reflection Loop", "desc": "Verifica suficiência de evidências e aciona re-retrieval se necessário."},
            {"id": "synthesize_answer", "title": "6. Synthesize Answer", "desc": "Gera a resposta final estritamente fundamentada com citações dos IDs de eventos."},
            {"id": "abstain", "title": "7. Abstain Node", "desc": "Emite resposta formal de abstenção (ABSTAIN) quando faltam fatos válidos."},
        ]
    }


@app.post("/api/query")
def execute_query(req: QueryRequest):
    if not query_is_safe(req.query, req.customer_id):
        raise HTTPException(status_code=400, detail="Consulta bloqueada: instrução adversarial ou outro cliente")
    if req.mode not in ("temporal", "naive", "both"):
        raise HTTPException(status_code=422, detail="Modo inválido")
    as_of_dt = None
    if req.as_of:
        try:
            as_of_dt = datetime.fromisoformat(req.as_of)
        except Exception:
            pass

    response_data = {}

    if req.mode in ("temporal", "both"):
        temp_resp = temporal_agent.answer(req.query, customer_id=req.customer_id, as_of=as_of_dt)
        response_data["temporal"] = public_response(temp_resp)

    if req.mode in ("naive", "both"):
        naive_resp = baseline_agent.answer(req.query, customer_id=req.customer_id, as_of=as_of_dt)
        response_data["naive"] = public_response(naive_resp)

    return response_data


@app.get("/api/metrics")
def get_metrics():
    """Latência em memória por rota; sem perguntas ou dados pessoais."""
    result = {}
    for route, values in _latencies.items():
        ordered = sorted(values)
        if ordered:
            result[route] = {
                "count": len(ordered),
                "p50_ms": round(ordered[int((len(ordered) - 1) * 0.50)], 2),
                "p95_ms": round(ordered[int((len(ordered) - 1) * 0.95)], 2),
            }
    return result


@app.post("/api/benchmark/run")
def run_benchmark(limit: int = 50):
    samples = BenchmarkDatasetGenerator.load_dataset()
    if limit and limit < len(samples):
        samples = samples[:limit]
    evaluator = BenchmarkEvaluator(db)
    report = evaluator.run_benchmark(samples)
    return report


@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Temporal RAG para CRM - Dashboard com LangGraph</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <script>
        mermaid.initialize({ startOnLoad: false, theme: 'dark' });
    </script>
    <style>
        body { background-color: #0f172a; color: #e2e8f0; font-family: system-ui, -apple-system, sans-serif; }
        .card { background-color: #1e293b; border-color: #334155; color: #e2e8f0; }
        .form-control, .form-select { background-color: #0f172a; border-color: #334155; color: #f8fafc; }
        .form-control:focus, .form-select:focus { background-color: #0f172a; color: #fff; border-color: #38bdf8; box-shadow: 0 0 0 0.25rem rgba(56, 189, 248, 0.25); }
        .timeline-badge { width: 12px; height: 12px; border-radius: 50%; display: inline-block; margin-right: 8px; }
        .badge-plan { background-color: #38bdf8; }
        .badge-ticket { background-color: #f59e0b; }
        .badge-invoice { background-color: #10b981; }
        .badge-contact { background-color: #ec4899; }
        .badge-nps { background-color: #8b5cf6; }
        .comparison-box { border-left: 4px solid; padding: 12px; border-radius: 4px; }
        .comparison-naive { border-color: #ef4444; background: rgba(239, 68, 68, 0.08); }
        .comparison-temporal { border-color: #10b981; background: rgba(16, 185, 129, 0.08); }
        .code-pill { background: #334155; padding: 2px 6px; border-radius: 4px; font-family: monospace; font-size: 0.85em; }
        .trace-step { padding: 4px 8px; border-radius: 4px; margin-bottom: 4px; font-family: monospace; font-size: 0.82em; background: rgba(56, 189, 248, 0.1); border-left: 3px solid #38bdf8; }
    </style>
</head>
<body class="p-4">
    <div class="container-fluid">
        <header class="d-flex justify-content-between align-items-center mb-4 pb-3 border-bottom border-secondary">
            <div>
                <h2 class="fw-bold text-info mb-1"><i class="bi bi-diagram-3-fill me-2"></i>Temporal RAG com LangGraph</h2>
                <p class="text-secondary mb-0">Raciocínio Bitemporal, Agente com Grafo de Estados e Benchmark de Avaliação</p>
            </div>
            <div class="d-flex align-items-center gap-2">
                <button class="btn btn-outline-warning" onclick="showGraphModal()"><i class="bi bi-bezier2 me-1"></i>Ver Grafo LangGraph</button>
                <button class="btn btn-outline-info" onclick="loadBenchmark()"><i class="bi bi-speedometer2 me-1"></i>Rodar Benchmark</button>
                <span class="badge bg-success fs-6"><i class="bi bi-check2-circle me-1"></i>LangGraph Ativo</span>
            </div>
        </header>

        <div class="row g-4">
            <!-- Coluna da Esquerda: Seletor de Cliente e Consulta -->
            <div class="col-lg-5">
                <div class="card p-3 mb-4">
                    <h5 class="card-title text-info mb-3"><i class="bi bi-search me-2"></i>Consulta Temporal Interativa</h5>
                    
                    <div class="mb-3">
                        <label class="form-label">Cliente Selecionado:</label>
                        <select id="customerSelect" class="form-select" onchange="loadCustomerTimeline()"></select>
                    </div>

                    <div id="customerDetails" class="p-2 mb-3 rounded bg-dark border border-secondary" style="font-size: 0.9em;">
                        Carregando informações do cliente...
                    </div>

                    <div class="mb-3">
                        <label class="form-label">Data de Corte ("As-Of" Date):</label>
                        <input type="date" id="asOfInput" class="form-control" value="2024-03-15">
                        <small class="text-secondary">Simula a resposta histórica exata nesta data.</small>
                    </div>

                    <div class="mb-3">
                        <label class="form-label">Pergunta ao Assistente:</label>
                        <input type="text" id="queryInput" class="form-control" value="Qual era o meu plano contratado?">
                        <div class="mt-2 d-flex flex-wrap gap-1">
                            <button class="btn btn-sm btn-outline-secondary" onclick="setQuery('Qual era o meu plano contratado?')">Qual era o plano?</button>
                            <button class="btn btn-sm btn-outline-secondary" onclick="setQuery('Em qual data mudei de plano?')">Quando mudei?</button>
                            <button class="btn btn-sm btn-outline-secondary" onclick="setQuery('Qual era o endereço da minha sede?')">Endereço na data</button>
                            <button class="btn btn-sm btn-outline-secondary" onclick="setQuery('Quantas faturas foram emitidas entre 01/01/2023 e 31/12/2023?')">Contar faturas</button>
                        </div>
                    </div>

                    <button class="btn btn-primary w-100 py-2 fw-semibold" onclick="submitQuery()"><i class="bi bi-play-fill me-1"></i>Executar no Grafo LangGraph</button>
                </div>

                <!-- Comparação Lado a Lado -->
                <div class="card p-3" id="resultsCard" style="display: none;">
                    <h5 class="card-title text-warning mb-3"><i class="bi bi-columns-gap me-2"></i>Resultado Comparativo</h5>

                    <!-- Naive RAG -->
                    <div class="comparison-box comparison-naive mb-3">
                        <div class="d-flex justify-content-between align-items-center mb-1">
                            <span class="text-danger fw-bold"><i class="bi bi-x-circle me-1"></i>Naive RAG (Baseline)</span>
                            <span class="badge bg-danger">Sem Filtro Temporal</span>
                        </div>
                        <p id="naiveAnswer" class="mb-2">-</p>
                        <small class="text-secondary">Fontes citadas: <span id="naiveSources" class="code-pill">-</span></small>
                    </div>

                    <!-- Temporal RAG com LangGraph -->
                    <div class="comparison-box comparison-temporal">
                        <div class="d-flex justify-content-between align-items-center mb-1">
                            <span class="text-success fw-bold"><i class="bi bi-cpu me-1"></i>Temporal RAG (LangGraph)</span>
                            <div>
                                <span class="badge bg-primary me-1">StateGraph</span>
                                <span class="badge bg-success" id="tempConfidence">100%</span>
                            </div>
                        </div>
                        <p id="tempAnswer" class="mb-2 fw-semibold text-white">-</p>
                        <div class="mb-2"><small class="text-secondary">Fontes citadas: <span id="tempSources" class="code-pill">-</span></small></div>
                        
                        <div class="mt-2">
                            <span class="small fw-semibold text-info mb-1 d-block"><i class="bi bi-activity me-1"></i>Trilha de Execução dos Nós:</span>
                            <div id="tempTrace" style="max-height: 180px; overflow-y: auto;"></div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Coluna da Direita: Timeline de Eventos Bitemporais -->
            <div class="col-lg-7">
                <div class="card p-3">
                    <div class="d-flex justify-content-between align-items-center mb-3">
                        <h5 class="card-title text-info mb-0"><i class="bi bi-clock-history me-2"></i>Linha do Tempo Bitemporal do Cliente</h5>
                        <div class="d-flex gap-2" style="font-size: 0.8em;">
                            <span><span class="timeline-badge badge-plan"></span>Plano</span>
                            <span><span class="timeline-badge badge-invoice"></span>Fatura</span>
                            <span><span class="timeline-badge badge-ticket"></span>Ticket</span>
                            <span><span class="timeline-badge badge-contact"></span>Contato</span>
                        </div>
                    </div>
                    <div id="timelineContainer" style="max-height: 700px; overflow-y: auto;">
                        <p class="text-secondary">Selecione um cliente para carregar seus eventos...</p>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Modal do Grafo LangGraph -->
    <div class="modal fade" id="graphModal" tabindex="-1">
        <div class="modal-dialog modal-xl">
            <div class="modal-content bg-dark text-white border-secondary">
                <div class="modal-header border-secondary">
                    <h5 class="modal-title"><i class="bi bi-bezier2 me-2"></i>Grafo de Estados do Agente (LangGraph)</h5>
                    <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    <p class="text-secondary small mb-3">
                        O grafo abaixo orquestra o ciclo de raciocínio com nós determinísticos, roteamento para SQL Tool, verificação de fronteiras temporais e loop de reflexão.
                    </p>
                    <div id="mermaidGraphContainer" class="p-3 bg-black rounded border border-secondary text-center mb-3">
                        <div class="spinner-border text-info" role="status"></div>
                    </div>
                    <div id="nodesDescription" class="row g-2"></div>
                </div>
            </div>
        </div>
    </div>

    <!-- Modal de Benchmark -->
    <div class="modal fade" id="benchModal" tabindex="-1">
        <div class="modal-dialog modal-lg">
            <div class="modal-content bg-dark text-white border-secondary">
                <div class="modal-header border-secondary">
                    <h5 class="modal-title"><i class="bi bi-speedometer2 me-2"></i>Relatório do Benchmark Temporal</h5>
                    <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body" id="benchModalBody">
                    <div class="text-center py-4">
                        <div class="spinner-border text-info" role="status"></div>
                        <p class="mt-2">Executando harness de avaliação em background...</p>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        let currentCustomers = [];

        async function init() {
            try {
                const res = await fetch('/api/customers?limit=30');
                currentCustomers = await res.json();
                const sel = document.getElementById('customerSelect');
                sel.innerHTML = currentCustomers.map(c => 
                    `<option value="${c.id}">${c.id} - ${c.name} (${c.company})</option>`
                ).join('');
                if (currentCustomers.length > 0) {
                    loadCustomerTimeline();
                }
            } catch (err) {
                console.error("Erro ao carregar clientes:", err);
            }
        }

        async function loadCustomerTimeline() {
            const cid = document.getElementById('customerSelect').value;
            const cust = currentCustomers.find(c => c.id === cid);
            if (cust) {
                document.getElementById('customerDetails').innerHTML = `
                    <div><strong>Empresa:</strong> ${cust.company} | <strong>Email:</strong> ${cust.email}</div>
                    <div><strong>Plano Inicial:</strong> <span class="badge bg-secondary">${cust.initial_plan}</span> | <strong>Plano Atual:</strong> <span class="badge bg-info">${cust.current_plan || 'N/A'}</span></div>
                    <div><strong>Endereço Original:</strong> ${cust.address}</div>
                `;
            }

            try {
                const res = await fetch(`/api/customers/${cid}/events`);
                const events = await res.json();
                renderTimeline(events);
            } catch (err) {
                console.error("Erro ao carregar timeline:", err);
            }
        }

        function renderTimeline(events) {
            const container = document.getElementById('timelineContainer');
            if (!events || events.length === 0) {
                container.innerHTML = '<p class="text-secondary">Nenhum evento registrado.</p>';
                return;
            }

            const getBadgeClass = (type) => {
                if (type === 'plan_change') return 'badge-plan';
                if (type === 'invoice') return 'badge-invoice';
                if (type === 'ticket') return 'badge-ticket';
                if (type === 'contact_update') return 'badge-contact';
                return 'badge-nps';
            };

            container.innerHTML = events.map(e => `
                <div class="card mb-2 p-2 border-secondary bg-dark" style="border-left: 3px solid #38bdf8 !important;">
                    <div class="d-flex justify-content-between align-items-center mb-1">
                        <span class="fw-semibold">
                            <span class="timeline-badge ${getBadgeClass(e.event_type)}"></span>
                            ${e.title}
                        </span>
                        <span class="code-pill">${e.id}</span>
                    </div>
                    <div class="text-light small mb-2">${e.content}</div>
                    <div class="d-flex justify-content-between text-secondary" style="font-size: 0.75em;">
                        <span><i class="bi bi-calendar-check me-1"></i><strong>valid_from:</strong> ${new Date(e.valid_from).toLocaleString('pt-BR')} ${e.valid_to ? ' até ' + new Date(e.valid_to).toLocaleString('pt-BR') : '(ativo)'}</span>
                        <span><i class="bi bi-clock me-1"></i><strong>recorded_at:</strong> ${new Date(e.recorded_at).toLocaleTimeString('pt-BR')}</span>
                    </div>
                </div>
            `).join('');
        }

        function setQuery(text) {
            document.getElementById('queryInput').value = text;
        }

        async function submitQuery() {
            const cid = document.getElementById('customerSelect').value;
            const query = document.getElementById('queryInput').value;
            const asOf = document.getElementById('asOfInput').value;

            document.getElementById('resultsCard').style.display = 'block';
            document.getElementById('naiveAnswer').innerText = 'Consultando...';
            document.getElementById('tempAnswer').innerText = 'Consultando...';

            try {
                const res = await fetch('/api/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        customer_id: cid,
                        query: query,
                        as_of: asOf ? asOf + "T00:00:00" : null,
                        mode: 'both'
                    })
                });
                const data = await res.json();

                // Atualiza Naive
                if (data.naive) {
                    document.getElementById('naiveAnswer').innerText = data.naive.answer;
                    document.getElementById('naiveSources').innerText = (data.naive.cited_event_ids || []).join(', ') || 'nenhum';
                }

                // Atualiza Temporal
                if (data.temporal) {
                    document.getElementById('tempAnswer').innerText = data.temporal.answer;
                    document.getElementById('tempSources').innerText = (data.temporal.cited_event_ids || []).join(', ') || 'nenhum';
                    document.getElementById('tempConfidence').innerText = Math.round(data.temporal.confidence * 100) + '%';
                    const traceHtml = (data.temporal.reasoning_trace || []).map(t => `<div class="trace-step">${t}</div>`).join('');
                    document.getElementById('tempTrace').innerHTML = traceHtml;
                }
            } catch (err) {
                alert('Erro ao executar consulta: ' + err);
            }
        }

        async function showGraphModal() {
            const modal = new bootstrap.Modal(document.getElementById('graphModal'));
            modal.show();
            try {
                const res = await fetch('/api/agent/graph');
                const data = await res.json();
                
                const container = document.getElementById('mermaidGraphContainer');
                container.innerHTML = `<div class="mermaid">${data.mermaid}</div>`;
                await mermaid.run({ nodes: container.querySelectorAll('.mermaid') });

                const nodesDesc = document.getElementById('nodesDescription');
                nodesDesc.innerHTML = data.nodes.map(n => `
                    <div class="col-md-6">
                        <div class="card p-2 bg-dark border-secondary h-100">
                            <span class="text-info fw-bold small">${n.title}</span>
                            <span class="text-secondary small">${n.desc}</span>
                        </div>
                    </div>
                `).join('');
            } catch (err) {
                console.error("Erro ao carregar grafo LangGraph:", err);
            }
        }

        async function loadBenchmark() {
            const modal = new bootstrap.Modal(document.getElementById('benchModal'));
            modal.show();
            document.getElementById('benchModalBody').innerHTML = `
                <div class="text-center py-4">
                    <div class="spinner-border text-info" role="status"></div>
                    <p class="mt-2">Executando harness de avaliação em 50 amostras do benchmark...</p>
                </div>
            `;

            try {
                const res = await fetch('/api/benchmark/run?limit=50', { method: 'POST' });
                const data = await res.json();
                const n = data.results.naive_rag;
                const t = data.results.temporal_rag;

                document.getElementById('benchModalBody').innerHTML = `
                    <div class="table-responsive">
                        <table class="table table-dark table-bordered table-striped">
                            <thead>
                                <tr class="text-info">
                                    <th>Métrica</th>
                                    <th>Naive RAG (Baseline)</th>
                                    <th>Temporal RAG (LangGraph)</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td><strong>Acurácia Geral</strong></td>
                                    <td class="text-danger fw-bold">${n.overall_accuracy}%</td>
                                    <td class="text-success fw-bold">${t.overall_accuracy}%</td>
                                </tr>
                                <tr>
                                    <td>Recall de Citações</td>
                                    <td>${n.citation_recall}%</td>
                                    <td class="text-success">${t.citation_recall}%</td>
                                </tr>
                                <tr>
                                    <td>Precisão de Citações</td>
                                    <td>${n.citation_precision}%</td>
                                    <td class="text-success">${t.citation_precision}%</td>
                                </tr>
                                <tr>
                                    <td>Latência Média</td>
                                    <td>${n.avg_latency_ms} ms</td>
                                    <td>${t.avg_latency_ms} ms</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                    <div class="alert alert-info mt-3 small mb-0">
                        <i class="bi bi-info-circle me-1"></i>
                        O Temporal RAG com LangGraph supera o Naive RAG ao filtrar fatos invalidados temporalmente e orquestrar reflexão com rastreamento estrito de eventos.
                    </div>
                `;
            } catch (err) {
                document.getElementById('benchModalBody').innerHTML = `<div class="alert alert-danger">Erro ao rodar benchmark: ${err}</div>`;
            }
        }

        window.onload = init;
    </script>
</body>
</html>
    """
