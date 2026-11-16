"""CLI para Temporal RAG - Geração de dados, avaliação e consultas."""

import argparse
import sys
from datetime import datetime, timedelta

# Garante suporte a UTF-8 em terminais Windows (cp1252)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from temporal_rag.agent.agent import BaselineAgent, TemporalAgent
from temporal_rag.agent.intent import build_extractor
from temporal_rag.benchmark.dataset_generator import BenchmarkDatasetGenerator
from temporal_rag.config import settings
from temporal_rag.eval.evaluator import BenchmarkEvaluator
from temporal_rag.generator.synthetic import SyntheticDataGenerator
from temporal_rag.storage.db import Database


def cmd_seed(args):
    print(f"🌱 Gerando dados sintéticos (seed={args.seed}, clientes={args.customers})...")
    db = Database()
    db.clear_all()

    generator = SyntheticDataGenerator(seed=args.seed)
    customers, events = generator.generate_customers_and_timelines(count=args.customers)

    print(f"💾 Salvando {len(customers)} clientes e {len(events)} eventos bitemporais no banco...")
    for cust in customers:
        db.save_customer(cust)
    db.save_events(events)

    print("🎯 Gerando dataset de benchmark com ground truth temporal...")
    bench_gen = BenchmarkDatasetGenerator(customers, events)
    samples = bench_gen.generate_full_benchmark(limit_per_category=args.benchmark_per_cat)
    bench_gen.save_dataset(samples)

    from temporal_rag.benchmark.paraphrases import build_paraphrases
    para = build_paraphrases(samples)
    bench_gen.save_dataset(para, settings.paraphrase_file)
    print(f"🗣️  {len(para)} paráfrases salvas em: {settings.paraphrase_file}")
    print(f"✅ Concluído! {len(samples)} perguntas geradas em: {settings.benchmark_file}")
    print(f"📊 Banco de dados pronto em: {settings.database_url}")


def cmd_eval(args):
    db = Database()
    print("📖 Carregando dataset de benchmark...")
    try:
        samples = BenchmarkDatasetGenerator.load_dataset()
    except Exception as e:
        print(f"❌ Erro ao carregar benchmark: {e}. Execute primeiro: make seed ou python -m temporal_rag.cli seed")
        sys.exit(1)

    if args.split != "all":
        samples = [s for s in samples if s.split == args.split]
    if args.split in ("test", "all"):
        print("⚠️  Split de teste: use só na avaliação final, depois de congelar as decisões no split dev.")
    if args.limit and args.limit < len(samples):
        samples = samples[:args.limit]

    evaluator = BenchmarkEvaluator(db, extractor=build_extractor(args.extractor))
    evaluator.run_benchmark(samples, blind=args.blind)


def cmd_eval_paraphrase(args):
    import json
    from tabulate import tabulate
    from temporal_rag.benchmark.paraphrases import HELDOUT_VARIANTS, TUNING_VARIANTS, variant_of

    db = Database()
    try:
        samples = BenchmarkDatasetGenerator.load_dataset(settings.paraphrase_file)
    except Exception as e:
        print(f"❌ Erro ao carregar paráfrases: {e}. Execute primeiro: make seed")
        sys.exit(1)
    if args.split != "all":
        samples = [s for s in samples if s.split == args.split]
    if args.split in ("test", "all"):
        print("⚠️  Split de teste: use só na avaliação final.")
    evaluator = BenchmarkEvaluator(db, extractor=build_extractor(args.extractor))
    report = {"split": args.split, "groups": {}}
    rows = []
    for label, variants in (("ajuste (v0-v3)", TUNING_VARIANTS), ("verificação (v4-v5)", HELDOUT_VARIANTS)):
        group = [s for s in samples if variant_of(s) in variants]
        metrics = {
            name: evaluator.evaluate_model(name, agent, group, blind=True)
            for name, agent in (("naive", evaluator.baseline_agent), ("temporal", evaluator.temporal_agent))
        }
        report["groups"][label] = {k: {"n": v["total_questions"], "strict_accuracy": v["strict_accuracy"],
                                       "by_category": {c: d["strict_accuracy"] for c, d in v["category_breakdown"].items()}}
                                   for k, v in metrics.items()}
        rows.append([label, len(group), f"{metrics['naive']['strict_accuracy']}%", f"{metrics['temporal']['strict_accuracy']}%"])
    print(tabulate(rows, headers=["Grupo", "Perguntas", "Naive (estrita)", "Temporal (estrita)"], tablefmt="github"))
    cats = sorted(report["groups"]["ajuste (v0-v3)"]["temporal"]["by_category"])
    print()
    print(tabulate([[c] + [report["groups"][g]["temporal"]["by_category"].get(c, "-") for g in report["groups"]] for c in cats],
                   headers=["Categoria (Temporal, estrita %)"] + list(report["groups"]), tablefmt="github"))
    with open(settings.paraphrase_report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def cmd_ablate(args):
    from temporal_rag.eval.ablation import print_ablation, run_ablation

    db = Database()
    try:
        samples = BenchmarkDatasetGenerator.load_dataset()
    except Exception as e:
        print(f"❌ Erro ao carregar benchmark: {e}. Execute primeiro: make seed")
        sys.exit(1)
    if args.split in ("test", "all"):
        print("⚠️  Split de teste: use só na avaliação final, depois de congelar as decisões no split dev.")
    embedder = None
    if args.embedder:
        from temporal_rag.retrieval.embedders import build_embedder

        embedder = build_embedder(args.embedder)
    report = run_ablation(db, samples, split=args.split, embedder=embedder)
    print_ablation(report)
    print(f"\n💾 Relatório salvo em: {settings.ablation_report_file}")


def cmd_compare(args):
    db = Database()
    customer_id = args.customer
    query = args.query

    # Se nenhum cliente/query for passado, seleciona um caso emblemático do banco
    if not customer_id or not query:
        customers = db.list_customers(limit=10)
        if not customers:
            print("❌ Banco de dados vazio. Execute primeiro: make seed")
            sys.exit(1)
        cust = customers[0]
        customer_id = cust.id
        events = db.get_all_events(customer_id)
        plan_events = [e for e in events if e.event_type.value == "plan_change"]
        if len(plan_events) >= 2:
            first_plan = plan_events[0]
            # Data no passado onde o primeiro plano estava ativo
            query_date = first_plan.valid_from + timedelta(days=15)
            query = f"Qual era o meu plano em {query_date.strftime('%d/%m/%Y')}?"
        else:
            query = "Qual é o meu plano contratado?"

    print("\n" + "=" * 70)
    print("🔬 COMPARAÇÃO LADO A LADO: NAIVE RAG vs TEMPORAL RAG")
    print("=" * 70)
    print(f"👤 Cliente: {customer_id}")
    print(f"❓ Pergunta: {query}")
    print("-" * 70)

    # 1. Resposta Naive RAG
    naive_agent = BaselineAgent(db)
    naive_resp = naive_agent.answer(query, customer_id=customer_id)
    print("\n🔴 [NAIVE RAG (Baseline Sem Filtro Temporal)]")
    print(f"  Resposta:   {naive_resp.answer}")
    print(f"  Eventos:    {naive_resp.cited_event_ids}")
    print("  Diagnóstico: Frequentemente traz o plano mais recente ou com mais repetições,")
    print("               ignorando que na data perguntada o cliente possuía outro plano!")

    # 2. Resposta Temporal RAG
    temp_agent = TemporalAgent(db)
    temp_resp = temp_agent.answer(query, customer_id=customer_id)
    print("\n🟢 [TEMPORAL RAG (Bitemporal + Validade Temporal)]")
    print(f"  Resposta:   {temp_resp.answer}")
    print(f"  Eventos:    {temp_resp.cited_event_ids}")
    print(f"  Confiança:  {temp_resp.confidence * 100:.1f}%")
    print("  Raciocínio: " + " -> ".join(temp_resp.reasoning_trace))
    print("=" * 70 + "\n")


def cmd_ask(args):
    db = Database()
    agent = TemporalAgent(db)

    as_of_dt = None
    if args.as_of:
        try:
            as_of_dt = datetime.fromisoformat(args.as_of)
        except Exception:
            pass

    response = agent.answer(args.query, customer_id=args.customer, as_of=as_of_dt)
    print(f"\nResposta: {response.answer}")
    print(f"Confiança: {response.confidence * 100:.1f}%")
    print(f"Eventos Citados: {response.cited_event_ids}")
    if response.reasoning_trace:
        print("Trilha:")
        for step in response.reasoning_trace:
            print(f" - {step}")


def cmd_serve(args):
    import uvicorn
    print(f"🚀 Iniciando servidor web em http://{args.host}:{args.port}")
    uvicorn.run("temporal_rag.server.app:app", host=args.host, port=args.port, reload=False)


def main():
    parser = argparse.ArgumentParser(description="Temporal RAG para CRM - CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Seed
    p_seed = subparsers.add_parser("seed", help="Gera dados sintéticos e benchmark")
    p_seed.add_argument("--customers", type=int, default=250, help="Quantidade de clientes")
    p_seed.add_argument("--seed", type=int, default=42, help="Seed pseudoaleatória")
    p_seed.add_argument("--benchmark-per-cat", type=int, default=40, help="Perguntas por categoria de teste")
    p_seed.set_defaults(func=cmd_seed)

    # Eval
    p_eval = subparsers.add_parser("eval", help="Roda o benchmark e gera métricas")
    p_eval.add_argument("--limit", type=int, default=None, help="Limite de perguntas a avaliar")
    p_eval.add_argument("--split", choices=["dev", "test", "all"], default="dev", help="Conjunto avaliado (padrão: dev)")
    p_eval.add_argument("--extractor", choices=["rules", "llm"], default="rules", help="Extrator de intenção (llm exige TEMPORAL_RAG_LLM)")
    p_eval.add_argument("--blind", action="store_true", help="Não entrega categoria/data do gabarito ao agente")
    p_eval.set_defaults(func=cmd_eval)

    # Eval paraphrase
    p_para = subparsers.add_parser("eval-paraphrase", help="Avalia os agentes (modo cego) em perguntas parafraseadas")
    p_para.add_argument("--split", choices=["dev", "test", "all"], default="dev", help="Conjunto avaliado (padrão: dev)")
    p_para.add_argument("--extractor", choices=["rules", "llm"], default="rules", help="Extrator de intenção (llm exige TEMPORAL_RAG_LLM)")
    p_para.set_defaults(func=cmd_eval_paraphrase)

    # Ablate
    p_ablate = subparsers.add_parser("ablate", help="Ablação de recuperação (recall@k, MRR, nDCG)")
    p_ablate.add_argument("--split", choices=["dev", "test", "all"], default="dev", help="Conjunto avaliado (padrão: dev)")
    p_ablate.add_argument(
        "--embedder",
        default=None,
        help="Ativa variantes densas: 'sentence-transformers[:modelo]' ou 'pacote.modulo:funcao' (padrão: desligado)",
    )
    p_ablate.set_defaults(func=cmd_ablate)

    # Compare
    p_compare = subparsers.add_parser("compare", help="Compara Naive vs Temporal RAG lado a lado")
    p_compare.add_argument("--customer", type=str, default=None, help="ID do cliente")
    p_compare.add_argument("--query", type=str, default=None, help="Texto da pergunta")
    p_compare.set_defaults(func=cmd_compare)

    # Ask
    p_ask = subparsers.add_parser("ask", help="Faz uma pergunta ao Temporal RAG")
    p_ask.add_argument("--customer", type=str, required=True, help="ID do cliente (ex: CUST_0001)")
    p_ask.add_argument("--query", type=str, required=True, help="Pergunta em linguagem natural")
    p_ask.add_argument("--as-of", type=str, default=None, help="Data de referência ISO (ex: 2024-03-15)")
    p_ask.set_defaults(func=cmd_ask)

    # Serve
    p_serve = subparsers.add_parser("serve", help="Inicia o servidor web FastAPI e UI")
    p_serve.add_argument("--host", type=str, default="127.0.0.1", help="Host")
    p_serve.add_argument("--port", type=int, default=8000, help="Porta")
    p_serve.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
