.PHONY: help install seed eval eval-para ablate test run compare clean

help:
	@echo "Comandos disponíveis:"
	@echo "  make seed      - Gera dados sintéticos bitemporais (clientes e eventos de CRM)"
	@echo "  make eval      - Executa o benchmark e gera tabela de ablação (Naive vs Temporal RAG)"
	@echo "  make ablate    - Ablação de recuperação (recall@k, MRR, nDCG) no split dev"
	@echo "  make test      - Executa suite de testes unitários com pytest"
	@echo "  make run       - Inicia a API FastAPI e UI interativa em http://localhost:8000"
	@echo "  make compare   - Executa comparação demonstrativa no terminal"

install:
	pip install -e .

seed:
	python -m temporal_rag.cli seed --customers 250

eval:
	python -m temporal_rag.cli eval

eval-para:
	python -m temporal_rag.cli eval-paraphrase

ablate:
	python -m temporal_rag.cli ablate

test:
	pytest -v tests/

run:
	python -m temporal_rag.cli serve

compare:
	python -m temporal_rag.cli compare

clean:
	python -c "import shutil, os; [shutil.rmtree(p) for p in ['data', '.pytest_cache'] if os.path.exists(p)]"
