"""Configurações globais do projeto Temporal RAG."""

import os
from pathlib import Path
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

class Settings(BaseModel):
    # Database config
    # Fallback default is high-speed SQLite in data/crm_temporal.db
    # If DATABASE_URL is set, uses PostgreSQL (e.g. postgresql://user:pass@localhost:5432/temporal_crm)
    database_url: str = os.getenv(
        "DATABASE_URL", f"sqlite:///{DATA_DIR / 'crm_temporal.db'}"
    )
    
    # Synthetic generator config
    seed: int = int(os.getenv("SEED", "42"))
    default_customers_count: int = int(os.getenv("CUSTOMERS_COUNT", "250"))
    
    # Benchmark config
    benchmark_file: Path = DATA_DIR / "benchmark_dataset.json"
    evaluation_report_file: Path = DATA_DIR / "evaluation_report.json"
    
    # Retrieval config
    default_top_k: int = 5
    recency_decay_lambda: float = 0.005  # Decay factor per day of difference

settings = Settings()
