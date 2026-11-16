"""Embeddings plugáveis para a recuperação densa.

Nenhum modelo é baixado ou chamado por padrão. Duas formas de ativar:
- ``sentence-transformers[:modelo]``: carrega o modelo localmente (exige ``pip install sentence-transformers``
  e acesso ao Hugging Face na primeira execução).
- ``pacote.modulo:funcao``: função sua que recebe ``List[str]`` e devolve uma matriz (n, d).
"""

import importlib
from typing import Callable, Dict, List, Sequence

import numpy as np

DEFAULT_ST_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def _normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)


class Embedder:
    """Envolve uma função de embedding, normaliza os vetores e guarda cache por texto."""

    def __init__(self, encode: Callable[[List[str]], np.ndarray], name: str = "custom"):
        self.name = name
        self._encode = encode
        self._cache: Dict[str, np.ndarray] = {}

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            vectors = _normalize(self._encode(missing))
            if len(vectors) != len(missing):
                raise ValueError("a função de embedding devolveu um número de vetores diferente do de textos")
            self._cache.update(zip(missing, vectors))
        return np.vstack([self._cache[t] for t in texts]) if texts else np.zeros((0, 0))


def _sentence_transformers(model_name: str) -> Embedder:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - depende do ambiente do usuário
        raise RuntimeError("Instale com: pip install sentence-transformers") from exc
    model = SentenceTransformer(model_name)
    return Embedder(lambda texts: model.encode(texts, batch_size=64, show_progress_bar=False), f"st:{model_name}")


def build_embedder(spec: str) -> Embedder:
    """``sentence-transformers[:modelo]`` ou ``pacote.modulo:funcao``."""
    kind, _, rest = spec.partition(":")
    if kind == "sentence-transformers":
        return _sentence_transformers(rest or DEFAULT_ST_MODEL)
    if not kind or not rest:
        raise ValueError("Use 'sentence-transformers[:modelo]' ou 'pacote.modulo:funcao'")
    func = getattr(importlib.import_module(kind), rest)
    return Embedder(func, name=spec)
