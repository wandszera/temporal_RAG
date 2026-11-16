"""Módulo de vetorização e cálculo de similaridade semântica."""

import numpy as np
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TextVectorizer:
    """Vetorizador leve e determinístico baseado em TF-IDF com sublinear tf e n-grams."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode",
            lowercase=True,
        )
        self.is_fitted = False

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0))
        self.is_fitted = True
        return self.vectorizer.fit_transform(texts).toarray()

    def transform(self, texts: List[str]) -> np.ndarray:
        if not self.is_fitted or not texts:
            return np.empty((0, 0))
        return self.vectorizer.transform(texts).toarray()

    def compute_similarity(self, query: str, candidate_texts: List[str]) -> np.ndarray:
        if not candidate_texts:
            return np.array([])
        
        # Fit temporário no conjunto de candidatos + query para máxima precisão local
        all_texts = candidate_texts + [query]
        vecs = self.vectorizer.fit_transform(all_texts).toarray()
        
        cand_vecs = vecs[:-1]
        q_vec = vecs[-1:]
        
        sims = cosine_similarity(q_vec, cand_vecs)[0]
        return sims
