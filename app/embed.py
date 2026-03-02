from __future__ import annotations

"""Embedding adapters and similarity utilities."""

import os
import warnings
from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class EmbeddingConfig:
    """Configurable model names for hosted and local embedding paths."""

    model: str = "text-embedding-3-small"
    local_model: str = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingEngine:
    """Small embedding wrapper.

    Why this exists: the app should run in two modes with identical interfaces:
    OpenAI embeddings when a key is present, and local sentence-transformer embeddings otherwise.
    TF-IDF remains as an explicit fallback for offline test/eval stability.
    """

    def __init__(
        self,
        corpus: List[str],
        force_local: bool = False,
        local_backend: str = "sentence-transformers",
    ) -> None:
        """Initialize embedding providers for runtime usage.

        Args:
            corpus: Text corpus used to fit TF-IDF when that fallback is selected.
            force_local: Forces local embedding path even if OpenAI key is available.
            local_backend: Preferred local backend (`sentence-transformers` or `tfidf`).

        """
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.use_openai = bool(self.api_key) and not force_local
        self.config = EmbeddingConfig(model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
        self.config.local_model = os.getenv("LOCAL_EMBED_MODEL", self.config.local_model)
        self.local_backend = local_backend
        self.vectorizer: TfidfVectorizer | None = None
        self.local_model = None
        self.client = None

        if self.use_openai:
            from openai import OpenAI

            self.client = OpenAI(api_key=self.api_key)
        else:
            if self.local_backend == "sentence-transformers":
                try:
                    from sentence_transformers import SentenceTransformer

                    # This is the default local embedding path for better semantic similarity.
                    self.local_model = SentenceTransformer(self.config.local_model)
                except Exception as exc:
                    warnings.warn(
                        f"Failed to load sentence-transformers model ({exc}). "
                        "Falling back to TF-IDF.",
                        stacklevel=2,
                    )
                    self.local_backend = "tfidf"

            if self.local_backend == "tfidf":
                self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
                self.vectorizer.fit(corpus)

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts into a dense matrix.

        Args:
            texts: Documents to embed.

        Returns:
            2D array shaped `(len(texts), embedding_dim)`.
        """
        if self.use_openai:
            # Single API call for a batch keeps behavior straightforward.
            resp = self.client.embeddings.create(model=self.config.model, input=texts)
            vectors = [item.embedding for item in resp.data]
            return np.array(vectors, dtype=np.float32)

        if self.local_backend == "sentence-transformers" and self.local_model is not None:
            vectors = self.local_model.encode(texts, normalize_embeddings=True)
            return np.array(vectors, dtype=np.float32)

        assert self.vectorizer is not None
        matrix = self.vectorizer.transform(texts).toarray()
        return matrix.astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query text into a 1D vector."""
        return self.embed_documents([text])[0]


def cosine_similarity(query_vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between one vector and each row in a matrix."""
    q_norm = np.linalg.norm(query_vector)
    m_norm = np.linalg.norm(matrix, axis=1)
    denominator = np.maximum(q_norm * m_norm, 1e-8)
    return (matrix @ query_vector) / denominator
