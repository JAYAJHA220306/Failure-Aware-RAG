"""Hybrid retrieval: BM25 (sparse) + optional dense embeddings, fused with Reciprocal Rank Fusion."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from rank_bm25 import BM25Okapi

from . import models
from .config import settings
from .ingest import Chunk, load_store
from .textutils import content_terms

RRF_K = 60


@dataclass
class Hit:
    chunk: Chunk
    score: float  # fused retrieval score (higher = better)
    relevance: float = 0.0  # 0-1 relevance to the question, filled by the verifier
    sources: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.chunk.id,
            "doc_id": self.chunk.doc_id,
            "title": self.chunk.title,
            "authority": self.chunk.authority,
            "text": self.chunk.text,
            "score": round(self.score, 4),
            "relevance": round(self.relevance, 3),
        }


class Retriever:
    def __init__(self, chunks: list[Chunk] | None = None):
        self.chunks = chunks if chunks is not None else load_store()
        self._by_id = {c.id: c for c in self.chunks}
        self._bm25 = BM25Okapi([content_terms(c.text + " " + c.title) or ["_"] for c in self.chunks]) if self.chunks else None
        self._emb: np.ndarray | None = None

    # ------------------------------------------------------------------ dense
    def _embeddings(self) -> np.ndarray | None:
        model = models.embedder()
        if model is None or not self.chunks:
            return None
        if self._emb is None:
            path = settings.index_dir / "embeddings.npy"
            if path.exists():
                cached = np.load(path)
                if cached.shape[0] == len(self.chunks):
                    self._emb = cached
            if self._emb is None:
                self._emb = model.encode([c.text for c in self.chunks], normalize_embeddings=True, batch_size=32)
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, self._emb)
        return self._emb

    def _dense_scores(self, query: str) -> np.ndarray | None:
        emb = self._embeddings()
        if emb is None:
            return None
        q = models.embedder().encode([query], normalize_embeddings=True)[0]
        return emb @ q

    # ----------------------------------------------------------------- search
    def search(self, query: str, k: int | None = None, method: str = "hybrid") -> list[Hit]:
        """Return the top-k chunks for a query. method: hybrid | bm25 | dense."""
        k = k or settings.candidate_k
        if not self.chunks:
            return []
        rankings: dict[str, np.ndarray] = {}
        if method in {"hybrid", "bm25"}:
            rankings["bm25"] = np.asarray(self._bm25.get_scores(content_terms(query) or ["_"]))
        if method in {"hybrid", "dense"}:
            dense = self._dense_scores(query)
            if dense is not None:
                rankings["dense"] = dense
        fused = np.zeros(len(self.chunks))
        for scores in rankings.values():
            order = np.argsort(-scores)
            ranks = np.empty_like(order)
            ranks[order] = np.arange(len(order))
            fused += 1.0 / (RRF_K + ranks + 1)
        top = np.argsort(-fused)[:k]
        return [
            Hit(self.chunks[i], float(fused[i]), sources={n: float(s[i]) for n, s in rankings.items()})
            for i in top
            if any(s[i] > 0 for s in rankings.values())
        ]

    def get(self, chunk_id: str) -> Chunk | None:
        return self._by_id.get(chunk_id)

    @property
    def idf(self) -> dict[str, float]:
        return dict(self._bm25.idf) if self._bm25 else {}

    @property
    def method(self) -> str:
        return "BM25 + dense (RRF)" if self._embeddings() is not None else "BM25"
