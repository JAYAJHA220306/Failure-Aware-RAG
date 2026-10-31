"""Lazy loaders for the optional neural models (embedder, reranker, NLI).

All three need `pip install -r requirements-ml.txt`. If they are missing, or a model
cannot be downloaded, the pipeline falls back to lexical methods and keeps working.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from .config import ml_available, settings

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def embedder():
    if not (settings.use_dense and ml_available()):
        return None
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(settings.embed_model)
    except Exception as exc:  # network blocked, model missing, etc.
        log.warning("Dense retrieval disabled: %s", exc)
        return None


@lru_cache(maxsize=1)
def reranker():
    if not (settings.use_reranker and ml_available()):
        return None
    try:
        from sentence_transformers import CrossEncoder

        return CrossEncoder(settings.rerank_model)
    except Exception as exc:
        log.warning("Reranker disabled: %s", exc)
        return None


@lru_cache(maxsize=1)
def nli():
    if settings.verifier not in {"auto", "nli"} or not ml_available():
        return None
    try:
        from sentence_transformers import CrossEncoder

        return CrossEncoder(settings.nli_model)
    except Exception as exc:
        log.warning("NLI verifier disabled: %s", exc)
        return None


def status() -> dict:
    return {
        "ml_installed": ml_available(),
        "dense": embedder() is not None,
        "reranker": reranker() is not None,
        "nli": nli() is not None,
    }
