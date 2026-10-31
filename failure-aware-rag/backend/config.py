"""Central settings. Every value can be overridden in a .env file or environment variable."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env reader (KEY=value lines) so no extra dependency is needed."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")


def _env(key: str, default):
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    if isinstance(default, bool):
        return raw.lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


@dataclass
class Settings:
    # --- LLM -------------------------------------------------------------
    # "mock"   = no API key needed; extractive answers (for testing the pipeline and UI)
    # "openai" = any OpenAI-compatible endpoint: OpenAI, Groq, Together, Ollama, LM Studio
    llm_provider: str = _env("LLM_PROVIDER", "mock")
    llm_base_url: str = _env("LLM_BASE_URL", "https://api.openai.com/v1")
    llm_api_key: str = _env("LLM_API_KEY", "")
    llm_model: str = _env("LLM_MODEL", "gpt-4o-mini")
    llm_temperature: float = _env("LLM_TEMPERATURE", 0.0)
    llm_timeout_s: int = _env("LLM_TIMEOUT_S", 120)

    # --- Optional neural models (need requirements-ml.txt) ----------------
    use_dense: bool = _env("USE_DENSE", True)
    use_reranker: bool = _env("USE_RERANKER", True)
    verifier: str = _env("VERIFIER", "auto")  # auto | nli | lexical | llm
    embed_model: str = _env("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    rerank_model: str = _env("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    nli_model: str = _env("NLI_MODEL", "cross-encoder/nli-deberta-v3-small")

    # --- Chunking and retrieval ------------------------------------------
    chunk_words: int = _env("CHUNK_WORDS", 100)
    chunk_overlap: int = _env("CHUNK_OVERLAP", 30)
    top_k: int = _env("TOP_K", 6)
    candidate_k: int = _env("CANDIDATE_K", 30)

    # --- Failure detection thresholds (tune on the dev split) -------------
    min_top_relevance: float = _env("MIN_TOP_RELEVANCE", 0.45)
    passage_relevant: float = _env("PASSAGE_RELEVANT", 0.35)
    min_relevant_fraction: float = _env("MIN_RELEVANT_FRACTION", 0.5)
    claim_supported: float = _env("CLAIM_SUPPORTED", 0.6)
    max_unsupported_fraction: float = _env("MAX_UNSUPPORTED_FRACTION", 0.0)
    min_answer_coverage: float = _env("MIN_ANSWER_COVERAGE", 0.6)
    conflict_threshold: float = _env("CONFLICT_THRESHOLD", 0.7)
    ranking_margin: float = _env("RANKING_MARGIN", 0.1)
    max_repairs: int = _env("MAX_REPAIRS", 2)
    random_seed: int = _env("RANDOM_SEED", 13)

    # --- Paths ------------------------------------------------------------
    corpus_dir: Path = field(default_factory=lambda: ROOT / _env("CORPUS_DIR", "data/corpus"))
    index_dir: Path = field(default_factory=lambda: ROOT / _env("INDEX_DIR", "data/index"))
    eval_dir: Path = field(default_factory=lambda: ROOT / "data/eval")
    results_dir: Path = field(default_factory=lambda: ROOT / "results")


settings = Settings()


def ml_available() -> bool:
    """True when sentence-transformers (and torch) are installed."""
    try:
        import sentence_transformers  # noqa: F401
        return True
    except Exception:
        return False
