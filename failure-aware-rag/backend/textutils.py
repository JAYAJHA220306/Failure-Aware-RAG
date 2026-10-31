"""Small text helpers shared by retrieval, verification and evaluation."""
from __future__ import annotations

import re

STOPWORDS = set("""
a an the and or but if then else of to in on at by for with from as is are was were be been being
this that these those it its into than too very can could should would may might must will shall do does
did done have has had having not no nor so such what which who whom whose when where why how all any
both each few more most other some own same only just about above below between after before during
over under again further once here there also i you he she we they me him her us them my your our their
patient patients please tell explain describe give list according much many us
recommended recommend use used using take taken often long
""".split())

_TOKEN = re.compile(r"[a-z0-9]+(?:[./][0-9]+)?")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_CITE = re.compile(r"\s*\[(\d+(?:\s*,\s*\d+)*)\]")
_NUM = re.compile(r"\b\d+(?:[./]\d+)?\b")


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens, keeping numbers such as 0.5 or 130/80."""
    return _TOKEN.findall(text.lower())


def light_stem(token: str) -> str:
    """Very small suffix stripper so 'rechecked'/'recheck' and 'doses'/'dose' match."""
    if len(token) <= 4 or token[0].isdigit():
        return token
    for suffix, repl in (("ies", "y"), ("ing", ""), ("ed", ""), ("s", "")):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            if suffix == "s" and token.endswith(("ss", "us", "is")):
                return token
            return token[: -len(suffix)] + repl
    return token


def content_terms(text: str) -> list[str]:
    return [light_stem(t) for t in tokenize(text) if t not in STOPWORDS and len(t) > 1]


def numbers(text: str) -> set[str]:
    return set(_NUM.findall(text))


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [s.strip() for s in _SENT.split(text) if s.strip()]


def strip_citations(text: str) -> str:
    return _CITE.sub("", text).strip()


def cited_ids(text: str) -> list[int]:
    ids: list[int] = []
    for group in _CITE.findall(text):
        ids.extend(int(x) for x in re.split(r"\s*,\s*", group))
    return ids


def coverage(query_terms: list[str], text: str, weights: dict[str, float] | None = None) -> float:
    """Share of query content terms present in text (0-1), optionally IDF-weighted."""
    terms = set(query_terms)
    if not terms:
        return 0.0
    bag = set(content_terms(text))
    if not weights:
        return sum(1 for t in terms if t in bag) / len(terms)
    rare = max(weights.values(), default=1.0)
    w = {t: max(weights.get(t, rare), 0.05) for t in terms}
    return sum(w[t] for t in terms if t in bag) / sum(w.values())


_PARTS = re.compile(r"\?\s+|;\s+|,?\s+(?:and|also)\s+(?=(?:what|how|which|when|who|why|is|are|does|do|should|can)\b)", re.I)


def split_question(question: str) -> list[str]:
    """Rule-based split of a compositional question into its parts (no LLM call)."""
    parts = [p.strip(" ?") + "?" for p in _PARTS.split(question) if content_terms(p)]
    return parts if len(parts) > 1 else [question]
