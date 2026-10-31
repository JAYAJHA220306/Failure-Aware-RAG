"""Failure detectors: turn one (question, retrieved context, answer) into measurable signals.

Retrieval-side signals: top relevance, share of relevant passages, better passages left
below the cut-off, relevant passage stuck mid-context, conflicting passages.
Generation-side signals: claim-level support for each answer sentence, abstention.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from .config import settings
from .generator import Answer
from .retrieval import Hit
from .textutils import content_terms, coverage, split_question, split_sentences, strip_citations
from .verifier import Verifier, safe_mean

COMPOSITIONAL = re.compile(r"\b(both|compare|comparison|versus|vs\.?|difference between)\b|\?.+\?|\b(and|also)\s+(what|how|which|when|who|is|are|does|do|should)\b", re.I)


@dataclass
class Claim:
    text: str
    support: float
    contradiction: float
    passage: int | None  # 1-based index of the best-supporting passage
    supported: bool


@dataclass
class Signals:
    top_relevance: float = 0.0
    relevant_fraction: float = 0.0
    better_below_cutoff: float = 0.0
    best_passage_position: int | None = None
    position_flag: bool = False
    conflict: float = 0.0
    conflict_pair: tuple[int, int] | None = None
    compositional: bool = False
    uncovered_terms: list[str] = field(default_factory=list)
    answer_coverage: float = 1.0
    claims: list[Claim] = field(default_factory=list)
    unsupported_fraction: float = 0.0
    abstained: bool = False
    passed: bool = False
    failed_checks: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 3)
        return d


def _claims_from(answer: Answer, options: dict | None) -> list[str]:
    claims = [strip_citations(s) for s in split_sentences(answer.text)]
    if options and answer.choice:
        claims = [options[answer.choice]] + [c for c in claims if not c.lower().startswith("option ")]
    return [c for c in claims if len(content_terms(c)) >= 2]


def detect(
    verifier: Verifier,
    question: str,
    context: list[Hit],
    candidates: list[Hit],
    answer: Answer,
    options: dict | None = None,
) -> Signals:
    s = Signals(abstained=answer.abstained)
    rel = [h.relevance for h in context]
    s.top_relevance = max(rel, default=0.0)
    s.relevant_fraction = (sum(r >= settings.passage_relevant for r in rel) / len(rel)) if rel else 0.0

    # Ranking: is a clearly more relevant passage sitting below the top-k cut-off?
    in_ctx = {h.chunk.id for h in context}
    below = [h.relevance for h in candidates if h.chunk.id not in in_ctx]
    s.better_below_cutoff = max(0.0, max(below, default=0.0) - s.top_relevance)

    # Position: the most relevant passage is in the middle of a long context.
    if rel:
        best = max(range(len(rel)), key=lambda i: rel[i])
        s.best_passage_position = best + 1
        s.position_flag = len(rel) >= 5 and 2 <= best < len(rel) - 1 and rel[best] >= settings.passage_relevant

    # Conflict between relevant passages from different documents.
    relevant = [(i, h) for i, h in enumerate(context) if h.relevance >= settings.passage_relevant][:6]
    for a in range(len(relevant)):
        for b in range(a + 1, len(relevant)):
            (ia, ha), (ib, hb) = relevant[a], relevant[b]
            if ha.chunk.doc_id == hb.chunk.doc_id:
                continue
            score = verifier.conflict(ha.chunk.text, hb.chunk.text)
            if score > s.conflict:
                s.conflict, s.conflict_pair = score, (ia + 1, ib + 1)

    # Multi-hop: compositional question with parts no passage covers.
    s.compositional = bool(COMPOSITIONAL.search(question))
    covered = set()
    for h in context:
        covered |= set(content_terms(h.chunk.text))
    s.uncovered_terms = [t for t in dict.fromkeys(content_terms(question)) if t not in covered]

    # Claim-level support.
    passages = [h.chunk.text for h in context]
    if not answer.abstained:
        for text in _claims_from(answer, options):
            scores = verifier.support(text, passages)
            if not scores:
                continue
            best_i = max(range(len(scores)), key=lambda i: scores[i]["entail"])
            sup = scores[best_i]["entail"]
            con = scores[best_i]["contradict"]  # contradiction by the passage that best supports it
            s.claims.append(Claim(text, sup, con, best_i + 1, sup >= settings.claim_supported and con < settings.conflict_threshold))
        if s.claims:
            s.unsupported_fraction = sum(not c.supported for c in s.claims) / len(s.claims)

    # Completeness: does the answer address every part of the question?
    if not answer.abstained:
        answer_text = answer.text + (" " + options[answer.choice] if options and answer.choice else "")
        parts = split_question(question)
        s.answer_coverage = min(coverage(content_terms(part), answer_text, verifier.idf) for part in parts)

    # Verdict.
    checks = {
        "abstained": s.abstained,
        "low_relevance": s.top_relevance < settings.min_top_relevance,
        "unsupported_claims": s.unsupported_fraction > settings.max_unsupported_fraction,
        "conflict": s.conflict >= settings.conflict_threshold,
        "no_claims": not s.abstained and not s.claims,
        "incomplete_answer": not s.abstained and not options and s.compositional and s.answer_coverage < settings.min_answer_coverage,
    }
    s.failed_checks = [k for k, v in checks.items() if v]
    s.passed = not s.failed_checks
    s.confidence = 0.0 if s.abstained else s.top_relevance * safe_mean([c.support for c in s.claims] or [0.0])
    return s
