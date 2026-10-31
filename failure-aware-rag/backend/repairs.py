"""One repair action per failure type. Each repair edits the pipeline state and says what it did.

failure type          repair                                         source in the literature
missing_evidence  ->  HyDE query rewrite + wider re-retrieval       Wang et al. 2024; FLARE; DRAGIN
poor_ranking      ->  rerank the candidate pool by relevance         Wang et al. 2024; MEGA-RAG
noisy_context     ->  drop passages below the relevance threshold    Yoran et al. 2024; Chain-of-Note
position_effect   ->  move the best passages to the context edges   Liu et al. 2024 (lost in the middle)
unsupported       ->  remove unsupported sentences / strict regen   CoVe; FActScore; MEGA-RAG
multi_hop_gap     ->  decompose, retrieve per sub-question          Self-Ask; ReAct
conflicting       ->  keep the higher-authority source, flag it     RGB; MEGA-RAG
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .config import settings
from .detectors import Signals
from .diagnosis import Failure
from .generator import Answer
from .retrieval import Hit
from .textutils import split_sentences, strip_citations

if TYPE_CHECKING:
    from .pipeline import Pipeline


@dataclass
class State:
    question: str
    options: dict | None
    query: str
    candidates: list[Hit]
    context: list[Hit]
    answer: Answer | None = None
    signals: Signals | None = None
    notes: list[str] = field(default_factory=list)
    subquestions: list[str] | None = None
    excluded: set[str] = field(default_factory=set)  # passages set aside by a repair stay out


@dataclass
class RepairResult:
    name: str
    detail: str
    regenerate: bool = True
    strict: bool = False


def _dedupe(hits: list[Hit]) -> list[Hit]:
    seen, out = set(), []
    for h in hits:
        if h.chunk.id not in seen:
            seen.add(h.chunk.id)
            out.append(h)
    return out


def _by_relevance(hits: list[Hit]) -> list[Hit]:
    return sorted(hits, key=lambda h: (-h.relevance, -h.score))


def repair_missing(p: "Pipeline", st: State) -> RepairResult:
    new_query = p.generator.rewrite_query(st.question)
    fresh = p.retrieve(st.question, query=new_query, k=settings.candidate_k * 2)
    st.query = new_query
    st.candidates = _by_relevance(_dedupe(fresh + st.candidates))
    st.context = st.candidates[: settings.top_k]
    return RepairResult("Re-query (HyDE rewrite)", f"Searched again with a rewritten query; {len(fresh)} new candidates, kept the {len(st.context)} most relevant.")


def repair_ranking(p: "Pipeline", st: State) -> RepairResult:
    st.context = _by_relevance(st.candidates)[: settings.top_k]
    return RepairResult("Rerank candidates", f"Reranked {len(st.candidates)} candidates by relevance and kept the top {len(st.context)}.")


def repair_noisy(p: "Pipeline", st: State) -> RepairResult:
    keep = [h for h in st.context if h.relevance >= settings.passage_relevant]
    if not keep:
        keep = _by_relevance(st.context)[:1]
    removed = len(st.context) - len(keep)
    st.context = keep
    return RepairResult("Filter irrelevant passages", f"Removed {removed} passage(s) below relevance {settings.passage_relevant}.")


def repair_position(p: "Pipeline", st: State) -> RepairResult:
    ranked = _by_relevance(st.context)
    if len(ranked) >= 3:
        st.context = [ranked[0]] + ranked[2:] + [ranked[1]]
    else:
        st.context = ranked
    return RepairResult("Reorder to context edges", "Placed the most relevant passage first and the second most relevant last.")


def repair_unsupported(p: "Pipeline", st: State) -> RepairResult:
    sig, ans = st.signals, st.answer
    if ans and sig and sig.claims and not st.options:
        supported = {c.text for c in sig.claims if c.supported}
        kept = [s for s in split_sentences(ans.text) if strip_citations(s) in supported]
        if kept:
            dropped = len(split_sentences(ans.text)) - len(kept)
            st.answer = Answer(" ".join(kept), choice=ans.choice)
            return RepairResult("Remove unsupported claims", f"Dropped {dropped} unsupported sentence(s); kept {len(kept)} verified sentence(s).", regenerate=False)
    return RepairResult("Strict regeneration", "Regenerated the answer, allowing only facts stated explicitly in the passages.", strict=True)


def repair_multihop(p: "Pipeline", st: State) -> RepairResult:
    subs = p.generator.decompose(st.question)
    per_sub = max(2, settings.top_k // max(1, len(subs)))
    lists = []
    for sq in subs:
        hits = p.retrieve(sq, query=f"{sq} {st.question}", k=settings.candidate_k, score_against=sq)
        lists.append(_by_relevance(hits)[:per_sub])
    # Interleave so every sub-question's best passage sits near the front of the context.
    merged = [lst[i] for i in range(per_sub) for lst in lists if i < len(lst)]
    p.score(st.question, merged)
    st.context = _dedupe(merged + _by_relevance(st.context))[: settings.top_k + 2]
    st.candidates = _dedupe(st.candidates + merged)
    st.subquestions = subs
    return RepairResult("Decompose and retrieve", f"Split into {len(subs)} sub-question(s): " + " | ".join(subs))


def repair_conflict(p: "Pipeline", st: State) -> RepairResult:
    sig = st.signals
    if not sig or not sig.conflict_pair:
        return repair_noisy(p, st)
    a, b = (st.context[i - 1] for i in sig.conflict_pair)
    loser = min((a, b), key=lambda h: (h.chunk.authority_rank, h.relevance))
    winner = b if loser is a else a
    st.excluded.add(loser.chunk.id)
    st.context = [h for h in st.context if h.chunk.id != loser.chunk.id]
    note = f"Sources disagreed; kept '{winner.chunk.title}' ({winner.chunk.authority}) over '{loser.chunk.title}' ({loser.chunk.authority})."
    st.notes.append(note)
    return RepairResult("Resolve conflict by authority", note)


REPAIRS = {
    Failure.MISSING: repair_missing,
    Failure.RANKING: repair_ranking,
    Failure.NOISY: repair_noisy,
    Failure.POSITION: repair_position,
    Failure.UNSUPPORTED: repair_unsupported,
    Failure.MULTIHOP: repair_multihop,
    Failure.CONFLICT: repair_conflict,
}
