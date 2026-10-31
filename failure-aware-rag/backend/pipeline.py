"""The failure-aware RAG pipeline and every baseline / ablation from the experiment plan.

    retrieve -> generate -> detect --pass--> answer
                              |
                              fail -> diagnose -> matched repair -> (re)generate -> detect ...
                              budget spent -> abstain

Modes (same retriever, corpus, LLM and top-k for all):
    llm_only        B0  no retrieval
    vanilla         B1  single-pass RAG
    rerank          B2  hybrid retrieval + reranker
    crag_lite       B4* simplified CRAG: corrects retrieval only, triggered by retrieval score
    failure_aware   Ours
    detect_abstain  A1  detect, then abstain (no repair)
    generic         A2  any failure -> re-retrieve (no diagnosis)
    random          A3  random repair, same budget
    oracle          A4  true failure label given (evaluation only)
    no_abstain      A5  repairs, but never abstains
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable

from .config import settings
from .detectors import Signals, detect
from .diagnosis import LABELS, Diagnosis, Failure, diagnose
from .generator import ABSTAIN, Answer, Generator
from .llm import LLM
from .repairs import REPAIRS, State, repair_missing, repair_noisy
from .retrieval import Hit, Retriever
from .verifier import Verifier

MODES = {
    "failure_aware": "Failure-aware (ours)",
    "vanilla": "Vanilla RAG (B1)",
    "rerank": "Hybrid + rerank (B2)",
    "crag_lite": "CRAG-lite (B4, simplified)",
    "llm_only": "LLM only (B0)",
    "detect_abstain": "Detect + abstain (A1)",
    "generic": "Generic re-retrieve (A2)",
    "random": "Random repair (A3)",
    "oracle": "Oracle diagnosis (A4)",
    "no_abstain": "No abstention (A5)",
}
REPAIRING = {"failure_aware", "generic", "random", "oracle", "no_abstain"}


@dataclass
class Perturbation:
    """Hooks used by the evaluation to inject a known failure (see evaluation/inject.py)."""
    label: Failure = Failure.NONE
    on_candidates: Callable[[list[Hit]], list[Hit]] | None = None
    on_context: Callable[[list[Hit]], list[Hit]] | None = None


@dataclass
class Trace:
    steps: list[dict] = field(default_factory=list)
    _t0: float = field(default_factory=time.perf_counter)

    def add(self, kind: str, title: str, detail: str = "", **data) -> None:
        self.steps.append({"kind": kind, "title": title, "detail": detail, "t_ms": round((time.perf_counter() - self._t0) * 1000), **data})


class Pipeline:
    def __init__(self, retriever: Retriever | None = None, llm: LLM | None = None):
        self.retriever = retriever or Retriever()
        self.llm = llm or LLM()
        self.generator = Generator(self.llm)
        self.verifier = Verifier(self.llm)
        self.verifier.idf = self.retriever.idf
        self.generator.idf = self.retriever.idf
        self.rng = random.Random(settings.random_seed)

    # ---------------------------------------------------------------- helpers
    def score(self, question: str, hits: list[Hit]) -> list[Hit]:
        rel = self.verifier.relevance(question, [h.chunk.text for h in hits])
        for h, r in zip(hits, rel):
            h.relevance = float(r)
        return hits

    def retrieve(self, question: str, query: str | None = None, k: int | None = None, score_against: str | None = None) -> list[Hit]:
        hits = self.retriever.search(query or question, k or settings.candidate_k)
        return self.score(score_against or question, hits)

    def _generate(self, st: State, strict: bool = False) -> Answer:
        return self.generator.answer(st.question, st.context, st.options, strict=strict, subquestions=st.subquestions)

    def _detect(self, st: State) -> Signals:
        return detect(self.verifier, st.question, st.context, st.candidates, st.answer, st.options)

    # -------------------------------------------------------------------- run
    def run(self, question: str, mode: str = "failure_aware", options: dict | None = None, perturb: Perturbation | None = None) -> dict:
        if mode not in MODES:
            raise ValueError(f"Unknown mode '{mode}'. Choose from: {', '.join(MODES)}")
        self.llm.reset_counters()
        trace = Trace()
        perturb = perturb or Perturbation()
        diagnoses: list[Diagnosis] = []
        repairs: list[str] = []

        if mode == "llm_only":
            ans = self.generator.closed_book(question, options)
            trace.add("generate", "Closed-book answer", "No retrieval (baseline B0).")
            st = State(question, options, question, [], [], ans)
            st.signals = Signals(abstained=ans.abstained, passed=not ans.abstained)
            return self._result(st, mode, trace, diagnoses, repairs, first=ans)

        # 1. Retrieve
        candidates = self.retrieve(question)
        if perturb.on_candidates:
            candidates = perturb.on_candidates(candidates)
        if mode in {"rerank"}:
            context = sorted(candidates, key=lambda h: -h.relevance)[: settings.top_k]
        else:
            context = candidates[: settings.top_k]
        if perturb.on_context:
            context = perturb.on_context(context)
        st = State(question, options, question, candidates, context)
        trace.add("retrieve", "Retrieve", f"{self.retriever.method}: {len(candidates)} candidates, top {len(context)} sent to the LLM.",
                  passages=[h.to_dict() for h in context])

        # CRAG-lite: correct retrieval before generating, based on retrieval score alone.
        if mode == "crag_lite":
            top = max((h.relevance for h in st.context), default=0)
            if top < settings.min_top_relevance:
                r = repair_missing(self, st)
                repairs.append(r.name)
                trace.add("repair", r.name, f"Retrieval score {top:.2f} judged incorrect. {r.detail}")
            r = repair_noisy(self, st)
            trace.add("repair", "Refine passages", r.detail)

        # 2. Generate + 3. Detect
        st.answer = self._generate(st)
        first = st.answer
        trace.add("generate", "Generate", "Abstained: no answer in passages." if st.answer.abstained else "Draft answer written.", answer=st.answer.text)
        st.signals = self._detect(st)
        trace.add("detect", "Checks passed" if st.signals.passed else "Checks failed",
                  ", ".join(st.signals.failed_checks) or "Relevance, claim support and conflict checks passed.", signals=st.signals.to_dict())

        if mode in {"vanilla", "rerank", "crag_lite"}:
            return self._result(st, mode, trace, diagnoses, repairs, first=first, enforce=False)

        if mode == "detect_abstain":
            if not st.signals.passed:
                d = diagnose(st.signals)
                diagnoses.append(d)
                trace.add("diagnose", LABELS[d.failure], d.reason, failure=d.failure.value)
            return self._result(st, mode, trace, diagnoses, repairs, first=first)

        # 4. Diagnose -> repair loop
        budget = settings.max_repairs
        while not st.signals.passed and budget > 0:
            budget -= 1
            d = diagnose(st.signals)
            if mode == "oracle" and perturb.label != Failure.NONE and not repairs:
                d = Diagnosis(perturb.label, "Oracle: true injected failure type.")
            diagnoses.append(d)
            trace.add("diagnose", LABELS[d.failure], d.reason, failure=d.failure.value)
            if mode == "generic":
                fn = repair_missing
            elif mode == "random":
                fn = self.rng.choice(list(REPAIRS.values()))
            else:
                fn = REPAIRS.get(d.failure, repair_missing)
            r = fn(self, st)
            if st.excluded:
                st.context = [h for h in st.context if h.chunk.id not in st.excluded]
                st.candidates = [h for h in st.candidates if h.chunk.id not in st.excluded]
            repairs.append(r.name)
            trace.add("repair", r.name, r.detail, passages=[h.to_dict() for h in st.context])
            if r.regenerate:
                st.answer = self._generate(st, strict=r.strict)
                trace.add("generate", "Regenerate", "Abstained." if st.answer.abstained else "New answer written.", answer=st.answer.text)
            st.signals = self._detect(st)
            trace.add("detect", "Checks passed" if st.signals.passed else "Checks failed",
                      ", ".join(st.signals.failed_checks) or "All checks passed.", signals=st.signals.to_dict())

        return self._result(st, mode, trace, diagnoses, repairs, first=first, enforce=(mode != "no_abstain"))

    # ----------------------------------------------------------------- result
    def _result(self, st: State, mode: str, trace: Trace, diagnoses, repairs, first: Answer, enforce: bool = True) -> dict:
        sig = st.signals or Signals()
        ans = st.answer or Answer(ABSTAIN, abstained=True)
        abstained = ans.abstained or (enforce and not sig.passed)
        if abstained:
            reason = diagnoses[-1].reason if diagnoses else "the answer could not be verified against the sources"
            text = f"{ABSTAIN}. The system could not verify an answer in the indexed sources ({reason}) Please consult the primary guideline."
            status = "abstained"
            trace.add("abstain", "Abstain", "Budget spent or checks failed; no unverified answer is shown.")
        else:
            text = ans.text
            status = "repaired" if repairs and (first.abstained or first.text != ans.text or first.choice != ans.choice) else "answered"
            trace.add("answer", "Answer", "Returned with citations.")
        return {
            "question": st.question,
            "mode": mode,
            "mode_label": MODES[mode],
            "status": status,
            "abstained": abstained,
            "answer": text,
            "choice": None if abstained else ans.choice,
            "notes": st.notes,
            "first_answer": {"text": first.text, "choice": first.choice, "abstained": first.abstained},
            "passages": [h.to_dict() for h in st.context],
            "claims": [c.__dict__ for c in sig.claims],
            "signals": sig.to_dict(),
            "diagnoses": [d.to_dict() for d in diagnoses],
            "repairs": repairs,
            "confidence": round(sig.confidence, 3),
            "trace": trace.steps,
            "latency_ms": trace.steps[-1]["t_ms"] if trace.steps else 0,
            "llm_calls": self.llm.calls,
            "tokens": self.llm.tokens,
        }

    def info(self) -> dict:
        return {
            "llm": self.llm.describe(),
            "retriever": self.retriever.method,
            "verifier": self.verifier.mode,
            "chunks": len(self.retriever.chunks),
            "documents": len({c.doc_id for c in self.retriever.chunks}),
            "max_repairs": settings.max_repairs,
            "top_k": settings.top_k,
            "modes": MODES,
        }
