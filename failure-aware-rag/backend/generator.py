"""Everything that asks the LLM to write: answers, query rewrites, sub-questions.

Each function has an extractive fallback for LLM_PROVIDER=mock, so the whole pipeline
and UI run without an API key. The mock 'reads' only the first 3 and the last passage,
which imitates the lost-in-the-middle effect (Liu et al., TACL 2024) so the
position repair can be demonstrated offline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .llm import LLM
from .retrieval import Hit
from .textutils import content_terms, coverage, split_question, split_sentences

ABSTAIN = "INSUFFICIENT EVIDENCE"
MOCK_HEAD, MOCK_TAIL = 3, 1

ANSWER_SYSTEM = (
    "You are a clinical question-answering assistant for clinicians. "
    "Answer ONLY from the numbered passages provided. "
    "Put a citation like [2] at the end of every sentence. "
    f"If the passages do not contain the answer, reply exactly: {ABSTAIN}. "
    "Do not use outside knowledge. Keep the answer under 120 words."
)


@dataclass
class Answer:
    text: str
    choice: str | None = None
    abstained: bool = False
    notes: list[str] = field(default_factory=list)


def format_passages(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{i}] ({h.chunk.title}) {h.chunk.text}" for i, h in enumerate(hits, 1))


def parse_answer(raw: str, options: dict | None) -> Answer:
    text = raw.strip()
    choice = None
    if options:
        m = re.search(r"answer\s*[:\-]\s*\(?([A-Z])\)?", text, re.I)
        if m and m.group(1).upper() in options:
            choice = m.group(1).upper()
        text = re.sub(r"\n?\s*answer\s*[:\-].*$", "", text, flags=re.I | re.S).strip()
    abstained = ABSTAIN.lower() in raw.lower() and choice is None
    return Answer(text=ABSTAIN if abstained else text, choice=choice, abstained=abstained)


class Generator:
    def __init__(self, llm: LLM):
        self.llm = llm
        self.idf: dict[str, float] = {}  # term weights for the mock generator

    # ----------------------------------------------------------------- answer
    def answer(self, question: str, hits: list[Hit], options: dict | None = None, strict: bool = False,
               subquestions: list[str] | None = None) -> Answer:
        if not hits:
            return Answer(ABSTAIN, abstained=True)
        if self.llm.is_mock:
            if subquestions and len(subquestions) > 1 and not options:
                return self._mock_composed(question, subquestions, hits)
            return self._mock_answer(question, hits, options)
        user = f"Passages:\n{format_passages(hits)}\n\nQuestion: {question}"
        if subquestions and len(subquestions) > 1:
            user += "\nAnswer every part of the question:\n" + "\n".join(f"- {q}" for q in subquestions)
        if options:
            user += "\nOptions:\n" + "\n".join(f"{k}. {v}" for k, v in options.items())
            user += "\nExplain briefly with citations, then on the last line write 'Answer: <letter>'."
        if strict:
            user += "\nState only facts written explicitly in the passages; leave out anything else."
        return parse_answer(self.llm.chat(ANSWER_SYSTEM, user), options)

    def closed_book(self, question: str, options: dict | None = None) -> Answer:
        """Baseline B0: the LLM answers from its own knowledge, no retrieval."""
        if self.llm.is_mock:
            return Answer(ABSTAIN, abstained=True, notes=["Mock mode cannot answer without passages."])
        user = f"Question: {question}"
        if options:
            user += "\nOptions:\n" + "\n".join(f"{k}. {v}" for k, v in options.items())
            user += "\nThink briefly, then on the last line write 'Answer: <letter>'."
        raw = self.llm.chat("You are a careful clinical expert. Answer concisely.", user)
        return parse_answer(raw, options)

    def _visible(self, hits: list[Hit]) -> list[tuple[int, Hit]]:
        numbered = list(enumerate(hits, 1))
        if len(numbered) <= MOCK_HEAD + MOCK_TAIL:
            return numbered
        return numbered[:MOCK_HEAD] + numbered[-MOCK_TAIL:]

    def _mock_answer(self, question: str, hits: list[Hit], options: dict | None) -> Answer:
        q_terms = content_terms(question)
        visible = self._visible(hits)
        if options:
            best, best_score = None, 0.0
            for letter, text in options.items():
                o_terms = content_terms(text)
                score = max((coverage(o_terms, h.chunk.text) * (0.5 + coverage(q_terms, h.chunk.text)) for _, h in visible), default=0)
                if score > best_score:
                    best, best_score = letter, score
            if best is None or best_score < 0.2:
                return Answer(ABSTAIN, abstained=True)
            src = max(visible, key=lambda x: coverage(content_terms(options[best]), x[1].chunk.text))[0]
            return Answer(f"Option {best} ({options[best]}) is best supported by the passages [{src}].", choice=best)
        scored = []
        for idx, h in visible:
            for sent in split_sentences(h.chunk.text):
                if len(sent.split()) < 6:  # skip headings and fragments
                    continue
                scored.append((coverage(q_terms, sent, self.idf), idx, sent))
        scored.sort(key=lambda x: -x[0])
        top = scored[0][0] if scored else 0.0
        picked = [s for s in scored if s[0] >= max(0.34, 0.7 * top)][:2]
        if not picked:
            return Answer(ABSTAIN, abstained=True)
        return Answer(" ".join(f"{s.rstrip('.')} [{idx}]." for _, idx, s in picked))

    def _mock_composed(self, question: str, subquestions: list[str], hits: list[Hit]) -> Answer:
        """Self-Ask style: one best sentence per sub-question, tie-broken by relevance to the whole question."""
        main = content_terms(question)
        used, parts = set(), []
        for sq in subquestions:
            terms = content_terms(sq)
            best = None
            for idx, h in self._visible(hits):
                ctx = coverage(main, h.chunk.text, self.idf)
                for sent in split_sentences(h.chunk.text):
                    if len(sent.split()) < 6 or sent in used:
                        continue
                    score = coverage(terms, sent, self.idf) + 0.5 * ctx
                    if best is None or score > best[0]:
                        best = (score, idx, sent)
            if best and best[0] >= 0.5:
                used.add(best[2])
                parts.append(f"{best[2].rstrip('.')} [{best[1]}].")
        if not parts:
            return Answer(ABSTAIN, abstained=True)
        return Answer(" ".join(parts))

    # --------------------------------------------------------- repair helpers
    def rewrite_query(self, question: str) -> str:
        """HyDE-style rewrite: a hypothetical answer passage plus key terms, used as the search query."""
        if self.llm.is_mock:
            return " ".join(dict.fromkeys(content_terms(question)))
        hypo = self.llm.chat(
            "Write a 2-3 sentence passage, in the style of a clinical guideline, that would answer the question. "
            "Include key medical terms and common synonyms. It is used only as a search query.",
            question,
            max_tokens=150,
        )
        return f"{question} {hypo}"

    def decompose(self, question: str) -> list[str]:
        if self.llm.is_mock:
            return split_question(question)
        reply = self.llm.chat(
            "Split the question into 2-4 self-contained sub-questions that must each be answered "
            "to answer the original. One per line, no numbering.",
            question,
            max_tokens=200,
        )
        subs = [re.sub(r"^[\-\d.)\s]+", "", line).strip() for line in reply.splitlines() if line.strip()]
        return subs[:4] or [question]
