"""Failure injection: perturb retrieval on purpose so every failure has a known, true label.

This gives ground truth for diagnosis accuracy (RQ2), which no public benchmark provides.
Each injector needs the item's gold_doc (the document that holds the answer).
"""
from __future__ import annotations

import random

from backend.diagnosis import Failure
from backend.ingest import Chunk
from backend.pipeline import Perturbation, Pipeline
from backend.retrieval import Hit

INJECTIONS = {
    "missing": Failure.MISSING,
    "poor_ranking": Failure.RANKING,
    "noisy": Failure.NOISY,
    "position": Failure.POSITION,
    "conflict": Failure.CONFLICT,
}


def _is_gold(h: Hit, gold_doc: str) -> bool:
    return h.chunk.doc_id == gold_doc


def _distractors(pipeline: Pipeline, gold_doc: str, question: str, n: int, rng: random.Random) -> list[Hit]:
    pool = [c for c in pipeline.retriever.chunks if c.doc_id != gold_doc]
    picked = rng.sample(pool, min(n, len(pool)))
    return pipeline.score(question, [Hit(c, 0.0) for c in picked])


def make_perturbation(kind: str, item, pipeline: Pipeline, seed: int = 13) -> Perturbation | None:
    """Return a Perturbation for this item, or None if the injection does not apply to it."""
    if kind == "none":
        return Perturbation()
    if kind not in INJECTIONS or not item.gold_doc:
        return None
    if kind == "conflict" and not item.counterfactual:
        return None
    rng = random.Random(f"{seed}-{item.id}-{kind}")
    gold, label = item.gold_doc, INJECTIONS[kind]

    if kind == "missing":
        return Perturbation(label, on_candidates=lambda c: [h for h in c if not _is_gold(h, gold)])

    if kind == "poor_ranking":
        def push_down(c):
            golds = [h for h in c if _is_gold(h, gold)]
            return [h for h in c if not _is_gold(h, gold)] + golds
        return Perturbation(label, on_candidates=push_down)

    if kind == "noisy":
        def add_noise(ctx):
            golds = [h for h in ctx if _is_gold(h, gold)][:1]
            noise = _distractors(pipeline, gold, item.question, 5, rng)
            return golds + noise
        return Perturbation(label, on_context=add_noise)

    if kind == "position":
        def bury(ctx):
            golds = [h for h in ctx if _is_gold(h, gold)][:1]
            noise = _distractors(pipeline, gold, item.question, 6, rng)
            mid = len(noise) // 2
            return noise[:mid] + golds + noise[mid:]  # gold lands in the middle of a 7-passage context
        return Perturbation(label, on_context=bury)

    if kind == "conflict":
        fake = Chunk(id="injected#0", doc_id="injected_counterfactual", title="Injected counterfactual passage",
                     text=item.counterfactual, authority="forum")
        def contradict(ctx):
            hit = pipeline.score(item.question, [Hit(fake, 0.0)])[0]
            return ctx[:1] + [hit] + ctx[1:]
        return Perturbation(label, on_context=contradict)
    return None
