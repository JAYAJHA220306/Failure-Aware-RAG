"""Map detector signals to ONE failure type (the project's taxonomy).

Rules are checked in priority order; the first that fires is the diagnosis.
The taxonomy follows Barnett et al. (CAIN 2024), RGB (AAAI 2024) and RAGChecker (NeurIPS 2024).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .config import settings
from .detectors import Signals


class Failure(str, Enum):
    NONE = "none"
    CONFLICT = "conflicting_evidence"
    MISSING = "missing_evidence"
    RANKING = "poor_ranking"
    MULTIHOP = "multi_hop_gap"
    POSITION = "position_effect"
    NOISY = "noisy_context"
    UNSUPPORTED = "unsupported_claims"


LABELS = {
    Failure.NONE: "No failure",
    Failure.CONFLICT: "Conflicting evidence",
    Failure.MISSING: "Missing evidence",
    Failure.RANKING: "Poor ranking",
    Failure.MULTIHOP: "Multi-hop gap",
    Failure.POSITION: "Position effect",
    Failure.NOISY: "Noisy context",
    Failure.UNSUPPORTED: "Unsupported claims",
}


@dataclass
class Diagnosis:
    failure: Failure
    reason: str

    def to_dict(self) -> dict:
        return {"failure": self.failure.value, "label": LABELS[self.failure], "reason": self.reason}


def diagnose(s: Signals) -> Diagnosis:
    if s.passed:
        return Diagnosis(Failure.NONE, "All checks passed.")
    if s.conflict >= settings.conflict_threshold:
        a, b = s.conflict_pair or (0, 0)
        return Diagnosis(Failure.CONFLICT, f"Passages [{a}] and [{b}] disagree (score {s.conflict:.2f}).")
    if s.better_below_cutoff > settings.ranking_margin:
        return Diagnosis(Failure.RANKING, f"A passage below the top-k cut-off is {s.better_below_cutoff:.2f} more relevant than any passage used.")
    if s.top_relevance < settings.min_top_relevance:
        return Diagnosis(Failure.MISSING, f"Best passage relevance {s.top_relevance:.2f} is below {settings.min_top_relevance}.")
    if s.compositional and (len(s.uncovered_terms) >= 2 or "incomplete_answer" in s.failed_checks):
        detail = f"terms not covered by context: {', '.join(s.uncovered_terms[:5])}" if len(s.uncovered_terms) >= 2 else f"answer covers only {s.answer_coverage:.0%} of the question"
        return Diagnosis(Failure.MULTIHOP, f"Compositional question; {detail}.")
    if s.position_flag and (s.abstained or s.unsupported_fraction > 0):
        return Diagnosis(Failure.POSITION, f"Most relevant passage is [{s.best_passage_position}], in the middle of the context, and the answer did not use it.")
    if s.relevant_fraction < settings.min_relevant_fraction:
        return Diagnosis(Failure.NOISY, f"Only {s.relevant_fraction:.0%} of passages are relevant.")
    if s.unsupported_fraction > settings.max_unsupported_fraction:
        bad = sum(not c.supported for c in s.claims)
        return Diagnosis(Failure.UNSUPPORTED, f"{bad} of {len(s.claims)} answer claims are not supported by the passages.")
    if s.abstained:
        return Diagnosis(Failure.MISSING, "The generator could not find the answer in the passages.")
    return Diagnosis(Failure.UNSUPPORTED, "The answer has no checkable claims.")
