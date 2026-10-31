"""Run with:  pytest -q   (uses the demo corpus, mock LLM and lexical verifier; no downloads)."""
import os

os.environ["LLM_PROVIDER"] = "mock"
os.environ["VERIFIER"] = "lexical"
os.environ["USE_DENSE"] = "false"
os.environ["USE_RERANKER"] = "false"

import pytest  # noqa: E402

from backend.config import settings  # noqa: E402
from backend.ingest import build_store, chunk_text  # noqa: E402
from backend.pipeline import MODES, Pipeline  # noqa: E402
from backend.retrieval import Retriever  # noqa: E402
from backend.verifier import Verifier  # noqa: E402
from evaluation.datasets import load_dataset  # noqa: E402
from evaluation.inject import make_perturbation  # noqa: E402
from evaluation.metrics import is_correct  # noqa: E402
from evaluation.run_eval import evaluate  # noqa: E402


@pytest.fixture(scope="module")
def pipe(tmp_path_factory):
    index_dir = tmp_path_factory.mktemp("index")
    chunks = build_store(settings.corpus_dir, index_dir)
    return Pipeline(Retriever(chunks))


@pytest.fixture(scope="module")
def items():
    return {i.id: i for i in load_dataset("demo", eval_dir=settings.eval_dir)}


def test_chunking_respects_size():
    text = "\n\n".join(["word " * 50] * 4)
    chunks = chunk_text(text, size=100, overlap=20)
    assert all(len(c.split()) <= 100 for c in chunks)
    assert len(chunks) == 2


def test_answerable_question_is_answered_with_citation(pipe, items):
    r = pipe.run(items["ana-first"].question, "failure_aware")
    assert not r["abstained"]
    assert "[1]" in r["answer"]
    assert is_correct(items["ana-first"], r)


def test_unanswerable_question_abstains(pipe, items):
    r = pipe.run(items["un-inr"].question, "failure_aware")
    assert r["abstained"]
    assert r["diagnoses"][0]["failure"] == "missing_evidence"


def test_conflict_is_diagnosed_and_resolved_by_authority(pipe, items):
    r = pipe.run(items["hypo-tx"].question, "failure_aware")
    assert r["diagnoses"][0]["failure"] == "conflicting_evidence"
    assert all(p["authority"] != "forum" for p in r["passages"])
    assert is_correct(items["hypo-tx"], r)


def test_multi_hop_repair_completes_the_answer(pipe, items):
    item = items["ana-multi"]
    assert not is_correct(item, pipe.run(item.question, "vanilla"))
    r = pipe.run(item.question, "failure_aware")
    assert r["repairs"] == ["Decompose and retrieve"]
    assert is_correct(item, r)


def test_position_injection_is_diagnosed(pipe, items):
    item = items["ana-dose"]
    r = pipe.run(item.question, "failure_aware", item.options, make_perturbation("position", item, pipe))
    assert r["diagnoses"][0]["failure"] == "position_effect"
    assert is_correct(item, r)


def test_conflict_heuristic_ignores_ranges_in_same_text():
    v = Verifier()
    text = "Stage 1 hypertension is a systolic pressure of 130 to 139 mmHg or a diastolic pressure of 80 to 89 mmHg."
    assert v.number_conflict(text, text) == 0.0


def test_every_mode_runs(pipe, items):
    for mode in MODES:
        r = pipe.run(items["htn-drugs"].question, mode, items["htn-drugs"].options)
        assert r["mode"] == mode and r["trace"]


def test_evaluation_produces_summary(pipe, items):
    report = evaluate(pipe, list(items.values())[:4], ["vanilla", "failure_aware"], ["none", "conflict"])
    assert report["summary"] and report["rows"]
    assert {"accuracy_%", "diagnosis_acc_%", "aurc"} <= set(report["summary"][0])
