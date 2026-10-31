"""Load evaluation questions: the project's own JSONL format, or the MIRAGE benchmark.

JSONL format (one question per line):
  {"id": "q1", "question": "...", "answer_keywords": ["70 mg/dl"], "answerable": true,
   "gold_doc": "hypoglycaemia_guideline", "counterfactual": "optional contradicting passage"}
  {"id": "q2", "question": "...", "options": {"A": "...", "B": "..."}, "answer": "B", ...}

MIRAGE (Xiong et al., Findings of ACL 2024): download benchmark.json from
https://github.com/Teddy-XiongGZ/MIRAGE and pass --dataset mirage:medqa (or medmcqa, pubmedqa, bioasq, mmlu).
Note: MIRAGE questions need a large medical corpus (e.g. MedRAG's PubMed/StatPearls/Textbooks)
in data/corpus, not the 5-file demo corpus.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalItem:
    id: str
    question: str
    options: dict | None = None
    answer: str | None = None  # option letter for multiple choice
    answer_keywords: list[str] = field(default_factory=list)  # all must appear in an open answer
    answerable: bool = True
    gold_doc: str | None = None
    counterfactual: str | None = None
    split: str = "test"
    source: str = "custom"


def load_jsonl(path: str | Path) -> list[EvalItem]:
    items = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row.setdefault("id", f"q{n}")
            known = {k: v for k, v in row.items() if k in EvalItem.__dataclass_fields__}
            items.append(EvalItem(**known))
    return items


def load_mirage(path: str | Path, subset: str, limit: int | None = None, seed: int = 13) -> list[EvalItem]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if subset not in data:
        raise ValueError(f"Unknown MIRAGE subset '{subset}'. Available: {', '.join(data)}")
    rows = list(data[subset].items())
    if limit:
        random.Random(seed).shuffle(rows)
        rows = rows[:limit]
    return [
        EvalItem(id=f"{subset}-{qid}", question=r["question"], options=r["options"], answer=r["answer"], source=f"mirage:{subset}")
        for qid, r in rows
    ]


def split_dev_test(items: list[EvalItem], dev_fraction: float = 0.2, seed: int = 13) -> tuple[list[EvalItem], list[EvalItem]]:
    """Fixed random dev/test split. Tune thresholds on dev; report test once."""
    shuffled = items[:]
    random.Random(seed).shuffle(shuffled)
    cut = int(len(shuffled) * dev_fraction)
    for it in shuffled[:cut]:
        it.split = "dev"
    return shuffled[:cut], shuffled[cut:]


def load_dataset(spec: str, mirage_path: str | None = None, limit: int | None = None, eval_dir: Path | None = None) -> list[EvalItem]:
    """spec: 'demo' | 'mirage:<subset>' | path/to/file.jsonl"""
    if spec.startswith("mirage:"):
        if not mirage_path:
            raise ValueError("Pass the path to MIRAGE benchmark.json (--mirage-path).")
        return load_mirage(mirage_path, spec.split(":", 1)[1], limit)
    path = Path(spec)
    if spec == "demo" and eval_dir:
        path = eval_dir / "demo_questions.jsonl"
    items = load_jsonl(path)
    return items[:limit] if limit else items
