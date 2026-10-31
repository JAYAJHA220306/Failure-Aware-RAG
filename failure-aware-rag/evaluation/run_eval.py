"""Run baselines, ablations and failure injections; write results/<name>.json and .csv.

Examples
  python -m evaluation.run_eval                                   # demo set, main modes, no injection
  python -m evaluation.run_eval --modes all --inject all          # every mode x every injected failure
  python -m evaluation.run_eval --dataset mirage:medqa --mirage-path data/eval/benchmark.json --limit 200
  python -m evaluation.run_eval --split dev                        # tune thresholds on the dev split only
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Callable

from backend.config import settings
from backend.pipeline import MODES, Pipeline

from .datasets import EvalItem, load_dataset, split_dev_test
from .inject import INJECTIONS, make_perturbation
from .metrics import mcnemar, score_row, summarise

DEFAULT_MODES = ["vanilla", "rerank", "crag_lite", "failure_aware", "generic", "random"]


def evaluate(
    pipeline: Pipeline,
    items: list[EvalItem],
    modes: list[str],
    injections: list[str],
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    rows, total, done = [], len(items) * len(modes) * len(injections), 0
    for inj in injections:
        for item in items:
            for mode in modes:
                done += 1
                perturb = make_perturbation(inj, item, pipeline)
                if perturb is None:  # injection needs a gold_doc / counterfactual this item lacks
                    continue
                pipeline.rng.seed(f"{settings.random_seed}-{item.id}-{inj}")
                result = pipeline.run(item.question, mode, item.options, perturb)
                label = INJECTIONS[inj].value if inj in INJECTIONS else None
                rows.append(score_row(item, result, mode, inj, label))
                if progress:
                    progress(done, total)
    summary = summarise(rows)
    tests = []
    if "failure_aware" in modes:
        for other in modes:
            if other != "failure_aware":
                for inj in injections:
                    tests.append(mcnemar(rows, "failure_aware", other, inj))
    return {"rows": rows, "summary": summary, "significance": tests}


def save(report: dict, name: str) -> Path:
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    path = settings.results_dir / f"{name}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (settings.results_dir / f"{name}_summary.csv").open("w", newline="", encoding="utf-8") as f:
        if report["summary"]:
            w = csv.DictWriter(f, fieldnames=list(report["summary"][0]))
            w.writeheader()
            w.writerows(report["summary"])
    with (settings.results_dir / f"{name}_rows.csv").open("w", newline="", encoding="utf-8") as f:
        if report["rows"]:
            w = csv.DictWriter(f, fieldnames=list(report["rows"][0]))
            w.writeheader()
            w.writerows(report["rows"])
    return path


def print_table(summary: list[dict]) -> None:
    cols = ["injection", "mode", "n", "accuracy_%", "hallucination_%", "abstain_recall_%", "trigger_rate_%",
            "repair_gain_%", "repair_regression_%", "diagnosis_acc_%", "aurc", "median_latency_ms"]
    widths = {c: max(len(c), *(len(str(r.get(c))) for r in summary)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in summary:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="demo", help="demo | mirage:<subset> | path/to/questions.jsonl")
    ap.add_argument("--mirage-path", default=None)
    ap.add_argument("--modes", default=",".join(DEFAULT_MODES), help=f"comma list or 'all' ({', '.join(MODES)})")
    ap.add_argument("--inject", default="none", help=f"none | all | comma list of {', '.join(INJECTIONS)}")
    ap.add_argument("--split", default="all", choices=["all", "dev", "test"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--name", default=None, help="output file name (default: timestamp)")
    args = ap.parse_args()

    items = load_dataset(args.dataset, args.mirage_path, args.limit, settings.eval_dir)
    if args.split != "all":
        dev, test = split_dev_test(items)
        items = dev if args.split == "dev" else test
    modes = list(MODES) if args.modes == "all" else [m.strip() for m in args.modes.split(",")]
    modes = [m for m in modes if m != "oracle"] + (["oracle"] if "oracle" in modes and args.inject != "none" else [])
    injections = ["none"] + list(INJECTIONS) if args.inject == "all" else [i.strip() for i in args.inject.split(",")]

    pipeline = Pipeline()
    print(f"{len(items)} questions x {len(modes)} modes x {len(injections)} conditions | {pipeline.info()['llm']} | verifier: {pipeline.verifier.mode}")
    t0 = time.time()
    report = evaluate(pipeline, items, modes, injections, lambda d, t: print(f"  {d}/{t}", flush=True) if d % max(1, t // 10) == 0 else None)
    print(f"\nDone in {time.time() - t0:.1f}s")
    report["config"] = {"dataset": args.dataset, "modes": modes, "injections": injections, "split": args.split, **pipeline.info()}
    path = save(report, args.name or time.strftime("run_%Y%m%d_%H%M%S"))
    print_table(report["summary"])
    print(f"\nSaved {path} (+ _summary.csv, _rows.csv)")


if __name__ == "__main__":
    main()
