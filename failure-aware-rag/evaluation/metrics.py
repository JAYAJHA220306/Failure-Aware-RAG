"""Scoring one result, and aggregating many into the tables of the experiment plan."""
from __future__ import annotations

import math
import re
from collections import defaultdict
from statistics import median


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def is_correct(item, result: dict, answer_key: str = "final") -> bool:
    """Correct = right option / all keywords present; for unanswerable items, correct = abstained."""
    if answer_key == "first":
        ans = result["first_answer"]
        abstained, text, choice = ans["abstained"], ans["text"], ans["choice"]
    else:
        abstained, text, choice = result["abstained"], result["answer"], result["choice"]
    if not item.answerable:
        return abstained
    if abstained:
        return False
    if item.options:
        return choice == item.answer
    return all(_norm(k) in _norm(text) for k in item.answer_keywords)


def score_row(item, result: dict, mode: str, injection: str, injected_label: str | None) -> dict:
    final_ok = is_correct(item, result)
    first_ok = is_correct(item, result, "first")
    triggered = bool(result["diagnoses"])
    first_diag = result["diagnoses"][0]["failure"] if triggered else "none"
    return {
        "id": item.id,
        "split": item.split,
        "mode": mode,
        "injection": injection,
        "injected_label": injected_label,
        "answerable": item.answerable,
        "correct": final_ok,
        "first_correct": first_ok,
        "abstained": result["abstained"],
        "hallucinated": (not result["abstained"]) and any(not c["supported"] for c in result["claims"]),
        "triggered": triggered,
        "first_diagnosis": first_diag,
        "repairs": "|".join(result["repairs"]),
        "repair_gain": triggered and not first_ok and final_ok,
        "repair_regression": triggered and first_ok and not final_ok,
        "confidence": result["confidence"],
        "latency_ms": result["latency_ms"],
        "llm_calls": result["llm_calls"],
        "tokens": result["tokens"],
    }


def aurc(rows: list[dict]) -> float:
    """Area under the risk-coverage curve (lower is better). Abstentions rank last."""
    if not rows:
        return float("nan")
    ordered = sorted(rows, key=lambda r: (-1 if r["abstained"] else r["confidence"]), reverse=True)
    errors, risks = 0, []
    for i, r in enumerate(ordered, 1):
        errors += 0 if r["correct"] else 1
        risks.append(errors / i)
    return sum(risks) / len(risks)


def _pct(num: int, den: int) -> float | None:
    return round(100 * num / den, 1) if den else None


def summarise(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["mode"], r["injection"])].append(r)
    out = []
    for (mode, inj), rs in groups.items():
        ans = [r for r in rs if r["answerable"]]
        unans = [r for r in rs if not r["answerable"]]
        answered = [r for r in rs if not r["abstained"]]
        abst = [r for r in rs if r["abstained"]]
        trig = [r for r in rs if r["triggered"]]
        injected = [r for r in rs if r["injected_label"]]
        lat = sorted(r["latency_ms"] for r in rs)
        out.append({
            "mode": mode,
            "injection": inj,
            "n": len(rs),
            "accuracy_%": _pct(sum(r["correct"] for r in ans), len(ans)),
            "selective_acc_%": _pct(sum(r["correct"] for r in answered if r["answerable"]), len([r for r in answered if r["answerable"]])),
            "hallucination_%": _pct(sum(r["hallucinated"] for r in answered), len(answered)),
            "abstain_precision_%": _pct(sum(not r["answerable"] for r in abst), len(abst)),
            "abstain_recall_%": _pct(sum(r["abstained"] for r in unans), len(unans)),
            "trigger_rate_%": _pct(len(trig), len(rs)),
            "repair_gain_%": _pct(sum(r["repair_gain"] for r in trig), len(trig)),
            "repair_regression_%": _pct(sum(r["repair_regression"] for r in trig), len(trig)),
            "diagnosis_acc_%": _pct(sum(r["first_diagnosis"] == r["injected_label"] for r in injected), len(injected)),
            "aurc": round(aurc(rs), 3),
            "median_latency_ms": median(lat) if lat else None,
            "p95_latency_ms": lat[min(len(lat) - 1, math.ceil(0.95 * len(lat)) - 1)] if lat else None,
            "llm_calls_per_q": round(sum(r["llm_calls"] for r in rs) / len(rs), 2),
        })
    inj_order = ["none", "missing", "poor_ranking", "noisy", "position", "conflict"]
    return sorted(out, key=lambda r: (inj_order.index(r["injection"]) if r["injection"] in inj_order else 99, r["mode"]))


def mcnemar(rows: list[dict], mode_a: str, mode_b: str, injection: str = "none") -> dict:
    """Exact McNemar test on paired correctness of two modes over the same questions."""
    a = {r["id"]: r["correct"] for r in rows if r["mode"] == mode_a and r["injection"] == injection}
    b = {r["id"]: r["correct"] for r in rows if r["mode"] == mode_b and r["injection"] == injection}
    shared = a.keys() & b.keys()
    only_a = sum(a[i] and not b[i] for i in shared)
    only_b = sum(b[i] and not a[i] for i in shared)
    n = only_a + only_b
    if n == 0:
        p = 1.0
    else:
        k = min(only_a, only_b)
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"a": mode_a, "b": mode_b, "injection": injection, "pairs": len(shared), "a_only_correct": only_a, "b_only_correct": only_b, "p_value": round(p, 4)}
