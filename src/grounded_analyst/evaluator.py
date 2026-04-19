from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from grounded_analyst.pipeline import GroundedAnalystPipeline


@dataclass
class EvalRow:
    question: str
    expected_answer_contains: str


def load_eval_set(path: Path) -> list[EvalRow]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        EvalRow(
            question=row["question"],
            expected_answer_contains=row["expected_answer_contains"],
        )
        for row in data
    ]


def run_eval(pipeline: GroundedAnalystPipeline, eval_set: list[EvalRow]) -> dict:
    results = []
    passed = 0
    for row in eval_set:
        ans = pipeline.ask(row.question)
        ok = _normalized_contains(ans.answer, row.expected_answer_contains)
        if ok:
            passed += 1
        results.append(
            {
                "question": row.question,
                "expected_contains": row.expected_answer_contains,
                "answer": ans.answer,
                "citations": ans.citations,
                "clarification_requested": ans.clarification_requested,
                "pass": ok,
            }
        )
    return {"passed": passed, "total": len(eval_set), "details": results}


def _normalized_contains(answer: str, expected: str) -> bool:
    answer_n = _normalize_text(answer)
    expected_n = _normalize_text(expected)
    return expected_n in answer_n


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace(",", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text
