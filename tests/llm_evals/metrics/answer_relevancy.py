"""DeepEval metric adapter for the locked-rubric AnswerRelevancyJudge.

Wraps `graph.judges.answer_relevancy_judge.AnswerRelevancyJudge` — the
semantic ground-truth comparison evaluator that scores the response
against authored GT (expected_facts, expected_contains, expected_not_contains).
"""

from __future__ import annotations

from deepeval.test_case import LLMTestCase

from graph.judges.answer_relevancy_judge import AnswerRelevancyJudge
from tests.llm_evals.metrics.base import LockedRubricMetric


class AnswerRelevancyMetric(LockedRubricMetric):
    """DeepEval metric wrapping our AnswerRelevancyJudge (threshold=0.85)."""

    def __init__(self, threshold: float = 0.85):
        super().__init__(threshold=threshold)
        self._judge = AnswerRelevancyJudge(threshold=threshold)

    @property
    def __name__(self) -> str:
        return "answer_relevancy"

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        metadata = test_case.additional_metadata or {}
        ground_truth = _format_ground_truth(metadata)

        result = await self._judge.evaluate(
            response=test_case.actual_output or "",
            question=test_case.input or "",
            ground_truth=ground_truth,
        )
        return self._extract_judge_result(result)


def _format_ground_truth(metadata: dict) -> str:
    """Render GT fields into the text block AnswerRelevancyJudge expects.

    Mirrors `pytest_judge_plugin._format_ground_truth_for_judge` — keeps
    the format compact and explicit so the judge prompt can map sections
    directly.
    """
    lines: list[str] = []

    facts = metadata.get("expected_facts")
    if facts and isinstance(facts, dict):
        lines.append("expected_facts:")
        for key, value in facts.items():
            lines.append(f"  {key}: {value}")

    contains = metadata.get("expected_contains")
    if contains and isinstance(contains, list):
        lines.append("expected_contains:")
        for value in contains:
            lines.append(f"  - {value}")

    not_contains = metadata.get("expected_not_contains")
    if not_contains and isinstance(not_contains, list):
        lines.append("expected_not_contains:")
        for value in not_contains:
            lines.append(f"  - {value}")

    return "\n".join(lines) if lines else "(no structured ground truth)"
