"""DeepEval metric adapter for the locked-rubric FaithfulnessJudge.

Wraps `graph.judges.faithfulness_judge.FaithfulnessJudge` — the LLM
grounding evaluator that verifies every factual claim in the response
is traceable to source data (tool results + prompt context).
"""

from __future__ import annotations

from deepeval.test_case import LLMTestCase

from graph.judges.faithfulness_judge import FaithfulnessJudge
from graph.judges.utils import format_grounding_sources
from tests.llm_evals.metrics.base import LockedRubricMetric


class FaithfulnessMetric(LockedRubricMetric):
    """DeepEval metric wrapping our FaithfulnessJudge (threshold=0.9)."""

    def __init__(self, threshold: float = 0.9):
        super().__init__(threshold=threshold)
        self._judge = FaithfulnessJudge(threshold=threshold)

    @property
    def __name__(self) -> str:
        return "faithfulness"

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        metadata = test_case.additional_metadata or {}
        tool_results = metadata.get("tool_results", [])
        prompt_context = metadata.get("prompt_context")

        source_data = format_grounding_sources(
            tool_results=tool_results,
            prompt_context=prompt_context,
        )

        result = await self._judge.evaluate(
            response=test_case.actual_output or "",
            source_data=source_data,
        )
        return self._extract_judge_result(result)
