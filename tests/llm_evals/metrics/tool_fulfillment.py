"""DeepEval metric adapter for the locked-rubric ToolFulfillmentJudge.

Wraps `graph.judges.tool_fulfillment_judge.ToolFulfillmentJudge` — the
LLM-driven evaluator that scores whether the agent's tool-call pattern
was appropriate for the question. Keeps the semantic, LLM-based approach
(not deterministic matching) to avoid false positives when the agent
correctly answers from prompt context without calling tools.
"""

from __future__ import annotations

from deepeval.test_case import LLMTestCase

from graph.judges.tool_fulfillment_judge import ToolFulfillmentJudge
from graph.judges.utils import format_tool_results
from tests.llm_evals.metrics.base import LockedRubricMetric


class ToolFulfillmentMetric(LockedRubricMetric):
    """DeepEval metric wrapping our ToolFulfillmentJudge (threshold=0.8)."""

    def __init__(self, threshold: float = 0.8):
        super().__init__(threshold=threshold)
        self._judge = ToolFulfillmentJudge(threshold=threshold)

    @property
    def __name__(self) -> str:
        return "tool_fulfillment"

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        metadata = test_case.additional_metadata or {}
        tool_results = metadata.get("tool_results", [])
        prompt_context = (
            metadata.get("prompt_context") or "(no prompt context was provided)"
        )

        tool_calls_text = format_tool_results(tool_results)

        result = await self._judge.evaluate(
            question=test_case.input or "",
            response=test_case.actual_output or "",
            tool_calls=tool_calls_text,
            available_context=prompt_context,
        )
        return self._extract_judge_result(result)
