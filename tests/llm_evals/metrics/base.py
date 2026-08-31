"""Base adapter: DeepEval BaseMetric wrapping our locked-rubric judges.

Each concrete subclass instantiates one of our proven, prompt-tuned judges
(FaithfulnessJudge, AnswerRelevancyJudge, ToolFulfillmentJudge), calls
its async `.evaluate()` method inside `a_measure()`, and maps the
resulting `JudgeResult` into DeepEval's score/reason interface.

DeepEval's `assert_test` then uses `is_successful()` to gate pass/fail.
Our judges remain the source of truth for evaluation logic; DeepEval is
the execution and assertion framework only.
"""

from __future__ import annotations

from typing import Optional

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from graph.judges.schemas import EvaluationResult, JudgeResult


class LockedRubricMetric(BaseMetric):
    """Abstract adapter: DeepEval BaseMetric backed by a locked-rubric judge.

    Subclasses implement `a_measure()` to call the appropriate judge with
    the correct kwargs extracted from `test_case.additional_metadata`.

    Attributes:
        score: The 0.0-1.0 score returned by the judge.
        reason: The judge's rationale string.
        evaluation_result: The full EvaluationResult if the judge produced one.
        judge_result: The raw JudgeResult from the judge call.
    """

    def __init__(self, threshold: float):
        self.threshold = threshold
        self.score: Optional[float] = None
        self.reason: Optional[str] = None
        self.evaluation_result: Optional[EvaluationResult] = None
        self.judge_result: Optional[JudgeResult] = None

    @property
    def __name__(self) -> str:
        raise NotImplementedError

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        raise NotImplementedError

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        """Sync measurement not supported — use a_measure via async runner."""
        import asyncio

        return asyncio.run(self.a_measure(test_case, *args, **kwargs))

    def is_successful(self) -> bool:
        if self.score is None:
            return False
        return self.score >= self.threshold

    def _extract_judge_result(self, result: JudgeResult) -> float:
        """Store judge output fields and return the score."""
        self.judge_result = result
        self.score = result.score
        self.reason = result.reasoning
        eval_data = result.metadata.get("evaluation_result")
        if eval_data and isinstance(eval_data, dict):
            self.evaluation_result = EvaluationResult(**eval_data)
        return self.score
