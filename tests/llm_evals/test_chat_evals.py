"""DeepEval chat agent evaluation tests.

Parametrized tests that run the locked-rubric judges against live chat
agent responses, using DeepEval's `assert_test` as the assertion gate.
Ground truth is loaded from Langfuse Datasets.

Run via:
    make test-llm-evals            # all fixtures
    make test-llm-evals-smoke      # john_smith + jane_doe only
"""

from __future__ import annotations

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from tests.llm_evals.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ToolFulfillmentMetric,
)


@pytest.mark.llm
@pytest.mark.deepeval
@pytest.mark.asyncio
class TestChatEvals:
    """Chat agent evaluation tests using the locked rubric via DeepEval."""

    async def test_chat_quality(self, eval_test_case: LLMTestCase):
        """Assert all locked-rubric metrics pass for this GT question.

        DeepEval's `assert_test` raises `AssertionError` when any metric
        scores below its threshold, causing pytest to report a failure.
        """
        assert_test(
            eval_test_case,
            [
                FaithfulnessMetric(threshold=0.9),
                AnswerRelevancyMetric(threshold=0.85),
                ToolFulfillmentMetric(threshold=0.8),
            ],
        )
