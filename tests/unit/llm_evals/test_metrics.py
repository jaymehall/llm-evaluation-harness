"""Unit tests for DeepEval metric adapters.

Verifies the structural contract between our LockedRubricMetric adapters
and DeepEval's BaseMetric interface. Does NOT call real LLMs — mocks the
underlying judge `.evaluate()` method.

Performance budget: each test < 3 seconds.
"""

from unittest.mock import AsyncMock, patch

import pytest
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from graph.judges.schemas import JudgeResult
from tests.llm_evals.metrics.answer_relevancy import (
    AnswerRelevancyMetric,
    _format_ground_truth,
)
from tests.llm_evals.metrics.base import LockedRubricMetric
from tests.llm_evals.metrics.faithfulness import FaithfulnessMetric
from tests.llm_evals.metrics.tool_fulfillment import ToolFulfillmentMetric


@pytest.fixture
def passing_judge_result():
    """JudgeResult that passes all thresholds."""
    return JudgeResult(
        is_valid=True,
        score=0.95,
        reasoning="All claims are grounded in source data.",
        judge_name="TestJudge",
        metadata={
            "metric": "faithfulness",
            "violating_sentences": [],
            "evaluation_result": {
                "metric": "faithfulness",
                "score": 0.95,
                "violating_sentences": [],
                "rationale": "All claims are grounded.",
            },
        },
    )


@pytest.fixture
def failing_judge_result():
    """JudgeResult that fails the faithfulness threshold."""
    return JudgeResult(
        is_valid=False,
        score=0.6,
        reasoning="Multiple claims are ungrounded.",
        judge_name="TestJudge",
        metadata={
            "metric": "faithfulness",
            "violating_sentences": ["[UNGROUNDED] Claim X not in sources"],
            "evaluation_result": {
                "metric": "faithfulness",
                "score": 0.6,
                "violating_sentences": ["[UNGROUNDED] Claim X not in sources"],
                "rationale": "Multiple claims are ungrounded.",
            },
        },
    )


@pytest.fixture
def sample_test_case():
    """LLMTestCase with realistic metadata."""
    return LLMTestCase(
        input="What are the total medical bills?",
        actual_output="The total medical bills are $10,820.",
        expected_output="Medical bills total approximately $10,820.",
        additional_metadata={
            "tool_results": [
                {"tool_name": "get_entity_data", "result": {"amount": 10820}}
            ],
            "prompt_context": "Case involves medical bills totaling $10,820.",
            "expected_facts": {"medical_bills_total": "$10,820"},
            "expected_contains": ["$10,820"],
            "expected_not_contains": [],
            "entry_id": "medical_bills_total",
            "fixture_name": "john_smith",
            "item_id": "test-item-123",
        },
    )


class TestLockedRubricMetricBase:
    """Base adapter inherits BaseMetric and enforces the interface."""

    def test_inherits_base_metric(self):
        metric = FaithfulnessMetric()
        assert isinstance(metric, BaseMetric)

    def test_inherits_locked_rubric_metric(self):
        metric = FaithfulnessMetric()
        assert isinstance(metric, LockedRubricMetric)

    def test_initial_state(self):
        metric = FaithfulnessMetric()
        assert metric.score is None
        assert metric.reason is None
        assert metric.evaluation_result is None
        assert metric.judge_result is None

    def test_is_successful_none_score(self):
        metric = FaithfulnessMetric()
        assert metric.is_successful() is False

    def test_is_successful_passing(self):
        metric = FaithfulnessMetric(threshold=0.9)
        metric.score = 0.95
        assert metric.is_successful() is True

    def test_is_successful_failing(self):
        metric = FaithfulnessMetric(threshold=0.9)
        metric.score = 0.7
        assert metric.is_successful() is False


class TestFaithfulnessMetric:
    """FaithfulnessMetric wraps FaithfulnessJudge correctly."""

    def test_metric_name(self):
        metric = FaithfulnessMetric()
        assert metric.__name__ == "faithfulness"

    def test_default_threshold(self):
        metric = FaithfulnessMetric()
        assert metric.threshold == 0.9

    def test_custom_threshold(self):
        metric = FaithfulnessMetric(threshold=0.8)
        assert metric.threshold == 0.8

    @pytest.mark.asyncio
    async def test_a_measure_passing(self, sample_test_case, passing_judge_result):
        metric = FaithfulnessMetric()
        with patch.object(
            metric._judge, "evaluate", new_callable=AsyncMock
        ) as mock_eval:
            mock_eval.return_value = passing_judge_result
            score = await metric.a_measure(sample_test_case)

        assert score == 0.95
        assert metric.score == 0.95
        assert metric.is_successful() is True
        mock_eval.assert_called_once()
        call_kwargs = mock_eval.call_args.kwargs
        assert "response" in call_kwargs
        assert "source_data" in call_kwargs

    @pytest.mark.asyncio
    async def test_a_measure_failing(self, sample_test_case, failing_judge_result):
        metric = FaithfulnessMetric()
        with patch.object(
            metric._judge, "evaluate", new_callable=AsyncMock
        ) as mock_eval:
            mock_eval.return_value = failing_judge_result
            score = await metric.a_measure(sample_test_case)

        assert score == 0.6
        assert metric.is_successful() is False

    @pytest.mark.asyncio
    async def test_evaluation_result_extracted(
        self, sample_test_case, passing_judge_result
    ):
        metric = FaithfulnessMetric()
        with patch.object(
            metric._judge, "evaluate", new_callable=AsyncMock
        ) as mock_eval:
            mock_eval.return_value = passing_judge_result
            await metric.a_measure(sample_test_case)

        assert metric.evaluation_result is not None
        assert metric.evaluation_result.metric == "faithfulness"
        assert metric.evaluation_result.score == 0.95


class TestAnswerRelevancyMetric:
    """AnswerRelevancyMetric wraps AnswerRelevancyJudge correctly."""

    def test_metric_name(self):
        metric = AnswerRelevancyMetric()
        assert metric.__name__ == "answer_relevancy"

    def test_default_threshold(self):
        metric = AnswerRelevancyMetric()
        assert metric.threshold == 0.85

    @pytest.mark.asyncio
    async def test_a_measure_passes_correct_kwargs(self, sample_test_case):
        metric = AnswerRelevancyMetric()
        mock_result = JudgeResult(
            is_valid=True,
            score=0.9,
            reasoning="Good coverage.",
            judge_name="AnswerRelevancyJudge",
            metadata={
                "metric": "answer_relevancy",
                "violating_sentences": [],
                "evaluation_result": {
                    "metric": "answer_relevancy",
                    "score": 0.9,
                    "violating_sentences": [],
                    "rationale": "Good coverage.",
                },
            },
        )
        with patch.object(
            metric._judge, "evaluate", new_callable=AsyncMock
        ) as mock_eval:
            mock_eval.return_value = mock_result
            await metric.a_measure(sample_test_case)

        call_kwargs = mock_eval.call_args.kwargs
        assert call_kwargs["response"] == "The total medical bills are $10,820."
        assert call_kwargs["question"] == "What are the total medical bills?"
        assert "expected_facts" in call_kwargs["ground_truth"]


class TestToolFulfillmentMetric:
    """ToolFulfillmentMetric wraps ToolFulfillmentJudge correctly."""

    def test_metric_name(self):
        metric = ToolFulfillmentMetric()
        assert metric.__name__ == "tool_fulfillment"

    def test_default_threshold(self):
        metric = ToolFulfillmentMetric()
        assert metric.threshold == 0.8

    @pytest.mark.asyncio
    async def test_a_measure_passes_correct_kwargs(self, sample_test_case):
        metric = ToolFulfillmentMetric()
        mock_result = JudgeResult(
            is_valid=True,
            score=1.0,
            reasoning="Appropriate tool usage.",
            judge_name="ToolFulfillmentJudge",
            metadata={
                "metric": "tool_fulfillment",
                "violating_sentences": [],
                "evaluation_result": {
                    "metric": "tool_fulfillment",
                    "score": 1.0,
                    "violating_sentences": [],
                    "rationale": "Appropriate tool usage.",
                },
            },
        )
        with patch.object(
            metric._judge, "evaluate", new_callable=AsyncMock
        ) as mock_eval:
            mock_eval.return_value = mock_result
            await metric.a_measure(sample_test_case)

        call_kwargs = mock_eval.call_args.kwargs
        assert call_kwargs["question"] == "What are the total medical bills?"
        assert call_kwargs["response"] == "The total medical bills are $10,820."
        assert "tool_calls" in call_kwargs


class TestFormatGroundTruth:
    """_format_ground_truth renders GT metadata correctly."""

    def test_full_metadata(self):
        metadata = {
            "expected_facts": {"amount": "$10,820", "date": "September 2025"},
            "expected_contains": ["medical bills", "$10,820"],
            "expected_not_contains": ["$50,000"],
        }
        result = _format_ground_truth(metadata)
        assert "expected_facts:" in result
        assert "amount: $10,820" in result
        assert "expected_contains:" in result
        assert "- medical bills" in result
        assert "expected_not_contains:" in result
        assert "- $50,000" in result

    def test_empty_metadata(self):
        result = _format_ground_truth({})
        assert result == "(no structured ground truth)"

    def test_partial_metadata(self):
        metadata = {"expected_contains": ["test"]}
        result = _format_ground_truth(metadata)
        assert "expected_contains:" in result
        assert "- test" in result
        assert "expected_facts:" not in result
