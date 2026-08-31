"""Unit tests for the rubric report writer.

Verifies the report structure, metric ordering, and byte-identical layout
(template is fixed; only values differ between runs).

Performance budget: each test < 3 seconds.
"""

import pytest

from graph.judges.schemas import EvaluationResult
from tests.llm_evals.reporters.rubric_report import (
    METRIC_ORDER,
    METRIC_THRESHOLDS,
    QuestionResult,
    RubricReportWriter,
)


@pytest.fixture
def sample_question_result():
    """A passing QuestionResult with all metrics populated."""
    qr = QuestionResult(
        question="What are the total medical bills?",
        entry_id="medical_bills_total",
    )
    qr.set_metric(
        "faithfulness",
        0.95,
        EvaluationResult(
            metric="faithfulness",
            score=0.95,
            violating_sentences=[],
            rationale="All claims grounded.",
        ),
    )
    qr.set_metric(
        "answer_relevancy",
        0.90,
        EvaluationResult(
            metric="answer_relevancy",
            score=0.90,
            violating_sentences=[],
            rationale="Good coverage of GT.",
        ),
    )
    qr.set_metric(
        "tool_fulfillment",
        1.0,
        EvaluationResult(
            metric="tool_fulfillment",
            score=1.0,
            violating_sentences=[],
            rationale="Appropriate tool usage.",
        ),
    )
    return qr


@pytest.fixture
def failing_question_result():
    """A failing QuestionResult with violations."""
    qr = QuestionResult(
        question="When did the accident occur?",
        entry_id="accident_date",
    )
    qr.set_metric(
        "faithfulness",
        0.6,
        EvaluationResult(
            metric="faithfulness",
            score=0.6,
            violating_sentences=["[UNGROUNDED] The date was March 2024"],
            rationale="Multiple ungrounded claims.",
        ),
    )
    qr.set_metric(
        "answer_relevancy",
        0.7,
        EvaluationResult(
            metric="answer_relevancy",
            score=0.7,
            violating_sentences=["[MISSING] September 15, 2025"],
            rationale="Key date missing from response.",
        ),
    )
    qr.set_metric(
        "tool_fulfillment",
        0.8,
        EvaluationResult(
            metric="tool_fulfillment",
            score=0.8,
            violating_sentences=[],
            rationale="Tools used correctly.",
        ),
    )
    return qr


class TestQuestionResult:
    """QuestionResult tracks per-metric scores and pass/fail."""

    def test_initial_state(self):
        qr = QuestionResult(question="test?", entry_id="test_id")
        assert qr.question == "test?"
        assert qr.entry_id == "test_id"
        for metric in METRIC_ORDER:
            assert qr.scores[metric] == 0.0
            assert qr.passed[metric] is False
            assert qr.metrics[metric] is None

    def test_set_metric_passing(self):
        qr = QuestionResult(question="test?")
        qr.set_metric("faithfulness", 0.95, None)
        assert qr.scores["faithfulness"] == 0.95
        assert qr.passed["faithfulness"] is True

    def test_set_metric_failing(self):
        qr = QuestionResult(question="test?")
        qr.set_metric("faithfulness", 0.5, None)
        assert qr.scores["faithfulness"] == 0.5
        assert qr.passed["faithfulness"] is False


class TestRubricReportWriter:
    """RubricReportWriter produces the correct Markdown structure."""

    def test_empty_report(self):
        writer = RubricReportWriter(dataset_name="test_dataset")
        report = writer.render()
        assert "# DeepEval Run Report - test_dataset" in report
        assert "## Summary" in report
        assert "## Per-Question Results" in report

    def test_report_contains_all_metrics(self, sample_question_result):
        writer = RubricReportWriter(dataset_name="john_smith")
        writer.add_question_result(sample_question_result)
        report = writer.render()

        for metric in METRIC_ORDER:
            assert f"| {metric} |" in report

    def test_report_summary_pass_status(self, sample_question_result):
        writer = RubricReportWriter(dataset_name="john_smith")
        writer.add_question_result(sample_question_result)
        report = writer.render()

        assert "| PASS |" in report

    def test_report_summary_fail_status(self, failing_question_result):
        writer = RubricReportWriter(dataset_name="john_smith")
        writer.add_question_result(failing_question_result)
        report = writer.render()

        assert "| FAIL |" in report

    def test_per_question_section(self, sample_question_result):
        writer = RubricReportWriter(dataset_name="john_smith")
        writer.add_question_result(sample_question_result)
        report = writer.render()

        assert "### medical_bills_total:" in report
        assert "**faithfulness**: 0.9500 [PASS]" in report
        assert "**answer_relevancy**: 0.9000 [PASS]" in report
        assert "**tool_fulfillment**: 1.0000 [PASS]" in report

    def test_violations_shown(self, failing_question_result):
        writer = RubricReportWriter(dataset_name="john_smith")
        writer.add_question_result(failing_question_result)
        report = writer.render()

        assert "[UNGROUNDED] The date was March 2024" in report
        assert "[MISSING] September 15, 2025" in report

    def test_rationale_shown(self, sample_question_result):
        writer = RubricReportWriter(dataset_name="john_smith")
        writer.add_question_result(sample_question_result)
        report = writer.render()

        assert "rationale: All claims grounded." in report

    def test_write_creates_file(self, tmp_path, monkeypatch, sample_question_result):
        monkeypatch.setattr(
            "tests.llm_evals.reporters.rubric_report.REPORT_DIR", tmp_path
        )
        writer = RubricReportWriter(dataset_name="test_fixture")
        writer.add_question_result(sample_question_result)
        path = writer.write()

        assert path.exists()
        assert path.suffix == ".md"
        assert "test_fixture" in path.name
        content = path.read_text()
        assert "# DeepEval Run Report - test_fixture" in content

    def test_metric_order_is_locked(self):
        assert METRIC_ORDER == [
            "faithfulness",
            "answer_relevancy",
            "tool_fulfillment",
        ]

    def test_thresholds_match_judges(self):
        assert METRIC_THRESHOLDS["faithfulness"] == 0.9
        assert METRIC_THRESHOLDS["answer_relevancy"] == 0.85
        assert METRIC_THRESHOLDS["tool_fulfillment"] == 0.8
