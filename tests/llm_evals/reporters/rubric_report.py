"""Rubric report writer for DeepEval evaluation runs.

Produces a structured Markdown file using the locked rubric metric names
with a byte-identical layout (only values differ between runs on the same
dataset). Uses EvaluationResult.model_dump() for each question's output.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.judges.schemas import EvaluationResult, MetricName

logger = logging.getLogger(__name__)

REPORT_DIR = Path(__file__).parent.parent / "reports"

METRIC_THRESHOLDS: dict[MetricName, float] = {
    "faithfulness": 0.9,
    "answer_relevancy": 0.85,
    "tool_fulfillment": 0.8,
}

METRIC_ORDER: list[MetricName] = [
    "faithfulness",
    "answer_relevancy",
    "tool_fulfillment",
]


class QuestionResult:
    """Evaluation results for a single question across all metrics."""

    def __init__(self, question: str, entry_id: str = ""):
        self.question = question
        self.entry_id = entry_id
        self.metrics: dict[MetricName, EvaluationResult | None] = {
            m: None for m in METRIC_ORDER
        }
        self.scores: dict[MetricName, float] = {m: 0.0 for m in METRIC_ORDER}
        self.passed: dict[MetricName, bool] = {m: False for m in METRIC_ORDER}

    def set_metric(
        self,
        metric: MetricName,
        score: float,
        evaluation_result: EvaluationResult | None,
    ) -> None:
        self.scores[metric] = score
        self.metrics[metric] = evaluation_result
        self.passed[metric] = score >= METRIC_THRESHOLDS[metric]


class RubricReportWriter:
    """Writes a structured Markdown report after a DeepEval evaluation run.

    The report layout is fixed — only metric values, violations, and
    rationales differ between runs. This satisfies the byte-identical
    template requirement from the acceptance criteria.
    """

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.questions: list[QuestionResult] = []
        self._start_time = datetime.now(timezone.utc)

    def add_question_result(self, result: QuestionResult) -> None:
        self.questions.append(result)

    def _compute_summary(self) -> dict[MetricName, dict[str, Any]]:
        """Compute average scores and pass rates per metric."""
        summary: dict[MetricName, dict[str, Any]] = {}
        for metric in METRIC_ORDER:
            scores = [q.scores[metric] for q in self.questions]
            avg = sum(scores) / len(scores) if scores else 0.0
            pass_count = sum(1 for q in self.questions if q.passed[metric])
            summary[metric] = {
                "avg_score": avg,
                "threshold": METRIC_THRESHOLDS[metric],
                "pass_count": pass_count,
                "total": len(self.questions),
                "pass_rate": (
                    pass_count / len(self.questions) if self.questions else 0.0
                ),
                "overall_pass": avg >= METRIC_THRESHOLDS[metric],
            }
        return summary

    def render(self) -> str:
        """Render the report as a Markdown string."""
        lines: list[str] = []
        end_time = datetime.now(timezone.utc)
        summary = self._compute_summary()

        lines.append(f"# DeepEval Run Report - {self.dataset_name}")
        lines.append("")
        lines.append(f"**Generated:** {end_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        lines.append(f"**Questions evaluated:** {len(self.questions)}")
        lines.append(
            f"**Duration:** {(end_time - self._start_time).total_seconds():.1f}s"
        )
        lines.append("")

        lines.append("## Summary")
        lines.append("")
        lines.append("| Metric | Avg Score | Threshold | Pass Rate | Status |")
        lines.append("|--------|-----------|-----------|-----------|--------|")
        for metric in METRIC_ORDER:
            s = summary[metric]
            status = "PASS" if s["overall_pass"] else "FAIL"
            lines.append(
                f"| {metric} | {s['avg_score']:.4f} | "
                f"{s['threshold']:.2f} | "
                f"{s['pass_count']}/{s['total']} | {status} |"
            )
        lines.append("")

        lines.append("## Per-Question Results")
        lines.append("")
        for idx, q in enumerate(self.questions, 1):
            q_label = q.entry_id or f"Q{idx}"
            lines.append(f"### {q_label}: {q.question[:80]}")
            lines.append("")
            for metric in METRIC_ORDER:
                score = q.scores[metric]
                passed = "PASS" if q.passed[metric] else "FAIL"
                violations: list[str] = []
                rationale = ""
                if q.metrics[metric]:
                    violations = q.metrics[metric].violating_sentences
                    rationale = q.metrics[metric].rationale
                violation_str = "; ".join(violations) if violations else "none"
                lines.append(
                    f"- **{metric}**: {score:.4f} [{passed}] | "
                    f"violations: [{violation_str}]"
                )
                if rationale:
                    lines.append(f"  - rationale: {rationale}")
            lines.append("")

        return "\n".join(lines)

    def write(self) -> Path:
        """Write the report to disk and return the file path."""
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{self.dataset_name}_{timestamp}.md"
        path = REPORT_DIR / filename
        content = self.render()
        path.write_text(content, encoding="utf-8")
        logger.info("Rubric report written to %s", path)
        return path
