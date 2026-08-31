"""Answer Relevancy judge for harness ground-truth comparison.

Renamed from ``AccuracyJudge``. Compares the response to
authored ground truth using paraphrase-tolerant semantic matching —
"approximately $13K" ≈ "$12,800", "mid-September 2025" ≈ "September 15,
2025", etc.

Emits the locked rubric output (``EvaluationResult``). See
docs/architecture/evaluation-rubric.md §2.2 for the locked metric
definition.

Usage::

    from graph.judges.answer_relevancy_judge import AnswerRelevancyJudge

    judge = AnswerRelevancyJudge()
    result = await judge.evaluate(
        response="The accident happened in mid-September 2025 on I-25.",
        question="When did the accident occur and where?",
        ground_truth=(
            "expected_contains:\\n  - September\\n  - I-25\\n"
            "expected_facts:\\n  accident_date: September 15, 2025\\n"
        ),
    )
"""

from typing import ClassVar

from django.conf import settings

from graph.judges.base import LockedRubricJudge
from graph.judges.prompts import ANSWER_RELEVANCY_EVAL_PROMPT
from graph.judges.schemas import EvaluationResult, MetricName


class AnswerRelevancyJudge(LockedRubricJudge):
    """Locked-rubric judge for the Answer Relevancy metric.

    Scores ``response`` against the question's authored ground truth.
    Returns a score 0.0–1.0 representing the fraction of GT items
    (expected_facts + expected_contains + expected_not_contains) that
    the response handles correctly under paraphrase tolerance. Emits
    the locked ``EvaluationResult`` schema.

    Always runs on ``settings.OPENAI_MODEL_SMALL``.

    Args:
        threshold: Minimum score to count as a pass (default 0.85).
            Below 1.0 to allow paraphrase slack — the judge is supposed
            to accept "mid-September 2025" when GT says "September 15,
            2025". Inherited from the prior AccuracyJudge.
    """

    EXPECTED_METRIC: ClassVar[MetricName] = "answer_relevancy"

    def __init__(self, threshold: float = 0.85):
        super().__init__(
            name="AnswerRelevancyJudge",
            prompt_template=ANSWER_RELEVANCY_EVAL_PROMPT,
            evaluation_schema=EvaluationResult,
            model=settings.OPENAI_MODEL_SMALL,
            max_tokens=4096,
            temperature=0.0,
            threshold=threshold,
        )
