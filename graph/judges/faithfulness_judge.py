"""FaithfulnessJudge — locked-rubric grounding evaluator.

FaithfulnessJudge (renamed from GroundingJudge) verifies that
LLM response claims are grounded in the source data the chat agent saw —
both tool results AND the prompt-context block injected by
``OpenAIConversationalAssistant._get_rich_node_context`` (precision).

Emits the locked rubric output (``EvaluationResult``). See
docs/architecture/evaluation-rubric.md §2.1 for the locked metric
definition.

Usage::

    from graph.judges.faithfulness_judge import FaithfulnessJudge
    from graph.judges.utils import format_grounding_sources

    source_data = format_grounding_sources(
        tool_results=[{"tool_name": "get_entity_data", "result": {"amount": 10820}}],
        prompt_context="(node-context block from the chat agent system prompt)",
    )

    judge = FaithfulnessJudge()
    result = await judge.evaluate(
        response="The total medical bills are $10,820...",
        source_data=source_data,
    )
"""

from typing import ClassVar

from django.conf import settings

from graph.judges.base import LockedRubricJudge
from graph.judges.prompts import FAITHFULNESS_EVAL_PROMPT
from graph.judges.schemas import EvaluationResult, MetricName


class FaithfulnessJudge(LockedRubricJudge):
    """Locked-rubric judge for the Faithfulness metric.

    Verifies that every factual claim in the response is grounded in at
    least one source channel (tool_results or prompt_context). Emits the
    locked ``EvaluationResult`` schema. Replaces the previous
    ``GroundingJudge``.

    Always runs on ``settings.OPENAI_MODEL_SMALL`` — Faithfulness is a
    structured "is this claim traceable to that source?" check, and the
    small model is sufficient. Cuts judge cost ~5x vs. the large model
    without measurable score regression in harness fixtures.

    Args:
        threshold: Minimum score to pass (default 0.9).
    """

    EXPECTED_METRIC: ClassVar[MetricName] = "faithfulness"

    def __init__(self, threshold: float = 0.9):
        super().__init__(
            name="FaithfulnessJudge",
            prompt_template=FAITHFULNESS_EVAL_PROMPT,
            evaluation_schema=EvaluationResult,
            model=settings.OPENAI_MODEL_SMALL,
            max_tokens=4096,
            temperature=0.0,
            threshold=threshold,
        )
