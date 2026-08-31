"""Tool Fulfillment judge for harness tool-call evaluation.

Scores whether the agent's tool-call pattern was appropriate for the
question given the data already available in the agent's prompt context.
Does NOT score response correctness — that is the job of
FaithfulnessJudge (precision) and AnswerRelevancyJudge (recall against
GT).

The judge is context-aware: it receives the full prompt context injected
into the agent's system prompt so it can distinguish "no tools needed
because data was in context" from "no tools called when data retrieval
was needed." This eliminates false positives where agents correctly
answer from prompt context without calling tools.

This is the LLM-driven implementation of the locked rubric. The
DeepEval integration wraps this judge via a ``BaseMetric``
adapter rather than replacing it with a deterministic metric —
deterministic tool-call matching creates false positives when the agent
correctly answers from prompt context without calling tools, or when
multiple valid tool strategies exist. The ``EvaluationResult`` shape and
``metric='tool_fulfillment'`` tag are stable; callers will not need to
change.

See docs/architecture/evaluation-rubric.md §2.3 for the locked metric
definition.

Usage::

    from graph.judges.tool_fulfillment_judge import ToolFulfillmentJudge
    from graph.judges.utils import format_tool_results

    judge = ToolFulfillmentJudge()
    result = await judge.evaluate(
        question="What is the case status?",
        response="The case is open and pending review.",
        tool_calls=format_tool_results(tool_results),
        available_context="Case Type: MVA\\nStatus: Open",
    )
"""

from typing import ClassVar

from django.conf import settings

from graph.judges.base import LockedRubricJudge
from graph.judges.prompts import TOOL_FULFILLMENT_EVAL_PROMPT
from graph.judges.schemas import EvaluationResult, MetricName


class ToolFulfillmentJudge(LockedRubricJudge):
    """Locked-rubric judge for the Tool Fulfillment metric.

    Context-aware judge that scores the agent's tool-call pattern against
    the question's domain on a 0.0–1.0 scale, accounting for data already
    available in the agent's prompt context.

    Violation prefixes:
      - ``[WRONG_TOOL]``: wrong or unnecessary tool calls (wasted latency).
      - ``[MISSING_TOOL]``: no tool called when data retrieval was needed.
      - ``[OFF_TOPIC]``: entirely unrelated tool calls.

    Always runs on ``settings.OPENAI_MODEL_SMALL`` — tool-call adequacy
    is a structured check, and the small model is sufficient.

    Args:
        threshold: Minimum score to pass (default 0.8). Wider than
            Faithfulness/Answer Relevancy because the LLM judge for
            tool fit is inherently more variable than a per-claim
            structural check.
    """

    EXPECTED_METRIC: ClassVar[MetricName] = "tool_fulfillment"

    def __init__(self, threshold: float = 0.8):
        super().__init__(
            name="ToolFulfillmentJudge",
            prompt_template=TOOL_FULFILLMENT_EVAL_PROMPT,
            evaluation_schema=EvaluationResult,
            model=settings.OPENAI_MODEL_SMALL,
            max_tokens=4096,
            temperature=0.0,
            threshold=threshold,
        )
