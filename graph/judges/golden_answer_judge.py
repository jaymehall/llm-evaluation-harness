"""GoldenAnswerJudge — multi-turn conversational evaluation (RFC: Golden Answer Evaluation Engine).

Evaluates agent responses against human-blessed golden answers via a
structured 4-turn conversation. Each turn produces a rubric step that maps
directly to an Allure report section.

Unlike the locked-rubric judges (FaithfulnessJudge, AnswerRelevancyJudge,
ToolFulfillmentJudge) which are single-prompt evaluations, this judge uses
OpenAI's `previous_response_id` to maintain conversation context across
turns, enabling focused per-step evaluation without re-sending context.

Turn 1: Decompose test response into atomic claims
Turn 2: Check completeness (every golden fact addressed?)
Turn 3: Check faithfulness + tool correlation (contradictions? empty tools?)
Turn 4: Emit final verdict (PASS/FAIL with specific citation)

Usage::

    from graph.judges.golden_answer_judge import GoldenAnswerJudge

    judge = GoldenAnswerJudge()
    result = await judge.evaluate(
        golden_facts=["ER visit on Sept 15, 2025", "Physical therapy with Dr. Chen"],
        test_response="The client visited the ER on September 15th...",
        question="What are the medical treatments?",
        tool_calls=[{"tool": "search_medical_records", "result_count": 3}],
    )
    # result.is_valid = True/False (binary pass/fail)
    # result.metadata["rubric_steps"] = per-turn structured output for Allure
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any, Optional

from pydantic import BaseModel, Field

from graph.judges.schemas import JudgeResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic schemas for structured output per turn
# ---------------------------------------------------------------------------


class DecompositionOutput(BaseModel):
    """Turn 1: Decomposed atomic claims from the test response."""

    test_facts: list[str] = Field(
        description="List of atomic factual claims extracted from the test response."
    )


class CompletenessItem(BaseModel):
    """Single golden fact completeness check."""

    status: str = Field(description="PRESENT or MISSING")
    matched_by: str | None = Field(
        default="",
        description="Which test fact addresses this golden fact (if PRESENT).",
    )
    explanation: str | None = Field(
        default="",
        description="Why this golden fact is missing (if MISSING).",
    )


class CompletenessOutput(BaseModel):
    """Turn 2: Completeness check results."""

    results: dict[str, CompletenessItem] = Field(
        description="Per golden-fact completeness verdict."
    )


class FaithfulnessItem(BaseModel):
    """Single unmatched claim classification."""

    status: str = Field(description="ACCEPTABLE or CONTRADICTS")
    contradicts: str = Field(
        default="",
        description="What golden fact or source data it contradicts (if CONTRADICTS).",
    )
    explanation: str = Field(default="", description="Brief explanation.")

    @classmethod
    def from_flexible(cls, value):
        """Accept string shorthand ('ACCEPTABLE') or full dict."""
        if isinstance(value, str):
            return cls(status=value)
        if isinstance(value, dict):
            return cls(**value)
        return value


class ToolCorrelation(BaseModel):
    """Correlation between a missing fact and an empty tool result."""

    golden_fact: str = Field(description="The missing golden fact.")
    tool_name: str = Field(description="Tool that returned empty results.")
    explanation: str = Field(default="", description="Why this correlation is likely.")


class FaithfulnessOutput(BaseModel):
    """Turn 3: Faithfulness + tool correlation results."""

    claims: dict[str, Any] = Field(
        default_factory=dict,
        description="Per unmatched-claim faithfulness classification. Values are ACCEPTABLE or CONTRADICTS (string or object).",
    )
    tool_correlations: list[ToolCorrelation] = Field(
        default_factory=list,
        description="Missing facts that correlate with empty tool results.",
    )


class VerdictOutput(BaseModel):
    """Turn 4: Final verdict."""

    verdict: str = Field(description="PASS or FAIL")
    reason: str = Field(description="One-sentence explanation citing specifics.")
    missing_count: int = Field(default=0, description="Number of missing golden facts.")
    contradiction_count: int = Field(
        default=0, description="Number of contradicting claims."
    )
    tool_correlation_summary: str = Field(
        default="",
        description="Summary of tool call correlations with failures.",
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a factual accuracy judge evaluating an AI legal assistant's responses.

Your task: determine whether the agent's response covers the same factual \
content as a human-blessed golden answer. You evaluate in structured steps.

## Matching Standard

A golden fact is PRESENT if the test response conveys the same information \
through any of these means:
- Direct statement
- Paraphrase ("Sept 15" = "September 15, 2025")
- Synonym ("documented" = "filed" = "sent"; "scheduled" = "set")
- Medical equivalence ("cervical strain" = "whiplash"; "WAD Grade II" = \
"cervical strain with whiplash"; ICD codes = their named diagnoses;
"neck pain" = "cervical strain"; "lower back pain" = "lumbar strain")
- Logical implication (stating all premises = stating the conclusion)
- Greater specificity (listing individual symptoms subsumes "symptoms \
documented across visits")
- Coreference resolution: If the golden fact names an entity (e.g., \
"John Smith") and the test response refers to the same entity using \
a pronoun, title, or role (e.g., "the defendant", "the at-fault driver", \
"the other driver"), it counts as a MATCH if the context makes the \
reference unambiguous. The test response does not need to repeat a proper \
name if the entity is clearly identified by context.
- Role equivalence: "the defendant" = the named defendant; "the plaintiff" = \
the named client; "the insurer" = the named insurance company.

A golden fact is MISSING only if the test response does not address it \
by ANY of the above means.

A test fact CONTRADICTS only if it asserts a numerically or factually \
different value (e.g., "$8,445" vs "$9,200"; "no pre-existing" vs \
"degenerative changes found"). Differences in level of detail (e.g., \
"neck pain" vs "cervical strain at C4-C5") are NOT contradictions — the \
less specific version is simply less detailed.

## Key Principle

Extra detail in the test response is NEVER a failure. Only missing or \
contradicted golden facts cause failure.
"""

# Appended to the system prompt at evaluation time so the judge can reason about
# time-relative claims (SOL deadlines, past/future events). Without this the
# judge has no notion of "now" and cannot verify statements like "the deadline
# has not passed" or "the vacation has since ended".
DATE_CONTEXT_PROMPT = """

## Current Date Context

The current date is {current_date}. Use it whenever a golden fact or the test \
response makes a time-relative claim:
- A statute-of-limitations (or other) deadline that falls AFTER the current \
date has NOT expired; a deadline on or before it has passed.
- An event dated before the current date is in the past; one dated after it \
is in the future.
- Treat past-tense phrasing about a dated event ("was on vacation", "has \
returned", "has since passed") as consistent with that event being in the \
past when its date precedes the current date, and vice versa.

Apply this only to resolve time-relative claims; it does not change the \
matching standard for non-temporal facts.
"""

TURN1_PROMPT = """\
Here is the context for this evaluation:

QUESTION ASKED:
{question}

GOLDEN ANSWER (human-blessed correct response):
{golden_answer}

GOLDEN FACTS (required atomic facts from the golden answer):
{golden_facts_formatted}

TEST RESPONSE (what the agent actually produced):
{test_response}

TOOL CALLS MADE BY THE AGENT:
{tool_calls_formatted}

---

STEP 1: Decompose the test response into atomic factual claims.
Extract every distinct factual assertion. Do not include opinions,
hedging language, or structural elements (greetings, transitions).
Return ONLY the list of factual claims.
"""

TURN2_PROMPT = """\
STEP 2: Completeness check.

For EACH golden fact below, determine: does the test response address this \
fact? Apply the FULL matching standard from your instructions (paraphrase, \
synonym, medical equivalence, logical implication, greater specificity, \
coreference resolution, and role equivalence ALL count as valid matches).

Mark each fact:
- PRESENT — if any test fact from Step 1 conveys the same information \
through ANY valid matching method
- MISSING — if no test fact addresses it by any valid matching method

IMPORTANT: Be generous with matching. If the test response addresses the \
spirit and substance of the golden fact even with different wording, it is \
PRESENT. Only mark MISSING if the information is genuinely absent.
{eval_guidance}
Golden facts to check:
{golden_facts_formatted}

For each, return: status, which test fact matched (if PRESENT), or a brief \
explanation of why it's missing (if MISSING).
"""

TURN3_PROMPT = """\
STEP 3: Faithfulness + Tool Correlation.

Part A — Unmatched test facts: For each test fact from Step 1 that was NOT \
matched to any golden fact in Step 2, classify it:
- ACCEPTABLE: relevant extra detail, no conflict with golden answer
- CONTRADICTS: asserts something that directly conflicts with the golden answer

Part B — Retrieval correlation: For each MISSING golden fact from Step 2, \
check if it correlates with a tool call that returned 0 results. This \
distinguishes "agent didn't synthesize available data" from "data wasn't \
retrieved."

Tool calls:
{tool_calls_formatted}
"""

TURN4_PROMPT = """\
STEP 4: Final Verdict.
{eval_guidance_verdict}
Rules:
- ANY test fact CONTRADICTS → FAIL
- All golden facts PRESENT and no contradictions → PASS
{missing_fact_rule}
Emit verdict with a one-sentence reason citing the specific missing or \
contradicting facts. If there are tool retrieval correlations, note them.
"""


# ---------------------------------------------------------------------------
# Judge implementation
# ---------------------------------------------------------------------------


class GoldenAnswerJudge:
    """Multi-turn conversational judge using OpenAI Chat Completions.

    Evaluates a test response against golden facts through 4 structured
    turns, maintaining conversation context by accumulating messages.
    OpenAI's automatic prompt caching gives ~90% input token discount
    on the repeated prefix across turns.

    Each turn produces structured output that maps to an Allure rubric
    section, enabling step-by-step auditability.
    """

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        evaluation_model: Optional[str] = None,
        evaluation_date: Optional[str] = None,
    ):
        """Initialize the judge.

        Args:
            model: OpenAI model for Turn 1 (decomposition). Defaults to
                settings.OPENAI_MODEL_SMALL (gpt-4.1-mini).
            evaluation_model: OpenAI model for Turns 2-4 (completeness,
                faithfulness, verdict). Defaults to settings.OPENAI_MODEL_LARGE
                (gpt-4.1) for better semantic understanding of coreference,
                medical equivalence, and logical implication.
            evaluation_date: ISO date (YYYY-MM-DD) supplied to the judge as the
                "current date" for temporal reasoning (SOL deadlines, past/future
                events). Defaults to today; injectable for deterministic
                tests.
        """
        from django.conf import settings

        self.model = model or settings.OPENAI_MODEL_SMALL
        self.evaluation_model = evaluation_model or settings.OPENAI_MODEL_LARGE
        self.evaluation_date = evaluation_date

    async def evaluate(
        self,
        *,
        golden_facts: list[str],
        test_response: str,
        question: str,
        tool_calls: list[dict[str, Any]] | None = None,
        golden_answer: str = "",
        eval_guidance: str = "",
    ) -> JudgeResult:
        """Run the 4-turn evaluation conversation.

        Args:
            golden_facts: Pre-decomposed atomic facts from the golden answer.
            test_response: The agent's actual response to evaluate.
            question: The question that was asked.
            tool_calls: Tool call records for the question (from _TOOL_CALL_STATE).
            golden_answer: The full golden answer prose (for context in Turn 1).
            eval_guidance: Optional per-question evaluation guidance injected
                into the completeness check. Used for subjective/strategic
                questions where 1-to-1 fact mapping is inappropriate.

        Returns:
            JudgeResult with:
              - is_valid: True if PASS, False if FAIL
              - score: 1.0 for PASS, 0.0 for FAIL (binary)
              - reasoning: The verdict reason from Turn 4
              - metadata["rubric_steps"]: Full per-turn output for Allure
              - metadata["verdict"]: The VerdictOutput dict
        """
        start = time.perf_counter()
        rubric_steps: list[dict[str, Any]] = []

        golden_facts_formatted = "\n".join(
            f"  {i+1}. {fact}" for i, fact in enumerate(golden_facts)
        )
        tool_calls_formatted = self._format_tool_calls(tool_calls or [])

        # Inject the current date so the judge can reason about time-relative
        # claims (SOL deadlines, past/future events). Defaults to today.
        if self.evaluation_date:
            current_date = self.evaluation_date
        else:
            from django.utils import timezone

            current_date = timezone.now().date().isoformat()
        system_content = SYSTEM_PROMPT + DATE_CONTEXT_PROMPT.format(
            current_date=current_date
        )

        # Conversation state: accumulate messages across turns
        conversation: list[dict[str, str]] = [
            {"role": "system", "content": system_content},
        ]

        try:
            # Turn 1: Decomposition
            turn1_input = TURN1_PROMPT.format(
                question=question,
                golden_answer=golden_answer or "(see golden facts above)",
                golden_facts_formatted=golden_facts_formatted,
                test_response=test_response,
                tool_calls_formatted=tool_calls_formatted,
            )

            decomposition = await self._call_turn(
                conversation=conversation,
                user_message=turn1_input,
                schema=DecompositionOutput,
            )
            rubric_steps.append(
                {
                    "step": 1,
                    "name": "Test Facts Decomposition",
                    "output": decomposition.model_dump(),
                }
            )

            # Turn 2: Completeness (uses larger model for semantic matching)
            eval_guidance_block = ""
            if eval_guidance:
                eval_guidance_block = (
                    f"\n## Evaluation Guidance for This Question\n" f"{eval_guidance}\n"
                )
            turn2_input = TURN2_PROMPT.format(
                golden_facts_formatted=golden_facts_formatted,
                eval_guidance=eval_guidance_block,
            )
            completeness = await self._call_turn(
                conversation=conversation,
                user_message=turn2_input,
                schema=CompletenessOutput,
                model_override=self.evaluation_model,
            )
            rubric_steps.append(
                {
                    "step": 2,
                    "name": "Completeness (Recall)",
                    "output": completeness.model_dump(),
                }
            )

            # Turn 3: Faithfulness + Tool Correlation (uses larger model)
            turn3_input = TURN3_PROMPT.format(
                tool_calls_formatted=tool_calls_formatted,
            )
            faithfulness = await self._call_turn(
                conversation=conversation,
                user_message=turn3_input,
                schema=FaithfulnessOutput,
                model_override=self.evaluation_model,
            )
            rubric_steps.append(
                {
                    "step": 3,
                    "name": "Faithfulness + Tool Correlation",
                    "output": faithfulness.model_dump(),
                }
            )

            # Turn 4: Verdict (uses larger model)
            # For subjective questions with eval_guidance, relax the
            # strict "any missing = fail" rule to allow partial recall
            if eval_guidance:
                missing_fact_rule = (
                    "- If MOST golden facts are PRESENT (≥60% recall) and the "
                    "response provides a valid, well-reasoned analysis grounded "
                    "in case facts → PASS (subjective question allowance)\n"
                    "- If golden facts are MOSTLY MISSING (<60% recall) or the "
                    "response lacks substantive analysis → FAIL"
                )
                eval_guidance_verdict = (
                    "\nNOTE: This is a subjective/strategic analysis question. "
                    "The agent's response may use different reasoning paths "
                    "than the golden answer. Apply lenient matching — a valid "
                    "alternative analysis satisfies the intent of the golden facts.\n"
                )
            else:
                missing_fact_rule = "- ANY golden fact MISSING → FAIL"
                eval_guidance_verdict = ""

            turn4_input = TURN4_PROMPT.format(
                eval_guidance_verdict=eval_guidance_verdict,
                missing_fact_rule=missing_fact_rule,
            )
            verdict = await self._call_turn(
                conversation=conversation,
                user_message=turn4_input,
                schema=VerdictOutput,
                model_override=self.evaluation_model,
            )
            rubric_steps.append(
                {
                    "step": 4,
                    "name": "Final Verdict",
                    "output": verdict.model_dump(),
                }
            )

        except Exception as e:
            logger.error("GoldenAnswerJudge failed: %s", e)
            duration_ms = (time.perf_counter() - start) * 1000
            return JudgeResult(
                is_valid=False,
                score=0.0,
                reasoning=f"Judge error: {e}",
                judge_name="GoldenAnswerJudge",
                evaluation_duration_ms=duration_ms,
                metadata={"rubric_steps": rubric_steps, "error": str(e)},
            )

        duration_ms = (time.perf_counter() - start) * 1000
        is_pass = verdict.verdict.upper() == "PASS"

        return JudgeResult(
            is_valid=is_pass,
            score=1.0 if is_pass else 0.0,
            reasoning=verdict.reason,
            judge_name="GoldenAnswerJudge",
            evaluation_duration_ms=duration_ms,
            metadata={
                "rubric_steps": rubric_steps,
                "verdict": verdict.model_dump(),
                "completeness": completeness.model_dump(),
                "faithfulness": faithfulness.model_dump(),
                "decomposition": decomposition.model_dump(),
            },
        )

    async def _call_turn(
        self,
        *,
        conversation: list[dict[str, str]],
        user_message: str,
        schema: type[BaseModel],
        model_override: Optional[str] = None,
    ) -> BaseModel:
        """Execute one turn of the conversation via Chat Completions API.

        Appends the user message to the conversation, calls the LLM with
        structured output, appends the assistant response, and returns
        the parsed result. The conversation list is mutated in place so
        subsequent calls carry the full history.

        OpenAI's automatic prompt caching gives ~90% discount on the
        repeated message prefix across turns 2-4.

        Args:
            conversation: The accumulated message list (mutated in place).
            user_message: The user prompt for this turn.
            schema: Pydantic model for structured output parsing.
            model_override: Optional model to use instead of self.model.
                Used for turns 2-4 which benefit from a larger model.

        Returns:
            Parsed schema instance.
        """
        from openai import AsyncOpenAI

        client = AsyncOpenAI()

        # Add user message to conversation
        conversation.append({"role": "user", "content": user_message})

        response = await client.chat.completions.create(
            model=model_override or self.model,
            messages=conversation,
            temperature=0.0,
            max_tokens=4096,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": schema.model_json_schema(),
                    "strict": False,
                },
            },
        )

        # Extract and parse the structured output
        output_text = response.choices[0].message.content
        parsed = schema.model_validate_json(output_text)

        # Append assistant response to conversation for context continuity
        conversation.append({"role": "assistant", "content": output_text})

        return parsed

    def _format_tool_calls(self, tool_calls: list[dict[str, Any]]) -> str:
        """Format tool call records for the judge prompt."""
        if not tool_calls:
            return "(no tool calls were made)"

        lines = []
        for call in tool_calls:
            tool_name = call.get("tool", "unknown")
            result_count = call.get("result_count", "?")
            error = call.get("error")
            latency = call.get("latency_ms")

            line = f"- {tool_name}: {result_count} results"
            if latency is not None:
                line += f" ({latency:.0f}ms)"
            if error:
                line += f" [ERROR: {error}]"
            if result_count == 0:
                line += " ⚠️ EMPTY"
            lines.append(line)

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Golden facts cache management
# ---------------------------------------------------------------------------


def compute_golden_facts_hash(golden_answers: list[str]) -> str:
    """Compute the cache invalidation key for golden answers.

    Used by the harness to detect when golden answers have changed
    and the cached golden_facts need re-decomposition.
    """
    content = "\n---\n".join(golden_answers)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


async def decompose_golden_answer(
    golden_answer: str,
    *,
    model: Optional[str] = None,
) -> list[str]:
    """Decompose a golden answer into atomic facts for caching.

    Called once per golden answer on first evaluation run, then cached.
    Uses the same model as the judge for consistency.

    Args:
        golden_answer: The blessed golden answer prose.
        model: OpenAI model to use. Defaults to settings.OPENAI_MODEL_SMALL.

    Returns:
        List of atomic factual claims.
    """
    from django.conf import settings
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    _model = model or settings.OPENAI_MODEL_SMALL

    prompt = f"""\
Extract the atomic factual assertions from this golden answer. Each fact \
should be independently verifiable.

Guidelines:
- One assertion per fact. "X happened on date D" = 1 fact (not 2).
- "No X has been done/found" = 1 fact (not split into existence + action).
- Synonyms in a clause are 1 fact: "set or scheduled" = one assertion.
- Split only at genuine conjunctions of independent claims.
- Short answers (1-2 sentences) → typically 1-3 facts.
- Do NOT add facts not explicitly stated in the text.

GOLDEN ANSWER:
{golden_answer}
"""

    response = await client.chat.completions.create(
        model=_model,
        messages=[
            {
                "role": "system",
                "content": "You extract atomic factual claims from text.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=4096,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "DecompositionOutput",
                "schema": DecompositionOutput.model_json_schema(),
                "strict": False,
            },
        },
    )

    parsed = DecompositionOutput.model_validate_json(
        response.choices[0].message.content
    )
    return parsed.test_facts
