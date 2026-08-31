"""
BaseJudge — configurable LLM-as-a-Judge foundation.

A working, configurable judge class that can be used directly with a
prompt template + Pydantic evaluation schema, or subclassed for custom
deterministic logic (e.g., CypherJudge).

Most future judge use cases just need a new prompt + schema combination
passed to the constructor — no subclass required.

Usage:
    from graph.judges.base import BaseJudge
    from graph.judges.prompts import GRAPH_DATA_QUALITY_PROMPT
    from graph.judges.schemas import GraphDataQualityEvaluation

    judge = BaseJudge(
        name="GraphDataQualityJudge",
        prompt_template=GRAPH_DATA_QUALITY_PROMPT,
        evaluation_schema=GraphDataQualityEvaluation,
    )
    result = await judge.evaluate(entity_type="Treatment", ...)
"""

from __future__ import annotations

import logging
import time
from typing import Any, ClassVar, Optional, Type

from pydantic import BaseModel

from graph.judges.schemas import EvaluationResult, JudgeResult, MetricName

logger = logging.getLogger(__name__)


class BaseJudge:
    """Configurable LLM judge.

    Usable directly with a prompt + schema, or subclassed for custom logic.
    The evaluate() method formats the prompt with context kwargs, calls the
    LLM via invoke_structured_llm(), and wraps the result in a JudgeResult.

    Subclasses can override evaluate() to add deterministic checks before
    or instead of the LLM call (see CypherJudge).
    """

    def __init__(
        self,
        name: str,
        prompt_template: str,
        evaluation_schema: Type[BaseModel],
        *,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ):
        self.name = name
        self.prompt_template = prompt_template
        self.evaluation_schema = evaluation_schema
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    async def evaluate(self, **context: Any) -> JudgeResult:
        """Format prompt with context, call LLM, wrap in JudgeResult.

        Override in subclasses to add deterministic checks or conditional
        LLM invocation.

        Args:
            **context: Template variables for the prompt string.

        Returns:
            JudgeResult with evaluation outcome.
        """
        start = time.perf_counter()

        try:
            prompt = self.prompt_template.format(**context)
        except KeyError as exc:
            raise ValueError(
                f"Judge '{self.name}' prompt template missing variable: {exc}. "
                f"Provided keys: {sorted(context.keys())}"
            ) from exc
        llm_result = await self._invoke_llm(prompt)
        result = self._to_judge_result(llm_result)

        result.evaluation_duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "Judge %s evaluated in %.1fms: is_valid=%s score=%.2f",
            self.name,
            result.evaluation_duration_ms,
            result.is_valid,
            result.score,
        )
        return result

    async def invoke_llm(
        self, prompt: str, schema: Optional[Type[BaseModel]] = None
    ) -> BaseModel:
        """Call LLM with structured output — public API.

        Use this when you need the raw typed evaluation without the
        ``JudgeResult`` wrapper (e.g. ``RoleDetectionHandler``).

        Delegates to ``_invoke_llm`` which is the single implementation
        shared with ``evaluate()``.
        """
        return await self._invoke_llm(prompt, schema=schema)

    async def _invoke_llm(
        self, prompt: str, schema: Optional[Type[BaseModel]] = None
    ) -> BaseModel:
        """Call LLM with structured output. Isolated for easy mocking in tests.

        Args:
            prompt: The formatted prompt string.
            schema: Override the schema for this call. Defaults to
                self.evaluation_schema. Use this when a single judge needs to
                call the LLM with a different output schema (e.g. CypherJudge
                calling _repair_query with CypherRepairResult vs the normal
                CypherLLMEvaluation schema).
        """
        from core.langchain_throttle import invoke_structured_llm

        target_schema = schema or self.evaluation_schema

        # Auto-detect schemas with dict fields — OpenAI strict mode doesn't
        # support dict types, so we fall back to function_calling method.
        method = None
        for field_info in target_schema.model_fields.values():
            origin = getattr(field_info.annotation, "__origin__", None)
            if origin is dict:
                method = "function_calling"
                break

        # Pass prompt_cache_key for routing affinity so all judge
        # calls within a harness run route to the same backend, maximizing
        # prefix cache hits across questions (2 of 3 judges exceed 1024 tokens).
        return await invoke_structured_llm(
            target_schema,
            prompt,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            method=method,
            user="harness-judge",
        )

    def _to_judge_result(self, llm_result: BaseModel) -> JudgeResult:
        """Convert structured LLM output to standardized JudgeResult.

        Prefers an explicit .to_judge_result() method on the schema (type-safe,
        no magic string lookups). Falls back to duck-typing for schemas that
        don't implement the protocol — ensures forward compatibility with
        third-party or future schemas added before they adopt the protocol.
        """
        if hasattr(llm_result, "to_judge_result"):
            return llm_result.to_judge_result(self.name)

        # Fallback duck-typing path — only reached for schemas that don't
        # implement to_judge_result(). New schemas should implement the protocol.
        logger.warning(
            "Judge '%s' schema %s does not implement to_judge_result() — "
            "falling back to duck-typing. Add to_judge_result() to this schema.",
            self.name,
            type(llm_result).__name__,
        )
        result_dict = llm_result.model_dump()
        is_valid = True
        if "semantic_match" in result_dict:
            is_valid = result_dict["semantic_match"]
        elif "should_keep" in result_dict:
            is_valid = result_dict["should_keep"]
        score = result_dict.get(
            "confidence", result_dict.get("quality_score", 1.0 if is_valid else 0.0)
        )
        reasoning = result_dict.get(
            "semantic_reasoning", result_dict.get("reasoning", "")
        )
        return JudgeResult(
            is_valid=is_valid,
            score=score,
            reasoning=reasoning,
            judge_name=self.name,
        )


class LockedRubricJudge(BaseJudge):
    """Base class for judges that emit the locked EvaluationResult schema.

    Provides the shared ``_to_judge_result`` implementation that all
    locked-rubric judges need: a defensive metric-tag check (so the
    harness fails loud if the LLM returns the wrong tag) and a
    threshold-based ``is_valid`` override.

    Concrete subclasses declare ``EXPECTED_METRIC`` as a ClassVar and
    pass their threshold to ``__init__``.
    """

    EXPECTED_METRIC: ClassVar[MetricName]

    def __init__(self, *, threshold: float, **kwargs: Any):
        super().__init__(**kwargs)
        self.threshold = threshold

    def _to_judge_result(self, llm_result: BaseModel) -> JudgeResult:
        if (
            isinstance(llm_result, EvaluationResult)
            and llm_result.metric != self.EXPECTED_METRIC
        ):
            raise ValueError(
                f"{self.name} expected metric={self.EXPECTED_METRIC!r}, "
                f"LLM returned metric={llm_result.metric!r}"
            )
        result = super()._to_judge_result(llm_result)
        result.is_valid = result.score >= self.threshold
        return result
