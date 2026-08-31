"""
LLM-as-a-Judge framework for validating system outputs.

Provides a configurable BaseJudge that can be used directly with a
prompt + schema, or subclassed for custom deterministic logic.

Concrete judges (locked rubric):
- ChunkRelevanceJudge: Filters retrieved chunks by semantic relevance.
- FaithfulnessJudge: Verifies response claims are grounded in source channels.
- AnswerRelevancyJudge: Scores response against authored ground truth.
- ToolFulfillmentJudge: Scores tool-call pattern against question intent.
- Graph Data Quality Judge: BaseJudge instance for entity quality evaluation.

NOTE (V2 cutover): the legacy ``CypherJudge`` (Phase-1
deterministic checks + Phase-2/3 LLM repair) was deleted; its
deterministic checks moved to
``core.services.cypher_pipeline.linter.CypherLinter`` and its LLM
repair moved to ``core.services.cypher_pipeline.repair.CypherRepair``.
The ``GraphSchemaContext`` Pydantic schema stays here (it's reused by
the new linter and pipeline).

Usage:
    from graph.judges.base import BaseJudge
    from graph.judges.schemas import GraphSchemaContext, JudgeResult
"""

from graph.judges.answer_relevancy_judge import AnswerRelevancyJudge
from graph.judges.base import BaseJudge, LockedRubricJudge
from graph.judges.chunk_relevance_judge import ChunkRelevanceJudge
from graph.judges.faithfulness_judge import FaithfulnessJudge
from graph.judges.golden_answer_judge import GoldenAnswerJudge
from graph.judges.schemas import (
    VIOLATION_CATEGORIES,
    ChunkRelevance,
    ChunkRelevanceEvaluation,
    CypherLLMEvaluation,
    CypherRepairResult,
    EvaluationResult,
    GraphDataQualityEvaluation,
    GraphSchemaContext,
    JudgeIssue,
    JudgeResult,
    MetricName,
)
from graph.judges.tool_fulfillment_judge import ToolFulfillmentJudge

__all__ = [
    "AnswerRelevancyJudge",
    "BaseJudge",
    "ChunkRelevanceJudge",
    "GoldenAnswerJudge",
    "LockedRubricJudge",
    "FaithfulnessJudge",
    "ToolFulfillmentJudge",
    "VIOLATION_CATEGORIES",
    "ChunkRelevance",
    "ChunkRelevanceEvaluation",
    "CypherLLMEvaluation",
    "CypherRepairResult",
    "EvaluationResult",
    "GraphDataQualityEvaluation",
    "GraphSchemaContext",
    "JudgeIssue",
    "JudgeResult",
    "MetricName",
]
