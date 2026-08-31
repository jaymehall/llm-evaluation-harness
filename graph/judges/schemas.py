"""
Pydantic schemas for the LLM-as-a-Judge framework.

Provides standardized input/output models for all judge evaluations:
- JudgeIssue / JudgeResult: Universal output for any judge
- EvaluationResult: Locked rubric output — every harness judge emits this
- CypherEvaluateInput / CypherLLMEvaluation: CypherJudge-specific
- GraphSchemaContext: Schema context for validation
- GraphDataQualityEvaluation: Data quality judge output
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

# =============================================================================
# LOCKED EVALUATION RUBRIC
# =============================================================================

# The closed set of metric names. New metrics require a rubric-update ticket
# (see docs/architecture/evaluation-rubric.md §8).
MetricName = Literal["faithfulness", "answer_relevancy", "tool_fulfillment"]

# Closed-set vocabulary for violating_sentences inline prefixes. Judges
# instruct the LLM to emit one of these at the start of each violation, e.g.
# "[MISSING] September 15, 2025". The schema does NOT enforce the prefix
# (analysis §4.2) — downstream grep/UI workflows treat it as best-effort.
VIOLATION_CATEGORIES = frozenset(
    {
        "MISSING",  # Answer Relevancy: expected item not conveyed
        "WRONG",  # Answer Relevancy: expected_facts contradicted
        "FORBIDDEN",  # Answer Relevancy: expected_not_contains present
        "UNGROUNDED",  # Faithfulness: claim not traceable to source
        "OFF_TOPIC",  # Tool Fulfillment: irrelevant tool invocation
        "WRONG_TOOL",  # Tool Fulfillment: called wrong tool for domain
        "MISSING_TOOL",  # Tool Fulfillment: no tool called when data retrieval needed
    }
)

# Hard cap on rationale character length, enforced by Field(max_length=...).
# The 2-sentence trim is a separate post-LLM validator below.
RATIONALE_MAX_CHARS = 400


class EvaluationResult(BaseModel):
    """Locked rubric output produced by every harness judge.

    Every judge's LLM call returns this exact shape. The harness persists
    it byte-identically across runs (only values differ), and the
    lexicon-violation guard test re-parses every baseline JSON through
    this schema.

    See docs/architecture/evaluation-rubric.md for the locked metric
    definitions, scoring rules, and violation vocabulary.
    """

    model_config = ConfigDict(extra="forbid")

    metric: MetricName = Field(
        ...,
        description="The locked rubric metric this evaluation measures.",
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="0.0 (worst) to 1.0 (best). Threshold is per-judge.",
    )
    violating_sentences: list[str] = Field(
        default_factory=list,
        description=(
            "Flat list of violations, each prefixed with one of "
            "VIOLATION_CATEGORIES at start, e.g. '[MISSING] September 15, "
            "2025'. Empty list means no violations found."
        ),
    )
    rationale: str = Field(
        ...,
        max_length=RATIONALE_MAX_CHARS,
        description=(
            "Two-sentence summary. Post-LLM-trimmed by the validator below "
            "if the LLM emitted more than two sentences."
        ),
    )

    @model_validator(mode="after")
    def _trim_rationale_to_two_sentences(self) -> EvaluationResult:
        """Hard-cap rationale at 2 sentences, even if LLM ignored the prompt.

        Counts terminators (., !, ?). Rationales with more than two
        terminators are truncated to end-of-second-sentence. Rationales
        with two or fewer terminators are left untouched. The 400-char
        cap is enforced separately by Field(max_length=...).

        Walks character-by-character (rather than splitting on regex) so
        "Mr." stays a single sentence — naive split would break on every
        period.
        """
        text = self.rationale.strip()
        terminators_found = 0
        cut = len(text)
        for i, ch in enumerate(text):
            if ch in ".!?":
                terminators_found += 1
                if terminators_found == 2:
                    cut = i + 1
                    break
        if terminators_found > 2 or (terminators_found == 2 and cut < len(text)):
            self.rationale = text[:cut].rstrip()
        return self

    def to_judge_result(self, judge_name: str) -> JudgeResult:
        """Wrap the structured eval into the universal JudgeResult envelope.

        is_valid is left as True; concrete judges override _to_judge_result
        to apply their threshold (same pattern as the legacy judges).
        Metadata carries the metric name, the flat violation list, and a
        full dump of this record so downstream consumers (rollup, guard
        test) can read the rubric output without round-tripping JSON.
        """
        return JudgeResult(
            is_valid=True,  # overridden by judge threshold
            score=self.score,
            reasoning=self.rationale,
            judge_name=judge_name,
            metadata={
                "metric": self.metric,
                "violating_sentences": list(self.violating_sentences),
                "evaluation_result": self.model_dump(),
            },
        )


# =============================================================================
# UNIVERSAL JUDGE OUTPUT
# =============================================================================


class JudgeIssue(BaseModel):
    """Individual issue found during evaluation."""

    category: str = Field(
        ...,
        description=(
            "Issue category: schema_compliance, read_only, semantic, "
            "performance, syntax, data_quality"
        ),
    )
    severity: str = Field(
        ...,
        description="Issue severity: error, warning, info",
    )
    message: str = Field(..., description="Human-readable issue description")
    suggestion: Optional[str] = Field(None, description="Suggested fix or improvement")


class JudgeResult(BaseModel):
    """Standardized output from any Judge evaluation."""

    is_valid: bool = Field(..., description="Whether the evaluated input passes")
    score: float = Field(
        ..., ge=0.0, le=1.0, description="Quality score (0.0 = worst, 1.0 = best)"
    )
    reasoning: str = Field(..., description="Human-readable evaluation summary")
    issues: list[JudgeIssue] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    judge_name: str = Field(
        ..., description="Name of the judge that produced this result"
    )
    corrected_query: Optional[str] = Field(
        None,
        description="Self-healed/corrected version of the input",
    )
    evaluation_duration_ms: float = Field(
        0, description="Total evaluation time in milliseconds"
    )
    token_usage: Optional[dict[str, int]] = Field(
        None,
        description=(
            "LLM token usage: prompt_tokens, completion_tokens. "
            "Not yet populated — reserved for future observability work."
        ),
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extensible metadata — future judges add domain-specific data here",
    )


# =============================================================================
# CYPHER JUDGE SCHEMAS
# =============================================================================


class PropertyDescriptor(BaseModel):
    """Type and sample-value metadata for a single graph property.

    Populated by ``schema_context_builder._merge`` from the
    ``CRMSchemaRegistry``; consumed by the cypher-pipeline prompt builder
    (to render ``(number)`` / ``(string)`` annotations and safe sample
    values) and by the linter (to recognise numeric operands for
    SUM/AVG/MIN/MAX even when Neo4j storage is String).
    """

    types: list[str] = Field(
        default_factory=list,
        description=(
            "Type names observed for this property — drawn from the schema "
            "registry's ``data_types`` set (``str``, ``float``, ``int``, "
            "``bool``, ``date``, ...). Numeric-named String fields that "
            "parse as currency/numbers will carry both ``str`` and ``float``."
        ),
    )
    sample_values: list[Any] = Field(
        default_factory=list,
        description=(
            "Up to 3 sample values, included only for fields on the "
            "numeric-name allow-list (PHI safety: never set for "
            "name/email/address/id-typed fields)."
        ),
    )
    is_complex: bool = Field(
        False,
        description=(
            "True when the property stores a Map or List value (subscriptable "
            "with `prop[key]` or `prop[idx]`). False for scalar storage "
            "(String, Number, Boolean, Date). Drives the linter's "
            "SCHEMA_INVALID_SUBSCRIPT rule. Populated from live "
            "Neo4j discovery via `apoc.meta.nodeTypeProperties` or the "
            "fallback `valueType()` query."
        ),
    )


class GraphSchemaContext(BaseModel):
    """Graph schema context provided to CypherJudge for validation."""

    node_labels: list[str] = Field(..., description="Valid node labels in the graph")
    relationship_types: list[str] = Field(
        ..., description="Valid relationship types in the graph"
    )
    node_properties: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Map of node label -> list of property names",
    )
    node_property_types: dict[str, dict[str, PropertyDescriptor]] = Field(
        default_factory=dict,
        description=(
            "Map of node label -> property name -> PropertyDescriptor. "
            "Carries type and (PHI-gated) sample-value info for the "
            "cypher-pipeline prompt builder and linter. Optional — the "
            "older ``node_properties`` field stays as the canonical name "
            "list for backward compat."
        ),
    )
    organization_id: str = Field(
        ..., description="Organization scope for multi-tenancy"
    )
    schema_version: int = Field(
        0,
        description=(
            "Monotonically-increasing version stamped by "
            "`OrganizationKnowledgeService.register_graph_schema`. Used as "
            "a cache key suffix for prompt caches and few-shot filtering "
            "so caches turn over automatically when the schema changes."
        ),
    )
    built_at: Optional[str] = Field(
        None,
        description=(
            "ISO-8601 timestamp of when this context was built (from live "
            "Neo4j discovery + registry merge). Surfaced in pipeline "
            "telemetry so we can alarm on stale-schema queries."
        ),
    )
    node_count: int = Field(
        0,
        description=(
            "Total org-scoped node count captured at build time. The "
            "pipeline compares this against a cheap `count(n)` query to "
            "detect schema drift and trigger a background refresh."
        ),
    )
    crm_relationship_types: list[str] = Field(
        default_factory=list,
        description=(
            "CRM-specific relationship types for this organization, "
            "sourced from the CRM Schema Registry. These are the typed "
            "relationships (e.g., HAS_LIEN__C, HAS_FILE_NOTES__C) that "
            "connect (:Node) to (:DataEntity/:Entity) child records. "
            "Empty for docs-only orgs. Per-org, not global."
        ),
    )


class CypherEvaluateInput(BaseModel):
    """Input for CypherJudge.evaluate()."""

    cypher_query: str = Field(..., description="Generated Cypher query to evaluate")
    natural_language_query: str = Field(
        ..., description="Original natural language query from user"
    )
    schema_context: GraphSchemaContext = Field(
        ..., description="Graph schema for validation"
    )
    allow_mutations: bool = Field(
        False, description="Whether mutation operations are permitted"
    )


class CypherLLMEvaluation(BaseModel):
    """Structured output schema for LLM semantic evaluation of Cypher queries."""

    semantic_match: bool = Field(
        ..., description="Whether the Cypher accurately implements the NL intent"
    )
    semantic_reasoning: str = Field(..., description="Explanation of semantic analysis")
    performance_issues: list[str] = Field(
        default_factory=list,
        description=(
            "Identified performance concerns "
            "(Cartesian products, missing anchors, etc.)"
        ),
    )
    logic_issues: list[str] = Field(
        default_factory=list,
        description="Logic errors or incorrect relationship traversals",
    )
    suggestions: list[str] = Field(
        default_factory=list, description="Suggested improvements to the query"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in the evaluation"
    )

    def to_judge_result(self, judge_name: str) -> "JudgeResult":
        issues = [
            JudgeIssue(category="performance", severity="warning", message=m)
            for m in self.performance_issues
        ] + [
            JudgeIssue(category="logic", severity="warning", message=m)
            for m in self.logic_issues
        ]
        return JudgeResult(
            is_valid=self.semantic_match,
            score=self.confidence,
            reasoning=self.semantic_reasoning,
            issues=issues,
            suggestions=self.suggestions,
            judge_name=judge_name,
        )


class CypherRepairResult(BaseModel):
    """Structured output schema for LLM Cypher query repair."""

    corrected_query: str = Field(..., description="The corrected Cypher query")
    changes_made: list[str] = Field(
        default_factory=list,
        description="List of changes made to fix the query",
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in the repair"
    )
    reasoning: str = Field(
        ..., description="Explanation of what was wrong and how it was fixed"
    )


# =============================================================================
# GRAPH DATA QUALITY JUDGE SCHEMAS
# =============================================================================


class GraphDataQualityEvaluation(BaseModel):
    """Structured output schema for LLM data quality evaluation of graph entities."""

    should_keep: bool = Field(
        ..., description="Whether the entity has enough data to be useful"
    )
    quality_score: float = Field(
        ..., ge=0.0, le=1.0, description="Overall quality score"
    )
    missing_fields: list[str] = Field(
        default_factory=list,
        description=(
            "Fields that should be present for this entity type " "but are missing"
        ),
    )
    repair_suggestions: dict[str, str] = Field(
        default_factory=dict,
        description="Map of field name -> suggested value for repair",
    )
    reasoning: str = Field(..., description="Explanation of the quality assessment")

    def to_judge_result(self, judge_name: str) -> "JudgeResult":
        issues = [
            JudgeIssue(
                category="data_quality",
                severity="warning",
                message=f"Missing field: {f}",
            )
            for f in self.missing_fields
        ]
        return JudgeResult(
            is_valid=self.should_keep,
            score=self.quality_score,
            reasoning=self.reasoning,
            issues=issues,
            suggestions=list(self.repair_suggestions.keys()),
            judge_name=judge_name,
            metadata={"repair_suggestions": self.repair_suggestions},
        )


# =============================================================================
# CHUNK RELEVANCE JUDGE SCHEMAS
# =============================================================================


class ChunkRelevance(BaseModel):
    """Relevance assessment for a single retrieved chunk."""

    chunk_id: str = Field(..., description="The chunk identifier from the input")
    relevant: bool = Field(
        ..., description="Whether this chunk is relevant to the query"
    )
    reasoning: str = Field(..., description="Brief explanation of relevance decision")


class ChunkRelevanceEvaluation(BaseModel):
    """Structured output from the chunk relevance judge."""

    relevant_chunk_ids: list[str] = Field(
        ...,
        description="IDs of chunks that are relevant to the user's question",
    )
    chunks: list[ChunkRelevance] = Field(
        ...,
        description="Per-chunk relevance assessments",
    )

    def to_judge_result(self, judge_name: str) -> "JudgeResult":
        relevant_count = len(self.relevant_chunk_ids)
        total_count = len(self.chunks)
        return JudgeResult(
            is_valid=relevant_count > 0,
            score=relevant_count / total_count if total_count else 0.0,
            reasoning=f"{relevant_count}/{total_count} chunks relevant",
            judge_name=judge_name,
        )
