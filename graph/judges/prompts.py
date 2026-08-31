"""
Prompt constants for LLM-as-a-Judge evaluations.

Follows the pattern in core/constants/conversational_prompts.py
and core/services/pii_scanner_service.py — Python string constants
with .format() for variable substitution.
"""

# =============================================================================
# CYPHER VALIDATION PROMPT
# =============================================================================

CYPHER_VALIDATION_PROMPT = """\
You are a Cypher query validation expert for Neo4j.

Given a graph schema, a natural language query, and a generated Cypher query,
evaluate whether the Cypher accurately and safely implements the user's intent.

## Graph Schema

Node Labels: {node_labels}
Relationship Types: {relationship_types}
Node Properties:
{node_properties}

## User's Natural Language Query

{natural_language_query}

## Generated Cypher Query

```cypher
{cypher_query}
```

## Evaluation Criteria

1. **Semantic Accuracy**: Does the Cypher query correctly implement what the user \
asked for? Are the right node types, relationship traversals, and filters used?

2. **Performance**: Are there any Cartesian products (disconnected MATCH patterns), \
missing relationship anchors, or patterns that could cause excessive computation?

3. **Logic Correctness**: Are relationships traversed in the right direction? Are \
filters applied correctly? Does the RETURN clause return what the user expects?

Evaluate the query and provide your assessment.\
"""


# =============================================================================
# CYPHER REPAIR PROMPT
# =============================================================================

CYPHER_REPAIR_PROMPT = """\
You are a Cypher query repair expert for Neo4j.

A generated Cypher query has validation issues. Your job is to fix the query \
so it correctly and safely implements the user's intent.

## Graph Schema

Node Labels: {node_labels}
Relationship Types: {relationship_types}
Node Properties:
{node_properties}

## User's Natural Language Query

{natural_language_query}

## Original Cypher Query (has issues)

```cypher
{cypher_query}
```

## Issues Found

{issues_description}

## Repair Instructions

1. Fix all errors and warnings listed above.
2. The corrected query MUST be read-only (MATCH/RETURN only, no mutations).
3. The corrected query MUST include a LIMIT clause (max 100).
4. Use ONLY the node labels, relationship types, and properties from the schema above.
5. Preserve the user's original intent as closely as possible.
6. If the original query is completely unsalvageable, write a new query from scratch \
that implements the user's natural language intent.

Provide the corrected query and explain what you changed.\
"""


# =============================================================================
# GRAPH DATA QUALITY PROMPT
# =============================================================================

GRAPH_DATA_QUALITY_PROMPT = """\
You are a data quality expert evaluating entities extracted from legal documents \
for storage in a Neo4j knowledge graph.

## Entity Under Review

Entity Type: {entity_type}
Properties:
{entity_properties}

## Graph Schema Context

Valid entity types: {valid_entity_types}
Expected properties for this type: {expected_properties}

## Evaluation Criteria

1. **Identity**: Does this entity have enough identifying information to be useful? \
A name, ID, code, or other unique identifier should be present or synthesizable \
from the available properties.

2. **Completeness**: Are the key properties for this entity type populated? \
Missing critical fields reduce the entity's value in the knowledge graph.

3. **Validity**: Are the property values reasonable for a legal domain entity? \
Look for obvious extraction errors, garbled text, or nonsensical values.

4. **Usefulness**: Would this entity provide value in a legal case knowledge graph, \
or is it noise/junk from imperfect LLM extraction?

Decide whether to keep or discard this entity.

## Repair Instructions

If the entity is worth keeping but has issues, you MUST populate the \
`repair_suggestions` field with concrete fixes. For each missing or garbled \
field, provide the field name as key and the suggested corrected value as value.

Examples:
- Missing name on a Treatment with type=MRI, body_part=cervical spine: \
  repair_suggestions = {{"name": "Cervical Spine MRI"}}
- Garbled OCR name "J0hn Srni+h" with email john.smith@example.com: \
  repair_suggestions = {{"name": "John Smith"}}
- Missing description on Diagnosis with code M54.5: \
  repair_suggestions = {{"description": "Low back pain"}}

If the entity should be discarded, leave repair_suggestions empty.\
"""


# =============================================================================
# CHUNK RELEVANCE PROMPT
# =============================================================================

CHUNK_RELEVANCE_PROMPT = """\
You are a legal document relevance judge for a legal case management system.

Given a user's question and a list of retrieved document chunks from a legal case, \
determine which chunks are relevant to answering the question.

A chunk is RELEVANT if:
- It contains information that helps answer the question directly or indirectly
- It discusses the same broad subject area as the question (e.g., a chunk about \
medical treatment details is relevant to a question about treatment completion)
- It comes from a document that would reasonably be cited when answering the question

A chunk is NOT RELEVANT if:
- It merely mentions a keyword from the question in a different context. \
For example, if the question asks about a "police report", an insurance letter \
that references "review of the police report" is NOT relevant — only chunks that \
contain the actual police report content are relevant.
- It is about a completely different subject (e.g., a vehicle repair estimate \
when the question is about medical treatment)
- It contains only watermarks, scan codes, marginalia, or page headers/footers

Pay close attention to the FILENAME of each chunk. If the user asks about a \
specific document type (e.g., "police report", "MRI results", "demand letter"), \
strongly prefer chunks from files whose names match that document type.

USER'S QUESTION:
{query}

RETRIEVED CHUNKS:
{chunks}

Return the chunk IDs that are relevant to the user's question.\
"""


# =============================================================================
# FAITHFULNESS EVAL PROMPT
# =============================================================================
#
# Locked-rubric prompt for the Faithfulness metric. Replaces
# GROUNDING_VERIFICATION_PROMPT (orphaned at the end of a prior refactor)
# while preserving its dual-channel grounding logic. Constrains
# the LLM to emit an EvaluationResult with metric='faithfulness',
# [UNGROUNDED]-prefixed violations, and a 2-sentence rationale.
#
# See docs/architecture/evaluation-rubric.md §2.1 for the locked metric
# definition.

FAITHFULNESS_EVAL_PROMPT = """\
You are a faithfulness judge for a legal case management system.

CRITICAL FRAMING — READ THIS FIRST.

The chat agent's response was generated from MULTIPLE source channels, not \
just tool calls.  Your job is to verify that each factual claim in the \
response can be traced back to ANY of the source channels below — not only \
the tool results.

SOURCE CHANNELS (any of these is a legitimate source of truth):

1. **Tool Results** — data the agent retrieved by calling tools during this \
request (Salesforce queries, graph searches, document chunk lookups).
2. **Prompt Context** — markdown blocks injected into the system prompt \
before the agent saw the user's question.  This includes the active node / \
case overview (CRM vitals), case-agent analysis results, CRM child records, \
workflow summaries, and any tuning-override global context.

A claim grounded in EITHER channel is GROUNDED.  A claim grounded in NEITHER \
channel is UNGROUNDED — that is the only true hallucination.

GROUNDING RULES.

A claim is GROUNDED if AT LEAST ONE of the following is true:
- The exact value appears verbatim in TOOL RESULTS or PROMPT CONTEXT
- The value is a reasonable derivation (e.g., sum of line items present \
in either channel)
- A name matches a source name, allowing minor formatting differences \
("ProHealth Injury & Rehab" ≈ "ProHealth Injury and Rehab")

A claim is NOT GROUNDED if BOTH are true:
- No corresponding value appears in TOOL RESULTS, AND
- No corresponding value appears in PROMPT CONTEXT

STRICTNESS ON NUMERICS.

If the source shows "$10,820" and the response says "$10,800", that is \
NOT grounded — it is an LLM-fabricated approximation.  A different concrete \
value is never a paraphrase.

SCORING.

score = (count of grounded claims) / (total count of claims).  If the \
response contains no factual claims, score = 1.0.

OUTPUT — STRUCTURED EVALUATIONRESULT (LOCKED RUBRIC).

Return an EvaluationResult with:

- metric: must be the string literal "faithfulness".
- score: float in [0.0, 1.0] per scoring rule above.
- violating_sentences: list of strings.  For each ungrounded claim, add an \
entry that begins with the inline prefix "[UNGROUNDED] " followed by the \
ungrounded claim verbatim, e.g. "[UNGROUNDED] City Hospital on March 15, \
2025 (no source match)".  Empty list when score == 1.0.
- rationale: at most 2 sentences (about 200 characters).  Summarise the \
grounding outcome — do not enumerate individual claims (those go in \
violating_sentences).

INPUTS.

RESPONSE TO EVALUATE:
{response}

SOURCE DATA:
{source_data}

Evaluate every factual claim in the response.\
"""


# =============================================================================
# ANSWER RELEVANCY EVAL PROMPT
# =============================================================================
#
# Locked-rubric prompt for the Answer Relevancy metric. Replaces
# ACCURACY_VERIFICATION_PROMPT (orphaned at the end of a prior refactor)
# while preserving its paraphrase-tolerant scoring rules.
# Constrains the LLM to emit an EvaluationResult with
# metric='answer_relevancy', [MISSING]/[WRONG]/[FORBIDDEN]-prefixed
# violations, and a 2-sentence rationale.
#
# See docs/architecture/evaluation-rubric.md §2.2 for the locked metric
# definition.

ANSWER_RELEVANCY_EVAL_PROMPT = """\
You are an answer relevancy judge for a legal case management system.

CRITICAL FRAMING — READ THIS FIRST.

The GROUND TRUTH below is *minimal and scoped*.  It contains ONLY the \
facts the user explicitly asked about — nothing more.  The response is \
allowed and expected to contain additional correct information that is \
NOT mentioned in ground truth.  Extra correct content does NOT lower \
the score and is NOT a violation.  Hallucination relative to source \
data is a separate concern handled by the Faithfulness judge — your job \
is narrow: verify the response addresses every authored ground-truth \
item correctly.

GROUND TRUTH STRUCTURE.

Ground truth contains some combination of:
- **expected_facts**: key/value assertions.  Each VALUE must be conveyed \
by the response.
- **expected_contains**: concepts that MUST appear in the response.
- **expected_not_contains**: concepts that MUST NOT appear in the response.

PARAPHRASE TOLERANCE — APPLIES TO ALL THREE INPUT TYPES.

Treat the following as semantically equivalent:
- Numeric forms: "approximately $13K" ≈ "$12,800" ≈ "thirteen thousand"
- Date forms: "mid-September 2025" ≈ "September 15, 2025" ≈ "9/15/25"
- Synonyms / abbreviations: "MVA" ≈ "Motor Vehicle Accident"; "PI" ≈ \
"Personal Injury"
- Restated phrasing: "the client did not respond" ≈ "no contact"
- Format differences: bullet list vs prose; ordered vs unordered
- Dollar formatting: "$8,445" ≈ "$8,445.00" ≈ "$8445" ≈ "8,445" — \
trailing zeros, commas, and dollar signs are formatting, not substance.  \
"$9,264.46" ≈ "$9,264" when the context is a total or summary.
- Medical term synonyms: "cervical strain" ≈ "cervical sprain" ≈ \
"neck strain"; "lumbar" ≈ "lower back"; "chiropractic care" ≈ "chiro"; \
"MRI" ≈ "magnetic resonance imaging"
- Insurance terminology: "BI" ≈ "bodily injury"; "UM/UIM" ≈ \
"uninsured/underinsured motorist"; "PIP" ≈ "personal injury protection"
- Legal terminology: "SOL" ≈ "statute of limitations"; "demand letter" ≈ \
"demand package"; "comp neg" ≈ "comparative negligence"

STRICT RULES — APPLY EACH RULE TO EVERY GT ITEM SEPARATELY.

Rule 1 (expected_facts / expected_contains): Each item is CONVEYED if \
the response asserts the same fact in any equivalent form.  Each item \
is MISSING if the response is silent on it OR hedges ("unknown", "not \
available", "could not determine") when GT has a concrete answer.

Rule 2 (expected_facts / expected_contains): Each item is a WRONG_CLAIM \
if the response asserts a factually different value (e.g. GT says \
"September 15"; response says "September 25" — paraphrase tolerance \
does NOT extend to factually different values).

Rule 3 (expected_not_contains): Each item fails if it appears in the \
response verbatim OR as a semantic match (paraphrase, abbreviation, \
synonym).

Rule 4: NEVER flag a response claim as a violation just because GT \
doesn't mention it.  GT is minimal.  The response may legitimately add \
correct information from tool results.

SCORING — DETERMINISTIC PER-ITEM.

Let N = total count of (expected_facts items + expected_contains items + \
expected_not_contains items).

Let C = count of items that pass their rule:
  - expected_facts / expected_contains items that are conveyed (Rule 1) \
    AND not contradicted (Rule 2)
  - expected_not_contains items that are absent (Rule 3)

  score = C / N   (or 1.0 if N == 0)

OUTPUT — STRUCTURED EVALUATIONRESULT (LOCKED RUBRIC).

Return an EvaluationResult with:

- metric: must be the string literal "answer_relevancy".
- score: float in [0.0, 1.0] per scoring rule above.
- violating_sentences: list of strings.  Each entry begins with one of \
the inline prefixes:
    * "[MISSING] " — for expected_facts / expected_contains items the \
      response did not convey, followed by the GT item verbatim.
    * "[WRONG] " — for expected_facts items the response contradicts, \
      followed by "<response claim> (expected <GT value>)".
    * "[FORBIDDEN] " — for expected_not_contains items present in the \
      response, followed by the forbidden item verbatim.
  Empty list when score == 1.0.
- rationale: at most 2 sentences (about 200 characters).  Summarise the \
relevancy outcome — do not enumerate individual items.

WORKED EXAMPLES.

Example 1 — all items conveyed (score = 1.0):

  GT:
    expected_facts:
      accident_date: September 15, 2025
    expected_contains:
      - I-25
  Response: "The MVA occurred on September 15, 2025 on Interstate 25."

  Result: 2/2 items pass.  "September 15, 2025" matches verbatim; \
"Interstate 25" ≈ "I-25" under paraphrase tolerance.  score = 1.0, \
violating_sentences = [].

Example 2 — one missing, one wrong (score = 0.33):

  GT:
    expected_facts:
      total_medical: $10,820
      treating_provider: ProHealth Injury & Rehab
    expected_contains:
      - cervical strain
  Response: "Total medical bills are $9,500, treated at ProHealth \
Injury and Rehab for a cervical sprain."

  Result: 1/3 items pass.  "ProHealth Injury and Rehab" ≈ \
"ProHealth Injury & Rehab" (formatting).  "cervical sprain" ≈ \
"cervical strain" (medical synonym).  BUT "$9,500" is factually \
different from "$10,820" — that is a WRONG_CLAIM, not a paraphrase.  \
score = 0.33.  violating_sentences = ["[WRONG] $9,500 (expected \
$10,820)", "[MISSING] cervical strain"].

Wait — "cervical sprain" ≈ "cervical strain" passes; recounting: \
provider passes, cervical passes, total_medical fails.  score = 2/3 \
= 0.67.  violating_sentences = ["[WRONG] $9,500 (expected $10,820)"].

Example 3 — forbidden item present (score = 0.5):

  GT:
    expected_contains:
      - case is open
    expected_not_contains:
      - settlement amount
  Response: "The case is currently open. The settlement amount is \
$50,000."

  Result: 1/2 items pass.  "case is open" is conveyed.  \
"settlement amount" is present despite being forbidden.  score = 0.5.  \
violating_sentences = ["[FORBIDDEN] settlement amount"].

INPUTS.

QUESTION:
{question}

GROUND TRUTH:
{ground_truth}

RESPONSE TO EVALUATE:
{response}

Evaluate every authored GT item separately.\
"""

# =============================================================================
# TOOL FULFILLMENT EVAL PROMPT
# =============================================================================
#
# Locked-rubric prompt for the Tool Fulfillment metric. Scores whether the
# agent's tool-call pattern was appropriate for the question given the data
# already available in the agent's prompt context. The LLM judge receives
# the full prompt context so it can distinguish "no tools needed because
# data was in context" from "no tools called but data retrieval was needed."
#
# See docs/architecture/evaluation-rubric.md §2.3 for the locked metric
# definition.

TOOL_FULFILLMENT_EVAL_PROMPT = """\
You are a tool-fulfillment judge for a legal case management system.

CRITICAL FRAMING — READ THIS FIRST.

Given the user's question, the data ALREADY AVAILABLE to the agent in its \
system prompt (AVAILABLE CONTEXT), and the list of tool calls the agent \
made, rate whether the agent invoked an *appropriate* tool pattern for the \
question.  This metric does NOT score the correctness of the response — \
that is the job of the Faithfulness and Answer Relevancy judges.  Your job \
is narrow: were the right tools called, given what the user asked and what \
data the agent already had?

AVAILABLE CONTEXT — READ THIS BEFORE EVALUATING.

The AVAILABLE CONTEXT section below shows data that was injected into the \
agent's system prompt BEFORE the user asked their question.  The agent \
can answer from this data without calling any tools.  When evaluating \
whether tool calls were appropriate, first check whether the question \
can be answered from the AVAILABLE CONTEXT alone.

EVALUATION RULES.

Rule 1 (context-answerable — no tools needed): If the question can be \
answered from AVAILABLE CONTEXT alone, the agent does NOT need to call \
any tools.  Score 1.0 when ``tool_calls`` is empty.  Score LOW when \
``tool_calls`` is non-empty — this is wasted latency.  Use the \
"[WRONG_TOOL] called <tool> but answer was available in prompt context" \
prefix for each unnecessary tool call.

Rule 2 (context-answerable — greetings and small-talk): A question that \
is a greeting, small-talk, or requires no data lookup at all scores 1.0 \
when ``tool_calls`` is empty.

Rule 3 (data retrieval needed): A question that requires data NOT present \
in AVAILABLE CONTEXT (e.g., specific document content, detailed medical \
records, financial line items, file notes not summarised in context) \
scores 1.0 when at least one tool invocation matches the question's domain.

Rule 4 (failure modes): A question scores LOWER when:
- The agent called NO tools but the question clearly needs data that is \
  NOT in AVAILABLE CONTEXT → use the "[MISSING_TOOL] no tool called when \
  <domain> data was required" prefix.
- The agent called a tool that doesn't match the question's domain (e.g. \
  called ``pdf_search`` for a question that asks "what is the case status") \
  → use the "[WRONG_TOOL] called <X> but the question asks about <Y>" \
  prefix.
- The agent called tools entirely unrelated to the question → use the \
  "[OFF_TOPIC] called <X>; question is about <Y>" prefix.

Rule 5 (partial coverage is acceptable): Tool invocations that ARE \
relevant to the question's domain but do not exhaustively cover every \
possible angle do NOT lower the score.  Partial coverage of a relevant \
domain is acceptable — the judge does not penalize incomplete tool \
exploration.

SCORING — GUIDED.

- 1.0 = appropriate tool pattern (correct tools called, or correctly \
no tools when data was in context).
- 0.5 to 0.9 = partially appropriate (mix of relevant + irrelevant tool \
calls; or called tools that were unnecessary but domain-relevant).
- 0.0 to 0.4 = clearly inappropriate (no tools when data retrieval was \
needed and data was NOT in context; entirely off-topic tools; or called \
tools when the answer was obviously in the available context).

OUTPUT — STRUCTURED EVALUATIONRESULT (LOCKED RUBRIC).

Return an EvaluationResult with:

- metric: must be the string literal "tool_fulfillment".
- score: float in [0.0, 1.0] per scoring guide above.
- violating_sentences: list of strings.  Each entry begins with one of \
the inline prefixes:
    * "[WRONG_TOOL] " — for wrong or unnecessary tool calls (wasted \
      latency).
    * "[MISSING_TOOL] " — for questions needing data retrieval where no \
      tool was called and the data was NOT in available context.
    * "[OFF_TOPIC] " — for tool calls entirely unrelated to the question.
  Empty list when score == 1.0.
- rationale: at most 2 sentences (about 200 characters).  Summarise the \
tool-call pattern's fit — do not enumerate every tool call.

INPUTS.

AVAILABLE CONTEXT (data already in the agent's system prompt):
{available_context}

QUESTION:
{question}

TOOL CALLS MADE:
{tool_calls}

RESPONSE (for context only — do NOT score response correctness here):
{response}

Evaluate whether the tool-call pattern fits the question.\
"""
