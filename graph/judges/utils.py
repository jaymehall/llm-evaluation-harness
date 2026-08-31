"""
Utility functions for the LLM-as-a-Judge framework.

Provides:
- format_tool_results: Serialization of tool results for judge prompts
- format_grounding_sources: Multi-channel source assembly for FaithfulnessJudge
- ensure_dict: Re-exported from core.utils.json_utils for convenience
"""

from typing import Any, Optional

from core.utils.json_utils import ensure_dict  # noqa: F401 — re-export

# Stale marker: when case-agent analysis is pending, the chat agent
# replaces cached agent-output / workflow-summary blocks with this text so
# the LLM is forced to call tools instead of parroting stale data. The
# grounding judge must NOT count the marker text as a "source" — strip it
# before judging so a response asserting "the analysis says X" against a
# stale-marker context is correctly classified as ungrounded.
_STALE_MARKER_FRAGMENT = "Analysis in progress — cached agent outputs and workflow"
_STALE_MARKER_REPLACEMENT = (
    "(cached outputs unavailable — agent should have queried tools)"
)


def format_tool_results(tool_results: list[dict]) -> str:
    """Serialize tool results into text for the grounding judge prompt.

    Extracts the factually-relevant content (entities, amounts, names, dates,
    chunk text) from each tool result and formats it as a readable document
    the judge LLM can scan for source matches.

    Args:
        tool_results: List of tool result dicts from the agent response.
            Each dict has ``tool_name`` plus the body under ``result`` (the
            shape produced by ``OpenAIConversationalAssistant``) or ``content``
            (legacy/test fixtures and ``golden_validator``). When both keys
            are present, ``result`` wins because that is the production
            payload and ``content`` may be a stale alias.

    Returns:
        Formatted text representation of all tool results.
    """
    if not tool_results:
        return ""

    sections = []
    for result in tool_results:
        tool_name = result.get("tool_name", "unknown")
        body = result["result"] if "result" in result else result.get("content", {})
        sections.append(f"--- Tool: {tool_name} ---")
        if isinstance(body, dict):
            sections.append(_format_dict(body))
        elif isinstance(body, str):
            sections.append(body)
        else:
            sections.append(str(body))
    return "\n\n".join(sections)


def _format_dict(d: Any, indent: int = 0) -> str:
    """Recursively format a dict into readable key-value lines.

    Args:
        d: Dict to format.
        indent: Current indentation level.

    Returns:
        Formatted string with indented key-value pairs.
    """
    if not isinstance(d, dict):
        return str(d)

    lines = []
    prefix = "  " * indent
    for key, value in d.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.append(_format_dict(value, indent + 1))
        elif isinstance(value, list):
            lines.append(f"{prefix}{key}:")
            for item in value:
                if isinstance(item, dict):
                    lines.append(_format_dict(item, indent + 1))
                else:
                    lines.append(f"{prefix}  - {item}")
        else:
            lines.append(f"{prefix}{key}: {value}")
    return "\n".join(lines)


def _strip_stale_markers(prompt_context: Optional[str]) -> Optional[str]:
    """Replace stale-analysis markers with an explicit absence label.

    Returns the input unchanged when no marker is present so callers can
    distinguish "no prompt context" (None / empty) from "context contains
    a stale marker" (replaced inline).
    """
    if not prompt_context:
        return prompt_context
    if _STALE_MARKER_FRAGMENT in prompt_context:
        return prompt_context.replace(
            _STALE_MARKER_FRAGMENT,
            _STALE_MARKER_REPLACEMENT,
        )
    return prompt_context


def format_grounding_sources(
    tool_results: list[dict],
    prompt_context: Optional[str] = None,
) -> str:
    """Serialize all source channels the chat agent saw on this request.

    Returns a single text block split into clearly delimited sections so the
    grounding judge can attribute each claim to its origin. The tool-results
    section reuses ``format_tool_results`` verbatim — no behavior change to
    the existing serializer.

    Args:
        tool_results: Tool-call results captured during the chat request.
        prompt_context: The ``node_context`` markdown block produced by
            ``OpenAIConversationalAssistant._get_rich_node_context``. May be
            None for general (non-node-bound) chat requests.

    Returns:
        Formatted multi-section text. Empty sections emit an explicit
        ``(no <section>)`` placeholder so the LLM does not silently treat
        a missing channel as evidence.
    """
    tool_block = format_tool_results(tool_results) or "(no tool calls were made)"
    context_block = _strip_stale_markers(prompt_context) or (
        "(no prompt context — general chat or empty node)"
    )
    return (
        "=== TOOL RESULTS ===\n"
        f"{tool_block}\n"
        "\n=== PROMPT CONTEXT (system-prompt-injected) ===\n"
        f"{context_block}"
    )
