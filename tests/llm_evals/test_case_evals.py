"""DeepEval case agent evaluation tests.

Placeholder for case-agent-specific evaluations. The case agent produces
structured outputs (facts_of_the_case, injury_summary, etc.) that require
different ground-truth schemas than chat. This module will be implemented
when case-agent GT is migrated to Langfuse Datasets.

For now, the chat eval suite (test_chat_evals.py) covers the full chat
agent loop, which implicitly exercises case-agent outputs via prompt
context injection.
"""

from __future__ import annotations

import pytest


@pytest.mark.llm
@pytest.mark.deepeval
@pytest.mark.asyncio
class TestCaseEvals:
    """Case agent evaluation tests (placeholder for future GT migration)."""

    @pytest.mark.skip(reason="Case agent GT not yet migrated to Langfuse Datasets")
    async def test_case_agent_quality(self):
        """Placeholder — will be parametrized once case GT is in Langfuse."""
        pass
