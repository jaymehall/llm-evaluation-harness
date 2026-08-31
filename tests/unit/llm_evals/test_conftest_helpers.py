"""Unit tests for conftest helper functions in the DeepEval suite.

Verifies LLMTestCase construction, GT parsing, and dataset resolution.
Does NOT require Langfuse or a live agent — tests only the pure logic.

Performance budget: each test < 3 seconds.
"""

import pytest
from deepeval.test_case import LLMTestCase

from tests.agent_harness.langfuse_dataset_client import LangfuseDatasetItem
from tests.llm_evals.conftest import (
    _get_all_fixture_names,
    _parse_expected_output,
    build_test_case,
)


@pytest.fixture
def sample_langfuse_item():
    """A realistic LangfuseDatasetItem."""
    return LangfuseDatasetItem(
        item_id="item-abc-123",
        input="What are the total medical bills?",
        expected_output={
            "expected_facts": {"medical_bills_total": "$10,820"},
            "expected_contains": ["$10,820", "medical"],
            "expected_not_contains": ["$50,000"],
            "expected_narrative": "The total medical bills are approximately $10,820.",
            "type_hints": {},
            "expected_context": ["Medical records show $10,820 in total billing."],
        },
        metadata={
            "fixture_group": "docs_only",
            "fixture_name": "john_smith",
            "node_id": "1c9489b7-6ba1-4304-81df-60deba317748",
            "entry_id": "medical_bills_total",
            "source_format": "unified",
            "source_hash": "abcdef1234567890",
            "source_order": 0,
        },
        status="ACTIVE",
    )


@pytest.fixture
def sample_agent_response():
    """A realistic agent response dict."""
    return {
        "response": "The total medical bills are $10,820 based on the records.",
        "tool_results": [
            {"tool_name": "get_entity_data", "result": {"total_amount": 10820}}
        ],
        "node_context": "This case involves medical treatment bills totaling $10,820.",
        "session_id": "session-xyz",
    }


class TestBuildTestCase:
    """build_test_case constructs a proper LLMTestCase from Langfuse + agent."""

    def test_returns_llm_test_case(self, sample_langfuse_item, sample_agent_response):
        result = build_test_case(sample_langfuse_item, sample_agent_response)
        assert isinstance(result, LLMTestCase)

    def test_input_from_langfuse(self, sample_langfuse_item, sample_agent_response):
        result = build_test_case(sample_langfuse_item, sample_agent_response)
        assert result.input == "What are the total medical bills?"

    def test_actual_output_from_agent(
        self, sample_langfuse_item, sample_agent_response
    ):
        result = build_test_case(sample_langfuse_item, sample_agent_response)
        assert "The total medical bills are $10,820" in result.actual_output

    def test_expected_output_from_narrative(
        self, sample_langfuse_item, sample_agent_response
    ):
        result = build_test_case(sample_langfuse_item, sample_agent_response)
        assert result.expected_output == (
            "The total medical bills are approximately $10,820."
        )

    def test_retrieval_context_from_expected_context(
        self, sample_langfuse_item, sample_agent_response
    ):
        result = build_test_case(sample_langfuse_item, sample_agent_response)
        assert result.retrieval_context == [
            "Medical records show $10,820 in total billing."
        ]

    def test_metadata_carries_tool_results(
        self, sample_langfuse_item, sample_agent_response
    ):
        result = build_test_case(sample_langfuse_item, sample_agent_response)
        assert len(result.additional_metadata["tool_results"]) == 1
        assert (
            result.additional_metadata["tool_results"][0]["tool_name"]
            == "get_entity_data"
        )

    def test_metadata_carries_prompt_context(
        self, sample_langfuse_item, sample_agent_response
    ):
        result = build_test_case(sample_langfuse_item, sample_agent_response)
        assert "medical treatment bills" in (
            result.additional_metadata["prompt_context"]
        )

    def test_metadata_carries_gt_fields(
        self, sample_langfuse_item, sample_agent_response
    ):
        result = build_test_case(sample_langfuse_item, sample_agent_response)
        assert result.additional_metadata["expected_facts"] == {
            "medical_bills_total": "$10,820"
        }
        assert "$10,820" in result.additional_metadata["expected_contains"]

    def test_metadata_carries_item_id(
        self, sample_langfuse_item, sample_agent_response
    ):
        result = build_test_case(sample_langfuse_item, sample_agent_response)
        assert result.additional_metadata["item_id"] == "item-abc-123"

    def test_handles_empty_agent_response(self, sample_langfuse_item):
        result = build_test_case(sample_langfuse_item, {})
        assert result.actual_output == ""
        assert result.additional_metadata["tool_results"] == []
        assert result.additional_metadata["prompt_context"] is None


class TestParseExpectedOutput:
    """_parse_expected_output handles various input shapes."""

    def test_dict_input(self):
        raw = {
            "expected_facts": {"key": "value"},
            "expected_contains": ["test"],
            "expected_not_contains": [],
            "expected_narrative": "narrative",
            "type_hints": {},
            "expected_context": [],
        }
        result = _parse_expected_output(raw)
        assert result.expected_facts == {"key": "value"}
        assert result.expected_narrative == "narrative"

    def test_none_input(self):
        result = _parse_expected_output(None)
        assert result.expected_facts == {}
        assert result.expected_contains == []
        assert result.expected_narrative == ""

    def test_empty_dict(self):
        result = _parse_expected_output({})
        assert result.expected_facts == {}


class TestGetAllFixtureNames:
    """_get_all_fixture_names returns all known fixtures."""

    def test_returns_list(self):
        names = _get_all_fixture_names()
        assert isinstance(names, list)
        assert len(names) > 0

    def test_includes_known_fixtures(self):
        names = _get_all_fixture_names()
        assert "john_smith" in names
        assert "jane_doe" in names

    def test_includes_crm_fixtures(self):
        names = _get_all_fixture_names()
        assert "crm_case_1" in names
