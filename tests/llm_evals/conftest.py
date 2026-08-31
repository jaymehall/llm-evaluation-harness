"""Pytest configuration and fixtures for the DeepEval evaluation suite.

Responsibilities:
- Load ground truth from Langfuse Datasets (via LangfuseDatasetClient)
- Run live agent calls (via ContinuousAssistantManager)
- Construct DeepEval LLMTestCase instances per GT item
- Provide --dataset CLI filter for scoping to specific fixtures
- Write rubric reports after each session via the reporter hook
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
from deepeval.test_case import LLMTestCase

from tests.agent_harness.conftest_shared import (
    STAGING_FIXTURES,
    get_or_create_test_user_for_org,
    resolve_fixture_name,
)
from tests.agent_harness.langfuse_dataset_client import (
    LangfuseDatasetClient,
    LangfuseDatasetItem,
)
from tests.agent_harness.langfuse_envelope import LangfuseExpectedOutput

logger = logging.getLogger(__name__)


def pytest_addoption(parser):
    """Register --dataset CLI option for filtering evaluation datasets."""
    parser.addoption(
        "--dataset",
        action="append",
        default=None,
        help=(
            "Filter to specific dataset fixture(s) by name. "
            "Can be passed multiple times: --dataset john_smith --dataset jane_doe. "
            "When omitted, all fixtures run."
        ),
    )


def _get_all_fixture_names() -> list[str]:
    """Return all fixture names from STAGING_FIXTURES."""
    names = []
    for group in STAGING_FIXTURES.values():
        names.extend(group["nodes"].keys())
    return names


def _get_target_datasets(config) -> list[str]:
    """Resolve which datasets to run based on --dataset option."""
    explicit = config.getoption("--dataset")
    if explicit:
        return explicit
    return _get_all_fixture_names()


def _load_dataset_items(dataset_name: str) -> list[LangfuseDatasetItem]:
    """Load active items from a Langfuse dataset by fixture name.

    The Langfuse dataset naming convention is:
    `chat_agent_{fixture_name}` for chat agent evaluations.
    """
    client = LangfuseDatasetClient()
    langfuse_dataset_name = f"chat_agent_{dataset_name}"
    try:
        items = list(client.list_items(langfuse_dataset_name))
        active_items = [item for item in items if item.status == "ACTIVE"]
        logger.info(
            "Loaded %d active items from dataset '%s'",
            len(active_items),
            langfuse_dataset_name,
        )
        return active_items
    except Exception as e:
        logger.warning("Failed to load dataset '%s': %s", langfuse_dataset_name, e)
        return []


def _parse_expected_output(raw: Any) -> LangfuseExpectedOutput:
    """Parse the expected_output field into a validated schema."""
    if isinstance(raw, dict):
        return LangfuseExpectedOutput(**raw)
    return LangfuseExpectedOutput()


def build_test_case(
    item: LangfuseDatasetItem, agent_response: dict[str, Any]
) -> LLMTestCase:
    """Map a Langfuse item + agent response into a DeepEval LLMTestCase.

    The additional_metadata dict carries all context the metric adapters
    need to invoke the judges (tool_results, prompt_context, GT fields).
    """
    expected = _parse_expected_output(item.expected_output)
    input_text = item.input if isinstance(item.input, str) else str(item.input)
    metadata = item.metadata or {}

    return LLMTestCase(
        input=input_text,
        actual_output=agent_response.get("response", ""),
        expected_output=expected.expected_narrative,
        retrieval_context=expected.expected_context or None,
        additional_metadata={
            "tool_results": agent_response.get("tool_results", []),
            "prompt_context": agent_response.get("node_context"),
            "expected_facts": expected.expected_facts,
            "expected_contains": expected.expected_contains,
            "expected_not_contains": expected.expected_not_contains,
            "entry_id": metadata.get("entry_id", ""),
            "fixture_name": metadata.get("fixture_name", ""),
            "item_id": item.item_id,
        },
    )


async def _run_agent(user, node_id: str, message: str) -> dict[str, Any]:
    """Execute the chat agent via the production service path."""
    from core.services.continuous_assistant import ContinuousAssistantManager

    return await ContinuousAssistantManager.send_message_to_continuous_session(
        user=user,
        message=message,
        node_id=node_id,
    )


@pytest.fixture(scope="session")
def target_datasets(request) -> list[str]:
    """Resolved list of fixture names to evaluate."""
    return _get_target_datasets(request.config)


@pytest.fixture(scope="session")
def langfuse_dataset_items(target_datasets) -> dict[str, list[LangfuseDatasetItem]]:
    """Load all Langfuse dataset items for the target fixtures.

    Returns a dict keyed by fixture_name with lists of active items.
    """
    all_items: dict[str, list[LangfuseDatasetItem]] = {}
    for dataset_name in target_datasets:
        items = _load_dataset_items(dataset_name)
        if items:
            all_items[dataset_name] = items
        else:
            logger.warning("No items loaded for dataset '%s' — skipping", dataset_name)
    return all_items


def _build_eval_params(
    langfuse_items: dict[str, list[LangfuseDatasetItem]],
) -> list[tuple[str, LangfuseDatasetItem]]:
    """Flatten dataset items into parametrize-ready tuples."""
    params = []
    for fixture_name, items in langfuse_items.items():
        for item in items:
            params.append((fixture_name, item))
    return params


def pytest_generate_tests(metafunc):
    """Dynamically parametrize test functions that request eval_test_case."""
    if "eval_fixture_name" in metafunc.fixturenames:
        datasets = _get_target_datasets(metafunc.config)
        all_items: list[tuple[str, str]] = []
        for dataset_name in datasets:
            items = _load_dataset_items(dataset_name)
            for item in items:
                all_items.append((dataset_name, item.item_id))
        if all_items:
            metafunc.parametrize(
                "eval_fixture_name,eval_item_id",
                all_items,
                ids=[f"{name}:{iid[:8]}" for name, iid in all_items],
            )


@pytest.fixture
async def eval_test_case(
    eval_fixture_name: str,
    eval_item_id: str,
    langfuse_dataset_items: dict[str, list[LangfuseDatasetItem]],
) -> LLMTestCase:
    """Build a complete LLMTestCase by running the agent and assembling GT.

    This fixture:
    1. Finds the LangfuseDatasetItem by item_id
    2. Resolves the fixture's node_id and org_id
    3. Gets or creates a test user
    4. Runs the live agent
    5. Builds and returns the LLMTestCase
    """
    items = langfuse_dataset_items.get(eval_fixture_name, [])
    item = next((i for i in items if i.item_id == eval_item_id), None)
    if item is None:
        pytest.skip(f"Item {eval_item_id} not found in dataset {eval_fixture_name}")

    node_id, org_id = resolve_fixture_name(eval_fixture_name)

    user = await get_or_create_test_user_for_org(
        org_id=org_id,
        email="deepeval-harness@example.test",
        username="deepeval_harness",
    )
    if user is None:
        pytest.skip(
            f"Could not create test user for org {org_id} "
            f"(fixture={eval_fixture_name})"
        )

    input_text = item.input if isinstance(item.input, str) else str(item.input)
    agent_response = await _run_agent(user, node_id, input_text)

    return build_test_case(item, agent_response)
