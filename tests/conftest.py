"""Shared pytest fixtures for the Agentic AI Patterns Workshop.

This module provides reusable fixtures for testing:
- Configuration mocking
- Telemetry setup
- Azure client mocks
- Async event loop configuration
"""

import os
from collections.abc import AsyncIterator, Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.common.config import Settings, get_settings
from src.common.telemetry import reset_telemetry


# -----------------------------------------------------------------------------
# Configuration Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def env_vars() -> Iterator[dict[str, str]]:
    """Provide a clean environment with test variables.

    This fixture sets up test environment variables and cleans up
    after the test completes.

    Yields:
        Dictionary of test environment variables.
    """
    test_vars = {
        "AZURE_OPENAI_ENDPOINT": "https://test-endpoint.openai.azure.com/",
        "AZURE_OPENAI_API_KEY": "test-api-key-12345",
        "AZURE_OPENAI_API_VERSION": "2024-10-21",
        "AZURE_OPENAI_DEPLOYMENT": "gpt-4o",
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "text-embedding-3-small",
        "APPLICATIONINSIGHTS_CONNECTION_STRING": "InstrumentationKey=test-key",
        "ENABLE_OTEL": "true",
        "LOG_LEVEL": "DEBUG",
    }

    # Save original values
    original_env = {k: os.environ.get(k) for k in test_vars}

    # Set test values
    os.environ.update(test_vars)

    # Clear cached settings
    get_settings.cache_clear()

    yield test_vars

    # Restore original values
    for key, value in original_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

    # Clear cached settings again
    get_settings.cache_clear()


@pytest.fixture
def minimal_env_vars() -> Iterator[dict[str, str]]:
    """Provide minimal environment (Azure not configured).

    Yields:
        Dictionary of minimal test environment variables.
    """
    test_vars = {
        "AZURE_OPENAI_ENDPOINT": "",
        "AZURE_OPENAI_API_KEY": "",
        "ENABLE_OTEL": "false",
        "LOG_LEVEL": "INFO",
    }

    # Save original values
    original_env = {k: os.environ.get(k) for k in test_vars}

    # Clear existing vars
    for key in [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "APPLICATIONINSIGHTS_CONNECTION_STRING",
    ]:
        os.environ.pop(key, None)

    # Set minimal values
    os.environ.update(test_vars)

    # Clear cached settings
    get_settings.cache_clear()

    yield test_vars

    # Restore original values
    for key, value in original_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

    get_settings.cache_clear()


@pytest.fixture
def test_settings(env_vars: dict[str, str]) -> Settings:
    """Provide a Settings instance with test configuration.

    Args:
        env_vars: Environment variables fixture.

    Returns:
        Settings instance with test configuration.
    """
    return get_settings()


# -----------------------------------------------------------------------------
# Telemetry Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_telemetry_state() -> Iterator[None]:
    """Reset telemetry state before and after each test.

    This ensures tests don't interfere with each other's telemetry setup.
    """
    reset_telemetry()
    yield
    reset_telemetry()


@pytest.fixture
def mock_tracer() -> MagicMock:
    """Provide a mock tracer for testing span creation.

    Returns:
        Mock tracer with span creation capability.
    """
    tracer = MagicMock()
    span = MagicMock()
    span.__enter__ = MagicMock(return_value=span)
    span.__exit__ = MagicMock(return_value=None)
    tracer.start_as_current_span.return_value = span
    return tracer


@pytest.fixture
def mock_azure_monitor_exporter() -> Iterator[MagicMock]:
    """Mock the Azure Monitor exporter for telemetry tests.

    Yields:
        Mock AzureMonitorTraceExporter class.
    """
    with patch(
        "azure.monitor.opentelemetry.exporter.AzureMonitorTraceExporter"
    ) as mock_exporter:
        mock_exporter.return_value = MagicMock()
        yield mock_exporter


# -----------------------------------------------------------------------------
# Azure Client Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def mock_openai_client() -> AsyncMock:
    """Provide a mock Azure OpenAI client.

    Returns:
        AsyncMock configured as an OpenAI client.
    """
    client = AsyncMock()

    # Mock chat completions
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Test response"
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 20
    client.chat.completions.create = AsyncMock(return_value=mock_response)

    # Mock embeddings
    mock_embedding = MagicMock()
    mock_embedding.data = [MagicMock()]
    mock_embedding.data[0].embedding = [0.1] * 1536  # Standard embedding size
    client.embeddings.create = AsyncMock(return_value=mock_embedding)

    return client


@pytest.fixture
def mock_agent() -> AsyncMock:
    """Provide a mock agent for testing agent operations.

    Returns:
        AsyncMock configured as an agent.
    """
    agent = AsyncMock()
    agent.name = "test_agent"
    agent.run = AsyncMock(return_value="Agent response")
    return agent


# -----------------------------------------------------------------------------
# MCP Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def mock_mcp_server() -> MagicMock:
    """Provide a mock MCP server for testing.

    Returns:
        Mock MCP server with tool registration.
    """
    server = MagicMock()
    server.tools = {}

    def register_tool(name: str) -> Any:
        def decorator(func: Any) -> Any:
            server.tools[name] = func
            return func

        return decorator

    server.tool = register_tool
    return server


@pytest.fixture
def mock_mcp_client() -> AsyncMock:
    """Provide a mock MCP client for testing.

    Returns:
        AsyncMock configured as an MCP client.
    """
    client = AsyncMock()
    client.list_tools = AsyncMock(return_value=[])
    client.call_tool = AsyncMock(return_value={"result": "success"})
    return client


# -----------------------------------------------------------------------------
# AG-UI Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def mock_ag_ui_encoder() -> MagicMock:
    """Provide a mock AG-UI event encoder.

    Returns:
        Mock encoder for AG-UI events.
    """
    encoder = MagicMock()
    encoder.encode = MagicMock(return_value=b"encoded_event")
    return encoder


# -----------------------------------------------------------------------------
# A2A Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def mock_a2a_client() -> AsyncMock:
    """Provide a mock A2A client for testing.

    Returns:
        AsyncMock configured as an A2A client.
    """
    client = AsyncMock()
    client.discover = AsyncMock(return_value={"name": "test_agent"})
    client.send_task = AsyncMock(return_value={"task_id": "test-task-123"})
    return client


# -----------------------------------------------------------------------------
# Async Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
async def async_client() -> AsyncIterator[AsyncMock]:
    """Provide an async HTTP client mock.

    Yields:
        AsyncMock configured as an HTTP client.
    """
    client = AsyncMock()
    client.get = AsyncMock()
    client.post = AsyncMock()
    yield client


# -----------------------------------------------------------------------------
# Sample Data Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_tool_result() -> dict[str, Any]:
    """Provide sample tool execution result.

    Returns:
        Sample tool result dictionary.
    """
    return {
        "status": "success",
        "data": {"query": "test", "results": ["item1", "item2"]},
        "execution_time_ms": 150,
    }


@pytest.fixture
def sample_agent_response() -> dict[str, Any]:
    """Provide sample agent response.

    Returns:
        Sample agent response dictionary.
    """
    return {
        "content": "This is the agent's response.",
        "model": "gpt-4o",
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
    }


@pytest.fixture
def sample_workflow_steps() -> list[dict[str, Any]]:
    """Provide sample workflow step definitions.

    Returns:
        List of workflow step dictionaries.
    """
    return [
        {"name": "research", "agent": "researcher", "input": "query"},
        {"name": "analyze", "agent": "analyst", "input": "research_results"},
        {"name": "summarize", "agent": "writer", "input": "analysis"},
    ]
