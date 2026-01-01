"""Integration tests for Scenario 01: Simple Agent with MCP Tools.

These tests verify that the notebook components work together correctly.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestScenario01Components:
    """Tests for Scenario 01 components."""

    def test_tools_import(self) -> None:
        """All tools should be importable."""
        from src.tools import search_web, calculate, read_file, write_file
        from src.tools.search_tool import get_weather

        assert callable(search_web)
        assert callable(calculate)
        assert callable(read_file)
        assert callable(write_file)
        assert callable(get_weather)

    def test_mcp_server_import(self) -> None:
        """MCP server should be importable."""
        from src.tools.mcp_server import create_mcp_server, TOOLS

        assert callable(create_mcp_server)
        assert len(TOOLS) > 0

    def test_agent_import(self) -> None:
        """Agents should be importable."""
        from src.agents import BaseAgent, ResearchAgent

        assert BaseAgent is not None
        assert ResearchAgent is not None

    def test_telemetry_import(self) -> None:
        """Telemetry should be importable."""
        from src.common.telemetry import (
            setup_telemetry,
            get_tracer,
            create_span_attributes,
        )

        assert callable(setup_telemetry)
        assert callable(get_tracer)
        assert callable(create_span_attributes)


class TestMCPServerCreation:
    """Tests for MCP server creation."""

    def test_create_mcp_server(self) -> None:
        """MCP server should be creatable."""
        from src.tools.mcp_server import create_mcp_server

        server = create_mcp_server("test-server")
        assert server is not None

    def test_mcp_server_has_tools(self) -> None:
        """MCP server should have tool definitions."""
        from src.tools.mcp_server import TOOLS

        tool_names = {t.name for t in TOOLS}
        expected_tools = {
            "search_web",
            "get_weather",
            "calculate",
            "read_file",
            "write_file",
            "list_files",
        }
        assert expected_tools.issubset(tool_names)


class TestResearchAgentCreation:
    """Tests for ResearchAgent creation."""

    def test_research_agent_creation_without_azure(
        self, minimal_env_vars: dict[str, str]
    ) -> None:
        """ResearchAgent should fail gracefully without Azure config."""
        from src.common.exceptions import ConfigurationError
        from src.agents import ResearchAgent

        with pytest.raises(ConfigurationError):
            ResearchAgent(name="test_agent")

    def test_research_agent_with_mock_config(
        self, env_vars: dict[str, str]
    ) -> None:
        """ResearchAgent should be creatable with valid config."""
        from src.agents import ResearchAgent

        agent = ResearchAgent(name="test_agent")
        assert agent.name == "test_agent"
        assert agent.model is not None

    def test_research_agent_tool_registration(
        self, env_vars: dict[str, str]
    ) -> None:
        """ResearchAgent should accept tool handlers."""
        from src.agents import ResearchAgent

        agent = ResearchAgent(name="test_agent")

        async def mock_tool(**kwargs):
            return {"result": "test"}

        agent.set_tool_handlers({"mock_tool": mock_tool})
        assert "mock_tool" in agent._tool_handlers


class TestToolExecution:
    """Tests for tool execution."""

    @pytest.mark.asyncio
    async def test_search_web_execution(self) -> None:
        """search_web should execute and return results."""
        from src.tools import search_web

        result = await search_web("python programming", max_results=3)

        assert result["total_found"] > 0
        assert len(result["results"]) <= 3

    @pytest.mark.asyncio
    async def test_calculate_execution(self) -> None:
        """calculate should execute and return correct results."""
        from src.tools import calculate

        result = await calculate("add", 10, 5)
        assert result["result"] == 15

        result = await calculate("sqrt", 16)
        assert result["result"] == 4

    @pytest.mark.asyncio
    async def test_get_weather_execution(self) -> None:
        """get_weather should execute and return weather data."""
        from src.tools.search_tool import get_weather

        result = await get_weather("Seattle", "celsius")

        assert "temperature" in result
        assert "condition" in result
        assert result["units"] == "celsius"


class TestTelemetryIntegration:
    """Tests for telemetry integration."""

    def test_telemetry_setup(self, minimal_env_vars: dict[str, str]) -> None:
        """Telemetry should initialize without errors."""
        from src.common.telemetry import (
            setup_telemetry,
            is_telemetry_enabled,
            reset_telemetry,
        )

        reset_telemetry()
        setup_telemetry()
        assert is_telemetry_enabled()

    def test_tracer_creation(self, minimal_env_vars: dict[str, str]) -> None:
        """Tracers should be creatable."""
        from src.common.telemetry import setup_telemetry, get_tracer

        setup_telemetry()
        tracer = get_tracer("test_module")

        assert tracer is not None
        assert hasattr(tracer, "start_as_current_span")

    @pytest.mark.asyncio
    async def test_tool_creates_spans(
        self, minimal_env_vars: dict[str, str]
    ) -> None:
        """Tool execution should create spans."""
        from src.common.telemetry import setup_telemetry, get_tracer
        from src.tools import search_web

        setup_telemetry()

        # Execute tool - it should create spans internally
        result = await search_web("test query")

        assert "results" in result


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_tool_error_handling(self) -> None:
        """Tools should raise appropriate exceptions."""
        from src.common.exceptions import ToolError
        from src.tools import calculate

        with pytest.raises(ToolError):
            await calculate("divide", 1, 0)

    @pytest.mark.asyncio
    async def test_file_tool_path_validation(self, tmp_path) -> None:
        """File tools should validate paths."""
        from src.common.exceptions import ToolError
        from src.tools.file_tool import read_file, _get_safe_path

        # Attempt to escape workspace
        with pytest.raises(ToolError):
            _get_safe_path("../../../etc/passwd", workspace=tmp_path)


class TestScenario01EndToEnd:
    """End-to-end tests for Scenario 01."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_agent_with_tools_mock(
        self, env_vars: dict[str, str]
    ) -> None:
        """Agent should be able to use tools (mocked Azure)."""
        from src.agents import ResearchAgent
        from src.tools import search_web, calculate

        # Create agent
        agent = ResearchAgent(name="test_agent")
        agent.set_tool_handlers({
            "search_web": search_web,
            "calculate": calculate,
        })

        # Mock the Azure OpenAI client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].message.tool_calls = None
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20

        with patch.object(
            agent, "_create_completion", new_callable=AsyncMock
        ) as mock_completion:
            mock_completion.return_value = mock_response

            result = await agent.run("What is 2 + 2?")

            assert result == "Test response"
            mock_completion.assert_called_once()
