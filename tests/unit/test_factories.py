"""Unit tests for agent factory functions.

Tests for src/agents/factories.py - factory functions for creating agents
with Microsoft Agent Framework patterns.
"""

from __future__ import annotations

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Any

# =============================================================================
# Test Factory Module Import
# =============================================================================


class TestFactoriesImports:
    """Test that factory module can be imported."""

    def test_import_factories(self) -> None:
        """Test importing factories module."""
        from src.agents import factories
        assert factories is not None

    def test_import_create_research_agent(self) -> None:
        """Test importing create_research_agent."""
        from src.agents.factories import create_research_agent
        assert create_research_agent is not None

    def test_import_create_summarizer_agent(self) -> None:
        """Test importing create_summarizer_agent."""
        from src.agents.factories import create_summarizer_agent
        assert create_summarizer_agent is not None

    def test_import_create_analysis_agent(self) -> None:
        """Test importing create_analysis_agent."""
        from src.agents.factories import create_analysis_agent
        assert create_analysis_agent is not None

    def test_import_create_coordinator_agent(self) -> None:
        """Test importing create_coordinator_agent."""
        from src.agents.factories import create_coordinator_agent
        assert create_coordinator_agent is not None

    def test_import_create_custom_agent(self) -> None:
        """Test importing create_custom_agent."""
        from src.agents.factories import create_custom_agent
        assert create_custom_agent is not None

    def test_import_agent_factories_registry(self) -> None:
        """Test importing AGENT_FACTORIES registry."""
        from src.agents.factories import AGENT_FACTORIES
        assert isinstance(AGENT_FACTORIES, dict)

    def test_import_get_agent_factory(self) -> None:
        """Test importing get_agent_factory."""
        from src.agents.factories import get_agent_factory
        assert get_agent_factory is not None


# =============================================================================
# Test Agent Factories Registry
# =============================================================================


class TestAgentFactoriesRegistry:
    """Test the AGENT_FACTORIES registry."""

    def test_registry_contains_research(self) -> None:
        """Test registry contains research agent."""
        from src.agents.factories import AGENT_FACTORIES
        assert "research" in AGENT_FACTORIES

    def test_registry_contains_summarizer(self) -> None:
        """Test registry contains summarizer agent."""
        from src.agents.factories import AGENT_FACTORIES
        assert "summarizer" in AGENT_FACTORIES

    def test_registry_contains_analysis(self) -> None:
        """Test registry contains analysis agent."""
        from src.agents.factories import AGENT_FACTORIES
        assert "analysis" in AGENT_FACTORIES

    def test_registry_contains_coordinator(self) -> None:
        """Test registry contains coordinator agent."""
        from src.agents.factories import AGENT_FACTORIES
        assert "coordinator" in AGENT_FACTORIES

    def test_registry_values_are_callable(self) -> None:
        """Test all registry values are callable."""
        from src.agents.factories import AGENT_FACTORIES
        for name, factory in AGENT_FACTORIES.items():
            assert callable(factory), f"Factory '{name}' should be callable"


# =============================================================================
# Test get_agent_factory
# =============================================================================


class TestGetAgentFactory:
    """Test the get_agent_factory function."""

    def test_get_research_factory(self) -> None:
        """Test getting research factory."""
        from src.agents.factories import get_agent_factory, create_research_agent
        factory = get_agent_factory("research")
        assert factory is create_research_agent

    def test_get_summarizer_factory(self) -> None:
        """Test getting summarizer factory."""
        from src.agents.factories import get_agent_factory, create_summarizer_agent
        factory = get_agent_factory("summarizer")
        assert factory is create_summarizer_agent

    def test_get_analysis_factory(self) -> None:
        """Test getting analysis factory."""
        from src.agents.factories import get_agent_factory, create_analysis_agent
        factory = get_agent_factory("analysis")
        assert factory is create_analysis_agent

    def test_get_coordinator_factory(self) -> None:
        """Test getting coordinator factory."""
        from src.agents.factories import get_agent_factory, create_coordinator_agent
        factory = get_agent_factory("coordinator")
        assert factory is create_coordinator_agent

    def test_invalid_agent_type_raises_error(self) -> None:
        """Test that unknown agent type raises ValueError."""
        from src.agents.factories import get_agent_factory
        with pytest.raises(ValueError) as exc_info:
            get_agent_factory("unknown_type")
        assert "Unknown agent type: unknown_type" in str(exc_info.value)
        assert "Available:" in str(exc_info.value)


# =============================================================================
# Test Factory Functions with Mocks
# =============================================================================


class TestCreateResearchAgent:
    """Test create_research_agent factory."""

    @patch("src.agents.factories.AzureOpenAIChatClient")
    @patch("src.agents.factories.ChatAgent")
    @patch("src.agents.factories.get_settings")
    def test_create_with_defaults(
        self, mock_settings: Mock, mock_chat_agent: Mock, mock_client: Mock
    ) -> None:
        """Test creating research agent with default settings."""
        from src.agents.factories import create_research_agent

        # Setup mock settings
        mock_settings.return_value = Mock(
            azure_openai_endpoint="https://test.openai.azure.com",
            azure_openai_deployment="gpt-4",
            azure_openai_api_version="2024-02-15-preview",
        )

        agent = create_research_agent()

        # Verify ChatAgent was called
        mock_chat_agent.assert_called_once()
        call_kwargs = mock_chat_agent.call_args.kwargs
        assert call_kwargs["name"] == "research_agent"
        assert "Research Agent" in call_kwargs["instructions"]

    @patch("src.agents.factories.AzureOpenAIChatClient")
    @patch("src.agents.factories.ChatAgent")
    @patch("src.agents.factories.get_settings")
    def test_create_with_custom_name(
        self, mock_settings: Mock, mock_chat_agent: Mock, mock_client: Mock
    ) -> None:
        """Test creating research agent with custom name."""
        from src.agents.factories import create_research_agent

        mock_settings.return_value = Mock(
            azure_openai_endpoint="https://test.openai.azure.com",
            azure_openai_deployment="gpt-4",
            azure_openai_api_version="2024-02-15-preview",
        )

        create_research_agent(name="custom_researcher")

        call_kwargs = mock_chat_agent.call_args.kwargs
        assert call_kwargs["name"] == "custom_researcher"

    @patch("src.agents.factories.AzureOpenAIChatClient")
    @patch("src.agents.factories.ChatAgent")
    @patch("src.agents.factories.get_settings")
    def test_create_with_custom_instructions(
        self, mock_settings: Mock, mock_chat_agent: Mock, mock_client: Mock
    ) -> None:
        """Test creating research agent with custom instructions."""
        from src.agents.factories import create_research_agent

        mock_settings.return_value = Mock(
            azure_openai_endpoint="https://test.openai.azure.com",
            azure_openai_deployment="gpt-4",
            azure_openai_api_version="2024-02-15-preview",
        )

        custom_instructions = "You are a custom research agent."
        create_research_agent(instructions=custom_instructions)

        call_kwargs = mock_chat_agent.call_args.kwargs
        assert call_kwargs["instructions"] == custom_instructions

    @patch("src.agents.factories.AzureOpenAIChatClient")
    @patch("src.agents.factories.ChatAgent")
    @patch("src.agents.factories.get_settings")
    def test_create_with_temperature(
        self, mock_settings: Mock, mock_chat_agent: Mock, mock_client: Mock
    ) -> None:
        """Test creating research agent with custom temperature."""
        from src.agents.factories import create_research_agent

        mock_settings.return_value = Mock(
            azure_openai_endpoint="https://test.openai.azure.com",
            azure_openai_deployment="gpt-4",
            azure_openai_api_version="2024-02-15-preview",
        )

        create_research_agent(temperature=0.9)

        call_kwargs = mock_chat_agent.call_args.kwargs
        assert call_kwargs["model_settings"]["temperature"] == 0.9


class TestCreateSummarizerAgent:
    """Test create_summarizer_agent factory."""

    @patch("src.agents.factories.AzureOpenAIChatClient")
    @patch("src.agents.factories.ChatAgent")
    @patch("src.agents.factories.get_settings")
    def test_create_with_defaults(
        self, mock_settings: Mock, mock_chat_agent: Mock, mock_client: Mock
    ) -> None:
        """Test creating summarizer agent with default settings."""
        from src.agents.factories import create_summarizer_agent

        mock_settings.return_value = Mock(
            azure_openai_endpoint="https://test.openai.azure.com",
            azure_openai_deployment="gpt-4",
            azure_openai_api_version="2024-02-15-preview",
        )

        agent = create_summarizer_agent()

        mock_chat_agent.assert_called_once()
        call_kwargs = mock_chat_agent.call_args.kwargs
        assert call_kwargs["name"] == "summarizer_agent"
        assert "Summarization Agent" in call_kwargs["instructions"]
        assert call_kwargs["model_settings"]["temperature"] == 0.5


class TestCreateAnalysisAgent:
    """Test create_analysis_agent factory."""

    @patch("src.agents.factories.AzureOpenAIChatClient")
    @patch("src.agents.factories.ChatAgent")
    @patch("src.agents.factories.get_settings")
    def test_create_with_defaults(
        self, mock_settings: Mock, mock_chat_agent: Mock, mock_client: Mock
    ) -> None:
        """Test creating analysis agent with default settings."""
        from src.agents.factories import create_analysis_agent

        mock_settings.return_value = Mock(
            azure_openai_endpoint="https://test.openai.azure.com",
            azure_openai_deployment="gpt-4",
            azure_openai_api_version="2024-02-15-preview",
        )

        agent = create_analysis_agent()

        mock_chat_agent.assert_called_once()
        call_kwargs = mock_chat_agent.call_args.kwargs
        assert call_kwargs["name"] == "analysis_agent"
        assert "Analysis Agent" in call_kwargs["instructions"]
        assert call_kwargs["model_settings"]["temperature"] == 0.3


class TestCreateCoordinatorAgent:
    """Test create_coordinator_agent factory."""

    @patch("src.agents.factories.AzureOpenAIChatClient")
    @patch("src.agents.factories.ChatAgent")
    @patch("src.agents.factories.get_settings")
    def test_create_with_defaults(
        self, mock_settings: Mock, mock_chat_agent: Mock, mock_client: Mock
    ) -> None:
        """Test creating coordinator agent with default settings."""
        from src.agents.factories import create_coordinator_agent

        mock_settings.return_value = Mock(
            azure_openai_endpoint="https://test.openai.azure.com",
            azure_openai_deployment="gpt-4",
            azure_openai_api_version="2024-02-15-preview",
        )

        agent = create_coordinator_agent()

        mock_chat_agent.assert_called_once()
        call_kwargs = mock_chat_agent.call_args.kwargs
        assert call_kwargs["name"] == "coordinator_agent"
        assert "Coordinator Agent" in call_kwargs["instructions"]


class TestCreateCustomAgent:
    """Test create_custom_agent factory."""

    @patch("src.agents.factories.AzureOpenAIChatClient")
    @patch("src.agents.factories.ChatAgent")
    @patch("src.agents.factories.get_settings")
    def test_create_custom_agent(
        self, mock_settings: Mock, mock_chat_agent: Mock, mock_client: Mock
    ) -> None:
        """Test creating custom agent."""
        from src.agents.factories import create_custom_agent

        mock_settings.return_value = Mock(
            azure_openai_endpoint="https://test.openai.azure.com",
            azure_openai_deployment="gpt-4",
            azure_openai_api_version="2024-02-15-preview",
        )

        agent = create_custom_agent(
            name="my_custom_agent",
            instructions="You are a custom assistant.",
            temperature=0.8,
        )

        mock_chat_agent.assert_called_once()
        call_kwargs = mock_chat_agent.call_args.kwargs
        assert call_kwargs["name"] == "my_custom_agent"
        assert call_kwargs["instructions"] == "You are a custom assistant."
        assert call_kwargs["model_settings"]["temperature"] == 0.8

    @patch("src.agents.factories.AzureOpenAIChatClient")
    @patch("src.agents.factories.ChatAgent")
    @patch("src.agents.factories.get_settings")
    def test_create_custom_agent_with_deployment(
        self, mock_settings: Mock, mock_chat_agent: Mock, mock_client: Mock
    ) -> None:
        """Test creating custom agent with specific deployment."""
        from src.agents.factories import create_custom_agent

        mock_settings.return_value = Mock(
            azure_openai_endpoint="https://test.openai.azure.com",
            azure_openai_deployment="gpt-4",
            azure_openai_api_version="2024-02-15-preview",
        )

        create_custom_agent(
            name="my_agent",
            instructions="Custom instructions",
            deployment="gpt-35-turbo",
        )

        # Verify the client was called with custom deployment
        mock_client.assert_called_once()
        call_kwargs = mock_client.call_args.kwargs
        assert call_kwargs["deployment"] == "gpt-35-turbo"


# =============================================================================
# Test Azure Client Creation
# =============================================================================


class TestAzureClientCreation:
    """Test _create_azure_client helper."""

    @patch("src.agents.factories.AzureOpenAIChatClient")
    @patch("src.agents.factories.get_settings")
    def test_creates_client_from_settings(
        self, mock_settings: Mock, mock_client: Mock
    ) -> None:
        """Test client creation uses settings."""
        from src.agents.factories import _create_azure_client

        mock_settings.return_value = Mock(
            azure_openai_endpoint="https://myendpoint.openai.azure.com",
            azure_openai_deployment="gpt-4o",
            azure_openai_api_version="2024-06-01",
        )

        _create_azure_client()

        mock_client.assert_called_once_with(
            endpoint="https://myendpoint.openai.azure.com",
            deployment="gpt-4o",
            api_version="2024-06-01",
        )

    @patch("src.agents.factories.AzureOpenAIChatClient")
    @patch("src.agents.factories.get_settings")
    def test_accepts_custom_settings(
        self, mock_settings: Mock, mock_client: Mock
    ) -> None:
        """Test client creation with custom settings."""
        from src.agents.factories import _create_azure_client

        custom_settings = Mock(
            azure_openai_endpoint="https://custom.openai.azure.com",
            azure_openai_deployment="custom-deployment",
            azure_openai_api_version="2024-01-01",
        )

        _create_azure_client(settings=custom_settings)

        mock_client.assert_called_once_with(
            endpoint="https://custom.openai.azure.com",
            deployment="custom-deployment",
            api_version="2024-01-01",
        )
        # Should not call get_settings when custom settings provided
        mock_settings.assert_not_called()

    @patch("src.agents.factories.AzureOpenAIChatClient")
    @patch("src.agents.factories.get_settings")
    def test_deployment_override(
        self, mock_settings: Mock, mock_client: Mock
    ) -> None:
        """Test client creation with deployment override."""
        from src.agents.factories import _create_azure_client

        mock_settings.return_value = Mock(
            azure_openai_endpoint="https://myendpoint.openai.azure.com",
            azure_openai_deployment="default-deployment",
            azure_openai_api_version="2024-06-01",
        )

        _create_azure_client(deployment="override-deployment")

        call_kwargs = mock_client.call_args.kwargs
        assert call_kwargs["deployment"] == "override-deployment"
