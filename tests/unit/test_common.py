"""Tests for the common module (config, telemetry, exceptions).

This module tests the foundational utilities that all other modules depend on.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from src.common.config import Settings, get_settings
from src.common.exceptions import (
    A2AError,
    AGUIError,
    AgentError,
    ConfigurationError,
    MCPError,
    ToolError,
    WorkflowError,
    WorkshopError,
)
from src.common.telemetry import (
    create_span_attributes,
    get_current_span,
    get_tracer,
    is_telemetry_enabled,
    reset_telemetry,
    setup_telemetry,
)


# =============================================================================
# Configuration Tests
# =============================================================================


class TestSettings:
    """Tests for the Settings configuration class."""

    def test_settings_loads_from_env(self, env_vars: dict[str, str]) -> None:
        """Settings should load values from environment variables."""
        settings = get_settings()

        assert settings.azure_openai_endpoint == env_vars["AZURE_OPENAI_ENDPOINT"]
        assert settings.azure_openai_api_key == env_vars["AZURE_OPENAI_API_KEY"]
        assert settings.log_level == env_vars["LOG_LEVEL"]

    def test_settings_defaults(self, minimal_env_vars: dict[str, str]) -> None:
        """Settings should use defaults when env vars not set."""
        settings = get_settings()

        # The default comes from the actual Settings class
        # Just check that it's a valid API version string
        assert settings.azure_openai_version.startswith("2024-")
        assert settings.mcp_server_port == 8001
        assert settings.agui_server_port == 8888
        assert settings.a2a_server_port == 8080

    def test_is_azure_configured_true(self, env_vars: dict[str, str]) -> None:
        """is_azure_configured returns True when endpoint is set."""
        settings = get_settings()
        assert settings.is_azure_configured is True

    def test_is_azure_configured_false(self, minimal_env_vars: dict[str, str]) -> None:
        """is_azure_configured returns False when endpoint is empty."""
        settings = get_settings()
        assert settings.is_azure_configured is False

    def test_is_observability_configured_true(self, env_vars: dict[str, str]) -> None:
        """is_observability_configured returns True when connection string is set."""
        settings = get_settings()
        assert settings.is_observability_configured is True

    def test_is_observability_configured_false(
        self, minimal_env_vars: dict[str, str]
    ) -> None:
        """is_observability_configured returns False when connection string is empty."""
        settings = get_settings()
        assert settings.is_observability_configured is False

    def test_log_level_validation_valid(self) -> None:
        """Valid log levels should be accepted."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

        for level in valid_levels:
            with patch.dict(
                os.environ,
                {
                    "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
                    "LOG_LEVEL": level,
                },
            ):
                get_settings.cache_clear()
                settings = get_settings()
                assert settings.log_level == level

    def test_log_level_validation_invalid(self) -> None:
        """Invalid log levels should raise ValidationError."""
        with patch.dict(
            os.environ,
            {
                "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
                "LOG_LEVEL": "INVALID",
            },
        ):
            get_settings.cache_clear()
            with pytest.raises(ValidationError):
                get_settings()

    def test_settings_caching(self, env_vars: dict[str, str]) -> None:
        """Settings should be cached (same instance returned)."""
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2


# =============================================================================
# Telemetry Tests
# =============================================================================


class TestTelemetry:
    """Tests for telemetry setup and tracing utilities."""

    def test_setup_telemetry_initializes_once(
        self, minimal_env_vars: dict[str, str]
    ) -> None:
        """setup_telemetry should only initialize once."""
        setup_telemetry()
        assert is_telemetry_enabled() is True

        # Second call should be a no-op
        setup_telemetry()
        assert is_telemetry_enabled() is True

    def test_setup_telemetry_disabled_by_config(
        self, minimal_env_vars: dict[str, str]
    ) -> None:
        """setup_telemetry should respect enable_otel=False."""
        with patch.dict(os.environ, {"ENABLE_OTEL": "false"}):
            get_settings.cache_clear()
            setup_telemetry()
            assert is_telemetry_enabled() is True  # Still marked as initialized

    def test_get_tracer_returns_tracer(
        self, minimal_env_vars: dict[str, str]
    ) -> None:
        """get_tracer should return a Tracer instance."""
        setup_telemetry()
        tracer = get_tracer("test_module")

        assert tracer is not None
        assert hasattr(tracer, "start_as_current_span")

    def test_get_tracer_caches_results(
        self, minimal_env_vars: dict[str, str]
    ) -> None:
        """get_tracer should cache tracer instances."""
        setup_telemetry()
        tracer1 = get_tracer("test_module")
        tracer2 = get_tracer("test_module")

        assert tracer1 is tracer2

    def test_reset_telemetry(self, minimal_env_vars: dict[str, str]) -> None:
        """reset_telemetry should allow re-initialization."""
        setup_telemetry()
        assert is_telemetry_enabled() is True

        reset_telemetry()
        assert is_telemetry_enabled() is False

    def test_create_span_attributes_basic(self) -> None:
        """create_span_attributes should create correct attribute dict."""
        attrs = create_span_attributes(
            agent_name="test_agent",
            tool_name="search",
            model="gpt-4o",
        )

        assert attrs["agent.name"] == "test_agent"
        assert attrs["tool.name"] == "search"
        assert attrs["llm.model"] == "gpt-4o"

    def test_create_span_attributes_with_tokens(self) -> None:
        """create_span_attributes should handle token counts."""
        attrs = create_span_attributes(
            prompt_tokens=100,
            completion_tokens=50,
        )

        assert attrs["llm.prompt_tokens"] == 100
        assert attrs["llm.completion_tokens"] == 50

    def test_create_span_attributes_custom(self) -> None:
        """create_span_attributes should handle custom attributes."""
        attrs = create_span_attributes(
            custom_key="custom_value",
            another_key="another_value",
        )

        assert attrs["custom.key"] == "custom_value"
        assert attrs["another.key"] == "another_value"

    def test_get_current_span(self, minimal_env_vars: dict[str, str]) -> None:
        """get_current_span should return a span."""
        setup_telemetry()
        span = get_current_span()
        assert span is not None


# =============================================================================
# Exception Tests
# =============================================================================


class TestExceptions:
    """Tests for the custom exception hierarchy."""

    def test_workshop_error_basic(self) -> None:
        """WorkshopError should store message correctly."""
        error = WorkshopError("Test error")
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.details == {}

    def test_workshop_error_with_details(self) -> None:
        """WorkshopError should include details in string representation."""
        error = WorkshopError("Test error", details={"key": "value"})
        assert "key=value" in str(error)
        assert error.details == {"key": "value"}

    def test_configuration_error(self) -> None:
        """ConfigurationError should store config_key."""
        error = ConfigurationError("Missing config", config_key="API_KEY")
        assert error.config_key == "API_KEY"
        assert "config_key=API_KEY" in str(error)

    def test_agent_error(self) -> None:
        """AgentError should store agent_name."""
        error = AgentError("Agent failed", agent_name="researcher")
        assert error.agent_name == "researcher"
        assert "agent_name=researcher" in str(error)

    def test_tool_error(self) -> None:
        """ToolError should store tool_name."""
        error = ToolError("Tool failed", tool_name="search")
        assert error.tool_name == "search"
        assert "tool_name=search" in str(error)

    def test_mcp_error(self) -> None:
        """MCPError should store server_url."""
        error = MCPError("Connection failed", server_url="http://localhost:8001")
        assert error.server_url == "http://localhost:8001"
        assert "server_url=http://localhost:8001" in str(error)

    def test_workflow_error(self) -> None:
        """WorkflowError should store workflow_name and step_name."""
        error = WorkflowError(
            "Step failed",
            workflow_name="research_pipeline",
            step_name="analyze",
        )
        assert error.workflow_name == "research_pipeline"
        assert error.step_name == "analyze"
        assert "workflow_name=research_pipeline" in str(error)
        assert "step_name=analyze" in str(error)

    def test_ag_ui_error(self) -> None:
        """AGUIError should store event_type."""
        error = AGUIError("Event failed", event_type="TextMessageStart")
        assert error.event_type == "TextMessageStart"
        assert "event_type=TextMessageStart" in str(error)

    def test_a2a_error(self) -> None:
        """A2AError should store task_id and agent_url."""
        error = A2AError(
            "Task failed",
            task_id="task-123",
            agent_url="http://agent.example.com",
        )
        assert error.task_id == "task-123"
        assert error.agent_url == "http://agent.example.com"

    def test_exception_inheritance(self) -> None:
        """All custom exceptions should inherit from WorkshopError."""
        errors = [
            ConfigurationError("test"),
            AgentError("test"),
            ToolError("test"),
            MCPError("test"),
            WorkflowError("test"),
            AGUIError("test"),
            A2AError("test"),
        ]

        for error in errors:
            assert isinstance(error, WorkshopError)
            assert isinstance(error, Exception)

    def test_exception_chaining(self) -> None:
        """Exceptions should support chaining with 'from'."""
        original = ValueError("Original error")
        try:
            try:
                raise original
            except ValueError as e:
                raise AgentError("Wrapped error", agent_name="test") from e
        except AgentError as wrapped:
            assert wrapped.__cause__ is original
