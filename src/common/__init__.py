"""Common utilities for the Agentic AI Patterns Workshop.

This module provides shared functionality across all workshop scenarios:
- Configuration management (config.py)
- OpenTelemetry observability (telemetry.py)
- Custom exceptions (exceptions.py)

Example:
    from src.common import get_settings, setup_telemetry, WorkshopError

    settings = get_settings()
    tracer = setup_telemetry("my-scenario")
"""

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
from src.common.telemetry import get_tracer, setup_telemetry

__all__ = [
    # Configuration
    "Settings",
    "get_settings",
    # Telemetry
    "setup_telemetry",
    "get_tracer",
    # Exceptions
    "WorkshopError",
    "ConfigurationError",
    "AgentError",
    "ToolError",
    "MCPError",
    "WorkflowError",
    "AGUIError",
    "A2AError",
]
