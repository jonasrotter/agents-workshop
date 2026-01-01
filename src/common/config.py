"""Configuration management for the Agentic AI Patterns Workshop.

This module provides centralized configuration using pydantic-settings
for type-safe environment variable loading with validation.

Example:
    from src.common.config import get_settings

    settings = get_settings()
    print(f"Using endpoint: {settings.azure_openai_endpoint}")
"""

from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    All settings can be overridden via environment variables or a .env file.
    Variable names are case-insensitive.

    Attributes:
        azure_openai_endpoint: Azure OpenAI service endpoint URL.
        azure_openai_api_key: API key for Azure OpenAI (optional if using CLI auth).
        azure_openai_version: API version for Azure OpenAI.
        completion_deployment_name: Primary model deployment name.
        medium_deployment_model_name: Medium-tier model deployment.
        small_deployment_model_name: Small/fast model deployment.
        responses_deployment_name: Model for response generation.
        applicationinsights_connection_string: Azure Monitor connection string.
        enable_otel: Whether to enable OpenTelemetry tracing.
        enable_sensitive_data: Include sensitive data in traces (dev only).
        log_level: Logging verbosity level.
        mcp_server_port: Port for MCP tool server.
        agui_server_port: Port for AG-UI streaming server.
        a2a_server_port: Port for A2A protocol server.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Azure OpenAI Configuration
    azure_openai_endpoint: str = Field(
        default="",
        description="Azure OpenAI service endpoint URL",
    )
    azure_openai_api_key: Optional[str] = Field(
        default=None,
        description="API key for Azure OpenAI (optional if using CLI auth)",
    )
    azure_openai_version: str = Field(
        default="2024-10-01-preview",
        description="API version for Azure OpenAI",
    )

    # Model Deployments
    completion_deployment_name: str = Field(
        default="gpt-4o",
        description="Primary model deployment name",
    )
    medium_deployment_model_name: str = Field(
        default="gpt-4o-mini",
        description="Medium-tier model deployment",
    )
    small_deployment_model_name: str = Field(
        default="gpt-4o-mini",
        description="Small/fast model deployment",
    )
    responses_deployment_name: str = Field(
        default="gpt-4o",
        description="Model for response generation",
    )

    # Azure Monitor
    applicationinsights_connection_string: Optional[str] = Field(
        default=None,
        description="Azure Monitor Application Insights connection string",
    )

    # Observability Options
    enable_otel: bool = Field(
        default=True,
        description="Enable OpenTelemetry tracing",
    )
    enable_sensitive_data: bool = Field(
        default=False,
        description="Include sensitive data in traces (dev only!)",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )

    # Server Ports
    mcp_server_port: int = Field(
        default=8001,
        ge=1024,
        le=65535,
        description="Port for MCP tool server",
    )
    agui_server_port: int = Field(
        default=8888,
        ge=1024,
        le=65535,
        description="Port for AG-UI streaming server",
    )
    a2a_server_port: int = Field(
        default=8080,
        ge=1024,
        le=65535,
        description="Port for A2A protocol server",
    )

    # Server URLs (for clients)
    local_mcp_agent_server_url: str = Field(
        default="http://localhost:8001/sse",
        description="URL for local MCP agent server",
    )
    agui_server_url: str = Field(
        default="http://localhost:8888",
        description="URL for AG-UI server",
    )
    a2a_agent_host: str = Field(
        default="http://localhost:8080",
        description="URL for A2A agent host",
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is a valid Python logging level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper_v = v.upper()
        if upper_v not in valid_levels:
            msg = f"log_level must be one of {valid_levels}, got '{v}'"
            raise ValueError(msg)
        return upper_v

    @property
    def is_azure_configured(self) -> bool:
        """Check if Azure OpenAI is properly configured."""
        return bool(self.azure_openai_endpoint)

    @property
    def is_observability_configured(self) -> bool:
        """Check if Azure Monitor observability is configured."""
        return bool(self.applicationinsights_connection_string)

    def get_azure_openai_config(self) -> dict[str, str]:
        """Get Azure OpenAI configuration as a dictionary.

        Returns:
            Dictionary with endpoint, api_key (if set), and version.
        """
        config: dict[str, str] = {
            "azure_endpoint": self.azure_openai_endpoint,
            "api_version": self.azure_openai_version,
        }
        if self.azure_openai_api_key:
            config["api_key"] = self.azure_openai_api_key
        return config


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings.

    Settings are loaded once and cached for performance.
    The cache can be cleared by calling get_settings.cache_clear().

    Returns:
        Settings instance with values from environment.

    Example:
        settings = get_settings()
        if settings.is_azure_configured:
            print(f"Azure endpoint: {settings.azure_openai_endpoint}")
    """
    return Settings()


def reload_settings() -> Settings:
    """Force reload settings from environment.

    Clears the settings cache and reloads from environment variables.
    Useful for testing or when environment changes at runtime.

    Returns:
        Fresh Settings instance.
    """
    get_settings.cache_clear()
    return get_settings()
