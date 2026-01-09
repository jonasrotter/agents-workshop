"""Configuration management for the Agentic AI Patterns Workshop.

This module provides centralized configuration using pydantic-settings
for type-safe environment variable loading with validation.

Also provides helper functions for Azure AI Evaluation SDK configuration.

Example:
    from src.common.config import get_settings, get_model_config

    settings = get_settings()
    print(f"Using endpoint: {settings.azure_openai_endpoint}")
    
    # For SDK evaluators
    model_config = get_model_config()
    evaluator = RelevanceEvaluator(model_config)
"""

from functools import lru_cache
from typing import Any, Optional, TypedDict

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.common.exceptions import ConfigurationError


# =============================================================================
# TypedDicts for Azure AI Evaluation SDK
# =============================================================================


class ModelConfig(TypedDict):
    """Configuration for Azure OpenAI model used by AI-assisted evaluators.

    This matches the expected format for azure-ai-evaluation SDK evaluators.
    """
    azure_deployment: str
    api_key: str
    azure_endpoint: str
    api_version: str


class AzureAIProject(TypedDict, total=False):
    """Configuration for Azure AI Foundry project (optional).

    Used for logging evaluation results to Azure AI Foundry.
    """
    subscription_id: str
    resource_group_name: str
    project_name: str


# =============================================================================
# Settings Class
# =============================================================================


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
    azure_openai_api_version: str = Field(
        default="2024-10-01-preview",
        description="API version for Azure OpenAI",
    )

    # Model Deployments
    azure_openai_deployment: str = Field(
        default="gpt-5-nano",
        description="Primary model deployment name",
    )
    medium_deployment_model_name: str = Field(
        default="gpt-4.1-mini",
        description="Medium-tier model deployment",
    )
    small_deployment_model_name: str = Field(
        default="gpt-4.1-nano",
        description="Small/fast model deployment",
    )
    responses_deployment_name: str = Field(
        default="gpt-4.1-mini",
        description="Model for response generation",
    )

    # Azure AI Foundry (optional - for evaluation result logging)
    azure_ai_project_subscription_id: Optional[str] = Field(
        default=None,
        description="Azure subscription ID for AI Foundry project",
    )
    azure_ai_project_resource_group: Optional[str] = Field(
        default=None,
        description="Resource group name for AI Foundry project",
    )
    azure_ai_project_name: Optional[str] = Field(
        default=None,
        description="AI Foundry project name",
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

    @property
    def is_azure_ai_project_configured(self) -> bool:
        """Check if Azure AI Foundry project is configured."""
        return bool(
            self.azure_ai_project_subscription_id
            and self.azure_ai_project_resource_group
            and self.azure_ai_project_name
        )

    def get_azure_openai_config(self) -> dict[str, str]:
        """Get Azure OpenAI configuration as a dictionary.

        Returns:
            Dictionary with endpoint, api_key (if set), and version.
        """
        config: dict[str, str] = {
            "azure_endpoint": self.azure_openai_endpoint,
            "api_version": self.azure_openai_api_version,
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


# =============================================================================
# Azure AI Evaluation SDK Helper Functions
# =============================================================================


def get_model_config(
    *,
    azure_deployment: str | None = None,
    api_key: str | None = None,
    azure_endpoint: str | None = None,
    api_version: str | None = None,
    raise_on_missing: bool = True,
    settings: Settings | None = None,
) -> ModelConfig:
    """Get Azure OpenAI model configuration for SDK evaluators.

    Reads from Settings by default, with optional overrides.
    
    Supports both API key and Azure credential authentication:
    - If api_key is provided (via argument or AZURE_OPENAI_API_KEY), it will be used.
    - If api_key is empty, the SDK evaluators will use DefaultAzureCredential.
    
    Args:
        azure_deployment: Model deployment name. Falls back to settings.
        api_key: API key. Falls back to settings. Optional for credential auth.
        azure_endpoint: Azure OpenAI endpoint. Falls back to settings.
        api_version: API version. Falls back to settings.
        raise_on_missing: If True, raises ConfigurationError for missing values.
        settings: Optional Settings instance. If not provided, uses get_settings().

    Returns:
        ModelConfig dictionary ready for use with SDK evaluators.

    Raises:
        ConfigurationError: If raise_on_missing=True and required values missing.

    Example:
        >>> from src.common.config import get_model_config
        >>> config = get_model_config()
        >>> from azure.ai.evaluation import RelevanceEvaluator
        >>> evaluator = RelevanceEvaluator(config)
    """
    if settings is None:
        settings = get_settings()

    config: ModelConfig = {
        "azure_deployment": str(azure_deployment or settings.azure_openai_deployment or ""),
        "api_key": str(api_key or settings.azure_openai_api_key or ""),
        "azure_endpoint": str(azure_endpoint or settings.azure_openai_endpoint or ""),
        "api_version": str(api_version or settings.azure_openai_api_version or "2024-02-15-preview"),
    }

    if raise_on_missing:
        # api_key is optional (credential auth supported), but deployment and endpoint are required
        missing = [k for k, v in config.items() if not v and k not in ("api_version", "api_key")]
        if missing:
            env_var_map = {
                "azure_deployment": "AZURE_OPENAI_DEPLOYMENT",
                "azure_endpoint": "AZURE_OPENAI_ENDPOINT",
            }
            missing_vars = [env_var_map.get(k, k.upper()) for k in missing]
            raise ConfigurationError(
                f"Missing required environment variables: {', '.join(missing_vars)}. "
                f"Please set these in your .env file or pass them as arguments."
            )

    return config


def get_azure_ai_project(
    *,
    subscription_id: str | None = None,
    resource_group_name: str | None = None,
    project_name: str | None = None,
    settings: Settings | None = None,
) -> AzureAIProject | None:
    """Get Azure AI Foundry project configuration (optional).
    
    Used for logging evaluation results to Azure AI Foundry.
    Returns None if any required value is missing.
    
    Args:
        subscription_id: Azure subscription ID. Falls back to settings.
        resource_group_name: Resource group name. Falls back to settings.
        project_name: Project name. Falls back to settings.
        settings: Optional Settings instance. If not provided, uses get_settings().
    
    Returns:
        AzureAIProject dictionary if all values present, None otherwise.

    Example:
        >>> project = get_azure_ai_project()
        >>> if project:
        ...     result = evaluate(data=data, azure_ai_project=project, ...)
    """
    if settings is None:
        settings = get_settings()

    config: AzureAIProject = {
        "subscription_id": str(subscription_id or settings.azure_ai_project_subscription_id or ""),
        "resource_group_name": str(resource_group_name or settings.azure_ai_project_resource_group or ""),
        "project_name": str(project_name or settings.azure_ai_project_name or ""),
    }

    # Return None if any value is missing (Foundry is optional)
    if not all(config.values()):
        return None

    return config


def validate_model_config(config: ModelConfig) -> list[str]:
    """Validate model configuration and return list of issues.
    
    Note: api_key is optional as DefaultAzureCredential can be used instead.
    
    Args:
        config: ModelConfig dictionary to validate.
    
    Returns:
        List of validation error messages (empty if valid).
    """
    issues: list[str] = []

    if not config.get("azure_deployment"):
        issues.append("azure_deployment is required")

    # api_key is optional - DefaultAzureCredential can be used instead
    # if not config.get("api_key"):
    #     issues.append("api_key is required")

    if not config.get("azure_endpoint"):
        issues.append("azure_endpoint is required")
    elif not config["azure_endpoint"].startswith("https://"):
        issues.append("azure_endpoint must start with https://")

    if not config.get("api_version"):
        issues.append("api_version is required")

    return issues


def get_config_summary(config: ModelConfig) -> dict[str, Any]:
    """Get a safe summary of configuration (without sensitive values).
    
    Useful for logging and debugging without exposing API keys.
    
    Args:
        config: ModelConfig dictionary.
    
    Returns:
        Dictionary with masked sensitive values.
    """
    return {
        "azure_deployment": config.get("azure_deployment", ""),
        "api_key": "***" if config.get("api_key") else "(not set)",
        "azure_endpoint": config.get("azure_endpoint", ""),
        "api_version": config.get("api_version", ""),
    }
