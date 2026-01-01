"""
YAML configuration loader with Pydantic validation.

Provides utilities for loading and validating agent and workflow
configurations from YAML files.
"""

from pathlib import Path
from typing import Any, Optional, TypeVar
from enum import Enum
import yaml

from pydantic import BaseModel, Field, field_validator
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

T = TypeVar("T", bound=BaseModel)


class ModelConfig(BaseModel):
    """LLM model configuration."""
    
    provider: str = Field(
        ...,
        description="Model provider: 'azure_openai' or 'openai'",
    )
    deployment: str = Field(
        ...,
        description="Model deployment name",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    api_version: Optional[str] = Field(
        default=None,
        description="API version for Azure OpenAI",
    )
    
    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Validate provider is supported."""
        allowed = {"azure_openai", "openai"}
        if v not in allowed:
            raise ValueError(f"provider must be one of {allowed}")
        return v


class AgentConfig(BaseModel):
    """Declarative agent configuration."""
    
    name: str = Field(
        ...,
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Unique agent identifier (lowercase, underscores)",
    )
    model: ModelConfig = Field(
        ...,
        description="LLM model configuration",
    )
    instructions: str = Field(
        ...,
        min_length=10,
        description="System prompt for the agent",
    )
    tools: list[str] = Field(
        default_factory=list,
        description="List of tool names to enable",
    )
    max_tokens: int = Field(
        default=4096,
        ge=1,
        le=128000,
        description="Maximum response tokens",
    )
    
    model_config = {"extra": "forbid"}


class ErrorStrategy(str, Enum):
    """Error handling strategies for workflows."""
    
    FAIL = "fail"
    SKIP = "skip"
    RETRY = "retry"


class RetryConfig(BaseModel):
    """Retry configuration for workflow steps."""
    
    max_attempts: int = Field(default=3, ge=1, le=10)
    delay_seconds: float = Field(default=1.0, ge=0.0)
    backoff_multiplier: float = Field(default=2.0, ge=1.0)


class ErrorHandling(BaseModel):
    """Error handling configuration."""
    
    strategy: ErrorStrategy = Field(default=ErrorStrategy.FAIL)
    fallback_value: Optional[str] = None


class WorkflowStepConfig(BaseModel):
    """Single step in a workflow."""
    
    name: str = Field(
        ...,
        min_length=1,
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Step identifier",
    )
    agent: str = Field(
        ...,
        description="Agent name to execute this step",
    )
    prompt: str = Field(
        ...,
        min_length=1,
        description="Prompt template with {variable} placeholders",
    )
    outputs: list[str] = Field(
        ...,
        min_length=1,
        description="Output variable names",
    )
    condition: Optional[str] = Field(
        default=None,
        description="Optional condition expression",
    )
    retry: Optional[RetryConfig] = Field(
        default=None,
        description="Retry configuration",
    )


class WorkflowConfig(BaseModel):
    """Declarative workflow configuration."""
    
    name: str = Field(
        ...,
        min_length=1,
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Unique workflow identifier",
    )
    description: Optional[str] = Field(
        default=None,
        description="Human-readable description",
    )
    steps: list[WorkflowStepConfig] = Field(
        ...,
        min_length=1,
        description="Ordered list of workflow steps",
    )
    on_error: ErrorHandling = Field(
        default_factory=ErrorHandling,
        description="Error handling strategy",
    )
    
    model_config = {"extra": "forbid"}


class YAMLLoader:
    """Load and validate YAML configuration files."""
    
    def __init__(self, base_path: Optional[Path] = None) -> None:
        """Initialize loader with optional base path."""
        self.base_path = base_path or Path.cwd()
    
    @tracer.start_as_current_span("yaml_loader.load_file")
    def load_file(self, path: Path | str) -> dict[str, Any]:
        """Load raw YAML content from file."""
        file_path = self._resolve_path(path)
        
        span = trace.get_current_span()
        span.set_attribute("yaml.file_path", str(file_path))
        
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            content = yaml.safe_load(f)
        
        if not isinstance(content, dict):
            raise ValueError(f"YAML file must contain a mapping: {file_path}")
        
        return content
    
    @tracer.start_as_current_span("yaml_loader.load_agent")
    def load_agent(self, path: Path | str) -> AgentConfig:
        """Load and validate an agent configuration."""
        data = self.load_file(path)
        return AgentConfig.model_validate(data)
    
    @tracer.start_as_current_span("yaml_loader.load_workflow")
    def load_workflow(self, path: Path | str) -> WorkflowConfig:
        """Load and validate a workflow configuration."""
        data = self.load_file(path)
        return WorkflowConfig.model_validate(data)
    
    @tracer.start_as_current_span("yaml_loader.load_all_agents")
    def load_all_agents(self, directory: Path | str) -> dict[str, AgentConfig]:
        """Load all agent configurations from a directory."""
        dir_path = self._resolve_path(directory)
        agents: dict[str, AgentConfig] = {}
        
        if not dir_path.exists():
            return agents
        
        for file_path in dir_path.glob("*.yaml"):
            try:
                agent = self.load_agent(file_path)
                agents[agent.name] = agent
            except Exception as e:
                # Log but continue loading other files
                span = trace.get_current_span()
                span.add_event(
                    "load_error",
                    {"file": str(file_path), "error": str(e)},
                )
        
        return agents
    
    @tracer.start_as_current_span("yaml_loader.load_all_workflows")
    def load_all_workflows(
        self,
        directory: Path | str,
    ) -> dict[str, WorkflowConfig]:
        """Load all workflow configurations from a directory."""
        dir_path = self._resolve_path(directory)
        workflows: dict[str, WorkflowConfig] = {}
        
        if not dir_path.exists():
            return workflows
        
        for file_path in dir_path.glob("*.yaml"):
            try:
                workflow = self.load_workflow(file_path)
                workflows[workflow.name] = workflow
            except Exception as e:
                span = trace.get_current_span()
                span.add_event(
                    "load_error",
                    {"file": str(file_path), "error": str(e)},
                )
        
        return workflows
    
    def _resolve_path(self, path: Path | str) -> Path:
        """Resolve path relative to base path."""
        path = Path(path)
        if path.is_absolute():
            return path
        return self.base_path / path
    
    @staticmethod
    def validate_yaml_string(
        content: str,
        schema_class: type[T],
    ) -> T:
        """Validate YAML string against a Pydantic model."""
        data = yaml.safe_load(content)
        return schema_class.model_validate(data)


def load_agent_config(path: Path | str) -> AgentConfig:
    """Convenience function to load an agent config."""
    loader = YAMLLoader()
    return loader.load_agent(path)


def load_workflow_config(path: Path | str) -> WorkflowConfig:
    """Convenience function to load a workflow config."""
    loader = YAMLLoader()
    return loader.load_workflow(path)


def validate_agent_yaml(content: str) -> AgentConfig:
    """Validate agent YAML content string."""
    return YAMLLoader.validate_yaml_string(content, AgentConfig)


def validate_workflow_yaml(content: str) -> WorkflowConfig:
    """Validate workflow YAML content string."""
    return YAMLLoader.validate_yaml_string(content, WorkflowConfig)
