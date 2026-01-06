"""
Unit tests for declarative agent and workflow loading.

Tests YAML schema validation and configuration loading.
"""

import pytest
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
import tempfile
import os

from src.common.yaml_loader import (
    YAMLLoader,
    AgentConfig,
    WorkflowConfig,
    ModelConfig,
    WorkflowStepConfig,
    RetryConfig,
    ErrorHandling,
    ErrorStrategy,
    validate_agent_yaml,
    validate_workflow_yaml,
    load_agent_config,
    load_workflow_config,
)
from src.agents.declarative import (
    DeclarativeAgent,
    DeclarativeAgentLoader,
    DeclarativeWorkflowLoader,
    load_agents_from_config,
    load_workflows_from_config,
)


class TestModelConfig:
    """Test ModelConfig schema validation."""

    def test_valid_azure_openai(self) -> None:
        """Valid Azure OpenAI config."""
        config = ModelConfig(
            provider="azure_openai",
            deployment="gpt-4o",
            temperature=0.7,
        )
        assert config.provider == "azure_openai"
        assert config.deployment == "gpt-4o"
        assert config.temperature == 0.7

    def test_valid_openai(self) -> None:
        """Valid OpenAI config."""
        config = ModelConfig(
            provider="openai",
            deployment="gpt-4",
        )
        assert config.provider == "openai"

    def test_invalid_provider(self) -> None:
        """Invalid provider raises error."""
        with pytest.raises(ValueError, match="provider"):
            ModelConfig(
                provider="invalid_provider",
                deployment="gpt-4",
            )

    def test_temperature_range(self) -> None:
        """Temperature must be 0.0-2.0."""
        with pytest.raises(ValueError):
            ModelConfig(
                provider="azure_openai",
                deployment="gpt-4",
                temperature=3.0,
            )


class TestAgentConfig:
    """Test AgentConfig schema validation."""

    def test_valid_agent_config(self) -> None:
        """Valid agent configuration."""
        config = AgentConfig(
            name="test_agent",
            model=ModelConfig(
                provider="azure_openai",
                deployment="gpt-4o",
            ),
            instructions="You are a helpful assistant for testing.",
        )
        assert config.name == "test_agent"
        assert config.max_tokens == 4096  # default

    def test_name_pattern_lowercase(self) -> None:
        """Name must be lowercase with underscores."""
        with pytest.raises(ValueError):
            AgentConfig(
                name="InvalidName",  # uppercase
                model=ModelConfig(
                    provider="azure_openai",
                    deployment="gpt-4",
                ),
                instructions="Test instructions here.",
            )

    def test_name_pattern_start_letter(self) -> None:
        """Name must start with letter."""
        with pytest.raises(ValueError):
            AgentConfig(
                name="123_agent",
                model=ModelConfig(
                    provider="azure_openai",
                    deployment="gpt-4",
                ),
                instructions="Test instructions here.",
            )

    def test_instructions_min_length(self) -> None:
        """Instructions must be at least 10 chars."""
        with pytest.raises(ValueError):
            AgentConfig(
                name="test_agent",
                model=ModelConfig(
                    provider="azure_openai",
                    deployment="gpt-4",
                ),
                instructions="short",  # too short
            )

    def test_extra_fields_forbidden(self) -> None:
        """Extra fields are not allowed."""
        with pytest.raises(ValueError):
            AgentConfig(
                name="test_agent",
                model=ModelConfig(
                    provider="azure_openai",
                    deployment="gpt-4",
                ),
                instructions="Test instructions here.",
                unknown_field="not allowed",
            )


class TestWorkflowStepConfig:
    """Test WorkflowStepConfig schema."""

    def test_valid_step(self) -> None:
        """Valid workflow step."""
        step = WorkflowStepConfig(
            name="research",
            agent="research_agent",
            prompt="Research {topic}",
            outputs=["findings"],
        )
        assert step.name == "research"
        assert step.condition is None  # optional

    def test_step_with_retry(self) -> None:
        """Step with retry configuration."""
        step = WorkflowStepConfig(
            name="research",
            agent="research_agent",
            prompt="Research {topic}",
            outputs=["findings"],
            retry=RetryConfig(
                max_attempts=5,
                delay_seconds=2.0,
            ),
        )
        assert step.retry.max_attempts == 5

    def test_step_requires_outputs(self) -> None:
        """Step must have at least one output."""
        with pytest.raises(ValueError):
            WorkflowStepConfig(
                name="research",
                agent="research_agent",
                prompt="Research {topic}",
                outputs=[],  # must have at least one
            )


class TestWorkflowConfig:
    """Test WorkflowConfig schema validation."""

    def test_valid_workflow(self) -> None:
        """Valid workflow configuration."""
        config = WorkflowConfig(
            name="test_workflow",
            steps=[
                WorkflowStepConfig(
                    name="step1",
                    agent="agent1",
                    prompt="Do something",
                    outputs=["result"],
                ),
            ],
        )
        assert config.name == "test_workflow"
        assert len(config.steps) == 1

    def test_workflow_with_error_handling(self) -> None:
        """Workflow with error handling."""
        config = WorkflowConfig(
            name="test_workflow",
            steps=[
                WorkflowStepConfig(
                    name="step1",
                    agent="agent1",
                    prompt="Do something",
                    outputs=["result"],
                ),
            ],
            on_error=ErrorHandling(
                strategy=ErrorStrategy.RETRY,
            ),
        )
        assert config.on_error.strategy == ErrorStrategy.RETRY

    def test_workflow_requires_steps(self) -> None:
        """Workflow must have at least one step."""
        with pytest.raises(ValueError):
            WorkflowConfig(
                name="empty_workflow",
                steps=[],
            )


class TestValidateAgentYaml:
    """Test validate_agent_yaml function."""

    def test_valid_yaml(self) -> None:
        """Valid YAML parses correctly."""
        yaml_content = """
name: test_agent
model:
  provider: azure_openai
  deployment: gpt-4o
instructions: |
  You are a test agent.
  Be helpful.
tools:
  - search
max_tokens: 2048
"""
        config = validate_agent_yaml(yaml_content)
        assert config.name == "test_agent"
        assert "search" in config.tools

    def test_invalid_yaml_syntax(self) -> None:
        """Invalid YAML syntax raises error."""
        invalid_yaml = """
name: test_agent
model:
  - this is invalid
  deployment: gpt-4
"""
        with pytest.raises(Exception):
            validate_agent_yaml(invalid_yaml)


class TestValidateWorkflowYaml:
    """Test validate_workflow_yaml function."""

    def test_valid_yaml(self) -> None:
        """Valid workflow YAML parses correctly."""
        yaml_content = """
name: test_pipeline
steps:
  - name: step1
    agent: agent1
    prompt: "Process {input}"
    outputs:
      - result
on_error:
  strategy: skip
"""
        config = validate_workflow_yaml(yaml_content)
        assert config.name == "test_pipeline"
        assert config.on_error.strategy == ErrorStrategy.SKIP


class TestYAMLLoader:
    """Test YAMLLoader class."""

    def test_load_file(self, tmp_path: Path) -> None:
        """Load raw YAML file."""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("key: value\nnested:\n  inner: data")
        
        loader = YAMLLoader(tmp_path)
        data = loader.load_file("test.yaml")
        
        assert data["key"] == "value"
        assert data["nested"]["inner"] == "data"

    def test_load_missing_file(self, tmp_path: Path) -> None:
        """Missing file raises FileNotFoundError."""
        loader = YAMLLoader(tmp_path)
        
        with pytest.raises(FileNotFoundError):
            loader.load_file("missing.yaml")

    def test_load_agent(self, tmp_path: Path) -> None:
        """Load and validate agent config."""
        yaml_file = tmp_path / "agent.yaml"
        yaml_file.write_text("""
name: test_agent
model:
  provider: azure_openai
  deployment: gpt-4o
instructions: This is a test agent for unit testing.
""")
        
        loader = YAMLLoader(tmp_path)
        config = loader.load_agent("agent.yaml")
        
        assert isinstance(config, AgentConfig)
        assert config.name == "test_agent"

    def test_load_workflow(self, tmp_path: Path) -> None:
        """Load and validate workflow config."""
        yaml_file = tmp_path / "workflow.yaml"
        yaml_file.write_text("""
name: test_workflow
steps:
  - name: step1
    agent: agent1
    prompt: "Do {task}"
    outputs:
      - result
""")
        
        loader = YAMLLoader(tmp_path)
        config = loader.load_workflow("workflow.yaml")
        
        assert isinstance(config, WorkflowConfig)
        assert config.name == "test_workflow"


class TestDeclarativeAgent:
    """Test DeclarativeAgent class."""

    def test_create_from_config(self) -> None:
        """Create agent from config."""
        config = AgentConfig(
            name="test_agent",
            model=ModelConfig(
                provider="azure_openai",
                deployment="gpt-4o",
                temperature=0.5,
            ),
            instructions="You are a helpful test agent.",
            tools=["search"],
            max_tokens=2048,
        )
        
        agent = DeclarativeAgent(config)
        
        assert agent.name == "test_agent"
        assert agent.model_name == "gpt-4o"
        assert agent.temperature == 0.5
        assert agent.tools == ["search"]
        assert agent.max_tokens == 2048

    @pytest.mark.asyncio
    async def test_run_mock(self) -> None:
        """Run returns mock response without client."""
        config = AgentConfig(
            name="test_agent",
            model=ModelConfig(
                provider="azure_openai",
                deployment="gpt-4o",
            ),
            instructions="You are a helpful test agent.",
        )
        
        agent = DeclarativeAgent(config)
        response = await agent.run("Hello")
        
        assert "test_agent" in response
        assert "Hello" in response


class TestDeclarativeAgentLoader:
    """Test DeclarativeAgentLoader class."""

    def test_load_agent(self, tmp_path: Path) -> None:
        """Load single agent."""
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        
        agent_file = agents_dir / "test_agent.yaml"
        agent_file.write_text("""
name: test_agent
model:
  provider: azure_openai
  deployment: gpt-4o
instructions: This is a test agent for loading tests.
""")
        
        loader = DeclarativeAgentLoader(agents_dir)
        agent = loader.load_agent(agent_file)
        
        assert agent.name == "test_agent"
        assert loader.get_agent("test_agent") is agent

    def test_load_all(self, tmp_path: Path) -> None:
        """Load all agents from directory."""
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        
        for i in range(3):
            agent_file = agents_dir / f"agent_{i}.yaml"
            agent_file.write_text(f"""
name: agent_{i}
model:
  provider: azure_openai
  deployment: gpt-4o
instructions: This is test agent number {i}.
""")
        
        loader = DeclarativeAgentLoader(agents_dir)
        agents = loader.load_all()
        
        assert len(agents) == 3
        assert "agent_0" in agents
        assert "agent_1" in agents
        assert "agent_2" in agents

    def test_list_agents(self, tmp_path: Path) -> None:
        """List loaded agent names."""
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        
        agent_file = agents_dir / "my_agent.yaml"
        agent_file.write_text("""
name: my_agent
model:
  provider: azure_openai
  deployment: gpt-4o
instructions: This is my test agent for listing.
""")
        
        loader = DeclarativeAgentLoader(agents_dir)
        loader.load_all()
        
        names = loader.list_agents()
        assert "my_agent" in names


class TestDeclarativeWorkflowLoader:
    """Test DeclarativeWorkflowLoader class."""

    def test_load_workflow(self, tmp_path: Path) -> None:
        """Load single workflow."""
        workflows_dir = tmp_path / "workflows"
        workflows_dir.mkdir()
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        
        # Create agent
        agent_file = agents_dir / "test_agent.yaml"
        agent_file.write_text("""
name: test_agent
model:
  provider: azure_openai
  deployment: gpt-4o
instructions: This is a test agent for workflow loading.
""")
        
        # Create workflow
        workflow_file = workflows_dir / "test_workflow.yaml"
        workflow_file.write_text("""
name: test_workflow
steps:
  - name: step1
    agent: test_agent
    prompt: "Do {task}"
    outputs:
      - result
""")
        
        agent_loader = DeclarativeAgentLoader(agents_dir)
        agent_loader.load_all()
        
        loader = DeclarativeWorkflowLoader(workflows_dir, agent_loader)
        workflow = loader.load_workflow(workflow_file)
        
        assert workflow.name == "test_workflow"
        assert len(workflow.steps) == 1

    def test_load_all(self, tmp_path: Path) -> None:
        """Load all workflows from directory."""
        workflows_dir = tmp_path / "workflows"
        workflows_dir.mkdir()
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        
        agent_file = agents_dir / "agent1.yaml"
        agent_file.write_text("""
name: agent1
model:
  provider: azure_openai
  deployment: gpt-4o
instructions: This is agent 1 for workflow loading tests.
""")
        
        for i in range(2):
            workflow_file = workflows_dir / f"workflow_{i}.yaml"
            workflow_file.write_text(f"""
name: workflow_{i}
steps:
  - name: step1
    agent: agent1
    prompt: "Process {'{'}input{'}'}"
    outputs:
      - result
""")
        
        agent_loader = DeclarativeAgentLoader(agents_dir)
        agent_loader.load_all()
        
        loader = DeclarativeWorkflowLoader(workflows_dir, agent_loader)
        workflows = loader.load_all()
        
        assert len(workflows) == 2


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_load_agent_config(self, tmp_path: Path) -> None:
        """load_agent_config convenience function."""
        agent_file = tmp_path / "agent.yaml"
        agent_file.write_text("""
name: conv_agent
model:
  provider: azure_openai
  deployment: gpt-4o
instructions: This is a convenience function test agent.
""")
        
        config = load_agent_config(agent_file)
        assert config.name == "conv_agent"

    def test_load_workflow_config(self, tmp_path: Path) -> None:
        """load_workflow_config convenience function."""
        workflow_file = tmp_path / "workflow.yaml"
        workflow_file.write_text("""
name: conv_workflow
steps:
  - name: step1
    agent: agent1
    prompt: "Do {task}"
    outputs:
      - result
""")
        
        config = load_workflow_config(workflow_file)
        assert config.name == "conv_workflow"

    def test_load_agents_from_config(self, tmp_path: Path) -> None:
        """load_agents_from_config convenience function."""
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        
        agent_file = agents_dir / "simple_agent.yaml"
        agent_file.write_text("""
name: simple_agent
model:
  provider: azure_openai
  deployment: gpt-4o
instructions: This is a simple agent for function testing.
""")
        
        agents = load_agents_from_config(agents_dir)
        assert "simple_agent" in agents


class TestRetryConfig:
    """Test RetryConfig schema."""

    def test_defaults(self) -> None:
        """Default retry values."""
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.delay_seconds == 1.0
        assert config.backoff_multiplier == 2.0

    def test_custom_values(self) -> None:
        """Custom retry values."""
        config = RetryConfig(
            max_attempts=5,
            delay_seconds=0.5,
            backoff_multiplier=1.5,
        )
        assert config.max_attempts == 5

    def test_max_attempts_limit(self) -> None:
        """Max attempts limited to 10."""
        with pytest.raises(ValueError):
            RetryConfig(max_attempts=15)


class TestErrorHandling:
    """Test ErrorHandling schema."""

    def test_defaults(self) -> None:
        """Default error handling."""
        config = ErrorHandling()
        assert config.strategy == ErrorStrategy.FAIL
        assert config.fallback_value is None

    def test_with_fallback(self) -> None:
        """Error handling with fallback."""
        config = ErrorHandling(
            strategy=ErrorStrategy.SKIP,
            fallback_value="default",
        )
        assert config.fallback_value == "default"


# =============================================================================
# Tests for Agent Framework Declarative (New Format)
# =============================================================================

# Note: These tests require Azure OpenAI credentials to create real ChatAgent instances.
# They use the AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY from .env
# The AgentFactory requires a real connection to create agents.

def _has_azure_credentials() -> bool:
    """Check if Azure OpenAI credentials are available."""
    import os
    from dotenv import load_dotenv
    load_dotenv()
    return bool(os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_API_KEY"))

def _get_azure_endpoint() -> str:
    """Get Azure OpenAI endpoint from environment."""
    import os
    from dotenv import load_dotenv
    load_dotenv()
    return os.getenv("AZURE_OPENAI_ENDPOINT", "")


requires_azure = pytest.mark.skipif(
    not _has_azure_credentials(),
    reason="Requires Azure OpenAI credentials (AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY)"
)


class TestAgentFactoryLoader:
    """Test AgentFactoryLoader class for agent-framework-declarative format."""

    @requires_azure
    def test_load_agent_new_format(self, tmp_path: Path) -> None:
        """Load agent using new kind: Prompt format."""
        from src.agents.declarative import AgentFactoryLoader
        
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        
        agent_file = agents_dir / "test_agent.yaml"
        agent_file.write_text(f"""
kind: Prompt
name: test_agent
description: A test agent for unit testing
model:
  id: gpt-4.1-mini
  provider: AzureOpenAI.Chat
  connection:
    kind: remote
    endpoint: {_get_azure_endpoint()}
  options:
    temperature: 0.7
instructions: |
  You are a helpful test agent for unit testing purposes.
tools: []
""")
        
        loader = AgentFactoryLoader(agents_dir)
        agent = loader.load_agent(agent_file)
        
        # Agent should be a ChatAgent from agent_framework
        assert agent is not None
        assert loader.get_agent("test_agent") is agent

    @requires_azure
    def test_load_all_new_format(self, tmp_path: Path) -> None:
        """Load all agents using new format."""
        from src.agents.declarative import AgentFactoryLoader
        
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        
        endpoint = _get_azure_endpoint()
        for i in range(2):
            agent_file = agents_dir / f"agent_{i}.yaml"
            agent_file.write_text(f"""
kind: Prompt
name: agent_{i}
model:
  id: gpt-4.1-mini
  provider: AzureOpenAI.Chat
  connection:
    kind: remote
    endpoint: {endpoint}
instructions: This is test agent number {i} for testing.
tools: []
""")
        
        loader = AgentFactoryLoader(agents_dir)
        agents = loader.load_all()
        
        assert len(agents) == 2
        assert "agent_0" in agents
        assert "agent_1" in agents

    @requires_azure
    def test_list_agents_new_format(self, tmp_path: Path) -> None:
        """List loaded agent names."""
        from src.agents.declarative import AgentFactoryLoader
        
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        
        agent_file = agents_dir / "my_agent.yaml"
        agent_file.write_text(f"""
kind: Prompt
name: my_agent
model:
  id: gpt-4.1-mini
  provider: AzureOpenAI.Chat
  connection:
    kind: remote
    endpoint: {_get_azure_endpoint()}
instructions: This is my test agent for listing in tests.
tools: []
""")
        
        loader = AgentFactoryLoader(agents_dir)
        loader.load_all()
        
        names = loader.list_agents()
        assert "my_agent" in names

    def test_loader_initialization(self, tmp_path: Path) -> None:
        """Test AgentFactoryLoader can be initialized."""
        from src.agents.declarative import AgentFactoryLoader
        
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        
        loader = AgentFactoryLoader(agents_dir)
        assert loader.agents_dir == agents_dir
        assert loader.list_agents() == []


class TestLoadAgentFromYaml:
    """Test load_agent_from_yaml convenience function."""

    @requires_azure
    def test_load_single_agent(self, tmp_path: Path) -> None:
        """Load single agent with convenience function."""
        from src.agents.declarative import load_agent_from_yaml
        
        agent_file = tmp_path / "agent.yaml"
        agent_file.write_text(f"""
kind: Prompt
name: convenience_agent
model:
  id: gpt-4.1-mini
  provider: AzureOpenAI.Chat
  connection:
    kind: remote
    endpoint: {_get_azure_endpoint()}
instructions: This agent tests the convenience function.
tools: []
""")
        
        agent = load_agent_from_yaml(agent_file)
        assert agent is not None


class TestLoadAgentsWithFactory:
    """Test load_agents_with_factory convenience function."""

    @requires_azure
    def test_load_all_agents(self, tmp_path: Path) -> None:
        """Load all agents with convenience function."""
        from src.agents.declarative import load_agents_with_factory
        
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        
        agent_file = agents_dir / "factory_agent.yaml"
        agent_file.write_text(f"""
kind: Prompt
name: factory_agent
model:
  id: gpt-4.1-mini
  provider: AzureOpenAI.Chat
  connection:
    kind: remote
    endpoint: {_get_azure_endpoint()}
instructions: This agent tests factory loading convenience.
tools: []
""")
        
        agents = load_agents_with_factory(agents_dir)
        assert "factory_agent" in agents
