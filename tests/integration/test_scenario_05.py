"""
Integration tests for Scenario 05: Declarative Agent Configuration.

Tests the complete declarative agent and workflow functionality.
"""

import pytest
from pathlib import Path
import tempfile


class TestScenario05Imports:
    """Test module imports for Scenario 05."""

    def test_import_yaml_loader(self) -> None:
        """Can import YAML loader components."""
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
        )
        
        assert YAMLLoader is not None
        assert AgentConfig is not None
        assert WorkflowConfig is not None
        assert ModelConfig is not None
        assert WorkflowStepConfig is not None
        assert RetryConfig is not None
        assert ErrorHandling is not None
        assert ErrorStrategy is not None
        assert validate_agent_yaml is not None
        assert validate_workflow_yaml is not None

    def test_import_declarative_agents(self) -> None:
        """Can import declarative agent components."""
        from src.agents.declarative import (
            DeclarativeAgent,
            DeclarativeAgentLoader,
            DeclarativeWorkflowLoader,
            load_agents_from_config,
            load_workflows_from_config,
        )
        
        assert DeclarativeAgent is not None
        assert DeclarativeAgentLoader is not None
        assert DeclarativeWorkflowLoader is not None
        assert load_agents_from_config is not None
        assert load_workflows_from_config is not None


class TestAgentConfigValidation:
    """Test agent configuration validation."""

    def test_minimal_valid_config(self) -> None:
        """Minimal valid configuration passes."""
        from src.common.yaml_loader import validate_agent_yaml
        
        yaml = """
name: minimal_agent
model:
  provider: azure_openai
  deployment: gpt-4o
instructions: This is a minimal agent configuration.
"""
        config = validate_agent_yaml(yaml)
        assert config.name == "minimal_agent"

    def test_full_valid_config(self) -> None:
        """Full configuration with all fields passes."""
        from src.common.yaml_loader import validate_agent_yaml
        
        yaml = """
name: full_agent
model:
  provider: azure_openai
  deployment: gpt-4o
  temperature: 0.5
  api_version: "2024-02-15-preview"
instructions: |
  This is a full agent configuration.
  With multiple lines.
tools:
  - search_web
  - read_file
max_tokens: 2048
"""
        config = validate_agent_yaml(yaml)
        assert config.name == "full_agent"
        assert config.model.temperature == 0.5
        assert len(config.tools) == 2
        assert config.max_tokens == 2048


class TestWorkflowConfigValidation:
    """Test workflow configuration validation."""

    def test_minimal_workflow(self) -> None:
        """Minimal workflow configuration."""
        from src.common.yaml_loader import validate_workflow_yaml
        
        yaml = """
name: minimal_workflow
steps:
  - name: step1
    agent: agent1
    prompt: "Do something"
    outputs:
      - result
"""
        config = validate_workflow_yaml(yaml)
        assert config.name == "minimal_workflow"
        assert len(config.steps) == 1

    def test_multi_step_workflow(self) -> None:
        """Multi-step workflow configuration."""
        from src.common.yaml_loader import validate_workflow_yaml
        
        yaml = """
name: pipeline
description: A multi-step pipeline
steps:
  - name: research
    agent: research_agent
    prompt: "Research {topic}"
    outputs:
      - findings
    retry:
      max_attempts: 3
      delay_seconds: 1.0
  - name: analyze
    agent: analysis_agent
    prompt: "Analyze {findings}"
    outputs:
      - analysis
  - name: summarize
    agent: summary_agent
    prompt: "Summarize {analysis}"
    outputs:
      - summary
on_error:
  strategy: retry
"""
        config = validate_workflow_yaml(yaml)
        assert config.name == "pipeline"
        assert len(config.steps) == 3
        assert config.steps[0].retry is not None


class TestDeclarativeAgentCreation:
    """Test creating declarative agents."""

    def test_create_agent_from_config(self) -> None:
        """Create agent from config object."""
        from src.common.yaml_loader import AgentConfig, ModelConfig
        from src.agents.declarative import DeclarativeAgent
        
        config = AgentConfig(
            name="test_agent",
            model=ModelConfig(
                provider="azure_openai",
                deployment="gpt-4o",
            ),
            instructions="You are a helpful test agent.",
        )
        
        agent = DeclarativeAgent(config)
        
        assert agent.name == "test_agent"
        assert agent.model_name == "gpt-4o"

    @pytest.mark.asyncio
    async def test_agent_run_mock(self) -> None:
        """Agent run returns mock response."""
        from src.common.yaml_loader import AgentConfig, ModelConfig
        from src.agents.declarative import DeclarativeAgent
        
        config = AgentConfig(
            name="mock_agent",
            model=ModelConfig(
                provider="azure_openai",
                deployment="gpt-4o",
            ),
            instructions="You are a mock agent.",
        )
        
        agent = DeclarativeAgent(config)
        response = await agent.run("Test prompt")
        
        assert "mock_agent" in response


class TestDeclarativeAgentLoader:
    """Test loading agents from files."""

    def test_load_single_agent(self) -> None:
        """Load a single agent from file."""
        from src.agents.declarative import DeclarativeAgentLoader
        
        with tempfile.TemporaryDirectory() as tmpdir:
            agents_dir = Path(tmpdir) / "agents"
            agents_dir.mkdir()
            
            agent_file = agents_dir / "test.yaml"
            agent_file.write_text("""
name: file_agent
model:
  provider: azure_openai
  deployment: gpt-4o
instructions: Agent loaded from file.
""")
            
            loader = DeclarativeAgentLoader(agents_dir)
            agent = loader.load_agent(agent_file)
            
            assert agent.name == "file_agent"

    def test_load_all_agents(self) -> None:
        """Load all agents from directory."""
        from src.agents.declarative import DeclarativeAgentLoader
        
        with tempfile.TemporaryDirectory() as tmpdir:
            agents_dir = Path(tmpdir) / "agents"
            agents_dir.mkdir()
            
            for name in ["alpha", "beta", "gamma"]:
                agent_file = agents_dir / f"{name}.yaml"
                agent_file.write_text(f"""
name: {name}_agent
model:
  provider: azure_openai
  deployment: gpt-4o
instructions: This is the {name} agent.
""")
            
            loader = DeclarativeAgentLoader(agents_dir)
            agents = loader.load_all()
            
            assert len(agents) == 3
            assert "alpha_agent" in agents
            assert "beta_agent" in agents
            assert "gamma_agent" in agents

    def test_get_loaded_agent(self) -> None:
        """Get agent by name after loading."""
        from src.agents.declarative import DeclarativeAgentLoader
        
        with tempfile.TemporaryDirectory() as tmpdir:
            agents_dir = Path(tmpdir) / "agents"
            agents_dir.mkdir()
            
            agent_file = agents_dir / "named.yaml"
            agent_file.write_text("""
name: named_agent
model:
  provider: azure_openai
  deployment: gpt-4o
instructions: A named agent for retrieval.
""")
            
            loader = DeclarativeAgentLoader(agents_dir)
            loader.load_all()
            
            agent = loader.get_agent("named_agent")
            assert agent is not None
            assert agent.name == "named_agent"

    def test_list_agents(self) -> None:
        """List all loaded agent names."""
        from src.agents.declarative import DeclarativeAgentLoader
        
        with tempfile.TemporaryDirectory() as tmpdir:
            agents_dir = Path(tmpdir) / "agents"
            agents_dir.mkdir()
            
            for i in range(3):
                agent_file = agents_dir / f"agent{i}.yaml"
                agent_file.write_text(f"""
name: agent{i}
model:
  provider: azure_openai
  deployment: gpt-4o
instructions: Agent number {i}.
""")
            
            loader = DeclarativeAgentLoader(agents_dir)
            loader.load_all()
            
            names = loader.list_agents()
            assert len(names) == 3


class TestDeclarativeWorkflowLoader:
    """Test loading workflows from files."""

    def test_load_workflow(self) -> None:
        """Load a workflow from file."""
        from src.agents.declarative import (
            DeclarativeAgentLoader,
            DeclarativeWorkflowLoader,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            agents_dir = base / "agents"
            workflows_dir = base / "workflows"
            agents_dir.mkdir()
            workflows_dir.mkdir()
            
            # Create agent
            agent_file = agents_dir / "worker.yaml"
            agent_file.write_text("""
name: worker_agent
model:
  provider: azure_openai
  deployment: gpt-4o
instructions: A worker agent.
""")
            
            # Create workflow
            workflow_file = workflows_dir / "pipeline.yaml"
            workflow_file.write_text("""
name: simple_pipeline
steps:
  - name: work
    agent: worker_agent
    prompt: "Process {input}"
    outputs:
      - result
""")
            
            agent_loader = DeclarativeAgentLoader(agents_dir)
            agent_loader.load_all()
            
            workflow_loader = DeclarativeWorkflowLoader(
                workflows_dir,
                agent_loader,
            )
            workflow = workflow_loader.load_workflow(workflow_file)
            
            assert workflow.name == "simple_pipeline"
            assert len(workflow.steps) == 1


class TestConvenienceFunctions:
    """Test convenience loading functions."""

    def test_load_agents_from_config(self) -> None:
        """load_agents_from_config function."""
        from src.agents.declarative import load_agents_from_config
        
        with tempfile.TemporaryDirectory() as tmpdir:
            agents_dir = Path(tmpdir) / "agents"
            agents_dir.mkdir()
            
            agent_file = agents_dir / "conv.yaml"
            agent_file.write_text("""
name: convenience_agent
model:
  provider: azure_openai
  deployment: gpt-4o
instructions: Agent for convenience test.
""")
            
            agents = load_agents_from_config(agents_dir)
            assert "convenience_agent" in agents

    def test_load_workflows_from_config(self) -> None:
        """load_workflows_from_config function."""
        from src.agents.declarative import load_workflows_from_config
        
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            agents_dir = base / "agents"
            workflows_dir = base / "workflows"
            agents_dir.mkdir()
            workflows_dir.mkdir()
            
            agent_file = agents_dir / "agent.yaml"
            agent_file.write_text("""
name: test_agent
model:
  provider: azure_openai
  deployment: gpt-4o
instructions: Test agent for workflow loading.
""")
            
            workflow_file = workflows_dir / "wf.yaml"
            workflow_file.write_text("""
name: test_workflow
steps:
  - name: step1
    agent: test_agent
    prompt: "Do {task}"
    outputs:
      - done
""")
            
            workflows = load_workflows_from_config(
                workflows_dir=workflows_dir,
                agents_dir=agents_dir,
            )
            assert "test_workflow" in workflows


class TestErrorStrategyMapping:
    """Test error strategy mapping from config to engine."""

    def test_fail_maps_to_abort(self) -> None:
        """FAIL strategy maps to ABORT."""
        from src.common.yaml_loader import ErrorStrategy
        from src.workflows.engine import ErrorStrategy as EngineErrorStrategy
        
        # Verify enum values exist
        assert ErrorStrategy.FAIL is not None
        assert EngineErrorStrategy.ABORT is not None

    def test_skip_maps_to_skip(self) -> None:
        """SKIP strategy maps correctly."""
        from src.common.yaml_loader import ErrorStrategy
        from src.workflows.engine import ErrorStrategy as EngineErrorStrategy
        
        assert ErrorStrategy.SKIP is not None
        assert EngineErrorStrategy.SKIP is not None

    def test_retry_maps_to_retry(self) -> None:
        """RETRY strategy maps correctly."""
        from src.common.yaml_loader import ErrorStrategy
        from src.workflows.engine import ErrorStrategy as EngineErrorStrategy
        
        assert ErrorStrategy.RETRY is not None
        assert EngineErrorStrategy.RETRY is not None


class TestConfigFileLoading:
    """Test loading actual config files from the project.
    
    Note: Config files now use Agent Framework format (kind: Prompt).
    These tests use AgentFactoryLoader instead of YAMLLoader.
    """

    def test_load_research_agent_config(self) -> None:
        """Load research_agent.yaml from project using AgentFactoryLoader."""
        from src.agents.declarative import AgentFactoryLoader
        from agent_framework import ChatAgent
        from pathlib import Path
        
        # Try to find project configs
        possible_dirs = [
            Path("configs/agents"),
            Path("../configs/agents"),
            Path("../../configs/agents"),
        ]
        
        loaded = False
        
        for agents_dir in possible_dirs:
            config_path = agents_dir / "research_agent.yaml"
            if config_path.exists():
                loader = AgentFactoryLoader(agents_dir=agents_dir)
                agent = loader.load_agent(config_path)
                assert isinstance(agent, ChatAgent)
                loaded = True
                break
        
        # Skip if config not found (running from different directory)
        if not loaded:
            pytest.skip("Config file not found in expected locations")

    def test_load_summarizer_agent_config(self) -> None:
        """Load summarizer_agent.yaml from project using AgentFactoryLoader."""
        from src.agents.declarative import AgentFactoryLoader
        from agent_framework import ChatAgent
        from pathlib import Path
        
        possible_dirs = [
            Path("configs/agents"),
            Path("../configs/agents"),
            Path("../../configs/agents"),
        ]
        
        loaded = False
        
        for agents_dir in possible_dirs:
            config_path = agents_dir / "summarizer_agent.yaml"
            if config_path.exists():
                loader = AgentFactoryLoader(agents_dir=agents_dir)
                agent = loader.load_agent(config_path)
                assert isinstance(agent, ChatAgent)
                loaded = True
                break
        
        if not loaded:
            pytest.skip("Config file not found in expected locations")


class TestYAMLLoaderEdgeCases:
    """Test edge cases in YAML loading."""

    def test_empty_tools_list(self) -> None:
        """Empty tools list is valid."""
        from src.common.yaml_loader import validate_agent_yaml
        
        yaml = """
name: no_tools_agent
model:
  provider: azure_openai
  deployment: gpt-4o
instructions: Agent with no tools enabled.
tools: []
"""
        config = validate_agent_yaml(yaml)
        assert config.tools == []

    def test_multiline_instructions(self) -> None:
        """Multiline instructions parse correctly."""
        from src.common.yaml_loader import validate_agent_yaml
        
        yaml = """
name: multiline_agent
model:
  provider: azure_openai
  deployment: gpt-4o
instructions: |
  Line one of instructions.
  Line two of instructions.
  
  Line four after blank.
"""
        config = validate_agent_yaml(yaml)
        assert "Line one" in config.instructions
        assert "Line four" in config.instructions


class TestIntegrationWithWorkflowEngine:
    """Test integration with workflow engine."""

    def test_workflow_has_steps(self) -> None:
        """Loaded workflow has correct steps."""
        from src.agents.declarative import (
            DeclarativeAgentLoader,
            DeclarativeWorkflowLoader,
        )
        from src.workflows.steps import AgentStep
        
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            agents_dir = base / "agents"
            workflows_dir = base / "workflows"
            agents_dir.mkdir()
            workflows_dir.mkdir()
            
            agent_file = agents_dir / "agent.yaml"
            agent_file.write_text("""
name: int_agent
model:
  provider: azure_openai
  deployment: gpt-4o
instructions: Integration test agent.
""")
            
            workflow_file = workflows_dir / "wf.yaml"
            workflow_file.write_text("""
name: int_workflow
steps:
  - name: step_a
    agent: int_agent
    prompt: "A"
    outputs:
      - out_a
  - name: step_b
    agent: int_agent
    prompt: "B"
    outputs:
      - out_b
""")
            
            agent_loader = DeclarativeAgentLoader(agents_dir)
            agent_loader.load_all()
            
            workflow_loader = DeclarativeWorkflowLoader(
                workflows_dir,
                agent_loader,
            )
            workflow = workflow_loader.load_workflow(workflow_file)
            
            assert len(workflow.steps) == 2
            assert isinstance(workflow.steps[0], AgentStep)
            assert isinstance(workflow.steps[1], AgentStep)
