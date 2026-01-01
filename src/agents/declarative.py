"""
Declarative agent loader from YAML configuration.

Loads agent and workflow definitions from YAML files
and instantiates them at runtime.
"""

from pathlib import Path
from typing import Any, Callable, Optional

from opentelemetry import trace

from src.common.yaml_loader import (
    AgentConfig,
    WorkflowConfig,
    YAMLLoader,
    ModelConfig,
)
from src.workflows.steps import (
    WorkflowStep,
    AgentStep,
    WorkflowContext,
    StepResult,
    StepStatus,
)
from src.workflows.engine import (
    WorkflowEngine,
    ErrorConfig,
    ErrorStrategy as EngineErrorStrategy,
)
from src.common.yaml_loader import ErrorStrategy as ConfigErrorStrategy

tracer = trace.get_tracer(__name__)


class DeclarativeAgent:
    """Agent loaded from YAML configuration."""
    
    def __init__(
        self,
        config: AgentConfig,
        client: Optional[Any] = None,
    ) -> None:
        """Initialize from configuration."""
        self.config = config
        self.name = config.name
        self.instructions = config.instructions
        self.tools = config.tools
        self.max_tokens = config.max_tokens
        self.model = config.model
        self._client = client
    
    @property
    def model_name(self) -> str:
        """Get the model deployment name."""
        return self.model.deployment
    
    @property
    def temperature(self) -> float:
        """Get the model temperature."""
        return self.model.temperature
    
    @tracer.start_as_current_span("declarative_agent.run")
    async def run(self, prompt: str) -> str:
        """Execute the agent with a prompt."""
        span = trace.get_current_span()
        span.set_attribute("agent.name", self.name)
        span.set_attribute("agent.model", self.model_name)
        
        if self._client is None:
            # Mock response for workshop demonstrations
            return f"[{self.name}] Response to: {prompt[:100]}..."
        
        # Real implementation would use the client
        messages = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": prompt},
        ]
        
        response = await self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        
        return response.choices[0].message.content or ""
    
    def __repr__(self) -> str:
        """String representation."""
        return f"DeclarativeAgent(name={self.name!r}, model={self.model_name!r})"


class DeclarativeAgentLoader:
    """Load agents from YAML configuration files."""
    
    def __init__(
        self,
        agents_dir: Path | str = "configs/agents",
        client: Optional[Any] = None,
    ) -> None:
        """Initialize loader with agents directory."""
        self.agents_dir = Path(agents_dir)
        self._client = client
        self._yaml_loader = YAMLLoader()
        self._agents: dict[str, DeclarativeAgent] = {}
    
    @tracer.start_as_current_span("declarative_loader.load_agent")
    def load_agent(self, path: Path | str) -> DeclarativeAgent:
        """Load a single agent from YAML file."""
        config = self._yaml_loader.load_agent(path)
        agent = DeclarativeAgent(config, self._client)
        self._agents[agent.name] = agent
        return agent
    
    @tracer.start_as_current_span("declarative_loader.load_all")
    def load_all(self) -> dict[str, DeclarativeAgent]:
        """Load all agents from the agents directory."""
        configs = self._yaml_loader.load_all_agents(self.agents_dir)
        
        for name, config in configs.items():
            self._agents[name] = DeclarativeAgent(config, self._client)
        
        return self._agents
    
    def get_agent(self, name: str) -> Optional[DeclarativeAgent]:
        """Get a loaded agent by name."""
        return self._agents.get(name)
    
    def list_agents(self) -> list[str]:
        """List all loaded agent names."""
        return list(self._agents.keys())
    
    @property
    def agents(self) -> dict[str, DeclarativeAgent]:
        """Get all loaded agents."""
        return self._agents


class DeclarativeWorkflowLoader:
    """Load workflows from YAML configuration files."""
    
    def __init__(
        self,
        workflows_dir: Path | str = "configs/workflows",
        agent_loader: Optional[DeclarativeAgentLoader] = None,
    ) -> None:
        """Initialize loader with workflows directory."""
        self.workflows_dir = Path(workflows_dir)
        self._agent_loader = agent_loader or DeclarativeAgentLoader()
        self._yaml_loader = YAMLLoader()
        self._workflows: dict[str, WorkflowEngine] = {}
    
    @tracer.start_as_current_span("workflow_loader.load_workflow")
    def load_workflow(self, path: Path | str) -> WorkflowEngine:
        """Load a single workflow from YAML file."""
        config = self._yaml_loader.load_workflow(path)
        engine = self._build_engine(config)
        self._workflows[config.name] = engine
        return engine
    
    @tracer.start_as_current_span("workflow_loader.load_all")
    def load_all(self) -> dict[str, WorkflowEngine]:
        """Load all workflows from the workflows directory."""
        configs = self._yaml_loader.load_all_workflows(self.workflows_dir)
        
        for name, config in configs.items():
            self._workflows[name] = self._build_engine(config)
        
        return self._workflows
    
    def _build_engine(self, config: WorkflowConfig) -> WorkflowEngine:
        """Build a workflow engine from configuration."""
        # Map config error strategy to engine error strategy
        strategy_map = {
            ConfigErrorStrategy.FAIL: EngineErrorStrategy.ABORT,
            ConfigErrorStrategy.SKIP: EngineErrorStrategy.SKIP,
            ConfigErrorStrategy.RETRY: EngineErrorStrategy.RETRY,
        }
        
        error_config = ErrorConfig(
            strategy=strategy_map.get(
                config.on_error.strategy,
                EngineErrorStrategy.ABORT,
            ),
            fallback_value=config.on_error.fallback_value,
        )
        
        engine = WorkflowEngine(
            name=config.name,
            description=config.description,
            error_config=error_config,
        )
        
        # Register agents
        for agent_name in {step.agent for step in config.steps}:
            agent = self._agent_loader.get_agent(agent_name)
            if agent:
                engine.register_agent(agent_name, agent)
        
        # Add steps
        for step_config in config.steps:
            step = AgentStep(
                name=step_config.name,
                agent_name=step_config.agent,
                prompt_template=step_config.prompt,
                output_vars=step_config.outputs,
            )
            engine.add_step(step)
        
        return engine
    
    def get_workflow(self, name: str) -> Optional[WorkflowEngine]:
        """Get a loaded workflow by name."""
        return self._workflows.get(name)
    
    def list_workflows(self) -> list[str]:
        """List all loaded workflow names."""
        return list(self._workflows.keys())
    
    @property
    def workflows(self) -> dict[str, WorkflowEngine]:
        """Get all loaded workflows."""
        return self._workflows


def load_agents_from_config(
    agents_dir: Path | str = "configs/agents",
    client: Optional[Any] = None,
) -> dict[str, DeclarativeAgent]:
    """Load all agents from configuration directory."""
    loader = DeclarativeAgentLoader(agents_dir, client)
    return loader.load_all()


def load_workflows_from_config(
    workflows_dir: Path | str = "configs/workflows",
    agents_dir: Path | str = "configs/agents",
    client: Optional[Any] = None,
) -> dict[str, WorkflowEngine]:
    """Load all workflows from configuration directory."""
    agent_loader = DeclarativeAgentLoader(agents_dir, client)
    agent_loader.load_all()
    
    workflow_loader = DeclarativeWorkflowLoader(workflows_dir, agent_loader)
    return workflow_loader.load_all()
