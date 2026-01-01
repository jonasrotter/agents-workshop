"""
Deterministic Workflow Engine.

Provides the core engine for executing multi-agent workflows with:
- Sequential agent orchestration
- Data flow between steps
- Error handling strategies (retry, fallback, abort)
- Telemetry integration
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Callable
from pydantic import BaseModel, Field

from src.common.telemetry import get_tracer
from src.workflows.steps import (
    WorkflowStep,
    WorkflowContext,
    StepResult,
    StepStatus,
    RetryConfig,
    AgentStep,
)

tracer = get_tracer(__name__)


class ErrorStrategy(str, Enum):
    """Strategy for handling errors in workflows."""
    ABORT = "abort"  # Stop workflow on first error
    SKIP = "skip"    # Skip failed step and continue
    RETRY = "retry"  # Retry failed step according to config
    FALLBACK = "fallback"  # Use fallback value on failure


class WorkflowStatus(str, Enum):
    """Status of a workflow execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkflowResult:
    """Result of workflow execution."""
    workflow_name: str
    status: WorkflowStatus
    outputs: dict[str, Any] = field(default_factory=dict)
    step_results: dict[str, StepResult] = field(default_factory=dict)
    error: Optional[str] = None
    duration_ms: float = 0.0
    
    @property
    def succeeded(self) -> bool:
        """Check if workflow completed successfully."""
        return self.status == WorkflowStatus.COMPLETED
    
    def get_step_result(self, step_name: str) -> Optional[StepResult]:
        """Get result for a specific step."""
        return self.step_results.get(step_name)


class ErrorConfig(BaseModel):
    """Configuration for error handling."""
    strategy: ErrorStrategy = Field(default=ErrorStrategy.ABORT)
    fallback_value: Optional[Any] = None
    max_retries: int = Field(default=3, ge=1, le=10)
    retry_delay: float = Field(default=1.0, ge=0.0)


class WorkflowDefinition(BaseModel):
    """Definition of a workflow."""
    name: str = Field(..., min_length=1)
    description: str = Field(default="")
    on_error: ErrorConfig = Field(default_factory=ErrorConfig)


class WorkflowEngine:
    """
    Engine for executing deterministic multi-agent workflows.
    
    Supports:
    - Sequential step execution
    - Parallel step execution
    - Conditional branching
    - Error handling with retry/fallback
    - Telemetry integration
    
    Example:
        ```python
        engine = WorkflowEngine()
        engine.register_agent("research", research_agent)
        engine.register_agent("summarize", summarize_agent)
        
        engine.add_step(AgentStep(
            name="research",
            agent_name="research",
            prompt_template="Research: {topic}",
            output_vars=["findings"]
        ))
        engine.add_step(AgentStep(
            name="summarize",
            agent_name="summarize",
            prompt_template="Summarize: {findings}",
            output_vars=["summary"]
        ))
        
        result = await engine.execute({"topic": "AI agents"})
        ```
    """
    
    def __init__(
        self,
        name: str = "workflow",
        description: str = "",
        error_config: Optional[ErrorConfig] = None,
    ) -> None:
        """
        Initialize the workflow engine.
        
        Args:
            name: Workflow identifier
            description: Human-readable description
            error_config: Error handling configuration
        """
        self.name = name
        self.description = description
        self.error_config = error_config or ErrorConfig()
        self.steps: list[WorkflowStep] = []
        self.agents: dict[str, Any] = {}
        self._before_step_hooks: list[Callable] = []
        self._after_step_hooks: list[Callable] = []
    
    def register_agent(self, name: str, agent: Any) -> None:
        """
        Register an agent for use in the workflow.
        
        Args:
            name: Agent identifier
            agent: Agent instance
        """
        self.agents[name] = agent
    
    def add_step(self, step: WorkflowStep) -> "WorkflowEngine":
        """
        Add a step to the workflow.
        
        Args:
            step: Workflow step to add
            
        Returns:
            Self for method chaining
        """
        self.steps.append(step)
        return self
    
    def add_steps(self, steps: list[WorkflowStep]) -> "WorkflowEngine":
        """
        Add multiple steps to the workflow.
        
        Args:
            steps: List of workflow steps
            
        Returns:
            Self for method chaining
        """
        self.steps.extend(steps)
        return self
    
    def before_step(
        self,
        hook: Callable[[WorkflowStep, dict[str, Any]], None],
    ) -> None:
        """Register a hook to run before each step."""
        self._before_step_hooks.append(hook)
    
    def after_step(
        self,
        hook: Callable[[WorkflowStep, StepResult], None],
    ) -> None:
        """Register a hook to run after each step."""
        self._after_step_hooks.append(hook)
    
    async def execute(
        self,
        inputs: dict[str, Any],
        timeout: Optional[float] = None,
    ) -> WorkflowResult:
        """
        Execute the workflow.
        
        Args:
            inputs: Initial input data
            timeout: Optional execution timeout in seconds
            
        Returns:
            WorkflowResult with outputs and status
        """
        start = time.time()
        
        with tracer.start_as_current_span(f"workflow_{self.name}") as span:
            span.set_attribute("workflow.name", self.name)
            span.set_attribute("workflow.steps", len(self.steps))
            span.set_attribute("workflow.agents", len(self.agents))
            
            # Create execution context
            context = WorkflowContext(
                workflow_name=self.name,
                variables=inputs.copy(),
                metadata={"agents": self.agents},
            )
            
            current_inputs = inputs.copy()
            all_outputs: dict[str, Any] = {}
            
            try:
                if timeout:
                    result = await asyncio.wait_for(
                        self._execute_steps(current_inputs, context),
                        timeout=timeout,
                    )
                else:
                    result = await self._execute_steps(current_inputs, context)
                
                return result
                
            except asyncio.TimeoutError:
                return WorkflowResult(
                    workflow_name=self.name,
                    status=WorkflowStatus.FAILED,
                    step_results=context.step_results,
                    error="Workflow execution timed out",
                    duration_ms=(time.time() - start) * 1000,
                )
            except Exception as e:
                return WorkflowResult(
                    workflow_name=self.name,
                    status=WorkflowStatus.FAILED,
                    step_results=context.step_results,
                    error=str(e),
                    duration_ms=(time.time() - start) * 1000,
                )
    
    async def _execute_steps(
        self,
        inputs: dict[str, Any],
        context: WorkflowContext,
    ) -> WorkflowResult:
        """Execute all workflow steps sequentially."""
        start = time.time()
        current_inputs = inputs.copy()
        all_outputs: dict[str, Any] = {}
        
        for step in self.steps:
            # Run before hooks
            for hook in self._before_step_hooks:
                try:
                    hook(step, current_inputs)
                except Exception:
                    pass  # Hooks shouldn't break execution
            
            # Execute step with error handling
            result = await self._execute_step_with_retry(step, current_inputs, context)
            context.add_step_result(result)
            
            # Run after hooks
            for hook in self._after_step_hooks:
                try:
                    hook(step, result)
                except Exception:
                    pass
            
            # Handle step failure
            if not result.succeeded:
                if self.error_config.strategy == ErrorStrategy.ABORT:
                    return WorkflowResult(
                        workflow_name=self.name,
                        status=WorkflowStatus.FAILED,
                        outputs=all_outputs,
                        step_results=context.step_results,
                        error=f"Step '{step.name}' failed: {result.error}",
                        duration_ms=(time.time() - start) * 1000,
                    )
                elif self.error_config.strategy == ErrorStrategy.SKIP:
                    # Continue with current inputs
                    continue
                elif self.error_config.strategy == ErrorStrategy.FALLBACK:
                    if self.error_config.fallback_value is not None:
                        result.outputs = {"fallback": self.error_config.fallback_value}
            
            # Pass outputs to next step
            current_inputs.update(result.outputs)
            all_outputs.update(result.outputs)
            context.variables.update(result.outputs)
        
        return WorkflowResult(
            workflow_name=self.name,
            status=WorkflowStatus.COMPLETED,
            outputs=all_outputs,
            step_results=context.step_results,
            duration_ms=(time.time() - start) * 1000,
        )
    
    async def _execute_step_with_retry(
        self,
        step: WorkflowStep,
        inputs: dict[str, Any],
        context: WorkflowContext,
    ) -> StepResult:
        """Execute a step with retry logic."""
        max_attempts = step.retry.max_attempts if step.retry else 1
        delay = step.retry.delay_seconds if step.retry else 1.0
        backoff = step.retry.backoff_multiplier if step.retry else 2.0
        
        last_error: Optional[str] = None
        
        for attempt in range(max_attempts):
            with tracer.start_as_current_span(f"step_{step.name}") as span:
                span.set_attribute("step.name", step.name)
                span.set_attribute("step.attempt", attempt + 1)
                
                result = await step.execute(inputs, context)
                
                if result.succeeded:
                    return result
                
                last_error = result.error
                
                # Don't retry on last attempt
                if attempt < max_attempts - 1:
                    await asyncio.sleep(delay)
                    delay *= backoff
        
        return StepResult(
            step_name=step.name,
            status=StepStatus.FAILED,
            error=f"Failed after {max_attempts} attempts: {last_error}",
        )
    
    def validate(self) -> list[str]:
        """
        Validate the workflow configuration.
        
        Returns:
            List of validation error messages
        """
        errors: list[str] = []
        
        if not self.steps:
            errors.append("Workflow has no steps")
        
        # Check for duplicate step names
        step_names = [s.name for s in self.steps]
        duplicates = [n for n in step_names if step_names.count(n) > 1]
        if duplicates:
            errors.append(f"Duplicate step names: {set(duplicates)}")
        
        # Check agent steps have registered agents
        for step in self.steps:
            if isinstance(step, AgentStep):
                if step.agent_name not in self.agents:
                    errors.append(
                        f"Step '{step.name}' references unregistered agent "
                        f"'{step.agent_name}'"
                    )
        
        return errors


def create_workflow(
    name: str,
    steps: list[WorkflowStep],
    agents: Optional[dict[str, Any]] = None,
    description: str = "",
    error_strategy: ErrorStrategy = ErrorStrategy.ABORT,
) -> WorkflowEngine:
    """
    Factory function to create a workflow engine.
    
    Args:
        name: Workflow identifier
        steps: List of workflow steps
        agents: Optional dict of agent instances
        description: Human-readable description
        error_strategy: Error handling strategy
        
    Returns:
        Configured WorkflowEngine
    
    Example:
        ```python
        workflow = create_workflow(
            name="research_pipeline",
            steps=[
                AgentStep("search", "search_agent", "Search: {query}", ["results"]),
                AgentStep("analyze", "analysis_agent", "Analyze: {results}", ["analysis"]),
            ],
            agents={"search_agent": agent1, "analysis_agent": agent2}
        )
        result = await workflow.execute({"query": "AI trends"})
        ```
    """
    engine = WorkflowEngine(
        name=name,
        description=description,
        error_config=ErrorConfig(strategy=error_strategy),
    )
    
    engine.add_steps(steps)
    
    if agents:
        for agent_name, agent in agents.items():
            engine.register_agent(agent_name, agent)
    
    return engine


class WorkflowBuilder:
    """
    Fluent builder for creating workflows.
    
    Provides a more readable API for workflow construction.
    
    Example:
        ```python
        workflow = (
            WorkflowBuilder("my_workflow")
            .with_description("Research and summarize")
            .with_agent("researcher", research_agent)
            .with_agent("summarizer", summarize_agent)
            .add_agent_step(
                "research",
                agent="researcher",
                prompt="Research: {topic}",
                outputs=["findings"]
            )
            .add_agent_step(
                "summarize",
                agent="summarizer",
                prompt="Summarize: {findings}",
                outputs=["summary"]
            )
            .on_error(ErrorStrategy.RETRY)
            .build()
        )
        ```
    """
    
    def __init__(self, name: str) -> None:
        """Initialize the builder with a workflow name."""
        self._name = name
        self._description = ""
        self._steps: list[WorkflowStep] = []
        self._agents: dict[str, Any] = {}
        self._error_strategy = ErrorStrategy.ABORT
        self._error_config = ErrorConfig()
    
    def with_description(self, description: str) -> "WorkflowBuilder":
        """Set workflow description."""
        self._description = description
        return self
    
    def with_agent(self, name: str, agent: Any) -> "WorkflowBuilder":
        """Register an agent."""
        self._agents[name] = agent
        return self
    
    def add_step(self, step: WorkflowStep) -> "WorkflowBuilder":
        """Add a workflow step."""
        self._steps.append(step)
        return self
    
    def add_agent_step(
        self,
        name: str,
        agent: str,
        prompt: str,
        outputs: list[str],
        description: str = "",
        retry: Optional[RetryConfig] = None,
    ) -> "WorkflowBuilder":
        """Add an agent step."""
        step = AgentStep(
            name=name,
            agent_name=agent,
            prompt_template=prompt,
            output_vars=outputs,
            description=description,
            retry=retry,
        )
        self._steps.append(step)
        return self
    
    def on_error(
        self,
        strategy: ErrorStrategy,
        fallback_value: Optional[Any] = None,
    ) -> "WorkflowBuilder":
        """Configure error handling."""
        self._error_config = ErrorConfig(
            strategy=strategy,
            fallback_value=fallback_value,
        )
        return self
    
    def build(self) -> WorkflowEngine:
        """Build and return the workflow engine."""
        engine = WorkflowEngine(
            name=self._name,
            description=self._description,
            error_config=self._error_config,
        )
        
        engine.add_steps(self._steps)
        
        for name, agent in self._agents.items():
            engine.register_agent(name, agent)
        
        return engine
