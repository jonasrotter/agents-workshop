"""
Workflow Step Definitions.

.. deprecated:: 1.0.0
    This module is deprecated. Use Microsoft Agent Framework patterns instead:
    
    Before (deprecated):
        from src.workflows.steps import SequentialStep, ParallelStep
        seq = SequentialStep([step1, step2])
        par = ParallelStep([step_a, step_b])
    
    After (recommended):
        from agent_framework import WorkflowBuilder, ChatAgent
        import asyncio
        
        # Sequential: use chained add_edge()
        workflow = WorkflowBuilder().set_start_executor(a1).add_edge(a1, a2).build()
        
        # Parallel: use asyncio.gather()
        results = await asyncio.gather(agent1.run(x), agent2.run(x))
        
        # Conditional: use add_edge(condition=fn)
        workflow = WorkflowBuilder().add_edge(a1, a2, condition=my_fn).build()
    
    See notebooks/04_deterministic_workflows.ipynb for migration examples.

Provides step types for building deterministic multi-agent workflows:
- WorkflowStep: Base class for all steps
- SequentialStep: Executes steps in sequence
- ParallelStep: Executes steps in parallel
- ConditionalStep: Conditional branching
- DataTransform: Data transformation between steps
"""

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, TypeVar, Generic
from pydantic import BaseModel, Field

from src.common.telemetry import get_tracer

tracer = get_tracer(__name__)

# Emit deprecation warning when module is imported
warnings.warn(
    "src.workflows.steps is deprecated. Use Microsoft Agent Framework patterns instead. "
    "See notebooks/04_deterministic_workflows.ipynb for migration examples.",
    DeprecationWarning,
    stacklevel=2
)


class StepStatus(str, Enum):
    """Status of a workflow step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StepResult:
    """Result of executing a workflow step."""
    step_name: str
    status: StepStatus
    outputs: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    duration_ms: float = 0.0
    
    @property
    def succeeded(self) -> bool:
        """Check if step completed successfully."""
        return self.status == StepStatus.COMPLETED


class RetryConfig(BaseModel):
    """Retry configuration for workflow steps."""
    max_attempts: int = Field(default=3, ge=1, le=10)
    delay_seconds: float = Field(default=1.0, ge=0.0)
    backoff_multiplier: float = Field(default=2.0, ge=1.0)


class WorkflowStep(ABC):
    """
    Base class for workflow steps.
    
    All workflow steps inherit from this class and implement
    the execute method with their specific behavior.
    """
    
    def __init__(
        self,
        name: str,
        description: str = "",
        retry: Optional[RetryConfig] = None,
    ) -> None:
        """
        Initialize a workflow step.
        
        Args:
            name: Unique identifier for the step
            description: Human-readable description
            retry: Optional retry configuration
        """
        self.name = name
        self.description = description
        self.retry = retry or RetryConfig(max_attempts=1)
        self.status = StepStatus.PENDING
    
    @abstractmethod
    async def execute(
        self,
        inputs: dict[str, Any],
        context: "WorkflowContext",
    ) -> StepResult:
        """
        Execute the workflow step.
        
        Args:
            inputs: Input data for the step
            context: Workflow execution context
            
        Returns:
            StepResult with outputs and status
        """
        pass
    
    def validate_inputs(self, inputs: dict[str, Any]) -> list[str]:
        """
        Validate step inputs. Override for custom validation.
        
        Args:
            inputs: Input data to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        return []


@dataclass
class WorkflowContext:
    """
    Context for workflow execution.
    
    Stores workflow state, variables, and configuration.
    """
    workflow_name: str
    variables: dict[str, Any] = field(default_factory=dict)
    step_results: dict[str, StepResult] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def set_variable(self, name: str, value: Any) -> None:
        """Set a workflow variable."""
        self.variables[name] = value
    
    def get_variable(self, name: str, default: Any = None) -> Any:
        """Get a workflow variable."""
        return self.variables.get(name, default)
    
    def add_step_result(self, result: StepResult) -> None:
        """Record a step result."""
        self.step_results[result.step_name] = result


class SequentialStep(WorkflowStep):
    """
    A step that executes child steps in sequence.
    
    Each child step receives the outputs of the previous step
    as additional inputs.
    """
    
    def __init__(
        self,
        name: str,
        steps: list[WorkflowStep],
        description: str = "",
    ) -> None:
        """
        Initialize a sequential step container.
        
        Args:
            name: Step name
            steps: List of child steps to execute in order
            description: Optional description
        """
        super().__init__(name, description)
        self.steps = steps
    
    async def execute(
        self,
        inputs: dict[str, Any],
        context: WorkflowContext,
    ) -> StepResult:
        """Execute child steps sequentially."""
        import time
        start = time.time()
        
        with tracer.start_as_current_span(f"sequential_step_{self.name}"):
            current_inputs = inputs.copy()
            all_outputs: dict[str, Any] = {}
            
            for step in self.steps:
                self.status = StepStatus.RUNNING
                result = await step.execute(current_inputs, context)
                context.add_step_result(result)
                
                if not result.succeeded:
                    return StepResult(
                        step_name=self.name,
                        status=StepStatus.FAILED,
                        error=f"Child step '{step.name}' failed: {result.error}",
                        duration_ms=(time.time() - start) * 1000,
                    )
                
                # Pass outputs to next step
                current_inputs.update(result.outputs)
                all_outputs.update(result.outputs)
            
            self.status = StepStatus.COMPLETED
            return StepResult(
                step_name=self.name,
                status=StepStatus.COMPLETED,
                outputs=all_outputs,
                duration_ms=(time.time() - start) * 1000,
            )


class ParallelStep(WorkflowStep):
    """
    A step that executes child steps in parallel.
    
    All child steps receive the same inputs and their outputs
    are merged into a single result.
    """
    
    def __init__(
        self,
        name: str,
        steps: list[WorkflowStep],
        description: str = "",
        fail_fast: bool = True,
    ) -> None:
        """
        Initialize a parallel step container.
        
        Args:
            name: Step name
            steps: List of child steps to execute in parallel
            description: Optional description
            fail_fast: If True, fail immediately on first error
        """
        super().__init__(name, description)
        self.steps = steps
        self.fail_fast = fail_fast
    
    async def execute(
        self,
        inputs: dict[str, Any],
        context: WorkflowContext,
    ) -> StepResult:
        """Execute child steps in parallel."""
        import asyncio
        import time
        start = time.time()
        
        with tracer.start_as_current_span(f"parallel_step_{self.name}"):
            self.status = StepStatus.RUNNING
            
            # Execute all steps concurrently
            tasks = [
                step.execute(inputs.copy(), context)
                for step in self.steps
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            all_outputs: dict[str, Any] = {}
            errors: list[str] = []
            
            for step, result in zip(self.steps, results):
                if isinstance(result, Exception):
                    errors.append(f"{step.name}: {str(result)}")
                    context.add_step_result(StepResult(
                        step_name=step.name,
                        status=StepStatus.FAILED,
                        error=str(result),
                    ))
                elif not result.succeeded:
                    errors.append(f"{step.name}: {result.error}")
                    context.add_step_result(result)
                else:
                    all_outputs.update(result.outputs)
                    context.add_step_result(result)
            
            if errors and self.fail_fast:
                self.status = StepStatus.FAILED
                return StepResult(
                    step_name=self.name,
                    status=StepStatus.FAILED,
                    error="; ".join(errors),
                    duration_ms=(time.time() - start) * 1000,
                )
            
            self.status = StepStatus.COMPLETED
            return StepResult(
                step_name=self.name,
                status=StepStatus.COMPLETED,
                outputs=all_outputs,
                duration_ms=(time.time() - start) * 1000,
            )


class ConditionalStep(WorkflowStep):
    """
    A step that conditionally executes based on a condition.
    
    If the condition evaluates to True, executes the then_step.
    Otherwise, executes the else_step (if provided).
    """
    
    def __init__(
        self,
        name: str,
        condition: Callable[[dict[str, Any], WorkflowContext], bool],
        then_step: WorkflowStep,
        else_step: Optional[WorkflowStep] = None,
        description: str = "",
    ) -> None:
        """
        Initialize a conditional step.
        
        Args:
            name: Step name
            condition: Function that evaluates the condition
            then_step: Step to execute if condition is True
            else_step: Optional step to execute if condition is False
            description: Optional description
        """
        super().__init__(name, description)
        self.condition = condition
        self.then_step = then_step
        self.else_step = else_step
    
    async def execute(
        self,
        inputs: dict[str, Any],
        context: WorkflowContext,
    ) -> StepResult:
        """Execute the appropriate branch based on condition."""
        import time
        start = time.time()
        
        with tracer.start_as_current_span(f"conditional_step_{self.name}"):
            self.status = StepStatus.RUNNING
            
            try:
                condition_result = self.condition(inputs, context)
            except Exception as e:
                self.status = StepStatus.FAILED
                return StepResult(
                    step_name=self.name,
                    status=StepStatus.FAILED,
                    error=f"Condition evaluation failed: {str(e)}",
                    duration_ms=(time.time() - start) * 1000,
                )
            
            if condition_result:
                result = await self.then_step.execute(inputs, context)
            elif self.else_step:
                result = await self.else_step.execute(inputs, context)
            else:
                # No else branch, skip
                self.status = StepStatus.SKIPPED
                return StepResult(
                    step_name=self.name,
                    status=StepStatus.SKIPPED,
                    outputs=inputs,  # Pass through inputs
                    duration_ms=(time.time() - start) * 1000,
                )
            
            context.add_step_result(result)
            self.status = result.status
            
            return StepResult(
                step_name=self.name,
                status=result.status,
                outputs=result.outputs,
                error=result.error,
                duration_ms=(time.time() - start) * 1000,
            )


class DataTransform(WorkflowStep):
    """
    A step that transforms data between steps.
    
    Useful for mapping, filtering, or reshaping data
    in a workflow pipeline.
    """
    
    def __init__(
        self,
        name: str,
        transform: Callable[[dict[str, Any]], dict[str, Any]],
        description: str = "",
    ) -> None:
        """
        Initialize a data transform step.
        
        Args:
            name: Step name
            transform: Function that transforms input data
            description: Optional description
        """
        super().__init__(name, description)
        self.transform = transform
    
    async def execute(
        self,
        inputs: dict[str, Any],
        context: WorkflowContext,
    ) -> StepResult:
        """Apply the transformation to inputs."""
        import time
        start = time.time()
        
        with tracer.start_as_current_span(f"transform_step_{self.name}"):
            self.status = StepStatus.RUNNING
            
            try:
                outputs = self.transform(inputs)
                self.status = StepStatus.COMPLETED
                
                return StepResult(
                    step_name=self.name,
                    status=StepStatus.COMPLETED,
                    outputs=outputs,
                    duration_ms=(time.time() - start) * 1000,
                )
            except Exception as e:
                self.status = StepStatus.FAILED
                return StepResult(
                    step_name=self.name,
                    status=StepStatus.FAILED,
                    error=f"Transform failed: {str(e)}",
                    duration_ms=(time.time() - start) * 1000,
                )


class AgentStep(WorkflowStep):
    """
    A step that invokes an agent with a prompt template.
    
    This is the most common step type, executing an agent
    with variable substitution in the prompt.
    """
    
    def __init__(
        self,
        name: str,
        agent_name: str,
        prompt_template: str,
        output_vars: list[str],
        description: str = "",
        retry: Optional[RetryConfig] = None,
    ) -> None:
        """
        Initialize an agent step.
        
        Args:
            name: Step name
            agent_name: Name of the agent to invoke
            prompt_template: Prompt with {variable} placeholders
            output_vars: Names for output variables
            description: Optional description
            retry: Optional retry configuration
        """
        super().__init__(name, description, retry)
        self.agent_name = agent_name
        self.prompt_template = prompt_template
        self.output_vars = output_vars
    
    def _render_prompt(self, inputs: dict[str, Any]) -> str:
        """Render the prompt template with input values."""
        prompt = self.prompt_template
        for key, value in inputs.items():
            prompt = prompt.replace(f"{{{key}}}", str(value))
        return prompt
    
    async def execute(
        self,
        inputs: dict[str, Any],
        context: WorkflowContext,
    ) -> StepResult:
        """Execute the agent with the rendered prompt."""
        import time
        start = time.time()
        
        with tracer.start_as_current_span(f"agent_step_{self.name}") as span:
            span.set_attribute("agent_name", self.agent_name)
            self.status = StepStatus.RUNNING
            
            # Render the prompt
            prompt = self._render_prompt(inputs)
            span.set_attribute("prompt_length", len(prompt))
            
            # Get agent from context
            agents = context.metadata.get("agents", {})
            agent = agents.get(self.agent_name)
            
            if agent is None:
                self.status = StepStatus.FAILED
                return StepResult(
                    step_name=self.name,
                    status=StepStatus.FAILED,
                    error=f"Agent '{self.agent_name}' not found",
                    duration_ms=(time.time() - start) * 1000,
                )
            
            try:
                # Execute agent
                result = await agent.run(prompt)
                
                # Map result to output variables
                outputs = {}
                if len(self.output_vars) == 1:
                    outputs[self.output_vars[0]] = result
                else:
                    # For multiple outputs, expect dict from agent
                    if isinstance(result, dict):
                        for var in self.output_vars:
                            outputs[var] = result.get(var, "")
                    else:
                        outputs[self.output_vars[0]] = result
                
                self.status = StepStatus.COMPLETED
                return StepResult(
                    step_name=self.name,
                    status=StepStatus.COMPLETED,
                    outputs=outputs,
                    duration_ms=(time.time() - start) * 1000,
                )
                
            except Exception as e:
                self.status = StepStatus.FAILED
                return StepResult(
                    step_name=self.name,
                    status=StepStatus.FAILED,
                    error=str(e),
                    duration_ms=(time.time() - start) * 1000,
                )
