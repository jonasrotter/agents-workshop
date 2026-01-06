"""
Enhanced Workflow Builders for Agent Orchestration.

This module provides builder patterns for creating multi-agent workflows
following Microsoft Agent Framework Workflows patterns. Builders offer
a fluent API for composing complex agent orchestration scenarios.

Key patterns supported:
- SequentialBuilder: Execute agents one after another
- ConcurrentBuilder: Execute agents in parallel
- GroupChatBuilder: Multi-agent discussion/collaboration
- PipelineBuilder: Data transformation pipelines

Usage:
    from src.workflows.builders import (
        SequentialBuilder,
        ConcurrentBuilder,
        GroupChatBuilder,
    )
    
    # Sequential workflow
    workflow = (
        SequentialBuilder("research_pipeline")
        .add_agent("researcher", research_agent, "Research: {topic}")
        .add_agent("summarizer", summarizer_agent, "Summarize: {research}")
        .build()
    )
    
    # Concurrent workflow
    workflow = (
        ConcurrentBuilder("parallel_research")
        .add_agent("web_search", web_agent, "Search: {query}")
        .add_agent("doc_search", doc_agent, "Search docs: {query}")
        .with_aggregator(lambda results: combine_results(results))
        .build()
    )
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TypeVar, Generic

from src.common.telemetry import get_tracer
from src.workflows.steps import (
    WorkflowStep,
    WorkflowContext,
    StepResult,
    StepStatus,
    RetryConfig,
    AgentStep,
    SequentialStep,
    ParallelStep,
    DataTransform,
)
from src.workflows.engine import (
    WorkflowEngine,
    WorkflowResult,
    WorkflowStatus,
    ErrorStrategy,
    ErrorConfig,
)

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


@dataclass
class AgentConfig:
    """Configuration for an agent in a workflow."""
    name: str
    agent: Any
    prompt_template: str
    output_vars: list[str] = field(default_factory=lambda: ["result"])
    description: str = ""
    retry: Optional[RetryConfig] = None


class BaseBuilder:
    """Base class for workflow builders."""
    
    def __init__(self, name: str) -> None:
        """
        Initialize the builder.
        
        Args:
            name: Workflow identifier
        """
        self._name = name
        self._description = ""
        self._error_strategy = ErrorStrategy.ABORT
        self._error_config = ErrorConfig()
        self._hooks_before: list[Callable] = []
        self._hooks_after: list[Callable] = []
    
    def with_description(self, description: str) -> "BaseBuilder":
        """Set workflow description."""
        self._description = description
        return self
    
    def on_error(
        self,
        strategy: ErrorStrategy,
        fallback_value: Optional[Any] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> "BaseBuilder":
        """
        Configure error handling strategy.
        
        Args:
            strategy: Error handling strategy (ABORT, SKIP, RETRY, FALLBACK)
            fallback_value: Value to use on FALLBACK strategy
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries in seconds
            
        Returns:
            Self for method chaining
        """
        self._error_config = ErrorConfig(
            strategy=strategy,
            fallback_value=fallback_value,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        return self
    
    def before_step(self, hook: Callable[[WorkflowStep, dict[str, Any]], None]) -> "BaseBuilder":
        """Add a hook to run before each step."""
        self._hooks_before.append(hook)
        return self
    
    def after_step(self, hook: Callable[[WorkflowStep, StepResult], None]) -> "BaseBuilder":
        """Add a hook to run after each step."""
        self._hooks_after.append(hook)
        return self


class SequentialBuilder(BaseBuilder):
    """
    Builder for sequential agent workflows.
    
    Creates workflows where agents execute one after another,
    with each agent receiving the outputs of previous agents.
    
    Example:
        workflow = (
            SequentialBuilder("research_pipeline")
            .add_agent("researcher", research_agent, "Research: {topic}", ["findings"])
            .add_agent("summarizer", summarize_agent, "Summarize: {findings}", ["summary"])
            .on_error(ErrorStrategy.RETRY, max_retries=3)
            .build()
        )
        
        result = await workflow.execute({"topic": "AI agents"})
    """
    
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._agents: list[AgentConfig] = []
        self._transforms: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {}
    
    def add_agent(
        self,
        name: str,
        agent: Any,
        prompt_template: str,
        output_vars: Optional[list[str]] = None,
        description: str = "",
        retry: Optional[RetryConfig] = None,
    ) -> "SequentialBuilder":
        """
        Add an agent to the sequential workflow.
        
        Args:
            name: Step name for this agent
            agent: Agent instance
            prompt_template: Prompt template with {variable} placeholders
            output_vars: Names for output variables (default: ["result"])
            description: Optional step description
            retry: Optional retry configuration
            
        Returns:
            Self for method chaining
        """
        config = AgentConfig(
            name=name,
            agent=agent,
            prompt_template=prompt_template,
            output_vars=output_vars or ["result"],
            description=description,
            retry=retry,
        )
        self._agents.append(config)
        return self
    
    def add_transform(
        self,
        name: str,
        transform: Callable[[dict[str, Any]], dict[str, Any]],
        description: str = "",
    ) -> "SequentialBuilder":
        """
        Add a data transformation between steps.
        
        Args:
            name: Transform name
            transform: Function to transform data
            description: Optional description
            
        Returns:
            Self for method chaining
        """
        # Store transform to be inserted at current position
        self._transforms[len(self._agents)] = (name, transform, description)
        return self
    
    def build(self) -> WorkflowEngine:
        """
        Build and return the workflow engine.
        
        Returns:
            Configured WorkflowEngine for sequential execution
        """
        engine = WorkflowEngine(
            name=self._name,
            description=self._description,
            error_config=self._error_config,
        )
        
        # Build steps in order, inserting transforms where specified
        step_index = 0
        for i, config in enumerate(self._agents):
            # Check for transform before this agent
            if i in self._transforms:
                t_name, t_func, t_desc = self._transforms[i]
                engine.add_step(DataTransform(
                    name=t_name,
                    transform=t_func,
                    description=t_desc,
                ))
            
            # Add agent step
            engine.register_agent(config.name, config.agent)
            engine.add_step(AgentStep(
                name=config.name,
                agent_name=config.name,
                prompt_template=config.prompt_template,
                output_vars=config.output_vars,
                description=config.description,
                retry=config.retry,
            ))
        
        # Register hooks
        for hook in self._hooks_before:
            engine.before_step(hook)
        for hook in self._hooks_after:
            engine.after_step(hook)
        
        return engine


class ConcurrentBuilder(BaseBuilder):
    """
    Builder for concurrent agent workflows.
    
    Creates workflows where multiple agents execute in parallel,
    with results aggregated at the end.
    
    Example:
        workflow = (
            ConcurrentBuilder("parallel_search")
            .add_agent("web", web_agent, "Web search: {query}", ["web_results"])
            .add_agent("docs", doc_agent, "Doc search: {query}", ["doc_results"])
            .with_aggregator(combine_search_results)
            .build()
        )
        
        result = await workflow.execute({"query": "AI agents"})
    """
    
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._agents: list[AgentConfig] = []
        self._aggregator: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None
        self._fail_fast = True
    
    def add_agent(
        self,
        name: str,
        agent: Any,
        prompt_template: str,
        output_vars: Optional[list[str]] = None,
        description: str = "",
        retry: Optional[RetryConfig] = None,
    ) -> "ConcurrentBuilder":
        """
        Add an agent to execute concurrently.
        
        Args:
            name: Step name for this agent
            agent: Agent instance
            prompt_template: Prompt template with {variable} placeholders
            output_vars: Names for output variables (default: ["result"])
            description: Optional step description
            retry: Optional retry configuration
            
        Returns:
            Self for method chaining
        """
        config = AgentConfig(
            name=name,
            agent=agent,
            prompt_template=prompt_template,
            output_vars=output_vars or ["result"],
            description=description,
            retry=retry,
        )
        self._agents.append(config)
        return self
    
    def with_aggregator(
        self,
        aggregator: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> "ConcurrentBuilder":
        """
        Set a function to aggregate parallel results.
        
        Args:
            aggregator: Function that takes merged outputs and returns final result
            
        Returns:
            Self for method chaining
        """
        self._aggregator = aggregator
        return self
    
    def fail_fast(self, enabled: bool = True) -> "ConcurrentBuilder":
        """
        Configure fail-fast behavior.
        
        Args:
            enabled: If True, fail immediately on any error
            
        Returns:
            Self for method chaining
        """
        self._fail_fast = enabled
        return self
    
    def build(self) -> WorkflowEngine:
        """
        Build and return the workflow engine.
        
        Returns:
            Configured WorkflowEngine for concurrent execution
        """
        engine = WorkflowEngine(
            name=self._name,
            description=self._description,
            error_config=self._error_config,
        )
        
        # Create parallel step containing all agent steps
        parallel_steps = []
        for config in self._agents:
            engine.register_agent(config.name, config.agent)
            parallel_steps.append(AgentStep(
                name=config.name,
                agent_name=config.name,
                prompt_template=config.prompt_template,
                output_vars=config.output_vars,
                description=config.description,
                retry=config.retry,
            ))
        
        # Add parallel step
        engine.add_step(ParallelStep(
            name=f"{self._name}_parallel",
            steps=parallel_steps,
            fail_fast=self._fail_fast,
        ))
        
        # Add aggregator if specified
        if self._aggregator:
            engine.add_step(DataTransform(
                name=f"{self._name}_aggregate",
                transform=self._aggregator,
            ))
        
        # Register hooks
        for hook in self._hooks_before:
            engine.before_step(hook)
        for hook in self._hooks_after:
            engine.after_step(hook)
        
        return engine


class GroupChatBuilder(BaseBuilder):
    """
    Builder for multi-agent group discussion workflows.
    
    Creates workflows where agents participate in a collaborative
    discussion, taking turns to contribute to the conversation.
    
    Example:
        workflow = (
            GroupChatBuilder("design_discussion")
            .add_participant("architect", architect_agent)
            .add_participant("developer", dev_agent)
            .add_participant("reviewer", review_agent)
            .with_moderator(moderator_agent)
            .max_rounds(5)
            .build()
        )
        
        result = await workflow.execute({"topic": "API design"})
    """
    
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._participants: list[tuple[str, Any]] = []
        self._moderator: Optional[Any] = None
        self._max_rounds = 3
        self._termination_condition: Optional[Callable[[str, int], bool]] = None
    
    def add_participant(
        self,
        name: str,
        agent: Any,
        description: str = "",
    ) -> "GroupChatBuilder":
        """
        Add a participant to the group discussion.
        
        Args:
            name: Participant identifier
            agent: Agent instance
            description: Role description
            
        Returns:
            Self for method chaining
        """
        self._participants.append((name, agent, description))
        return self
    
    def with_moderator(self, moderator: Any) -> "GroupChatBuilder":
        """
        Set the discussion moderator.
        
        The moderator coordinates the discussion and decides
        when to conclude or which agent speaks next.
        
        Args:
            moderator: Moderator agent instance
            
        Returns:
            Self for method chaining
        """
        self._moderator = moderator
        return self
    
    def max_rounds(self, rounds: int) -> "GroupChatBuilder":
        """
        Set maximum discussion rounds.
        
        Args:
            rounds: Maximum number of turns
            
        Returns:
            Self for method chaining
        """
        self._max_rounds = rounds
        return self
    
    def with_termination(
        self,
        condition: Callable[[str, int], bool],
    ) -> "GroupChatBuilder":
        """
        Set custom termination condition.
        
        Args:
            condition: Function(last_message, round_num) -> bool
            
        Returns:
            Self for method chaining
        """
        self._termination_condition = condition
        return self
    
    def build(self) -> WorkflowEngine:
        """
        Build and return the workflow engine.
        
        Creates a workflow that simulates group discussion
        by running agents in sequence for multiple rounds.
        
        Returns:
            Configured WorkflowEngine for group discussion
        """
        engine = WorkflowEngine(
            name=self._name,
            description=self._description,
            error_config=self._error_config,
        )
        
        # Register moderator if present
        if self._moderator:
            engine.register_agent("moderator", self._moderator)
        
        # Register all participants
        for name, agent, _ in self._participants:
            engine.register_agent(name, agent)
        
        # Create discussion steps
        # For each round, each participant contributes
        for round_num in range(self._max_rounds):
            # Moderator introduction/direction for the round
            if self._moderator:
                engine.add_step(AgentStep(
                    name=f"moderator_round_{round_num}",
                    agent_name="moderator",
                    prompt_template=(
                        "Round {round}: Guide the discussion on {topic}. "
                        "Previous discussion: {discussion_history}"
                    ),
                    output_vars=[f"moderator_r{round_num}"],
                    description=f"Moderator guidance for round {round_num}",
                ))
            
            # Each participant takes a turn
            for name, agent, desc in self._participants:
                engine.add_step(AgentStep(
                    name=f"{name}_round_{round_num}",
                    agent_name=name,
                    prompt_template=(
                        "Contribute to the discussion on {topic}. "
                        "Previous contributions: {discussion_history}"
                    ),
                    output_vars=[f"{name}_r{round_num}"],
                    description=f"{name}'s contribution in round {round_num}",
                ))
        
        # Final summary step
        if self._moderator:
            engine.add_step(AgentStep(
                name="final_summary",
                agent_name="moderator",
                prompt_template=(
                    "Summarize the discussion outcomes on {topic}. "
                    "Full discussion: {discussion_history}"
                ),
                output_vars=["summary"],
                description="Final discussion summary",
            ))
        
        # Register hooks
        for hook in self._hooks_before:
            engine.before_step(hook)
        for hook in self._hooks_after:
            engine.after_step(hook)
        
        return engine


class PipelineBuilder(BaseBuilder):
    """
    Builder for data transformation pipelines.
    
    Creates workflows focused on transforming data through
    a series of steps, with optional agent processing.
    
    Example:
        pipeline = (
            PipelineBuilder("data_pipeline")
            .add_transform("parse", parse_input)
            .add_agent("enrich", enrichment_agent, "Enrich: {data}")
            .add_transform("format", format_output)
            .build()
        )
    """
    
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._steps: list[tuple[str, Any, str]] = []  # (type, step, description)
        self._agents: dict[str, Any] = {}
    
    def add_transform(
        self,
        name: str,
        transform: Callable[[dict[str, Any]], dict[str, Any]],
        description: str = "",
    ) -> "PipelineBuilder":
        """
        Add a data transformation step.
        
        Args:
            name: Step name
            transform: Transformation function
            description: Optional description
            
        Returns:
            Self for method chaining
        """
        self._steps.append(("transform", DataTransform(
            name=name,
            transform=transform,
            description=description,
        ), description))
        return self
    
    def add_agent(
        self,
        name: str,
        agent: Any,
        prompt_template: str,
        output_vars: Optional[list[str]] = None,
        description: str = "",
        retry: Optional[RetryConfig] = None,
    ) -> "PipelineBuilder":
        """
        Add an agent processing step.
        
        Args:
            name: Step name
            agent: Agent instance
            prompt_template: Prompt with {variable} placeholders
            output_vars: Output variable names
            description: Optional description
            retry: Optional retry configuration
            
        Returns:
            Self for method chaining
        """
        self._agents[name] = agent
        self._steps.append(("agent", AgentStep(
            name=name,
            agent_name=name,
            prompt_template=prompt_template,
            output_vars=output_vars or ["result"],
            description=description,
            retry=retry,
        ), description))
        return self
    
    def add_filter(
        self,
        name: str,
        predicate: Callable[[dict[str, Any]], bool],
        description: str = "",
    ) -> "PipelineBuilder":
        """
        Add a filter step that passes or blocks data.
        
        Args:
            name: Step name
            predicate: Function returning True to pass, False to block
            description: Optional description
            
        Returns:
            Self for method chaining
        """
        def filter_transform(inputs: dict[str, Any]) -> dict[str, Any]:
            if predicate(inputs):
                return inputs
            return {"__filtered": True}
        
        self._steps.append(("transform", DataTransform(
            name=name,
            transform=filter_transform,
            description=description or f"Filter: {name}",
        ), description))
        return self
    
    def build(self) -> WorkflowEngine:
        """
        Build and return the workflow engine.
        
        Returns:
            Configured WorkflowEngine for the pipeline
        """
        engine = WorkflowEngine(
            name=self._name,
            description=self._description,
            error_config=self._error_config,
        )
        
        # Register agents
        for name, agent in self._agents.items():
            engine.register_agent(name, agent)
        
        # Add all steps
        for _, step, _ in self._steps:
            engine.add_step(step)
        
        # Register hooks
        for hook in self._hooks_before:
            engine.before_step(hook)
        for hook in self._hooks_after:
            engine.after_step(hook)
        
        return engine


# Convenience factory functions for creating builders
def sequential(name: str) -> SequentialBuilder:
    """Create a sequential workflow builder."""
    return SequentialBuilder(name)


def concurrent(name: str) -> ConcurrentBuilder:
    """Create a concurrent workflow builder."""
    return ConcurrentBuilder(name)


def group_chat(name: str) -> GroupChatBuilder:
    """Create a group chat workflow builder."""
    return GroupChatBuilder(name)


def pipeline(name: str) -> PipelineBuilder:
    """Create a pipeline workflow builder."""
    return PipelineBuilder(name)
