"""
Workflow Engine Module.

This module provides deterministic multi-agent workflow orchestration capabilities.
It includes workflow step definitions, execution engine, and coordination utilities.

Exports:
    - WorkflowEngine: Main engine for executing workflows
    - WorkflowStep: Step definition for workflow nodes
    - SequentialStep: Step that executes sequentially
    - ParallelStep: Step that executes in parallel
    - ConditionalStep: Step with conditional branching
    - ErrorStrategy: Error handling strategies (retry, fallback, abort)
    - WorkflowResult: Result of workflow execution
    - create_workflow: Factory for creating workflows from definitions
    
Enhanced Builders:
    - SequentialBuilder: Fluent builder for sequential workflows
    - ConcurrentBuilder: Fluent builder for parallel workflows
    - GroupChatBuilder: Fluent builder for multi-agent discussions
    - PipelineBuilder: Fluent builder for data transformation pipelines
    - sequential: Factory function for SequentialBuilder
    - concurrent: Factory function for ConcurrentBuilder
    - group_chat: Factory function for GroupChatBuilder
    - pipeline: Factory function for PipelineBuilder
"""

from src.workflows.steps import (
    WorkflowStep,
    SequentialStep,
    ParallelStep,
    ConditionalStep,
    DataTransform,
)
from src.workflows.engine import (
    WorkflowEngine,
    WorkflowResult,
    WorkflowContext,
    ErrorStrategy,
    create_workflow,
    WorkflowBuilder,
)
from src.workflows.builders import (
    SequentialBuilder,
    ConcurrentBuilder,
    GroupChatBuilder,
    PipelineBuilder,
    sequential,
    concurrent,
    group_chat,
    pipeline,
)

__all__ = [
    # Steps
    "WorkflowStep",
    "SequentialStep",
    "ParallelStep",
    "ConditionalStep",
    "DataTransform",
    # Engine
    "WorkflowEngine",
    "WorkflowResult",
    "WorkflowContext",
    "ErrorStrategy",
    "create_workflow",
    "WorkflowBuilder",
    # Enhanced Builders
    "SequentialBuilder",
    "ConcurrentBuilder",
    "GroupChatBuilder",
    "PipelineBuilder",
    "sequential",
    "concurrent",
    "group_chat",
    "pipeline",
]
