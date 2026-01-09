"""
Integration tests for Scenario 04: Deterministic Multi-Agent Workflows.

Tests the complete workflow functionality including:
- Module imports
- Workflow construction
- Step execution flows
- Error handling strategies
- End-to-end workflow execution

Note: The src.workflows module is deprecated in favor of Microsoft Agent Framework's
WorkflowBuilder. These tests ensure backward compatibility during migration.
See notebooks/04_deterministic_workflows.ipynb for the new recommended patterns.
"""

import pytest
import warnings
from unittest.mock import AsyncMock, MagicMock
from typing import Any


# Configure pytest to filter deprecation warnings for legacy workflow imports
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning:src.workflows")


class TestScenario04Imports:
    """Test module imports for Scenario 04."""

    def test_import_workflow_steps(self) -> None:
        """Can import workflow step classes."""
        from src.workflows.steps import (
            WorkflowStep,
            StepResult,
            StepStatus,
            WorkflowContext,
            RetryConfig,
            SequentialStep,
            ParallelStep,
            ConditionalStep,
            DataTransform,
            AgentStep,
        )
        
        assert WorkflowStep is not None
        assert StepResult is not None
        assert StepStatus is not None
        assert WorkflowContext is not None
        assert RetryConfig is not None
        assert SequentialStep is not None
        assert ParallelStep is not None
        assert ConditionalStep is not None
        assert DataTransform is not None
        assert AgentStep is not None

    def test_import_workflow_engine(self) -> None:
        """Can import workflow engine classes."""
        from src.workflows.engine import (
            WorkflowEngine,
            WorkflowResult,
            WorkflowStatus,
            ErrorStrategy,
            ErrorConfig,
            create_workflow,
            WorkflowBuilder,
        )
        
        assert WorkflowEngine is not None
        assert WorkflowResult is not None
        assert WorkflowStatus is not None
        assert ErrorStrategy is not None
        assert ErrorConfig is not None
        assert create_workflow is not None
        assert WorkflowBuilder is not None

    def test_import_from_module_init(self) -> None:
        """Can import from module __init__."""
        from src.workflows import (
            WorkflowStep,
            SequentialStep,
            ParallelStep,
            ConditionalStep,
            DataTransform,
            WorkflowEngine,
            WorkflowResult,
            WorkflowContext,
            ErrorStrategy,
            create_workflow,
        )
        
        assert WorkflowStep is not None
        assert SequentialStep is not None
        assert ParallelStep is not None
        assert ConditionalStep is not None
        assert DataTransform is not None
        assert WorkflowEngine is not None
        assert WorkflowResult is not None
        assert WorkflowContext is not None
        assert ErrorStrategy is not None
        assert create_workflow is not None


class TestWorkflowEngineCreation:
    """Test workflow engine instantiation."""

    def test_create_basic_engine(self) -> None:
        """Create a basic workflow engine."""
        from src.workflows import WorkflowEngine
        
        engine = WorkflowEngine(name="test_workflow")
        
        assert engine.name == "test_workflow"
        assert len(engine.steps) == 0
        assert len(engine.agents) == 0

    def test_create_engine_with_description(self) -> None:
        """Create engine with description."""
        from src.workflows import WorkflowEngine
        
        engine = WorkflowEngine(
            name="test_workflow",
            description="A test workflow",
        )
        
        assert engine.description == "A test workflow"

    def test_create_workflow_factory(self) -> None:
        """Use create_workflow factory."""
        from src.workflows import create_workflow
        from src.workflows.steps import WorkflowStep, StepResult, StepStatus, WorkflowContext
        
        class DummyStep(WorkflowStep):
            async def execute(self, inputs, context):
                return StepResult(
                    step_name=self.name,
                    status=StepStatus.COMPLETED,
                    outputs={},
                )
        
        workflow = create_workflow(
            name="factory_test",
            steps=[DummyStep("step1")],
            description="Created by factory",
        )
        
        assert workflow.name == "factory_test"
        assert len(workflow.steps) == 1


class TestWorkflowContext:
    """Test workflow context management."""

    def test_create_context(self) -> None:
        """Create workflow context."""
        from src.workflows import WorkflowContext
        
        context = WorkflowContext(workflow_name="test")
        
        assert context.workflow_name == "test"
        assert context.variables == {}
        assert context.step_results == {}

    def test_context_variable_management(self) -> None:
        """Context manages variables."""
        from src.workflows import WorkflowContext
        
        context = WorkflowContext(
            workflow_name="test",
            variables={"initial": "value"},
        )
        
        context.set_variable("new_key", "new_value")
        
        assert context.get_variable("initial") == "value"
        assert context.get_variable("new_key") == "new_value"

    def test_context_metadata(self) -> None:
        """Context stores metadata."""
        from src.workflows import WorkflowContext
        
        context = WorkflowContext(
            workflow_name="test",
            metadata={"agent_count": 3},
        )
        
        assert context.metadata["agent_count"] == 3


class TestStepExecution:
    """Test individual step execution."""

    @pytest.mark.asyncio
    async def test_data_transform_step(self) -> None:
        """DataTransform step transforms data."""
        from src.workflows import DataTransform, WorkflowContext
        from src.workflows.steps import StepStatus
        
        transform = DataTransform(
            name="uppercase",
            transform=lambda inputs: {
                "text": inputs.get("text", "").upper()
            },
        )
        
        context = WorkflowContext(workflow_name="test")
        result = await transform.execute({"text": "hello"}, context)
        
        assert result.status == StepStatus.COMPLETED
        assert result.outputs["text"] == "HELLO"

    @pytest.mark.asyncio
    async def test_conditional_with_context(self) -> None:
        """ConditionalStep uses context in condition."""
        from src.workflows import ConditionalStep, DataTransform, WorkflowContext
        from src.workflows.steps import StepStatus
        
        def check_threshold(inputs, context):
            return inputs.get("value", 0) > 10
        
        conditional = ConditionalStep(
            name="threshold_check",
            condition=check_threshold,
            then_step=DataTransform(
                "high",
                transform=lambda x: {"category": "high"},
            ),
            else_step=DataTransform(
                "low",
                transform=lambda x: {"category": "low"},
            ),
        )
        
        context = WorkflowContext(workflow_name="test")
        
        # Test high value
        result_high = await conditional.execute({"value": 15}, context)
        assert result_high.outputs["category"] == "high"
        
        # Test low value
        result_low = await conditional.execute({"value": 5}, context)
        assert result_low.outputs["category"] == "low"


class TestSequentialWorkflows:
    """Test sequential workflow execution."""

    @pytest.mark.asyncio
    async def test_three_step_sequence(self) -> None:
        """Execute three steps in sequence."""
        from src.workflows import SequentialStep, DataTransform, WorkflowContext
        from src.workflows.steps import StepStatus
        
        seq = SequentialStep(
            name="pipeline",
            steps=[
                DataTransform(
                    "add_prefix",
                    transform=lambda x: {"text": "PREFIX_" + x.get("text", "")},
                ),
                DataTransform(
                    "uppercase",
                    transform=lambda x: {"text": x.get("text", "").upper()},
                ),
                DataTransform(
                    "add_suffix",
                    transform=lambda x: {"text": x.get("text", "") + "_SUFFIX"},
                ),
            ],
        )
        
        context = WorkflowContext(workflow_name="test")
        result = await seq.execute({"text": "hello"}, context)
        
        assert result.status == StepStatus.COMPLETED
        assert result.outputs["text"] == "PREFIX_HELLO_SUFFIX"

    @pytest.mark.asyncio
    async def test_sequence_data_flow(self) -> None:
        """Data flows through sequential steps."""
        from src.workflows import WorkflowEngine, DataTransform, WorkflowContext
        from src.workflows.steps import StepStatus
        
        engine = WorkflowEngine(name="data_flow_test")
        engine.add_step(
            DataTransform("step1", lambda x: {"a": 1, **x})
        ).add_step(
            DataTransform("step2", lambda x: {"b": 2, **x})
        ).add_step(
            DataTransform("step3", lambda x: {"c": 3, **x})
        )
        
        result = await engine.execute({"initial": 0})
        
        assert result.outputs["a"] == 1
        assert result.outputs["b"] == 2
        assert result.outputs["c"] == 3
        assert result.outputs["initial"] == 0


class TestParallelWorkflows:
    """Test parallel workflow execution."""

    @pytest.mark.asyncio
    async def test_parallel_merge(self) -> None:
        """Parallel steps merge outputs."""
        from src.workflows import ParallelStep, DataTransform, WorkflowContext
        from src.workflows.steps import StepStatus
        
        parallel = ParallelStep(
            name="fan_out",
            steps=[
                DataTransform("branch_a", lambda x: {"result_a": "A"}),
                DataTransform("branch_b", lambda x: {"result_b": "B"}),
                DataTransform("branch_c", lambda x: {"result_c": "C"}),
            ],
        )
        
        context = WorkflowContext(workflow_name="test")
        result = await parallel.execute({}, context)
        
        assert result.status == StepStatus.COMPLETED
        assert result.outputs["result_a"] == "A"
        assert result.outputs["result_b"] == "B"
        assert result.outputs["result_c"] == "C"


class TestErrorHandling:
    """Test workflow error handling strategies."""

    @pytest.mark.asyncio
    async def test_abort_strategy(self) -> None:
        """ABORT strategy stops on first error."""
        from src.workflows import WorkflowEngine, DataTransform
        from src.workflows.engine import ErrorStrategy, ErrorConfig, WorkflowStatus
        
        engine = WorkflowEngine(
            name="abort_test",
            error_config=ErrorConfig(strategy=ErrorStrategy.ABORT),
        )
        
        def failing_transform(inputs):
            raise ValueError("Intentional error")
        
        engine.add_step(DataTransform("step1", lambda x: {"a": 1}))
        engine.add_step(DataTransform("step2", failing_transform))
        engine.add_step(DataTransform("step3", lambda x: {"c": 3}))
        
        result = await engine.execute({})
        
        assert result.status == WorkflowStatus.FAILED
        assert "step2" in result.error

    @pytest.mark.asyncio
    async def test_skip_strategy(self) -> None:
        """SKIP strategy continues on error."""
        from src.workflows import WorkflowEngine, DataTransform
        from src.workflows.engine import ErrorStrategy, ErrorConfig, WorkflowStatus
        
        engine = WorkflowEngine(
            name="skip_test",
            error_config=ErrorConfig(strategy=ErrorStrategy.SKIP),
        )
        
        def failing_transform(inputs):
            raise ValueError("Intentional error")
        
        engine.add_step(DataTransform("step1", lambda x: {"a": 1}))
        engine.add_step(DataTransform("step2", failing_transform))
        engine.add_step(DataTransform("step3", lambda x: {"c": 3}))
        
        result = await engine.execute({})
        
        assert result.status == WorkflowStatus.COMPLETED
        assert result.outputs.get("a") == 1
        assert result.outputs.get("c") == 3


class TestWorkflowBuilder:
    """Test fluent WorkflowBuilder API."""

    def test_builder_chain(self) -> None:
        """Builder methods chain correctly."""
        from src.workflows.engine import WorkflowBuilder, ErrorStrategy
        from src.workflows import DataTransform
        
        workflow = (
            WorkflowBuilder("builder_test")
            .with_description("Test workflow")
            .add_step(DataTransform("step1", lambda x: x))
            .add_step(DataTransform("step2", lambda x: x))
            .on_error(ErrorStrategy.SKIP)
            .build()
        )
        
        assert workflow.name == "builder_test"
        assert workflow.description == "Test workflow"
        assert len(workflow.steps) == 2
        assert workflow.error_config.strategy == ErrorStrategy.SKIP

    @pytest.mark.asyncio
    async def test_builder_with_agents(self) -> None:
        """Builder registers agents correctly."""
        from src.workflows.engine import WorkflowBuilder
        from src.workflows import DataTransform
        
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value="Agent response")
        
        workflow = (
            WorkflowBuilder("agent_test")
            .with_agent("test_agent", mock_agent)
            .add_step(DataTransform("step1", lambda x: {"result": "done"}))
            .build()
        )
        
        assert "test_agent" in workflow.agents
        assert workflow.agents["test_agent"] is mock_agent


class TestWorkflowValidation:
    """Test workflow validation."""

    def test_validate_empty_workflow(self) -> None:
        """Validation catches empty workflow."""
        from src.workflows import WorkflowEngine
        
        engine = WorkflowEngine(name="empty")
        errors = engine.validate()
        
        assert len(errors) > 0
        assert any("no steps" in e for e in errors)

    def test_validate_missing_agent(self) -> None:
        """Validation catches missing agent."""
        from src.workflows import WorkflowEngine
        from src.workflows.steps import AgentStep
        
        engine = WorkflowEngine(name="test")
        engine.add_step(AgentStep(
            name="agent_step",
            agent_name="nonexistent",
            prompt_template="Hello",
            output_vars=["result"],
        ))
        
        errors = engine.validate()
        
        assert any("nonexistent" in e for e in errors)

    def test_validation_passes(self) -> None:
        """Valid workflow passes validation."""
        from src.workflows import WorkflowEngine, DataTransform
        
        engine = WorkflowEngine(name="valid")
        engine.add_step(DataTransform("step1", lambda x: x))
        
        errors = engine.validate()
        
        assert len(errors) == 0


class TestEndToEndWorkflow:
    """End-to-end workflow tests."""

    @pytest.mark.asyncio
    async def test_document_processing_workflow(self) -> None:
        """Complete document processing workflow."""
        from src.workflows import (
            WorkflowEngine,
            SequentialStep,
            ParallelStep,
            ConditionalStep,
            DataTransform,
        )
        from src.workflows.engine import WorkflowStatus
        
        # Build a realistic workflow
        engine = WorkflowEngine(
            name="document_processor",
            description="Process documents through multiple stages",
        )
        
        # Stage 1: Extract metadata
        engine.add_step(DataTransform(
            "extract_metadata",
            transform=lambda x: {
                **x,
                "word_count": len(x.get("content", "").split()),
                "char_count": len(x.get("content", "")),
            },
        ))
        
        # Stage 2: Parallel analysis
        engine.add_step(ParallelStep(
            name="parallel_analysis",
            steps=[
                DataTransform(
                    "sentiment",
                    transform=lambda x: {**x, "sentiment": "positive"},
                ),
                DataTransform(
                    "topics",
                    transform=lambda x: {**x, "topics": ["ai", "workflows"]},
                ),
            ],
        ))
        
        # Stage 3: Conditional categorization
        engine.add_step(ConditionalStep(
            name="categorize",
            condition=lambda inputs, ctx: inputs.get("word_count", 0) > 5,
            then_step=DataTransform(
                "long_doc",
                transform=lambda x: {**x, "category": "detailed"},
            ),
            else_step=DataTransform(
                "short_doc",
                transform=lambda x: {**x, "category": "brief"},
            ),
        ))
        
        # Execute workflow
        result = await engine.execute({
            "content": "This is a test document for our workflow system.",
        })
        
        assert result.status == WorkflowStatus.COMPLETED
        assert result.outputs["word_count"] == 9
        assert result.outputs["sentiment"] == "positive"
        assert result.outputs["topics"] == ["ai", "workflows"]
        assert result.outputs["category"] == "detailed"

    @pytest.mark.asyncio
    async def test_workflow_with_timeout(self) -> None:
        """Workflow respects timeout."""
        from src.workflows import WorkflowEngine, DataTransform
        from src.workflows.engine import WorkflowStatus
        import asyncio
        
        async def slow_transform(inputs):
            await asyncio.sleep(10)
            return inputs
        
        engine = WorkflowEngine(name="slow_workflow")
        
        # Add a slow step using a custom step
        from src.workflows.steps import WorkflowStep, StepResult, StepStatus
        
        class SlowStep(WorkflowStep):
            async def execute(self, inputs, context):
                await asyncio.sleep(10)
                return StepResult(
                    step_name=self.name,
                    status=StepStatus.COMPLETED,
                    outputs={},
                )
        
        engine.add_step(SlowStep("slow"))
        
        result = await engine.execute({}, timeout=0.1)
        
        assert result.status == WorkflowStatus.FAILED
        assert "timed out" in result.error


class TestWorkflowHooks:
    """Test workflow execution hooks."""

    @pytest.mark.asyncio
    async def test_before_step_hook(self) -> None:
        """Before step hook is called."""
        from src.workflows import WorkflowEngine, DataTransform
        from src.workflows.steps import StepResult, StepStatus
        
        hook_calls = []
        
        def before_hook(step, inputs) -> None:
            hook_calls.append(("before", step.name))
        
        engine = WorkflowEngine(name="hook_test")
        engine.before_step(before_hook)
        engine.add_step(DataTransform("step1", lambda x: x))
        engine.add_step(DataTransform("step2", lambda x: x))
        
        await engine.execute({})
        
        assert ("before", "step1") in hook_calls
        assert ("before", "step2") in hook_calls

    @pytest.mark.asyncio
    async def test_after_step_hook(self) -> None:
        """After step hook is called."""
        from src.workflows import WorkflowEngine, DataTransform
        from src.workflows.steps import StepResult
        
        hook_calls = []
        
        def after_hook(step, result: StepResult) -> None:
            hook_calls.append(("after", step.name, result.succeeded))
        
        engine = WorkflowEngine(name="hook_test")
        engine.after_step(after_hook)
        engine.add_step(DataTransform("step1", lambda x: {"done": True}))
        
        await engine.execute({})
        
        assert len(hook_calls) == 1
        assert hook_calls[0][0] == "after"
        assert hook_calls[0][1] == "step1"
        assert hook_calls[0][2] is True


class TestStepStatus:
    """Test step status values."""

    def test_status_values(self) -> None:
        """All expected status values exist."""
        from src.workflows.steps import StepStatus
        
        assert StepStatus.PENDING is not None
        assert StepStatus.RUNNING is not None
        assert StepStatus.COMPLETED is not None
        assert StepStatus.FAILED is not None
        assert StepStatus.SKIPPED is not None


class TestWorkflowStatus:
    """Test workflow status values."""

    def test_status_values(self) -> None:
        """All expected workflow status values exist."""
        from src.workflows.engine import WorkflowStatus
        
        assert WorkflowStatus.PENDING is not None
        assert WorkflowStatus.RUNNING is not None
        assert WorkflowStatus.COMPLETED is not None
        assert WorkflowStatus.FAILED is not None
        assert WorkflowStatus.CANCELLED is not None
