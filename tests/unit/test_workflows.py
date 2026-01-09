"""
Unit tests for workflow engine and step definitions.

Tests the workflow engine logic including:
- Step execution and composition
- Error handling strategies
- Retry configuration
- Data flow between steps

Note: The src.workflows module is deprecated in favor of Microsoft Agent Framework's
WorkflowBuilder. These tests ensure backward compatibility during migration.
"""

import pytest
import warnings
from unittest.mock import AsyncMock, MagicMock
from typing import Any

# Suppress deprecation warnings for these tests (testing legacy code)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="src.workflows")
    from src.workflows.steps import (
        WorkflowStep,
        WorkflowContext,
        StepResult,
        StepStatus,
        RetryConfig,
        SequentialStep,
        ParallelStep,
        ConditionalStep,
        DataTransform,
        AgentStep,
    )
    from src.workflows.engine import (
        WorkflowEngine,
        WorkflowResult,
        WorkflowStatus,
        ErrorStrategy,
        ErrorConfig,
        create_workflow,
        WorkflowBuilder,
    )


class SimpleStep(WorkflowStep):
    """Simple step for testing."""
    
    def __init__(
        self,
        name: str,
        output_key: str = "result",
        output_value: Any = "success",
    ) -> None:
        super().__init__(name)
        self.output_key = output_key
        self.output_value = output_value
    
    async def execute(
        self,
        inputs: dict[str, Any],
        context: WorkflowContext,
    ) -> StepResult:
        return StepResult(
            step_name=self.name,
            status=StepStatus.COMPLETED,
            outputs={self.output_key: self.output_value},
        )


class FailingStep(WorkflowStep):
    """Step that always fails."""
    
    async def execute(
        self,
        inputs: dict[str, Any],
        context: WorkflowContext,
    ) -> StepResult:
        return StepResult(
            step_name=self.name,
            status=StepStatus.FAILED,
            error="Intentional failure",
        )


class InputEchoStep(WorkflowStep):
    """Step that echoes input to output."""
    
    async def execute(
        self,
        inputs: dict[str, Any],
        context: WorkflowContext,
    ) -> StepResult:
        return StepResult(
            step_name=self.name,
            status=StepStatus.COMPLETED,
            outputs=inputs.copy(),
        )


class TestStepResult:
    """Test StepResult class."""

    def test_succeeded_completed(self) -> None:
        """Completed status means succeeded."""
        result = StepResult(
            step_name="test",
            status=StepStatus.COMPLETED,
        )
        assert result.succeeded is True

    def test_succeeded_failed(self) -> None:
        """Failed status means not succeeded."""
        result = StepResult(
            step_name="test",
            status=StepStatus.FAILED,
        )
        assert result.succeeded is False

    def test_default_values(self) -> None:
        """Default values are set correctly."""
        result = StepResult(
            step_name="test",
            status=StepStatus.PENDING,
        )
        assert result.outputs == {}
        assert result.error is None
        assert result.duration_ms == 0.0


class TestWorkflowContext:
    """Test WorkflowContext class."""

    def test_set_get_variable(self) -> None:
        """Variables can be set and retrieved."""
        context = WorkflowContext(workflow_name="test")
        context.set_variable("key", "value")
        
        assert context.get_variable("key") == "value"

    def test_get_missing_variable(self) -> None:
        """Missing variables return default."""
        context = WorkflowContext(workflow_name="test")
        
        assert context.get_variable("missing") is None
        assert context.get_variable("missing", "default") == "default"

    def test_add_step_result(self) -> None:
        """Step results are recorded."""
        context = WorkflowContext(workflow_name="test")
        result = StepResult(step_name="step1", status=StepStatus.COMPLETED)
        
        context.add_step_result(result)
        
        assert "step1" in context.step_results


class TestSimpleStep:
    """Test basic step execution."""

    @pytest.mark.asyncio
    async def test_execute_returns_result(self) -> None:
        """Step execute returns result."""
        step = SimpleStep("test_step")
        context = WorkflowContext(workflow_name="test")
        
        result = await step.execute({}, context)
        
        assert result.status == StepStatus.COMPLETED
        assert result.step_name == "test_step"

    @pytest.mark.asyncio
    async def test_execute_produces_output(self) -> None:
        """Step execute produces configured output."""
        step = SimpleStep("test_step", "my_key", "my_value")
        context = WorkflowContext(workflow_name="test")
        
        result = await step.execute({}, context)
        
        assert result.outputs["my_key"] == "my_value"


class TestSequentialStep:
    """Test sequential step execution."""

    @pytest.mark.asyncio
    async def test_executes_in_order(self) -> None:
        """Child steps execute in order."""
        execution_order = []
        
        class OrderTrackingStep(WorkflowStep):
            async def execute(self, inputs, context):
                execution_order.append(self.name)
                return StepResult(
                    step_name=self.name,
                    status=StepStatus.COMPLETED,
                    outputs={},
                )
        
        seq = SequentialStep(
            name="sequential",
            steps=[
                OrderTrackingStep("step1"),
                OrderTrackingStep("step2"),
                OrderTrackingStep("step3"),
            ],
        )
        
        context = WorkflowContext(workflow_name="test")
        await seq.execute({}, context)
        
        assert execution_order == ["step1", "step2", "step3"]

    @pytest.mark.asyncio
    async def test_passes_outputs_to_next(self) -> None:
        """Outputs from one step become inputs to next."""
        class OutputStep(WorkflowStep):
            async def execute(self, inputs, context):
                return StepResult(
                    step_name=self.name,
                    status=StepStatus.COMPLETED,
                    outputs={"from_" + self.name: True},
                )
        
        seq = SequentialStep(
            name="sequential",
            steps=[
                OutputStep("step1"),
                InputEchoStep("step2"),
            ],
        )
        
        context = WorkflowContext(workflow_name="test")
        result = await seq.execute({}, context)
        
        assert "from_step1" in result.outputs

    @pytest.mark.asyncio
    async def test_fails_on_child_failure(self) -> None:
        """Sequential step fails if child fails."""
        seq = SequentialStep(
            name="sequential",
            steps=[
                SimpleStep("step1"),
                FailingStep("step2"),
                SimpleStep("step3"),
            ],
        )
        
        context = WorkflowContext(workflow_name="test")
        result = await seq.execute({}, context)
        
        assert result.status == StepStatus.FAILED


class TestParallelStep:
    """Test parallel step execution."""

    @pytest.mark.asyncio
    async def test_merges_outputs(self) -> None:
        """Outputs from parallel steps are merged."""
        par = ParallelStep(
            name="parallel",
            steps=[
                SimpleStep("step1", "key1", "value1"),
                SimpleStep("step2", "key2", "value2"),
            ],
        )
        
        context = WorkflowContext(workflow_name="test")
        result = await par.execute({}, context)
        
        assert result.outputs["key1"] == "value1"
        assert result.outputs["key2"] == "value2"

    @pytest.mark.asyncio
    async def test_fail_fast(self) -> None:
        """Fails immediately if fail_fast is True."""
        par = ParallelStep(
            name="parallel",
            steps=[
                SimpleStep("step1"),
                FailingStep("step2"),
            ],
            fail_fast=True,
        )
        
        context = WorkflowContext(workflow_name="test")
        result = await par.execute({}, context)
        
        assert result.status == StepStatus.FAILED

    @pytest.mark.asyncio
    async def test_continues_without_fail_fast(self) -> None:
        """Continues on failure if fail_fast is False."""
        par = ParallelStep(
            name="parallel",
            steps=[
                SimpleStep("step1", "key1", "value1"),
                FailingStep("step2"),
            ],
            fail_fast=False,
        )
        
        context = WorkflowContext(workflow_name="test")
        result = await par.execute({}, context)
        
        # Should complete with partial results
        assert "key1" in result.outputs


class TestConditionalStep:
    """Test conditional step execution."""

    @pytest.mark.asyncio
    async def test_executes_then_branch(self) -> None:
        """Executes then_step when condition is True."""
        cond = ConditionalStep(
            name="conditional",
            condition=lambda inputs, ctx: True,
            then_step=SimpleStep("then", "branch", "then"),
            else_step=SimpleStep("else", "branch", "else"),
        )
        
        context = WorkflowContext(workflow_name="test")
        result = await cond.execute({}, context)
        
        assert result.outputs["branch"] == "then"

    @pytest.mark.asyncio
    async def test_executes_else_branch(self) -> None:
        """Executes else_step when condition is False."""
        cond = ConditionalStep(
            name="conditional",
            condition=lambda inputs, ctx: False,
            then_step=SimpleStep("then", "branch", "then"),
            else_step=SimpleStep("else", "branch", "else"),
        )
        
        context = WorkflowContext(workflow_name="test")
        result = await cond.execute({}, context)
        
        assert result.outputs["branch"] == "else"

    @pytest.mark.asyncio
    async def test_skips_without_else(self) -> None:
        """Skips when condition is False and no else_step."""
        cond = ConditionalStep(
            name="conditional",
            condition=lambda inputs, ctx: False,
            then_step=SimpleStep("then"),
        )
        
        context = WorkflowContext(workflow_name="test")
        result = await cond.execute({}, context)
        
        assert result.status == StepStatus.SKIPPED


class TestDataTransform:
    """Test data transform step."""

    @pytest.mark.asyncio
    async def test_transforms_data(self) -> None:
        """Transform function is applied to inputs."""
        transform = DataTransform(
            name="transform",
            transform=lambda inputs: {"doubled": inputs.get("value", 0) * 2},
        )
        
        context = WorkflowContext(workflow_name="test")
        result = await transform.execute({"value": 5}, context)
        
        assert result.outputs["doubled"] == 10

    @pytest.mark.asyncio
    async def test_handles_transform_error(self) -> None:
        """Transform errors are captured."""
        def failing_transform(inputs):
            raise ValueError("Transform error")
        
        transform = DataTransform(
            name="transform",
            transform=failing_transform,
        )
        
        context = WorkflowContext(workflow_name="test")
        result = await transform.execute({}, context)
        
        assert result.status == StepStatus.FAILED
        assert "Transform failed" in result.error


class TestAgentStep:
    """Test agent step execution."""

    def test_render_prompt(self) -> None:
        """Prompt template renders with variables."""
        step = AgentStep(
            name="agent_step",
            agent_name="test_agent",
            prompt_template="Hello {name}, how are you?",
            output_vars=["response"],
        )
        
        rendered = step._render_prompt({"name": "World"})
        
        assert rendered == "Hello World, how are you?"

    @pytest.mark.asyncio
    async def test_missing_agent_fails(self) -> None:
        """Fails when agent is not found."""
        step = AgentStep(
            name="agent_step",
            agent_name="missing_agent",
            prompt_template="Hello",
            output_vars=["response"],
        )
        
        context = WorkflowContext(
            workflow_name="test",
            metadata={"agents": {}},
        )
        result = await step.execute({}, context)
        
        assert result.status == StepStatus.FAILED
        assert "not found" in result.error


class TestWorkflowEngine:
    """Test workflow engine."""

    @pytest.mark.asyncio
    async def test_executes_steps(self) -> None:
        """Engine executes all steps."""
        engine = WorkflowEngine(name="test")
        engine.add_step(SimpleStep("step1", "key1", "value1"))
        engine.add_step(SimpleStep("step2", "key2", "value2"))
        
        result = await engine.execute({})
        
        assert result.status == WorkflowStatus.COMPLETED
        assert result.outputs["key1"] == "value1"
        assert result.outputs["key2"] == "value2"

    @pytest.mark.asyncio
    async def test_abort_on_failure(self) -> None:
        """Aborts on failure with ABORT strategy."""
        engine = WorkflowEngine(
            name="test",
            error_config=ErrorConfig(strategy=ErrorStrategy.ABORT),
        )
        engine.add_step(SimpleStep("step1"))
        engine.add_step(FailingStep("step2"))
        engine.add_step(SimpleStep("step3"))
        
        result = await engine.execute({})
        
        assert result.status == WorkflowStatus.FAILED
        assert "step2" in result.error

    @pytest.mark.asyncio
    async def test_skip_on_failure(self) -> None:
        """Continues on failure with SKIP strategy."""
        engine = WorkflowEngine(
            name="test",
            error_config=ErrorConfig(strategy=ErrorStrategy.SKIP),
        )
        engine.add_step(SimpleStep("step1", "key1", "value1"))
        engine.add_step(FailingStep("step2"))
        engine.add_step(SimpleStep("step3", "key3", "value3"))
        
        result = await engine.execute({})
        
        assert result.status == WorkflowStatus.COMPLETED
        assert "key1" in result.outputs
        assert "key3" in result.outputs

    @pytest.mark.asyncio
    async def test_timeout(self) -> None:
        """Workflow times out."""
        import asyncio
        
        class SlowStep(WorkflowStep):
            async def execute(self, inputs, context):
                await asyncio.sleep(5)
                return StepResult(
                    step_name=self.name,
                    status=StepStatus.COMPLETED,
                )
        
        engine = WorkflowEngine(name="test")
        engine.add_step(SlowStep("slow"))
        
        result = await engine.execute({}, timeout=0.1)
        
        assert result.status == WorkflowStatus.FAILED
        assert "timed out" in result.error

    def test_register_agent(self) -> None:
        """Agents can be registered."""
        engine = WorkflowEngine(name="test")
        mock_agent = MagicMock()
        
        engine.register_agent("test_agent", mock_agent)
        
        assert "test_agent" in engine.agents

    def test_add_step_chaining(self) -> None:
        """add_step returns self for chaining."""
        engine = WorkflowEngine(name="test")
        
        result = engine.add_step(SimpleStep("step1"))
        
        assert result is engine

    def test_validate_empty(self) -> None:
        """Validates empty workflow."""
        engine = WorkflowEngine(name="test")
        
        errors = engine.validate()
        
        assert "no steps" in errors[0]

    def test_validate_missing_agent(self) -> None:
        """Validates missing agent references."""
        engine = WorkflowEngine(name="test")
        engine.add_step(AgentStep(
            name="step1",
            agent_name="missing",
            prompt_template="Hello",
            output_vars=["result"],
        ))
        
        errors = engine.validate()
        
        assert any("missing" in e for e in errors)


class TestCreateWorkflow:
    """Test create_workflow factory."""

    def test_creates_engine(self) -> None:
        """Creates configured engine."""
        workflow = create_workflow(
            name="test",
            steps=[SimpleStep("step1")],
            description="Test workflow",
        )
        
        assert workflow.name == "test"
        assert workflow.description == "Test workflow"
        assert len(workflow.steps) == 1

    def test_registers_agents(self) -> None:
        """Registers provided agents."""
        mock_agent = MagicMock()
        workflow = create_workflow(
            name="test",
            steps=[],
            agents={"agent1": mock_agent},
        )
        
        assert "agent1" in workflow.agents


class TestWorkflowBuilder:
    """Test WorkflowBuilder fluent API."""

    def test_builds_workflow(self) -> None:
        """Builder creates valid workflow."""
        workflow = (
            WorkflowBuilder("test")
            .with_description("Test workflow")
            .add_step(SimpleStep("step1"))
            .build()
        )
        
        assert workflow.name == "test"
        assert workflow.description == "Test workflow"
        assert len(workflow.steps) == 1

    def test_adds_agent_step(self) -> None:
        """Builder adds agent steps."""
        workflow = (
            WorkflowBuilder("test")
            .with_agent("agent1", MagicMock())
            .add_agent_step(
                name="step1",
                agent="agent1",
                prompt="Hello",
                outputs=["result"],
            )
            .build()
        )
        
        assert len(workflow.steps) == 1
        assert isinstance(workflow.steps[0], AgentStep)

    def test_configures_error_handling(self) -> None:
        """Builder configures error handling."""
        workflow = (
            WorkflowBuilder("test")
            .on_error(ErrorStrategy.SKIP)
            .build()
        )
        
        assert workflow.error_config.strategy == ErrorStrategy.SKIP


class TestRetryConfig:
    """Test retry configuration."""

    def test_default_values(self) -> None:
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
        assert config.delay_seconds == 0.5
        assert config.backoff_multiplier == 1.5
