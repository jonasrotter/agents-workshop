# Refactoring Plan: 04_deterministic_workflows.ipynb with Microsoft Agent Framework

**Date**: 2026-01-09 | **Spec**: [spec.md](./spec.md) | **Tasks**: [refactor-04-workflows-tasks.md](./refactor-04-workflows-tasks.md)

**Status**: ✅ COMPLETE (2026-01-09)

## Completion Summary

| Phase | Tasks | Status |
|-------|-------|--------|
| Phase 1-2: Setup & Foundation | T001-T007 | ✅ Complete |
| Phase 3: Sequential Orchestration | T008-T012 | ✅ Complete |
| Phase 4: Parallel Execution | T013-T016 | ✅ Complete |
| Phase 5: Conditional Branching | T017-T020 | ✅ Complete |
| Phase 6: Data Transforms | T021-T023 | ✅ Complete |
| Phase 7: Error Handling | T024-T028 | ✅ Complete |
| Phase 8: Builder Pattern | T029-T033 | ✅ Complete |
| Phase 9: Test Compatibility | T034-T036 | ✅ Complete (63/63 tests pass) |
| Phase 10: Deprecation | T037-T040 | ✅ Complete |
| Phase 11: Polish | T042-T043 | ✅ Complete |
| **Remaining** | T041, T044, T045 | ⏳ Manual validation pending |

---

## Executive Summary

This document outlines a plan to refactor `notebooks/04_deterministic_workflows.ipynb` to use Microsoft Agent Framework's `WorkflowBuilder` API instead of the custom workflow engine. The goal is to **minimize custom code** while maintaining educational value.

---

## Gap Analysis: Current vs. Spec

### Current Implementation

The notebook uses a **custom workflow engine** (`src/workflows/engine.py`) with:

| Component | Lines of Code | Purpose |
|-----------|---------------|---------|
| `WorkflowEngine` | ~150 | Sequential step orchestration |
| `WorkflowStep` (base) | ~50 | Abstract step interface |
| `SequentialStep` | ~30 | Group steps sequentially |
| `ParallelStep` | ~40 | Execute steps concurrently |
| `ConditionalStep` | ~30 | Branch based on condition |
| `DataTransform` | ~20 | Transform data between steps |
| `WorkflowBuilder` | ~100 | Fluent API for workflow construction |
| **Total Custom Code** | **~420 lines** | |

### Spec Requirement (from plan.md, research.md)

> "Use Microsoft Agent Framework (`agent-framework` Python package) for building agents with Azure OpenAI"

The research.md (Section 6) states:
> "Build a lightweight workflow engine for Scenario 4 (~100 lines)"

But this was written **before** Microsoft Agent Framework added `WorkflowBuilder` with:
- `add_edge()` for sequential flows
- Parallel execution via concurrent agent runs
- Conditional routing via `condition` parameter
- Native streaming with `WorkflowEvent` types

### Gap Summary

| Feature | Custom Engine | Agent Framework |
|---------|---------------|-----------------|
| Sequential orchestration | `SequentialStep` | `WorkflowBuilder.add_edge()` |
| Parallel execution | `ParallelStep` | Concurrent `ChatAgent.run()` calls |
| Conditional branching | `ConditionalStep` | `add_edge(condition=fn)` |
| Data transforms | `DataTransform` | Native prompt templating |
| Error handling | `ErrorStrategy` enum | Native exception handling |
| Retry logic | `RetryConfig` | Middleware pattern |
| Telemetry | Manual span creation | Built-in middleware |
| Streaming | Not implemented | `WorkflowEvent` stream |

---

## Recommended Refactoring Approach

### Option A: Replace Custom Engine Entirely (RECOMMENDED)

**Pros**: 
- Minimizes custom code (spec goal: "as little custom code as possible")
- Aligns with spec's Microsoft Agent Framework requirement
- Students learn production patterns
- Native streaming support

**Cons**:
- Loses some educational examples of building workflow engines
- Requires updating test fixtures

### Option B: Hybrid - Keep Custom Engine for Educational Value

**Pros**: Shows "how the sausage is made"
**Cons**: Doesn't align with spec's framework requirement

### Recommendation: **Option A** - Full refactor to Microsoft Agent Framework

---

## Implementation Plan

### Phase 1: Agent Framework Workflow Patterns (Replace Parts 3-6)

#### Current Custom Pattern (Part 3-6):
```python
# Custom step classes
class EchoStep(WorkflowStep):
    async def execute(self, inputs, context):
        ...

engine = WorkflowEngine(name="echo_pipeline")
engine.add_step(EchoStep("step1", "[Step 1]"))
result = await engine.execute({"message": "Hello"})
```

#### Refactored Pattern:
```python
from agent_framework import WorkflowBuilder
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import DefaultAzureCredential

# Create chat client once
chat_client = AzureOpenAIChatClient(credential=DefaultAzureCredential())

# Create specialized agents (instead of custom Step classes)
step1_agent = chat_client.create_agent(
    name="step1",
    instructions="Add '[Step 1]:' prefix to the input message."
)
step2_agent = chat_client.create_agent(
    name="step2", 
    instructions="Add '[Step 2]:' prefix to the input message."
)

# Build workflow with edges (instead of add_step)
workflow = (
    WorkflowBuilder()
    .set_start_executor(step1_agent)
    .add_edge(step1_agent, step2_agent)
    .build()
)

# Execute with streaming
async for event in workflow.run_stream("Hello, Workflow!"):
    if hasattr(event, 'data'):
        print(f"{event.executor_id}: {event.data}")
```

### Phase 2: Parallel Execution (Replace Part 5)

#### Current Custom Pattern:
```python
parallel_analysis = ParallelStep(
    name="parallel_analysis",
    steps=[
        AnalysisStep("sentiment", "sentiment"),
        AnalysisStep("entities", "entities"),
        AnalysisStep("topics", "topics"),
    ]
)
```

#### Refactored Pattern:
```python
import asyncio

# Create parallel agents
sentiment_agent = chat_client.create_agent(name="sentiment", instructions="Analyze sentiment...")
entities_agent = chat_client.create_agent(name="entities", instructions="Extract entities...")
topics_agent = chat_client.create_agent(name="topics", instructions="Identify topics...")

# Parallel execution using asyncio.gather
async def parallel_analysis(text: str):
    """Execute multiple agents in parallel."""
    results = await asyncio.gather(
        sentiment_agent.run(text),
        entities_agent.run(text),
        topics_agent.run(text),
    )
    return {
        "sentiment": results[0],
        "entities": results[1],
        "topics": results[2],
    }

result = await parallel_analysis("The quick brown fox...")
```

### Phase 3: Conditional Branching (Replace Part 6)

#### Current Custom Pattern:
```python
conditional_processing = ConditionalStep(
    name="length_check",
    condition=is_long_text,
    then_step=ProcessingStep("summarize"),
    else_step=ProcessingStep("expand"),
)
```

#### Refactored Pattern:
```python
def route_by_length(state: dict) -> str:
    """Route to different agents based on text length."""
    text = state.get("text", "")
    return "summarize_agent" if len(text) > 100 else "expand_agent"

# Build conditional workflow
workflow = (
    WorkflowBuilder()
    .set_start_executor(router_agent)
    .add_edge(router_agent, summarize_agent, condition=lambda s: len(s.get("text", "")) > 100)
    .add_edge(router_agent, expand_agent, condition=lambda s: len(s.get("text", "")) <= 100)
    .build()
)
```

### Phase 4: Error Handling (Replace Part 8-9)

#### Current Custom Pattern:
```python
abort_engine = WorkflowEngine(
    name="abort_test",
    error_config=ErrorConfig(strategy=ErrorStrategy.ABORT)
)
```

#### Refactored Pattern:
```python
from agent_framework import ChatAgent, ChatMessage

class RetryMiddleware:
    """Simple retry middleware for agent calls."""
    def __init__(self, max_retries: int = 3, delay: float = 1.0):
        self.max_retries = max_retries
        self.delay = delay
    
    async def __call__(self, next_handler, request):
        for attempt in range(self.max_retries):
            try:
                return await next_handler(request)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(self.delay)

# Agent with retry middleware
agent_with_retry = ChatAgent(
    chat_client=chat_client,
    instructions="...",
    middleware=[RetryMiddleware(max_retries=3)]
)

# Error handling in workflow
async def safe_workflow(task: str):
    """Workflow with error handling."""
    try:
        async for event in workflow.run_stream(task):
            yield event
    except Exception as e:
        # Fallback behavior
        yield {"error": str(e), "fallback": "Default response"}
```

### Phase 5: Builder Pattern (Replace Part 10)

#### Current Custom Pattern:
```python
workflow = (
    WorkflowBuilder("research_pipeline")
    .with_agent("researcher", MockAgent(...))
    .with_agent("analyzer", MockAgent(...))
    .add_agent_step(name="research", agent="researcher", ...)
    .add_agent_step(name="analyze", agent="analyzer", ...)
    .build()
)
```

#### Refactored Pattern:
```python
from agent_framework import WorkflowBuilder, AgentRunEvent

# Create agents
researcher = chat_client.create_agent(
    name="researcher",
    instructions="Research the given topic thoroughly."
)
analyzer = chat_client.create_agent(
    name="analyzer", 
    instructions="Analyze findings and provide insights."
)
summarizer = chat_client.create_agent(
    name="summarizer",
    instructions="Create a concise summary."
)

# Build workflow pipeline
workflow = (
    WorkflowBuilder()
    .set_start_executor(researcher)
    .add_edge(researcher, analyzer)
    .add_edge(analyzer, summarizer)
    .build()
)

# Execute and collect results
events = await workflow.run("Research: AI agents in production")
for event in events:
    if isinstance(event, AgentRunEvent):
        print(f"[{event.executor_id}]: {event.data}")
```

---

## Code Reduction Summary

| Component | Before (Custom) | After (Framework) | Reduction |
|-----------|-----------------|-------------------|-----------|
| Sequential orchestration | ~150 lines | ~10 lines | **93%** |
| Parallel execution | ~40 lines | ~15 lines | **63%** |
| Conditional branching | ~30 lines | ~10 lines | **67%** |
| Error handling | ~80 lines | ~20 lines | **75%** |
| Workflow builder | ~100 lines | ~0 lines (use framework) | **100%** |
| **Total** | **~420 lines** | **~55 lines** | **87%** |

---

## Files to Modify

### Notebook Changes

| Section | Current Content | Refactored Content |
|---------|-----------------|-------------------|
| Part 2 | Import custom `WorkflowEngine` | Import `WorkflowBuilder`, `AzureOpenAIChatClient` |
| Part 3 | Custom `EchoStep` class | `ChatAgent` with instructions |
| Part 4 | `SequentialStep` demo | `WorkflowBuilder.add_edge()` demo |
| Part 5 | Custom `ParallelStep` | `asyncio.gather()` with agents |
| Part 6 | Custom `ConditionalStep` | `add_edge(condition=...)` |
| Part 7 | `DataTransform` class | Native prompt templating |
| Part 8-9 | `ErrorStrategy` enum | Middleware pattern |
| Part 10 | Custom `WorkflowBuilder` | Framework `WorkflowBuilder` |
| Part 11 | Custom validation | Framework validation |

### Source Code Impact

| File | Action | Notes |
|------|--------|-------|
| `src/workflows/engine.py` | **DEPRECATE** | Keep for reference, mark as deprecated |
| `src/workflows/steps.py` | **DEPRECATE** | Keep for reference, mark as deprecated |
| `tests/unit/test_workflows.py` | **UPDATE** | Test new WorkflowBuilder patterns |
| `tests/integration/test_scenario_04.py` | **UPDATE** | Verify notebook execution |

---

## Migration Tasks

### T201 [REFACTOR] Study WorkflowBuilder API
- Review `WorkflowBuilder.set_start_executor()`, `add_edge()`, `build()` 
- Test streaming with `run_stream()` and `WorkflowEvent` types
- Document parallel execution patterns

### T202 [REFACTOR] Refactor Part 3-4: Sequential Orchestration
- Replace `WorkflowEngine.add_step()` with `WorkflowBuilder.add_edge()`
- Replace custom `EchoStep` with `ChatAgent` instances
- Update telemetry to use built-in tracing

### T203 [REFACTOR] Refactor Part 5: Parallel Execution
- Replace `ParallelStep` with `asyncio.gather()` pattern
- Demonstrate concurrent agent execution
- Show result aggregation

### T204 [REFACTOR] Refactor Part 6: Conditional Branching
- Replace `ConditionalStep` with `add_edge(condition=fn)` 
- Show routing function pattern
- Demo multiple conditional paths

### T205 [REFACTOR] Refactor Part 7: Data Transforms
- Replace `DataTransform` with prompt templating
- Show context passing between agents
- Demo variable substitution

### T206 [REFACTOR] Refactor Part 8-9: Error Handling
- Replace `ErrorStrategy` enum with middleware pattern
- Implement `RetryMiddleware` example
- Show fallback patterns

### T207 [REFACTOR] Refactor Part 10-11: Builder & Validation
- Replace custom `WorkflowBuilder` with framework version
- Show validation through type system
- Update hands-on exercise

### T208 [REFACTOR] Update Tests
- Update `tests/unit/test_workflows.py` for new patterns
- Ensure `tests/integration/test_scenario_04.py` passes
- Add contract tests for `WorkflowBuilder`

### T209 [REFACTOR] Mark Custom Engine as Deprecated
- Add deprecation warnings to `src/workflows/engine.py`
- Add deprecation warnings to `src/workflows/steps.py`
- Update docstrings with migration guidance

---

## Learning Objectives (Updated)

The refactored notebook will teach:

1. **WorkflowBuilder Pattern** - Official framework for multi-agent orchestration
2. **Sequential Flows** - Using `add_edge()` for step chains
3. **Parallel Execution** - Using `asyncio.gather()` with agents
4. **Conditional Routing** - Using condition functions with edges
5. **Error Resilience** - Middleware patterns for retry/fallback
6. **Streaming Events** - `WorkflowEvent` types for real-time updates
7. **Production Patterns** - Framework-native approaches vs custom code

---

## Appendix: Key Microsoft Agent Framework Imports

```python
# Core imports for workflows
from agent_framework import (
    WorkflowBuilder,      # Build multi-agent workflows
    AgentRunEvent,        # Event from agent execution
    ChatAgent,            # Basic agent class
    ChatMessage,          # Message structure
)
from agent_framework.azure import (
    AzureOpenAIChatClient,  # Azure OpenAI client
)
from azure.identity import (
    DefaultAzureCredential,  # Azure authentication
)
```

---

## References

- [Microsoft Agent Framework - Python README](https://github.com/microsoft/agent-framework/blob/main/python/README.md)
- [WorkflowBuilder Examples](https://github.com/microsoft/agent-framework/tree/main/python/samples)
- [Multi-Agent Orchestration](https://github.com/microsoft/agent-framework/blob/main/dotnet/samples/AzureFunctions/03_AgentOrchestration_Concurrency/)
- [research.md](./research.md) - Section 6 (Workflow Engine) and Section 8 (Discussion Framework)
