# Feature Specification: AG-UI Interface Refactoring

**Date**: 2026-01-07 | **Priority**: P1 (High) | **Type**: Refactoring

## Problem Statement

The current AG-UI (Agent-User Interface) implementation in `src/agents/agui_server.py` is a **custom implementation** built from scratch. However, the project's research.md (Section 3) and plan.md clearly specify using the **Microsoft Agent Framework's native AG-UI support**.

### Current State (Custom Implementation)
- `src/agents/agui_server.py` (551 lines) implements:
  - Custom `EventType` enum mirroring AG-UI event types
  - Custom Pydantic models: `BaseEvent`, `RunStartedEvent`, `TextMessageContentEvent`, etc.
  - Custom `AGUIEventEmitter` dataclass for SSE formatting
  - Custom `AGUIServer` class with manual FastAPI route registration
  - Manual SSE streaming via `StreamingResponse`

### What Should Be Used (research.md Section 3)
```python
from agent_framework import ChatAgent
from agent_framework.azure import AzureOpenAIChatClient
from agent_framework.ag_ui import add_agent_framework_fastapi_endpoint
from fastapi import FastAPI

agent = ChatAgent(
    name="AGUIAssistant",
    instructions="You are helpful.",
    chat_client=AzureOpenAIChatClient(...)
)

app = FastAPI()
add_agent_framework_fastapi_endpoint(app, agent, "/")
```

### Also Available
- `ag-ui-core` package with official event types and `EventEncoder`

## Gap Analysis

| Aspect | Current Custom | Official SDK |
|--------|----------------|--------------|
| Event Types | Custom `EventType` enum | `ag_ui.core.EventType` |
| Event Models | Custom Pydantic models | `ag_ui.core.*Event` |
| SSE Encoding | Manual `_format_sse()` | `ag_ui.encoder.EventEncoder` |
| Server Setup | Custom `AGUIServer` class | `add_agent_framework_fastapi_endpoint()` |
| Agent Integration | Manual `_process_with_agent()` | Native `ChatAgent.run_stream()` |
| Tool Handling | Basic tool call events | Full tool execution lifecycle |

## Requirements

### Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-1 | Use `add_agent_framework_fastapi_endpoint()` for AG-UI server | Must |
| FR-2 | Event types from `ag-ui-core` package | Must |
| FR-3 | Native `ChatAgent` streaming integration | Must |
| FR-4 | Support for custom event emitter (for backward compat) | Should |
| FR-5 | Preserve existing endpoint paths (`/`, `/run`, `/health`) | Must |

### Non-Functional Requirements

| ID | Requirement | Metric |
|----|-------------|--------|
| NFR-1 | Test coverage | ≥80% |
| NFR-2 | Existing notebook works | 02_agui_interface.ipynb passes |
| NFR-3 | Contract tests pass | test_agui_schemas.py green |

## Acceptance Criteria

- [ ] AC-1: AG-UI server uses `add_agent_framework_fastapi_endpoint()` or `ag-ui-core` encoder
- [ ] AC-2: Event types imported from `ag-ui-core`, not custom definitions
- [ ] AC-3: Streaming integrates with `ChatAgent.run_stream()` natively
- [ ] AC-4: Notebook `02_agui_interface.ipynb` executes without errors
- [ ] AC-5: All AG-UI contract tests pass
- [ ] AC-6: Test coverage ≥80%

## Scope

### In Scope
- Refactor `src/agents/agui_server.py` to use official AG-UI SDK
- Update `notebooks/02_agui_interface.ipynb` to demonstrate native integration
- Update tests in `tests/contract/test_agui_schemas.py`
- Update integration tests in `tests/integration/test_scenario_02.py`

### Out of Scope
- A2A protocol refactoring (separate spec)
- Discussion framework (Phase 13, already done)
- MCP tools (working correctly)

## References
- research.md Section 3: AG-UI Protocol research
- contracts/agui-events.md: Event schemas
- plan.md: Original implementation plan
