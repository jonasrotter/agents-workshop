# Feature Specification: A2A Server Refactoring

**Date**: 2026-01-06 | **Priority**: P0 (Critical) | **Type**: Refactoring

## Problem Statement

The current A2A (Agent-to-Agent) implementation in `src/agents/a2a_server.py` is a **custom implementation** built from scratch using FastAPI and Pydantic. However, the project's `requirements.txt` includes official packages that should be used instead:

1. **`a2a-sdk==0.3.17`** - Official Google A2A SDK with:
   - `A2AFastAPIApplication` - Ready-to-use FastAPI app
   - `DefaultRequestHandler` - Standard request handling
   - `a2a.types.*` - All A2A types (AgentCard, Task, Message, etc.)
   - Task management, push notifications, streaming support

2. **`agent-framework-a2a==1.0.0b251120`** - Microsoft Agent Framework A2A integration with:
   - `A2AAgent` - Client wrapper for calling A2A agents
   - Integration with Agent Framework's `ChatAgent`

3. **`a2a-server`** - Server utilities package

## Current State Analysis

### What Exists (Custom Implementation)
- `src/agents/a2a_server.py` (713 lines) implements:
  - Custom Pydantic models: `AgentCard`, `Skill`, `Task`, `Message`, `TextPart`, etc.
  - Custom `TaskManager` class for in-memory task storage
  - Custom `A2AServer` class with FastAPI endpoint creation
  - JSON-RPC handlers: `message/send`, `tasks/get`, `tasks/list`, `tasks/cancel`
  - Agent Card endpoint at `/.well-known/agent-card.json`

### What Should Be Used (Official SDK)
- `a2a.types.*` - Standard types (AgentCard, Task, TaskState, etc.)
- `a2a.server.apps.A2AFastAPIApplication` - Standard FastAPI app builder
- `a2a.server.request_handlers.DefaultRequestHandler` - Standard handlers
- `agent_framework_a2a.A2AAgent` - Client for calling A2A agents

## Gap Analysis

| Aspect | Current Custom | Official SDK |
|--------|----------------|--------------|
| Types | Custom Pydantic models | `a2a.types.*` |
| Server App | Custom `A2AServer` class | `A2AFastAPIApplication` |
| Request Handling | Custom `_handle_rpc()` | `DefaultRequestHandler` |
| Task Store | Custom `TaskManager` | SDK's `TaskStore` interface |
| Agent Executor | Direct agent call | SDK's `AgentExecutor` interface |
| Push Notifications | Not implemented | Built-in support |
| Streaming | Partial | Full SSE support |

## Requirements

### Functional Requirements

**FR-001**: Replace custom Pydantic models with `a2a.types` imports
- Use `a2a.types.AgentCard`, `a2a.types.Task`, `a2a.types.Message`, etc.
- Remove duplicate type definitions from `a2a_server.py`

**FR-002**: Implement A2A server using `A2AFastAPIApplication`
- Create a new implementation using the SDK's `A2AFastAPIApplication.build()` method
- Provide custom `AgentExecutor` to wrap existing agents
- Implement `TaskStore` for task persistence

**FR-003**: Maintain backward compatibility for notebook imports
- Keep existing module exports in `src/agents/__init__.py`
- Ensure `notebooks/03_a2a_protocol.ipynb` works without changes
- Alias or re-export SDK types under existing names if needed

**FR-004**: Integrate with Microsoft Agent Framework
- Use `A2AAgent` from `agent-framework-a2a` for client-side A2A calls
- Demonstrate calling external A2A agents from workshop agents

### Non-Functional Requirements

**NFR-001**: Preserve OpenTelemetry integration
- Maintain tracing spans for A2A operations
- Use SDK hooks/middleware for telemetry injection

**NFR-002**: Maintain test coverage
- Update `tests/contract/test_a2a_schemas.py` for SDK types
- Update `tests/integration/test_scenario_03.py` as needed

**NFR-003**: Update documentation
- Update notebook comments to reference SDK usage
- Document migration from custom to SDK implementation

## Success Criteria

1. ✅ All A2A types imported from `a2a.types` (no custom duplicates)
2. ✅ Server built using `A2AFastAPIApplication`
3. ✅ Custom `AgentExecutor` wraps existing agents
4. ✅ All existing tests pass
5. ✅ Notebook 03 executes successfully
6. ✅ OpenTelemetry traces still captured

## Out of Scope

- Push notification implementation (can be added later)
- Task resubscription streaming
- External A2A agent registry integration
