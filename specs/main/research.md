# Research: A2A Server Refactoring

**Date**: 2026-01-06 | **Plan**: [plan.md](./plan.md) | **Spec**: [spec.md](./spec.md)

This document consolidates research findings for refactoring the A2A server implementation to use the official `a2a-sdk` package.

---

## 1. Current Implementation Analysis

### Custom Implementation Details

The current `src/agents/a2a_server.py` is a **completely custom implementation** (713 lines) that does not use any of the official A2A packages available in `requirements.txt`.

#### Custom Components

| Component | Lines | Description |
|-----------|-------|-------------|
| `TaskState` enum | 10 | Task state values (submitted, working, completed, etc.) |
| `AuthType` enum | 8 | Authentication types |
| `ErrorCode` class | 12 | JSON-RPC error codes |
| `Skill` model | 10 | Agent skill definition |
| `AuthConfig` model | 6 | Auth configuration |
| `Capabilities` model | 8 | Agent capabilities |
| `Provider` model | 6 | Provider info |
| `AgentCard` model | 20 | Agent card structure |
| `TextPart`, `FilePart`, `DataPart` | 30 | Message parts |
| `Message` model | 10 | A2A message |
| `JSONRPCError`, `JSONRPCRequest`, `JSONRPCResponse` | 25 | JSON-RPC models |
| `TaskStatus`, `Artifact`, `Task` | 30 | Task models |
| Parameter models | 40 | Method parameter schemas |
| `TaskManager` class | 80 | In-memory task storage |
| `A2AServer` class | 200 | Server implementation |
| Request handlers | 150 | `_handle_message_send`, `_handle_tasks_get`, etc. |
| Helper functions | 50 | `create_a2a_server`, `create_default_app` |

### What's Missing vs SDK

| Feature | Custom | SDK |
|---------|--------|-----|
| Task store persistence | In-memory only | Pluggable `TaskStore` interface |
| Push notifications | Not implemented | Full support |
| Streaming responses | Partial | Full SSE support |
| Task resubscription | Not implemented | Built-in |
| Extended agent cards | Not implemented | Full support |
| Content length limits | Not implemented | Configurable |

---

## 2. Official SDK Package Analysis

### a2a-sdk (v0.3.17)

**Purpose**: Official Google A2A protocol SDK for Python

#### Package Structure

```
a2a/
├── types.py              # All Pydantic models (AgentCard, Task, Message, etc.)
├── server/
│   ├── apps/
│   │   └── A2AFastAPIApplication  # FastAPI app builder
│   ├── request_handlers/
│   │   ├── RequestHandler         # Interface/protocol
│   │   └── DefaultRequestHandler  # Standard implementation
│   ├── agent_executor.py          # AgentExecutor interface
│   ├── task_store.py              # TaskStore interface + InMemoryTaskStore
│   └── context.py                 # ServerCallContext
└── client/                        # Client-side utilities
```

#### Key Classes

**`A2AFastAPIApplication`**
```python
class A2AFastAPIApplication:
    """FastAPI application implementing A2A protocol server endpoints."""
    
    def __init__(
        self,
        agent_card: AgentCard,
        http_handler: RequestHandler,
        extended_agent_card: AgentCard | None = None,
        context_builder: CallContextBuilder | None = None,
        card_modifier: Callable[[AgentCard], AgentCard] | None = None,
        max_content_length: int | None = 10485760
    ): ...
    
    def build(self) -> FastAPI: ...
    def add_routes_to_app(self, app: FastAPI) -> None: ...
```

**`RequestHandler`** (Interface)
```python
class RequestHandler:
    """A2A request handler interface."""
    
    async def on_message_send(self, request: SendMessageRequest, context: ServerCallContext) -> Task: ...
    async def on_message_send_stream(self, request, context) -> AsyncIterator[Event]: ...
    async def on_get_task(self, request: GetTaskRequest, context: ServerCallContext) -> Task: ...
    async def on_cancel_task(self, request: CancelTaskRequest, context: ServerCallContext) -> Task: ...
    async def on_resubscribe_to_task(self, request, context) -> AsyncIterator[Event]: ...
    # Push notification handlers...
```

**`DefaultRequestHandler`**
```python
class DefaultRequestHandler(RequestHandler):
    """Default request handler coordinating AgentExecutor, TaskStore, QueueManager."""
    
    def __init__(
        self,
        agent_executor: AgentExecutor,
        task_store: TaskStore,
        queue_manager: QueueManager | None = None,
        push_notifier: PushNotifier | None = None
    ): ...
```

**`InMemoryTaskStore`**
```python
class InMemoryTaskStore(TaskStore):
    """In-memory task storage implementation."""
    
    async def create(self, task: Task) -> Task: ...
    async def get(self, task_id: str) -> Task | None: ...
    async def update(self, task: Task) -> Task: ...
    async def delete(self, task_id: str) -> bool: ...
    async def list(self, filter: TaskFilter) -> list[Task]: ...
```

#### Types Available (`a2a.types`)

```python
# Agent Card
AgentCard, AgentCapabilities, AgentSkill, AgentProvider, AgentExtension

# Authentication
SecurityScheme, APIKeySecurityScheme, HTTPAuthSecurityScheme, OAuth2SecurityScheme

# Tasks
Task, TaskState, TaskStatus, Artifact

# Messages  
Message, Role, Part, TextPart, FilePart, DataPart, FileWithBytes, FileWithUri

# JSON-RPC
JSONRPCRequest, JSONRPCResponse, JSONRPCSuccessResponse, JSONRPCErrorResponse, JSONRPCError

# Requests/Responses
SendMessageRequest, SendMessageResponse, SendMessageSuccessResponse
SendStreamingMessageRequest, SendStreamingMessageResponse
GetTaskRequest, GetTaskResponse, GetTaskSuccessResponse
CancelTaskRequest, CancelTaskResponse, CancelTaskSuccessResponse

# Events (for streaming)
TaskStatusUpdateEvent, TaskArtifactUpdateEvent

# Errors
A2AError, InvalidParamsError, InvalidRequestError, MethodNotFoundError
TaskNotFoundError, TaskNotCancelableError, InternalError
```

---

## 3. agent-framework-a2a (v1.0.0b251120)

**Purpose**: Microsoft Agent Framework integration for calling A2A agents

#### Key Class

```python
from agent_framework_a2a import A2AAgent

class A2AAgent:
    """Agent2Agent (A2A) protocol implementation.
    
    Wraps an A2A Client to connect the Agent Framework with external 
    A2A-compliant agents via HTTP/JSON-RPC.
    """
    
    @classmethod
    def from_url(cls, url: str, **kwargs) -> A2AAgent: ...
    
    @classmethod  
    def from_dict(cls, data: dict) -> A2AAgent: ...
    
    async def run(self, prompt: str) -> str: ...
    async def run_stream(self, prompt: str) -> AsyncIterator[str]: ...
    
    def as_tool(self) -> Tool: ...  # Use A2A agent as a tool
```

#### Usage Pattern

```python
# Create client from URL
external_agent = A2AAgent.from_url("https://research-agent.example.com")

# Call synchronously
result = await external_agent.run("Research quantum computing")

# Call with streaming
async for chunk in external_agent.run_stream("Research AI safety"):
    print(chunk, end="")

# Use as a tool for another agent
tool = external_agent.as_tool()
coordinator = ChatAgent(tools=[tool])
```

---

## 4. Decision: Refactoring Approach

### Option A: Full SDK Integration (RECOMMENDED)

Replace all custom code with SDK components:
- Use `A2AFastAPIApplication` for app creation
- Implement custom `RequestHandler` for workshop agents
- Use `InMemoryTaskStore` for task management
- Import all types from `a2a.types`

**Pros**:
- Maximum code reduction (~80%)
- Production-ready features (streaming, push notifications)
- Future SDK updates automatically available
- Matches spec/research recommendations

**Cons**:
- Need to adapt workshop agents to `RequestHandler` interface
- Slight learning curve for SDK patterns

### Option B: Partial SDK Integration

Keep custom `A2AServer` class but use SDK types:
- Import types from `a2a.types`
- Keep custom request handling logic
- Keep custom `TaskManager`

**Pros**:
- Smaller change surface
- More control over implementation

**Cons**:
- Still maintaining custom code
- Miss SDK features (streaming, push notifications)
- Types may drift from SDK updates

### Decision: **Option A - Full SDK Integration**

**Rationale**: 
1. The spec explicitly recommends using the SDK
2. Code reduction from 713 lines to ~150 lines
3. SDK provides battle-tested features
4. Workshop should demonstrate best practices

---

## 5. Implementation Strategy

### Phase 1: Custom RequestHandler

Create a `WorkshopRequestHandler` that:
1. Wraps existing workshop agents (ChatAgent, ResearchAgent)
2. Converts A2A messages to agent prompts
3. Converts agent responses to A2A tasks/artifacts
4. Integrates OpenTelemetry tracing

### Phase 2: Simplified A2AServer

Create a thin wrapper around `A2AFastAPIApplication`:
1. Builds `AgentCard` from configuration
2. Creates `InMemoryTaskStore`
3. Instantiates `WorkshopRequestHandler`
4. Calls `A2AFastAPIApplication.build()`

### Phase 3: Backward Compatibility

Maintain exports in `src/agents/__init__.py`:
```python
# Re-export SDK types with aliases for backward compat
from a2a.types import AgentSkill as Skill
from a2a.types import AgentCapabilities as Capabilities
```

### Phase 4: Notebook Updates

Update `notebooks/03_a2a_protocol.ipynb` to:
1. Demonstrate SDK usage
2. Show `A2AAgent` client for calling external agents
3. Maintain existing learning objectives

---

## 6. References

- [A2A Protocol Specification](https://google.github.io/A2A)
- [a2a-sdk GitHub Repository](https://github.com/google/a2a)
- [Microsoft Agent Framework](https://github.com/microsoft/agent-framework)
- Original spec: [specs/001-agentic-patterns-workshop/research.md](../001-agentic-patterns-workshop/research.md)
