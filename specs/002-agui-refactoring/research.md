# Research: AG-UI Interface Refactoring# Research: AG-UI Interface Refactoring



































































































































































































































































































































































































- Existing research.md Section 3: AG-UI Protocol- [agent-framework-ag-ui Source](`.venv312/Lib/site-packages/agent_framework_ag_ui/`)- [AG-UI Protocol Specification](https://ag-ui-protocol.github.io/ag-ui)- [Microsoft Agent Framework GitHub](https://github.com/microsoft/agent-framework)## References---3. Document when to use each approach2. Add comparison section: custom vs SDK1. Update notebook exercises### Phase 3: Update Documentation3. Remove redundant code (~300 lines)2. Use `EventEncoder` instead of manual SSE formatting1. Replace custom event models with `ag-ui-core` imports### Phase 2: Simplify Custom Implementation4. Update notebook to show both approaches3. Keep existing custom implementation2. Add `AGUIClient` wrapper class1. Add `create_agui_endpoint()` helper function### Phase 1: Add SDK Integration (Non-Breaking)## 10. Migration Plan---### No New Dependencies Required| `httpx` | (installed) | HTTP client || `fastapi` | (installed) | HTTP server || `ag-ui-core` | (transitive) | Event types and encoder || `agent-framework-ag-ui` | 1.0.0b251223 | AG-UI integration || `agent-framework` | 1.0.0b251120 | Core Agent Framework ||---------|---------|---------|| Package | Version | Purpose |### Already Installed ✅## 9. Dependencies---- Test event emission with `EventEncoder`- Test `create_agui_endpoint` helper- Test `AGUIClient` wrapper### Unit Tests (new)- Validate streaming behavior- Test both custom and SDK-based endpoints- Verify notebook executes without errors### Integration Tests (`tests/integration/test_scenario_02.py`)- Validate request/response types- Test `EventEncoder` output format- Verify event schemas match `ag-ui-core` specification### Contract Tests (`tests/contract/test_agui_schemas.py`)## 8. Test Strategy---| `Tool` | `ag_ui.core.Tool` || `Message` | `ag_ui.core.Message` || `RunAgentInput` | `ag_ui.core.RunAgentInput` ||-------------|----------|| Custom Type | SDK Type |### Input/Output Mapping| `TextMessageContentEvent` | `ag_ui.core.TextMessageContentEvent` | Use SDK model || `RunStartedEvent` | `ag_ui.core.RunStartedEvent` | Use SDK model || `EventType.TEXT_MESSAGE_CONTENT` | `ag_ui.core.EventType.TEXT_MESSAGE_CONTENT` | Same name || `EventType.RUN_STARTED` | `ag_ui.core.EventType.RUN_STARTED` | Same name ||-------------|----------|-------|| Custom Type | SDK Type | Notes |### Event Type Mapping (Custom → SDK)## 7. Type Mapping---```    pass    """Wrapper around AGUIChatClient with simplified interface."""class AGUIClient:# Part 3: Client    add_agent_framework_fastapi_endpoint(app, agent, path)    from agent_framework.ag_ui import add_agent_framework_fastapi_endpoint    """Production wrapper using add_agent_framework_fastapi_endpoint.""") -> None:    path: str = "/"    agent: ChatAgent,    app: FastAPI,def create_agui_endpoint(# Part 2: Production - SDK-based implementation    pass    """Custom server showing how AG-UI works internally."""class AGUIServer:    pass    """Simplified event emitter for teaching protocol concepts."""class AGUIEventEmitter:# Part 1: Educational - Simplified custom implementation# src/agents/agui_server.py```python### Implementation Structure3. The value of using official SDKs (less code, more features)2. How to use production SDK (Agent Framework integration)1. How AG-UI protocol works (custom implementation)The workshop's purpose is **educational**. Students should understand:### Rationale## 6. Recommended Approach: Option C (Dual Implementation)---**Code Size**: ~300 lines (simplified custom + SDK wrapper)- Potential confusion- More code to maintain**Cons**:- Students learn both patterns- Backward compatible- Educational: shows custom vs SDK approaches**Pros**:**Approach**: Keep both implementations, demonstrate comparison### Option C: Dual Implementation**Code Reduction**: 551 lines → ~150 lines- Must manually handle streaming- More code than Option A**Cons**:- Allows custom event handling demonstrations- Maintains existing notebook structure- Shows protocol mechanics for educational purposes**Pros**:**Approach**: Use `ag-ui-core` types + custom FastAPI with `EventEncoder`### Option B: Hybrid Approach**Code Reduction**: 551 lines → ~20 lines- Less customization for workshop exercises- Less educational visibility into protocol internals**Cons**:- Maintained by Microsoft- Full SDK feature support- Minimal code (10-20 lines)**Pros**:**Approach**: Replace all custom code with `add_agent_framework_fastapi_endpoint()`### Option A: Full SDK Replacement## 5. Refactoring Decision Matrix---```                print(content.text, end="", flush=True)            if hasattr(content, "text"):        for content in update.contents:    if update.contents:async for update in client.get_streaming_response("Tell me a story"):```python#### Streaming```response2 = await agent.run("How are you?", thread=thread)response = await agent.run("Hello!", thread=thread)# ChatAgent automatically maintains historythread = await agent.get_new_thread()agent = ChatAgent(name="assistant", client=client)client = AGUIChatClient(endpoint="http://localhost:8000/")from agent_framework.ag_ui import AGUIChatClientfrom agent_framework import ChatAgent```python#### With ChatAgent (Client Manages History) - Recommended```    )        metadata={"thread_id": thread_id}        "How are you?",    response2 = await client.get_response(    # Second message - server retrieves history using thread_id        thread_id = response.additional_properties.get("thread_id")    response = await client.get_response("Hello!")    # First message - thread ID auto-generatedasync with AGUIChatClient(endpoint="http://localhost:8000/") as client:from agent_framework.ag_ui import AGUIChatClient```python#### Direct Usage (Server Manages History)### Usage Examples5. **Hybrid Tool Execution**: Client tools execute locally, server tools remotely4. **Event Conversion**: Converts AG-UI events to Agent Framework types3. **SSE Streaming**: Handles Server-Sent Events automatically2. **State Synchronization**: Extracts/sends state between client and server1. **Thread ID Management**: Auto-generates or uses provided `thread_id`### Key Features```    ) -> None:        **kwargs: Any,        additional_properties: dict[str, Any] | None = None,        timeout: float = 60.0,        http_client: httpx.AsyncClient | None = None,        endpoint: str,        *,        self,    def __init__(class AGUIChatClient(BaseChatClient):```python### SignatureClient for communicating with AG-UI compliant servers. Implements `BaseChatClient` interface.### Purpose## 4. `AGUIChatClient` API---```# Run: uvicorn app:app --reloadadd_agent_framework_fastapi_endpoint(app, agent, "/")app = FastAPI(title="AG-UI Server")# Create FastAPI app with AG-UI endpointagent = ChatAgent(name="assistant", chat_client=client, instructions="You are helpful."))    credential=DefaultAzureCredential(),    model="gpt-4",client = AzureOpenAIChatClient(# Create agentfrom azure.identity import DefaultAzureCredentialfrom fastapi import FastAPIfrom agent_framework.ag_ui import add_agent_framework_fastapi_endpointfrom agent_framework.azure import AzureOpenAIChatClientfrom agent_framework import ChatAgent```python### Usage Example4. **Returns `StreamingResponse`** with correct headers3. **Handles streaming** via `wrapped_agent.run_agent(input_data)`2. **Uses `EventEncoder`** for SSE formatting automatically1. **Auto-wraps AgentProtocol** with `AgentFrameworkAgent`### Implementation Details| `default_state` | `dict` | Initial state when client doesn't provide || `allow_origins` | `list[str]` | CORS origins (not yet implemented) || `predict_state_config` | `dict` | Predictive state updates config || `state_schema` | `Any \| None` | State schema for shared state (dict or Pydantic) || `path` | `str` | Endpoint path (default: "/") || `agent` | `AgentProtocol \| AgentFrameworkAgent` | Agent to expose (auto-wrapped if raw) || `app` | `FastAPI` | FastAPI application instance ||-----------|------|-------------|| Parameter | Type | Description |### Parameters```) -> None:    default_state: dict[str, Any] | None = None,    allow_origins: list[str] | None = None,    predict_state_config: dict[str, dict[str, str]] | None = None,    state_schema: Any | None = None,    path: str = "/",    agent: AgentProtocol | AgentFrameworkAgent,    app: FastAPI,def add_agent_framework_fastapi_endpoint(```python### Signature## 3. `add_agent_framework_fastapi_endpoint()` API---```from ag_ui.encoder import EventEncoder)    Tool,    Message,    RunAgentInput,    EventType,from ag_ui.core import (```python**Note**: Required by `agent-framework-ag-ui` for event types and encoding.### Package: `ag-ui-core````)    DocumentWriterConfirmationStrategy,    RecipeConfirmationStrategy,    TaskPlannerConfirmationStrategy,    DefaultConfirmationStrategy,    ConfirmationStrategy,    # Confirmation strategies        AGUIChatClient,                         # BaseChatClient for AG-UI servers    # Client-side        AGUIHttpService,                        # HTTP client for AG-UI servers    AGUIEventConverter,                     # Converts AG-UI events to Agent Framework types    AgentFrameworkAgent,                    # Wrapper for AgentProtocol    add_agent_framework_fastapi_endpoint,  # One-line FastAPI setup    # Server-sidefrom agent_framework.ag_ui import (```python#### Available Components**Installed**: ✅ Already available in `.venv312`### Package: `agent-framework-ag-ui` (v1.0.0b251223)## 2. Microsoft Agent Framework AG-UI Components---- No confirmation strategies for tool calls- No predictive state updates (`predict_state_config` parameter)- No state schema support (`state_schema` parameter)- No integration with `ChatAgent` streaming- No `AGUIChatClient` usage for client-side AG-UI communication### What's Missing in Current Implementation| Agent streaming | 100+ | Manual stream handling (SDK handles natively) || `AGUIServer` | 200+ | Manual FastAPI routes (SDK has `add_agent_framework_fastapi_endpoint`) || `AGUIEventEmitter` | 100+ | Custom SSE formatter (SDK has `EventEncoder`) || Pydantic event models | 100+ | Duplicates `ag_ui.core.*Event` models || `EventType` enum | 15 | Duplicates `ag_ui.core.EventType` ||-----------|-------|----------------------|| Component | Lines | Custom Implementation |The existing implementation is a **fully custom AG-UI server** that duplicates functionality already provided by the Microsoft Agent Framework:### Current State (`src/agents/agui_server.py` - 551 lines)## 1. Current Implementation Analysis---This document consolidates research findings for refactoring the AG-UI implementation to use Microsoft Agent Framework's native AG-UI support.**Date**: 2026-01-07 | **Plan**: [plan.md](./plan.md) | **Spec**: [spec.md](./spec.md)
**Date**: 2026-01-07 | **Plan**: [plan.md](./plan.md) | **Spec**: [spec.md](./spec.md)

This document consolidates research findings for the AG-UI refactoring to use Microsoft Agent Framework's native support.

---

## 1. Current Implementation Analysis

### Decision
The current `src/agents/agui_server.py` is a custom implementation that should be refactored to use official SDK components.

### Current Structure (551 lines)

```python
# Custom event types (lines 35-54)
class EventType(str, Enum):
    RUN_STARTED = "RUN_STARTED"
    RUN_FINISHED = "RUN_FINISHED"
    # ... 10 event types

# Custom Pydantic models (lines 56-197)
class BaseEvent(BaseModel): ...
class RunStartedEvent(BaseEvent): ...
class TextMessageContentEvent(BaseEvent): ...
# ... 12 event classes

# Custom emitter (lines 199-305)
@dataclass
class AGUIEventEmitter:
    def _format_sse(self, event: BaseEvent) -> str: ...
    def emit_text_content(self, delta: str) -> str: ...
    # ... 12 emit methods

# Custom server (lines 307-551)
class AGUIServer:
    def create_app(self) -> FastAPI: ...
    async def _stream_response(...) -> AsyncGenerator[str, None]: ...
    # ... manual route registration
```

### Issues with Current Implementation

1. **Duplicated Code**: Event types and models duplicate `ag-ui-core` package
2. **Manual SSE**: Custom `_format_sse()` reimplements `EventEncoder`
3. **No Native Agent Integration**: Manual streaming instead of `ChatAgent.run_stream()`
4. **Maintenance Burden**: 551 lines vs ~100 lines with SDK

---

## 2. Official SDK Components

### ag-ui-core Package

**Package Version**: `ag-ui-core` (installed)

**Key Exports**:
```python
from ag_ui.core import (
    EventType,           # Enum: RUN_STARTED, TEXT_MESSAGE_CONTENT, etc.
    RunAgentInput,       # Request Pydantic model
    Message,             # Message model
    Tool,                # Tool definition model
    # Events
    RunStartedEvent,
    RunFinishedEvent,
    TextMessageStartEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    ToolCallStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
)

from ag_ui.encoder import EventEncoder
```

**EventEncoder Usage**:
```python
encoder = EventEncoder()

# Encode any event to SSE format
sse_data = encoder.encode({
    "type": EventType.TEXT_MESSAGE_CONTENT,
    "message_id": "msg-123",
    "delta": "Hello"
})
# Returns: "event: message\ndata: {...}\n\n"

# Get content type for StreamingResponse
media_type = encoder.get_content_type()
# Returns: "text/event-stream"
```

---

## 3. agent-framework-ag-ui Package

### Decision
Use `agent_framework.ag_ui` module for high-level integration with ChatAgent.

### Package Version
`agent-framework-ag-ui==1.0.0b251223`

### Key Exports

```python
from agent_framework.ag_ui import (
    add_agent_framework_fastapi_endpoint,  # One-liner server setup
    AGUIChatClient,                        # Client for remote AG-UI servers
    AGUIEventConverter,                    # Event conversion utilities
    AGUIHttpService,                       # HTTP service wrapper
    AgentFrameworkAgent,                   # Agent wrapper for AG-UI
    ConfirmationStrategy,                  # Human-in-the-loop strategies
    DefaultConfirmationStrategy,
)
```

### add_agent_framework_fastapi_endpoint

**Signature**:
```python
def add_agent_framework_fastapi_endpoint(
    app: FastAPI,
    agent: AgentProtocol | AgentFrameworkAgent,
    path: str = "/",
    state_schema: Any | None = None,
    predict_state_config: dict[str, dict[str, str]] | None = None,
    allow_origins: list[str] | None = None,
    default_state: dict[str, Any] | None = None,
) -> None:
```

**Usage**:
```python
from agent_framework import ChatAgent
from agent_framework.openai import OpenAIChatClient
from agent_framework.ag_ui import add_agent_framework_fastapi_endpoint
from fastapi import FastAPI

# Create agent
agent = ChatAgent(
    name="assistant",
    instructions="You are helpful.",
    chat_client=OpenAIChatClient()
)

# Create app and add endpoint - that's it!
app = FastAPI()
add_agent_framework_fastapi_endpoint(app, agent, "/")
```

**What It Handles Automatically**:
- SSE streaming setup
- Event encoding
- Tool execution lifecycle
- State management
- Thread ID handling
- Error handling

---

## 4. AGUIChatClient (Client-Side)

### Decision
`AGUIChatClient` is for consuming AG-UI servers, not building them. Useful for testing.

### Usage Pattern

```python
from agent_framework.ag_ui import AGUIChatClient
from agent_framework import ChatAgent

# Connect to remote AG-UI server
client = AGUIChatClient(endpoint="http://localhost:8000/")

# Wrap in ChatAgent for conversation history
agent = ChatAgent(name="remote", chat_client=client)
thread = await agent.get_new_thread()

# Use like any other agent
response = await agent.run("Hello!", thread=thread)
```

### Key Features
- Automatic thread ID management
- State synchronization
- Server-Sent Events parsing
- Tool handling (hybrid client/server execution)

---

## 5. Migration Strategy

### Option A: Maximum Simplicity (3 lines)

**Pros**: Minimal code, fully maintained by Microsoft
**Cons**: Less educational value, hidden complexity

```python
from agent_framework.ag_ui import add_agent_framework_fastapi_endpoint
from fastapi import FastAPI

app = FastAPI()
add_agent_framework_fastapi_endpoint(app, agent, "/")
```

### Option B: Hybrid Approach (Educational)

**Pros**: Shows AG-UI concepts, maintains control
**Cons**: More code than Option A

```python
from ag_ui.core import EventType, RunAgentInput
from ag_ui.encoder import EventEncoder
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()
encoder = EventEncoder()

@app.post("/")
async def run_agent(input_data: RunAgentInput):
    async def event_generator():
        yield encoder.encode({"type": EventType.RUN_STARTED, ...})
        
        async for chunk in agent.run_stream(input_data.messages[-1].content):
            yield encoder.encode({
                "type": EventType.TEXT_MESSAGE_CONTENT,
                "delta": chunk
            })
        
        yield encoder.encode({"type": EventType.RUN_FINISHED, ...})
    
    return StreamingResponse(
        event_generator(),
        media_type=encoder.get_content_type()
    )
```

### Recommendation: Both Options

For the workshop, demonstrate **BOTH** approaches:
1. **Simple**: `add_agent_framework_fastapi_endpoint()` for production use
2. **Educational**: `ag-ui-core` with custom FastAPI for learning

---

## 6. Backward Compatibility

### Affected Components

| Component | Current Usage | Migration Path |
|-----------|---------------|----------------|
| `AGUIServer` | `notebooks/02_agui_interface.ipynb` | Keep as wrapper |
| `AGUIEventEmitter` | Same notebook, tests | Keep as wrapper |
| `EventType` | Contract tests | Re-export from `ag_ui.core` |
| `create_agui_server()` | Factory function | Keep, use SDK internally |

### Wrapper Strategy

```python
# src/agents/agui_server.py (refactored)

# Re-export from SDK for backward compatibility
from ag_ui.core import EventType
from ag_ui.encoder import EventEncoder

class AGUIEventEmitter:
    """Backward-compatible wrapper around EventEncoder."""
    
    def __init__(self, thread_id: str, run_id: str):
        self._encoder = EventEncoder()
        self._thread_id = thread_id
        self._run_id = run_id
        self._message_id: str | None = None
    
    def emit_text_content(self, delta: str) -> str:
        """Emit TEXT_MESSAGE_CONTENT event."""
        return self._encoder.encode({
            "type": EventType.TEXT_MESSAGE_CONTENT,
            "message_id": self._message_id,
            "delta": delta
        })
    # ... other emit methods
```

---

## 7. Test Impact Analysis

### Contract Tests (`tests/contract/test_agui_schemas.py`)

**Current**: Tests custom Pydantic models
**After**: Tests against `ag-ui-core` types

```python
# Before
from src.agents.agui_server import EventType, TextMessageContentEvent

# After
from ag_ui.core import EventType
from src.agents.agui_server import AGUIEventEmitter  # Wrapper
```

### Integration Tests (`tests/integration/test_scenario_02.py`)

**No changes needed** - tests notebook execution, not internals

### Unit Tests

**May need new tests** for:
- `AGUIEventEmitter` wrapper methods
- `AGUIServer` with SDK integration
- `create_agui_server()` factory

---

## 8. Type Mapping

| Custom Type | SDK Type | Notes |
|-------------|----------|-------|
| `EventType` | `ag_ui.core.EventType` | Direct replacement |
| `RunAgentInput` | `ag_ui.core.RunAgentInput` | Direct replacement |
| `Message` | `ag_ui.core.Message` | Direct replacement |
| `Tool` | `ag_ui.core.Tool` | Direct replacement |
| `BaseEvent` | N/A | Remove, use dict with encoder |
| `AGUIEventEmitter` | `EventEncoder` | Wrapper for backward compat |
| `AGUIServer` | `add_agent_framework_fastapi_endpoint` | Simplified |

---

## 9. Implementation Checklist

- [ ] Replace custom `EventType` enum with `ag_ui.core.EventType`
- [ ] Replace custom event models with `ag-ui-core` types
- [ ] Replace manual SSE formatting with `EventEncoder`
- [ ] Update `AGUIServer` to use SDK components
- [ ] Create backward-compatible `AGUIEventEmitter` wrapper
- [ ] Update notebook to show both approaches
- [ ] Update contract tests for SDK types
- [ ] Verify ≥80% test coverage
- [ ] Run full test suite

---

## References

- [AG-UI Protocol Documentation](https://docs.ag-ui.com)
- [agent-framework-ag-ui Package](https://pypi.org/project/agent-framework-ag-ui/)
- [ag-ui-core Package](https://pypi.org/project/ag-ui-core/)
- research.md Section 3: Original AG-UI research
- contracts/agui-events.md: Event schemas
