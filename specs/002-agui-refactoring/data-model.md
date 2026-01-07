# Data Model: AG-UI Interface Refactoring

**Date**: 2026-01-07 | **Plan**: [plan.md](./plan.md) | **Research**: [research.md](./research.md)

This document defines the data models for the AG-UI refactoring.

---

## 1. SDK Types (From `ag-ui-core`)

### EventType Enum

```python
from ag_ui.core import EventType

# Available values:
EventType.RUN_STARTED          # "RUN_STARTED"
EventType.RUN_FINISHED         # "RUN_FINISHED"
EventType.RUN_ERROR            # "RUN_ERROR"
EventType.TEXT_MESSAGE_START   # "TEXT_MESSAGE_START"
EventType.TEXT_MESSAGE_CONTENT # "TEXT_MESSAGE_CONTENT"
EventType.TEXT_MESSAGE_END     # "TEXT_MESSAGE_END"
EventType.TOOL_CALL_START      # "TOOL_CALL_START"
EventType.TOOL_CALL_ARGS       # "TOOL_CALL_ARGS"
EventType.TOOL_CALL_END        # "TOOL_CALL_END"
EventType.STATE_SNAPSHOT       # "STATE_SNAPSHOT"
EventType.STATE_DELTA          # "STATE_DELTA"
EventType.RAW                  # "RAW"
```

### RunAgentInput

```python
from ag_ui.core import RunAgentInput

# Pydantic model for request body
class RunAgentInput(BaseModel):
    thread_id: str
    run_id: str
    messages: list[Message]
    tools: list[Tool] | None = None
    context: dict[str, Any] | None = None
    state: dict[str, Any] | None = None
```

### Message

```python
from ag_ui.core import Message

class Message(BaseModel):
    role: Literal["user", "assistant", "tool", "system"]
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
```

---

## 2. Custom Types (Simplified for Workshop)

### AGUIEventEmitter (Backward Compatible)

```python
@dataclass
class AGUIEventEmitter:
    """Simplified event emitter wrapping ag-ui-core EventEncoder.
    
    Maintains backward compatibility with existing notebook code
    while using the official SDK internally.
    """
    thread_id: str
    run_id: str
    _encoder: EventEncoder = field(default_factory=EventEncoder, init=False)
    _message_id: str | None = field(default=None, init=False)
    _tool_call_id: str | None = field(default=None, init=False)
    
    def emit_run_started(self) -> str: ...
    def emit_run_finished(self) -> str: ...
    def emit_text_start(self, message_id: str | None = None) -> str: ...
    def emit_text_content(self, delta: str) -> str: ...
    def emit_text_end(self) -> str: ...
    def emit_tool_call_start(self, tool_name: str, ...) -> str: ...
    def emit_tool_call_args(self, delta: str) -> str: ...
    def emit_tool_call_end(self) -> str: ...
    def emit_state_snapshot(self, state: dict) -> str: ...
    def emit_state_delta(self, delta: dict) -> str: ...
    def emit_run_error(self, message: str, code: str | None = None) -> str: ...
```

### AGUIServer (Simplified)

```python
class AGUIServer:
    """AG-UI server supporting both custom and SDK modes.
    
    Attributes:
        agent: Agent instance for handling requests
        use_sdk: If True, uses add_agent_framework_fastapi_endpoint
        title: API documentation title
        description: API documentation description
    """
    agent: Any | None
    use_sdk: bool
    title: str
    description: str
    _app: FastAPI | None
    
    def create_app(self) -> FastAPI: ...
    def set_agent(self, agent: Any) -> None: ...
```

### AGUIClient (New)

```python
class AGUIClient:
    """Client for communicating with AG-UI servers.
    
    Wraps AGUIChatClient with simplified interface for workshop use.
    """
    endpoint: str
    _client: AGUIChatClient | None
    _timeout: float
    
    async def send(self, message: str) -> str: ...
    async def stream(self, message: str) -> AsyncGenerator[str, None]: ...
    async def close(self) -> None: ...
```

---

## 3. Event Flow

### Standard Event Sequence

```
Client Request (POST /)
    │
    ▼
RUN_STARTED { thread_id, run_id }
    │
    ▼
TEXT_MESSAGE_START { message_id, role: "assistant" }
    │
    ▼
TEXT_MESSAGE_CONTENT { message_id, delta: "..." } (repeated)
    │
    ▼
TEXT_MESSAGE_END { message_id }
    │
    ▼
RUN_FINISHED { thread_id, run_id }
```

### With Tool Calls

```
Client Request (POST /)
    │
    ▼
RUN_STARTED { thread_id, run_id }
    │
    ▼
TOOL_CALL_START { tool_call_id, tool_call_name }
    │
    ▼
TOOL_CALL_ARGS { tool_call_id, delta: "{...}" } (repeated)
    │
    ▼
TOOL_CALL_END { tool_call_id }
    │
    ▼
TEXT_MESSAGE_START { message_id, role: "assistant" }
    │
    ▼
TEXT_MESSAGE_CONTENT { message_id, delta: "..." } (repeated)
    │
    ▼
TEXT_MESSAGE_END { message_id }
    │
    ▼
RUN_FINISHED { thread_id, run_id }
```

---

## 4. SSE Format

### Event Encoding

```
event: message
data: {"type":"RUN_STARTED","thread_id":"thread-123","run_id":"run-456","timestamp":1704672000000}

event: message
data: {"type":"TEXT_MESSAGE_START","message_id":"msg-789","role":"assistant","timestamp":1704672000001}

event: message
data: {"type":"TEXT_MESSAGE_CONTENT","message_id":"msg-789","delta":"Hello","timestamp":1704672000002}

event: message
data: {"type":"TEXT_MESSAGE_END","message_id":"msg-789","timestamp":1704672000003}

event: message
data: {"type":"RUN_FINISHED","thread_id":"thread-123","run_id":"run-456","timestamp":1704672000004}
```

### Using EventEncoder

```python
from ag_ui.encoder import EventEncoder
from ag_ui.core import EventType

encoder = EventEncoder()

# Encode event to SSE format
sse = encoder.encode({
    "type": EventType.TEXT_MESSAGE_CONTENT,
    "message_id": "msg-123",
    "delta": "Hello"
})
# Result: "event: message\ndata: {...}\n\n"
```

---

## 5. HTTP Response Headers

```python
StreamingResponse(
    event_generator(),
    media_type="text/event-stream",
    headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",  # Disable nginx buffering
    }
)
```

---

## 6. Endpoint Contracts

### POST `/` (Main Endpoint)

**Request**:
```json
{
    "thread_id": "string",
    "run_id": "string",
    "messages": [
        {"role": "user", "content": "Hello"}
    ],
    "tools": [],
    "context": {},
    "state": {}
}
```

**Response**: SSE stream of events

### GET `/health`

**Response**:
```json
{
    "status": "healthy",
    "protocol": "AG-UI"
}
```

---

## 7. State Schema Support

### With `add_agent_framework_fastapi_endpoint`

```python
from pydantic import BaseModel

class ConversationState(BaseModel):
    topic: str = ""
    turns: int = 0
    context: dict[str, Any] = {}

add_agent_framework_fastapi_endpoint(
    app,
    agent,
    "/",
    state_schema=ConversationState,
    default_state={"topic": "", "turns": 0}
)
```

### State Events

```json
// STATE_SNAPSHOT
{"type": "STATE_SNAPSHOT", "state": {"topic": "AI", "turns": 3}}

// STATE_DELTA  
{"type": "STATE_DELTA", "delta": {"turns": 4}}
```

---

## 8. Error Handling

### RUN_ERROR Event

```json
{
    "type": "RUN_ERROR",
    "message": "Agent failed to process request",
    "code": "AGENT_ERROR",
    "timestamp": 1704672000000
}
```

### Custom Error Response

```python
def emit_run_error(self, message: str, code: str | None = None) -> str:
    return self._encoder.encode({
        "type": EventType.RUN_ERROR,
        "message": message,
        "code": code or "UNKNOWN_ERROR"
    })
```
