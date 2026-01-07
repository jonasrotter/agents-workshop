# Data Model: A2A Server Refactoring

**Date**: 2026-01-06 | **Plan**: [plan.md](./plan.md)

This document maps current custom types to SDK types and defines the data model for the refactored implementation.

---

## Type Mapping

### Agent Card Types

| Current (`a2a_server.py`) | SDK (`a2a.types`) | Migration Notes |
|---------------------------|-------------------|-----------------|
| `Skill` | `AgentSkill` | **Name change** - update imports |
| `AuthConfig` | `SecurityScheme` subclasses | More granular in SDK |
| `Capabilities` | `AgentCapabilities` | Same fields, different class |
| `Provider` | `AgentProvider` | Same structure |
| `AgentCard` | `AgentCard` | Field names match |

### Task Types

| Current | SDK | Migration Notes |
|---------|-----|-----------------|
| `TaskState` (enum) | `TaskState` | Values match: `submitted`, `working`, `completed`, `failed`, `cancelled`, `input-required` |
| `TaskStatus` | `TaskStatus` | Same structure |
| `Task` | `Task` | Field names match |
| `Artifact` | `Artifact` | Same structure |

### Message Types

| Current | SDK | Migration Notes |
|---------|-----|-----------------|
| `TextPart` | `TextPart` | Same structure |
| `FilePart` | `FilePart` / `FileWithBytes` / `FileWithUri` | SDK has more specific types |
| `DataPart` | `DataPart` | Same structure |
| `Message` | `Message` | Same structure |
| `MessagePart` (union) | `Part` | SDK uses `Part` as union type |

### JSON-RPC Types

| Current | SDK | Migration Notes |
|---------|-----|-----------------|
| `JSONRPCError` | `JSONRPCError` | Same structure |
| `JSONRPCRequest` | `JSONRPCRequest` | Same structure |
| `JSONRPCResponse` | `JSONRPCResponse` / `JSONRPCSuccessResponse` / `JSONRPCErrorResponse` | SDK has more specific types |
| `ErrorCode` (class) | N/A | Use SDK error classes instead |

### Request/Response Types

| Current | SDK | Migration Notes |
|---------|-----|-----------------|
| `MessageSendParams` | `MessageSendParams` | Same structure |
| `TaskGetParams` | `TaskIdParams` | **Name change** |
| `TaskListParams` | `TaskQueryParams` | **Name change** |
| `TaskCancelParams` | `TaskIdParams` | Use same params class |

### Server Types

| Current | SDK | Migration Notes |
|---------|-----|-----------------|
| `TaskManager` | `InMemoryTaskStore` | Different interface - adapt |
| `A2AServer` | `A2AFastAPIApplication` | Different pattern - wrap |

---

## SDK Type Details

### AgentCard

```python
from a2a.types import AgentCard, AgentCapabilities, AgentSkill, AgentProvider

agent_card = AgentCard(
    name="Research Agent",
    description="Researches topics and provides summaries",
    url="http://localhost:8000",
    version="1.0.0",
    capabilities=AgentCapabilities(
        streaming=True,
        pushNotifications=False,
        stateManagement=True,
    ),
    skills=[
        AgentSkill(
            id="research_topic",
            name="Research Topic",
            description="Research a topic and return findings",
            inputSchema={
                "type": "object",
                "properties": {"topic": {"type": "string"}},
                "required": ["topic"],
            },
        ),
    ],
    provider=AgentProvider(name="Workshop Demo"),
)
```

### Task

```python
from a2a.types import Task, TaskState, TaskStatus, Artifact, TextPart

task = Task(
    id="task-abc123",
    contextId="ctx-xyz789",
    status=TaskStatus(
        state=TaskState.completed,
        message=Message(
            role=Role.agent,
            parts=[TextPart(text="Research complete")],
        ),
    ),
    artifacts=[
        Artifact(
            artifactId="art-001",
            name="research_result",
            parts=[TextPart(text="Findings...")],
        ),
    ],
)
```

### Message

```python
from a2a.types import Message, Role, TextPart, FilePart, DataPart

message = Message(
    role=Role.user,
    parts=[
        TextPart(text="Research quantum computing"),
        DataPart(data={"depth": "detailed"}, mimeType="application/json"),
    ],
    messageId="msg-123",
)
```

---

## Backward Compatibility Aliases

To maintain backward compatibility with notebook imports:

```python
# src/agents/__init__.py

from a2a.types import (
    AgentCard,
    AgentSkill as Skill,  # Backward compat alias
    AgentCapabilities as Capabilities,  # Backward compat alias
    AgentProvider as Provider,  # Backward compat alias
    Task,
    TaskState,
    TaskStatus,
    Artifact,
    Message,
    TextPart,
    FilePart,
    DataPart,
)

# These imports will work unchanged:
# from src.agents import AgentCard, Skill, Task, TaskState, Message, TextPart
```

---

## New Interface: RequestHandler

The SDK requires implementing `RequestHandler` interface:

```python
from a2a.server.request_handlers import RequestHandler
from a2a.server.context import ServerCallContext
from a2a.types import (
    SendMessageRequest,
    GetTaskRequest,
    CancelTaskRequest,
    Task,
)
from typing import AsyncIterator

class WorkshopRequestHandler(RequestHandler):
    """Custom request handler for workshop agents."""
    
    def __init__(self, agent, task_store):
        self.agent = agent
        self.task_store = task_store
    
    async def on_message_send(
        self,
        request: SendMessageRequest,
        context: ServerCallContext,
    ) -> Task:
        """Handle incoming message, execute agent, return task."""
        ...
    
    async def on_message_send_stream(
        self,
        request: SendMessageRequest,
        context: ServerCallContext,
    ) -> AsyncIterator:
        """Handle streaming message requests."""
        ...
    
    async def on_get_task(
        self,
        request: GetTaskRequest,
        context: ServerCallContext,
    ) -> Task:
        """Get task by ID."""
        ...
    
    async def on_cancel_task(
        self,
        request: CancelTaskRequest,
        context: ServerCallContext,
    ) -> Task:
        """Cancel a task."""
        ...
```

---

## Task Store Interface

SDK provides `InMemoryTaskStore` with this interface:

```python
from a2a.server.task_store import TaskStore, InMemoryTaskStore

# Use SDK's in-memory implementation
task_store = InMemoryTaskStore()

# Or implement custom TaskStore for persistence:
class DatabaseTaskStore(TaskStore):
    async def create(self, task: Task) -> Task: ...
    async def get(self, task_id: str) -> Task | None: ...
    async def update(self, task: Task) -> Task: ...
    async def delete(self, task_id: str) -> bool: ...
    async def list(self, filter: TaskFilter | None = None) -> list[Task]: ...
```

---

## Entity Relationships

```
┌─────────────┐         ┌─────────────┐
│  AgentCard  │────────▶│  AgentSkill │ (1:N)
│             │         └─────────────┘
│             │────────▶│AgentProvider│ (1:1)
│             │         └─────────────┘
│             │────────▶│AgentCaps    │ (1:1)
└─────────────┘         └─────────────┘

┌─────────────┐         ┌─────────────┐
│    Task     │────────▶│ TaskStatus  │ (1:1)
│             │         └─────────────┘
│             │────────▶│  Artifact   │ (1:N)
│             │         └─────────────┘
│             │────────▶│  Message    │ (N - history)
└─────────────┘         └─────────────┘

┌─────────────┐         ┌─────────────┐
│   Message   │────────▶│    Part     │ (1:N)
│             │         │ (TextPart,  │
│             │         │  FilePart,  │
│             │         │  DataPart)  │
└─────────────┘         └─────────────┘
```
