# A2A SDK Interface Contracts

**Date**: 2026-01-06 | **Plan**: [../plan.md](../plan.md)

This document defines the interface contracts for integrating with the `a2a-sdk` package.

---

## RequestHandler Interface

The core interface that must be implemented to handle A2A requests.

### Python Interface

```python
from abc import ABC, abstractmethod
from typing import AsyncIterator
from a2a.server.context import ServerCallContext
from a2a.types import (
    SendMessageRequest,
    SendStreamingMessageRequest,
    GetTaskRequest,
    CancelTaskRequest,
    Task,
    TaskResubscriptionRequest,
    SetTaskPushNotificationConfigRequest,
    GetTaskPushNotificationConfigRequest,
    DeleteTaskPushNotificationConfigRequest,
    ListTaskPushNotificationConfigRequest,
    TaskPushNotificationConfig,
)

class RequestHandler(ABC):
    """A2A request handler interface."""
    
    @abstractmethod
    async def on_message_send(
        self,
        request: SendMessageRequest,
        context: ServerCallContext,
    ) -> Task:
        """Handle synchronous message send.
        
        Args:
            request: The message send request containing:
                - params.message: The incoming message
                - params.context_id: Optional context ID
                - params.configuration: Optional message config
            context: Server call context with:
                - task_id: Generated task ID
                - context_id: Context ID for the conversation
                
        Returns:
            Task with status and optionally artifacts
        """
        ...
    
    async def on_message_send_stream(
        self,
        request: SendStreamingMessageRequest,
        context: ServerCallContext,
    ) -> AsyncIterator:
        """Handle streaming message send.
        
        Yields events as the task progresses:
        - TaskStatusUpdateEvent: When task state changes
        - TaskArtifactUpdateEvent: When artifacts are created
        
        Default implementation calls on_message_send and yields final task.
        """
        ...
    
    @abstractmethod
    async def on_get_task(
        self,
        request: GetTaskRequest,
        context: ServerCallContext,
    ) -> Task:
        """Get task by ID.
        
        Args:
            request: Contains params.id (task ID) and optional params.history_length
            context: Server call context
            
        Returns:
            Task if found
            
        Raises:
            TaskNotFoundError: If task ID doesn't exist
        """
        ...
    
    @abstractmethod
    async def on_cancel_task(
        self,
        request: CancelTaskRequest,
        context: ServerCallContext,
    ) -> Task:
        """Cancel a task.
        
        Args:
            request: Contains params.id (task ID to cancel)
            context: Server call context
            
        Returns:
            Task with cancelled status
            
        Raises:
            TaskNotFoundError: If task ID doesn't exist
            TaskNotCancelableError: If task cannot be cancelled
        """
        ...
    
    # Optional: Push notification handlers
    async def on_set_task_push_notification_config(
        self, request, context
    ) -> TaskPushNotificationConfig:
        """Configure push notifications for a task."""
        raise NotImplementedError("Push notifications not supported")
    
    async def on_get_task_push_notification_config(
        self, request, context
    ) -> TaskPushNotificationConfig:
        """Get push notification config for a task."""
        raise NotImplementedError("Push notifications not supported")
    
    async def on_delete_task_push_notification_config(
        self, request, context
    ) -> None:
        """Delete push notification config for a task."""
        raise NotImplementedError("Push notifications not supported")
    
    async def on_list_task_push_notification_config(
        self, request, context
    ) -> list[TaskPushNotificationConfig]:
        """List push notification configs."""
        raise NotImplementedError("Push notifications not supported")
    
    async def on_resubscribe_to_task(
        self, request, context
    ) -> AsyncIterator:
        """Resubscribe to task events."""
        raise NotImplementedError("Task resubscription not supported")
```

---

## TaskStore Interface

Interface for task persistence.

```python
from abc import ABC, abstractmethod
from a2a.types import Task

class TaskFilter:
    context_id: str | None = None
    state: str | None = None
    limit: int = 100
    offset: int = 0

class TaskStore(ABC):
    """Task storage interface."""
    
    @abstractmethod
    async def create(self, task: Task) -> Task:
        """Create a new task.
        
        Args:
            task: Task to create (ID may be pre-generated)
            
        Returns:
            Created task with ID
        """
        ...
    
    @abstractmethod
    async def get(self, task_id: str) -> Task | None:
        """Get task by ID.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Task if found, None otherwise
        """
        ...
    
    @abstractmethod
    async def update(self, task: Task) -> Task:
        """Update an existing task.
        
        Args:
            task: Task with updates (ID must exist)
            
        Returns:
            Updated task
            
        Raises:
            TaskNotFoundError: If task doesn't exist
        """
        ...
    
    @abstractmethod
    async def delete(self, task_id: str) -> bool:
        """Delete a task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            True if deleted, False if not found
        """
        ...
    
    @abstractmethod
    async def list(self, filter: TaskFilter | None = None) -> list[Task]:
        """List tasks with optional filtering.
        
        Args:
            filter: Optional filter criteria
            
        Returns:
            List of matching tasks
        """
        ...
```

---

## WorkshopRequestHandler Contract

Our custom implementation wrapping workshop agents.

```python
from typing import Any
from a2a.server.request_handlers import RequestHandler
from a2a.server.task_store import InMemoryTaskStore
from a2a.server.context import ServerCallContext
from a2a.types import (
    SendMessageRequest,
    GetTaskRequest,
    CancelTaskRequest,
    Task,
    TaskState,
    TaskStatus,
    Message,
    TextPart,
    Artifact,
    TaskNotFoundError,
)

class WorkshopRequestHandler(RequestHandler):
    """Request handler wrapping workshop agents.
    
    Attributes:
        agent: Workshop agent with async `run(prompt: str) -> str` method
        task_store: InMemoryTaskStore for task persistence
    """
    
    def __init__(
        self,
        agent: Any,
        task_store: InMemoryTaskStore,
    ):
        """Initialize handler.
        
        Args:
            agent: Agent with `run(str) -> str` method
            task_store: Task storage implementation
        """
        self.agent = agent
        self.task_store = task_store
    
    async def on_message_send(
        self,
        request: SendMessageRequest,
        context: ServerCallContext,
    ) -> Task:
        """Process message through workshop agent.
        
        Flow:
        1. Create task in SUBMITTED state
        2. Extract text from message parts
        3. Update task to WORKING state
        4. Run agent with extracted text
        5. Create artifact with response
        6. Update task to COMPLETED (or FAILED)
        7. Return final task
        """
        ...
    
    async def on_get_task(
        self,
        request: GetTaskRequest,
        context: ServerCallContext,
    ) -> Task:
        """Retrieve task from store."""
        ...
    
    async def on_cancel_task(
        self,
        request: CancelTaskRequest,
        context: ServerCallContext,
    ) -> Task:
        """Cancel task if in cancellable state."""
        ...
    
    def _extract_text(self, message: Message) -> str:
        """Extract text content from message parts.
        
        Iterates through parts and returns first TextPart's text.
        Returns empty string if no TextPart found.
        """
        ...
```

---

## A2AServer Contract

Simplified server wrapper for workshop.

```python
from typing import Any
from fastapi import FastAPI
from a2a.types import AgentCard, AgentSkill, AgentCapabilities

class A2AServer:
    """Simplified A2A server for workshop.
    
    Wraps A2AFastAPIApplication with workshop-specific defaults.
    
    Attributes:
        agent: Workshop agent to handle requests
        name: Agent name for Agent Card
        description: Agent description
        url: Base URL for Agent Card
        version: Semantic version string
        skills: List of agent skills
        task_store: InMemoryTaskStore instance
    """
    
    def __init__(
        self,
        agent: Any = None,
        name: str = "A2A Agent",
        description: str = "A2A Protocol Agent",
        url: str = "http://localhost:8000",
        version: str = "1.0.0",
        skills: list[AgentSkill] | None = None,
    ):
        """Initialize A2A server.
        
        Args:
            agent: Optional agent to handle requests
            name: Agent name (displayed in Agent Card)
            description: Agent description
            url: Base URL (used for Agent Card)
            version: Semantic version
            skills: Skills to advertise
        """
        ...
    
    @property
    def agent_card(self) -> AgentCard:
        """Get agent card for this server.
        
        Returns:
            AgentCard with configured name, skills, capabilities
        """
        ...
    
    def create_app(self) -> FastAPI:
        """Create FastAPI application.
        
        Creates:
        1. WorkshopRequestHandler with agent and task_store
        2. A2AFastAPIApplication with agent_card and handler
        3. Calls build() to get FastAPI app
        
        Returns:
            Configured FastAPI application
        """
        ...


def create_a2a_server(
    agent: Any = None,
    name: str = "A2A Agent",
    description: str = "A2A Protocol Agent",
    url: str = "http://localhost:8000",
    skills: list[AgentSkill] | None = None,
) -> FastAPI:
    """Create A2A FastAPI server (convenience function).
    
    Args:
        agent: Optional agent to handle requests
        name: Agent name
        description: Agent description
        url: Base URL
        skills: Skills to advertise
        
    Returns:
        FastAPI application ready to run with uvicorn
    """
    ...
```

---

## HTTP Endpoints (SDK-provided)

The `A2AFastAPIApplication` provides these endpoints:

### Agent Card Discovery
```
GET /.well-known/agent-card.json

Response: AgentCard JSON
```

### JSON-RPC Endpoint
```
POST /

Content-Type: application/json
Accept: application/json (or text/event-stream for streaming)

Request Body: JSONRPCRequest
Response: JSONRPCResponse or SSE stream
```

### Supported Methods

| Method | Description | Params Type | Result Type |
|--------|-------------|-------------|-------------|
| `message/send` | Send message to agent | `MessageSendParams` | `Task` |
| `message/stream` | Stream message to agent | `MessageSendParams` | SSE stream |
| `tasks/get` | Get task by ID | `TaskIdParams` | `Task` |
| `tasks/cancel` | Cancel task | `TaskIdParams` | `Task` |
| `tasks/pushNotificationConfig/set` | Set push config | `PushNotificationConfig` | `PushNotificationConfig` |
| `tasks/pushNotificationConfig/get` | Get push config | `TaskIdParams` | `PushNotificationConfig` |
| `tasks/pushNotificationConfig/delete` | Delete push config | `TaskIdParams` | `void` |
| `tasks/resubscribe` | Resubscribe to task | `TaskIdParams` | SSE stream |

---

## Error Codes

Standard A2A/JSON-RPC error codes:

| Code | Constant | Description |
|------|----------|-------------|
| -32700 | `PARSE_ERROR` | Invalid JSON |
| -32600 | `INVALID_REQUEST` | Invalid request |
| -32601 | `METHOD_NOT_FOUND` | Method not found |
| -32602 | `INVALID_PARAMS` | Invalid params |
| -32603 | `INTERNAL_ERROR` | Internal error |
| -32000 | `TASK_NOT_FOUND` | Task not found |
| -32001 | `TASK_NOT_CANCELABLE` | Task cannot be cancelled |
| -32002 | `PUSH_NOTIFICATION_NOT_SUPPORTED` | Push not supported |
