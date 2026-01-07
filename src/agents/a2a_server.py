"""A2A (Agent-to-Agent) Protocol Server implementation using official SDK.

This module implements the A2A protocol for agent interoperability using the
official a2a-sdk package, enabling agents to discover and invoke each other remotely.

A2A Protocol enables:
- Agent discovery via Agent Cards
- Remote task invocation via JSON-RPC
- Task status tracking and cancellation
- Streaming support

Reference: specs/001-agentic-patterns-workshop/contracts/a2a-protocol.md
SDK: https://github.com/google/a2a-python
"""

from __future__ import annotations

import uuid
from typing import Any, AsyncIterator

from fastapi import FastAPI

# SDK imports
from a2a.server.apps import A2AFastAPIApplication
from a2a.server.context import ServerCallContext
from a2a.server.request_handlers import RequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentProvider,
    AgentSkill,
    Artifact,
    DeleteTaskPushNotificationConfigParams,
    GetTaskPushNotificationConfigParams,
    ListTaskPushNotificationConfigParams,
    Message,
    MessageSendParams,
    Part,
    Task,
    TaskArtifactUpdateEvent,
    TaskIdParams,
    TaskPushNotificationConfig,
    TaskQueryParams,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)

from src.common.exceptions import A2AError
from src.common.telemetry import get_tracer, record_exception

tracer = get_tracer(__name__)


# Backward compatibility aliases
Skill = AgentSkill  # Alias for backward compatibility


class WorkshopRequestHandler(RequestHandler):
    """Request handler that wraps workshop agents.
    
    This handler implements the A2A RequestHandler interface to process
    messages through our workshop agents, managing task state via the SDK's
    InMemoryTaskStore.
    
    Attributes:
        agent: Workshop agent with async `run(prompt: str) -> str` method
        task_store: InMemoryTaskStore for task persistence
    """

    def __init__(
        self,
        agent: Any | None = None,
        task_store: InMemoryTaskStore | None = None,
    ) -> None:
        """Initialize handler.
        
        Args:
            agent: Agent with `run(str) -> str` method
            task_store: Task storage implementation (defaults to new store)
        """
        self.agent = agent
        self.task_store = task_store or InMemoryTaskStore()

    async def on_message_send(
        self,
        params: MessageSendParams,
        context: ServerCallContext | None = None,
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
        
        Args:
            params: Message parameters containing the message to process
            context: Server call context (optional)
            
        Returns:
            Completed task with artifacts
        """
        with tracer.start_as_current_span("a2a_message_send") as span:
            # Generate IDs
            task_id = f"task-{uuid.uuid4().hex[:12]}"
            context_id = f"ctx-{uuid.uuid4().hex[:12]}"
            
            span.set_attribute("task_id", task_id)
            
            # Create task in SUBMITTED state
            task = Task(
                id=task_id,
                context_id=context_id,
                status=TaskStatus(state=TaskState.submitted),
                history=[params.message] if params.message else [],
            )
            await self.task_store.save(task)
            
            # Extract text from message
            text = self._extract_text(params.message)
            span.set_attribute("input_length", len(text))
            
            # Process with agent if available
            if self.agent:
                # Update to WORKING state
                task.status = TaskStatus(state=TaskState.working)
                await self.task_store.save(task)
                
                try:
                    # Run agent
                    response = await self.agent.run(text)
                    
                    # Create artifact with response
                    artifact = Artifact(
                        artifact_id=f"art-{uuid.uuid4().hex[:8]}",
                        name="response",
                        parts=[TextPart(text=response)],
                    )
                    
                    # Update to COMPLETED
                    task.status = TaskStatus(state=TaskState.completed)
                    task.artifacts = [artifact]
                    task.history = (task.history or []) + [
                        Message(
                            role="agent",
                            parts=[TextPart(text=response)],
                        )
                    ]
                    
                except Exception as e:
                    record_exception(e)
                    # Update to FAILED
                    task.status = TaskStatus(
                        state=TaskState.failed,
                        message=Message(
                            role="agent",
                            parts=[TextPart(text=f"Error: {e}")],
                        ),
                    )
            else:
                # No agent - mark as completed with echo
                task.status = TaskStatus(state=TaskState.completed)
                task.artifacts = [
                    Artifact(
                        artifact_id=f"art-{uuid.uuid4().hex[:8]}",
                        name="echo",
                        parts=[TextPart(text=f"Echo: {text}")],
                    )
                ]
            
            await self.task_store.save(task)
            return task

    async def on_message_send_stream(
        self,
        params: MessageSendParams,
        context: ServerCallContext | None = None,
    ) -> AsyncIterator[Task | Message]:
        """Stream message processing (delegates to non-streaming).
        
        Args:
            params: Message parameters
            context: Server call context
            
        Yields:
            Final task result
        """
        task = await self.on_message_send(params, context)
        yield task

    async def on_get_task(
        self,
        params: TaskQueryParams,
        context: ServerCallContext | None = None,
    ) -> Task:
        """Get task by ID.
        
        Args:
            params: Task query parameters with ID
            context: Server call context
            
        Returns:
            Task if found
            
        Raises:
            A2AError: If task not found
        """
        task = await self.task_store.get(params.id)
        if not task:
            raise A2AError(
                f"Task not found: {params.id}",
                details={"task_id": params.id},
            )
        
        # Limit history if requested
        if params.history_length and task.history:
            task.history = task.history[-params.history_length:]
        
        return task

    async def on_cancel_task(
        self,
        params: TaskIdParams,
        context: ServerCallContext | None = None,
    ) -> Task:
        """Cancel a task.
        
        Args:
            params: Task ID parameters
            context: Server call context
            
        Returns:
            Canceled task
            
        Raises:
            A2AError: If task not found
        """
        task = await self.task_store.get(params.id)
        if not task:
            raise A2AError(
                f"Task not found: {params.id}",
                details={"task_id": params.id},
            )
        
        task.status = TaskStatus(state=TaskState.canceled)
        await self.task_store.save(task)
        return task

    def _extract_text(self, message: Message | None) -> str:
        """Extract text content from message parts.
        
        Iterates through parts and returns first TextPart's text.
        Returns empty string if no TextPart found.
        
        Args:
            message: Message with parts
            
        Returns:
            Extracted text or empty string
        """
        if not message or not message.parts:
            return ""
        
        for part in message.parts:
            if isinstance(part, TextPart):
                return part.text
            # Handle dict-style parts from JSON
            if isinstance(part, dict) and part.get("kind") == "text":
                return part.get("text", "")
            # Handle Part union types
            if hasattr(part, "root") and hasattr(part.root, "text"):
                return part.root.text
        
        return ""

    # Push notification methods (not supported in workshop)
    
    async def on_set_task_push_notification_config(
        self,
        params: TaskPushNotificationConfig,
        context: ServerCallContext | None = None,
    ) -> TaskPushNotificationConfig:
        """Set push notification config (not supported).
        
        Args:
            params: Push notification config
            context: Server call context
            
        Raises:
            A2AError: Push notifications not supported
        """
        raise A2AError(
            "Push notifications not supported",
            details={"feature": "push_notifications"},
        )

    async def on_get_task_push_notification_config(
        self,
        params: TaskIdParams | GetTaskPushNotificationConfigParams,
        context: ServerCallContext | None = None,
    ) -> TaskPushNotificationConfig:
        """Get push notification config (not supported).
        
        Args:
            params: Task ID params
            context: Server call context
            
        Raises:
            A2AError: Push notifications not supported
        """
        raise A2AError(
            "Push notifications not supported",
            details={"feature": "push_notifications"},
        )

    async def on_delete_task_push_notification_config(
        self,
        params: DeleteTaskPushNotificationConfigParams,
        context: ServerCallContext | None = None,
    ) -> None:
        """Delete push notification config (not supported).
        
        Args:
            params: Delete params
            context: Server call context
            
        Raises:
            A2AError: Push notifications not supported
        """
        raise A2AError(
            "Push notifications not supported",
            details={"feature": "push_notifications"},
        )

    async def on_list_task_push_notification_config(
        self,
        params: ListTaskPushNotificationConfigParams,
        context: ServerCallContext | None = None,
    ) -> list[TaskPushNotificationConfig]:
        """List push notification configs (not supported).
        
        Args:
            params: List params
            context: Server call context
            
        Raises:
            A2AError: Push notifications not supported
        """
        raise A2AError(
            "Push notifications not supported",
            details={"feature": "push_notifications"},
        )

    async def on_resubscribe_to_task(
        self,
        params: TaskIdParams,
        context: ServerCallContext | None = None,
    ) -> AsyncIterator[Message | Task | TaskStatusUpdateEvent | TaskArtifactUpdateEvent]:
        """Resubscribe to task events (not supported).
        
        Args:
            params: Task ID params
            context: Server call context
            
        Raises:
            A2AError: Task resubscription not supported
        """
        raise A2AError(
            "Task resubscription not supported",
            details={"feature": "resubscription"},
        )
        # This is an async generator, so we need a yield to make it valid
        # But we raise before getting here
        yield  # type: ignore[misc]


class A2AServer:
    """A2A Protocol Server using official SDK.

    This server wraps the A2AFastAPIApplication from the a2a-sdk,
    providing a simplified interface for the workshop.

    Example:
        server = A2AServer(agent=my_agent, name="Research Agent")
        app = server.create_app()
        # Run with uvicorn: uvicorn app:app --reload
    """

    def __init__(
        self,
        agent: Any | None = None,
        name: str = "A2A Agent",
        description: str = "A2A Protocol Agent",
        url: str = "http://localhost:8000",
        version: str = "1.0.0",
        skills: list[AgentSkill] | None = None,
    ) -> None:
        """Initialize A2A server.

        Args:
            agent: Agent instance to handle requests
            name: Agent name for Agent Card
            description: Agent description for Agent Card
            url: Base URL for Agent Card
            version: Agent version
            skills: List of agent skills
        """
        self.agent = agent
        self.name = name
        self.description = description
        self.url = url
        self.version = version
        self.skills = skills or []
        self._task_store = InMemoryTaskStore()
        self._handler: WorkshopRequestHandler | None = None
        self._app: FastAPI | None = None

    @property
    def agent_card(self) -> AgentCard:
        """Get the Agent Card for this server."""
        return AgentCard(
            name=self.name,
            description=self.description,
            url=self.url,
            version=self.version,
            default_input_modes=["text"],
            default_output_modes=["text"],
            capabilities=AgentCapabilities(
                streaming=True,
                push_notifications=False,
                state_transition_history=True,
            ),
            skills=self.skills,
            provider=AgentProvider(
                organization="Workshop Demo",
                url="https://github.com/workshop",
            ),
        )

    def create_app(self) -> FastAPI:
        """Create FastAPI application with A2A endpoints.

        Returns:
            Configured FastAPI application
        """
        # Create handler with agent and task store
        self._handler = WorkshopRequestHandler(
            agent=self.agent,
            task_store=self._task_store,
        )
        
        # Create A2A application using SDK
        a2a_app = A2AFastAPIApplication(
            agent_card=self.agent_card,
            http_handler=self._handler,
        )
        
        # Build and configure FastAPI app
        self._app = a2a_app.build()
        
        # Add health endpoint (not provided by SDK)
        @self._app.get("/health")
        async def health() -> dict[str, str]:
            """Health check endpoint."""
            return {"status": "healthy", "protocol": "A2A", "agent": self.name}
        
        return self._app

    def add_skill(self, skill: AgentSkill) -> None:
        """Add a skill to the agent.

        Args:
            skill: Skill definition to add
        """
        self.skills.append(skill)

    def set_agent(self, agent: Any) -> None:
        """Set or replace the agent instance.

        Args:
            agent: Agent instance to handle requests
        """
        self.agent = agent
        if self._handler:
            self._handler.agent = agent


def create_a2a_server(
    agent: Any | None = None,
    name: str = "A2A Agent",
    description: str = "A2A Protocol Agent",
    url: str = "http://localhost:8000",
    skills: list[AgentSkill] | None = None,
) -> FastAPI:
    """Create A2A FastAPI server.

    Args:
        agent: Optional agent instance to handle requests
        name: Agent name for Agent Card
        description: Agent description
        url: Base URL for Agent Card
        skills: List of agent skills

    Returns:
        Configured FastAPI application
    """
    server = A2AServer(
        agent=agent,
        name=name,
        description=description,
        url=url,
        skills=skills or [],
    )
    return server.create_app()


# For running directly with uvicorn
def create_default_app() -> FastAPI:
    """Create default A2A app for development.

    Returns:
        FastAPI application with demo agent card
    """
    return create_a2a_server(
        name="Demo A2A Agent",
        description="Demonstration A2A agent for workshop",
        skills=[
            AgentSkill(
                id="echo",
                name="Echo",
                description="Echo back the input message",
                tags=["demo", "echo"],
            ),
        ],
    )


app = create_default_app()
