"""A2A (Agent-to-Agent) Protocol Server implementation.

This module implements the A2A protocol for agent interoperability,
enabling agents to discover and invoke each other remotely.

A2A Protocol enables:
- Agent discovery via Agent Cards
- Remote task invocation via JSON-RPC
- Task status tracking and cancellation
- Streaming and push notifications

Reference: specs/001-agentic-patterns-workshop/contracts/a2a-protocol.md
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.common.config import get_settings
from src.common.exceptions import A2AError
from src.common.telemetry import get_tracer, record_exception

tracer = get_tracer(__name__)


# Task States


class TaskState(str, Enum):
    """A2A task states."""

    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AuthType(str, Enum):
    """Authentication types for A2A."""

    NONE = "none"
    BEARER = "bearer"
    API_KEY = "api_key"
    OAUTH2 = "oauth2"


# JSON-RPC Error Codes


class ErrorCode:
    """Standard JSON-RPC and A2A error codes."""

    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    TASK_NOT_FOUND = -32000
    CONTEXT_NOT_FOUND = -32001
    RATE_LIMITED = -32002
    UNAUTHORIZED = -32003


# Pydantic Models


class Skill(BaseModel):
    """Agent skill definition."""

    id: str
    name: str
    description: str
    inputSchema: dict[str, Any] | None = None
    outputSchema: dict[str, Any] | None = None


class AuthConfig(BaseModel):
    """Authentication configuration."""

    type: AuthType = AuthType.NONE
    scheme: str | None = None


class Capabilities(BaseModel):
    """Agent capabilities."""

    streaming: bool = False
    pushNotifications: bool = False
    stateManagement: bool = False


class Provider(BaseModel):
    """Agent provider information."""

    name: str
    url: str | None = None


class AgentCard(BaseModel):
    """A2A Agent Card for capability advertisement."""

    name: str
    description: str | None = None
    url: str
    version: str = "1.0.0"
    capabilities: Capabilities = Field(default_factory=Capabilities)
    skills: list[Skill]
    authentication: AuthConfig = Field(default_factory=AuthConfig)
    provider: Provider | None = None


# Message Parts


class TextPart(BaseModel):
    """Text message part."""

    kind: str = "text"
    text: str


class FilePart(BaseModel):
    """File message part."""

    kind: str = "file"
    mimeType: str
    data: str | None = None
    uri: str | None = None


class DataPart(BaseModel):
    """Structured data message part."""

    kind: str = "data"
    mimeType: str = "application/json"
    data: Any


MessagePart = TextPart | FilePart | DataPart


class Message(BaseModel):
    """A2A message structure."""

    role: str
    parts: list[MessagePart]
    messageId: str | None = None


# JSON-RPC Models


class JSONRPCError(BaseModel):
    """JSON-RPC error object."""

    code: int
    message: str
    data: Any | None = None


class JSONRPCRequest(BaseModel):
    """JSON-RPC request structure."""

    jsonrpc: str = "2.0"
    id: str | int
    method: str
    params: dict[str, Any] | None = None


class JSONRPCResponse(BaseModel):
    """JSON-RPC response structure."""

    jsonrpc: str = "2.0"
    id: str | int
    result: Any | None = None
    error: JSONRPCError | None = None


# Task Models


class TaskStatus(BaseModel):
    """Task status information."""

    state: TaskState
    message: Message | None = None
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class Artifact(BaseModel):
    """Task artifact (output)."""

    artifactId: str
    name: str
    parts: list[MessagePart]


class Task(BaseModel):
    """A2A task representation."""

    kind: str = "task"
    id: str
    contextId: str
    status: TaskStatus
    artifacts: list[Artifact] | None = None
    history: list[Message] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# Method Parameters


class MessageSendParams(BaseModel):
    """Parameters for message/send method."""

    message: Message
    contextId: str | None = None


class TaskGetParams(BaseModel):
    """Parameters for tasks/get method."""

    id: str
    historyLength: int | None = None


class TaskListParams(BaseModel):
    """Parameters for tasks/list method."""

    contextId: str | None = None
    status: TaskState | None = None
    pageSize: int = 10
    pageToken: str | None = None


class TaskCancelParams(BaseModel):
    """Parameters for tasks/cancel method."""

    id: str


@dataclass
class TaskManager:
    """Manages A2A tasks in memory.

    In production, this would be backed by a database.
    """

    tasks: dict[str, Task] = field(default_factory=dict)
    contexts: dict[str, list[str]] = field(default_factory=dict)

    def create_task(
        self, context_id: str | None = None, message: Message | None = None
    ) -> Task:
        """Create a new task."""
        task_id = f"task-{uuid.uuid4().hex[:12]}"
        context_id = context_id or f"ctx-{uuid.uuid4().hex[:12]}"

        task = Task(
            id=task_id,
            contextId=context_id,
            status=TaskStatus(
                state=TaskState.SUBMITTED,
                message=message,
            ),
            history=[message] if message else [],
        )

        self.tasks[task_id] = task

        if context_id not in self.contexts:
            self.contexts[context_id] = []
        self.contexts[context_id].append(task_id)

        return task

    def get_task(self, task_id: str) -> Task | None:
        """Get task by ID."""
        return self.tasks.get(task_id)

    def update_task_status(
        self,
        task_id: str,
        state: TaskState,
        message: Message | None = None,
        artifacts: list[Artifact] | None = None,
    ) -> Task | None:
        """Update task status."""
        task = self.tasks.get(task_id)
        if task:
            task.status = TaskStatus(state=state, message=message)
            if artifacts:
                task.artifacts = artifacts
        return task

    def list_tasks(
        self,
        context_id: str | None = None,
        status: TaskState | None = None,
        page_size: int = 10,
    ) -> list[Task]:
        """List tasks with optional filtering."""
        tasks = list(self.tasks.values())

        if context_id:
            task_ids = self.contexts.get(context_id, [])
            tasks = [t for t in tasks if t.id in task_ids]

        if status:
            tasks = [t for t in tasks if t.status.state == status]

        return tasks[:page_size]

    def cancel_task(self, task_id: str) -> Task | None:
        """Cancel a task."""
        return self.update_task_status(task_id, TaskState.CANCELLED)


class A2AServer:
    """A2A Protocol Server using FastAPI.

    This server implements the A2A protocol for agent interoperability.
    It provides endpoints for agent discovery and JSON-RPC task invocation.

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
        skills: list[Skill] | None = None,
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
        self._app: FastAPI | None = None
        self._task_manager = TaskManager()

    def _build_agent_card(self) -> AgentCard:
        """Build Agent Card from configuration."""
        return AgentCard(
            name=self.name,
            description=self.description,
            url=self.url,
            version=self.version,
            capabilities=Capabilities(
                streaming=True,
                pushNotifications=False,
                stateManagement=True,
            ),
            skills=self.skills,
            authentication=AuthConfig(type=AuthType.NONE),
            provider=Provider(name="Workshop Demo"),
        )

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI) -> AsyncGenerator[None, None]:
        """Application lifespan manager."""
        with tracer.start_as_current_span("a2a_server_startup"):
            yield

    def create_app(self) -> FastAPI:
        """Create FastAPI application with A2A endpoints.

        Returns:
            Configured FastAPI application
        """
        self._app = FastAPI(
            title=f"{self.name} - A2A Server",
            description=self.description,
            lifespan=self._lifespan,
        )

        # Configure CORS
        self._app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Register routes
        self._register_routes()

        return self._app

    def _register_routes(self) -> None:
        """Register API routes."""
        if not self._app:
            raise A2AError("App not created", details={})

        @self._app.get("/health")
        async def health() -> dict[str, str]:
            """Health check endpoint."""
            return {"status": "healthy", "protocol": "A2A", "agent": self.name}

        @self._app.get("/.well-known/agent-card.json")
        async def agent_card() -> dict[str, Any]:
            """Agent Card endpoint for capability discovery."""
            return self._build_agent_card().model_dump()

        @self._app.post("/")
        async def json_rpc(request: Request) -> JSONResponse:
            """JSON-RPC endpoint for A2A protocol."""
            try:
                body = await request.json()
                rpc_request = JSONRPCRequest.model_validate(body)
                response = await self._handle_rpc(rpc_request)
                return JSONResponse(content=response.model_dump())
            except Exception as e:
                record_exception(e)
                error_response = JSONRPCResponse(
                    id=body.get("id", 0) if isinstance(body, dict) else 0,
                    error=JSONRPCError(
                        code=ErrorCode.INTERNAL_ERROR,
                        message=str(e),
                    ),
                )
                return JSONResponse(content=error_response.model_dump())

    async def _handle_rpc(self, request: JSONRPCRequest) -> JSONRPCResponse:
        """Handle JSON-RPC request and route to appropriate method.

        Args:
            request: Parsed JSON-RPC request

        Returns:
            JSON-RPC response
        """
        with tracer.start_as_current_span("a2a_handle_rpc") as span:
            span.set_attribute("method", request.method)
            span.set_attribute("request_id", str(request.id))

            method_handlers = {
                "message/send": self._handle_message_send,
                "tasks/get": self._handle_tasks_get,
                "tasks/list": self._handle_tasks_list,
                "tasks/cancel": self._handle_tasks_cancel,
            }

            handler = method_handlers.get(request.method)
            if not handler:
                return JSONRPCResponse(
                    id=request.id,
                    error=JSONRPCError(
                        code=ErrorCode.METHOD_NOT_FOUND,
                        message=f"Method not found: {request.method}",
                    ),
                )

            try:
                result = await handler(request.params or {})
                return JSONRPCResponse(id=request.id, result=result)
            except A2AError as e:
                return JSONRPCResponse(
                    id=request.id,
                    error=JSONRPCError(
                        code=ErrorCode.INTERNAL_ERROR,
                        message=e.message,
                        data=e.details,
                    ),
                )

    async def _handle_message_send(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle message/send method.

        Args:
            params: Method parameters

        Returns:
            Task or message result
        """
        parsed = MessageSendParams.model_validate(params)

        # Create task for processing
        task = self._task_manager.create_task(
            context_id=parsed.contextId,
            message=parsed.message,
        )

        # Process with agent if available
        if self.agent:
            # Update to working state
            self._task_manager.update_task_status(
                task.id, TaskState.WORKING
            )

            # Extract text from message
            text = ""
            for part in parsed.message.parts:
                if isinstance(part, TextPart):
                    text = part.text
                    break

            try:
                # Run agent
                response = await self.agent.run(text)

                # Create artifact with response
                artifact = Artifact(
                    artifactId=f"art-{uuid.uuid4().hex[:8]}",
                    name="response",
                    parts=[TextPart(text=response)],
                )

                # Update to completed
                self._task_manager.update_task_status(
                    task.id,
                    TaskState.COMPLETED,
                    message=Message(
                        role="agent",
                        parts=[TextPart(text=response)],
                    ),
                    artifacts=[artifact],
                )

            except Exception as e:
                self._task_manager.update_task_status(
                    task.id,
                    TaskState.FAILED,
                    message=Message(
                        role="agent",
                        parts=[TextPart(text=f"Error: {e}")],
                    ),
                )

        # Get updated task
        task = self._task_manager.get_task(task.id)
        return task.model_dump() if task else {}

    async def _handle_tasks_get(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle tasks/get method.

        Args:
            params: Method parameters

        Returns:
            Task details
        """
        parsed = TaskGetParams.model_validate(params)
        task = self._task_manager.get_task(parsed.id)

        if not task:
            raise A2AError(
                f"Task not found: {parsed.id}",
                details={"task_id": parsed.id},
            )

        result = task.model_dump()

        # Limit history if requested
        if parsed.historyLength and task.history:
            result["history"] = result["history"][-parsed.historyLength :]

        return result

    async def _handle_tasks_list(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle tasks/list method.

        Args:
            params: Method parameters

        Returns:
            List of tasks
        """
        parsed = TaskListParams.model_validate(params)
        tasks = self._task_manager.list_tasks(
            context_id=parsed.contextId,
            status=parsed.status,
            page_size=parsed.pageSize,
        )

        return {
            "tasks": [t.model_dump() for t in tasks],
            "totalSize": len(tasks),
            "pageSize": parsed.pageSize,
            "nextPageToken": "",
        }

    async def _handle_tasks_cancel(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle tasks/cancel method.

        Args:
            params: Method parameters

        Returns:
            Cancelled task
        """
        parsed = TaskCancelParams.model_validate(params)
        task = self._task_manager.cancel_task(parsed.id)

        if not task:
            raise A2AError(
                f"Task not found: {parsed.id}",
                details={"task_id": parsed.id},
            )

        return task.model_dump()

    def add_skill(self, skill: Skill) -> None:
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


def create_a2a_server(
    agent: Any | None = None,
    name: str = "A2A Agent",
    description: str = "A2A Protocol Agent",
    url: str = "http://localhost:8000",
    skills: list[Skill] | None = None,
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
            Skill(
                id="echo",
                name="Echo",
                description="Echo back the input message",
                inputSchema={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                },
            ),
        ],
    )


app = create_default_app()
