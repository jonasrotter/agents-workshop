"""AG-UI Protocol Server for streaming chat interfaces.

This module implements the AG-UI (Agent-User Interface) protocol for building
streaming chat interfaces that communicate with agents.

AG-UI Protocol enables:
- Token-by-token streaming responses
- Tool execution status updates
- State management for conversations
- Rich event types for UI rendering

Reference: specs/001-agentic-patterns-workshop/contracts/agui-events.md
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.common.config import get_settings
from src.common.exceptions import AGUIError
from src.common.telemetry import get_tracer, record_exception

tracer = get_tracer(__name__)


class EventType(str, Enum):
    """AG-UI event types for streaming responses."""

    RUN_STARTED = "RUN_STARTED"
    RUN_FINISHED = "RUN_FINISHED"
    RUN_ERROR = "RUN_ERROR"
    TEXT_MESSAGE_START = "TEXT_MESSAGE_START"
    TEXT_MESSAGE_CONTENT = "TEXT_MESSAGE_CONTENT"
    TEXT_MESSAGE_END = "TEXT_MESSAGE_END"
    TOOL_CALL_START = "TOOL_CALL_START"
    TOOL_CALL_ARGS = "TOOL_CALL_ARGS"
    TOOL_CALL_END = "TOOL_CALL_END"
    STATE_SNAPSHOT = "STATE_SNAPSHOT"
    STATE_DELTA = "STATE_DELTA"
    RAW = "RAW"


# Request Models


class ToolCall(BaseModel):
    """Tool call within a message."""

    id: str
    type: str = "function"
    function: dict[str, Any]


class Message(BaseModel):
    """Message in conversation history."""

    role: str = Field(..., pattern="^(user|assistant|tool|system)$")
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


class Tool(BaseModel):
    """Tool definition for AG-UI."""

    name: str
    description: str
    parameters: dict[str, Any]


class RunAgentInput(BaseModel):
    """Request body for the agentic chat endpoint."""

    thread_id: str
    run_id: str
    messages: list[Message]
    tools: list[Tool] | None = None
    context: dict[str, Any] | None = None


# Event Models


class BaseEvent(BaseModel):
    """Base event schema for AG-UI protocol."""

    type: EventType
    timestamp: int = Field(default_factory=lambda: int(time.time() * 1000))
    raw_event: Any | None = None


class RunStartedEvent(BaseEvent):
    """Emitted when a run begins."""

    type: EventType = EventType.RUN_STARTED
    thread_id: str
    run_id: str


class RunFinishedEvent(BaseEvent):
    """Emitted when a run completes successfully."""

    type: EventType = EventType.RUN_FINISHED
    thread_id: str
    run_id: str


class RunErrorEvent(BaseEvent):
    """Emitted when a run fails."""

    type: EventType = EventType.RUN_ERROR
    message: str
    code: str | None = None


class TextMessageStartEvent(BaseEvent):
    """Emitted when an assistant message begins."""

    type: EventType = EventType.TEXT_MESSAGE_START
    message_id: str
    role: str = "assistant"


class TextMessageContentEvent(BaseEvent):
    """Emitted for each chunk of text content."""

    type: EventType = EventType.TEXT_MESSAGE_CONTENT
    message_id: str
    delta: str = Field(..., min_length=1)


class TextMessageEndEvent(BaseEvent):
    """Emitted when a text message completes."""

    type: EventType = EventType.TEXT_MESSAGE_END
    message_id: str


class ToolCallStartEvent(BaseEvent):
    """Emitted when a tool call begins."""

    type: EventType = EventType.TOOL_CALL_START
    tool_call_id: str
    tool_call_name: str
    parent_message_id: str | None = None


class ToolCallArgsEvent(BaseEvent):
    """Emitted for chunks of tool call arguments."""

    type: EventType = EventType.TOOL_CALL_ARGS
    tool_call_id: str
    delta: str


class ToolCallEndEvent(BaseEvent):
    """Emitted when a tool call completes."""

    type: EventType = EventType.TOOL_CALL_END
    tool_call_id: str


class StateSnapshotEvent(BaseEvent):
    """Emitted with complete state snapshot."""

    type: EventType = EventType.STATE_SNAPSHOT
    state: dict[str, Any]


class StateDeltaEvent(BaseEvent):
    """Emitted with state delta update."""

    type: EventType = EventType.STATE_DELTA
    delta: dict[str, Any]


class RawEvent(BaseEvent):
    """Raw event for provider-specific data."""

    type: EventType = EventType.RAW
    data: Any


@dataclass
class AGUIEventEmitter:
    """Event emitter for AG-UI streaming responses.

    This class handles the creation and streaming of AG-UI protocol events.
    It manages message and tool call lifecycle events, converting them to
    Server-Sent Events (SSE) format for streaming to clients.

    Example:
        emitter = AGUIEventEmitter(thread_id="t1", run_id="r1")
        async for event in emitter.stream_response(agent, messages):
            yield event
    """

    thread_id: str
    run_id: str
    _message_id: str | None = field(default=None, init=False)
    _tool_call_id: str | None = field(default=None, init=False)

    def _generate_id(self, prefix: str = "msg") -> str:
        """Generate unique identifier."""
        return f"{prefix}-{uuid.uuid4().hex[:12]}"

    def _format_sse(self, event: BaseEvent) -> str:
        """Format event as Server-Sent Event."""
        data = event.model_dump_json()
        return f"event: message\ndata: {data}\n\n"

    def emit_run_started(self) -> str:
        """Emit RUN_STARTED event."""
        event = RunStartedEvent(thread_id=self.thread_id, run_id=self.run_id)
        return self._format_sse(event)

    def emit_run_finished(self) -> str:
        """Emit RUN_FINISHED event."""
        event = RunFinishedEvent(thread_id=self.thread_id, run_id=self.run_id)
        return self._format_sse(event)

    def emit_run_error(self, message: str, code: str | None = None) -> str:
        """Emit RUN_ERROR event."""
        event = RunErrorEvent(message=message, code=code)
        return self._format_sse(event)

    def emit_text_start(self, message_id: str | None = None) -> str:
        """Emit TEXT_MESSAGE_START event."""
        self._message_id = message_id or self._generate_id("msg")
        event = TextMessageStartEvent(message_id=self._message_id)
        return self._format_sse(event)

    def emit_text_content(self, delta: str) -> str:
        """Emit TEXT_MESSAGE_CONTENT event."""
        if not self._message_id:
            raise AGUIError("No active message", details={"delta": delta})
        if not delta:
            raise AGUIError("Delta cannot be empty", details={})
        event = TextMessageContentEvent(message_id=self._message_id, delta=delta)
        return self._format_sse(event)

    def emit_text_end(self) -> str:
        """Emit TEXT_MESSAGE_END event."""
        if not self._message_id:
            raise AGUIError("No active message", details={})
        event = TextMessageEndEvent(message_id=self._message_id)
        result = self._format_sse(event)
        self._message_id = None
        return result

    def emit_tool_call_start(
        self, tool_name: str, tool_call_id: str | None = None
    ) -> str:
        """Emit TOOL_CALL_START event."""
        self._tool_call_id = tool_call_id or self._generate_id("tc")
        event = ToolCallStartEvent(
            tool_call_id=self._tool_call_id,
            tool_call_name=tool_name,
            parent_message_id=self._message_id,
        )
        return self._format_sse(event)

    def emit_tool_call_args(self, delta: str) -> str:
        """Emit TOOL_CALL_ARGS event."""
        if not self._tool_call_id:
            raise AGUIError("No active tool call", details={"delta": delta})
        event = ToolCallArgsEvent(tool_call_id=self._tool_call_id, delta=delta)
        return self._format_sse(event)

    def emit_tool_call_end(self) -> str:
        """Emit TOOL_CALL_END event."""
        if not self._tool_call_id:
            raise AGUIError("No active tool call", details={})
        event = ToolCallEndEvent(tool_call_id=self._tool_call_id)
        result = self._format_sse(event)
        self._tool_call_id = None
        return result

    def emit_state_snapshot(self, state: dict[str, Any]) -> str:
        """Emit STATE_SNAPSHOT event."""
        event = StateSnapshotEvent(state=state)
        return self._format_sse(event)

    def emit_state_delta(self, delta: dict[str, Any]) -> str:
        """Emit STATE_DELTA event."""
        event = StateDeltaEvent(delta=delta)
        return self._format_sse(event)

    def emit_raw(self, data: Any) -> str:
        """Emit RAW event."""
        event = RawEvent(data=data)
        return self._format_sse(event)


class AGUIServer:
    """AG-UI Protocol Server using FastAPI.

    This server implements the AG-UI protocol for streaming chat interfaces.
    It provides endpoints for running agents and streaming responses.

    Example:
        server = AGUIServer(agent=my_agent)
        app = server.create_app()
        # Run with uvicorn: uvicorn app:app --reload
    """

    def __init__(
        self,
        agent: Any | None = None,
        title: str = "AG-UI Server",
        description: str = "Agent-User Interface Protocol Server",
    ) -> None:
        """Initialize AG-UI server.

        Args:
            agent: Agent instance to handle requests
            title: API title for documentation
            description: API description for documentation
        """
        self.agent = agent
        self.title = title
        self.description = description
        self._app: FastAPI | None = None

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI) -> AsyncGenerator[None, None]:
        """Application lifespan manager."""
        with tracer.start_as_current_span("agui_server_startup"):
            yield

    def create_app(self) -> FastAPI:
        """Create FastAPI application with AG-UI endpoints.

        Returns:
            Configured FastAPI application
        """
        self._app = FastAPI(
            title=self.title,
            description=self.description,
            lifespan=self._lifespan,
        )

        # Configure CORS for frontend access
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
            raise AGUIError("App not created", details={})

        @self._app.get("/health")
        async def health() -> dict[str, str]:
            """Health check endpoint."""
            return {"status": "healthy", "protocol": "AG-UI"}

        @self._app.post("/")
        async def run_agent(request: RunAgentInput) -> StreamingResponse:
            """Run agent and stream response via AG-UI protocol."""
            return StreamingResponse(
                self._stream_response(request),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        @self._app.post("/run")
        async def run_agent_alt(request: RunAgentInput) -> StreamingResponse:
            """Alternative endpoint for running agent."""
            return await run_agent(request)

    async def _stream_response(
        self, request: RunAgentInput
    ) -> AsyncGenerator[str, None]:
        """Stream AG-UI events for agent response.

        Args:
            request: Run agent input with messages and context

        Yields:
            SSE-formatted events
        """
        emitter = AGUIEventEmitter(
            thread_id=request.thread_id,
            run_id=request.run_id,
        )

        with tracer.start_as_current_span("agui_stream_response") as span:
            span.set_attribute("thread_id", request.thread_id)
            span.set_attribute("run_id", request.run_id)
            span.set_attribute("message_count", len(request.messages))

            try:
                # Emit run started
                yield emitter.emit_run_started()

                # Process with agent if available
                if self.agent:
                    async for event_str in self._process_with_agent(
                        emitter, request
                    ):
                        yield event_str
                else:
                    # Default echo response for testing
                    async for event_str in self._echo_response(emitter, request):
                        yield event_str

                # Emit run finished
                yield emitter.emit_run_finished()

            except Exception as e:
                record_exception(e)
                yield emitter.emit_run_error(str(e), code="AGENT_ERROR")

    async def _process_with_agent(
        self, emitter: AGUIEventEmitter, request: RunAgentInput
    ) -> AsyncGenerator[str, None]:
        """Process request with configured agent.

        Args:
            emitter: Event emitter for AG-UI events
            request: Run agent input

        Yields:
            SSE-formatted events
        """
        # Convert messages to agent format
        messages = [
            {"role": msg.role, "content": msg.content or ""}
            for msg in request.messages
        ]

        # Get last user message
        last_message = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                last_message = msg["content"]
                break

        # Run agent and stream response
        yield emitter.emit_text_start()

        # Check if agent supports streaming
        if hasattr(self.agent, "stream"):
            async for chunk in self.agent.stream(last_message):
                if chunk:
                    yield emitter.emit_text_content(chunk)
        else:
            # Non-streaming fallback
            response = await self.agent.run(last_message)
            # Simulate streaming by chunking response
            chunk_size = 5
            for i in range(0, len(response), chunk_size):
                chunk = response[i : i + chunk_size]
                if chunk:
                    yield emitter.emit_text_content(chunk)

        yield emitter.emit_text_end()

    async def _echo_response(
        self, emitter: AGUIEventEmitter, request: RunAgentInput
    ) -> AsyncGenerator[str, None]:
        """Echo response for testing without agent.

        Args:
            emitter: Event emitter for AG-UI events
            request: Run agent input

        Yields:
            SSE-formatted events
        """
        yield emitter.emit_text_start()

        # Get last user message and echo
        last_content = "Hello! I received your message."
        for msg in reversed(request.messages):
            if msg.role == "user" and msg.content:
                last_content = f"Echo: {msg.content}"
                break

        # Stream character by character for demo
        for char in last_content:
            yield emitter.emit_text_content(char)

        yield emitter.emit_text_end()

    def set_agent(self, agent: Any) -> None:
        """Set or replace the agent instance.

        Args:
            agent: Agent instance to handle requests
        """
        self.agent = agent


def create_agui_server(
    agent: Any | None = None,
    title: str = "AG-UI Server",
    description: str = "Agent-User Interface Protocol Server",
) -> FastAPI:
    """Create AG-UI FastAPI server.

    Args:
        agent: Optional agent instance to handle requests
        title: API title for documentation
        description: API description for documentation

    Returns:
        Configured FastAPI application
    """
    server = AGUIServer(agent=agent, title=title, description=description)
    return server.create_app()


# For running directly with uvicorn
def create_default_app() -> FastAPI:
    """Create default AG-UI app for development.

    Returns:
        FastAPI application with echo mode (no agent)
    """
    return create_agui_server()


app = create_default_app()
