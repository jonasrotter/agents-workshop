"""Integration tests for Scenario 02: AG-UI Protocol for Streaming Chat.

These tests verify that the AG-UI components work together correctly.
"""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


class TestScenario02Imports:
    """Tests for Scenario 02 imports."""

    def test_agui_imports(self) -> None:
        """All AG-UI components should be importable."""
        from src.agents import (
            AGUIServer,
            AGUIEventEmitter,
            EventType,
            create_agui_server,
        )

        assert AGUIServer is not None
        assert AGUIEventEmitter is not None
        assert EventType is not None
        assert callable(create_agui_server)

    def test_event_models_import(self) -> None:
        """Event models should be importable."""
        from src.agents.agui_server import (
            BaseEvent,
            RunStartedEvent,
            RunFinishedEvent,
            RunErrorEvent,
            TextMessageStartEvent,
            TextMessageContentEvent,
            TextMessageEndEvent,
            ToolCallStartEvent,
            ToolCallArgsEvent,
            ToolCallEndEvent,
            StateSnapshotEvent,
            StateDeltaEvent,
            RawEvent,
        )

        # All models should be classes
        assert isinstance(BaseEvent, type)
        assert isinstance(RunStartedEvent, type)
        assert isinstance(TextMessageContentEvent, type)

    def test_request_models_import(self) -> None:
        """Request models should be importable."""
        from src.agents.agui_server import (
            RunAgentInput,
            Message,
            Tool,
            ToolCall,
        )

        assert isinstance(RunAgentInput, type)
        assert isinstance(Message, type)
        assert isinstance(Tool, type)
        assert isinstance(ToolCall, type)


class TestAGUIServerCreation:
    """Tests for AG-UI server creation."""

    def test_create_server_without_agent(self) -> None:
        """Server should be creatable without agent (echo mode)."""
        from src.agents import create_agui_server

        app = create_agui_server(title="Test Server")
        assert app is not None
        assert app.title == "Test Server"

    def test_create_server_with_mock_agent(self) -> None:
        """Server should be creatable with mock agent."""
        from src.agents import AGUIServer

        mock_agent = MagicMock()
        server = AGUIServer(agent=mock_agent)
        app = server.create_app()

        assert app is not None
        assert server.agent is mock_agent


class TestAGUIServerEndpoints:
    """Tests for AG-UI server endpoints."""

    def test_health_endpoint(self) -> None:
        """Health endpoint should return healthy status."""
        from src.agents import create_agui_server

        app = create_agui_server()
        client = TestClient(app)

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["protocol"] == "AG-UI"

    def test_run_endpoint_exists(self) -> None:
        """Run endpoint should exist."""
        from src.agents import create_agui_server

        app = create_agui_server()
        client = TestClient(app)

        # POST to root endpoint
        response = client.post(
            "/",
            json={
                "thread_id": "t1",
                "run_id": "r1",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")


class TestStreamingResponse:
    """Tests for streaming response functionality."""

    def test_echo_mode_response(self) -> None:
        """Echo mode should stream events correctly."""
        from src.agents import create_agui_server

        app = create_agui_server()
        client = TestClient(app)

        events = []
        with client.stream(
            "POST",
            "/",
            json={
                "thread_id": "t1",
                "run_id": "r1",
                "messages": [{"role": "user", "content": "Hello AG-UI!"}],
            },
        ) as response:
            for line in response.iter_lines():
                if line.startswith("data:"):
                    events.append(json.loads(line[5:].strip()))

        # Should have lifecycle events
        event_types = [e["type"] for e in events]
        assert "RUN_STARTED" in event_types
        assert "TEXT_MESSAGE_START" in event_types
        assert "TEXT_MESSAGE_CONTENT" in event_types
        assert "TEXT_MESSAGE_END" in event_types
        assert "RUN_FINISHED" in event_types

    def test_echo_mode_echoes_message(self) -> None:
        """Echo mode should echo user message."""
        from src.agents import create_agui_server

        app = create_agui_server()
        client = TestClient(app)

        content_parts = []
        with client.stream(
            "POST",
            "/",
            json={
                "thread_id": "t1",
                "run_id": "r1",
                "messages": [{"role": "user", "content": "Test message"}],
            },
        ) as response:
            for line in response.iter_lines():
                if line.startswith("data:"):
                    event = json.loads(line[5:].strip())
                    if event["type"] == "TEXT_MESSAGE_CONTENT":
                        content_parts.append(event["delta"])

        full_content = "".join(content_parts)
        assert "Test message" in full_content


class TestEventEmitterIntegration:
    """Tests for event emitter integration."""

    def test_emitter_produces_valid_sse(self) -> None:
        """Emitter should produce valid SSE format."""
        from src.agents import AGUIEventEmitter

        emitter = AGUIEventEmitter(thread_id="t1", run_id="r1")

        sse = emitter.emit_run_started()

        # Parse SSE
        lines = sse.strip().split("\n")
        assert lines[0] == "event: message"
        assert lines[1].startswith("data: ")

        # Verify JSON
        data = json.loads(lines[1][6:])
        assert data["type"] == "RUN_STARTED"
        assert data["thread_id"] == "t1"
        assert data["run_id"] == "r1"

    def test_full_conversation_flow(self) -> None:
        """Full conversation should emit correct event sequence."""
        from src.agents import AGUIEventEmitter

        emitter = AGUIEventEmitter(thread_id="t1", run_id="r1")

        events = []

        # Simulate conversation
        events.append(emitter.emit_run_started())
        events.append(emitter.emit_text_start())
        events.append(emitter.emit_text_content("Hello "))
        events.append(emitter.emit_text_content("world!"))
        events.append(emitter.emit_text_end())
        events.append(emitter.emit_run_finished())

        # All should be valid SSE
        for sse in events:
            assert "event: message\n" in sse
            assert "data: " in sse


class TestErrorHandling:
    """Tests for error handling."""

    def test_emitter_error_without_message(self) -> None:
        """Emitter should raise error for content without message."""
        from src.agents import AGUIEventEmitter
        from src.common.exceptions import AGUIError

        emitter = AGUIEventEmitter(thread_id="t1", run_id="r1")

        with pytest.raises(AGUIError):
            emitter.emit_text_content("Hello")

    def test_emitter_error_without_tool_call(self) -> None:
        """Emitter should raise error for tool args without tool call."""
        from src.agents import AGUIEventEmitter
        from src.common.exceptions import AGUIError

        emitter = AGUIEventEmitter(thread_id="t1", run_id="r1")

        with pytest.raises(AGUIError):
            emitter.emit_tool_call_args('{"query": "test"}')

    def test_run_error_event(self) -> None:
        """Run error event should be emittable."""
        from src.agents import AGUIEventEmitter

        emitter = AGUIEventEmitter(thread_id="t1", run_id="r1")
        sse = emitter.emit_run_error("Something went wrong", code="TEST_ERROR")

        # Parse and verify
        for line in sse.split("\n"):
            if line.startswith("data:"):
                data = json.loads(line[6:])
                assert data["type"] == "RUN_ERROR"
                assert data["message"] == "Something went wrong"
                assert data["code"] == "TEST_ERROR"


class TestAgentIntegration:
    """Tests for agent integration with AG-UI server."""

    @pytest.mark.asyncio
    async def test_server_with_mock_agent(
        self, env_vars: dict[str, str]
    ) -> None:
        """Server should work with mock agent."""
        from src.agents import AGUIServer, ResearchAgent

        # Create mock agent
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value="Mock response")

        # Create server with mock agent
        server = AGUIServer(agent=mock_agent)
        app = server.create_app()
        client = TestClient(app)

        events = []
        with client.stream(
            "POST",
            "/",
            json={
                "thread_id": "t1",
                "run_id": "r1",
                "messages": [{"role": "user", "content": "Test"}],
            },
        ) as response:
            for line in response.iter_lines():
                if line.startswith("data:"):
                    events.append(json.loads(line[5:].strip()))

        event_types = [e["type"] for e in events]
        assert "RUN_STARTED" in event_types
        assert "RUN_FINISHED" in event_types

    def test_set_agent_after_creation(self) -> None:
        """Agent should be settable after server creation."""
        from src.agents import AGUIServer

        server = AGUIServer()
        assert server.agent is None

        mock_agent = MagicMock()
        server.set_agent(mock_agent)

        assert server.agent is mock_agent


class TestToolCallStreaming:
    """Tests for tool call streaming."""

    def test_tool_call_event_sequence(self) -> None:
        """Tool call should emit correct event sequence."""
        from src.agents import AGUIEventEmitter

        emitter = AGUIEventEmitter(thread_id="t1", run_id="r1")
        events = []

        # Message with tool call
        events.append(emitter.emit_text_start())
        events.append(emitter.emit_tool_call_start("search_web", "tc1"))
        events.append(emitter.emit_tool_call_args('{"query":'))
        events.append(emitter.emit_tool_call_args('"test"}'))
        events.append(emitter.emit_tool_call_end())
        events.append(emitter.emit_text_end())

        # Extract event types
        event_types = []
        for sse in events:
            for line in sse.split("\n"):
                if line.startswith("data:"):
                    data = json.loads(line[6:])
                    event_types.append(data["type"])

        assert event_types == [
            "TEXT_MESSAGE_START",
            "TOOL_CALL_START",
            "TOOL_CALL_ARGS",
            "TOOL_CALL_ARGS",
            "TOOL_CALL_END",
            "TEXT_MESSAGE_END",
        ]


class TestStateManagement:
    """Tests for state management events."""

    def test_state_snapshot_event(self) -> None:
        """State snapshot should emit correctly."""
        from src.agents import AGUIEventEmitter

        emitter = AGUIEventEmitter(thread_id="t1", run_id="r1")
        state = {"message_count": 5, "tokens": 100}

        sse = emitter.emit_state_snapshot(state)

        for line in sse.split("\n"):
            if line.startswith("data:"):
                data = json.loads(line[6:])
                assert data["type"] == "STATE_SNAPSHOT"
                assert data["state"] == state

    def test_state_delta_event(self) -> None:
        """State delta should emit correctly."""
        from src.agents import AGUIEventEmitter

        emitter = AGUIEventEmitter(thread_id="t1", run_id="r1")
        delta = {"message_count": 6}

        sse = emitter.emit_state_delta(delta)

        for line in sse.split("\n"):
            if line.startswith("data:"):
                data = json.loads(line[6:])
                assert data["type"] == "STATE_DELTA"
                assert data["delta"] == delta


class TestTelemetryIntegration:
    """Tests for telemetry integration."""

    def test_tracer_import(self) -> None:
        """Tracer should be importable from agui_server."""
        from src.agents.agui_server import tracer

        assert tracer is not None


@pytest.mark.integration
class TestScenario02EndToEnd:
    """End-to-end tests for Scenario 02."""

    def test_complete_chat_flow(self) -> None:
        """Complete chat flow should work."""
        from src.agents import create_agui_server

        app = create_agui_server()
        client = TestClient(app)

        # Simulate multi-turn conversation
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is AG-UI?"},
        ]

        events = []
        with client.stream(
            "POST",
            "/",
            json={"thread_id": "t1", "run_id": "r1", "messages": messages},
        ) as response:
            for line in response.iter_lines():
                if line.startswith("data:"):
                    events.append(json.loads(line[5:].strip()))

        # Verify event sequence
        assert events[0]["type"] == "RUN_STARTED"
        assert events[-1]["type"] == "RUN_FINISHED"

        # Should have message content
        content_events = [e for e in events if e["type"] == "TEXT_MESSAGE_CONTENT"]
        assert len(content_events) > 0
