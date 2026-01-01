"""Contract tests for AG-UI event schemas.

These tests validate that AG-UI event implementations conform to
the contract defined in contracts/agui-events.md.
"""

import json
import time
from typing import Any

import pytest

from src.agents.agui_server import (
    AGUIEventEmitter,
    BaseEvent,
    EventType,
    Message,
    RunAgentInput,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    Tool,
    ToolCall,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    StateSnapshotEvent,
    StateDeltaEvent,
    RawEvent,
)


class TestEventTypeEnum:
    """Tests for EventType enumeration."""

    def test_all_event_types_defined(self) -> None:
        """All contract event types should be defined."""
        expected_types = {
            "RUN_STARTED",
            "RUN_FINISHED",
            "RUN_ERROR",
            "TEXT_MESSAGE_START",
            "TEXT_MESSAGE_CONTENT",
            "TEXT_MESSAGE_END",
            "TOOL_CALL_START",
            "TOOL_CALL_ARGS",
            "TOOL_CALL_END",
            "STATE_SNAPSHOT",
            "STATE_DELTA",
            "RAW",
        }
        actual_types = {e.value for e in EventType}
        assert expected_types == actual_types

    def test_event_type_string_values(self) -> None:
        """Event types should have string values."""
        for event_type in EventType:
            assert isinstance(event_type.value, str)
            assert event_type.value == event_type.name


class TestRequestSchemas:
    """Tests for request schemas."""

    def test_message_schema(self) -> None:
        """Message should conform to contract schema."""
        # User message
        user_msg = Message(role="user", content="Hello")
        assert user_msg.role == "user"
        assert user_msg.content == "Hello"

        # Assistant message with tool calls
        tool_call = ToolCall(
            id="tc-1",
            type="function",
            function={"name": "search", "arguments": "{}"},
        )
        assistant_msg = Message(
            role="assistant", content="Let me search", tool_calls=[tool_call]
        )
        assert assistant_msg.role == "assistant"
        assert len(assistant_msg.tool_calls) == 1

        # Tool result message
        tool_msg = Message(role="tool", content="Result", tool_call_id="tc-1")
        assert tool_msg.role == "tool"
        assert tool_msg.tool_call_id == "tc-1"

    def test_message_role_validation(self) -> None:
        """Message role should be validated."""
        valid_roles = ["user", "assistant", "tool", "system"]
        for role in valid_roles:
            msg = Message(role=role, content="test")
            assert msg.role == role

    def test_tool_schema(self) -> None:
        """Tool should conform to contract schema."""
        tool = Tool(
            name="search_web",
            description="Search the web",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        )
        assert tool.name == "search_web"
        assert tool.description == "Search the web"
        assert "properties" in tool.parameters

    def test_run_agent_input_schema(self) -> None:
        """RunAgentInput should conform to contract schema."""
        request = RunAgentInput(
            thread_id="thread-123",
            run_id="run-456",
            messages=[Message(role="user", content="Hello")],
            tools=[
                Tool(name="search", description="Search", parameters={"type": "object"})
            ],
            context={"key": "value"},
        )
        assert request.thread_id == "thread-123"
        assert request.run_id == "run-456"
        assert len(request.messages) == 1
        assert len(request.tools) == 1
        assert request.context == {"key": "value"}

    def test_run_agent_input_required_fields(self) -> None:
        """RunAgentInput should require thread_id, run_id, messages."""
        # Valid minimal request
        request = RunAgentInput(
            thread_id="t1",
            run_id="r1",
            messages=[Message(role="user", content="Hi")],
        )
        assert request.tools is None
        assert request.context is None


class TestLifecycleEventSchemas:
    """Tests for lifecycle event schemas."""

    def test_run_started_event_schema(self) -> None:
        """RunStartedEvent should conform to contract."""
        event = RunStartedEvent(thread_id="t-123", run_id="r-456")
        assert event.type == EventType.RUN_STARTED
        assert event.thread_id == "t-123"
        assert event.run_id == "r-456"
        assert isinstance(event.timestamp, int)

    def test_run_finished_event_schema(self) -> None:
        """RunFinishedEvent should conform to contract."""
        event = RunFinishedEvent(thread_id="t-123", run_id="r-456")
        assert event.type == EventType.RUN_FINISHED
        assert event.thread_id == "t-123"
        assert event.run_id == "r-456"

    def test_run_error_event_schema(self) -> None:
        """RunErrorEvent should conform to contract."""
        event = RunErrorEvent(message="Connection failed", code="CONN_ERROR")
        assert event.type == EventType.RUN_ERROR
        assert event.message == "Connection failed"
        assert event.code == "CONN_ERROR"

    def test_run_error_optional_code(self) -> None:
        """RunErrorEvent code should be optional."""
        event = RunErrorEvent(message="Unknown error")
        assert event.message == "Unknown error"
        assert event.code is None


class TestTextMessageEventSchemas:
    """Tests for text message event schemas."""

    def test_text_message_start_schema(self) -> None:
        """TextMessageStartEvent should conform to contract."""
        event = TextMessageStartEvent(message_id="msg-123", role="assistant")
        assert event.type == EventType.TEXT_MESSAGE_START
        assert event.message_id == "msg-123"
        assert event.role == "assistant"

    def test_text_message_content_schema(self) -> None:
        """TextMessageContentEvent should conform to contract."""
        event = TextMessageContentEvent(message_id="msg-123", delta="Hello")
        assert event.type == EventType.TEXT_MESSAGE_CONTENT
        assert event.message_id == "msg-123"
        assert event.delta == "Hello"

    def test_text_message_content_min_length(self) -> None:
        """TextMessageContentEvent delta should have min length 1."""
        # Valid single character
        event = TextMessageContentEvent(message_id="m1", delta="a")
        assert event.delta == "a"

        # Empty delta should fail validation
        with pytest.raises(Exception):  # Pydantic validation error
            TextMessageContentEvent(message_id="m1", delta="")

    def test_text_message_end_schema(self) -> None:
        """TextMessageEndEvent should conform to contract."""
        event = TextMessageEndEvent(message_id="msg-123")
        assert event.type == EventType.TEXT_MESSAGE_END
        assert event.message_id == "msg-123"


class TestToolCallEventSchemas:
    """Tests for tool call event schemas."""

    def test_tool_call_start_schema(self) -> None:
        """ToolCallStartEvent should conform to contract."""
        event = ToolCallStartEvent(
            tool_call_id="tc-123",
            tool_call_name="search_web",
            parent_message_id="msg-456",
        )
        assert event.type == EventType.TOOL_CALL_START
        assert event.tool_call_id == "tc-123"
        assert event.tool_call_name == "search_web"
        assert event.parent_message_id == "msg-456"

    def test_tool_call_start_optional_parent(self) -> None:
        """ToolCallStartEvent parent_message_id should be optional."""
        event = ToolCallStartEvent(tool_call_id="tc-1", tool_call_name="calc")
        assert event.parent_message_id is None

    def test_tool_call_args_schema(self) -> None:
        """ToolCallArgsEvent should conform to contract."""
        event = ToolCallArgsEvent(tool_call_id="tc-123", delta='{"query":')
        assert event.type == EventType.TOOL_CALL_ARGS
        assert event.tool_call_id == "tc-123"
        assert event.delta == '{"query":'

    def test_tool_call_end_schema(self) -> None:
        """ToolCallEndEvent should conform to contract."""
        event = ToolCallEndEvent(tool_call_id="tc-123")
        assert event.type == EventType.TOOL_CALL_END
        assert event.tool_call_id == "tc-123"


class TestStateEventSchemas:
    """Tests for state event schemas."""

    def test_state_snapshot_schema(self) -> None:
        """StateSnapshotEvent should conform to contract."""
        state = {"key": "value", "count": 42}
        event = StateSnapshotEvent(state=state)
        assert event.type == EventType.STATE_SNAPSHOT
        assert event.state == state

    def test_state_delta_schema(self) -> None:
        """StateDeltaEvent should conform to contract."""
        delta = {"count": 43}
        event = StateDeltaEvent(delta=delta)
        assert event.type == EventType.STATE_DELTA
        assert event.delta == delta


class TestRawEventSchema:
    """Tests for raw event schema."""

    def test_raw_event_schema(self) -> None:
        """RawEvent should conform to contract."""
        data = {"provider": "azure", "details": {"model": "gpt-4"}}
        event = RawEvent(data=data)
        assert event.type == EventType.RAW
        assert event.data == data


class TestBaseEventSchema:
    """Tests for base event schema."""

    def test_timestamp_auto_generated(self) -> None:
        """Timestamp should be auto-generated."""
        before = int(time.time() * 1000)
        event = RunStartedEvent(thread_id="t1", run_id="r1")
        after = int(time.time() * 1000)

        assert before <= event.timestamp <= after

    def test_raw_event_optional(self) -> None:
        """raw_event field should be optional."""
        event = RunStartedEvent(thread_id="t1", run_id="r1")
        assert event.raw_event is None

        event_with_raw = RunStartedEvent(
            thread_id="t1", run_id="r1", raw_event={"original": "data"}
        )
        assert event_with_raw.raw_event == {"original": "data"}


class TestSSEFormat:
    """Tests for SSE formatting."""

    def test_sse_event_format(self) -> None:
        """Events should be formatted as SSE."""
        emitter = AGUIEventEmitter(thread_id="t1", run_id="r1")
        sse = emitter.emit_run_started()

        # Should have event: message line
        assert "event: message\n" in sse
        # Should have data: JSON line
        assert "data: " in sse
        # Should end with double newline
        assert sse.endswith("\n\n")

    def test_sse_data_is_valid_json(self) -> None:
        """SSE data should be valid JSON."""
        emitter = AGUIEventEmitter(thread_id="t1", run_id="r1")
        sse = emitter.emit_run_started()

        # Extract data line
        for line in sse.split("\n"):
            if line.startswith("data:"):
                json_str = line[5:].strip()
                data = json.loads(json_str)
                assert data["type"] == "RUN_STARTED"
                assert data["thread_id"] == "t1"
                assert data["run_id"] == "r1"


class TestEventEmitterLifecycle:
    """Tests for event emitter lifecycle management."""

    def test_message_lifecycle(self) -> None:
        """Message events should follow proper lifecycle."""
        emitter = AGUIEventEmitter(thread_id="t1", run_id="r1")

        # Start message
        emitter.emit_text_start(message_id="m1")
        assert emitter._message_id == "m1"

        # Add content
        emitter.emit_text_content("Hello")

        # End message
        emitter.emit_text_end()
        assert emitter._message_id is None

    def test_tool_call_lifecycle(self) -> None:
        """Tool call events should follow proper lifecycle."""
        emitter = AGUIEventEmitter(thread_id="t1", run_id="r1")

        # Start message first
        emitter.emit_text_start()

        # Start tool call
        emitter.emit_tool_call_start("search", tool_call_id="tc1")
        assert emitter._tool_call_id == "tc1"

        # Add args
        emitter.emit_tool_call_args('{"query": "test"}')

        # End tool call
        emitter.emit_tool_call_end()
        assert emitter._tool_call_id is None


class TestEventSerialization:
    """Tests for event serialization."""

    def test_event_to_json(self) -> None:
        """Events should serialize to valid JSON."""
        event = RunStartedEvent(thread_id="t1", run_id="r1")
        json_str = event.model_dump_json()
        data = json.loads(json_str)

        assert data["type"] == "RUN_STARTED"
        assert data["thread_id"] == "t1"
        assert data["run_id"] == "r1"
        assert "timestamp" in data

    def test_event_from_json(self) -> None:
        """Events should deserialize from JSON."""
        json_data = {
            "type": "TEXT_MESSAGE_CONTENT",
            "message_id": "m1",
            "delta": "Hello",
            "timestamp": 1234567890,
        }
        event = TextMessageContentEvent.model_validate(json_data)
        assert event.message_id == "m1"
        assert event.delta == "Hello"
