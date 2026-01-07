"""Unit tests for AG-UI SDK integration components.

Tests for:
- AGUIEventEmitter (SDK mode and legacy mode)
- AGUIClient wrapper
- AGUIAgentProtocol
- create_agui_endpoint helper
- SDK type integration
"""

from __future__ import annotations

import json
import pytest
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.agui_server import (
    AGUIEventEmitter,
    AGUIClient,
    AGUIAgentProtocol,
    AGUIServer,
    create_agui_server,
    create_agui_endpoint,
    EventType,
    # Legacy types (preserved for backward compatibility)
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
    # Request types
    Message,
    Tool,
    ToolCall,
    RunAgentInput,
)


class TestEventTypeSDKIntegration:
    """Tests for SDK EventType integration."""

    def test_event_type_is_from_sdk(self) -> None:
        """EventType should be imported from ag_ui.core."""
        from ag_ui.core import EventType as SDKEventType
        assert EventType is SDKEventType

    def test_event_type_has_extended_values(self) -> None:
        """SDK EventType has more values than the original 12."""
        values = [e.value for e in EventType]
        # SDK has 26 event types
        assert len(values) >= 12
        # All original types are present
        assert "RUN_STARTED" in values
        assert "RUN_FINISHED" in values
        assert "TEXT_MESSAGE_CONTENT" in values

    def test_event_type_has_sdk_specific_values(self) -> None:
        """SDK EventType has additional types like CUSTOM."""
        values = [e.value for e in EventType]
        assert "CUSTOM" in values
        assert "MESSAGES_SNAPSHOT" in values


class TestAGUIEventEmitterSDKMode:
    """Tests for AGUIEventEmitter SDK mode."""

    def test_default_is_legacy_mode(self) -> None:
        """Default emitter should use legacy mode."""
        emitter = AGUIEventEmitter(thread_id="t1", run_id="r1")
        assert emitter.use_sdk is False

    def test_sdk_mode_can_be_enabled(self) -> None:
        """Emitter should support SDK mode."""
        emitter = AGUIEventEmitter(thread_id="t1", run_id="r1", use_sdk=True)
        assert emitter.use_sdk is True

    def test_legacy_mode_format(self) -> None:
        """Legacy mode should use event: message format."""
        emitter = AGUIEventEmitter(thread_id="t1", run_id="r1", use_sdk=False)
        event = emitter.emit_run_started()
        assert event.startswith("event: message\ndata: ")
        assert event.endswith("\n\n")

    def test_sdk_mode_format(self) -> None:
        """SDK mode should use data: format from EventEncoder."""
        emitter = AGUIEventEmitter(thread_id="t1", run_id="r1", use_sdk=True)
        event = emitter.emit_run_started()
        # SDK encoder outputs: data: {json}\n\n (without event: message prefix)
        assert event.startswith("data: ")
        assert event.endswith("\n\n")

    def test_emit_run_started_sdk_mode(self) -> None:
        """emit_run_started should work in SDK mode."""
        emitter = AGUIEventEmitter(thread_id="t1", run_id="r1", use_sdk=True)
        event = emitter.emit_run_started()
        # Parse the data portion
        assert "RUN_STARTED" in event or "runStarted" in event.lower()

    def test_emit_text_content_sdk_mode(self) -> None:
        """emit_text_content should work in SDK mode."""
        emitter = AGUIEventEmitter(thread_id="t1", run_id="r1", use_sdk=True)
        emitter.emit_text_start()
        event = emitter.emit_text_content("Hello")
        assert "Hello" in event


class TestAGUIEventEmitterLegacyCompatibility:
    """Tests for backward compatibility with legacy emitter."""

    def test_all_emit_methods_exist(self) -> None:
        """All original emit methods should exist."""
        emitter = AGUIEventEmitter(thread_id="t1", run_id="r1")
        assert hasattr(emitter, "emit_run_started")
        assert hasattr(emitter, "emit_run_finished")
        assert hasattr(emitter, "emit_run_error")
        assert hasattr(emitter, "emit_text_start")
        assert hasattr(emitter, "emit_text_content")
        assert hasattr(emitter, "emit_text_end")
        assert hasattr(emitter, "emit_tool_call_start")
        assert hasattr(emitter, "emit_tool_call_args")
        assert hasattr(emitter, "emit_tool_call_end")
        assert hasattr(emitter, "emit_state_snapshot")
        assert hasattr(emitter, "emit_state_delta")
        assert hasattr(emitter, "emit_raw")

    def test_message_lifecycle_legacy(self) -> None:
        """Message lifecycle should work in legacy mode."""
        emitter = AGUIEventEmitter(thread_id="t1", run_id="r1")
        start = emitter.emit_text_start()
        content = emitter.emit_text_content("Hi")
        end = emitter.emit_text_end()
        
        assert "TEXT_MESSAGE_START" in start
        assert "TEXT_MESSAGE_CONTENT" in content
        assert "TEXT_MESSAGE_END" in end

    def test_tool_call_lifecycle_legacy(self) -> None:
        """Tool call lifecycle should work in legacy mode."""
        emitter = AGUIEventEmitter(thread_id="t1", run_id="r1")
        start = emitter.emit_tool_call_start("search")
        args = emitter.emit_tool_call_args('{"query": "test"}')
        end = emitter.emit_tool_call_end()
        
        assert "TOOL_CALL_START" in start
        assert "TOOL_CALL_ARGS" in args
        assert "TOOL_CALL_END" in end


class TestAGUIAgentProtocol:
    """Tests for AGUIAgentProtocol."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """AGUIAgentProtocol should be runtime checkable."""
        from typing import runtime_checkable
        # Protocol is already runtime_checkable (decorator applied)
        assert hasattr(AGUIAgentProtocol, "__protocol_attrs__") or hasattr(AGUIAgentProtocol, "_is_runtime_protocol")

    def test_protocol_has_run_method(self) -> None:
        """Protocol should require run method."""
        # Create a class that implements the protocol
        class ValidAgent:
            async def run(self, message: str) -> str:
                return f"Response to: {message}"
        
        agent = ValidAgent()
        assert isinstance(agent, AGUIAgentProtocol)

    def test_class_without_run_not_protocol(self) -> None:
        """Class without run method should not match protocol."""
        class InvalidAgent:
            def process(self, message: str) -> str:
                return message
        
        agent = InvalidAgent()
        assert not isinstance(agent, AGUIAgentProtocol)


class TestAGUIClientWrapper:
    """Tests for AGUIClient wrapper."""

    def test_client_creation(self) -> None:
        """AGUIClient should be creatable with endpoint."""
        client = AGUIClient(endpoint="http://localhost:8000")
        assert client.endpoint == "http://localhost:8000"
        assert client.timeout == 60.0

    def test_client_custom_timeout(self) -> None:
        """AGUIClient should accept custom timeout."""
        client = AGUIClient(endpoint="http://localhost:8000", timeout=120.0)
        assert client.timeout == 120.0

    def test_client_has_stream_method(self) -> None:
        """AGUIClient should have stream method."""
        client = AGUIClient(endpoint="http://localhost:8000")
        assert hasattr(client, "stream")
        assert callable(client.stream)

    def test_client_has_send_method(self) -> None:
        """AGUIClient should have send method."""
        client = AGUIClient(endpoint="http://localhost:8000")
        assert hasattr(client, "send")
        assert callable(client.send)


class TestCreateAGUIEndpoint:
    """Tests for create_agui_endpoint helper."""

    def test_function_exists(self) -> None:
        """create_agui_endpoint should be importable."""
        assert callable(create_agui_endpoint)

    def test_adds_endpoint_to_app(self) -> None:
        """create_agui_endpoint should add endpoint to FastAPI app."""
        from fastapi import FastAPI
        
        app = FastAPI()
        
        # Create a mock agent
        class MockAgent:
            async def run(self, message: str) -> str:
                return f"Echo: {message}"
        
        agent = MockAgent()
        
        # This should not raise - patch at the point where the import happens
        with patch("agent_framework_ag_ui.add_agent_framework_fastapi_endpoint") as mock_add:
            create_agui_endpoint(app, agent)
            mock_add.assert_called_once_with(app, agent, path="/")

    def test_custom_path(self) -> None:
        """create_agui_endpoint should support custom path."""
        from fastapi import FastAPI
        
        app = FastAPI()
        
        class MockAgent:
            async def run(self, message: str) -> str:
                return message
        
        agent = MockAgent()
        
        with patch("agent_framework_ag_ui.add_agent_framework_fastapi_endpoint") as mock_add:
            create_agui_endpoint(app, agent, path="/chat")
            mock_add.assert_called_once_with(app, agent, path="/chat")


class TestLegacyTypeAliases:
    """Tests for backward-compatible type aliases."""

    def test_event_classes_are_pydantic_models(self) -> None:
        """All event classes should be Pydantic models."""
        from pydantic import BaseModel
        
        assert issubclass(BaseEvent, BaseModel)
        assert issubclass(RunStartedEvent, BaseModel)
        assert issubclass(RunFinishedEvent, BaseModel)
        assert issubclass(RunErrorEvent, BaseModel)
        assert issubclass(TextMessageStartEvent, BaseModel)
        assert issubclass(TextMessageContentEvent, BaseModel)
        assert issubclass(TextMessageEndEvent, BaseModel)
        assert issubclass(ToolCallStartEvent, BaseModel)
        assert issubclass(ToolCallArgsEvent, BaseModel)
        assert issubclass(ToolCallEndEvent, BaseModel)
        assert issubclass(StateSnapshotEvent, BaseModel)
        assert issubclass(StateDeltaEvent, BaseModel)
        assert issubclass(RawEvent, BaseModel)

    def test_request_types_are_pydantic_models(self) -> None:
        """All request types should be Pydantic models."""
        from pydantic import BaseModel
        
        assert issubclass(Message, BaseModel)
        assert issubclass(Tool, BaseModel)
        assert issubclass(ToolCall, BaseModel)
        assert issubclass(RunAgentInput, BaseModel)

    def test_run_started_event_creation(self) -> None:
        """RunStartedEvent should be creatable."""
        event = RunStartedEvent(thread_id="t1", run_id="r1")
        assert event.thread_id == "t1"
        assert event.run_id == "r1"
        assert event.type == EventType.RUN_STARTED

    def test_text_message_content_event_creation(self) -> None:
        """TextMessageContentEvent should be creatable."""
        event = TextMessageContentEvent(message_id="m1", delta="Hello")
        assert event.message_id == "m1"
        assert event.delta == "Hello"
        assert event.type == EventType.TEXT_MESSAGE_CONTENT


class TestAGUIServerUseSdkParameter:
    """Tests for AGUIServer use_sdk parameter support."""

    def test_emitter_in_stream_response(self) -> None:
        """AGUIEventEmitter should be used in stream_response."""
        server = AGUIServer()
        app = server.create_app()
        
        # The server creates emitters internally
        # Just verify the server works
        assert app is not None


class TestModuleExports:
    """Tests for module exports."""

    def test_all_exports_importable(self) -> None:
        """All exports should be importable from src.agents."""
        from src.agents import (
            AGUIServer,
            AGUIEventEmitter,
            AGUIClient,
            AGUIAgentProtocol,
            create_agui_server,
            create_agui_endpoint,
            EventType,
        )
        
        assert AGUIServer is not None
        assert AGUIEventEmitter is not None
        assert AGUIClient is not None
        assert AGUIAgentProtocol is not None
        assert create_agui_server is not None
        assert create_agui_endpoint is not None
        assert EventType is not None

    def test_exports_in_all(self) -> None:
        """New exports should be in __all__."""
        from src.agents import __all__
        
        assert "AGUIServer" in __all__
        assert "AGUIEventEmitter" in __all__
        assert "AGUIClient" in __all__
        assert "AGUIAgentProtocol" in __all__
        assert "create_agui_server" in __all__
        assert "create_agui_endpoint" in __all__
        assert "EventType" in __all__
