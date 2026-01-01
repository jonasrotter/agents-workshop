"""
Contract tests for A2A Protocol schemas.

Validates implementations against contracts/a2a-protocol.md.
These tests ensure the A2A server produces valid protocol messages.
"""

import json
import pytest
from typing import Any

from src.agents.a2a_server import (
    # Core classes
    A2AServer,
    TaskManager,
    # Models
    AgentCard,
    Skill,
    AgentCapabilities,
    Task,
    TaskState,
    TaskStatus,
    JSONRPCRequest,
    JSONRPCResponse,
    JSONRPCError,
    Message,
    TextPart,
    FilePart,
    DataPart,
    Artifact,
    ErrorCode,
)


class TestAgentCardSchema:
    """Test Agent Card schema compliance."""

    def test_agent_card_required_fields(self) -> None:
        """Agent Card must have name and url."""
        card = AgentCard(name="Test Agent", url="http://localhost:8000")
        
        assert card.name == "Test Agent"
        assert card.url == "http://localhost:8000"

    def test_agent_card_optional_fields(self) -> None:
        """Agent Card supports optional fields."""
        card = AgentCard(
            name="Test Agent",
            description="A test agent",
            url="http://localhost:8000",
            version="1.0.0",
        )
        
        assert card.description == "A test agent"
        assert card.version == "1.0.0"

    def test_agent_card_with_skills(self) -> None:
        """Agent Card can include skills."""
        card = AgentCard(
            name="Test Agent",
            url="http://localhost:8000",
            skills=[
                Skill(id="skill_1", name="Skill One"),
                Skill(id="skill_2", name="Skill Two", description="Second skill"),
            ],
        )
        
        assert len(card.skills) == 2
        assert card.skills[0].id == "skill_1"
        assert card.skills[1].description == "Second skill"

    def test_agent_card_serialization(self) -> None:
        """Agent Card serializes to valid JSON."""
        card = AgentCard(
            name="Test Agent",
            url="http://localhost:8000",
            skills=[Skill(id="test", name="Test")],
        )
        
        data = card.model_dump()
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        
        assert parsed["name"] == "Test Agent"
        assert parsed["url"] == "http://localhost:8000"
        assert len(parsed["skills"]) == 1


class TestSkillSchema:
    """Test Skill schema compliance."""

    def test_skill_required_fields(self) -> None:
        """Skill must have id and name."""
        skill = Skill(id="test_skill", name="Test Skill")
        
        assert skill.id == "test_skill"
        assert skill.name == "Test Skill"

    def test_skill_with_description(self) -> None:
        """Skill can have optional description."""
        skill = Skill(
            id="test_skill",
            name="Test Skill",
            description="A skill for testing",
        )
        
        assert skill.description == "A skill for testing"

    def test_skill_with_input_schema(self) -> None:
        """Skill can define input schema."""
        skill = Skill(
            id="test_skill",
            name="Test Skill",
            inputSchema={
                "type": "object",
                "properties": {
                    "input": {"type": "string"},
                },
                "required": ["input"],
            },
        )
        
        assert skill.inputSchema is not None
        assert skill.inputSchema["type"] == "object"

    def test_skill_with_output_schema(self) -> None:
        """Skill can define output schema."""
        skill = Skill(
            id="test_skill",
            name="Test Skill",
            outputSchema={
                "type": "object",
                "properties": {
                    "result": {"type": "string"},
                },
            },
        )
        
        assert skill.outputSchema is not None


class TestAgentCapabilitiesSchema:
    """Test Agent Capabilities schema compliance."""

    def test_default_capabilities(self) -> None:
        """Default capabilities should be minimal."""
        caps = AgentCapabilities()
        
        assert caps.streaming is False
        assert caps.pushNotifications is False
        assert caps.stateManagement is False

    def test_enabled_capabilities(self) -> None:
        """Capabilities can be enabled."""
        caps = AgentCapabilities(
            streaming=True,
            pushNotifications=True,
            stateManagement=True,
        )
        
        assert caps.streaming is True
        assert caps.pushNotifications is True
        assert caps.stateManagement is True


class TestTaskStateEnum:
    """Test TaskState enum compliance."""

    def test_all_states_present(self) -> None:
        """All required task states must be present."""
        required_states = [
            "submitted",
            "working",
            "input-required",
            "completed",
            "failed",
            "cancelled",
        ]
        
        actual_states = [state.value for state in TaskState]
        
        for state in required_states:
            assert state in actual_states, f"Missing state: {state}"

    def test_state_values(self) -> None:
        """Task states have correct string values."""
        assert TaskState.SUBMITTED.value == "submitted"
        assert TaskState.WORKING.value == "working"
        assert TaskState.INPUT_REQUIRED.value == "input-required"
        assert TaskState.COMPLETED.value == "completed"
        assert TaskState.FAILED.value == "failed"
        assert TaskState.CANCELLED.value == "cancelled"


class TestTaskStatusSchema:
    """Test TaskStatus schema compliance."""

    def test_status_with_state(self) -> None:
        """TaskStatus must have state."""
        status = TaskStatus(state=TaskState.SUBMITTED)
        
        assert status.state == TaskState.SUBMITTED

    def test_status_with_message(self) -> None:
        """TaskStatus can have optional message."""
        status = TaskStatus(
            state=TaskState.WORKING,
            message=Message(
                role="assistant",
                parts=[TextPart(text="Processing...")],
                messageId="msg-1",
            ),
        )
        
        assert status.message is not None
        assert status.message.role == "assistant"


class TestTaskSchema:
    """Test Task schema compliance."""

    def test_task_required_fields(self) -> None:
        """Task must have id, contextId, and status."""
        task = Task(
            id="task-001",
            contextId="ctx-001",
            status=TaskStatus(state=TaskState.SUBMITTED),
        )
        
        assert task.id == "task-001"
        assert task.contextId == "ctx-001"
        assert task.status.state == TaskState.SUBMITTED

    def test_task_with_artifacts(self) -> None:
        """Task can have artifacts."""
        task = Task(
            id="task-001",
            contextId="ctx-001",
            status=TaskStatus(state=TaskState.COMPLETED),
            artifacts=[
                Artifact(
                    artifactId="art-001",
                    name="result",
                    parts=[TextPart(text="Result data")],
                ),
            ],
        )
        
        assert len(task.artifacts) == 1
        assert task.artifacts[0].artifactId == "art-001"


class TestMessageSchema:
    """Test Message schema compliance."""

    def test_message_required_fields(self) -> None:
        """Message must have role and parts."""
        message = Message(
            role="user",
            parts=[TextPart(text="Hello")],
            messageId="msg-001",
        )
        
        assert message.role == "user"
        assert len(message.parts) == 1
        assert message.messageId == "msg-001"

    def test_message_roles(self) -> None:
        """Message supports user and assistant roles."""
        user_msg = Message(
            role="user",
            parts=[TextPart(text="Hello")],
            messageId="msg-u",
        )
        assistant_msg = Message(
            role="assistant",
            parts=[TextPart(text="Hi there!")],
            messageId="msg-a",
        )
        
        assert user_msg.role == "user"
        assert assistant_msg.role == "assistant"


class TestMessagePartSchemas:
    """Test message part type schemas."""

    def test_text_part(self) -> None:
        """TextPart schema compliance."""
        part = TextPart(text="Hello, world!")
        
        assert part.kind == "text"
        assert part.text == "Hello, world!"

    def test_file_part(self) -> None:
        """FilePart schema compliance."""
        part = FilePart(
            fileId="file-001",
            name="document.pdf",
            mimeType="application/pdf",
        )
        
        assert part.kind == "file"
        assert part.fileId == "file-001"
        assert part.name == "document.pdf"
        assert part.mimeType == "application/pdf"

    def test_data_part(self) -> None:
        """DataPart schema compliance."""
        part = DataPart(data={"key": "value"})
        
        assert part.kind == "data"
        assert part.data == {"key": "value"}

    def test_part_serialization(self) -> None:
        """Parts serialize with kind discriminator."""
        text = TextPart(text="test")
        file = FilePart(fileId="f1", name="test.txt", mimeType="text/plain")
        data = DataPart(data={})
        
        assert text.model_dump()["kind"] == "text"
        assert file.model_dump()["kind"] == "file"
        assert data.model_dump()["kind"] == "data"


class TestArtifactSchema:
    """Test Artifact schema compliance."""

    def test_artifact_required_fields(self) -> None:
        """Artifact must have artifactId and parts."""
        artifact = Artifact(
            artifactId="art-001",
            parts=[TextPart(text="Content")],
        )
        
        assert artifact.artifactId == "art-001"
        assert len(artifact.parts) == 1

    def test_artifact_with_name(self) -> None:
        """Artifact can have optional name."""
        artifact = Artifact(
            artifactId="art-001",
            name="result.txt",
            parts=[TextPart(text="Content")],
        )
        
        assert artifact.name == "result.txt"


class TestJSONRPCRequestSchema:
    """Test JSON-RPC request schema compliance."""

    def test_request_required_fields(self) -> None:
        """Request must have jsonrpc, id, and method."""
        request = JSONRPCRequest(
            id="req-001",
            method="message/send",
            params={},
        )
        
        assert request.jsonrpc == "2.0"
        assert request.id == "req-001"
        assert request.method == "message/send"

    def test_request_with_params(self) -> None:
        """Request can have params."""
        request = JSONRPCRequest(
            id="req-001",
            method="message/send",
            params={"contextId": "ctx-001"},
        )
        
        assert request.params["contextId"] == "ctx-001"

    def test_request_serialization(self) -> None:
        """Request serializes to valid JSON-RPC format."""
        request = JSONRPCRequest(
            id="req-001",
            method="test/method",
            params={"key": "value"},
        )
        
        data = request.model_dump()
        
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == "req-001"
        assert data["method"] == "test/method"
        assert data["params"]["key"] == "value"


class TestJSONRPCResponseSchema:
    """Test JSON-RPC response schema compliance."""

    def test_success_response(self) -> None:
        """Response with result."""
        response = JSONRPCResponse(
            id="req-001",
            result={"status": "ok"},
        )
        
        assert response.jsonrpc == "2.0"
        assert response.id == "req-001"
        assert response.result == {"status": "ok"}
        assert response.error is None

    def test_error_response(self) -> None:
        """Response with error."""
        response = JSONRPCResponse(
            id="req-001",
            error=JSONRPCError(
                code=ErrorCode.METHOD_NOT_FOUND,
                message="Method not found",
            ),
        )
        
        assert response.result is None
        assert response.error is not None
        assert response.error.code == ErrorCode.METHOD_NOT_FOUND


class TestJSONRPCErrorSchema:
    """Test JSON-RPC error schema compliance."""

    def test_error_required_fields(self) -> None:
        """Error must have code and message."""
        error = JSONRPCError(
            code=ErrorCode.INTERNAL_ERROR,
            message="Something went wrong",
        )
        
        assert error.code == ErrorCode.INTERNAL_ERROR
        assert error.message == "Something went wrong"

    def test_error_with_data(self) -> None:
        """Error can have optional data."""
        error = JSONRPCError(
            code=ErrorCode.INVALID_PARAMS,
            message="Missing required field",
            data={"field": "contextId"},
        )
        
        assert error.data == {"field": "contextId"}


class TestErrorCodeValues:
    """Test error code values match JSON-RPC spec."""

    def test_standard_error_codes(self) -> None:
        """Standard JSON-RPC error codes."""
        assert ErrorCode.PARSE_ERROR == -32700
        assert ErrorCode.INVALID_REQUEST == -32600
        assert ErrorCode.METHOD_NOT_FOUND == -32601
        assert ErrorCode.INVALID_PARAMS == -32602
        assert ErrorCode.INTERNAL_ERROR == -32603

    def test_a2a_specific_codes(self) -> None:
        """A2A-specific error codes."""
        assert ErrorCode.TASK_NOT_FOUND == -32000
        assert ErrorCode.CONTEXT_NOT_FOUND == -32001
        assert ErrorCode.RATE_LIMITED == -32002
        assert ErrorCode.UNAUTHORIZED == -32003


class TestTaskManagerContract:
    """Test TaskManager behavior contract."""

    def test_create_task(self) -> None:
        """TaskManager creates tasks with correct initial state."""
        manager = TaskManager()
        message = Message(
            role="user",
            parts=[TextPart(text="Test")],
            messageId="msg-1",
        )
        
        task = manager.create_task(context_id="ctx-1", message=message)
        
        assert task.id.startswith("task-")
        assert task.contextId == "ctx-1"
        assert task.status.state == TaskState.SUBMITTED

    def test_get_task(self) -> None:
        """TaskManager retrieves tasks by ID."""
        manager = TaskManager()
        message = Message(
            role="user",
            parts=[TextPart(text="Test")],
            messageId="msg-1",
        )
        
        created = manager.create_task(context_id="ctx-1", message=message)
        retrieved = manager.get_task(created.id)
        
        assert retrieved is not None
        assert retrieved.id == created.id

    def test_get_nonexistent_task(self) -> None:
        """TaskManager returns None for nonexistent tasks."""
        manager = TaskManager()
        
        task = manager.get_task("nonexistent-id")
        
        assert task is None

    def test_update_task_status(self) -> None:
        """TaskManager updates task status."""
        manager = TaskManager()
        message = Message(
            role="user",
            parts=[TextPart(text="Test")],
            messageId="msg-1",
        )
        
        task = manager.create_task(context_id="ctx-1", message=message)
        updated = manager.update_task_status(task.id, TaskState.WORKING)
        
        assert updated is not None
        assert updated.status.state == TaskState.WORKING

    def test_list_tasks_by_context(self) -> None:
        """TaskManager lists tasks by context."""
        manager = TaskManager()
        msg1 = Message(role="user", parts=[TextPart(text="1")], messageId="m1")
        msg2 = Message(role="user", parts=[TextPart(text="2")], messageId="m2")
        
        manager.create_task(context_id="ctx-1", message=msg1)
        manager.create_task(context_id="ctx-1", message=msg2)
        manager.create_task(context_id="ctx-2", message=msg1)
        
        ctx1_tasks = manager.list_tasks(context_id="ctx-1")
        
        assert len(ctx1_tasks) == 2


class TestA2AServerContract:
    """Test A2AServer behavior contract."""

    def test_server_creates_app(self) -> None:
        """Server creates FastAPI app."""
        server = A2AServer(
            name="Test Agent",
            skills=[Skill(id="test", name="Test")],
        )
        
        app = server.create_app()
        
        assert app is not None

    def test_agent_card_properties(self) -> None:
        """Server Agent Card has correct properties."""
        server = A2AServer(
            name="Test Agent",
            description="A test agent",
            url="http://localhost:8000",
            skills=[Skill(id="test", name="Test")],
        )
        
        assert server.agent_card.name == "Test Agent"
        assert server.agent_card.description == "A test agent"
        assert server.agent_card.url == "http://localhost:8000"
        assert len(server.agent_card.skills) == 1


class TestA2AMethodNames:
    """Test A2A method names match specification."""

    def test_message_send_method(self) -> None:
        """message/send is valid method."""
        request = JSONRPCRequest(
            id="1",
            method="message/send",
            params={},
        )
        assert request.method == "message/send"

    def test_tasks_get_method(self) -> None:
        """tasks/get is valid method."""
        request = JSONRPCRequest(
            id="1",
            method="tasks/get",
            params={},
        )
        assert request.method == "tasks/get"

    def test_tasks_list_method(self) -> None:
        """tasks/list is valid method."""
        request = JSONRPCRequest(
            id="1",
            method="tasks/list",
            params={},
        )
        assert request.method == "tasks/list"

    def test_tasks_cancel_method(self) -> None:
        """tasks/cancel is valid method."""
        request = JSONRPCRequest(
            id="1",
            method="tasks/cancel",
            params={},
        )
        assert request.method == "tasks/cancel"
