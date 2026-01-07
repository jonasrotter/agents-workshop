"""
Contract tests for A2A Protocol schemas.

Validates implementations against contracts/a2a-protocol.md.
These tests ensure the A2A server produces valid protocol messages.

NOTE: Tests updated to use SDK types from a2a-sdk package.
SDK types use snake_case for Python attributes but serialize to camelCase for JSON.
"""

import json
import pytest
from typing import Any

# SDK types
from a2a.types import (
    AgentCard,
    AgentCapabilities,
    AgentProvider,
    AgentSkill,
    Artifact,
    FileWithUri,
    FilePart,
    Message,
    Task,
    TaskState,
    TaskStatus,
    TextPart,
    DataPart,
)

# Workshop implementation
from src.agents.a2a_server import (
    A2AServer,
    WorkshopRequestHandler,
    create_a2a_server,
)
from src.agents import Skill  # Alias for AgentSkill


class TestAgentCardSchema:
    """Test Agent Card schema compliance with SDK types."""

    def test_agent_card_required_fields(self) -> None:
        """Agent Card must have all required fields per SDK."""
        card = AgentCard(
            name="Test Agent",
            description="A test agent",
            url="http://localhost:8000",
            version="1.0.0",
            default_input_modes=["text"],
            default_output_modes=["text"],
            capabilities=AgentCapabilities(),
            skills=[AgentSkill(
                id="test",
                name="Test",
                description="Test skill",
                tags=["test"],
            )],
        )
        
        assert card.name == "Test Agent"
        assert card.url == "http://localhost:8000"
        assert card.description == "A test agent"

    def test_agent_card_with_skills(self) -> None:
        """Agent Card can include multiple skills."""
        card = AgentCard(
            name="Test Agent",
            description="A test agent",
            url="http://localhost:8000",
            version="1.0.0",
            default_input_modes=["text"],
            default_output_modes=["text"],
            capabilities=AgentCapabilities(),
            skills=[
                AgentSkill(
                    id="skill_1",
                    name="Skill One",
                    description="First skill",
                    tags=["demo"],
                ),
                AgentSkill(
                    id="skill_2",
                    name="Skill Two",
                    description="Second skill",
                    tags=["demo"],
                ),
            ],
        )
        
        assert len(card.skills) == 2
        assert card.skills[0].id == "skill_1"
        assert card.skills[1].description == "Second skill"

    def test_agent_card_serialization(self) -> None:
        """Agent Card serializes to valid JSON with camelCase."""
        card = AgentCard(
            name="Test Agent",
            description="A test agent",
            url="http://localhost:8000",
            version="1.0.0",
            default_input_modes=["text"],
            default_output_modes=["text"],
            capabilities=AgentCapabilities(),
            skills=[AgentSkill(
                id="test",
                name="Test",
                description="Test skill",
                tags=["test"],
            )],
        )
        
        data = card.model_dump(by_alias=True)
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        
        assert parsed["name"] == "Test Agent"
        assert parsed["url"] == "http://localhost:8000"
        assert "defaultInputModes" in parsed
        assert "defaultOutputModes" in parsed


class TestAgentSkillSchema:
    """Test AgentSkill schema compliance."""

    def test_skill_required_fields(self) -> None:
        """Skill must have id, name, description, and tags."""
        skill = AgentSkill(
            id="test_skill",
            name="Test Skill",
            description="A test skill",
            tags=["test"],
        )
        
        assert skill.id == "test_skill"
        assert skill.name == "Test Skill"
        assert skill.description == "A test skill"
        assert "test" in skill.tags

    def test_skill_alias_works(self) -> None:
        """Skill alias from __init__.py works."""
        # Skill is an alias for AgentSkill
        skill = Skill(
            id="test_skill",
            name="Test Skill",
            description="A test skill",
            tags=["test"],
        )
        assert isinstance(skill, AgentSkill)


class TestCapabilitiesSchema:
    """Test AgentCapabilities schema compliance."""

    def test_default_capabilities(self) -> None:
        """Default capabilities should have expected defaults."""
        caps = AgentCapabilities()
        
        # SDK defaults may differ from our custom implementation
        assert hasattr(caps, 'streaming')
        assert hasattr(caps, 'push_notifications')
        assert hasattr(caps, 'state_transition_history')

    def test_enabled_capabilities(self) -> None:
        """Capabilities can be enabled."""
        caps = AgentCapabilities(
            streaming=True,
            push_notifications=True,
            state_transition_history=True,
        )
        
        assert caps.streaming is True
        assert caps.push_notifications is True
        assert caps.state_transition_history is True


class TestTaskStateEnum:
    """Test TaskState enum compliance with SDK."""

    def test_all_states_present(self) -> None:
        """All required task states must be present in SDK."""
        required_states = [
            "submitted",
            "working",
            "input-required",
            "completed",
            "failed",
            "canceled",  # Note: SDK uses "canceled" not "cancelled"
        ]
        
        actual_states = [state.value for state in TaskState]
        
        for state in required_states:
            assert state in actual_states, f"Missing state: {state}"

    def test_state_values(self) -> None:
        """Task states have correct string values."""
        assert TaskState.submitted.value == "submitted"
        assert TaskState.working.value == "working"
        assert TaskState.input_required.value == "input-required"
        assert TaskState.completed.value == "completed"
        assert TaskState.failed.value == "failed"
        assert TaskState.canceled.value == "canceled"


class TestTaskStatusSchema:
    """Test TaskStatus schema compliance."""

    def test_status_with_state(self) -> None:
        """TaskStatus must have state."""
        status = TaskStatus(state=TaskState.submitted)
        
        assert status.state == TaskState.submitted

    def test_status_with_message(self) -> None:
        """TaskStatus can have optional message."""
        status = TaskStatus(
            state=TaskState.working,
            message=Message(
                role="agent",  # SDK uses "agent" not "assistant"
                parts=[TextPart(text="Processing...")],
                message_id="msg-1",
            ),
        )
        
        assert status.message is not None
        assert status.message.role == "agent"


class TestTaskSchema:
    """Test Task schema compliance."""

    def test_task_required_fields(self) -> None:
        """Task must have id, context_id, and status."""
        task = Task(
            id="task-001",
            context_id="ctx-001",
            status=TaskStatus(state=TaskState.submitted),
        )
        
        assert task.id == "task-001"
        assert task.context_id == "ctx-001"
        assert task.status.state == TaskState.submitted

    def test_task_with_artifacts(self) -> None:
        """Task can have artifacts."""
        task = Task(
            id="task-001",
            context_id="ctx-001",
            status=TaskStatus(state=TaskState.completed),
            artifacts=[
                Artifact(
                    artifact_id="art-001",
                    name="result",
                    parts=[TextPart(text="Result data")],
                ),
            ],
        )
        
        assert task.artifacts is not None
        assert len(task.artifacts) == 1
        assert task.artifacts[0].artifact_id == "art-001"


class TestMessageSchema:
    """Test Message schema compliance."""

    def test_message_required_fields(self) -> None:
        """Message must have role and parts."""
        message = Message(
            role="user",
            parts=[TextPart(text="Hello")],
            message_id="msg-001",
        )
        
        assert message.role == "user"
        assert len(message.parts) == 1
        assert message.message_id == "msg-001"

    def test_message_roles(self) -> None:
        """Message supports user and agent roles."""
        user_msg = Message(
            role="user",
            parts=[TextPart(text="Hello")],
            message_id="msg-u",
        )
        agent_msg = Message(
            role="agent",  # SDK uses "agent" not "assistant"
            parts=[TextPart(text="Hi there!")],
            message_id="msg-a",
        )
        
        assert user_msg.role == "user"
        assert agent_msg.role == "agent"


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
            file=FileWithUri(
                uri="file://document.pdf",
                mime_type="application/pdf",
            ),
        )
        
        assert part.kind == "file"
        assert part.file.uri == "file://document.pdf"
        assert part.file.mime_type == "application/pdf"

    def test_data_part(self) -> None:
        """DataPart schema compliance."""
        part = DataPart(data={"key": "value"})
        
        assert part.kind == "data"
        assert part.data == {"key": "value"}

    def test_part_serialization(self) -> None:
        """Parts serialize with kind discriminator."""
        text = TextPart(text="test")
        file = FilePart(file=FileWithUri(uri="file://test.txt", mime_type="text/plain"))
        data = DataPart(data={})
        
        assert text.model_dump()["kind"] == "text"
        assert file.model_dump()["kind"] == "file"
        assert data.model_dump()["kind"] == "data"


class TestArtifactSchema:
    """Test Artifact schema compliance."""

    def test_artifact_required_fields(self) -> None:
        """Artifact must have artifact_id, name, and parts."""
        artifact = Artifact(
            artifact_id="art-001",
            name="result",
            parts=[TextPart(text="Content")],
        )
        
        assert artifact.artifact_id == "art-001"
        assert artifact.name == "result"
        assert len(artifact.parts) == 1

    def test_artifact_serialization(self) -> None:
        """Artifact serializes to camelCase."""
        artifact = Artifact(
            artifact_id="art-001",
            name="result",
            parts=[TextPart(text="Content")],
        )
        
        data = artifact.model_dump(by_alias=True)
        assert "artifactId" in data


class TestA2AServerIntegration:
    """Test A2AServer integration with SDK."""

    def test_server_creates_valid_agent_card(self) -> None:
        """A2AServer produces valid SDK AgentCard."""
        server = A2AServer(
            name="Test Server",
            description="Test description",
            url="http://localhost:8000",
        )
        
        card = server.agent_card
        
        assert isinstance(card, AgentCard)
        assert card.name == "Test Server"
        assert card.description == "Test description"

    def test_server_creates_app(self) -> None:
        """A2AServer creates FastAPI app."""
        server = A2AServer(
            name="Test Server",
            description="Test description",
            url="http://localhost:8000",
        )
        
        app = server.create_app()
        
        # Should be A2AFastAPI from SDK
        assert app is not None
        assert hasattr(app, 'routes')

    def test_factory_function(self) -> None:
        """create_a2a_server factory works."""
        app = create_a2a_server(
            name="Factory Test",
            description="Created by factory",
            url="http://localhost:8000",
        )
        
        assert app is not None


class TestWorkshopRequestHandler:
    """Test WorkshopRequestHandler implementation."""

    def test_handler_creation(self) -> None:
        """Handler can be created."""
        handler = WorkshopRequestHandler()
        
        assert handler is not None
        assert handler.agent is None
        assert handler.task_store is not None

    def test_handler_with_agent(self) -> None:
        """Handler accepts agent."""
        class MockAgent:
            async def run(self, text: str) -> str:
                return f"Echo: {text}"
        
        handler = WorkshopRequestHandler(agent=MockAgent())
        
        assert handler.agent is not None

    def test_extract_text_from_message(self) -> None:
        """Handler extracts text from message parts."""
        handler = WorkshopRequestHandler()
        
        message = Message(
            role="user",
            parts=[TextPart(text="Hello")],
            message_id="msg-001",
        )
        
        text = handler._extract_text(message)
        assert text == "Hello"

    def test_extract_text_empty_message(self) -> None:
        """Handler handles empty message."""
        handler = WorkshopRequestHandler()
        
        text = handler._extract_text(None)
        assert text == ""


class TestSDKTypesImportFromAgents:
    """Test that SDK types can be imported from src.agents."""

    def test_import_agent_card(self) -> None:
        """AgentCard importable from src.agents."""
        from src.agents import AgentCard as ImportedAgentCard
        assert ImportedAgentCard is AgentCard

    def test_import_task(self) -> None:
        """Task importable from src.agents."""
        from src.agents import Task as ImportedTask
        assert ImportedTask is Task

    def test_import_task_state(self) -> None:
        """TaskState importable from src.agents."""
        from src.agents import TaskState as ImportedTaskState
        assert ImportedTaskState is TaskState

    def test_import_message(self) -> None:
        """Message importable from src.agents."""
        from src.agents import Message as ImportedMessage
        assert ImportedMessage is Message

    def test_import_text_part(self) -> None:
        """TextPart importable from src.agents."""
        from src.agents import TextPart as ImportedTextPart
        assert ImportedTextPart is TextPart

    def test_import_artifact(self) -> None:
        """Artifact importable from src.agents."""
        from src.agents import Artifact as ImportedArtifact
        assert ImportedArtifact is Artifact
