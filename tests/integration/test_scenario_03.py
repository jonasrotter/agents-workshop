"""
Integration tests for Scenario 03: A2A Protocol.

Tests the complete A2A workflow including:
- Server creation and configuration
- Agent Card endpoint
- JSON-RPC message handling
- Task lifecycle management
- Error handling
"""

import pytest
from fastapi.testclient import TestClient

from src.agents import (
    A2AServer,
    create_a2a_server,
    AgentCard,
    Skill,
    TaskState,
    Task,
)
from src.agents.a2a_server import (
    JSONRPCRequest,
    JSONRPCResponse,
    Message,
    TextPart,
    TaskManager,
    ErrorCode,
)


class TestScenario03Imports:
    """Test that all Scenario 03 components are importable."""

    def test_import_a2a_server(self) -> None:
        """A2AServer is importable."""
        assert A2AServer is not None

    def test_import_create_function(self) -> None:
        """create_a2a_server factory is importable."""
        assert create_a2a_server is not None

    def test_import_agent_card(self) -> None:
        """AgentCard is importable."""
        assert AgentCard is not None

    def test_import_skill(self) -> None:
        """Skill is importable."""
        assert Skill is not None

    def test_import_task_state(self) -> None:
        """TaskState is importable."""
        assert TaskState is not None

    def test_import_task(self) -> None:
        """Task is importable."""
        assert Task is not None


class TestA2AServerCreation:
    """Test A2A server creation."""

    def test_create_server_minimal(self) -> None:
        """Create server with minimal configuration."""
        server = A2AServer(name="Test Agent")
        
        assert server.agent_card.name == "Test Agent"

    def test_create_server_with_skills(self) -> None:
        """Create server with skills."""
        skills = [
            Skill(id="skill1", name="Skill 1"),
            Skill(id="skill2", name="Skill 2"),
        ]
        server = A2AServer(name="Test Agent", skills=skills)
        
        assert len(server.agent_card.skills) == 2

    def test_create_server_factory(self) -> None:
        """Create server using factory function."""
        app = create_a2a_server(
            name="Factory Agent",
            description="Created via factory",
        )
        
        assert app is not None

    def test_server_creates_app(self) -> None:
        """Server creates FastAPI app."""
        server = A2AServer(name="Test Agent")
        app = server.create_app()
        
        # Verify it's a FastAPI app
        assert hasattr(app, "routes")


class TestHealthEndpoint:
    """Test health check endpoint."""

    @pytest.fixture
    def client(self) -> TestClient:
        """Create test client."""
        app = create_a2a_server(name="Test Agent")
        return TestClient(app)

    def test_health_returns_200(self, client: TestClient) -> None:
        """Health endpoint returns 200."""
        response = client.get("/health")
        
        assert response.status_code == 200

    def test_health_returns_healthy(self, client: TestClient) -> None:
        """Health endpoint returns healthy status."""
        response = client.get("/health")
        
        assert response.json()["status"] == "healthy"


class TestAgentCardEndpoint:
    """Test Agent Card endpoint."""

    @pytest.fixture
    def client(self) -> TestClient:
        """Create test client with skills."""
        app = create_a2a_server(
            name="Test Agent",
            description="A test agent for testing",
            url="http://localhost:8000",
            skills=[
                Skill(id="test", name="Test Skill", description="For testing"),
            ],
        )
        return TestClient(app)

    def test_agent_card_returns_200(self, client: TestClient) -> None:
        """Agent Card endpoint returns 200."""
        response = client.get("/.well-known/agent-card.json")
        
        assert response.status_code == 200

    def test_agent_card_content_type(self, client: TestClient) -> None:
        """Agent Card has JSON content type."""
        response = client.get("/.well-known/agent-card.json")
        
        assert "application/json" in response.headers["content-type"]

    def test_agent_card_has_name(self, client: TestClient) -> None:
        """Agent Card contains name."""
        response = client.get("/.well-known/agent-card.json")
        card = response.json()
        
        assert card["name"] == "Test Agent"

    def test_agent_card_has_url(self, client: TestClient) -> None:
        """Agent Card contains URL."""
        response = client.get("/.well-known/agent-card.json")
        card = response.json()
        
        assert card["url"] == "http://localhost:8000"

    def test_agent_card_has_skills(self, client: TestClient) -> None:
        """Agent Card contains skills."""
        response = client.get("/.well-known/agent-card.json")
        card = response.json()
        
        assert len(card["skills"]) == 1
        assert card["skills"][0]["id"] == "test"

    def test_agent_card_has_capabilities(self, client: TestClient) -> None:
        """Agent Card contains capabilities."""
        response = client.get("/.well-known/agent-card.json")
        card = response.json()
        
        assert "capabilities" in card


class TestJSONRPCEndpoint:
    """Test JSON-RPC endpoint."""

    @pytest.fixture
    def client(self) -> TestClient:
        """Create test client."""
        app = create_a2a_server(name="Test Agent")
        return TestClient(app)

    def test_rpc_endpoint_accepts_post(self, client: TestClient) -> None:
        """RPC endpoint accepts POST requests."""
        request = JSONRPCRequest(
            id="1",
            method="message/send",
            params={
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "Hello"}],
                    "messageId": "msg-1",
                },
                "contextId": "ctx-1",
            },
        )
        
        response = client.post("/", json=request.model_dump())
        
        assert response.status_code == 200

    def test_rpc_response_format(self, client: TestClient) -> None:
        """RPC response has correct format."""
        request = JSONRPCRequest(
            id="test-id",
            method="message/send",
            params={
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "Hello"}],
                    "messageId": "msg-1",
                },
            },
        )
        
        response = client.post("/", json=request.model_dump())
        data = response.json()
        
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == "test-id"


class TestMessageSendMethod:
    """Test message/send JSON-RPC method."""

    @pytest.fixture
    def client(self) -> TestClient:
        """Create test client."""
        app = create_a2a_server(name="Test Agent")
        return TestClient(app)

    def test_message_send_creates_task(self, client: TestClient) -> None:
        """message/send creates a task."""
        request = JSONRPCRequest(
            id="1",
            method="message/send",
            params={
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "Test message"}],
                    "messageId": "msg-1",
                },
                "contextId": "ctx-1",
            },
        )
        
        response = client.post("/", json=request.model_dump())
        data = response.json()
        
        assert "result" in data
        assert "id" in data["result"]

    def test_message_send_returns_task_status(self, client: TestClient) -> None:
        """message/send returns task with status."""
        request = JSONRPCRequest(
            id="1",
            method="message/send",
            params={
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "Test"}],
                    "messageId": "msg-1",
                },
            },
        )
        
        response = client.post("/", json=request.model_dump())
        data = response.json()
        
        assert "status" in data["result"]
        assert "state" in data["result"]["status"]


class TestTasksGetMethod:
    """Test tasks/get JSON-RPC method."""

    @pytest.fixture
    def client_with_task(self) -> tuple[TestClient, str]:
        """Create client and a task."""
        app = create_a2a_server(name="Test Agent")
        client = TestClient(app)
        
        # Create a task first
        create_request = JSONRPCRequest(
            id="create",
            method="message/send",
            params={
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "Test"}],
                    "messageId": "msg-1",
                },
                "contextId": "ctx-test",
            },
        )
        response = client.post("/", json=create_request.model_dump())
        task_id = response.json()["result"]["id"]
        
        return client, task_id

    def test_get_existing_task(self, client_with_task: tuple[TestClient, str]) -> None:
        """tasks/get returns existing task."""
        client, task_id = client_with_task
        
        request = JSONRPCRequest(
            id="get",
            method="tasks/get",
            params={"id": task_id},
        )
        
        response = client.post("/", json=request.model_dump())
        data = response.json()
        
        assert "result" in data
        assert data["result"]["id"] == task_id

    def test_get_nonexistent_task(self) -> None:
        """tasks/get returns error for nonexistent task."""
        app = create_a2a_server(name="Test Agent")
        client = TestClient(app)
        
        request = JSONRPCRequest(
            id="get",
            method="tasks/get",
            params={"id": "nonexistent-task"},
        )
        
        response = client.post("/", json=request.model_dump())
        data = response.json()
        
        assert "error" in data
        assert data["error"]["code"] == ErrorCode.TASK_NOT_FOUND


class TestTasksListMethod:
    """Test tasks/list JSON-RPC method."""

    @pytest.fixture
    def client(self) -> TestClient:
        """Create test client."""
        app = create_a2a_server(name="Test Agent")
        return TestClient(app)

    def test_list_empty(self, client: TestClient) -> None:
        """tasks/list returns empty list initially."""
        request = JSONRPCRequest(
            id="list",
            method="tasks/list",
            params={"contextId": "ctx-empty"},
        )
        
        response = client.post("/", json=request.model_dump())
        data = response.json()
        
        assert "result" in data
        assert data["result"]["tasks"] == []

    def test_list_with_tasks(self, client: TestClient) -> None:
        """tasks/list returns created tasks."""
        # Create tasks
        for i in range(3):
            request = JSONRPCRequest(
                id=f"create-{i}",
                method="message/send",
                params={
                    "message": {
                        "role": "user",
                        "parts": [{"kind": "text", "text": f"Task {i}"}],
                        "messageId": f"msg-{i}",
                    },
                    "contextId": "ctx-list",
                },
            )
            client.post("/", json=request.model_dump())
        
        # List tasks
        list_request = JSONRPCRequest(
            id="list",
            method="tasks/list",
            params={"contextId": "ctx-list"},
        )
        
        response = client.post("/", json=list_request.model_dump())
        data = response.json()
        
        assert len(data["result"]["tasks"]) == 3


class TestTasksCancelMethod:
    """Test tasks/cancel JSON-RPC method."""

    @pytest.fixture
    def client_with_task(self) -> tuple[TestClient, str]:
        """Create client and a task."""
        app = create_a2a_server(name="Test Agent")
        client = TestClient(app)
        
        create_request = JSONRPCRequest(
            id="create",
            method="message/send",
            params={
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "Test"}],
                    "messageId": "msg-1",
                },
            },
        )
        response = client.post("/", json=create_request.model_dump())
        task_id = response.json()["result"]["id"]
        
        return client, task_id

    def test_cancel_task(self, client_with_task: tuple[TestClient, str]) -> None:
        """tasks/cancel cancels existing task."""
        client, task_id = client_with_task
        
        request = JSONRPCRequest(
            id="cancel",
            method="tasks/cancel",
            params={"id": task_id},
        )
        
        response = client.post("/", json=request.model_dump())
        data = response.json()
        
        assert "result" in data
        assert data["result"]["status"]["state"] == "cancelled"


class TestMethodNotFound:
    """Test method not found handling."""

    @pytest.fixture
    def client(self) -> TestClient:
        """Create test client."""
        app = create_a2a_server(name="Test Agent")
        return TestClient(app)

    def test_unknown_method_returns_error(self, client: TestClient) -> None:
        """Unknown method returns method not found error."""
        request = JSONRPCRequest(
            id="1",
            method="unknown/method",
            params={},
        )
        
        response = client.post("/", json=request.model_dump())
        data = response.json()
        
        assert "error" in data
        assert data["error"]["code"] == ErrorCode.METHOD_NOT_FOUND


class TestTaskManagerIntegration:
    """Test TaskManager integration."""

    def test_manager_tracks_tasks(self) -> None:
        """TaskManager tracks created tasks."""
        manager = TaskManager()
        
        message = Message(
            role="user",
            parts=[TextPart(text="Test")],
            messageId="msg-1",
        )
        
        task1 = manager.create_task("ctx-1", message)
        task2 = manager.create_task("ctx-1", message)
        task3 = manager.create_task("ctx-2", message)
        
        assert len(manager.list_tasks()) == 3
        assert len(manager.list_tasks(context_id="ctx-1")) == 2
        assert len(manager.list_tasks(context_id="ctx-2")) == 1

    def test_manager_state_transitions(self) -> None:
        """TaskManager handles state transitions."""
        manager = TaskManager()
        
        message = Message(
            role="user",
            parts=[TextPart(text="Test")],
            messageId="msg-1",
        )
        
        task = manager.create_task("ctx-1", message)
        assert task.status.state == TaskState.SUBMITTED
        
        manager.update_task_status(task.id, TaskState.WORKING)
        updated = manager.get_task(task.id)
        assert updated.status.state == TaskState.WORKING
        
        manager.update_task_status(task.id, TaskState.COMPLETED)
        final = manager.get_task(task.id)
        assert final.status.state == TaskState.COMPLETED


class TestTelemetryIntegration:
    """Test telemetry integration."""

    def test_server_has_tracer(self) -> None:
        """Server initializes with tracer."""
        server = A2AServer(name="Test Agent")
        
        # Server should have tracer attribute
        assert hasattr(server, "tracer")


class TestScenario03EndToEnd:
    """End-to-end tests for Scenario 03."""

    def test_full_workflow(self) -> None:
        """Test complete A2A workflow."""
        # 1. Create server
        app = create_a2a_server(
            name="E2E Agent",
            description="End-to-end test agent",
            skills=[
                Skill(id="process", name="Process", description="Process data"),
            ],
        )
        client = TestClient(app)
        
        # 2. Check health
        health = client.get("/health")
        assert health.status_code == 200
        
        # 3. Get Agent Card
        card = client.get("/.well-known/agent-card.json")
        assert card.json()["name"] == "E2E Agent"
        
        # 4. Send message
        send_request = JSONRPCRequest(
            id="e2e-1",
            method="message/send",
            params={
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "Process this data"}],
                    "messageId": "msg-e2e",
                },
                "contextId": "ctx-e2e",
            },
        )
        send_response = client.post("/", json=send_request.model_dump())
        task_id = send_response.json()["result"]["id"]
        
        # 5. Get task
        get_request = JSONRPCRequest(
            id="e2e-2",
            method="tasks/get",
            params={"id": task_id},
        )
        get_response = client.post("/", json=get_request.model_dump())
        assert get_response.json()["result"]["id"] == task_id
        
        # 6. List tasks
        list_request = JSONRPCRequest(
            id="e2e-3",
            method="tasks/list",
            params={"contextId": "ctx-e2e"},
        )
        list_response = client.post("/", json=list_request.model_dump())
        assert len(list_response.json()["result"]["tasks"]) >= 1
        
        # 7. Cancel task
        cancel_request = JSONRPCRequest(
            id="e2e-4",
            method="tasks/cancel",
            params={"id": task_id},
        )
        cancel_response = client.post("/", json=cancel_request.model_dump())
        assert cancel_response.json()["result"]["status"]["state"] == "cancelled"
