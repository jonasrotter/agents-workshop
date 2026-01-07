# Implementation Plan: A2A Server Refactoring to Official SDK

**Version**: 1.0 | **Status**: PLANNED | **Branch**: `main`
**Spec**: [spec.md](./spec.md) | **Tasks**: [tasks.md](./tasks.md)

---

## Technical Context

| Aspect | Value |
|--------|-------|
| **Language** | Python 3.11+ |
| **Framework** | FastAPI, Pydantic v2 |
| **Primary SDK** | `a2a-sdk==0.3.17` with `a2a-server` |
| **Secondary SDK** | `agent-framework-a2a==1.0.0b251120` |
| **Database** | In-memory task store (`InMemoryTaskStore` from SDK) |
| **Project Type** | Single project (workshop modules) |

### Key Dependencies

```toml
# Already in requirements.txt
a2a-sdk==0.3.17
agent-framework-a2a==1.0.0b251120
a2a-server  # Provides A2AFastAPIApplication
```

### Primary File Impact

| File | Action | Rationale |
|------|--------|-----------|
| `src/agents/a2a_server.py` | **MAJOR REFACTOR** | Replace 713-line custom impl with ~150-line SDK integration |
| `src/agents/__init__.py` | UPDATE | Add backward-compatible re-exports |
| `tests/contract/test_a2a_schemas.py` | UPDATE | Use SDK types instead of custom |
| `tests/integration/test_scenario_03.py` | VERIFY | Ensure A2A scenario still passes |
| `notebooks/03_a2a_protocol.ipynb` | VERIFY | Update if needed |

---

## Constitution Check (Post-Design)

| Principle | Status | Notes |
|-----------|--------|-------|
| **I. Type Safety** | ✅ COMPLIANT | SDK types are fully typed Pydantic models |
| **II. Test-First** | ✅ COMPLIANT | Existing tests cover A2A; will update for SDK types |
| **III. Clean Code** | ✅ IMPROVES | 80% code reduction; cleaner abstractions |
| **IV. Dependencies** | ✅ COMPLIANT | Uses already-declared SDK packages |
| **V. Observability** | ✅ COMPLIANT | SDK provides structured error handling |

### Gate Evaluation

- **Type Safety Gate**: ✅ PASS - All SDK types have full type hints
- **Test Coverage Gate**: ✅ PASS - Contract tests exist; integration tests verify scenarios
- **Clean Code Gate**: ✅ PASS - Reduces complexity significantly
- **Dependency Gate**: ✅ PASS - No new dependencies (SDK already in requirements.txt)

---

## Architecture Overview

### Current State (BEFORE)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Custom A2A Implementation                         │
│                        (713 lines)                                   │
├─────────────────────────────────────────────────────────────────────┤
│  Custom Models (~200 lines):                                         │
│  - TaskState, Skill, Provider, AgentCard                            │
│  - Message, TextPart, Artifact                                       │
│  - JSONRPCRequest/Response                                           │
├─────────────────────────────────────────────────────────────────────┤
│  Custom TaskManager (~80 lines):                                     │
│  - In-memory dict storage                                            │
│  - Manual state transitions                                          │
├─────────────────────────────────────────────────────────────────────┤
│  Custom A2AServer (~200 lines):                                      │
│  - Manual FastAPI route setup                                        │
│  - Manual JSON-RPC handling                                          │
│  - Manual Agent Card endpoint                                        │
├─────────────────────────────────────────────────────────────────────┤
│  Request Handlers (~150 lines):                                      │
│  - _handle_rpc(), _handle_message_send()                            │
│  - _handle_get_task(), etc.                                          │
└─────────────────────────────────────────────────────────────────────┘
```

### Target State (AFTER)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SDK-Based A2A Implementation                      │
│                        (~150 lines)                                  │
├─────────────────────────────────────────────────────────────────────┤
│  FROM SDK (a2a.types):                                               │
│  - AgentCard, AgentSkill, AgentCapabilities                         │
│  - Task, TaskState, TaskStatus                                       │
│  - Message, TextPart, Artifact                                       │
│  - All JSON-RPC types                                                │
├─────────────────────────────────────────────────────────────────────┤
│  FROM SDK (a2a.server):                                              │
│  - A2AFastAPIApplication                                             │
│  - InMemoryTaskStore                                                 │
│  - RequestHandler (base interface)                                   │
├─────────────────────────────────────────────────────────────────────┤
│  WORKSHOP CUSTOM (~150 lines):                                       │
│  - WorkshopRequestHandler (wraps our agents)                         │
│  - A2AServer (thin wrapper, convenience)                             │
│  - create_a2a_server() factory function                              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Phase 0: Research ✅ COMPLETE

See [research.md](./research.md) for SDK analysis.

Key findings:
- `a2a.server.apps.A2AFastAPIApplication` handles all route setup
- `a2a.server.request_handlers.RequestHandler` is the interface to implement
- `a2a.server.task_store.InMemoryTaskStore` replaces custom TaskManager
- All Pydantic models available in `a2a.types`

### Phase 1: Design ✅ COMPLETE

Artifacts:
- [data-model.md](./data-model.md) - Type mapping from custom to SDK
- [contracts/a2a-sdk-interfaces.md](./contracts/a2a-sdk-interfaces.md) - Interface contracts
- [quickstart.md](./quickstart.md) - Migration guide

### Phase 2: Implementation Tasks

| ID | Task | Est. | Priority | Dependencies |
|----|------|------|----------|--------------|
| T001 | Create `WorkshopRequestHandler` class | 1h | HIGH | None |
| T002 | Refactor `A2AServer` to use `A2AFastAPIApplication` | 1h | HIGH | T001 |
| T003 | Remove custom Pydantic models | 30m | HIGH | T002 |
| T004 | Update `__init__.py` exports (backward compat) | 15m | MEDIUM | T003 |
| T005 | Update contract tests for SDK types | 30m | MEDIUM | T003 |
| T006 | Verify notebook 03 works | 30m | MEDIUM | T002 |
| T007 | Add `A2AAgent` client demo (optional) | 30m | LOW | T002 |

**Total Estimated Time**: 4-5 hours

---

## Task Details

### T001: Create WorkshopRequestHandler

**File**: `src/agents/a2a_server.py`

**Implementation**:

```python
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
)

class WorkshopRequestHandler(RequestHandler):
    """Request handler that wraps workshop agents."""
    
    def __init__(self, agent: Any, task_store: InMemoryTaskStore):
        self.agent = agent
        self.task_store = task_store
    
    async def on_message_send(
        self,
        request: SendMessageRequest,
        context: ServerCallContext,
    ) -> Task:
        # Create task
        task = Task(
            id=context.task_id,
            context_id=context.context_id,
            status=TaskStatus(state=TaskState.SUBMITTED),
        )
        await self.task_store.create(task)
        
        # Extract text from message
        text = self._extract_text(request.params.message)
        
        # Update to WORKING
        task.status.state = TaskState.WORKING
        await self.task_store.update(task)
        
        # Run agent
        try:
            result = await self.agent.run(text)
            task.artifacts = [
                Artifact(parts=[TextPart(text=result)])
            ]
            task.status.state = TaskState.COMPLETED
        except Exception as e:
            task.status.state = TaskState.FAILED
            task.status.message = str(e)
        
        await self.task_store.update(task)
        return task
    
    async def on_get_task(
        self,
        request: GetTaskRequest,
        context: ServerCallContext,
    ) -> Task:
        task = await self.task_store.get(request.params.id)
        if not task:
            raise TaskNotFoundError(request.params.id)
        return task
    
    async def on_cancel_task(
        self,
        request: CancelTaskRequest,
        context: ServerCallContext,
    ) -> Task:
        task = await self.task_store.get(request.params.id)
        if not task:
            raise TaskNotFoundError(request.params.id)
        task.status.state = TaskState.CANCELED
        await self.task_store.update(task)
        return task
    
    def _extract_text(self, message: Message) -> str:
        for part in message.parts:
            if isinstance(part, TextPart):
                return part.text
        return ""
```

---

### T002: Refactor A2AServer

**File**: `src/agents/a2a_server.py`

**Implementation**:

```python
from a2a.server.apps import A2AFastAPIApplication
from a2a.types import AgentCard, AgentSkill, AgentCapabilities

class A2AServer:
    """A2A server using official SDK."""
    
    def __init__(
        self,
        agent: Any = None,
        name: str = "A2A Agent",
        description: str = "A2A Protocol Agent",
        url: str = "http://localhost:8000",
        version: str = "1.0.0",
        skills: list[AgentSkill] | None = None,
    ):
        self.agent = agent
        self.name = name
        self.description = description
        self.url = url
        self.version = version
        self.skills = skills or []
        self.task_store = InMemoryTaskStore()
    
    @property
    def agent_card(self) -> AgentCard:
        return AgentCard(
            name=self.name,
            description=self.description,
            url=self.url,
            version=self.version,
            skills=self.skills,
            capabilities=AgentCapabilities(streaming=False),
        )
    
    def create_app(self) -> FastAPI:
        handler = WorkshopRequestHandler(self.agent, self.task_store)
        a2a_app = A2AFastAPIApplication(
            agent_card=self.agent_card,
            http_handler=handler,
        )
        return a2a_app.build()


def create_a2a_server(
    agent: Any = None,
    name: str = "A2A Agent",
    description: str = "A2A Protocol Agent",
    url: str = "http://localhost:8000",
    skills: list[AgentSkill] | None = None,
) -> FastAPI:
    """Convenience function to create A2A server."""
    server = A2AServer(
        agent=agent,
        name=name,
        description=description,
        url=url,
        skills=skills,
    )
    return server.create_app()
```

---

### T003: Remove Custom Models

**File**: `src/agents/a2a_server.py`

Delete all custom Pydantic models and replace with imports:

```python
# DELETE these (~200 lines):
# - class TaskState(str, Enum)
# - class Skill(BaseModel)
# - class Provider(BaseModel)
# - class AgentCard(BaseModel)
# - class TextPart(BaseModel)
# - class Message(BaseModel)
# - class Artifact(BaseModel)
# - class TaskStatus(BaseModel)
# - class Task(BaseModel)
# - class JSONRPCRequest(BaseModel)
# - class JSONRPCResponse(BaseModel)
# - class TaskManager

# REPLACE with:
from a2a.types import (
    AgentCard,
    AgentSkill,
    AgentCapabilities,
    Task,
    TaskState,
    TaskStatus,
    Message,
    TextPart,
    Artifact,
    SendMessageRequest,
    GetTaskRequest,
    CancelTaskRequest,
)
from a2a.server.task_store import InMemoryTaskStore
```

---

### T004: Update Exports

**File**: `src/agents/__init__.py`

```python
# A2A Server (SDK-based)
from .a2a_server import (
    A2AServer,
    WorkshopRequestHandler,
    create_a2a_server,
)

# Re-export SDK types for backward compatibility
from a2a.types import (
    AgentCard,
    AgentSkill,
    AgentCapabilities,
    Task,
    TaskState,
    TaskStatus,
    Message,
    TextPart,
    Artifact,
)

__all__ = [
    # A2A
    "A2AServer",
    "WorkshopRequestHandler", 
    "create_a2a_server",
    # SDK Types (re-exports)
    "AgentCard",
    "AgentSkill",
    "Task",
    "TaskState",
    "TaskStatus",
    "Message",
    "TextPart",
    "Artifact",
]
```

---

### T005: Update Contract Tests

**File**: `tests/contract/test_a2a_schemas.py`

Update imports to use SDK types:

```python
# FROM:
from src.agents.a2a_server import (
    AgentCard,
    Task,
    TaskState,
    # ...
)

# TO:
from a2a.types import (
    AgentCard,
    Task,
    TaskState,
    # ...
)
```

---

### T006: Verify Notebook

**File**: `notebooks/03_a2a_protocol.ipynb`

- Run all cells
- Verify agent card endpoint works
- Verify message send works
- Update any custom type references if needed

---

### T007: Add A2AAgent Client Demo (Optional)

Add to notebook or create example:

```python
from agent_framework_a2a import A2AAgent

# Create client to call external A2A agent
client = A2AAgent(agent_url="http://localhost:8001")

# Send message
response = await client.send_message("Hello from workshop!")
print(response)
```

---

## Success Criteria

1. **Code Reduction**: 713 lines → ~150 lines (~80% reduction) ✅
2. **SDK Integration**: All A2A types from `a2a.types` ✅
3. **Tests Pass**: All existing tests pass with SDK types ✅
4. **Notebook Works**: 03_a2a_protocol runs successfully ✅
5. **Backward Compatible**: Existing imports don't break ✅

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| SDK API changes | LOW | MEDIUM | Pin exact version `0.3.17` |
| Breaking changes | MEDIUM | MEDIUM | Preserve interface, add aliases |
| Missing SDK features | LOW | LOW | Custom code if needed |

---

## Rollback Plan

If refactoring fails:
1. Preserve original `a2a_server.py` as `a2a_server_custom.py`
2. Git revert to previous commit
3. Document specific incompatibilities found

---

## Next Steps

1. ✅ Complete Phase 0 (Research)
2. ✅ Complete Phase 1 (Design)
3. ⏳ **Execute Phase 2** (Implementation)
   - Start with T001 (WorkshopRequestHandler)
   - Progress sequentially through T002-T007
4. Run tests after each task
5. Final verification with notebook
