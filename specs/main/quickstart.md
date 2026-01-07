# Quickstart: A2A Server Migration Guide

**Date**: 2026-01-06 | **Plan**: [plan.md](./plan.md)

This guide shows how to migrate from the custom A2A implementation to the official `a2a-sdk`.

---

## Before You Start

Ensure these packages are installed (already in `requirements.txt`):

```bash
pip install a2a-sdk==0.3.17
pip install agent-framework-a2a==1.0.0b251120
```

---

## Migration Examples

### 1. Creating an Agent Card

**Before (Custom)**:
```python
from src.agents.a2a_server import AgentCard, Skill, Capabilities

agent_card = AgentCard(
    name="Research Agent",
    description="Researches topics",
    url="http://localhost:8000",
    version="1.0.0",
    capabilities=Capabilities(streaming=True),
    skills=[
        Skill(
            id="research",
            name="Research",
            description="Research a topic",
        )
    ],
)
```

**After (SDK)**:
```python
from a2a.types import AgentCard, AgentSkill, AgentCapabilities

agent_card = AgentCard(
    name="Research Agent",
    description="Researches topics",
    url="http://localhost:8000",
    version="1.0.0",
    capabilities=AgentCapabilities(streaming=True),
    skills=[
        AgentSkill(  # Note: Skill → AgentSkill
            id="research",
            name="Research",
            description="Research a topic",
        )
    ],
)
```

### 2. Creating a Server

**Before (Custom)**:
```python
from src.agents import A2AServer, create_a2a_server

# Option 1: Class-based
server = A2AServer(
    agent=my_agent,
    name="Research Agent",
    description="Researches topics",
    skills=[Skill(id="research", name="Research", description="...")],
)
app = server.create_app()

# Option 2: Function-based
app = create_a2a_server(
    agent=my_agent,
    name="Research Agent",
    skills=[Skill(id="research", ...)],
)
```

**After (SDK)**:
```python
from a2a.server.apps import A2AFastAPIApplication
from a2a.types import AgentCard, AgentSkill

# Build agent card
agent_card = AgentCard(
    name="Research Agent",
    url="http://localhost:8000",
    version="1.0.0",
    skills=[AgentSkill(id="research", name="Research", description="...")],
)

# Create handler (wraps your agent)
handler = WorkshopRequestHandler(agent=my_agent)

# Build app
app = A2AFastAPIApplication(agent_card, handler).build()
```

**Or use the maintained convenience function** (backward compatible):
```python
from src.agents import create_a2a_server, AgentSkill

app = create_a2a_server(
    agent=my_agent,
    name="Research Agent",
    skills=[AgentSkill(id="research", name="Research", description="...")],
)
```

### 3. Working with Tasks

**Before (Custom)**:
```python
from src.agents.a2a_server import Task, TaskState, TaskStatus, Message, TextPart

task = Task(
    id="task-123",
    contextId="ctx-456",
    status=TaskStatus(state=TaskState.COMPLETED),
    history=[Message(role="agent", parts=[TextPart(text="Done")])],
)
```

**After (SDK)**:
```python
from a2a.types import Task, TaskState, TaskStatus, Message, Role, TextPart

task = Task(
    id="task-123",
    contextId="ctx-456",
    status=TaskStatus(state=TaskState.completed),  # Note: lowercase
    history=[Message(role=Role.agent, parts=[TextPart(text="Done")])],
)
```

### 4. Calling External A2A Agents (NEW!)

The SDK enables calling external A2A agents as clients:

```python
from agent_framework_a2a import A2AAgent

# Connect to an external A2A agent
external_agent = A2AAgent.from_url("https://research-agent.example.com")

# Call it like any agent
result = await external_agent.run("Research quantum computing")

# Or with streaming
async for chunk in external_agent.run_stream("Research AI safety"):
    print(chunk, end="")

# Use as a tool for orchestration
coordinator = ChatAgent(
    tools=[external_agent.as_tool()],
    instructions="Use the research agent to gather information",
)
```

---

## Import Changes Summary

| Before | After |
|--------|-------|
| `from src.agents.a2a_server import Skill` | `from a2a.types import AgentSkill` |
| `from src.agents.a2a_server import Capabilities` | `from a2a.types import AgentCapabilities` |
| `from src.agents.a2a_server import TaskState` | `from a2a.types import TaskState` |
| `from src.agents.a2a_server import TaskManager` | `from a2a.server.task_store import InMemoryTaskStore` |
| `from src.agents.a2a_server import A2AServer` | `from a2a.server.apps import A2AFastAPIApplication` |

**Backward-compatible imports** (maintained in `src/agents/__init__.py`):
```python
# These still work:
from src.agents import (
    AgentCard,
    Skill,  # Alias for AgentSkill
    Task,
    TaskState,
    Message,
    TextPart,
    create_a2a_server,  # Convenience function
)
```

---

## Key Differences

| Aspect | Custom Implementation | SDK |
|--------|----------------------|-----|
| Task state values | `TaskState.COMPLETED` (uppercase) | `TaskState.completed` (lowercase) |
| Skill class name | `Skill` | `AgentSkill` |
| Message role | String (`"agent"`) | Enum (`Role.agent`) |
| App creation | `server.create_app()` | `A2AFastAPIApplication(...).build()` |
| Request handling | Custom `_handle_*` methods | `RequestHandler` interface |
| Task storage | Custom `TaskManager` | `TaskStore` interface |

---

## Running the Server

```python
import uvicorn
from src.agents import create_a2a_server, AgentSkill

# Create your agent
from src.agents import ResearchAgent
agent = ResearchAgent()

# Create A2A server
app = create_a2a_server(
    agent=agent,
    name="Research Agent",
    description="Researches topics using web search",
    skills=[
        AgentSkill(
            id="research_topic",
            name="Research Topic",
            description="Research a topic and return findings with sources",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "depth": {"type": "string", "enum": ["brief", "detailed"]},
                },
                "required": ["topic"],
            },
        ),
    ],
)

# Run with uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## Verification

After migration, verify:

1. **Agent Card endpoint**: `GET /.well-known/agent-card.json`
2. **Health endpoint**: `GET /health` (if maintained)
3. **JSON-RPC endpoint**: `POST /` with message/send method
4. **Tests pass**: `pytest tests/contract/test_a2a_schemas.py`
5. **Notebook works**: Run `notebooks/03_a2a_protocol.ipynb`
