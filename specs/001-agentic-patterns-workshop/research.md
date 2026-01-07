# Research: Agentic AI Patterns Workshop

**Date**: 2026-01-01 | **Plan**: [plan.md](./plan.md) | **Spec**: [spec.md](./spec.md)

This document consolidates research findings for all technology decisions in the Agentic AI Patterns Workshop.

---

## 1. Microsoft Agent Framework

### Decision
Use Microsoft Agent Framework (`agent-framework` Python package) for building agents with Azure OpenAI.

### Rationale
- **Official Microsoft SDK**: First-party support with Azure AI Agent Service
- **Python-native**: Pythonic async/await patterns align with workshop's Python 3.11+ requirement
- **Built-in protocols**: Native AG-UI endpoint support via `add_agent_framework_fastapi_endpoint()`
- **Tool integration**: Decorator-based `@ai_function` for easy tool definition
- **Middleware support**: Extensible pipeline for security, logging, telemetry

### Alternatives Considered
| Alternative | Reason Rejected |
|-------------|-----------------|
| Semantic Kernel | More complex, .NET-focused despite Python support |
| LangChain | Third-party, less Azure-native integration |
| AutoGen | Microsoft but more research-focused, less production patterns |

### Key Patterns

#### Agent Creation with Azure OpenAI
```python
from agent_framework.azure import AzureOpenAIResponsesClient
from azure.identity import AzureCliCredential

agent = AzureOpenAIResponsesClient(
    credential=AzureCliCredential()
).create_agent(
    name="WorkshopAgent",
    instructions="You are a helpful assistant."
)

# Non-streaming
result = await agent.run("Hello!")

# Streaming
async for chunk in agent.run_stream("Hello!"):
    if chunk.text:
        print(chunk.text, end="", flush=True)
```

#### Tool Definition
```python
from typing import Annotated
from pydantic import Field

def get_weather(
    location: Annotated[str, Field(description="City name")],
) -> str:
    """Get the weather for a given location."""
    return f"The weather in {location} is sunny."

# Attach tools to ChatAgent
from agent_framework import ChatAgent
from agent_framework.openai import OpenAIChatClient

agent = ChatAgent(
    chat_client=OpenAIChatClient(),
    instructions="You are a helpful assistant.",
    tools=[get_weather]
)
```

#### AG-UI FastAPI Endpoint
```python
from agent_framework.ag_ui import add_agent_framework_fastapi_endpoint
from fastapi import FastAPI

app = FastAPI()
add_agent_framework_fastapi_endpoint(app, agent, "/")
```

### Dependencies
- `agent-framework` (Microsoft Agent Framework)
- `azure-ai-projects` (Azure AI Agent Service client)
- `azure-identity` (Azure CLI/DefaultAzureCredential auth)

### References
- [Microsoft Agent Framework GitHub](https://github.com/microsoft/agent-framework)
- [Azure AI Foundry Documentation](https://learn.microsoft.com/azure/ai-foundry)

---

## 2. Model Context Protocol (MCP)

### Decision
Use MCP Python SDK (`mcp`) for building tool servers that agents can connect to.

### Rationale
- **Open standard**: Anthropic-created protocol adopted across industry
- **Microsoft support**: Microsoft MCP servers available for Azure services
- **Tool/Resource separation**: Clean abstraction for tools (actions) vs resources (data)
- **Transport agnostic**: Supports stdio, HTTP, SSE transports
- **Type-safe**: Pydantic-based schemas with validation

### Alternatives Considered
| Alternative | Reason Rejected |
|-------------|-----------------|
| Custom HTTP APIs | No standardized discovery/schema mechanism |
| OpenAI function calling only | Limited to single vendor, no external tool servers |
| gRPC | More complex, less AI-ecosystem adoption |

### Key Patterns

#### FastMCP Server with Tools and Resources
```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Workshop Tools", json_response=True)

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.resource("config://{key}")
def get_config(key: str) -> str:
    """Get configuration value"""
    return f"Value for {key}"

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
```

#### Low-Level Server with Lifespan Management
```python
import mcp.server.stdio
import mcp.types as types
from mcp.server.lowlevel import Server

server = Server("example-server")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="calculate",
            description="Perform calculations",
            inputSchema={
                "type": "object",
                "properties": {
                    "operation": {"type": "string"},
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["operation", "a", "b"]
            }
        )
    ]

@server.call_tool()
async def handle_tool(name: str, arguments: dict) -> str:
    if name == "calculate":
        return str(arguments["a"] + arguments["b"])
    raise ValueError(f"Unknown tool: {name}")
```

#### MCP Client Usage
```python
from mcp import ClientSession
from mcp.types import AnyUrl

# List available tools
tools = await session.list_tools()
print([t.name for t in tools.tools])

# Call a tool
result = await session.call_tool("add", arguments={"a": 5, "b": 3})

# Read a resource
content = await session.read_resource(AnyUrl("config://api_key"))
```

### Dependencies
- `mcp` (MCP Python SDK)

### References
- [MCP Specification](https://modelcontextprotocol.io/specification)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [Microsoft MCP Servers](https://github.com/microsoft/mcp)

---

## 3. AG-UI Protocol (Agent User Interaction)

### Decision
Use AG-UI protocol via `ag-ui-core` package for streaming chat interfaces.

### Rationale
- **Event-driven streaming**: Real-time token streaming with structured events
- **Human-in-the-loop**: Native support for tool confirmation, interruptions
- **Framework agnostic**: Works with any frontend (React, Vue, vanilla JS)
- **CopilotKit integration**: Proven ecosystem with production implementations
- **Microsoft Agent Framework native**: Built-in `add_agent_framework_fastapi_endpoint()`

### Alternatives Considered
| Alternative | Reason Rejected |
|-------------|-----------------|
| Raw SSE | No structured event types, harder to handle tool calls |
| WebSocket | More complex state management for request-response |
| OpenAI streaming format | Vendor-specific, limited event types |

### Key Patterns

#### AG-UI Event Types
```python
from ag_ui.core import (
    EventType,
    RunStartedEvent,
    RunFinishedEvent,
    TextMessageStartEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    ToolCallStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
)
```

#### FastAPI AG-UI Server
```python
import uuid
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from ag_ui.core import RunAgentInput, EventType
from ag_ui.encoder import EventEncoder

app = FastAPI()

@app.post("/")
async def agentic_chat(input_data: RunAgentInput, request: Request):
    accept_header = request.headers.get("accept")
    encoder = EventEncoder(accept=accept_header)

    async def event_generator():
        # Run started
        yield encoder.encode({
            "type": EventType.RUN_STARTED,
            "thread_id": input_data.thread_id,
            "run_id": input_data.run_id
        })

        message_id = str(uuid.uuid4())

        # Message start
        yield encoder.encode({
            "type": EventType.TEXT_MESSAGE_START,
            "message_id": message_id,
            "role": "assistant"
        })

        # Stream content
        for chunk in ["Hello", " ", "World", "!"]:
            yield encoder.encode({
                "type": EventType.TEXT_MESSAGE_CONTENT,
                "message_id": message_id,
                "delta": chunk
            })

        # Message end
        yield encoder.encode({
            "type": EventType.TEXT_MESSAGE_END,
            "message_id": message_id
        })

        # Run finished
        yield encoder.encode({
            "type": EventType.RUN_FINISHED,
            "thread_id": input_data.thread_id,
            "run_id": input_data.run_id
        })

    return StreamingResponse(
        event_generator(),
        media_type=encoder.get_content_type()
    )
```

#### Simplified with Microsoft Agent Framework
```python
from agent_framework import ChatAgent
from agent_framework.azure import AzureOpenAIChatClient
from agent_framework.ag_ui import add_agent_framework_fastapi_endpoint
from fastapi import FastAPI

agent = ChatAgent(
    name="AGUIAssistant",
    instructions="You are helpful.",
    chat_client=AzureOpenAIChatClient(...)
)

app = FastAPI()
add_agent_framework_fastapi_endpoint(app, agent, "/")
```

### Dependencies
- `ag-ui-core` (AG-UI Python SDK)
- `fastapi` (HTTP framework)
- `uvicorn` (ASGI server)

### References
- [AG-UI Documentation](https://docs.ag-ui.com)
- [AG-UI Protocol Specification](https://ag-ui-protocol.github.io/ag-ui)

---

## 4. A2A Protocol (Agent-to-Agent)

### Decision
Use Google's A2A protocol for exposing agents to other agents.

### Rationale
- **Open interoperability**: Agents from different vendors can communicate
- **Capability discovery**: Agent Cards advertise what an agent can do
- **Task-oriented**: Long-running tasks with status tracking
- **Enterprise-ready**: HTTP-based, integrates with existing infrastructure
- **Complementary to MCP**: MCP for tools, A2A for agent-to-agent

### Alternatives Considered
| Alternative | Reason Rejected |
|-------------|-----------------|
| Custom REST APIs | No standardized discovery or task management |
| MCP for agents | MCP is tool-focused, not agent-to-agent |
| Direct LLM chaining | No interoperability between vendors |

### Key Patterns

#### Agent Card (Capability Advertisement)
```json
{
  "name": "Research Agent",
  "description": "Researches topics and provides summaries",
  "url": "https://agent.example.com",
  "version": "1.0.0",
  "capabilities": {
    "streaming": true,
    "pushNotifications": false
  },
  "skills": [
    {
      "id": "research",
      "name": "Research Topic",
      "description": "Research a topic and return findings"
    }
  ]
}
```

#### A2A Server Implementation
```python
from a2a_sdk import A2AServer, Message, Task, TaskStatus

server = A2AServer(name="Research Agent")

@server.on_message
async def handle_message(message: Message) -> Task:
    # Create task for long-running operation
    task = Task(
        id=str(uuid.uuid4()),
        context_id=message.context_id,
        status=TaskStatus(state="working")
    )
    
    # Process asynchronously
    result = await research(message.parts[0].text)
    
    task.status = TaskStatus(state="completed")
    task.artifacts = [{"kind": "text", "text": result}]
    return task
```

#### A2A Client Usage
```python
from a2a_sdk import A2AClient, Message, TextPart

client = A2AClient.from_url("https://agent.example.com")

# Send message
message = Message(
    role="user",
    parts=[TextPart(text="Research quantum computing")]
)

response = await client.send_message(message)

if response.is_task:
    # Poll for completion
    task = await client.get_task(response.task.id)
    while task.status.state == "working":
        await asyncio.sleep(1)
        task = await client.get_task(task.id)
    print(task.artifacts[0]["text"])
```

#### JSON-RPC Message Format
```json
// Request
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "message/send",
  "params": {
    "message": {
      "role": "user",
      "parts": [{"kind": "text", "text": "Research AI safety"}]
    }
  }
}

// Response (task created)
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "id": "task-uuid",
    "contextId": "context-uuid",
    "status": {"state": "working"},
    "kind": "task"
  }
}
```

### Dependencies
- `a2a-sdk` (A2A Python SDK, if available) or custom implementation
- `httpx` (async HTTP client)
- `fastapi` (server implementation)

### References
- [A2A Protocol Specification](https://google.github.io/A2A)
- [A2A Samples Repository](https://github.com/google/a2a)

---

## 5. OpenTelemetry + Azure Monitor

### Decision
Use OpenTelemetry SDK with Azure Monitor exporter for observability.

### Rationale
- **Vendor neutral**: OTel is the industry standard
- **Azure integration**: Native exporter for Application Insights
- **Full stack**: Traces, metrics, logs in one framework
- **Agent tracing**: Trace LLM calls, tool invocations, agent handoffs
- **Constitution compliance**: Principle V requires observability

### Key Patterns

#### Setup OpenTelemetry with Azure Monitor
```python
from azure.monitor.opentelemetry import configure_azure_monitor
from opentelemetry import trace

# Configure once at startup
configure_azure_monitor(
    connection_string=os.environ["APPLICATIONINSIGHTS_CONNECTION_STRING"]
)

tracer = trace.get_tracer(__name__)
```

#### Trace Agent Execution
```python
from opentelemetry import trace
from opentelemetry.trace import SpanKind

tracer = trace.get_tracer("workshop.agents")

async def run_agent(query: str) -> str:
    with tracer.start_as_current_span(
        "agent.run",
        kind=SpanKind.INTERNAL,
        attributes={
            "agent.name": "research_agent",
            "agent.query": query
        }
    ) as span:
        result = await agent.run(query)
        span.set_attribute("agent.result_length", len(result))
        return result
```

#### Trace Tool Calls
```python
@tracer.start_as_current_span("tool.search")
async def search_tool(query: str) -> str:
    span = trace.get_current_span()
    span.set_attribute("tool.name", "search")
    span.set_attribute("tool.query", query)
    # ... implementation
```

### Dependencies
- `opentelemetry-api`
- `opentelemetry-sdk`
- `azure-monitor-opentelemetry`

### References
- [Azure Monitor OpenTelemetry](https://learn.microsoft.com/azure/azure-monitor/app/opentelemetry-python)
- [OpenTelemetry Python](https://opentelemetry.io/docs/languages/python/)

---

## 6. Deterministic Workflow Engine

### Decision
Build a lightweight workflow engine for Scenario 4 (deterministic multi-agent workflows).

### Rationale
- **Explicit control flow**: Workshop teaches when NOT to use pure LLM orchestration
- **Reproducibility**: Same inputs produce same outputs
- **Educational value**: Shows the spectrum from deterministic to autonomous
- **Simple implementation**: ~100 lines, fits in notebook

### Key Patterns

#### Workflow Definition
```python
from dataclasses import dataclass
from typing import Callable, Any

@dataclass
class WorkflowStep:
    name: str
    agent: str
    prompt_template: str
    input_mapper: Callable[[dict], str]
    output_mapper: Callable[[str], dict]

@dataclass
class Workflow:
    name: str
    steps: list[WorkflowStep]

    async def run(self, agents: dict, initial_input: dict) -> dict:
        context = initial_input.copy()
        for step in self.steps:
            agent = agents[step.agent]
            prompt = step.prompt_template.format(**context)
            result = await agent.run(prompt)
            context.update(step.output_mapper(result))
        return context
```

#### Example: Research Pipeline
```python
research_workflow = Workflow(
    name="research_pipeline",
    steps=[
        WorkflowStep(
            name="search",
            agent="search_agent",
            prompt_template="Find information about: {topic}",
            input_mapper=lambda ctx: ctx["topic"],
            output_mapper=lambda r: {"search_results": r}
        ),
        WorkflowStep(
            name="analyze",
            agent="analysis_agent",
            prompt_template="Analyze these findings:\n{search_results}",
            input_mapper=lambda ctx: ctx["search_results"],
            output_mapper=lambda r: {"analysis": r}
        ),
        WorkflowStep(
            name="summarize",
            agent="summary_agent",
            prompt_template="Summarize this analysis:\n{analysis}",
            input_mapper=lambda ctx: ctx["analysis"],
            output_mapper=lambda r: {"summary": r}
        )
    ]
)
```

---

## 7. Declarative Agent Configuration (YAML)

### Decision
Use YAML files for declarative agent/workflow configuration in Scenario 5.

### Rationale
- **No-code configuration**: Operators can modify behavior without Python
- **Version control friendly**: YAML diffs are readable
- **Pydantic validation**: Strong typing via Pydantic models
- **Educational contrast**: Shows declarative vs imperative patterns

### Key Patterns

#### Agent Configuration Schema
```yaml
# configs/agents/research_agent.yaml
name: research_agent
model:
  provider: azure_openai
  deployment: gpt-4o
  temperature: 0.7
instructions: |
  You are a research assistant. Your role is to:
  1. Search for relevant information
  2. Synthesize findings
  3. Cite sources appropriately
tools:
  - search_web
  - read_document
max_tokens: 4096
```

#### Workflow Configuration Schema
```yaml
# configs/workflows/research_pipeline.yaml
name: research_pipeline
description: Multi-step research workflow
steps:
  - name: gather
    agent: research_agent
    prompt: "Research the topic: {topic}"
    outputs: [search_results]
  - name: analyze
    agent: analysis_agent
    prompt: "Analyze: {search_results}"
    outputs: [analysis]
  - name: report
    agent: summary_agent
    prompt: "Write a report on: {analysis}"
    outputs: [report]
```

#### Config Loader
```python
from pathlib import Path
from pydantic import BaseModel
import yaml

class AgentConfig(BaseModel):
    name: str
    model: dict
    instructions: str
    tools: list[str]
    max_tokens: int = 4096

def load_agent_config(path: Path) -> AgentConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return AgentConfig(**data)
```

---

## 8. Protocol Comparison Matrix

| Aspect | MCP | AG-UI | A2A |
|--------|-----|-------|-----|
| **Purpose** | Tool/resource access | Human-agent UI | Agent-agent communication |
| **Direction** | Agent → Tool Server | Agent → Human | Agent → Agent |
| **Transport** | stdio, HTTP, SSE | HTTP SSE | HTTP JSON-RPC |
| **State** | Stateless tools | Session-based threads | Task-based contexts |
| **Discovery** | capabilities endpoint | N/A | Agent Card |
| **Use in Workshop** | Scenario 1 (tools) | Scenario 2 (UI) | Scenario 3 (interop) |

---

## 9. Package Dependencies Summary

```toml
# pyproject.toml dependencies for workshop
[project.dependencies]
python = ">=3.11"

# Microsoft Agent Framework
agent-framework = ">=0.1.0"
azure-ai-projects = ">=1.0.0"
azure-identity = ">=1.15.0"

# Protocols
mcp = ">=1.0.0"
ag-ui-core = ">=0.1.0"

# HTTP/API
fastapi = ">=0.110.0"
uvicorn = ">=0.27.0"
httpx = ">=0.26.0"

# Observability
opentelemetry-api = ">=1.22.0"
opentelemetry-sdk = ">=1.22.0"
azure-monitor-opentelemetry = ">=1.2.0"

# Core utilities
pydantic = ">=2.5.0"
pyyaml = ">=6.0.0"

# Notebooks
jupyter = ">=1.0.0"
ipykernel = ">=6.0.0"

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=4.1.0",
    "pytest-asyncio>=0.23.0",
    "mypy>=1.8.0",
    "ruff>=0.2.0",
]
```

---

## Summary of Resolved Clarifications

All NEEDS CLARIFICATION items from Technical Context have been resolved:

1. ✅ **Microsoft Agent Framework**: Use `agent-framework` Python package with Azure OpenAI
2. ✅ **MCP**: Use `mcp` Python SDK for tool servers
3. ✅ **AG-UI**: Use `ag-ui-core` with FastAPI, or built-in Agent Framework support
4. ✅ **A2A**: Implement per Google specification, use custom server if SDK unavailable
5. ✅ **OpenTelemetry**: Use `azure-monitor-opentelemetry` for Azure Monitor integration
6. ✅ **Workflow Engine**: Build lightweight custom engine (~100 lines)
7. ✅ **Declarative Config**: Use YAML with Pydantic validation

---

## 8. Discussion Framework Refactoring (Phase 13)

### Decision
Use `GroupChatBuilder` from Microsoft Agent Framework for multi-agent discussion orchestration.

### Rationale
- **Native agent-framework support**: GroupChatBuilder is first-party Microsoft SDK
- **Manager pattern**: `set_manager(agent)` directly maps to ModeratorAgent concept
- **Round-based**: `with_max_rounds(n)` supports our discussion round model
- **HITL support**: `with_request_info()` enables human intervention points
- **Type-safe**: GroupChatStateSnapshot provides structured round state

### Alternatives Considered
| Alternative | Reason Rejected |
|-------------|-----------------|
| MagenticBuilder | Better for task planning; discussions are more conversational |
| Custom orchestration | Reinvents wheel; GroupChatBuilder handles coordination |
| Raw ChatAgent loops | Loses structured speaker selection and state management |

### Type Mappings (T102)

| Current Implementation | Agent Framework Type | Notes |
|------------------------|----------------------|-------|
| `DiscussionTurn` | `GroupChatTurn` | speaker, content, timestamp |
| `RoundResult` | `GroupChatStateSnapshot` | round_index, history, conversation |
| `ModeratorAgent` | `ChatAgent` as manager | Via `GroupChatBuilder.set_manager()` |
| `AgentProtocol` | `ChatAgent` | Already interface-compatible |
| `DiscussionConfig` | Builder method calls | participants, max_rounds |

### Key Patterns

#### GroupChatBuilder for Discussions
```python
from agent_framework import GroupChatBuilder, GroupChatStateSnapshot, ChatAgent
from agent_framework.azure import AzureOpenAIChatClient

# Create chat client
chat_client = AzureOpenAIChatClient()

# Create discussion participants
optimist = chat_client.create_agent(
    name="optimist",
    instructions="You are an optimistic perspective. Find the positive aspects."
)

pessimist = chat_client.create_agent(
    name="pessimist",
    instructions="You are a skeptical perspective. Identify risks and concerns."
)

# Create moderator
moderator = chat_client.create_agent(
    name="moderator",
    instructions="""You coordinate discussion. Pick the next speaker based on:
    - Balance: Ensure all perspectives are heard
    - Relevance: Choose speakers who can best address current point
    - Progress: Guide toward synthesis after sufficient exploration
    Return ONLY the speaker name (optimist or pessimist) or DONE if complete."""
)

# Build discussion workflow
discussion = (
    GroupChatBuilder()
    .set_manager(moderator, display_name="Moderator")
    .participants([optimist, pessimist])
    .with_max_rounds(10)
    .build()
)

# Run discussion
async for event in discussion.run_stream("Discuss: Should we adopt AI in education?"):
    if hasattr(event, 'text'):
        print(f"[{event.speaker}]: {event.text}")
```

#### Function-Based Speaker Selection (Deterministic)
```python
from agent_framework import GroupChatBuilder, GroupChatStateSnapshot

def select_round_robin(state: GroupChatStateSnapshot) -> str | None:
    """Deterministic round-robin speaker selection."""
    if state["round_index"] >= 6:  # 3 rounds per speaker
        return None  # End discussion
    
    speakers = list(state["participants"].keys())
    return speakers[state["round_index"] % len(speakers)]

# Build with function-based selection
discussion = (
    GroupChatBuilder()
    .set_select_speakers_func(select_round_robin)
    .participants([optimist, pessimist])
    .build()
)
```

#### Event Streaming to AG-UI
```python
from ag_ui.core import EventType, RunStartedEvent, TextMessageStartEvent

async def stream_discussion_to_agui(discussion_workflow, topic: str):
    """Stream GroupChatBuilder events to AG-UI format."""
    
    yield {"type": EventType.RUN_STARTED}
    
    async for event in discussion_workflow.run_stream(topic):
        if isinstance(event, WorkflowStartedEvent):
            continue  # Internal event
        
        if hasattr(event, 'speaker') and hasattr(event, 'text'):
            # Map GroupChatTurn to AG-UI message
            message_id = str(uuid.uuid4())
            
            yield {
                "type": EventType.TEXT_MESSAGE_START,
                "message_id": message_id,
                "role": "assistant",
                "name": event.speaker
            }
            
            yield {
                "type": EventType.TEXT_MESSAGE_CONTENT,
                "message_id": message_id,
                "delta": event.text
            }
            
            yield {
                "type": EventType.TEXT_MESSAGE_END,
                "message_id": message_id
            }
    
    yield {"type": EventType.RUN_FINISHED}
```

### Migration Path

1. **T104**: Create `DiscussionAgent` wrapper around `ChatAgent`
2. **T105**: Refactor `ModeratorAgent` to use `GroupChatBuilder.set_manager()`
3. **T106**: Update `DiscussionProtocol` to extend `GroupChatStateSnapshot`
4. **T107**: Align protocol types with agent-framework
5. **T108-T109**: Wire streaming events to AG-UI endpoint

### References
- [GroupChatBuilder API](https://github.com/microsoft/agent-framework/blob/main/src/agent_framework/_workflows/_group_chat.py)
- [MagenticBuilder API](https://github.com/microsoft/agent-framework/blob/main/src/agent_framework/_workflows/_magentic.py)
- [WorkflowEvent Types](https://github.com/microsoft/agent-framework/blob/main/src/agent_framework/_workflows/_workflow.py)
