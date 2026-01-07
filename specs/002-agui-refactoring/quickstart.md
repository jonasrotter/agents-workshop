# Quickstart: AG-UI Interface Refactoring

**Time**: ~15 minutes | **Plan**: [plan.md](./plan.md)

## Prerequisites

- Python 3.11+ with `.venv312` activated
- All dependencies installed (`pip install -r requirements.txt`)
- Understanding of AG-UI protocol from Scenario 02 notebook

## Quick Reference

### Option 1: SDK-Based Server (Recommended for Production)

```python
from agent_framework import ChatAgent
from agent_framework.azure import AzureOpenAIChatClient
from agent_framework.ag_ui import add_agent_framework_fastapi_endpoint
from fastapi import FastAPI
from azure.identity import DefaultAzureCredential

# Create agent
agent = ChatAgent(
    name="assistant",
    instructions="You are a helpful assistant.",
    chat_client=AzureOpenAIChatClient(
        model="gpt-4",
        credential=DefaultAzureCredential()
    )
)

# Create server with ONE line
app = FastAPI()
add_agent_framework_fastapi_endpoint(app, agent, "/")

# Run: uvicorn app:app --reload
```

### Option 2: Custom Server with ag-ui-core (Educational)

```python
from ag_ui.core import EventType, RunAgentInput
from ag_ui.encoder import EventEncoder
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()
encoder = EventEncoder()

@app.post("/")
async def run_agent(input_data: RunAgentInput):
    async def stream_events():
        # Run started
        yield encoder.encode({
            "type": EventType.RUN_STARTED,
            "thread_id": input_data.thread_id,
            "run_id": input_data.run_id
        })
        
        # Message start
        message_id = f"msg-{uuid.uuid4().hex[:8]}"
        yield encoder.encode({
            "type": EventType.TEXT_MESSAGE_START,
            "message_id": message_id,
            "role": "assistant"
        })
        
        # Stream content
        response = "Hello! How can I help you?"
        for char in response:
            yield encoder.encode({
                "type": EventType.TEXT_MESSAGE_CONTENT,
                "message_id": message_id,
                "delta": char
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
        stream_events(),
        media_type=encoder.get_content_type(),
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
```

### Option 3: AG-UI Client (Consuming Servers)

```python
from agent_framework import ChatAgent
from agent_framework.ag_ui import AGUIChatClient

# Connect to remote AG-UI server
client = AGUIChatClient(endpoint="http://localhost:8000/")
agent = ChatAgent(name="remote-assistant", chat_client=client)
thread = await agent.get_new_thread()

# Use like any agent
response = await agent.run("Hello!", thread=thread)
print(response)

# Streaming
async for update in client.get_streaming_response("Tell me a story"):
    for content in update.contents:
        if hasattr(content, "text"):
            print(content.text, end="", flush=True)
```

## Workshop Module Usage

### Using Existing AGUIServer (Backward Compatible)

```python
from src.agents import AGUIServer, AGUIEventEmitter, EventType

# Create server (uses SDK internally)
server = AGUIServer(agent=my_agent)
app = server.create_app()

# Or use SDK directly
from src.agents.agui_server import create_agui_endpoint
create_agui_endpoint(app, agent, "/chat")
```

### Using AGUIClient

```python
from src.agents.agui_server import AGUIClient

async with AGUIClient("http://localhost:8000/") as client:
    response = await client.send("Hello!")
    print(response)
```

## Key Imports

```python
# SDK Types
from ag_ui.core import EventType, RunAgentInput, Message, Tool
from ag_ui.encoder import EventEncoder

# Agent Framework Integration
from agent_framework.ag_ui import (
    add_agent_framework_fastapi_endpoint,
    AGUIChatClient,
    AgentFrameworkAgent,
)

# Workshop Module (backward compatible)
from src.agents import AGUIServer, AGUIEventEmitter, create_agui_server
```

## Next Steps

1. Run notebook `02_agui_interface.ipynb` to see refactored implementation
2. Compare custom vs SDK approaches in the notebook
3. Try modifying the server to add custom event handling
