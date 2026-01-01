# AG-UI Event Schemas

This file defines event schemas for AG-UI protocol implementation (Scenario 2).

## Event Types Enum

```json
{
  "EventType": {
    "RUN_STARTED": "RUN_STARTED",
    "RUN_FINISHED": "RUN_FINISHED",
    "RUN_ERROR": "RUN_ERROR",
    "TEXT_MESSAGE_START": "TEXT_MESSAGE_START",
    "TEXT_MESSAGE_CONTENT": "TEXT_MESSAGE_CONTENT",
    "TEXT_MESSAGE_END": "TEXT_MESSAGE_END",
    "TOOL_CALL_START": "TOOL_CALL_START",
    "TOOL_CALL_ARGS": "TOOL_CALL_ARGS",
    "TOOL_CALL_END": "TOOL_CALL_END",
    "STATE_SNAPSHOT": "STATE_SNAPSHOT",
    "STATE_DELTA": "STATE_DELTA",
    "RAW": "RAW"
  }
}
```

## Request Schema

### RunAgentInput

Request body for the agentic chat endpoint.

```json
{
  "type": "object",
  "properties": {
    "thread_id": {
      "type": "string",
      "description": "Conversation thread identifier"
    },
    "run_id": {
      "type": "string",
      "description": "Unique run identifier"
    },
    "messages": {
      "type": "array",
      "items": { "$ref": "#/$defs/Message" },
      "description": "Conversation history"
    },
    "tools": {
      "type": "array",
      "items": { "$ref": "#/$defs/Tool" },
      "description": "Available tools for this run"
    },
    "context": {
      "type": "object",
      "description": "Additional context for the run"
    }
  },
  "required": ["thread_id", "run_id", "messages"],
  "$defs": {
    "Message": {
      "type": "object",
      "properties": {
        "role": {
          "type": "string",
          "enum": ["user", "assistant", "tool", "system"]
        },
        "content": { "type": "string" },
        "tool_calls": {
          "type": "array",
          "items": { "$ref": "#/$defs/ToolCall" }
        },
        "tool_call_id": { "type": "string" }
      },
      "required": ["role"]
    },
    "Tool": {
      "type": "object",
      "properties": {
        "name": { "type": "string" },
        "description": { "type": "string" },
        "parameters": { "type": "object" }
      },
      "required": ["name", "description", "parameters"]
    },
    "ToolCall": {
      "type": "object",
      "properties": {
        "id": { "type": "string" },
        "type": { "type": "string", "const": "function" },
        "function": {
          "type": "object",
          "properties": {
            "name": { "type": "string" },
            "arguments": { "type": "string" }
          }
        }
      }
    }
  }
}
```

## Event Schemas

### BaseEvent

Base schema for all events.

```json
{
  "type": "object",
  "properties": {
    "type": {
      "type": "string",
      "description": "Event type identifier"
    },
    "timestamp": {
      "type": "integer",
      "description": "Unix timestamp in milliseconds"
    },
    "raw_event": {
      "description": "Original event data from provider"
    }
  },
  "required": ["type"]
}
```

### RunStartedEvent

Emitted when a run begins.

```json
{
  "allOf": [{ "$ref": "#/BaseEvent" }],
  "properties": {
    "type": { "const": "RUN_STARTED" },
    "thread_id": {
      "type": "string",
      "description": "Thread identifier"
    },
    "run_id": {
      "type": "string",
      "description": "Run identifier"
    }
  },
  "required": ["type", "thread_id", "run_id"]
}
```

### RunFinishedEvent

Emitted when a run completes successfully.

```json
{
  "allOf": [{ "$ref": "#/BaseEvent" }],
  "properties": {
    "type": { "const": "RUN_FINISHED" },
    "thread_id": { "type": "string" },
    "run_id": { "type": "string" }
  },
  "required": ["type", "thread_id", "run_id"]
}
```

### RunErrorEvent

Emitted when a run fails.

```json
{
  "allOf": [{ "$ref": "#/BaseEvent" }],
  "properties": {
    "type": { "const": "RUN_ERROR" },
    "message": {
      "type": "string",
      "description": "Error message"
    },
    "code": {
      "type": "string",
      "description": "Error code"
    }
  },
  "required": ["type", "message"]
}
```

### TextMessageStartEvent

Emitted when an assistant message begins.

```json
{
  "allOf": [{ "$ref": "#/BaseEvent" }],
  "properties": {
    "type": { "const": "TEXT_MESSAGE_START" },
    "message_id": {
      "type": "string",
      "description": "Unique message identifier"
    },
    "role": {
      "type": "string",
      "const": "assistant"
    }
  },
  "required": ["type", "message_id", "role"]
}
```

### TextMessageContentEvent

Emitted for each chunk of text content.

```json
{
  "allOf": [{ "$ref": "#/BaseEvent" }],
  "properties": {
    "type": { "const": "TEXT_MESSAGE_CONTENT" },
    "message_id": { "type": "string" },
    "delta": {
      "type": "string",
      "minLength": 1,
      "description": "Text chunk (must not be empty)"
    }
  },
  "required": ["type", "message_id", "delta"]
}
```

### TextMessageEndEvent

Emitted when a text message completes.

```json
{
  "allOf": [{ "$ref": "#/BaseEvent" }],
  "properties": {
    "type": { "const": "TEXT_MESSAGE_END" },
    "message_id": { "type": "string" }
  },
  "required": ["type", "message_id"]
}
```

### ToolCallStartEvent

Emitted when a tool call begins.

```json
{
  "allOf": [{ "$ref": "#/BaseEvent" }],
  "properties": {
    "type": { "const": "TOOL_CALL_START" },
    "tool_call_id": {
      "type": "string",
      "description": "Unique tool call identifier"
    },
    "tool_call_name": {
      "type": "string",
      "description": "Name of the tool being called"
    },
    "parent_message_id": {
      "type": "string",
      "description": "ID of the assistant message containing this tool call"
    }
  },
  "required": ["type", "tool_call_id", "tool_call_name"]
}
```

### ToolCallArgsEvent

Emitted for chunks of tool call arguments.

```json
{
  "allOf": [{ "$ref": "#/BaseEvent" }],
  "properties": {
    "type": { "const": "TOOL_CALL_ARGS" },
    "tool_call_id": { "type": "string" },
    "delta": {
      "type": "string",
      "description": "JSON argument chunk"
    }
  },
  "required": ["type", "tool_call_id", "delta"]
}
```

### ToolCallEndEvent

Emitted when a tool call completes.

```json
{
  "allOf": [{ "$ref": "#/BaseEvent" }],
  "properties": {
    "type": { "const": "TOOL_CALL_END" },
    "tool_call_id": { "type": "string" }
  },
  "required": ["type", "tool_call_id"]
}
```

## SSE Format

Events are encoded as Server-Sent Events:

```
event: message
data: {"type":"RUN_STARTED","thread_id":"thread-123","run_id":"run-456"}

event: message
data: {"type":"TEXT_MESSAGE_START","message_id":"msg-789","role":"assistant"}

event: message
data: {"type":"TEXT_MESSAGE_CONTENT","message_id":"msg-789","delta":"Hello"}

event: message
data: {"type":"TEXT_MESSAGE_CONTENT","message_id":"msg-789","delta":" World"}

event: message
data: {"type":"TEXT_MESSAGE_END","message_id":"msg-789"}

event: message
data: {"type":"RUN_FINISHED","thread_id":"thread-123","run_id":"run-456"}
```

## Content Types

- Request: `application/json`
- Response: `text/event-stream` (SSE) or `application/x-ndjson` (NDJSON)
