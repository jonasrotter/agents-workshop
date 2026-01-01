# A2A Protocol Schemas

This file defines schemas for A2A (Agent-to-Agent) protocol implementation (Scenario 3).

## Agent Card Schema

The Agent Card is served at `/.well-known/agent-card.json` and advertises agent capabilities.

```json
{
  "type": "object",
  "properties": {
    "name": {
      "type": "string",
      "description": "Human-readable agent name"
    },
    "description": {
      "type": "string",
      "description": "What this agent does"
    },
    "url": {
      "type": "string",
      "format": "uri",
      "description": "Base URL for the agent's A2A endpoint"
    },
    "version": {
      "type": "string",
      "pattern": "^\\d+\\.\\d+\\.\\d+$",
      "description": "Semantic version"
    },
    "capabilities": {
      "type": "object",
      "properties": {
        "streaming": {
          "type": "boolean",
          "description": "Supports SSE streaming"
        },
        "pushNotifications": {
          "type": "boolean",
          "description": "Supports webhook callbacks"
        },
        "stateManagement": {
          "type": "boolean",
          "description": "Maintains conversation state"
        }
      }
    },
    "skills": {
      "type": "array",
      "items": { "$ref": "#/$defs/Skill" },
      "description": "List of agent skills/capabilities"
    },
    "authentication": {
      "$ref": "#/$defs/AuthConfig"
    },
    "provider": {
      "type": "object",
      "properties": {
        "name": { "type": "string" },
        "url": { "type": "string", "format": "uri" }
      }
    }
  },
  "required": ["name", "url", "version", "skills"],
  "$defs": {
    "Skill": {
      "type": "object",
      "properties": {
        "id": {
          "type": "string",
          "description": "Unique skill identifier"
        },
        "name": {
          "type": "string",
          "description": "Human-readable skill name"
        },
        "description": {
          "type": "string",
          "description": "What this skill does"
        },
        "inputSchema": {
          "type": "object",
          "description": "JSON Schema for skill input"
        },
        "outputSchema": {
          "type": "object",
          "description": "JSON Schema for skill output"
        }
      },
      "required": ["id", "name", "description"]
    },
    "AuthConfig": {
      "type": "object",
      "properties": {
        "type": {
          "type": "string",
          "enum": ["none", "bearer", "api_key", "oauth2"]
        },
        "scheme": { "type": "string" }
      }
    }
  }
}
```

### Example Agent Card

```json
{
  "name": "Research Agent",
  "description": "Researches topics and provides comprehensive summaries",
  "url": "https://agent.example.com/a2a",
  "version": "1.0.0",
  "capabilities": {
    "streaming": true,
    "pushNotifications": false,
    "stateManagement": true
  },
  "skills": [
    {
      "id": "research_topic",
      "name": "Research Topic",
      "description": "Research a topic and return detailed findings with sources",
      "inputSchema": {
        "type": "object",
        "properties": {
          "topic": { "type": "string" },
          "depth": { "type": "string", "enum": ["brief", "detailed", "comprehensive"] }
        },
        "required": ["topic"]
      },
      "outputSchema": {
        "type": "object",
        "properties": {
          "summary": { "type": "string" },
          "sources": { "type": "array", "items": { "type": "string" } }
        }
      }
    }
  ],
  "authentication": {
    "type": "bearer"
  },
  "provider": {
    "name": "Workshop Demo",
    "url": "https://workshop.example.com"
  }
}
```

## JSON-RPC Message Schemas

### Base Request

```json
{
  "type": "object",
  "properties": {
    "jsonrpc": {
      "type": "string",
      "const": "2.0"
    },
    "id": {
      "oneOf": [
        { "type": "string" },
        { "type": "integer" }
      ]
    },
    "method": {
      "type": "string"
    },
    "params": {
      "type": "object"
    }
  },
  "required": ["jsonrpc", "id", "method"]
}
```

### Base Response

```json
{
  "type": "object",
  "properties": {
    "jsonrpc": {
      "type": "string",
      "const": "2.0"
    },
    "id": {
      "oneOf": [
        { "type": "string" },
        { "type": "integer" }
      ]
    },
    "result": {},
    "error": {
      "$ref": "#/$defs/Error"
    }
  },
  "required": ["jsonrpc", "id"],
  "$defs": {
    "Error": {
      "type": "object",
      "properties": {
        "code": { "type": "integer" },
        "message": { "type": "string" },
        "data": {}
      },
      "required": ["code", "message"]
    }
  }
}
```

## Method: message/send

Send a message to the agent.

### Request

```json
{
  "jsonrpc": "2.0",
  "id": "req-001",
  "method": "message/send",
  "params": {
    "message": {
      "role": "user",
      "parts": [
        {
          "kind": "text",
          "text": "Research quantum computing applications"
        }
      ],
      "messageId": "msg-123"
    },
    "contextId": "ctx-456"
  }
}
```

### Response (Direct Message)

```json
{
  "jsonrpc": "2.0",
  "id": "req-001",
  "result": {
    "kind": "message",
    "messageId": "msg-789",
    "contextId": "ctx-456",
    "parts": [
      {
        "kind": "text",
        "text": "Quantum computing has several key applications..."
      }
    ],
    "metadata": {}
  }
}
```

### Response (Task Created)

```json
{
  "jsonrpc": "2.0",
  "id": "req-001",
  "result": {
    "kind": "task",
    "id": "task-abc",
    "contextId": "ctx-456",
    "status": {
      "state": "working",
      "message": {
        "role": "agent",
        "parts": [{ "kind": "text", "text": "Researching..." }]
      },
      "timestamp": "2026-01-01T10:00:00Z"
    },
    "metadata": {}
  }
}
```

## Method: tasks/get

Get task status and results.

### Request

```json
{
  "jsonrpc": "2.0",
  "id": "req-002",
  "method": "tasks/get",
  "params": {
    "id": "task-abc",
    "historyLength": 10
  }
}
```

### Response (Completed)

```json
{
  "jsonrpc": "2.0",
  "id": "req-002",
  "result": {
    "kind": "task",
    "id": "task-abc",
    "contextId": "ctx-456",
    "status": {
      "state": "completed",
      "timestamp": "2026-01-01T10:05:00Z"
    },
    "artifacts": [
      {
        "artifactId": "art-001",
        "name": "research_findings",
        "parts": [
          {
            "kind": "text",
            "text": "# Quantum Computing Applications\n\n..."
          }
        ]
      }
    ],
    "history": [
      {
        "role": "user",
        "parts": [{ "kind": "text", "text": "Research quantum computing" }],
        "messageId": "msg-123"
      }
    ],
    "metadata": {}
  }
}
```

## Method: tasks/list

List tasks for a context.

### Request

```json
{
  "jsonrpc": "2.0",
  "id": "req-003",
  "method": "tasks/list",
  "params": {
    "contextId": "ctx-456",
    "status": "working",
    "pageSize": 10
  }
}
```

### Response

```json
{
  "jsonrpc": "2.0",
  "id": "req-003",
  "result": {
    "tasks": [
      {
        "id": "task-abc",
        "contextId": "ctx-456",
        "status": { "state": "working" }
      }
    ],
    "totalSize": 1,
    "pageSize": 10,
    "nextPageToken": ""
  }
}
```

## Method: tasks/cancel

Cancel a running task.

### Request

```json
{
  "jsonrpc": "2.0",
  "id": "req-004",
  "method": "tasks/cancel",
  "params": {
    "id": "task-abc"
  }
}
```

### Response

```json
{
  "jsonrpc": "2.0",
  "id": "req-004",
  "result": {
    "kind": "task",
    "id": "task-abc",
    "status": {
      "state": "cancelled",
      "timestamp": "2026-01-01T10:03:00Z"
    }
  }
}
```

## Task States

```json
{
  "TaskState": {
    "submitted": "Task received but not yet started",
    "working": "Task is being processed",
    "input-required": "Task needs user input to continue",
    "completed": "Task finished successfully",
    "failed": "Task failed with an error",
    "cancelled": "Task was cancelled"
  }
}
```

## Message Parts

### TextPart

```json
{
  "type": "object",
  "properties": {
    "kind": { "const": "text" },
    "text": { "type": "string" }
  },
  "required": ["kind", "text"]
}
```

### FilePart

```json
{
  "type": "object",
  "properties": {
    "kind": { "const": "file" },
    "mimeType": { "type": "string" },
    "data": {
      "type": "string",
      "description": "Base64-encoded file data"
    },
    "uri": {
      "type": "string",
      "format": "uri",
      "description": "Alternative: URI to fetch file"
    }
  },
  "required": ["kind", "mimeType"]
}
```

### DataPart

```json
{
  "type": "object",
  "properties": {
    "kind": { "const": "data" },
    "mimeType": { "type": "string" },
    "data": {
      "description": "Structured data (e.g., JSON object)"
    }
  },
  "required": ["kind", "data"]
}
```

## Error Codes

| Code | Message | Description |
|------|---------|-------------|
| -32700 | Parse error | Invalid JSON |
| -32600 | Invalid Request | Invalid JSON-RPC request |
| -32601 | Method not found | Unknown method |
| -32602 | Invalid params | Invalid method parameters |
| -32603 | Internal error | Internal server error |
| -32000 | Task not found | Referenced task doesn't exist |
| -32001 | Context not found | Referenced context doesn't exist |
| -32002 | Rate limited | Too many requests |
| -32003 | Unauthorized | Authentication required |
