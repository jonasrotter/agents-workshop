# MCP Tool Schemas

This file defines JSON Schemas for workshop MCP tools (Scenario 1).

## search_web

Search the web for information.

```json
{
  "name": "search_web",
  "description": "Search the web for information on a topic",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Search query",
        "minLength": 1,
        "maxLength": 500
      },
      "max_results": {
        "type": "integer",
        "description": "Maximum number of results to return",
        "minimum": 1,
        "maximum": 20,
        "default": 5
      }
    },
    "required": ["query"]
  },
  "outputSchema": {
    "type": "object",
    "properties": {
      "results": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "title": { "type": "string" },
            "url": { "type": "string", "format": "uri" },
            "snippet": { "type": "string" }
          },
          "required": ["title", "url", "snippet"]
        }
      },
      "total_found": { "type": "integer" }
    },
    "required": ["results"]
  }
}
```

## calculate

Perform mathematical calculations.

```json
{
  "name": "calculate",
  "description": "Perform mathematical calculations",
  "inputSchema": {
    "type": "object",
    "properties": {
      "operation": {
        "type": "string",
        "enum": ["add", "subtract", "multiply", "divide", "power", "sqrt"],
        "description": "Mathematical operation to perform"
      },
      "a": {
        "type": "number",
        "description": "First operand"
      },
      "b": {
        "type": "number",
        "description": "Second operand (not required for sqrt)"
      }
    },
    "required": ["operation", "a"]
  },
  "outputSchema": {
    "type": "object",
    "properties": {
      "result": { "type": "number" },
      "operation": { "type": "string" },
      "expression": { "type": "string" }
    },
    "required": ["result", "operation"]
  }
}
```

## read_file

Read contents of a file.

```json
{
  "name": "read_file",
  "description": "Read the contents of a file from the workspace",
  "inputSchema": {
    "type": "object",
    "properties": {
      "path": {
        "type": "string",
        "description": "Relative path to the file",
        "pattern": "^[\\w\\-./]+$"
      },
      "encoding": {
        "type": "string",
        "description": "File encoding",
        "enum": ["utf-8", "ascii", "latin-1"],
        "default": "utf-8"
      }
    },
    "required": ["path"]
  },
  "outputSchema": {
    "type": "object",
    "properties": {
      "content": { "type": "string" },
      "size_bytes": { "type": "integer" },
      "encoding": { "type": "string" }
    },
    "required": ["content"]
  }
}
```

## write_file

Write content to a file.

```json
{
  "name": "write_file",
  "description": "Write content to a file in the workspace",
  "inputSchema": {
    "type": "object",
    "properties": {
      "path": {
        "type": "string",
        "description": "Relative path to the file",
        "pattern": "^[\\w\\-./]+$"
      },
      "content": {
        "type": "string",
        "description": "Content to write"
      },
      "mode": {
        "type": "string",
        "enum": ["overwrite", "append"],
        "default": "overwrite"
      }
    },
    "required": ["path", "content"]
  },
  "outputSchema": {
    "type": "object",
    "properties": {
      "success": { "type": "boolean" },
      "bytes_written": { "type": "integer" },
      "path": { "type": "string" }
    },
    "required": ["success"]
  }
}
```

## get_weather

Get weather for a location (mock implementation for workshop).

```json
{
  "name": "get_weather",
  "description": "Get current weather for a location",
  "inputSchema": {
    "type": "object",
    "properties": {
      "location": {
        "type": "string",
        "description": "City name or location",
        "minLength": 1
      },
      "units": {
        "type": "string",
        "enum": ["celsius", "fahrenheit"],
        "default": "celsius"
      }
    },
    "required": ["location"]
  },
  "outputSchema": {
    "type": "object",
    "properties": {
      "location": { "type": "string" },
      "temperature": { "type": "number" },
      "units": { "type": "string" },
      "condition": { 
        "type": "string",
        "enum": ["sunny", "cloudy", "rainy", "snowy", "stormy"]
      },
      "humidity_percent": { "type": "integer" }
    },
    "required": ["location", "temperature", "condition"]
  }
}
```

## send_notification

Send a notification (mock for workshop).

```json
{
  "name": "send_notification",
  "description": "Send a notification message",
  "inputSchema": {
    "type": "object",
    "properties": {
      "channel": {
        "type": "string",
        "enum": ["email", "slack", "teams"],
        "description": "Notification channel"
      },
      "recipient": {
        "type": "string",
        "description": "Recipient identifier"
      },
      "subject": {
        "type": "string",
        "description": "Notification subject/title"
      },
      "message": {
        "type": "string",
        "description": "Notification body"
      }
    },
    "required": ["channel", "recipient", "message"]
  },
  "outputSchema": {
    "type": "object",
    "properties": {
      "success": { "type": "boolean" },
      "message_id": { "type": "string" },
      "timestamp": { "type": "string", "format": "date-time" }
    },
    "required": ["success"]
  }
}
```
