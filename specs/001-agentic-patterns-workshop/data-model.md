# Data Model: Agentic AI Patterns Workshop

**Date**: 2026-01-01 | **Plan**: [plan.md](./plan.md) | **Research**: [research.md](./research.md)

This document defines the core data structures and schemas for the workshop.

---

## 1. Workshop Module Structure

### Entity: WorkshopModule

Represents a single workshop scenario/notebook.

| Field | Type | Description |
|-------|------|-------------|
| `id` | `int` | Unique scenario number (1-7) |
| `name` | `str` | Short identifier (e.g., "simple_agent_mcp") |
| `title` | `str` | Display title for the notebook |
| `description` | `str` | Learning objectives summary |
| `duration_minutes` | `int` | Estimated completion time |
| `prerequisites` | `list[int]` | IDs of modules that should be completed first |
| `concepts` | `list[str]` | Key concepts covered |
| `notebook_path` | `Path` | Path to .ipynb file |

### Workshop Module Registry

```python
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class WorkshopModule:
    id: int
    name: str
    title: str
    description: str
    duration_minutes: int
    prerequisites: list[int] = field(default_factory=list)
    concepts: list[str] = field(default_factory=list)
    
    @property
    def notebook_path(self) -> Path:
        return Path(f"notebooks/{self.id:02d}_{self.name}.ipynb")

MODULES = [
    WorkshopModule(
        id=0,
        name="setup",
        title="Environment Setup",
        description="Install dependencies and verify Azure connections",
        duration_minutes=15,
        concepts=["environment", "azure_openai", "authentication"]
    ),
    WorkshopModule(
        id=1,
        name="simple_agent_mcp",
        title="Simple Agent with MCP Tools",
        description="Build a basic agent with MCP tool integration and observability",
        duration_minutes=45,
        prerequisites=[0],
        concepts=["agent", "mcp", "tools", "opentelemetry"]
    ),
    WorkshopModule(
        id=2,
        name="agui_interface",
        title="AG-UI Protocol for Streaming Chat",
        description="Expose agents via AG-UI for real-time streaming interfaces",
        duration_minutes=45,
        prerequisites=[1],
        concepts=["ag_ui", "streaming", "sse", "fastapi"]
    ),
    WorkshopModule(
        id=3,
        name="a2a_protocol",
        title="A2A Protocol for Agent Interoperability",
        description="Expose agents to other agents using A2A protocol",
        duration_minutes=45,
        prerequisites=[1],
        concepts=["a2a", "agent_card", "json_rpc", "interoperability"]
    ),
    WorkshopModule(
        id=4,
        name="deterministic_workflows",
        title="Deterministic Multi-Agent Workflows",
        description="Build predictable multi-agent pipelines with explicit control flow",
        duration_minutes=45,
        prerequisites=[1],
        concepts=["workflow", "orchestration", "deterministic", "pipeline"]
    ),
    WorkshopModule(
        id=5,
        name="declarative_agents",
        title="Declarative Agent Configuration",
        description="Define agents and workflows using YAML configuration",
        duration_minutes=30,
        prerequisites=[4],
        concepts=["yaml", "declarative", "configuration", "no_code"]
    ),
    WorkshopModule(
        id=6,
        name="agent_discussions",
        title="Moderated Agent Discussions",
        description="Implement multi-agent debates with a moderator agent",
        duration_minutes=45,
        prerequisites=[4],
        concepts=["multi_agent", "debate", "moderator", "consensus"]
    ),
    WorkshopModule(
        id=7,
        name="evaluation_evolution",
        title="Evaluation and Prompt Evolution",
        description="Evaluate agent outputs and iterate on prompts",
        duration_minutes=30,
        prerequisites=[1],
        concepts=["evaluation", "prompts", "iteration", "metrics"]
    ),
]
```

---

## 2. Agent Configuration Schema

### Entity: AgentConfig

Configuration for a declarative agent definition (Scenario 5).

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | `str` | Yes | Unique agent identifier |
| `model` | `ModelConfig` | Yes | LLM configuration |
| `instructions` | `str` | Yes | System prompt/instructions |
| `tools` | `list[str]` | No | List of tool names to enable |
| `max_tokens` | `int` | No | Maximum response tokens (default: 4096) |
| `temperature` | `float` | No | Sampling temperature (default: 0.7) |

### Entity: ModelConfig

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `provider` | `str` | Yes | Model provider (azure_openai, openai) |
| `deployment` | `str` | Yes | Deployment/model name |
| `temperature` | `float` | No | Override for sampling temperature |
| `api_version` | `str` | No | API version (Azure OpenAI) |

### Pydantic Schema

```python
from pydantic import BaseModel, Field
from typing import Optional

class ModelConfig(BaseModel):
    """LLM model configuration."""
    provider: str = Field(
        ..., 
        description="Model provider: 'azure_openai' or 'openai'"
    )
    deployment: str = Field(
        ..., 
        description="Model deployment name"
    )
    temperature: float = Field(
        default=0.7, 
        ge=0.0, 
        le=2.0,
        description="Sampling temperature"
    )
    api_version: Optional[str] = Field(
        default=None,
        description="API version for Azure OpenAI"
    )

class AgentConfig(BaseModel):
    """Declarative agent configuration."""
    name: str = Field(
        ..., 
        min_length=1, 
        max_length=64,
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Unique agent identifier (lowercase, underscores)"
    )
    model: ModelConfig = Field(
        ..., 
        description="LLM model configuration"
    )
    instructions: str = Field(
        ..., 
        min_length=10,
        description="System prompt for the agent"
    )
    tools: list[str] = Field(
        default_factory=list,
        description="List of tool names to enable"
    )
    max_tokens: int = Field(
        default=4096, 
        ge=1, 
        le=128000,
        description="Maximum response tokens"
    )
    
    class Config:
        extra = "forbid"
```

### YAML Example

```yaml
# configs/agents/research_agent.yaml
name: research_agent
model:
  provider: azure_openai
  deployment: gpt-4o
  temperature: 0.7
instructions: |
  You are a research assistant specializing in technology topics.
  
  Your responsibilities:
  1. Search for relevant information using available tools
  2. Synthesize findings into clear summaries
  3. Always cite your sources
  
  Be thorough but concise.
tools:
  - search_web
  - read_document
max_tokens: 4096
```

---

## 3. Workflow Definition Schema

### Entity: WorkflowConfig

Configuration for a declarative workflow (Scenario 5).

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | `str` | Yes | Unique workflow identifier |
| `description` | `str` | No | Human-readable description |
| `steps` | `list[WorkflowStep]` | Yes | Ordered list of workflow steps |
| `on_error` | `ErrorHandling` | No | Error handling strategy |

### Entity: WorkflowStep

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | `str` | Yes | Step identifier |
| `agent` | `str` | Yes | Agent name to execute this step |
| `prompt` | `str` | Yes | Prompt template (supports {variable} substitution) |
| `outputs` | `list[str]` | Yes | Variable names for step outputs |
| `condition` | `str` | No | Optional condition expression |
| `retry` | `RetryConfig` | No | Retry configuration |

### Pydantic Schema

```python
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

class ErrorStrategy(str, Enum):
    FAIL = "fail"
    SKIP = "skip"
    RETRY = "retry"

class RetryConfig(BaseModel):
    """Retry configuration for workflow steps."""
    max_attempts: int = Field(default=3, ge=1, le=10)
    delay_seconds: float = Field(default=1.0, ge=0.0)
    backoff_multiplier: float = Field(default=2.0, ge=1.0)

class ErrorHandling(BaseModel):
    """Error handling configuration."""
    strategy: ErrorStrategy = Field(default=ErrorStrategy.FAIL)
    fallback_value: Optional[str] = None

class WorkflowStep(BaseModel):
    """Single step in a workflow."""
    name: str = Field(
        ..., 
        min_length=1,
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Step identifier"
    )
    agent: str = Field(
        ..., 
        description="Agent name to execute this step"
    )
    prompt: str = Field(
        ..., 
        min_length=1,
        description="Prompt template with {variable} placeholders"
    )
    outputs: list[str] = Field(
        ..., 
        min_length=1,
        description="Output variable names"
    )
    condition: Optional[str] = Field(
        default=None,
        description="Optional condition expression"
    )
    retry: Optional[RetryConfig] = Field(
        default=None,
        description="Retry configuration"
    )

class WorkflowConfig(BaseModel):
    """Declarative workflow configuration."""
    name: str = Field(
        ..., 
        min_length=1,
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Unique workflow identifier"
    )
    description: Optional[str] = Field(
        default=None,
        description="Human-readable description"
    )
    steps: list[WorkflowStep] = Field(
        ..., 
        min_length=1,
        description="Ordered list of workflow steps"
    )
    on_error: ErrorHandling = Field(
        default_factory=ErrorHandling,
        description="Error handling strategy"
    )
    
    class Config:
        extra = "forbid"
```

### YAML Example

```yaml
# configs/workflows/research_pipeline.yaml
name: research_pipeline
description: Multi-step research workflow with error handling

steps:
  - name: search
    agent: research_agent
    prompt: |
      Research the following topic thoroughly:
      {topic}
      
      Provide at least 3 relevant sources.
    outputs:
      - search_results
    retry:
      max_attempts: 3
      delay_seconds: 2.0

  - name: analyze
    agent: analysis_agent
    prompt: |
      Analyze the following research findings:
      {search_results}
      
      Identify key themes and patterns.
    outputs:
      - analysis

  - name: summarize
    agent: summary_agent
    prompt: |
      Create an executive summary based on:
      {analysis}
      
      Keep it under 300 words.
    outputs:
      - summary

on_error:
  strategy: retry
```

---

## 4. Tool Definition Schema

### Entity: ToolDefinition

Represents an MCP tool that agents can use.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | `str` | Yes | Tool identifier |
| `description` | `str` | Yes | Human-readable description |
| `input_schema` | `dict` | Yes | JSON Schema for input parameters |
| `output_schema` | `dict` | No | JSON Schema for output |
| `category` | `str` | No | Tool category for organization |

### Pydantic Schema

```python
from pydantic import BaseModel, Field
from typing import Any, Optional

class ToolDefinition(BaseModel):
    """MCP tool definition."""
    name: str = Field(
        ..., 
        min_length=1,
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Tool identifier"
    )
    description: str = Field(
        ..., 
        min_length=10,
        description="Human-readable description"
    )
    input_schema: dict[str, Any] = Field(
        ..., 
        description="JSON Schema for input parameters"
    )
    output_schema: Optional[dict[str, Any]] = Field(
        default=None,
        description="JSON Schema for output"
    )
    category: str = Field(
        default="general",
        description="Tool category"
    )
```

### Workshop Tools Registry

| Tool Name | Category | Description |
|-----------|----------|-------------|
| `search_web` | search | Search the web for information |
| `calculate` | math | Perform mathematical calculations |
| `read_file` | file | Read contents of a file |
| `write_file` | file | Write content to a file |
| `get_weather` | data | Get weather for a location |
| `send_notification` | communication | Send a notification |

---

## 5. Observability Entities

### Entity: AgentSpan

Represents a traced agent execution.

| Field | Type | Description |
|-------|------|-------------|
| `trace_id` | `str` | Unique trace identifier |
| `span_id` | `str` | Unique span identifier |
| `parent_span_id` | `str | None` | Parent span (if nested) |
| `agent_name` | `str` | Name of the agent |
| `operation` | `str` | Operation type (run, tool_call, etc.) |
| `start_time` | `datetime` | Span start timestamp |
| `end_time` | `datetime` | Span end timestamp |
| `status` | `SpanStatus` | Success/error status |
| `attributes` | `dict` | Custom attributes |

### Trace Attributes Convention

```python
# Agent execution attributes
ATTR_AGENT_NAME = "agent.name"
ATTR_AGENT_QUERY = "agent.query"
ATTR_AGENT_RESULT_LENGTH = "agent.result.length"
ATTR_AGENT_TOKEN_COUNT = "agent.tokens.total"

# Tool call attributes
ATTR_TOOL_NAME = "tool.name"
ATTR_TOOL_ARGS = "tool.arguments"
ATTR_TOOL_RESULT = "tool.result"

# Workflow attributes
ATTR_WORKFLOW_NAME = "workflow.name"
ATTR_WORKFLOW_STEP = "workflow.step"
ATTR_WORKFLOW_CONTEXT = "workflow.context"

# Error attributes
ATTR_ERROR_TYPE = "error.type"
ATTR_ERROR_MESSAGE = "error.message"
```

---

## 6. State Transitions

### Workflow Execution States

```
┌──────────┐
│ PENDING  │
└────┬─────┘
     │ start()
     ▼
┌──────────┐
│ RUNNING  │◄────────────────┐
└────┬─────┘                 │
     │                       │ retry()
     ├── step_complete() ────┼──►┌──────────┐
     │                       │   │ WAITING  │
     │                       │   └──────────┘
     │
     ├── all_steps_done() ──►┌───────────┐
     │                       │ COMPLETED │
     │                       └───────────┘
     │
     └── error() ───────────►┌──────────┐
                             │  FAILED  │
                             └──────────┘
```

### Task States (A2A Protocol)

```
┌───────────┐
│ SUBMITTED │
└─────┬─────┘
      │ agent_accepts()
      ▼
┌───────────┐
│  WORKING  │◄──────┐
└─────┬─────┘       │
      │             │ needs_input()
      ├─────────────┴──►┌─────────────────┐
      │                 │ INPUT_REQUIRED  │
      │                 └─────────────────┘
      │
      ├── complete() ──►┌───────────┐
      │                 │ COMPLETED │
      │                 └───────────┘
      │
      └── fail() ──────►┌──────────┐
                        │  FAILED  │
                        └──────────┘
```

---

## 7. Validation Rules

### Agent Configuration Validation

1. **Name uniqueness**: Agent names must be unique within a configuration set
2. **Tool existence**: All tools referenced must exist in the tool registry
3. **Model validity**: Provider must be "azure_openai" or "openai"
4. **Temperature range**: Must be between 0.0 and 2.0
5. **Instructions length**: Must be at least 10 characters

### Workflow Configuration Validation

1. **Step name uniqueness**: Step names must be unique within a workflow
2. **Agent existence**: All agents referenced must be defined
3. **Output variables**: Must not conflict with input variables
4. **Circular dependencies**: Workflows must not have circular step dependencies
5. **Prompt templates**: All {variable} references must be resolvable

### Tool Definition Validation

1. **JSON Schema validity**: input_schema must be valid JSON Schema
2. **Required fields**: name, description, input_schema are required
3. **Name format**: Must match `^[a-z][a-z0-9_]*$` pattern
