# Architecture Documentation

## Overview

This workshop implements a modular architecture for teaching agentic AI patterns. The design prioritizes clarity, extensibility, and educational value over production optimizations.

## System Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                              WORKSHOP LAYER                                  │
│                                                                             │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│   │ Notebook 01 │  │ Notebook 02 │  │ Notebook 03 │  │    ...      │       │
│   │  MCP Agent  │  │   AG-UI     │  │    A2A      │  │             │       │
│   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘       │
│          │                │                │                │               │
└──────────┼────────────────┼────────────────┼────────────────┼───────────────┘
           │                │                │                │
           ▼                ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AGENT LAYER                                     │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                        src/agents/                                   │   │
│   │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │   │
│   │  │  BaseAgent   │  │ ResearchAgent│  │ModeratorAgent│              │   │
│   │  │  - execute() │  │ - research() │  │ - moderate() │              │   │
│   │  │  - trace()   │  │ - summarize()│  │ - resolve()  │              │   │
│   │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘              │   │
│   │         │                 │                 │                        │   │
│   └─────────┼─────────────────┼─────────────────┼────────────────────────┘   │
│             │                 │                 │                             │
└─────────────┼─────────────────┼─────────────────┼─────────────────────────────┘
              │                 │                 │
              ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            PROTOCOL LAYER                                    │
│                                                                             │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│   │    MCP      │  │   AG-UI     │  │    A2A      │  │ Discussion  │       │
│   │   Tools     │  │   Server    │  │  Protocol   │  │  Protocol   │       │
│   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘       │
│          │                │                │                │               │
└──────────┼────────────────┼────────────────┼────────────────┼───────────────┘
           │                │                │                │
           ▼                ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            COMMON LAYER                                      │
│                                                                             │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│   │   Config    │  │  Telemetry  │  │ Evaluation  │  │PromptTuning │       │
│   │  (config.py)│  │(telemetry.py│  │(evaluation.py│  │(prompt_tuning│      │
│   └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Module Relationships

### Core Modules

#### `src/common/config.py`
Central configuration management using Pydantic settings.

**Responsibilities:**
- Load environment variables
- Validate configuration values
- Provide typed settings access

**Key Classes:**
- `Settings`: Main settings container
- `AzureOpenAISettings`: Azure-specific configuration
- `TelemetrySettings`: Observability configuration

**Dependencies:** None (foundational module)

#### `src/common/telemetry.py`
OpenTelemetry integration for distributed tracing.

**Responsibilities:**
- Initialize tracing provider
- Create span contexts
- Export to Azure Monitor

**Key Classes:**
- `TelemetryConfig`: Tracer configuration
- `SpanContext`: Context manager for spans
- `TracedOperation`: Decorator for traced functions

**Dependencies:** `config.py`

#### `src/common/exceptions.py`
Structured exception hierarchy for consistent error handling.

**Key Classes:**
- `AgentError`: Base exception for agent errors
- `ToolError`: MCP tool execution errors
- `ProtocolError`: Protocol-specific errors
- `WorkflowError`: Workflow execution errors

**Dependencies:** None

### Agent Modules

#### `src/agents/base_agent.py`
Foundation for all agent implementations.

**Responsibilities:**
- Define agent interface
- Integrate telemetry
- Manage agent lifecycle

**Key Classes:**
- `BaseAgent`: Abstract base with `execute()` method
- `AgentConfig`: Agent configuration
- `AgentContext`: Execution context with tracing

**Dependencies:** `config.py`, `telemetry.py`, `exceptions.py`

#### `src/agents/research_agent.py`
Specialized agent for research tasks.

**Key Methods:**
- `research()`: Execute research queries
- `summarize()`: Synthesize findings
- `cite()`: Generate citations

**Dependencies:** `base_agent.py`, `search_tool.py`

#### `src/agents/moderator_agent.py`
Discussion and debate moderation.

**Key Methods:**
- `moderate()`: Manage turn-taking
- `resolve_conflict()`: Handle disagreements
- `synthesize()`: Generate summaries

**Dependencies:** `base_agent.py`, `discussion.py`

### Tool Modules

#### `src/tools/search_tool.py`
Web search MCP tool implementation.

**MCP Schema:**
```json
{
  "name": "web_search",
  "parameters": {
    "query": "string",
    "max_results": "integer"
  }
}
```

#### `src/tools/calculator_tool.py`
Mathematical computation tool.

**Supported Operations:** add, subtract, multiply, divide, power, sqrt

#### `src/tools/file_tool.py`
File system operations (sandboxed).

**Operations:** read, write, list, exists

### Protocol Modules

#### `src/protocols/agui.py`
AG-UI streaming protocol implementation.

**Event Types:**
- `TEXT_DELTA`: Streaming text
- `TOOL_CALL_START/END`: Tool execution
- `RUN_STARTED/FINISHED`: Lifecycle events

**Server:** FastAPI with SSE streaming

#### `src/protocols/a2a.py`
Agent-to-Agent interoperability protocol.

**Key Classes:**
- `A2AServer`: FastAPI server exposing agent
- `AgentCard`: Agent capability description
- `A2AClient`: Client for invoking remote agents

#### `src/protocols/discussion.py`
Multi-agent discussion coordination.

**Protocols:**
- `DebateProtocol`: Structured pro/con debate
- `RoundRobinProtocol`: Equal-turn discussions
- `DiscussionProtocol`: Generic discussion handling

### Workflow Module

#### `src/workflows/engine.py`
Deterministic workflow orchestration.

**Key Classes:**
- `WorkflowEngine`: Execution coordinator
- `WorkflowStep`: Individual step definition
- `WorkflowContext`: Shared execution state

**Execution Modes:**
- Sequential: Steps run in order
- Parallel: Independent steps run concurrently
- Conditional: Branch based on results

### Evaluation Modules

#### `src/common/evaluation_config.py`
Configuration utilities for Azure AI Evaluation SDK.

**Key Functions:**
- `get_model_config()`: Get Azure OpenAI configuration for SDK evaluators
- `get_azure_ai_project()`: Get Azure AI Foundry project configuration (optional)
- `validate_config()`: Validate model configuration
- `get_config_summary()`: Get safe config summary (masks API keys)

**Key Types:**
- `ModelConfig`: TypedDict for SDK evaluator configuration
- `AzureAIProject`: TypedDict for Azure AI Foundry integration
- `ConfigurationError`: Exception for missing configuration

#### `src/common/evaluation.py`
Metrics collection and SDK evaluator wrappers.

**SDK Evaluator Wrappers (azure-ai-evaluation integration):**
- Quality Evaluators:
  - `create_relevance_evaluator()`: Response relevance to query (1-5 scale)
  - `create_coherence_evaluator()`: Logical flow and structure (1-5 scale)
  - `create_fluency_evaluator()`: Linguistic quality (1-5 scale)
  - `create_groundedness_evaluator()`: Factual accuracy against context (1-5 scale)
- Agent Evaluators:
  - `create_intent_resolution_evaluator()`: User intent understanding (pass/fail)
  - `create_task_adherence_evaluator()`: Task completion (pass/fail)
  - `create_tool_call_accuracy_evaluator()`: Tool usage correctness (0-1 scale)
- Batch Evaluation:
  - `batch_evaluate()`: Run multiple evaluators on datasets

**Custom Metric Types (retained for cost tracking):**
- Latency: Execution time tracking
- Accuracy: Response correctness (simple matchers)
- Cost: Token/API usage estimation
- Quality: Custom scoring dimensions

**Key Classes:**
- `MetricsCollector`: Central metrics aggregator
- `OpenAICostCalculator`: Token cost estimation
- `ExactMatchEvaluator`, `ContainsEvaluator`: Simple matchers (deprecated, use SDK)

#### `src/common/prompt_tuning.py`
Prompt iteration and improvement.

**Key Classes:**
- `PromptTuner`: Version management
- `PromptAnalyzer`: Quality analysis
- `ABTestRunner`: Comparison testing

## Design Decisions

### 1. Notebook-First Architecture

**Decision:** Use Jupyter Notebooks as primary delivery format with shared `src/` modules.

**Rationale:**
- Interactive learning experience
- Immediate feedback for participants
- Combines documentation with executable code

**Trade-offs:**
- ✅ Excellent for education
- ❌ Not suitable for production deployment

### 2. Protocol Abstraction

**Decision:** Each protocol (MCP, AG-UI, A2A) has its own module with consistent interfaces.

**Rationale:**
- Clear separation of concerns
- Easy to understand individual protocols
- Enables mixing protocols in advanced scenarios

### 3. Evaluation Integration

**Decision:** Built-in evaluation from the start, not bolted on.

**Rationale:**
- Teaches best practices early
- Enables improvement feedback loops
- Demonstrates production patterns

### 4. Configuration via Environment

**Decision:** Use environment variables with Pydantic validation.

**Rationale:**
- Secure credential management
- Easy local/cloud switching
- Standard 12-factor app pattern

## Data Flow

### Request Flow

```text
User Input → Notebook Cell → Agent.execute()
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
              Direct LLM Call              Tool Invocation
                    │                             │
                    │                      MCP Tool Handler
                    │                             │
                    └──────────────┬──────────────┘
                                   ▼
                            Response Assembly
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
              MetricsCollector              TelemetrySpan
                    │                             │
                    └──────────────┬──────────────┘
                                   ▼
                             User Output
```

### Telemetry Flow

```text
Agent Operation → SpanContext → TracerProvider → Azure Monitor
                      │
                      ├── span_id
                      ├── trace_id
                      ├── attributes
                      └── events
```

## Extension Points

### Adding New Tools

1. Create tool class in `src/tools/`
2. Implement MCP schema
3. Register in tool registry
4. Add to agent configuration

### Adding New Protocols

1. Create protocol module in `src/protocols/`
2. Define event/message schemas
3. Implement server/client
4. Add integration notebook

### Adding New Agents

1. Extend `BaseAgent` in `src/agents/`
2. Implement `execute()` method
3. Add configuration schema
4. Create declarative config option

## Performance Considerations

This workshop prioritizes clarity over performance. For production systems:

- **Caching:** Add LLM response caching
- **Batching:** Batch API calls where possible
- **Async:** Use async throughout (partially implemented)
- **Connection Pooling:** Pool HTTP connections

## Security Notes

- **API Keys:** Never commit to source control
- **Sandboxing:** File tools are sandboxed
- **Validation:** All inputs validated via Pydantic
- **Tracing:** Sensitive data excluded from spans
