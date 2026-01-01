# Feature Specification: Agentic AI Patterns Workshop

**Feature Branch**: `001-agentic-patterns-workshop`  
**Created**: 2026-01-01  
**Status**: Draft  
**Input**: User description: "Build a workshop for learning different agentic AI patterns"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Build Simple Agent with MCP Tools and Observability (Priority: P1)

As a workshop participant, I want to build a simple agent, connect tools via the Model Context Protocol (MCP), and inspect traces & metrics, so that I understand the foundational building block of agentic AI systems with proper observability.

**Why this priority**: The single agent with MCP tool integration is the most fundamental pattern. Adding observability from the start teaches best practices and enables debugging for all subsequent scenarios.

**Independent Test**: Can be fully tested by running a standalone agent that connects to an MCP server, invokes tools, and produces observable traces in a tracing UI. Delivers immediate value as participants see the full agent lifecycle.

**Acceptance Scenarios**:

1. **Given** a participant has set up the workshop environment, **When** they run the simple agent example, **Then** the agent connects to an MCP server, discovers available tools, and executes a task using those tools
2. **Given** an agent has completed a task, **When** the participant opens the tracing dashboard, **Then** they can see the full execution trace including LLM calls, tool invocations, and timing metrics
3. **Given** a participant completes the hands-on exercise, **When** they add a new MCP tool, **Then** the agent discovers and can use the new tool without code changes

---

### User Story 2 - User Interface for Agent via AG-UI Protocol (Priority: P2)

As a workshop participant, I want to build a user interface for my agent using the AG-UI protocol, so that I can create interactive chat experiences with streaming responses and rich UI components.

**Why this priority**: User interfaces are essential for making agents accessible. AG-UI provides a standardized protocol for agent-UI communication, building on the single agent from Scenario 1.

**Independent Test**: Can be fully tested by running a web-based chat interface that communicates with an agent via AG-UI, showing streaming responses and tool execution status.

**Acceptance Scenarios**:

1. **Given** a participant runs the AG-UI example, **When** they interact with the chat interface, **Then** they see streaming token-by-token responses from the agent
2. **Given** the agent executes a tool, **When** the participant observes the UI, **Then** they see real-time status updates showing tool execution progress
3. **Given** a participant completes the hands-on exercise, **When** they customize the UI component rendering, **Then** the agent's structured outputs display using their custom components

---

### User Story 3 - Exposing Agents via A2A Protocol (Priority: P3)

As a workshop participant, I want to expose my agent as a service using the Agent-to-Agent (A2A) protocol, so that other agents or systems can discover and interact with my agent programmatically.

**Why this priority**: A2A enables agent interoperability and is foundational for multi-agent scenarios. Participants learn how to make agents discoverable and callable as services.

**Independent Test**: Can be fully tested by running an agent as an A2A server and having a client discover its capabilities and invoke it remotely.

**Acceptance Scenarios**:

1. **Given** a participant runs the A2A server example, **When** they query the agent's capabilities endpoint, **Then** they receive a structured description of the agent's skills and input/output schemas
2. **Given** an A2A client connects to the agent, **When** it sends a task request, **Then** the agent processes the request and returns a structured response via A2A protocol
3. **Given** a participant completes the hands-on exercise, **When** they register their agent with a discovery service, **Then** other agents can discover and invoke it by capability

---

### User Story 4 - Deterministic Workflows Across Multiple Agents (Priority: P4)

As a workshop participant, I want to build deterministic workflows that coordinate multiple agents in a predictable sequence, so that I can create reliable multi-agent pipelines with guaranteed execution order.

**Why this priority**: Deterministic workflows are essential for production systems where reliability matters. This builds on A2A by orchestrating multiple agents in a controlled manner.

**Independent Test**: Can be fully tested by running a workflow that chains 3+ agents in sequence, with each step's output feeding the next, and verifying the execution order matches the workflow definition.

**Acceptance Scenarios**:

1. **Given** a participant runs the deterministic workflow example, **When** they provide an input, **Then** the workflow executes agents in the defined sequence with each agent receiving the previous agent's output
2. **Given** an agent in the workflow fails, **When** the workflow engine detects the failure, **Then** it executes the defined error handling strategy (retry, fallback, or abort)
3. **Given** a participant reviews the workflow execution log, **When** they examine the trace, **Then** they can see the exact order of agent invocations and data transformations

---

### User Story 5 - Declarative Agents and Workflows (Priority: P5)

As a workshop participant, I want to define agents and workflows declaratively using configuration files, so that I can create and modify agent behaviors without writing code.

**Why this priority**: Declarative approaches enable rapid iteration and make agent systems accessible to non-developers. This pattern supports governance and version control of agent behaviors.

**Independent Test**: Can be fully tested by modifying a YAML/JSON configuration file and observing the agent's behavior change without any code changes.

**Acceptance Scenarios**:

1. **Given** a participant runs the declarative agent example, **When** they modify the agent's system prompt in the config file, **Then** the agent exhibits the new behavior on restart
2. **Given** a participant defines a workflow in YAML, **When** they run the workflow engine, **Then** it orchestrates agents exactly as specified in the configuration
3. **Given** a participant completes the hands-on exercise, **When** they add a new agent to the declarative workflow, **Then** the workflow incorporates it without code changes

---

### User Story 6 - Moderating a Discussion Between Agents (Priority: P6)

As a workshop participant, I want to moderate a discussion between multiple agents with different perspectives, so that I can build debate-style systems that explore problems from multiple angles.

**Why this priority**: Agent discussions enable collaborative problem-solving and can surface insights that single agents miss. This advanced pattern demonstrates dynamic multi-agent interaction.

**Independent Test**: Can be fully tested by running a moderated debate between 2-3 agents on a topic, observing turn-taking, and seeing the moderator synthesize conclusions.

**Acceptance Scenarios**:

1. **Given** a participant runs the agent discussion example, **When** they provide a debate topic, **Then** the moderator agent facilitates structured turn-taking between participant agents
2. **Given** agents are discussing, **When** one agent references another's point, **Then** the discussion demonstrates agents building on each other's contributions
3. **Given** the discussion reaches a conclusion, **When** the moderator synthesizes the results, **Then** the output reflects the key points from all participating agents

---

### User Story 7 - Observability, Evaluation, and Evolving Agents (Priority: P7)

As a workshop participant, I want to implement comprehensive observability, evaluate agent performance, and use evaluation results to evolve agent behaviors, so that I can build agents that improve over time.

**Why this priority**: This capstone scenario ties together all learnings by showing how to measure, evaluate, and improve agent systems—essential for production deployments.

**Independent Test**: Can be fully tested by running an evaluation suite against an agent, reviewing the metrics dashboard, and applying an improvement based on evaluation feedback.

**Acceptance Scenarios**:

1. **Given** a participant runs the evaluation example, **When** they execute the test suite against an agent, **Then** they receive quantitative metrics (accuracy, latency, cost) and qualitative assessments
2. **Given** evaluation results identify a weakness, **When** the participant applies the suggested improvement (prompt tuning, tool addition), **Then** re-evaluation shows measurable improvement
3. **Given** a participant configures continuous evaluation, **When** the agent processes requests, **Then** metrics are collected and displayed in a real-time dashboard

---

### Edge Cases

- What happens when an MCP server is unavailable? Agent gracefully degrades with clear error messages indicating which tools are offline.
- What happens when AG-UI connection drops mid-stream? UI shows reconnection status and resumes streaming when connection restores.
- What happens when an A2A agent times out? Configurable timeouts with clear error responses following A2A protocol error schema.
- What happens when a workflow step fails in a deterministic workflow? Workflow engine executes defined error handling (retry with backoff, skip, or abort).
- What happens when declarative config is invalid? Validation errors are reported at load time with specific line numbers and fix suggestions.
- What happens when agents in a discussion produce conflicting outputs? Moderator agent applies conflict resolution strategy (voting, synthesis, or escalation).
- What happens when evaluation metrics conflict (e.g., faster but less accurate)? Dashboard shows trade-off analysis and lets users define weighting priorities.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Workshop MUST provide executable code examples for each scenario (MCP tools, AG-UI, A2A, workflows, declarative, discussions, observability)
- **FR-002**: Workshop MUST include hands-on exercises with clear instructions and expected outcomes for each module
- **FR-003**: Each scenario module MUST be self-contained and runnable independently
- **FR-004**: Workshop MUST provide clear documentation explaining the concepts and protocols behind each scenario
- **FR-005**: All code examples MUST follow the project's constitution (type hints, tests, clean code principles)
- **FR-006**: Workshop MUST demonstrate MCP protocol for tool integration with at least 3 example tools
- **FR-007**: Workshop MUST demonstrate AG-UI protocol for streaming chat interfaces with real-time updates
- **FR-008**: Workshop MUST demonstrate A2A protocol for agent discovery and remote invocation
- **FR-009**: Workshop MUST include an observability stack using OpenTelemetry instrumentation with Azure Monitor for traces and metrics visualization
- **FR-010**: Workshop MUST provide estimated completion times for each module
- **FR-011**: Workshop documentation MUST include guidance on token usage and API cost awareness (without enforcing limits in code)
- **FR-012**: Workshop MUST demonstrate deterministic workflow orchestration with error handling strategies
- **FR-013**: Workshop MUST demonstrate declarative agent/workflow configuration using YAML or JSON
- **FR-014**: Workshop MUST demonstrate multi-agent discussions with moderation and turn-taking
- **FR-015**: Workshop MUST demonstrate agent evaluation metrics collection and systematic prompt engineering iteration based on evaluation feedback

### Key Entities

- **Scenario Module**: A self-contained learning unit covering one agentic pattern/protocol; includes concept documentation, code examples, hands-on exercises, and verification steps
- **Agent**: An autonomous entity that perceives its environment, reasons about actions, and executes tools to achieve goals
- **MCP Server**: A Model Context Protocol server that exposes tools for agents to discover and invoke
- **AG-UI Client**: A user interface component that communicates with agents via the AG-UI protocol for streaming interactions
- **A2A Endpoint**: An Agent-to-Agent protocol endpoint that exposes agent capabilities for discovery and remote invocation
- **Workflow**: A deterministic sequence of agent invocations with defined data flow and error handling
- **Declarative Config**: A YAML/JSON file that defines agent behaviors and workflow orchestration without code
- **Moderator Agent**: A specialized agent that facilitates discussions between other agents
- **Evaluation Suite**: A collection of tests and metrics for measuring agent performance

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Participants can complete each scenario module (including exercises) within the documented time estimate (±20%)
- **SC-002**: 90% of participants successfully run all code examples on first attempt after following setup instructions
- **SC-003**: Each scenario module delivers a working, demonstrable outcome that participants can show to others
- **SC-004**: Participants can trace agent execution end-to-end using the observability dashboard
- **SC-005**: All code examples pass automated tests, demonstrating they work correctly
- **SC-006**: Workshop documentation is complete enough that participants can work through modules without instructor assistance
- **SC-007**: Participants can modify declarative configs and observe behavior changes without code changes

## Assumptions

- Participants have basic Python programming experience (can read/write Python functions, use pip)
- Participants have access to an LLM provider API (Azure OpenAI or OpenAI) with valid credentials
- Workshop will be run locally on participant machines (not a hosted environment)
- Participants have Python 3.11+ installed or can install it
- Workshop focuses on patterns, protocols, and concepts, not production deployment concerns
- Workshop uses **Microsoft Agent Framework** (Azure AI Agent Service) as the primary agentic framework
- Participants have basic familiarity with REST APIs and JSON
- Participants have access to an Azure subscription for Azure Monitor (or can use free tier)
- Docker is available for running local services (optional but recommended)

## Clarifications

### Session 2026-01-01

- Q: Which agentic AI framework should the workshop use for building agents? → A: Microsoft Agent Framework (Azure AI Agent Service)
- Q: Should the workshop include token budget limits and cost estimation features in the examples? → A: Documentation only (no code enforcement)
- Q: Should the workshop use a single consistent example domain across all pattern modules? → A: Single domain (e.g., research assistant) with evolving complexity
- Q: Which observability stack should the workshop use for tracing and metrics? → A: OpenTelemetry + Azure Monitor (cloud-native Azure observability)
- Q: What level of agent evolution/improvement should the workshop demonstrate? → A: Prompt engineering iteration (systematic prompt tuning based on eval results)
