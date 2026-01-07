# Tasks: Agentic AI Patterns Workshop

**Input**: Design documents from `/specs/001-agentic-patterns-workshop/`
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/ ✅, quickstart.md ✅

**Tests**: Tests verify notebook execution and protocol schemas. TDD per cell is adapted for educational context.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each scenario.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1-US7)
- Include exact file paths in descriptions

## Path Conventions

Based on plan.md structure:
- `src/common/` - Shared utilities
- `src/tools/` - MCP tool implementations  
- `src/agents/` - Reusable agent definitions
- `src/workflows/` - Workflow definitions
- `notebooks/` - Jupyter notebooks (00-07)
- `configs/` - Declarative YAML configs
- `tests/` - Unit, integration, and contract tests

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and development environment

- [X] T001 Create project structure with pyproject.toml at repository root
- [X] T002 [P] Configure .gitignore for Python, Jupyter, and environment files
- [X] T003 [P] Create .env.example template with Azure OpenAI and Monitor placeholders
- [X] T004 [P] Configure ruff.toml for linting and formatting
- [X] T005 [P] Configure mypy.ini for strict type checking
- [X] T006 [P] Create pytest.ini with asyncio and coverage settings

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No notebook development can begin until this phase is complete

- [X] T007 Create src/common/__init__.py with module exports
- [X] T008 Implement Azure/OpenAI configuration loader in src/common/config.py
- [X] T009 [P] Implement OpenTelemetry + Azure Monitor setup in src/common/telemetry.py
- [X] T010 [P] Implement custom exception hierarchy in src/common/exceptions.py
- [X] T011 [P] Create tests/conftest.py with shared fixtures (mock Azure client, test tracer)
- [X] T012 [P] Create tests/unit/test_common.py for config and telemetry validation
- [X] T013 Implement setup verification notebook in notebooks/00_setup.ipynb

**Checkpoint**: Foundation ready - user story implementation can now begin ✅

---

## Phase 3: User Story 1 - Simple Agent with MCP Tools and Observability (Priority: P1) 🎯 MVP

**Goal**: Build a simple agent, connect tools via MCP, inspect traces & metrics

**Independent Test**: Run standalone agent that connects to MCP server, invokes tools, produces observable traces

### Implementation for User Story 1

- [X] T014 Create src/tools/__init__.py with tool module exports
- [X] T015 [P] [US1] Implement search_web tool in src/tools/search_tool.py per contracts/mcp-tools.md
- [X] T016 [P] [US1] Implement calculate tool in src/tools/calculator_tool.py per contracts/mcp-tools.md
- [X] T017 [P] [US1] Implement file operations tools in src/tools/file_tool.py per contracts/mcp-tools.md
- [X] T018 [US1] Create MCP server wrapper in src/tools/mcp_server.py using FastMCP
- [X] T019 Create src/agents/__init__.py with agent module exports
- [X] T020 [US1] Implement BaseAgent with telemetry integration in src/agents/base_agent.py
- [X] T021 [US1] Implement ResearchAgent extending BaseAgent in src/agents/research_agent.py
- [X] T022 [US1] Create Scenario 1 notebook in notebooks/01_simple_agent_mcp.ipynb with:
  - Agent creation with Azure OpenAI
  - MCP tool discovery and invocation
  - OpenTelemetry trace inspection
  - Hands-on exercise: Add new MCP tool
  - Edge case: MCP server unavailability with graceful degradation
- [X] T023 [P] [US1] Create tests/contract/test_mcp_schemas.py to validate tool schemas
- [X] T024 [P] [US1] Create tests/integration/test_scenario_01.py to verify notebook executes

**Checkpoint**: User Story 1 complete - participants can build agents with MCP tools and see traces ✅

---

## Phase 4: User Story 2 - AG-UI Protocol for Streaming Chat (Priority: P2)

**Goal**: Build user interface for agent using AG-UI protocol with streaming responses

**Independent Test**: Run web-based chat interface communicating with agent via AG-UI, showing streaming responses

### Implementation for User Story 2

- [X] T025 [P] [US2] Create AG-UI FastAPI server module in src/agents/agui_server.py per contracts/agui-events.md
- [X] T026 [US2] Implement streaming event emitter in src/agents/agui_server.py supporting all event types
- [X] T027 [US2] Create Scenario 2 notebook in notebooks/02_agui_interface.ipynb with:
  - AG-UI server setup with FastAPI
  - Streaming token-by-token responses
  - Tool execution status updates
  - Hands-on exercise: Customize UI component rendering
  - Edge case: Connection drop with reconnection status UI
- [X] T028 [P] [US2] Create tests/contract/test_agui_schemas.py to validate event schemas
- [X] T029 [P] [US2] Create tests/integration/test_scenario_02.py to verify notebook executes

**Checkpoint**: User Story 2 complete - participants can build streaming chat interfaces ✅

---

## Phase 5: User Story 3 - A2A Protocol for Agent Interoperability (Priority: P3)

**Goal**: Expose agent as service using A2A protocol for discovery and remote invocation

**Independent Test**: Run agent as A2A server, have client discover capabilities and invoke remotely

### Implementation for User Story 3

- [X] T030 [P] [US3] Create A2A server module in src/agents/a2a_server.py per contracts/a2a-protocol.md
- [X] T031 [US3] Implement Agent Card generation at /.well-known/agent-card.json endpoint
- [X] T032 [US3] Implement A2A JSON-RPC message handler for tasks/send and tasks/get
- [X] T033 [US3] Create Scenario 3 notebook in notebooks/03_a2a_protocol.ipynb with:
  - A2A server setup with Agent Card
  - Capability discovery endpoint
  - Remote task invocation
  - Hands-on exercise: Register agent with discovery service (mock)
  - Edge case: Configurable timeouts with A2A error responses
- [X] T034 [P] [US3] Create tests/contract/test_a2a_schemas.py to validate Agent Card and message schemas
- [X] T035 [P] [US3] Create tests/integration/test_scenario_03.py to verify notebook executes

**Checkpoint**: User Story 3 complete - participants can expose agents via A2A protocol ✅

---

## Phase 6: User Story 4 - Deterministic Multi-Agent Workflows (Priority: P4)

**Goal**: Build deterministic workflows coordinating multiple agents in predictable sequence

**Independent Test**: Run workflow chaining 3+ agents, verify execution order matches definition

### Implementation for User Story 4

- [X] T036 Create src/workflows/__init__.py with workflow module exports
- [X] T037 [US4] Implement workflow step definitions in src/workflows/steps.py using data-model.md schemas
- [X] T038 [US4] Implement deterministic workflow engine in src/workflows/engine.py with:
  - Sequential agent orchestration
  - Data flow between steps
  - Error handling strategies (retry, fallback, abort)
- [X] T039 [US4] Create Scenario 4 notebook in notebooks/04_deterministic_workflows.ipynb with:
  - Workflow definition and execution
  - Multi-agent pipeline example
  - Error handling demonstration
  - Hands-on exercise: Build custom workflow
- [X] T040 [P] [US4] Create tests/unit/test_workflows.py for workflow engine logic
- [X] T041 [P] [US4] Create tests/integration/test_scenario_04.py to verify notebook executes

**Checkpoint**: User Story 4 complete - participants can build deterministic multi-agent workflows ✅

---

## Phase 7: User Story 5 - Declarative Agents and Workflows (Priority: P5)

**Goal**: Define agents and workflows using YAML configuration without code

**Independent Test**: Modify YAML config file, observe agent behavior change without code changes

### Implementation for User Story 5

- [X] T042 [P] [US5] Create YAML schema definitions in src/common/yaml_loader.py with Pydantic validation
- [X] T043 [P] [US5] Create research_agent.yaml in configs/agents/ per data-model.md AgentConfig schema
- [X] T044 [P] [US5] Create summarizer_agent.yaml in configs/agents/ per data-model.md AgentConfig schema
- [X] T045 [US5] Create research_pipeline.yaml in configs/workflows/ per data-model.md WorkflowConfig schema
- [X] T046 [US5] Implement declarative agent loader in src/agents/declarative.py
- [X] T047 [US5] Create Scenario 5 notebook in notebooks/05_declarative_agents.ipynb with:
  - Loading agents from YAML
  - Loading workflows from YAML
  - Runtime behavior modification
  - Hands-on exercise: Add new agent via config
- [X] T048 [P] [US5] Create tests/unit/test_declarative.py for YAML loading validation
- [X] T049 [P] [US5] Create tests/integration/test_scenario_05.py to verify notebook executes

**Checkpoint**: User Story 5 complete - participants can define agents declaratively ✅

---

## Phase 8: User Story 6 - Moderating Agent Discussions (Priority: P6)

**Goal**: Moderate discussion between multiple agents with different perspectives

**Independent Test**: Run moderated debate between 2-3 agents, observe turn-taking and conclusion synthesis

### Implementation for User Story 6

- [X] T050 [US6] Implement ModeratorAgent in src/agents/moderator_agent.py with:
  - Turn-taking facilitation
  - Conflict resolution strategy
  - Conclusion synthesis
- [X] T051 [US6] Implement discussion protocol in src/agents/discussion.py with:
  - Participant registration
  - Round management
  - Cross-reference detection
- [X] T052 [US6] Create Scenario 6 notebook in notebooks/06_agent_discussions.ipynb with:
  - Multi-agent debate setup
  - Moderator configuration
  - Turn-by-turn observation
  - Hands-on exercise: Add debate participant
- [X] T053 [P] [US6] Create tests/unit/test_discussion.py for discussion protocol
- [X] T054 [P] [US6] Create tests/integration/test_scenario_06.py to verify notebook executes

**Checkpoint**: User Story 6 complete - participants can build moderated agent discussions ✅

---

## Phase 9: User Story 7 - Observability, Evaluation, and Evolution (Priority: P7)

**Goal**: Implement comprehensive observability, evaluate performance, evolve agents

**Independent Test**: Run evaluation suite, review metrics dashboard, apply improvement from feedback

### Implementation for User Story 7

- [X] T055 [P] [US7] Implement evaluation metrics collector in src/common/evaluation.py with:
  - Accuracy metrics
  - Latency tracking
  - Cost estimation
- [X] T056 [P] [US7] Implement prompt iteration framework in src/common/prompt_tuning.py with:
  - Version tracking
  - A/B comparison support
  - Improvement suggestions
- [X] T057 [US7] Create Scenario 7 notebook in notebooks/07_evaluation_evolution.ipynb with:
  - Evaluation suite execution
  - Metrics dashboard visualization
  - Prompt tuning iteration
  - Hands-on exercise: Apply evaluation-driven improvement
- [X] T058 [P] [US7] Create tests/unit/test_evaluation.py for metrics collection
- [X] T059 [P] [US7] Create tests/integration/test_scenario_07.py to verify notebook executes

**Checkpoint**: User Story 7 complete - participants understand evaluation-driven agent evolution ✅

---

## Phase 10: Polish & Cross-Cutting Concerns

**Purpose**: Final improvements affecting multiple scenarios

- [X] T060 [P] Update README.md with workshop overview, prerequisites, and getting started guide
- [X] T061 [P] Create docs/ARCHITECTURE.md documenting module relationships and design decisions
- [X] T062 [P] Create docs/CONTRIBUTING.md for workshop extension guidelines
- [X] T066 [P] Add estimated completion times to each notebook header (FR-010: 15-45 min per scenario)
- [X] T067 [P] Create docs/COST_GUIDANCE.md with token usage estimates and Azure pricing tier recommendations (FR-011)
- [X] T063 Run quickstart.md validation - verify all setup steps work end-to-end
- [X] T064 Code review pass - ensure all modules follow Constitution (type hints, docstrings, ≤30 line functions)
- [X] T065 Create GitHub Actions workflow in .github/workflows/ci.yml for linting, type checking, and tests

**Checkpoint**: Phase 10 complete - workshop fully polished and ready for delivery ✅

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-9)**: All depend on Foundational phase completion
  - US1 (P1) must complete first (provides base agent pattern)
  - US2-US7 can proceed after US1 (build on foundation)
  - US4 should complete before US5 (workflows needed for declarative)
- **Polish (Phase 10)**: Depends on all user stories being complete

### User Story Dependencies

```text
Foundation (Phase 2)
       │
       ▼
┌──────────────┐
│  US1 (P1)    │ ◄─── MVP: Simple Agent + MCP + Observability
│  MCP + Obs   │
└──────┬───────┘
       │
       ├─────────────────┬─────────────────┬─────────────────┐
       ▼                 ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  US2 (P2)    │  │  US3 (P3)    │  │  US4 (P4)    │  │  US7 (P7)    │
│  AG-UI       │  │  A2A         │  │  Workflows   │  │  Evaluation  │
└──────────────┘  └──────────────┘  └──────┬───────┘  └──────────────┘
                                          │
                                          ▼
                                   ┌──────────────┐
                                   │  US5 (P5)    │
                                   │  Declarative │
                                   └──────┬───────┘
                                          │
                                          ▼
                                   ┌──────────────┐
                                   │  US6 (P6)    │
                                   │  Discussions │
                                   └──────────────┘
```

### Within Each User Story

- Source modules before notebooks
- Notebooks include hands-on exercises
- Tests can run in parallel after implementation

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational [P] tasks can run in parallel (within Phase 2)
- Once US1 completes: US2, US3, US4, US7 can start in parallel
- US5 requires US4 completion (workflow foundation)
- US6 requires US5 completion (declarative agents)
- All test tasks marked [P] within a story can run in parallel

---

## Parallel Example: User Story 1 (Phase 3)

```bash
# After T014 (tools __init__), launch all tool implementations together:
Task T015: "Implement search_web tool in src/tools/search_tool.py"
Task T016: "Implement calculate tool in src/tools/calculator_tool.py"
Task T017: "Implement file operations tools in src/tools/file_tool.py"

# After T022 (notebook complete), launch tests together:
Task T023: "Create tests/contract/test_mcp_schemas.py"
Task T024: "Create tests/integration/test_scenario_01.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1 (Simple Agent + MCP + Observability)
4. **STOP and VALIDATE**: Test scenario independently - participant can:
   - Build an agent with Azure OpenAI
   - Connect MCP tools
   - See execution traces in Azure Monitor
5. Deploy/demo if ready - **this is a complete, valuable workshop module**

### Incremental Delivery

1. Setup + Foundational → Foundation ready
2. Add US1 (MCP + Observability) → Test → **MVP Complete!**
3. Add US2 (AG-UI) → Test → Streaming UI capability
4. Add US3 (A2A) → Test → Agent interoperability
5. Add US4 (Workflows) → Test → Multi-agent orchestration
6. Add US5 (Declarative) → Test → No-code configuration
7. Add US6 (Discussions) → Test → Advanced multi-agent patterns
8. Add US7 (Evaluation) → Test → **Full Workshop Complete!**
9. **Refactor to Microsoft Agent Framework** → Align implementation with spec

Each story adds a complete scenario module without breaking previous ones.

### Parallel Team Strategy

With multiple developers after US1 completion:

| Developer | User Story | Prerequisite |
|-----------|------------|--------------|
| A | US2 (AG-UI) | US1 complete |
| B | US3 (A2A) | US1 complete |
| C | US4 (Workflows) | US1 complete |
| D | US7 (Evaluation) | US1 complete |

Then:
- Developer A → US5 (needs US4)
- Developer B → US6 (needs US5)

---

## Phase 11: Refactor to Microsoft Agent Framework (Priority: P0 - Critical)

**Goal**: Align implementation with spec by replacing raw OpenAI SDK with Microsoft Agent Framework

**Rationale**: The spec (plan.md, research.md) specifies using `agent-framework` package, but implementation uses raw `openai` SDK. This creates a gap between documentation and code.

**⚠️ BREAKING CHANGE**: This refactor will modify core agent classes and protocol servers

### Core Agent Refactoring

- [X] T068 [REFACTOR] Study agent-framework package API by running example code in notebooks/00_setup.ipynb
- [X] T069 [REFACTOR] Replace BaseAgent with ChatAgent from agent-framework in src/agents/base_agent.py:
  - Remove custom `AsyncAzureOpenAI` client management
  - Use `AzureOpenAIChatClient` from `agent_framework.azure`
  - Migrate `system_prompt` to `instructions` parameter
  - Migrate `_tools` list to `tools` parameter with `@ai_function` decorators
  - Preserve OpenTelemetry integration via middleware
- [X] T070 [REFACTOR] Update ResearchAgent in src/agents/research_agent.py to extend refactored BaseAgent
- [X] T071 [REFACTOR] Update ModeratorAgent in src/agents/moderator_agent.py to use ChatAgent patterns (N/A - uses AgentProtocol, doesn't extend BaseAgent)

### Tool Definition Refactoring

- [X] T072 [P] [REFACTOR] Convert src/tools/search_tool.py to use `@ai_function` decorator pattern
- [X] T073 [P] [REFACTOR] Convert src/tools/calculator_tool.py to use `@ai_function` decorator pattern
- [X] T074 [P] [REFACTOR] Convert src/tools/file_tool.py to use `@ai_function` decorator pattern

### AG-UI Protocol Refactoring

- [ ] T075 [REFACTOR] Replace custom AG-UI server in src/agents/agui_server.py with:
  - Use `add_agent_framework_fastapi_endpoint()` from `agent_framework.ag_ui`
  - Remove custom streaming event emitter (use built-in)
  - Preserve custom event types if needed via middleware
- [ ] T076 [REFACTOR] Update notebooks/02_agui_interface.ipynb to use new AG-UI integration

### A2A Protocol Refactoring

- [X] T077 [REFACTOR] Evaluate if src/agents/a2a_server.py can use agent-framework-a2a package (Result: Custom implementation needed - no agent-framework-a2a package exists)
- [X] T078 [REFACTOR] Update A2A server to leverage agent-framework-a2a if applicable, or document why custom implementation is needed (Result: Custom A2A is appropriate - matches Google A2A spec)

### Declarative Agent Refactoring

- [X] T079 [REFACTOR] Update src/agents/declarative.py to use agent-framework-declarative package (PARTIAL - see T095-T100 for full integration):
  - Import AgentFactory from `agent_framework_declarative`
  - Added AgentFactoryLoader class using `AgentFactory.create_agent_from_yaml_path()`
  - Added load_agent_from_yaml() and load_agents_with_factory() convenience functions
  - Keep DeclarativeWorkflowLoader (uses custom WorkflowEngine - no workflow factory in agent-framework-declarative)
  - Provider format: `AzureOpenAI.Chat` (not just `AzureOpenAI`)
  - **Note**: T095-T100 complete full integration with tests and notebook updates
- [X] T080 [REFACTOR] Update YAML configs in configs/agents/ to match agent-framework-declarative format (PARTIAL - see T096-T097 for complete conversion):
  - research_agent.yaml: Converted to kind: Prompt with model.provider: AzureOpenAI.Chat and model.options
  - summarizer_agent.yaml: Converted to kind: Prompt with model.provider: AzureOpenAI.Chat and model.options
  - **Note**: T096-T097 complete the YAML schema alignment

### Test Updates

- [ ] T081 [P] [REFACTOR] Update tests/unit/test_common.py for new agent patterns:
  - Test TelemetryMiddleware accepts model parameter
  - Test agent creation with ChatAgent interface
- [ ] T082 [P] [REFACTOR] Update tests/contract/test_mcp_schemas.py for @ai_function tools:
  - Verify tool schemas match agent-framework conventions
- [ ] T083 [P] [REFACTOR] Update tests/contract/test_agui_schemas.py for agent-framework AG-UI:
  - Test AG-UI endpoint integration
- [X] T084 [P] [REFACTOR] Update tests/contract/test_a2a_schemas.py if A2A changes (No changes needed - custom A2A implementation retained)
- [X] T085 [REFACTOR] Run all integration tests to verify notebooks still execute (468/468 tests pass)

### Notebook Updates

- [ ] T086 [REFACTOR] Update notebooks/01_simple_agent_mcp.ipynb with ChatAgent examples:
  - Show ChatAgent instantiation with AzureOpenAIChatClient
  - Demonstrate tool registration with @ai_function
  - Update trace inspection to match new middleware
- [ ] T087 [P] [REFACTOR] Update notebooks/03_a2a_protocol.ipynb if A2A implementation changed (Minimal changes - verify imports)
- [ ] T088 [P] [REFACTOR] Update notebooks/04_deterministic_workflows.ipynb for new agent interface:
  - Show workflow with ChatAgent instances
  - Update agent registration pattern
- [ ] T089 [P] [REFACTOR] Update notebooks/05_declarative_agents.ipynb for agent-framework-declarative:
  - Show AgentFactory.create_agent_from_yaml_path() usage
  - Update YAML config examples to kind: Prompt format
  - Demonstrate behavior modification via config changes
- [ ] T090 [P] [REFACTOR] Update notebooks/06_agent_discussions.ipynb for new ModeratorAgent:
  - Verify DiscussionProtocol works with ChatAgent instances
  - Update participant registration examples
- [ ] T091 [P] [REFACTOR] Update notebooks/07_evaluation_evolution.ipynb for new agent interface:
  - Update evaluation metrics collection for ChatAgent
  - Verify prompt tuning framework compatibility

### Documentation Updates

- [ ] T092 [P] [REFACTOR] Update README.md code examples to show agent-framework usage:
  - Update quick start examples with ChatAgent
  - Show @ai_function tool definition
  - Update architecture diagram if needed
- [ ] T093 [P] [REFACTOR] Update docs/ARCHITECTURE.md to reflect agent-framework integration:
  - Document ChatAgent vs custom BaseAgent
  - Update module relationships diagram
  - Document middleware pipeline
- [ ] T094 [REFACTOR] Verify .github/agents/copilot-instructions.md is accurate post-refactor:
  - Update package references
  - Update import patterns

**Checkpoint**: Implementation aligned with spec - Microsoft Agent Framework properly integrated ✅

---

## Phase 12: Declarative Agent Framework Integration (Priority: P0 - Critical)

**Goal**: Complete the declarative agent integration using agent_framework_declarative package

**Rationale**: The spec requires using declarative agent patterns from agent-framework-declarative. 
The AgentFactory class creates proper ChatAgent instances from YAML configuration.

**⚠️ BREAKING CHANGE**: This will change the YAML schema format for agent configs

### agent_framework_declarative Integration

- [ ] T095 [REFACTOR] Study agent_framework_declarative API:
  - Review AgentFactory.create_agent_from_yaml() signature and expected YAML schema
  - Review AgentFactory.create_agent_from_yaml_path() for file-based loading
  - Identify required fields: kind, name, description, instructions, model
  - Document bindings parameter for custom tool functions

- [ ] T096 [REFACTOR] Convert configs/agents/research_agent.yaml to agent-framework-declarative format:
  ```yaml
  kind: Prompt
  name: research_agent
  description: Research assistant agent
  instructions: |
    You are a research assistant...
  model:
    id: gpt-4o
    connection:
      kind: azure_openai
      endpoint: ${AZURE_OPENAI_ENDPOINT}
    options:
      temperature: 0.7
  tools:
    - kind: function
      name: search_web
    - kind: function
      name: read_document
  ```

- [ ] T097 [REFACTOR] Convert configs/agents/summarizer_agent.yaml to agent-framework-declarative format

- [ ] T098 [REFACTOR] Refactor src/agents/declarative.py:
  - Import AgentFactory from agent_framework_declarative
  - Create wrapper class DeclarativeAgentWrapper to maintain backward compatibility
  - Update DeclarativeAgentLoader.load_agent() to use AgentFactory.create_agent_from_yaml_path()
  - Update DeclarativeAgentLoader.load_all() to iterate YAML files and create agents
  - Keep DeclarativeWorkflowLoader unchanged (custom WorkflowEngine is appropriate)

- [ ] T099 [REFACTOR] Update tests/unit/test_declarative.py:
  - Update test YAML content to use kind: Prompt format
  - Update test assertions for ChatAgent return type
  - Add tests for AgentFactory integration

- [ ] T100 [REFACTOR] Update notebooks/05_declarative_agents.ipynb:
  - Show new YAML schema with kind: Prompt
  - Demonstrate AgentFactory usage
  - Update hands-on exercise for new format

**Checkpoint**: Declarative agents use agent-framework-declarative package properly ✅

---

## Phase 13: Agent Discussion Framework Refactoring (Priority: P0 - Critical)

**Goal**: Align agent discussion implementation with Microsoft Agent Framework Magentic patterns

**Rationale**: The current discussion implementation uses custom classes (`ModeratorAgent`, `DiscussionProtocol`, `AgentProtocol`) instead of the Microsoft Agent Framework's orchestration primitives (`ChatAgent`, `MagenticBuilder`, Magentic events). This creates educational inconsistency with other scenarios.

**Gap Analysis**:
| Component | Current | Expected (per spec) |
|-----------|---------|---------------------|
| Agent base | Custom `AgentProtocol` | `ChatAgent` from agent-framework |
| Moderator | Custom `ModeratorAgent` | `ChatAgent` with moderation instructions |
| Events | Custom `DiscussionTurn`, `RoundResult` | `MagenticAgentMessageEvent`, `WorkflowEvent` |
| Orchestration | Custom `DiscussionProtocol` | `MagenticBuilder` for multi-agent orchestration |
| Streaming | Not implemented | `MagenticAgentDeltaEvent` |

### Phase 13.1: Research & Design

- [X] T101 [RESEARCH] Study agent-framework Magentic APIs in notebooks/00_setup.ipynb:
  - Review MagenticBuilder class and multi-agent orchestration patterns
  - Document MagenticAgentDeltaEvent, MagenticAgentMessageEvent, MagenticFinalResultEvent schemas
  - Identify how WorkflowEvent maps to discussion rounds
  - Review MagenticOrchestratorMessageEvent for moderator coordination
  - Test HostedWebSearchTool integration for research discussions
  - **Result**: Magentic patterns ARE available in agent-framework package:
    - **GroupChatBuilder** - Best fit for structured multi-agent discussions
      - `set_manager(agent)` - LLM-based speaker selection (maps to ModeratorAgent)
      - `set_select_speakers_func(fn)` - Function-based selection (for deterministic rounds)
      - `participants([agent1, agent2])` - Add discussion participants
      - `with_max_rounds(n)` - Limit discussion rounds
      - `with_request_info()` - HITL pause before agent responses
      - GroupChatStateSnapshot contains: task, participants, conversation, history, round_index
    - **MagenticBuilder** - For complex task planning workflows (less suited for debates)
      - `participants(**agents)` - Named agent roles
      - `with_standard_manager(chat_client, max_round_count, max_stall_count)`
      - `with_plan_review(enable=True)` - HITL plan approval
      - `with_checkpointing(storage)` - Resume capability
    - **Key types for discussion mapping**:
      - GroupChatTurn → maps to DiscussionTurn
      - ChatMessage → maps to agent responses
      - WorkflowEvent → base event for streaming
      - GroupChatStateSnapshot → round state for speaker selection
    - **Recommendation**: Use GroupChatBuilder with `set_manager()` for ModeratorAgent pattern

- [X] T102 [DESIGN] Define mapping from current to agent-framework types:
  - DiscussionTurn → GroupChatTurn (speaker, content, timestamp)
  - RoundResult → GroupChatStateSnapshot (round_index, history, conversation)
  - ModeratorAgent → ChatAgent as manager in GroupChatBuilder.set_manager()
  - AgentProtocol → ChatAgent interface (already compatible)
  - DiscussionConfig → GroupChatBuilder configuration methods
  - **Type mapping documented in research.md**

- [X] T103 [DESIGN] Design event streaming architecture:
  - GroupChatBuilder events map to AG-UI event types (see data-model.md Section 8)
  - WorkflowStartedEvent → RUN_STARTED, GroupChatTurn → TEXT_MESSAGE_*, WorkflowFinishedEvent → RUN_FINISHED
  - Event adapter function `stream_discussion_to_agui()` documented with full implementation
  - FastAPI endpoint integration pattern for /discussion route
  - **Documented in data-model.md Section 8**

### Phase 13.2: Core Agent Refactoring

- [X] T104 [REFACTOR] Create DiscussionAgent wrapper class in src/agents/discussion.py:
  - Wrap ChatAgent from agent-framework (via DiscussionAgent class)
  - Add perspective/role configuration via system prompt (_build_role_instructions)
  - Implement async run() that delegates to ChatAgent with turn tracking
  - Implement async run_stream() for streaming responses
  - Preserve backward compatibility with AgentProtocol interface
  - Add factory function create_discussion_agent()
  - Export from src/agents/__init__.py
  - Added 28 unit tests in tests/unit/test_discussion.py

- [X] T105 [REFACTOR] Refactor ModeratorAgent in src/agents/moderator_agent.py to use ChatAgent:
  - Added optional chat_agent parameter for LLM-based moderation
  - Added is_llm_based property to check moderation mode
  - Added select_speaker() method for intelligent speaker selection
  - Added run() method for ChatAgent delegation
  - Added run_stream() method for streaming responses
  - Added DEFAULT_MODERATOR_INSTRUCTIONS constant
  - Created create_moderator_agent() factory function
  - Export from src/agents/__init__.py
  - Added 29 unit tests: TestModeratorAgentChatAgentIntegration (22 tests) + TestCreateModeratorAgentFactory (7 tests)
  - Preserved existing ConflictStrategy enum and round-robin logic as fallback

- [X] T106 [REFACTOR] Update DiscussionProtocol in src/agents/discussion.py to use Magentic patterns:
  - Updated DiscussionProtocol constructor: added use_group_chat=False and group_chat_manager=None params
  - Added register_discussion_agent() method to register DiscussionAgent wrappers
  - Added discussion_agents property to return list of registered DiscussionAgents
  - Added _build_group_chat() method to lazily construct GroupChatBuilder workflow
  - Added _round_robin_selector() method for deterministic speaker selection
  - Added run_discussion_stream() async iterator method for streaming discussion events
  - Event types: discussion_started, round_started, turn_started, turn_delta, turn_completed, round_completed, discussion_completed
  - Preserved existing register_participant(), on_turn(), on_round() APIs for backward compatibility
  - Added 20 unit tests for T106 features: TestDiscussionProtocolGroupChatMode (7), TestRoundRobinSelector (3), TestBuildGroupChat (2), TestRunDiscussionStream (8)
  - Total tests in suite: 594 passed, 80.64% coverage maintained

- [ ] T107 [REFACTOR] Update DebateProtocol and RoundRobinProtocol in src/agents/discussion.py:
  - Inherit from refactored DiscussionProtocol
  - Use Magentic events for turn notifications
  - Preserve role-based prioritization logic (proponent/opponent/neutral)
  - Update create_debate() and create_roundtable() factory functions

### Phase 13.3: Streaming & Events

- [ ] T108 [REFACTOR] Add streaming support with MagenticAgentDeltaEvent in src/agents/discussion.py:
  - Implement async iterator run_discussion_stream() for live turn streaming
  - Yield MagenticAgentDeltaEvent for token-by-token agent responses
  - Yield MagenticAgentMessageEvent for complete turns
  - Yield MagenticFinalResultEvent for discussion completion
  - Support AG-UI integration for real-time discussion UI

- [ ] T109 [REFACTOR] Add AG-UI endpoint for discussion streaming in src/agents/agui_server.py:
  - Create /discussion endpoint for streaming discussion events
  - Map Magentic events to AG-UI event format
  - Support multiple concurrent discussion sessions
  - Add thread_id tracking for discussion context

### Phase 13.4: Tool Integration

- [ ] T110 [P] [REFACTOR] Integrate HostedWebSearchTool for research discussions in src/agents/discussion.py:
  - Add tools parameter to DiscussionAgent and DiscussionConfig
  - Enable HostedWebSearchTool for fact-checking during debates
  - Use @ai_function decorator pattern for custom discussion tools
  - Track tool usage in participant statistics

- [ ] T111 [P] [REFACTOR] Add tool-augmented discussion protocol in src/agents/discussion.py:
  - Create ToolAugmentedDiscussionProtocol subclass
  - Allow agents to invoke tools during their turns
  - Emit tool call events via MagenticAgentDeltaEvent
  - Add tool_calls field to participant statistics

### Phase 13.5: Notebook Updates

- [ ] T112 [REFACTOR] Update notebooks/06_agent_discussions.ipynb Part 1-2:
  - Replace MockDebateAgent with ChatAgent-based DiscussionAgent
  - Show DiscussionAgent creation with perspective parameter
  - Demonstrate ChatAgent-based ModeratorAgent
  - Keep mock agents as fallback for offline testing

- [ ] T113 [REFACTOR] Update notebooks/06_agent_discussions.ipynb Part 3-5:
  - Show MagenticBuilder orchestration pattern
  - Demonstrate streaming events with MagenticAgentDeltaEvent
  - Update DiscussionProtocol examples with new event model
  - Show real LLM-powered discussion example

- [ ] T114 [REFACTOR] Update notebooks/06_agent_discussions.ipynb Part 6-7 and Exercise:
  - Update conflict resolution to use ChatAgent synthesis
  - Show tool-augmented discussions with HostedWebSearchTool
  - Update exercise to create ChatAgent-based participant
  - Add streaming UI example if AG-UI integration complete

### Phase 13.6: Test Updates

- [ ] T115 [P] [REFACTOR] Update tests/unit/test_discussion.py:
  - Test DiscussionAgent wraps ChatAgent correctly
  - Test MagenticBuilder integration in DiscussionProtocol
  - Test event emission (MagenticAgentMessageEvent, WorkflowEvent)
  - Test ChatAgent-based ModeratorAgent synthesis
  - Mock ChatAgent for deterministic unit tests

- [ ] T116 [P] [REFACTOR] Update tests/integration/test_scenario_06.py:
  - Test full discussion with ChatAgent participants (requires API)
  - Verify Magentic event streaming works
  - Test AG-UI integration for discussions
  - Add skip markers for tests requiring live API

### Phase 13.7: Documentation

- [ ] T117 [P] [REFACTOR] Update docs/ARCHITECTURE.md for discussion refactoring:
  - Document Magentic integration for multi-agent discussions
  - Update module relationship diagram to show ChatAgent usage
  - Add event flow documentation for discussion streaming
  - Document DiscussionAgent vs ChatAgent relationship

- [ ] T118 [REFACTOR] Update research.md with Magentic patterns documentation:
  - Add section on MagenticBuilder for discussions
  - Document Magentic event types and their usage
  - Add code examples for discussion orchestration

**Checkpoint**: Agent discussions use Microsoft Agent Framework Magentic patterns properly ✅

---

## Dependencies: Phase 13 Execution Order

```text
T101 (Research Magentic APIs)
    │
    ├──► T102 (Type mapping) ──► T103 (Event architecture)
    │
    ▼
T104 (DiscussionAgent) ◄─── Foundation for all subsequent tasks
    │
    ▼
T105 (ModeratorAgent) ──► T106 (DiscussionProtocol) ──► T107 (Protocols)
                                    │
                                    ▼
                              T108 (Streaming) ──► T109 (AG-UI endpoint)
                                    │
                                    ▼
                         ┌─────────────────────────┐
                         │ T110, T111 (Tools) [P]  │
                         └─────────────────────────┘
                                    │
                                    ▼
                         T112 ──► T113 ──► T114 (Notebooks)
                                    │
                                    ▼
                         ┌─────────────────────────┐
                         │ T115, T116 (Tests) [P]  │
                         └─────────────────────────┘
                                    │
                                    ▼
                         ┌─────────────────────────┐
                         │ T117, T118 (Docs) [P]   │
                         └─────────────────────────┘
```

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story (US1-US7)
- [REFACTOR] label marks Microsoft Agent Framework alignment tasks
- Each scenario module is independently completable and testable
- Notebooks are the primary delivery format - source modules support reuse
- Commit after each task or logical group
- Stop at any checkpoint to validate scenario independently
- Estimated times: Setup 15min, Foundational 30min, each User Story 45-60min, Refactor 2-3 hours
- Total: ~118 tasks across 13 phases

---

## Task Status Summary

### Completed Tasks: 85
- Phase 1 (Setup): 6/6 ✅
- Phase 2 (Foundational): 7/7 ✅
- Phase 3 (US1 - MCP + Obs): 11/11 ✅
- Phase 4 (US2 - AG-UI): 5/5 ✅
- Phase 5 (US3 - A2A): 6/6 ✅
- Phase 6 (US4 - Workflows): 6/6 ✅
- Phase 7 (US5 - Declarative): 8/8 ✅
- Phase 8 (US6 - Discussions): 5/5 ✅
- Phase 9 (US7 - Evaluation): 5/5 ✅
- Phase 10 (Polish): 8/8 ✅
- Phase 11 (Refactor): 10/27 (T068-T074 complete, T077-T078 complete, T084-T085 complete)

### Remaining Tasks: 33
- Phase 11 (Refactor): 17 remaining
  - AG-UI: T075-T076 (2 tasks)
  - Declarative: T079-T080 (2 tasks, superseded by Phase 12)
  - Tests: T081-T083 (3 tasks)
  - Notebooks: T086-T091 (6 tasks)
  - Documentation: T092-T094 (3 tasks)
- Phase 12 (Declarative Framework): 6 tasks (T095-T100)
- Phase 13 (Discussion Framework): 18 tasks (T101-T118)

### Priority Order for Remaining Work
1. **T101-T103** (Phase 13.1): Research Magentic APIs first - blocks all discussion refactoring
2. **T095-T100** (Phase 12): Complete declarative agent integration (can parallel with T101-T103)
3. **T104-T109** (Phase 13.2-13.3): Core discussion refactoring
4. **T075-T076**: AG-UI refactoring (can parallel with T104-T109)
5. **T110-T111** (Phase 13.4): Tool integration for discussions
6. **T112-T114** (Phase 13.5): Update discussion notebook
7. **T081-T083, T115-T116**: Update tests
8. **T086-T091**: Update other notebooks
9. **T092-T094, T117-T118**: Update documentation
